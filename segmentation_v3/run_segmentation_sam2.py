r"""
分割 pipeline v3：Grounding DINO（transformers） + SAM 2.1（ultralytics）
在 v2.1 基础上升级"范围限制 + 误检过滤 + 堆叠分离"，用于回应组长反馈：
    - 容易识别到杂七杂八的东西  -> ROI 裁剪 + 面积/形状过滤 + 词表收敛
    - 堆叠的识别不清晰          -> 显式 NMS 调参 + mask 粘连分离（可选深/浅开运算）

输出仍对齐《检测分割模块输出格式定义草案》（pipeline-1.0，seg-output-0.1）：
    - segmentation.json（机器读；v3 增加 roi / postprocess / filtered 三个字段，原字段不变）
    - report.md（人读汇总；v3 增加过滤与分离统计）
    - masks/ 每实例一张二值 PNG（0/255，与原图同尺寸）
    - 深度信息首版仅引用+状态

用法（在项目根目录 D:/AIE3905 下）：
    :: 基线（等价 v2.1，不加任何新参数）
    python scripts\run_segmentation_sam2.py ^
        --image "data\Chad_frames_002-005_20261013\images\xxx_color.jpg" ^
        --prompt "box. bag. bottle. carton. packet." ^
        --out-dir output\frame_211 --frame-id xxx

    :: 收紧版：限定 ROI + 过滤 + 分离（推荐先用 --grid 量好 ROI 坐标）
    python scripts\run_segmentation_sam2.py ^
        --image "..." --prompt "..." --out-dir output\tight ^
        --preset tight --roi "300,120,1000,780" --split-touching

    :: 只出网格图，用于目测 ROI 坐标（不出结果）
    python scripts\run_segmentation_sam2.py --image "..." --prompt "box." --grid-only

兼容性：run_segmentation(image_path, prompt, box_thresh, text_thresh) 签名不变，
评估脚本可继续用 --engine sam2 调用（默认不过滤，行为等于 v2.1）。
"""
import argparse
import json
import os
import sys
import time
from collections import deque

import numpy as np
from PIL import Image, ImageDraw

try:
    import torch
except ImportError:          # --grid-only 等纯后处理路径不强制依赖 torch
    torch = None

# ---------- 运行配置（CLI 会覆盖；评估脚本直接调 run_segmentation 时保持默认=基线） ----------
CFG = {
    "roi": None,              # (x1,y1,x2,y2)，None=整图
    "roi_margin": 60,         # 裁剪外扩像素数（防贴边物体被切断），白名单判定不用它
    "roi_crop": False,        # False=整图推理+中心点过滤（默认，召回优先）；True=裁剪推理（速度优先）
    "nms_iou": 0.0,           # 0=不额外做 NMS（DINO 自带）；>0 则按该 IoU 阈值再抑制
    "min_mask_area": 0,       # mask 像素数下限，0=不过滤
    "max_area_ratio": 1.0,    # mask 占 ROI 面积比例上限，1.0=不过滤
    "max_aspect": 0.0,        # 框长宽比上限（长边/短边），0=不过滤
    "split_touching": False,  # 是否对粘连 mask 做分离
    "split_max_erode": 40,    # 自适应分离的最大腐蚀层数（逐层找分裂层）
    "split_min_seed": 30,     # 分离后每个子实例的最小种子面积
    "split_min_area": 2000,   # 只对 mask 面积 >= 该值的实例尝试分离（省算力）
}
# 记录本次运行实际用到的分离腐蚀层数（写进 JSON 便于复现）
SPLIT_STATE = {"used_erode": 0, "attempts": 0}

# ---------- 设备 ----------
DEVICE = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
print(f"[info] 使用设备: {DEVICE}")

# ---------- 模型按需加载（--grid-only / --help 不需要加载大模型） ----------
_MODELS = {"gdino_proc": None, "gdino": None, "sam2": None}

SAM2_CKPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "models", "sam2.1_t.pt")
GDINO_NAME = "IDEA-Research/grounding-dino-tiny"


def _ensure_models():
    """首次调用时加载 Grounding DINO 与 SAM 2.1，之后复用。"""
    if _MODELS["sam2"] is not None:
        return

    # 权重通常已缓存：默认走离线模式，避免每次启动联网检查卡在重试。
    # 若要强制重新下载，先设环境变量 SEG_ALLOW_DOWNLOAD=1
    if os.environ.get("SEG_ALLOW_DOWNLOAD") != "1":
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    print("[info] 正在加载 Grounding DINO（离线模式，使用本地缓存）...")
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    _MODELS["gdino_proc"] = AutoProcessor.from_pretrained(GDINO_NAME)
    _MODELS["gdino"] = AutoModelForZeroShotObjectDetection.from_pretrained(
        GDINO_NAME).to(DEVICE)
    print("[info] Grounding DINO 加载完成。")

    print("[info] 正在加载 SAM 2.1（Tiny，ultralytics）...")
    from ultralytics import SAM
    if not os.path.exists(SAM2_CKPT):
        print(f"[error] 找不到 SAM 2.1 权重: {SAM2_CKPT}")
        print("[error] 请把权重放到 models\\ 目录下（sam2.1_t.pt）。")
        sys.exit(1)
    _MODELS["sam2"] = SAM(SAM2_CKPT)
    print("[info] SAM 2.1 加载完成。")


# ==================== 后处理工具（纯 numpy，无新增依赖） ====================

def _iou_one_to_many(box, boxes):
    """单框 vs 多框的 IoU 数组。"""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    w = np.maximum(0.0, x2 - x1)
    h = np.maximum(0.0, y2 - y1)
    inter = w * h
    area_a = (box[2] - box[0]) * (box[3] - box[1])
    area_b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area_a + area_b - inter
    return inter / np.maximum(union, 1e-9)


def nms(boxes, scores, iou_thr):
    """标准 NMS。返回保留的下标列表（按分数降序）。"""
    if len(boxes) == 0:
        return []
    arr = np.asarray(boxes, dtype=np.float32)
    sc = np.asarray(scores, dtype=np.float32)
    order = sc.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        ious = _iou_one_to_many(arr[i], arr[order[1:]])
        surviv = np.where(ious <= iou_thr)[0]
        order = order[surviv + 1]
    return keep


def _erode(mask):
    """4 邻域腐蚀一次（padding 实现，避免 np.roll 的环绕伪连通）。"""
    h, w = mask.shape
    pad = np.zeros((h + 2, w + 2), dtype=bool)
    pad[1:-1, 1:-1] = mask
    return (pad[1:-1, 1:-1] & pad[:-2, 1:-1] & pad[2:, 1:-1]
            & pad[1:-1, :-2] & pad[1:-1, 2:])


def _label_components(mask, max_rounds=0):
    """连通域标记（向量化传播，4 邻域）。返回 (标签图, 连通域数)。

    不用逐像素 Python BFS：在 200x300 级别的局部数组上，Python BFS 要 ~1s，
    而"每轮让标签向邻居取最小"的 numpy 实现只要 ~0.1s。
    """
    if not mask.any():
        return np.zeros(mask.shape, np.int32), 0
    if max_rounds <= 0:                      # 传播轮数需覆盖最长链
        max_rounds = int(2 * (mask.shape[0] + mask.shape[1])) + 50
    BIG = np.int64(1 << 40)
    lab = np.where(mask, np.arange(mask.size, dtype=np.int64).reshape(mask.shape) + 1, BIG)
    for _ in range(max_rounds):
        prev = lab.copy()
        p = lab.copy()
        p[1:, :] = np.minimum(p[1:, :], prev[:-1, :])
        p[:-1, :] = np.minimum(p[:-1, :], prev[1:, :])
        p[:, 1:] = np.minimum(p[:, 1:], prev[:, :-1])
        p[:, :-1] = np.minimum(p[:, :-1], prev[:, 1:])
        p = np.where(mask, p, BIG)
        if np.array_equal(p, prev):
            lab = p
            break
        lab = p
    lab = np.where(mask, lab, 0).astype(np.int64)
    # 把稀疏 id 压成 1..n
    _, inv = np.unique(lab[lab > 0], return_inverse=True)
    compact = np.zeros(lab.shape, np.int32)
    compact[lab > 0] = inv.reshape(-1).astype(np.int32) + 1
    return compact, int(compact.max())


def split_touching_mask(mask, max_erode=40, min_seed_area=30):
    """把粘连成一整块的 mask 分离成多个子 mask（自适应腐蚀 + 多源生长的近似分水岭）。

    为什么不用固定腐蚀次数：两个形状互相重叠的物体，其并集 mask 往往没有细颈，
    固定腐蚀 3~5 次根本分不开；必须一直腐蚀到"芯"彼此断开的那一层。

    做法：逐层腐蚀，找第一个能分成 >=2 个芯的层 -> 以这些芯为种子，
    在原始 mask 范围内向外生长（谁先到归谁，近似分水岭）。
    返回 (list[np.bool_], 实际使用的腐蚀层数)；无法拆分时返回 ([mask], 0)。
    """
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return [mask], 0
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    local = mask[y0:y1, x0:x1].copy()
    local_area = int(local.sum())

    # 先测出"腐蚀到消失"的总层数（≈ 最大内切半径）
    total = 0
    layer = local.copy()
    while layer.any() and total < max(1, max_erode):
        layer = _erode(layer)
        total += 1
    if total < 3:
        return [mask], 0   # 太瘦，不具备拆分条件

    def erode_n(n):
        m = local.copy()
        for _ in range(n):
            m = _erode(m)
        return m

    rel_min = max(min_seed_area, int(0.01 * local_area))
    best = None
    lo, hi = 1, total - 1
    while lo <= hi:                      # 二分找"最小的分裂层"
        mid = (lo + hi) // 2
        lab_m, n_m = _label_components(erode_n(mid))
        keep_m = None
        if n_m >= 2:
            sizes = np.bincount(lab_m.ravel(), minlength=n_m + 1)
            keep_m = [c for c in range(1, n_m + 1) if sizes[c] >= rel_min]
        if keep_m and len(keep_m) >= 2:
            best = (mid, lab_m, keep_m)
            hi = mid - 1
        else:
            lo = mid + 1

    if best is None:
        return [mask], 0
    used_k, lab, keep = best

    # 多源生长：所有芯的像素入队，向内未标注前景扩张
    grown = np.where(np.isin(lab, keep), lab, 0).astype(np.int32)
    h, w = local.shape
    q = deque((int(y), int(x)) for y, x in np.argwhere(grown > 0))

    while q:
        y, x = q.popleft()
        my_lab = grown[y, x]
        if my_lab == 0:
            continue
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and local[ny, nx] and grown[ny, nx] == 0:
                    grown[ny, nx] = my_lab
                    q.append((ny, nx))

    parts = []
    for c in keep:
        sub = (grown == c)
        if sub.any():
            full = np.zeros(mask.shape, dtype=bool)
            full[y0:y1, x0:x1] = sub
            parts.append(full)
    if len(parts) < 2:
        return [mask], 0
    return parts, used_k


def _prompt_classes(prompt):
    """把 prompt 拆成类别词表（"box. bag. bottle." -> ["box","bag","bottle"]）。"""
    return [c.strip().lower() for c in prompt.split(".") if c.strip()]


def _clean_label(raw, classes):
    """清洗 transformers 的 text_labels 残留（subword "^##on"、跨类拼接）。

    优先用 prompt 里的类别词做包含匹配；都不命中时去掉 ## 片段兜底。
    """
    s = str(raw).lower().strip()
    hits = [(s.find(c), c) for c in classes if c and c in s]
    if hits:
        hits.sort(key=lambda x: (x[0], -len(x[1])))
        return hits[0][1]
    words = [w for w in s.replace(".", " ").split() if not w.startswith("##")]
    return " ".join(words) if words else "object"


def mask_to_bbox_int(mask):
    """从二值 mask 非零像素重算整数半开区间 bbox [xmin,ymin,xmax+1,ymax+1]。"""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


# ==================== 核心推理（签名不变，评估脚本兼容） ====================

def run_segmentation(image_path, prompt, box_thresh=0.25, text_thresh=0.20):
    """返回 (image_full, instances, filtered)。

    instances: list[dict] 每项含 instance_id/bbox_xyxy(全图坐标)/det_score/label/mask(全图尺寸)
    filtered:  list[dict] 被后处理丢弃的候选及原因（用于审计，不静默丢弃）
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"找不到图片: {image_path}")

    _ensure_models()
    import torch as _torch   # 局部再导一次，确保 no_grad 可用
    SPLIT_STATE["used_erode"] = 0
    SPLIT_STATE["attempts"] = 0
    image_full = Image.open(image_path).convert("RGB")
    W, H = image_full.size
    print(f"[info] 图片尺寸: {image_full.size}")

    # 0) ROI：只把作业区送进模型（相机固定 -> ROI 固定）
    roi = CFG["roi"]
    roi_box = None
    if roi:
        rx1, ry1, rx2, ry2 = [int(v) for v in roi]
        rx1, ry1 = max(0, rx1), max(0, ry1)
        rx2, ry2 = min(W, rx2), min(H, ry2)
        if rx2 - rx1 < 16 or ry2 - ry1 < 16:
            raise ValueError(f"ROI 太小或非法: {roi}")
        roi_box = (rx1, ry1, rx2, ry2)
        if CFG["roi_crop"]:
            # 裁剪模式（追求速度）：向外扩 margin，避免贴边物体被切断
            margin = max(0, int(CFG["roi_margin"]))
            cx1, cy1 = max(0, rx1 - margin), max(0, ry1 - margin)
            cx2, cy2 = min(W, rx2 + margin), min(H, ry2 + margin)
            crop = image_full.crop((cx1, cy1, cx2, cy2))
            print(f"[info] ROI 生效(裁剪模式): 白名单 ({rx1},{ry1})-({rx2},{ry2})，"
                  f"裁剪 ({cx1},{cy1})-({cx2},{cy2}) 外扩 {margin}px，尺寸 {crop.size}")
        else:
            # 过滤模式（默认，召回优先）：整图推理，出结果后按中心点丢框
            cx1, cy1 = 0, 0
            crop = image_full
            print(f"[info] ROI 生效(过滤模式): 白名单 ({rx1},{ry1})-({rx2},{ry2})，"
                  f"整图推理 + 中心点归属过滤（不裁剪，召回无损）")
    else:
        cx1, cy1 = 0, 0
        crop = image_full
    crop_np = np.array(crop)
    cw, ch = crop.size

    # 1) Grounding DINO：文本 -> bbox（在 ROI 内搜索）
    t0 = time.time()
    proc, gdino = _MODELS["gdino_proc"], _MODELS["gdino"]
    inputs = proc(images=crop, text=prompt, return_tensors="pt").to(DEVICE)
    with _torch.no_grad():
        outputs = gdino(**inputs)
    import inspect
    _params = set(inspect.signature(
        proc.post_process_grounded_object_detection).parameters.keys())
    _post_kwargs = {"target_sizes": [crop.size[::-1]]}
    if "box_threshold" in _params:
        _post_kwargs["box_threshold"] = box_thresh
    if "text_threshold" in _params:
        _post_kwargs["text_threshold"] = text_thresh
    if "threshold" in _params:
        _post_kwargs["threshold"] = box_thresh
    results = proc.post_process_grounded_object_detection(
        outputs, inputs.input_ids, **_post_kwargs)[0]

    boxes = [b.cpu().numpy().tolist() for b in results["boxes"]]
    scores = [float(s) for s in results["scores"]]
    # transformers >=4.51 起 labels 变整数 id，优先取 text_labels
    classes = _prompt_classes(prompt)
    labels = [_clean_label(r, classes)
              for r in results.get("text_labels", results.get("labels"))]
    t1 = time.time()
    print(f"[info] Grounding DINO 检出 {len(boxes)} 个候选，耗时 {t1 - t0:.2f}s")

    filtered = []

    # 2) 显式 NMS（可选）：抑制同一物体的重复框
    if CFG["nms_iou"] and len(boxes) > 1:
        keep_idx = nms(boxes, scores, CFG["nms_iou"])
        for i in range(len(boxes)):
            if i not in keep_idx:
                bx = boxes[i]   # 注意：DINO 在裁剪图上推理，坐标需加 ROI 偏移
                filtered.append({"stage": "nms", "reason": f"nms_iou>{CFG['nms_iou']}",
                                 "bbox": [round(float(bx[0]) + cx1, 1),
                                          round(float(bx[1]) + cy1, 1),
                                          round(float(bx[2]) + cx1, 1),
                                          round(float(bx[3]) + cy1, 1)],
                                 "det_score": round(scores[i], 3),
                                 "object_label": str(labels[i])})
        boxes = [boxes[i] for i in keep_idx]
        scores = [scores[i] for i in keep_idx]
        labels = [labels[i] for i in keep_idx]
        print(f"[info] NMS 抑制后保留 {len(boxes)} 个候选")

    # 3) SAM 2.1：每个 bbox -> mask（在 ROI 裁剪图上推理，mask 再贴回全图尺寸）
    instances = []
    t2 = time.time()
    if boxes:
        sam2_results = _MODELS["sam2"].predict(source=crop_np, bboxes=boxes, verbose=False)
        masks_data = None
        if sam2_results and sam2_results[0].masks is not None:
            masks_data = sam2_results[0].masks.data.cpu().numpy()

        roi_area = (float((roi_box[2] - roi_box[0]) * (roi_box[3] - roi_box[1]))
                    if roi_box else float(cw * ch))
        for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            if masks_data is not None and i < len(masks_data):
                m_crop = masks_data[i] > 0.5
            else:
                m_crop = np.zeros((ch, cw), dtype=bool)

            full_mask = np.zeros((H, W), dtype=bool)
            full_mask[cy1:cy1 + ch, cx1:cx1 + cw] = m_crop

            # --- 后处理过滤：面积 / 占比 / 长宽比 ---
            area = int(full_mask.sum())
            bbox = mask_to_bbox_int(full_mask)
            if bbox is None:
                bx = [round(float(v)) for v in box]
                bbox_tmp = [bx[0] + cx1, bx[1] + cy1, bx[2] + cx1, bx[3] + cy1]
            else:
                bbox_tmp = bbox

            drop_reason = None
            if roi_box and not (roi_box[0] <= (bbox_tmp[0] + bbox_tmp[2]) / 2.0 <= roi_box[2]
                                and roi_box[1] <= (bbox_tmp[1] + bbox_tmp[3]) / 2.0 <= roi_box[3]):
                drop_reason = "outside_roi"
            if drop_reason is None and CFG["min_mask_area"] and area < CFG["min_mask_area"]:
                drop_reason = f"mask_area<{CFG['min_mask_area']}"
            elif drop_reason is None and CFG["max_area_ratio"] < 1.0 and area / max(roi_area, 1.0) > CFG["max_area_ratio"]:
                drop_reason = f"area_ratio>{CFG['max_area_ratio']}"
            elif drop_reason is None and CFG["max_aspect"]:
                bw = max(1, bbox_tmp[2] - bbox_tmp[0])
                bh = max(1, bbox_tmp[3] - bbox_tmp[1])
                if max(bw, bh) / min(bw, bh) > CFG["max_aspect"]:
                    drop_reason = f"aspect>{CFG['max_aspect']}"
            if drop_reason:
                filtered.append({"stage": "geometry", "reason": drop_reason,
                                 "bbox": bbox_tmp,
                                 "mask_area": area,
                                 "det_score": round(float(score), 3),
                                 "object_label": str(label)})
                continue

            # --- 堆叠分离：只对"明显偏大"的 mask 尝试拆分，小块直接跳过 ---
            if CFG["split_touching"] and area >= CFG["split_min_area"]:
                SPLIT_STATE["attempts"] += 1
                parts, used_k = split_touching_mask(
                    full_mask, CFG["split_max_erode"], CFG["split_min_seed"])
                if used_k:
                    SPLIT_STATE["used_erode"] = max(SPLIT_STATE["used_erode"], used_k)
            else:
                parts, used_k = [full_mask], 0

            if len(parts) > 1:
                print(f"[info] 实例 {i} 粘连 mask 已分离为 {len(parts)} 块")
            for j, part in enumerate(parts):
                instances.append({
                    "instance_id": f"i{len(instances):03d}",
                    "bbox_xyxy": [float(v) for v in bbox_tmp],
                    "det_score": float(score),
                    "label": str(label),
                    "mask": part,
                    "split_from": f"det{i}" if len(parts) > 1 else None,
                })
    t3 = time.time()
    print(f"[info] SAM 2.1 分割完成，耗时 {t3 - t2:.2f}s")

    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return image_full, instances, filtered


# ---------- 深度信息（首版仅引用+状态） ----------
def build_depth_info(depth_path, out_dir):
    warnings = []
    if depth_path and os.path.exists(depth_path):
        rel = os.path.relpath(depth_path, out_dir).replace("\\", "/")
        try:
            with Image.open(depth_path) as d:
                size = [d.width, d.height]
                mode = d.mode
        except Exception as e:
            warnings.append(f"depth_read_failed:{e}")
            size, mode = None, "unknown"
        dtype = {"I;16": "uint16", "I": "uint32", "L": "uint8"}.get(mode, "unconfirmed")
        info = {
            "depth_path": rel, "depth_size": size, "depth_dtype": dtype,
            "depth_unit": "unconfirmed", "depth_scale_to_m": None,
            "invalid_depth_values": None, "alignment": "unconfirmed",
            "intrinsics": None, "depth_usable": False,
        }
        warnings.append("depth_metadata_unconfirmed")
    else:
        info = {
            "depth_path": None, "depth_size": None, "depth_dtype": None,
            "depth_unit": None, "depth_scale_to_m": None,
            "invalid_depth_values": None, "alignment": None,
            "intrinsics": None, "depth_usable": False,
        }
        warnings.append(f"depth_file_not_found:{depth_path}" if depth_path else "depth_not_provided")
    return info, warnings


# ---------- 可视化（展示用，不作下游输入） ----------
def draw_overlay(image, instances, filtered=None, roi=None, grid=False):
    img = np.array(image).copy()
    colors = [(0, 255, 0), (0, 200, 255), (255, 200, 0), (255, 0, 255),
              (0, 255, 180), (255, 120, 0)]
    for k, ins in enumerate(instances):
        mask = ins["mask"]
        color = np.array(colors[k % len(colors)], dtype=np.uint8)
        img[mask] = (0.5 * img[mask] + 0.5 * color).astype(np.uint8)
        pil = Image.fromarray(img)
        draw = ImageDraw.Draw(pil)
        x1, y1, x2, y2 = ins["bbox_xyxy"]
        ink = tuple(int(v) for v in color)
        draw.rectangle([x1, y1, x2, y2], outline=ink, width=2)
        draw.text((x1, max(0, y1 - 12)), f"{ins['label']} {ins['det_score']:.2f}", fill=ink)
        img = np.array(pil)

    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    if grid:
        W, H = pil.size
        for x in range(0, W, 100):
            draw.line([(x, 0), (x, H)], fill=(200, 200, 200), width=1)
            draw.text((x + 2, 2), str(x), fill=(120, 120, 120))
        for y in range(0, H, 100):
            draw.line([(0, y), (W, y)], fill=(200, 200, 200), width=1)
            draw.text((2, y + 1), str(y), fill=(120, 120, 120))
    if roi:
        draw.rectangle(list(roi), outline=(255, 255, 0), width=3)
        draw.text((roi[0] + 4, roi[1] + 4), "ROI", fill=(255, 255, 0))
    return pil


# ---------- 写 segmentation.json + report.md ----------
def write_outputs(out_dir, image, instances, args, seg_seconds, depth_info,
                  depth_warnings, filtered):
    masks_dir = os.path.join(out_dir, "masks")
    os.makedirs(masks_dir, exist_ok=True)

    json_instances = []
    errors = []
    warnings = list(depth_warnings)

    for ins in instances:
        mask_path = os.path.join(masks_dir, f"{ins['instance_id']}.png")
        Image.fromarray((ins["mask"] * 255).astype(np.uint8)).save(mask_path)

        bbox = mask_to_bbox_int(ins["mask"])
        if bbox is None:
            bx = [round(float(x)) for x in ins["bbox_xyxy"]]
            bbox = [bx[0], bx[1], bx[2] + 1, bx[3] + 1]
            errors.append({"stage": "segmentation", "instance_id": ins["instance_id"],
                           "code": "mask_empty_fallback_box",
                           "message": "SAM 未生成有效前景，bbox 回退为检测框"})

        item = {
            "instance_id": ins["instance_id"],
            "object_label": ins["label"],
            "bbox_xyxy": bbox,
            "mask_path": os.path.relpath(mask_path, out_dir).replace("\\", "/"),
            "det_score": round(ins["det_score"], 3),
            "seg_score": None,
        }
        if ins.get("split_from"):
            item["split_from"] = ins["split_from"]
        json_instances.append(item)

    status = "no_match" if len(json_instances) == 0 else ("partial" if errors else "ok")

    result = {
        "schema_version": "pipeline-1.0",
        "run_id": args.run_id,
        "frame_id": args.frame_id,
        "stage": "segmentation",
        "status": status,
        "prompt": args.prompt,
        "preprocess_ref": args.preprocess,
        "image_size": [image.size[0], image.size[1]],
        "model_version": "seg_dino_tiny_sam21_v3",
        "roi": list(CFG["roi"]) if CFG["roi"] else None,
        "roi_margin": CFG["roi_margin"] if (CFG["roi"] and CFG["roi_crop"]) else None,
        "roi_mode": ("crop" if CFG["roi_crop"] else "filter") if CFG["roi"] else "full",
        "postprocess": {
            "nms_iou": CFG["nms_iou"],
            "min_mask_area": CFG["min_mask_area"],
            "max_area_ratio": CFG["max_area_ratio"],
            "max_aspect": CFG["max_aspect"],
            "split_touching": CFG["split_touching"],
            "split_max_erode": CFG["split_max_erode"],
            "split_used_erode": SPLIT_STATE.get("used_erode", 0),
            "split_min_seed": CFG["split_min_seed"],
            "split_min_area": CFG["split_min_area"],
            "split_attempts": SPLIT_STATE["attempts"],
            "filtered_count": len(filtered),
        },
        "instance_count": len(json_instances),
        "instances": json_instances,
        "filtered": filtered,
        "depth": depth_info,
        "warnings": warnings,
        "errors": errors,
    }
    json_path = os.path.join(out_dir, "segmentation.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    md_lines = [
        f"# 分割结果汇总 · {args.frame_id}",
        "",
        f"- run_id: {args.run_id}",
        f"- prompt: {args.prompt}",
        f"- 状态: {status}（检出 {len(json_instances)} 个实例）",
        f"- 模型: seg_dino_tiny_sam21_v3",
        f"- ROI: {CFG['roi'] if CFG['roi'] else '整图'}"
        + ((f"（裁剪模式，外扩 {CFG['roi_margin']}px）" if CFG["roi_crop"]
             else "（过滤模式：整图推理，按中心点丢弃作业区外实例）") if CFG["roi"] else ""),
        f"- 后处理: NMS IoU={CFG['nms_iou']}，最小 mask 面积={CFG['min_mask_area']}，"
        f"面积占比上限={CFG['max_area_ratio']}，长宽比上限={CFG['max_aspect']}，"
        f"粘连分离={CFG['split_touching']}",
        f"- 后处理过滤: {len(filtered)} 个候选被丢弃",
          f"- 分离尝试: {SPLIT_STATE['attempts']} 次（面积 < {CFG['split_min_area']} 的直接跳过）",
        f"- 耗时: 分割推理约 {seg_seconds:.1f}s（CPU，纯推理）",
        f"- 深度: depth_usable={depth_info['depth_usable']}（首版仅保留引用与状态）",
        "",
        "| instance_id | object_label | bbox_xyxy | det_score | mask |",
        "|---|---|---|---|---|",
    ]
    for j in json_instances:
        md_lines.append(f"| {j['instance_id']} | {j['object_label']} | {j['bbox_xyxy']} "
                        f"| {j['det_score']} | {j['mask_path']} |")

    if filtered:
        md_lines += ["", "## 被过滤的候选（可审计）", "",
                     "| object_label | bbox_xyxy | mask_area | det_score | reason |",
                     "|---|---|---|---|---|"]
        for f_ in filtered:
            md_lines.append(f"| {f_.get('object_label', '-')} | {f_.get('bbox', '-')} "
                            f"| {f_.get('mask_area', '-')} | {f_.get('det_score', '-')} "
                            f"| {f_.get('reason', '-')} |")

    md_lines += ["", "## 说明",
                 "- bbox 由最终 mask 非零像素重算（整数半开区间 [xmin,ymin,xmax+1,ymax+1]）。",
                 "- 材质/几何信息由下游模块输出，本模块不判定。",
                 "- 深度信息首版仅引用+状态，未做像素处理（等高线/点云留待后续版本）。"]
    if any(i.get("split_from") for i in json_instances):
        md_lines.append("- 带 split_from 字段的实例由粘连 mask 分离得到，与同源实例共享检测框与分数。")
    if warnings:
        md_lines += ["", "## 警告"] + [f"- {w}" for w in warnings]
    if errors:
        md_lines += ["", "## 错误"] + [f"- [{e['code']}] {e['message']} ({e['instance_id']})"
                                       for e in errors]

    md_path = os.path.join(out_dir, "report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    return json_path, md_path, masks_dir


# ---------- CLI ----------
def parse_roi(s):
    parts = [p.strip() for p in s.replace("，", ",").split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("ROI 需为 x1,y1,x2,y2 四个整数")
    return tuple(int(float(p)) for p in parts)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="RGB 图路径")
    ap.add_argument("--prompt", required=True,
                    help='提示词，每类后加句号，例如 "cardboard box on the conveyor. plastic bag on the conveyor."')
    ap.add_argument("--box-thresh", type=float, default=0.25, help="检测阈值")
    ap.add_argument("--text-thresh", type=float, default=0.20, help="文本匹配阈值")
    ap.add_argument("--out-dir", default="output/seg_demo_sam2", help="输出目录")
    ap.add_argument("--run-id", default=None, help="一次运行标识，缺省自动生成")
    ap.add_argument("--frame-id", default=None, help="源帧标识，缺省取图片文件名")
    ap.add_argument("--depth", default=None, help="深度 PNG 路径（可选，首版仅引用+状态）")
    ap.add_argument("--preprocess", default=None, help="preprocess.json 相对路径（可选）")
    ap.add_argument("--roi", type=parse_roi, default=None, help="作业区 x1,y1,x2,y2，只在此区域推理")
    ap.add_argument("--roi-margin", type=int, default=60,
                    help="ROI 裁剪时外扩的像素数，防止贴边物体被切断，默认 60")
    ap.add_argument("--roi-crop", action="store_true",
                    help="裁剪 ROI 后再推理（快，但贴边/小目标召回会掉）；默认整图推理+中心点过滤")
    ap.add_argument("--preset", choices=["baseline", "tight"], default="baseline",
                    help="tight=一键开启几何过滤与 NMS 的推荐组合")
    ap.add_argument("--nms-iou", type=float, default=0.0, help="额外 NMS 的 IoU 阈值，0=不做")
    ap.add_argument("--min-mask-area", type=int, default=0, help="mask 像素数下限，0=不过滤")
    ap.add_argument("--max-area-ratio", type=float, default=1.0, help="mask 占 ROI 面积比例上限，1.0=不过滤")
    ap.add_argument("--max-aspect", type=float, default=0.0, help="框长宽比上限，0=不过滤")
    ap.add_argument("--split-touching", action="store_true", help="对粘连 mask 做分离")
    ap.add_argument("--split-max-erode", type=int, default=40, help="自适应分离最大腐蚀层数，默认 40")
    ap.add_argument("--split-min-seed", type=int, default=30, help="子实例最小种子面积，默认 30")
    ap.add_argument("--split-min-area", type=int, default=2000,
                    help="只对 mask 面积 >= 该值的实例尝试分离，默认 2000")
    ap.add_argument("--grid", action="store_true", help="在 overlay 上画 100px 坐标网格（便于量 ROI）")
    ap.add_argument("--grid-only", action="store_true", help="只输出带网格的原图，不做推理")
    args = ap.parse_args()

    if args.run_id is None:
        args.run_id = "run_" + time.strftime("%Y%m%d_%H%M%S")
    if args.frame_id is None:
        args.frame_id = os.path.splitext(os.path.basename(args.image))[0]

    os.makedirs(args.out_dir, exist_ok=True)

    # 配置落到 CFG
    CFG["roi"] = args.roi
    CFG["roi_margin"] = args.roi_margin
    CFG["roi_crop"] = args.roi_crop
    CFG["nms_iou"] = args.nms_iou
    CFG["min_mask_area"] = args.min_mask_area
    CFG["max_area_ratio"] = args.max_area_ratio
    CFG["max_aspect"] = args.max_aspect
    CFG["split_touching"] = args.split_touching
    CFG["split_max_erode"] = args.split_max_erode
    CFG["split_min_seed"] = args.split_min_seed
    CFG["split_min_area"] = args.split_min_area
    if args.preset == "tight":
        CFG["nms_iou"] = CFG["nms_iou"] or 0.5
        CFG["min_mask_area"] = CFG["min_mask_area"] or 200
        CFG["max_area_ratio"] = 0.35 if CFG["max_area_ratio"] >= 1.0 else CFG["max_area_ratio"]
        CFG["max_aspect"] = CFG["max_aspect"] or 6.0

    # 只出网格图
    if args.grid_only:
        img = Image.open(args.image).convert("RGB")
        draw_overlay(img, [], None, args.roi, grid=True).save(
            os.path.join(args.out_dir, "grid.jpg"), quality=92)
        print(f"[info] 已输出网格图: {os.path.join(args.out_dir, 'grid.jpg')}")
        sys.exit(0)

    print("=" * 60)
    print(f"[info] preset={args.preset}  ROI={CFG['roi']}  split={CFG['split_touching']}")
    t_start = time.time()
    image, instances, filtered = run_segmentation(
        args.image, args.prompt, args.box_thresh, args.text_thresh)
    seg_seconds = time.time() - t_start

    overlay = draw_overlay(image, instances, filtered, CFG["roi"], grid=args.grid)
    overlay.save(os.path.join(args.out_dir, "overlay.jpg"), quality=90)

    depth_info, depth_warnings = build_depth_info(args.depth, args.out_dir)
    json_path, md_path, masks_dir = write_outputs(
        args.out_dir, image, instances, args, seg_seconds, depth_info, depth_warnings, filtered)

    print("=" * 60)
    print(f"完成！共检出 {len(instances)} 个物体，过滤掉 {len(filtered)} 个候选。")
    print(f"segmentation.json: {os.path.abspath(json_path)}")
    print(f"report.md: {os.path.abspath(md_path)}")
    print(f"Mask 目录: {os.path.abspath(masks_dir)}")

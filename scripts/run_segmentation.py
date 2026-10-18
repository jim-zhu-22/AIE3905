"""
最小分割 pipeline：Grounding DINO（transformers） + SAM（ultralytics）
放在项目的 scripts/ 目录下运行。

用法（在项目根目录 D:/AIE3905 下）：
    python scripts/run_segmentation.py ^
        --image "data/Chad_frames_002-005_20261013/images/xxx_color.jpg" ^
        --prompt "bottle. box. bag. can. cup."
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image, ImageDraw

# ---------- 设备 ----------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[info] 使用设备: {DEVICE}")

# ---------- 模型加载（首次联网下载权重，之后走缓存） ----------
print("[info] 正在加载 Grounding DINO（首次约 1.7GB，需联网下载）...")
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

GDINO_NAME = "IDEA-Research/grounding-dino-tiny"
gdino_processor = AutoProcessor.from_pretrained(GDINO_NAME)
gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO_NAME).to(DEVICE)
print("[info] Grounding DINO 加载完成。")

print("[info] 正在加载 SAM（ultralytics）...")
from ultralytics import SAM

SAM_CKPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "models", "sam_b.pt")
if not os.path.exists(SAM_CKPT):
    print(f"[error] 找不到 SAM 权重: {SAM_CKPT}")
    print("[error] 请先运行下载命令（见教程），或把权重放到 models\\ 目录下。")
    sys.exit(1)

sam_model = SAM(SAM_CKPT)
print("[info] SAM 加载完成。")


# ---------- 核心推理 ----------
def run_segmentation(image_path, prompt, box_thresh=0.25, text_thresh=0.20):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"找不到图片: {image_path}")

    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    print(f"[info] 图片尺寸: {image.size}")

    # 1) Grounding DINO：文本 -> bbox
    t0 = time.time()
    inputs = gdino_processor(images=image, text=prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = gdino_model(**inputs)
    # 兼容新旧版 transformers：参数名可能不同，自动适配
    import inspect
    _sig = inspect.signature(gdino_processor.post_process_grounded_object_detection)
    _params = set(_sig.parameters.keys())
    _post_kwargs = {"target_sizes": [image.size[::-1]]}
    if "box_threshold" in _params:
        _post_kwargs["box_threshold"] = box_thresh
    if "text_threshold" in _params:
        _post_kwargs["text_threshold"] = text_thresh
    if "threshold" in _params:
        _post_kwargs["threshold"] = box_thresh
    results = gdino_processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        **_post_kwargs,
    )[0]
    boxes = [b.cpu().numpy().tolist() for b in results["boxes"]]
    scores = [float(s) for s in results["scores"]]
    labels = list(results["labels"])
    t1 = time.time()
    print(f"[info] Grounding DINO 检出 {len(boxes)} 个候选，耗时 {t1 - t0:.2f}s")

    # 2) SAM：每个 bbox -> mask
    instances = []
    t2 = time.time()
    if boxes:
        sam_results = sam_model.predict(source=image_np, bboxes=boxes, verbose=False)
        masks_data = None
        if sam_results and sam_results[0].masks is not None:
            masks_data = sam_results[0].masks.data.cpu().numpy()

        for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            if masks_data is not None and i < len(masks_data):
                mask = masks_data[i] > 0.5
            else:
                mask = np.zeros((image.size[1], image.size[0]), dtype=bool)
            instances.append({
                "instance_id": f"i{i:03d}",
                "bbox_xyxy": [round(float(x), 1) for x in box],
                "det_score": round(float(score), 3),
                "label": label,
                "mask": mask,
            })
    t3 = time.time()
    print(f"[info] SAM 分割完成，耗时 {t3 - t2:.2f}s")

    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return image, instances


# ---------- 可视化 ----------
def draw_overlay(image, instances):
    img = np.array(image).copy()
    for ins in instances:
        mask = ins["mask"]
        color = np.array([0, 255, 0], dtype=np.uint8)
        img[mask] = (0.5 * img[mask] + 0.5 * color).astype(np.uint8)
        pil = Image.fromarray(img)
        draw = ImageDraw.Draw(pil)
        x1, y1, x2, y2 = ins["bbox_xyxy"]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, max(0, y1 - 14)),
                  f"{ins['label']} {ins['det_score']:.2f}", fill="red")
        img = np.array(pil)
    return Image.fromarray(img)


# ---------- CLI ----------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="RGB 图路径")
    ap.add_argument("--prompt", required=True,
                    help='提示词，每个类别后加句号，例如 "bottle. box. bag. can."')
    ap.add_argument("--box-thresh", type=float, default=0.25, help="检测阈值，越小检出越多、误检也越多")
    ap.add_argument("--out-dir", default="output/seg_demo", help="输出目录")
    args = ap.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    masks_dir = os.path.join(out_dir, "masks")
    os.makedirs(masks_dir, exist_ok=True)

    print("=" * 60)
    image, instances = run_segmentation(args.image, args.prompt, args.box_thresh)

    overlay = draw_overlay(image, instances)
    overlay_path = os.path.join(out_dir, "overlay.jpg")
    overlay.save(overlay_path, quality=90)

    json_instances = []
    for ins in instances:
        mask_path = os.path.join(masks_dir, f"{ins['instance_id']}.png")
        Image.fromarray((ins["mask"] * 255).astype(np.uint8)).save(mask_path)
        json_instances.append({
            "instance_id": ins["instance_id"],
            "bbox_xyxy": ins["bbox_xyxy"],
            "det_score": ins["det_score"],
            "label": ins["label"],
            "mask_ref": os.path.relpath(mask_path, out_dir).replace("\\", "/"),
        })

    result = {
        "image": args.image,
        "prompt": args.prompt,
        "box_threshold": args.box_thresh,
        "device": DEVICE,
        "num_instances": len(instances),
        "instances": json_instances,
    }
    json_path = os.path.join(out_dir, "result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print(f"完成！共检出 {len(instances)} 个物体。")
    print(f"叠加图: {os.path.abspath(overlay_path)}")
    print(f"结果 JSON: {os.path.abspath(json_path)}")
    print(f"Mask 目录: {os.path.abspath(masks_dir)}")

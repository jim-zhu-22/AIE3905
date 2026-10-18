"""
批量评估脚本：对数据集的 4 帧依次跑 Grounding DINO + SAM，
并与人工标注（COCO 格式）做 IoU 匹配，输出漏检/误检报告。

用法（在项目根目录 D:/AIE3905 下）：
    python scripts/evaluate_segmentation.py                  # 默认 SAM 1
    python scripts/evaluate_segmentation.py --engine sam2    # SAM 2.1
    python scripts/evaluate_segmentation.py --engine yoloe   # YOLOE（对照）
输出到 output/seg_eval/（sam2 版 → seg_eval_sam2/，yoloe 版 → seg_eval_yoloe/）
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

# 选择引擎：sam1 = Grounding DINO + SAM 1；sam2 = Grounding DINO + SAM 2.1；
# yoloe = YOLOE-11s-Seg（开放词汇，无 DINO）
_ap = argparse.ArgumentParser()
_ap.add_argument("--engine", choices=["sam1", "sam2", "yoloe"], default="sam1",
                 help="sam1=老版SAM，sam2=SAM 2.1，yoloe=YOLOE")
_engine = _ap.parse_args().engine

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if _engine == "sam2":
    from run_segmentation_sam2 import run_segmentation
    ENGINE_NAME = "GroundingDINO + SAM2.1-Tiny"
    OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "output", "seg_eval_sam2")
elif _engine == "yoloe":
    from run_segmentation_yoloe import run_segmentation
    ENGINE_NAME = "YOLOE-11s-Seg (open-vocab, conf=0.05 imgsz=1280)"
    OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "output", "seg_eval_yoloe")
else:
    from run_segmentation import run_segmentation
    ENGINE_NAME = "GroundingDINO + SAM1-ViT-B"
    OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "output", "seg_eval")

DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "data", "Chad_frames_002-005_20261013")
ANN_FILE = os.path.join(DATA_ROOT, "annotations", "instances_bbox_draft.coco.json")

PROMPT = "box. bag. bottle. carton. packet."
IOU_THRESH = 0.5  # 框 IoU 大于等于 0.5 算匹配上


def box_iou(a, b):
    """两个 xyxy 框的 IoU"""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_boxes(gt_boxes, pred_boxes):
    """1对1 贪心匹配，返回 (匹配数, 未匹配的 gt 索引, 未匹配的 pred 索引)"""
    matched_pred = set()
    matched_gt = set()
    pairs = []
    for gi, g in enumerate(gt_boxes):
        best_iou, best_pi = 0.0, -1
        for pi, p in enumerate(pred_boxes):
            if pi in matched_pred:
                continue
            iou = box_iou(g, p)
            if iou > best_iou:
                best_iou, best_pi = iou, pi
        if best_pi >= 0 and best_iou >= IOU_THRESH:
            matched_gt.add(gi)
            matched_pred.add(best_pi)
            pairs.append((gi, best_pi, best_iou))
    miss_gt = sorted(set(range(len(gt_boxes))) - matched_gt)
    false_pred = sorted(set(range(len(pred_boxes))) - matched_pred)
    return len(pairs), miss_gt, false_pred, pairs


def draw_compare(image, gt_boxes, pred_boxes, miss_gt, false_pred, out_path):
    """蓝框=标注，红框=模型；绿虚线=漏检的标注，黄虚线=误检的模型框"""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    for i, g in enumerate(gt_boxes):
        color = "green" if i in miss_gt else "blue"
        draw.rectangle(g, outline=color, width=2)
        draw.text((g[0], g[1] - 12), f"gt{i}", fill=color)
    for i, p in enumerate(pred_boxes):
        color = "yellow" if i in false_pred else "red"
        draw.rectangle(p, outline=color, width=2)
        draw.text((p[0], max(0, p[1] - 12)), f"p{i}", fill=color)
    img.save(out_path)
    return img


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 读标注
    with open(ANN_FILE, "r", encoding="utf-8") as f:
        coco = json.load(f)
    img_meta = {im["id"]: im for im in coco["images"]}
    ann_by_img = {}
    for a in coco["annotations"]:
        ann_by_img.setdefault(a["image_id"], []).append(a)

    summary = []
    all_matched = all_gt = all_fp = 0
    for im in coco["images"]:
        frame_id = im["frame_id"]
        img_path = os.path.join(DATA_ROOT, im["file_name"])
        print(f"\n===== 处理 {frame_id} =====")

        # 标注框（COCO xywh -> xyxy）
        gts = []
        for a in ann_by_img.get(im["id"], []):
            x, y, w, h = a["bbox"]
            gts.append([x, y, x + w, y + h])

        # 跑模型
        t_start = time.time()
        image, instances = run_segmentation(img_path, PROMPT)
        elapsed = time.time() - t_start
        preds = [ins["bbox_xyxy"] for ins in instances]

        # 匹配
        n_match, miss_gt, false_pred, pairs = match_boxes(gts, preds)
        recall = n_match / len(gts) if gts else None
        precision = n_match / len(preds) if preds else None

        print(f"标注 {len(gts)} | 检出 {len(preds)} | 匹配 {n_match} | "
              f"漏检 {len(miss_gt)} | 误检 {len(false_pred)} | 整帧耗时 {elapsed:.1f}s")
        print(f"召回率 {recall:.1%}（标注中找回来的比例）| "
              f"精确率 {precision:.1%}（检出中真的比例）")

        # 对比图
        out_img = os.path.join(OUT_DIR, f"{frame_id}_compare.jpg")
        draw_compare(image, gts, preds, miss_gt, false_pred, out_img)

        summary.append({
            "frame_id": frame_id,
            "gt_count": len(gts),
            "pred_count": len(preds),
            "matched": n_match,
            "miss": len(miss_gt),
            "false_positive": len(false_pred),
            "recall": round(recall, 4) if recall is not None else None,
            "precision": round(precision, 4) if precision is not None else None,
            "latency_s": round(elapsed, 2),
            "compare_image": out_img,
        })
        all_matched += n_match
        all_gt += len(gts)
        all_fp += len(false_pred)

    # 汇总
    overall = {
        "engine": ENGINE_NAME,
        "prompt": PROMPT,
        "iou_threshold": IOU_THRESH,
        "frames": summary,
        "total": {
            "gt_count": all_gt,
            "matched": all_matched,
            "miss": all_gt - all_matched,
            "false_positive": all_fp,
            "recall": round(all_matched / all_gt, 4) if all_gt else None,
            "avg_latency_s": round(sum(s["latency_s"] for s in summary) / len(summary), 2),
        },
        "note": "标注为 needs_review 状态，本报告为参考性对照，不等同于正式验收指标。",
    }
    out_json = os.path.join(OUT_DIR, "summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(overall, f, ensure_ascii=False, indent=2)

    print("\n================ 汇总 ================")
    print(f"{'帧':<25}{'标注':<6}{'检出':<6}{'匹配':<6}{'漏检':<6}{'误检':<6}{'耗时s':<8}")
    for s in summary:
        print(f"{s['frame_id']:<25}{s['gt_count']:<6}{s['pred_count']:<6}{s['matched']:<6}"
              f"{s['miss']:<6}{s['false_positive']:<6}{s['latency_s']:<8}")
    t = overall["total"]
    print(f"\n总计：标注 {t['gt_count']}，匹配 {t['matched']}，漏检 {t['miss']}，"
          f"误检 {t['false_positive']}，召回率 {t['recall']:.1%}")
    print(f"完整报告: {os.path.abspath(out_json)}")
    print(f"对比图目录: {os.path.abspath(OUT_DIR)}")


if __name__ == "__main__":
    main()

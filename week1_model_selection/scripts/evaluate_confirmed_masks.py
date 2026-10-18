"""Evaluate predicted instance masks against complete, human-confirmed frames.

Only JSON frames with frame_annotation_complete=true and all instances marked
confirmed are included. Mask PNGs must match the original RGB dimensions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as source:
        if source.size != size:
            raise ValueError(f"Mask size {source.size} differs from RGB size {size}: {path}")
        mask = np.asarray(source.convert("L"))
    if not np.isin(mask, [0, 255]).all():
        raise ValueError(f"Mask must contain only 0 and 255: {path}")
    return mask > 0


def mask_scores(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    intersection = int(np.logical_and(a, b).sum())
    a_count, b_count = int(a.sum()), int(b.sum())
    union = a_count + b_count - intersection
    return (intersection / union if union else 0.0,
            2 * intersection / (a_count + b_count) if a_count + b_count else 0.0)


def match_masks(gt_masks: list[np.ndarray], pred_masks: list[np.ndarray], threshold: float = 0.5):
    candidates = []
    for gi, gt in enumerate(gt_masks):
        for pi, pred in enumerate(pred_masks):
            iou, dice = mask_scores(gt, pred)
            candidates.append((iou, dice, gi, pi))
    candidates.sort(reverse=True)
    used_gt, used_pred = set(), set()
    pairs = []
    for iou, dice, gi, pi in candidates:
        if iou < threshold:
            break
        if gi not in used_gt and pi not in used_pred:
            used_gt.add(gi)
            used_pred.add(pi)
            pairs.append({"gt_index": gi, "pred_index": pi,
                          "mask_iou": round(iou, 4), "dice": round(dice, 4)})
    return pairs, len(gt_masks) - len(used_gt), len(pred_masks) - len(used_pred)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-dir", required=True, type=Path)
    parser.add_argument("--pred-root", type=Path, default=ROOT / "results" / "own_segmentation")
    parser.add_argument("--mode", choices=["full", "tiled"], required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    rows, skipped = [], []
    for ann_path in sorted(args.annotations_dir.glob("*.json")):
        truth = json.loads(ann_path.read_text(encoding="utf-8"))
        if truth.get("frame_annotation_complete") is not True:
            skipped.append({"file": str(ann_path), "reason": "frame not confirmed complete"})
            continue
        instances = truth.get("instances", [])
        if any(item.get("annotation_status") != "confirmed" for item in instances):
            skipped.append({"file": str(ann_path), "reason": "instance needs review"})
            continue
        frame_id = truth["frame_id"]
        pred_file = args.pred_root / args.mode / frame_id / "result.json"
        if not pred_file.is_file():
            skipped.append({"file": str(ann_path), "reason": "prediction missing"})
            continue
        width, height = truth["image_size"]
        size = (width, height)
        gt_masks = [load_mask((ann_path.parent / item["mask_path"]).resolve(), size)
                    for item in instances]
        prediction = json.loads(pred_file.read_text(encoding="utf-8"))
        pred_masks = [load_mask(pred_file.parent / item["mask_ref"], size)
                      for item in prediction["instances"]]
        pairs, fn, fp = match_masks(gt_masks, pred_masks, args.threshold)
        rows.append({"frame_id": frame_id, "gt": len(gt_masks), "pred": len(pred_masks),
                     "tp": len(pairs), "fn": fn, "fp": fp, "pairs": pairs})
    tp, fn, fp = (sum(row[key] for row in rows) for key in ("tp", "fn", "fp"))
    matched = [pair for row in rows for pair in row["pairs"]]
    result = {
        "mode": args.mode,
        "matching": f"one-to-one mask IoU >= {args.threshold}",
        "included_frames": len(rows),
        "skipped": skipped,
        "total": {
            "tp": tp, "fn": fn, "fp": fp,
            "recall": round(tp / (tp + fn), 4) if tp + fn else None,
            "precision": round(tp / (tp + fp), 4) if tp + fp else None,
            "mean_matched_mask_iou": round(sum(p["mask_iou"] for p in matched) / len(matched), 4) if matched else None,
            "mean_matched_dice": round(sum(p["dice"] for p in matched) / len(matched), 4) if matched else None,
        },
        "frames": rows,
    }
    target = args.pred_root / f"confirmed_mask_eval_{args.mode}.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["total"], ensure_ascii=False, indent=2))
    print(f"Included frames: {len(rows)}, skipped: {len(skipped)}. Output: {target.resolve()}")


if __name__ == "__main__":
    main()

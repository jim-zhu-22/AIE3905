r"""Independent full-frame vs fixed-tile Grounding DINO + SAM practice.

Example (PowerShell, repository root):
  ..\.venv\Scripts\python.exe scripts\run_own_segmentation.py --mode both

The four COCO boxes are draft labels. Metrics from them are exploratory only.
The fifth image has no annotation and is exported for visual review only.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
from segmentation_practice.pipeline import GroundedSegmenter, Prediction, box_iou, overlay, save_prediction

DEFAULT_DATA_ROOT = PROJECT_ROOT / "Chad_frames_002-005_20261013"
DEFAULT_KIT_ROOT = PROJECT_ROOT / "AIE3905_Annotation_Kit"
DEFAULT_EXTRA_ID = "2026_08_26_09_51_53_346"
DEFAULT_PROMPT = "box. bag. bottle. carton. packet."


def match_boxes(gt_boxes: list[list[float]], pred_boxes: list[list[float]], threshold: float = 0.5):
    """One-to-one global IoU greedy matching; fixed for every experiment arm."""
    candidates = [
        (box_iou(gt, pred), gi, pi)
        for gi, gt in enumerate(gt_boxes)
        for pi, pred in enumerate(pred_boxes)
    ]
    candidates.sort(reverse=True)
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    pairs: list[dict[str, float | int]] = []
    for iou, gi, pi in candidates:
        if iou < threshold:
            break
        if gi not in used_gt and pi not in used_pred:
            used_gt.add(gi)
            used_pred.add(pi)
            pairs.append({"gt_index": gi, "pred_index": pi, "box_iou": round(iou, 4)})
    misses = sorted(set(range(len(gt_boxes))) - used_gt)
    false_positives = sorted(set(range(len(pred_boxes))) - used_pred)
    return pairs, misses, false_positives


def draw_diagnostics(prediction: Prediction, gt_boxes, misses, false_positives, target: Path) -> None:
    image = overlay(prediction)
    draw = ImageDraw.Draw(image)
    for index, box in enumerate(gt_boxes):
        color = "#ff7a00" if index in misses else "#174fff"
        draw.rectangle(box, outline=color, width=3)
        draw.text((box[0], max(0, box[1] - 26)), f"GT{index}", fill=color)
    for index in false_positives:
        box = prediction.instances[index].bbox_xyxy
        draw.rectangle(box, outline="#ff00ff", width=3)
        draw.text((box[0], box[1]), f"FP{index}", fill="#ff00ff")
    image.save(target, quality=92)


def summarize(rows: list[dict]) -> dict:
    annotated = [row for row in rows if row["gt_count"] is not None]
    tp = sum(row["tp"] for row in annotated)
    gt = sum(row["gt_count"] for row in annotated)
    fp = sum(row["fp"] for row in annotated)
    latencies = [row["latency_ms"]["total"] for row in rows]
    return {
        "annotated_frames": len(annotated),
        "all_frames": len(rows),
        "gt_draft_count": gt,
        "tp_box_iou_0_5": tp,
        "fn": gt - tp,
        "fp": fp,
        "recall_exploratory": round(tp / gt, 4) if gt else None,
        "precision_exploratory": round(tp / (tp + fp), 4) if tp + fp else None,
        "median_total_latency_ms": round(statistics.median(latencies), 2) if latencies else None,
        "mean_total_latency_ms": round(statistics.mean(latencies), 2) if latencies else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--kit-root", type=Path, default=DEFAULT_KIT_ROOT)
    parser.add_argument("--extra-frame-id", default=DEFAULT_EXTRA_ID)
    parser.add_argument("--no-extra", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None, help="Smoke test using the first N frames")
    parser.add_argument("--mode", choices=["full", "tiled", "both"], default="both")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--dino-model", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--sam-weight", default=str(PROJECT_ROOT / "sam2.1_t.pt"))
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.20)
    parser.add_argument("--nms-threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "own_segmentation")
    args = parser.parse_args()

    annotation_file = args.data_root / "annotations" / "instances_bbox_draft.coco.json"
    coco = json.loads(annotation_file.read_text(encoding="utf-8"))
    by_image: dict[int, list[dict]] = {}
    for annotation in coco["annotations"]:
        by_image.setdefault(annotation["image_id"], []).append(annotation)
    frames = []
    for image_meta in coco["images"]:
        gt = []
        for annotation in by_image.get(image_meta["id"], []):
            x, y, w, h = annotation["bbox"]
            gt.append([x, y, x + w, y + h])
        frames.append((image_meta["frame_id"], args.data_root / image_meta["file_name"], gt))
    if not args.no_extra:
        extra_path = args.kit_root / "data" / "raw" / f"{args.extra_frame_id}_color.jpg"
        frames.append((args.extra_frame_id, extra_path, None))
    if args.max_frames is not None:
        frames = frames[: args.max_frames]
    for frame_id, path, _ in frames:
        if not path.is_file():
            raise FileNotFoundError(f"Missing input frame {frame_id}: {path}")
    args.output.mkdir(parents=True, exist_ok=True)

    model = GroundedSegmenter(
        dino_model=args.dino_model,
        sam_weight=args.sam_weight,
        device=args.device,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        nms_threshold=args.nms_threshold,
    )
    modes = ["full", "tiled"] if args.mode == "both" else [args.mode]
    rows: dict[str, list[dict]] = {mode: [] for mode in modes}
    for frame_id, path, gt_boxes in frames:
        image = Image.open(path).convert("RGB")
        for mode in modes:
            print(f"{frame_id} {mode}: start", flush=True)
            prediction = model.predict(image, args.prompt, mode)
            frame_dir = args.output / mode / frame_id
            save_prediction(prediction, frame_dir)
            row = {
                "frame_id": frame_id,
                "image_path": str(path),
                "mode": mode,
                "gt_status": "needs_review" if gt_boxes is not None else "unlabeled",
                "gt_count": len(gt_boxes) if gt_boxes is not None else None,
                "pred_count": len(prediction.instances),
                "raw_detections": prediction.raw_detections,
                "latency_ms": prediction.latency_ms,
            }
            if gt_boxes is not None:
                pairs, misses, false_positives = match_boxes(
                    gt_boxes, [item.bbox_xyxy for item in prediction.instances]
                )
                row.update({
                    "tp": len(pairs),
                    "fn": len(misses),
                    "fp": len(false_positives),
                    "pairs": pairs,
                    "miss_gt_indices": misses,
                    "false_pred_indices": false_positives,
                })
                draw_diagnostics(prediction, gt_boxes, misses, false_positives,
                                 frame_dir / "diagnostics.jpg")
            rows[mode].append(row)
            print(f"  predictions={row['pred_count']} time={prediction.latency_ms['total']/1000:.2f}s", flush=True)

    totals = {mode: summarize(items) for mode, items in rows.items()}
    result = {
        "evaluation_status": "exploratory_bbox_only; draft boxes, incomplete-frame status unknown; no ground-truth masks",
        "config": {
            "dino_model": args.dino_model,
            "sam_weight": args.sam_weight,
            "device": model.device,
            "prompt": args.prompt,
            "box_threshold": args.box_threshold,
            "text_threshold": args.text_threshold,
            "nms_threshold": args.nms_threshold,
            "tile_size": [704, 528],
            "matching": "one-to-one global-greedy box IoU >= 0.5",
            "tile_method": "detect on four tiles, merge boxes, run SAM once on full RGB image",
        },
        "frames": rows,
        "totals": totals,
    }
    (args.output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(totals, ensure_ascii=False, indent=2))
    print(f"Results: {args.output.resolve()}")


if __name__ == "__main__":
    main()

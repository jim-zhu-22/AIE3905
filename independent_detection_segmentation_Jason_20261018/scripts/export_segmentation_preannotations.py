"""Convert one model result into schema-v2 box drafts for the offline annotation kit.

The kit import replaces the current browser draft. Export that draft first.
Mask PNGs remain in the model result directory for separate contour review.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-id", default="2026_08_26_09_51_53_346")
    parser.add_argument("--mode", choices=["full", "tiled"], default="tiled")
    parser.add_argument("--kit-root", type=Path, default=PROJECT_ROOT / "AIE3905_Annotation_Kit")
    parser.add_argument("--pred-root", type=Path, default=ROOT / "results" / "own_segmentation")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "own_segmentation" / "preannotations.json")
    args = parser.parse_args()

    with (args.kit_root / "demo" / "pairs.csv").open(encoding="utf-8-sig", newline="") as stream:
        frames = {row["frame_id"]: row for row in csv.DictReader(stream)}
    if args.frame_id not in frames:
        raise ValueError(f"Frame not found in annotation kit: {args.frame_id}")
    frame = frames[args.frame_id]
    width, height = int(frame["width"]), int(frame["height"])
    result_file = args.pred_root / args.mode / args.frame_id / "result.json"
    result = json.loads(result_file.read_text(encoding="utf-8"))
    objects = []
    for index, prediction in enumerate(result["instances"], 1):
        x1, y1, x2, y2 = prediction["bbox_xyxy"]
        box = [max(0, math.floor(x1)), max(0, math.floor(y1)),
               min(width, math.ceil(x2)), min(height, math.ceil(y2))]
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        objects.append({
            "object_id": f"object_{index:04d}",
            "frame_id": args.frame_id,
            "rgb_file": f"{args.frame_id}_color.jpg",
            "depth_file": f"{args.frame_id}_depth.png",
            "image_size": [width, height],
            "time_group": frame["time_group"],
            "object_bbox_xyxy": box,
            "contact_bbox_xyxy": None,
            "coordinate_convention": "original_image_pixels_xyxy_half_open",
            "material": "",
            "sku_id": "",
            "annotation_status": "needs_review",
        })
    exported = {
        "schema_version": 2,
        "dataset_id": "yingjie_rgbd_20260826_245",
        "annotator": "model_preannotation",
        "depth_unit": "unconfirmed",
        "alignment": "unconfirmed",
        "objects": objects,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(exported, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Exported {len(objects)} draft boxes to {args.output.resolve()}")
    print("Before importing into the annotation kit, export any existing browser draft: import replaces it.")


if __name__ == "__main__":
    main()

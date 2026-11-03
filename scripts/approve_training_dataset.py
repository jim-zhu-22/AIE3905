"""Export only human-confirmed preannotations as a material training manifest."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from material_recognition import LABELS


def export_manifest(reviewed: Path, output: Path) -> dict:
    data = json.loads(reviewed.read_text(encoding="utf-8"))
    if data.get("schema_version") != "2.0-preannotation":
        raise ValueError("expected schema_version=2.0-preannotation")
    rows = []
    rejected = 0
    for item in data.get("objects", []):
        status = item.get("annotation_status")
        if status != "confirmed":
            rejected += 1
            continue
        label = item.get("final_material")
        if label not in LABELS:
            raise ValueError(f"{item.get('object_id')}: confirmed final_material must be one of {LABELS}")
        split = item.get("split")
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"{item.get('object_id')}: confirmed item needs train/validation/test split")
        crop = reviewed.parent / item["crop_path"]
        if not crop.is_file():
            raise FileNotFoundError(f"missing crop for {item['object_id']}: {crop}")
        bbox = item.get("object_bbox_xyxy") or ["", "", "", ""]
        rows.append({
            "sample_id": item["object_id"], "split": split, "label": label,
            "label_id": LABELS.index(label),
            "image_path": item["crop_path"], "source": "model_preannotation_human_confirmed",
            "sku_id": item.get("sku_id", ""), "batch_id": item.get("time_group", ""),
            "parent_image_id": item.get("frame_id", ""),
            "roi_x1": bbox[0], "roi_y1": bbox[1], "roi_x2": bbox[2], "roi_y2": bbox[3],
            "annotation_status": "confirmed",
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["sample_id", "split", "label", "label_id", "image_path", "source", "sku_id",
              "batch_id", "parent_image_id", "roi_x1", "roi_y1", "roi_x2", "roi_y2",
              "annotation_status"]
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {"output": str(output), "confirmed": len(rows), "not_exported": rejected,
               "status": "training_ready" if rows else "empty"}
    output.with_name("training_manifest_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_manifest(args.reviewed, args.output), ensure_ascii=False))


if __name__ == "__main__":
    main()

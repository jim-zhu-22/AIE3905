"""Validate the exported files and joins in a pipeline-1.0 frame bundle."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from perception.io.pipeline_contract import SavedSegmenter
from perception.io.manifest import write_json


def validate(root):
    root = Path(root).resolve()
    stages = {name: json.loads((root / f"{name}.json").read_text(encoding="utf-8"))
              for name in ("preprocess", "segmentation", "material_input", "material_result", "result")}
    result = stages["result"]
    for stage in stages.values():
        for key in ("schema_version", "run_id", "frame_id"):
            if stage[key] != result[key]:
                raise ValueError(f"Stage join mismatch: {key}")
    segmentation = SavedSegmenter(root / "segmentation.json")
    with Image.open(root / result["rgb_path"]) as image:
        segmentation.predict(image, result["prompt"], "imported")
    rows = stages["segmentation"]["instances"]
    ids = [v["instance_id"] for v in rows]
    for values in (result["instances"], stages["material_input"]["items"], stages["material_result"]["items"]):
        if [v["instance_id"] for v in values] != ids:
            raise ValueError("Instance IDs/order differ between stages")
    if result["instance_count"] != len(ids):
        raise ValueError("Final instance_count mismatch")
    for item, roi, material, final in zip(rows, stages["material_input"]["items"], stages["material_result"]["items"], result["instances"]):
        for key in ("object_label", "bbox_xyxy", "mask_path"):
            if item[key] != final[key]:
                raise ValueError(f"Final instance changed segmentation {key}")
        if roi["roi_id"] != material["roi_id"] or material != final["material"]:
            raise ValueError("Material/ROI join mismatch")
        if roi["crop_path"]:
            with Image.open(root / roi["crop_path"]) as crop, Image.open(root / roi["crop_mask_path"]) as mask:
                box = roi["contact_bbox_xyxy"]
                size = (box[2] - box[0], box[3] - box[1])
                if crop.size != size or mask.size != size or list(size) != roi["crop_size"]:
                    raise ValueError("ROI crop size mismatch")
        if material["scores"] is not None:
            values = np.array(list(material["scores"].values()), dtype=float)
            if not np.isfinite(values).all() or np.any(values < 0) or not np.isclose(values.sum(), 1, atol=1e-5):
                raise ValueError("Invalid material probabilities")
        if material["status"] != "accepted" and not material["reason"]:
            raise ValueError("Unaccepted material requires reason")
    expected = "no_match" if not ids else "ok" if all(v["material"]["status"] == "accepted" for v in result["instances"]) else "partial"
    if result["status"] != expected:
        raise ValueError("Final status does not propagate material states")
    for key in ("preprocess_ref", "segmentation_ref", "material_input_ref", "material_result_ref"):
        if not (root / result[key]).is_file():
            raise ValueError(f"Missing stage file {key}")
    if result["depth"].get("depth_path") and "depth_file_missing" not in result["warnings"]:
        if not (root / result["depth"]["depth_path"]).is_file():
            raise ValueError("Missing depth reference without warning")
    return {"path": str(root), "status": "passed", "instance_count": len(ids), "pipeline_status": result["status"],
            "checks": ["stage_joins", "unique_ids", "binary_masks", "mask_boxes", "rgb_size", "roi_files_and_sizes",
                       "material_scores", "final_status", "depth_reference"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundles", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results = [validate(root) for root in args.bundles]
    if args.output:
        write_json(args.output, results)
    print(json.dumps(results, ensure_ascii=False, indent=2))

"""Compare saved predictions by spatial masks, never by instance list position.

Agreement between two predictions is not accuracy against human ground truth.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from perception.io.manifest import write_json
from perception.io.pipeline_contract import mask_box
from perception.runtime import sha256


def read(path):
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    rows = []
    for item in data["instances"]:
        with Image.open(path.parent / (item.get("mask_path") or item["mask_ref"])) as im:
            mask = np.array(im) > 0
        rows.append({**item, "mask": mask, "mask_box": mask_box(mask)})
    return rows


def compare(left, right, threshold=.5):
    a, b = read(left), read(right)
    candidates = []
    for i, first in enumerate(a):
        for j, second in enumerate(b):
            if first["mask"].shape != second["mask"].shape:
                raise ValueError("Prediction images have different dimensions")
            box1, box2 = first["mask_box"], second["mask_box"]
            x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
            x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
            if x1 >= x2 or y1 >= y2:
                continue
            inter = int((first["mask"][y1:y2, x1:x2] & second["mask"][y1:y2, x1:x2]).sum())
            union = int(first["mask"].sum() + second["mask"].sum()) - inter
            iou = inter / union
            if iou >= threshold:
                candidates.append((iou, i, j))
    used_a, used_b, pairs = set(), set(), []
    for iou, i, j in sorted(candidates, reverse=True):
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        first, second = a[i], b[j]
        pairs.append({"left_id": first["instance_id"], "right_id": second["instance_id"], "mask_iou": iou,
                      "left_label": first.get("object_label", first.get("label")),
                      "right_label": second.get("object_label", second.get("label")),
                      "left_material": first.get("material", {}).get("top1_label", first.get("material", {}).get("label")),
                      "right_material": second.get("material", {}).get("top1_label", second.get("material", {}).get("label"))})
    return {"comparison": "prediction_agreement_not_ground_truth", "matching": "greedy_one_to_one_descending_mask_iou",
            "threshold": threshold, "left": str(left), "right": str(right),
            "left_sha256": sha256(left), "right_sha256": sha256(right),
            "left_count": len(a), "right_count": len(b), "matched_count": len(pairs),
            "mean_mask_iou": float(np.mean([p["mask_iou"] for p in pairs])) if pairs else None,
            "min_mask_iou": min([p["mask_iou"] for p in pairs], default=None),
            "different_instance_id_count": sum(p["left_id"] != p["right_id"] for p in pairs),
            "unmatched_left": [v["instance_id"] for i, v in enumerate(a) if i not in used_a],
            "unmatched_right": [v["instance_id"] for i, v in enumerate(b) if i not in used_b], "pairs": pairs}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", required=True, type=Path)
    parser.add_argument("--right", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = compare(args.left.resolve(), args.right.resolve())
    write_json(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k != "pairs"}, ensure_ascii=False, indent=2))

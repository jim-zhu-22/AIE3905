"""Build model-preannotated training candidates from RGB-D frame pairs.

This command never creates confirmed labels.  It runs the selected instance
pipeline (Grounding DINO + SAM 2.1 Tiny), crops each predicted instance, runs
the ResNet-18 material classifier, and writes candidates with
``annotation_status=needs_review``.  Human review is exported by
``approve_training_dataset.py``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image


MATERIALS = ("paper", "plastic", "metal", "wood", "fabric")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mask_box(mask) -> tuple[int, int, int, int] | None:
    ys, xs = mask.nonzero()
    if len(xs) == 0:
        return None
    return int(min(xs)), int(min(ys)), int(max(xs)) + 1, int(max(ys)) + 1


def _priority(det_score: float, material_score: float, top2_gap: float,
              high_score: float, high_gap: float) -> str:
    if det_score >= high_score and material_score >= high_score and top2_gap >= high_gap:
        return "quick_confirm"
    if material_score < 0.60 or top2_gap < 0.15:
        return "重点复核"
    return "常规复核"


def _load_pairs(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    required = {"frame_id", "rgb_path", "depth_path"}
    missing = required - set(rows[0] if rows else ())
    if missing:
        raise ValueError(f"pairs.csv missing columns: {sorted(missing)}")
    return rows


def _resolve(value: str, base: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def build(args: argparse.Namespace) -> dict[str, Any]:
    # Imports are delayed so --help and schema-only tests do not require GPU packages.
    from week1_model_selection.segmentation_practice.pipeline import GroundedSegmenter
    from material_recognition.infer import Predictor

    pairs = _load_pairs(args.pairs)
    if args.max_frames:
        pairs = pairs[:args.max_frames]
    args.output.mkdir(parents=True, exist_ok=True)
    crops_dir = args.output / "crops"
    masks_dir = args.output / "masks"
    crops_dir.mkdir(exist_ok=True)
    masks_dir.mkdir(exist_ok=True)

    segmenter = GroundedSegmenter(
        sam_weight=str(args.sam_weight), device=args.device,
        box_threshold=args.box_threshold, text_threshold=args.text_threshold,
        nms_threshold=args.nms_threshold,
    )
    classifier = Predictor(args.material_checkpoint)
    objects: list[dict[str, Any]] = []
    for frame in pairs:
        rgb_path = _resolve(frame["rgb_path"], args.pairs.parent)
        with Image.open(rgb_path) as source:
            image = source.convert("RGB")
        prediction = segmenter.predict(image, args.prompt, args.mode)
        frame_dir = args.output / "frames" / frame["frame_id"]
        frame_dir.mkdir(parents=True, exist_ok=True)
        for index, instance in enumerate(prediction.instances):
            instance_id = f"{frame['frame_id']}_object_{index:04d}"
            box = _mask_box(instance.mask)
            if box is None:
                continue
            x1, y1, x2, y2 = box
            crop = image.crop(box)
            crop_path = crops_dir / f"{instance_id}.png"
            crop.save(crop_path)
            mask_path = masks_dir / f"{instance_id}.png"
            Image.fromarray((instance.mask.astype("uint8") * 255)).save(mask_path)
            material = classifier.predict(crop)
            scores = sorted(material["scores"].values(), reverse=True)
            gap = scores[0] - scores[1] if len(scores) > 1 else scores[0]
            objects.append({
                "object_id": instance_id,
                "frame_id": frame["frame_id"],
                "rgb_path": str(rgb_path),
                "depth_path": frame["depth_path"],
                "image_size": [image.width, image.height],
                "time_group": frame.get("time_group", ""),
                "sku_id": frame.get("sku_id", ""),
                "sequence_id": frame.get("sequence_id", frame.get("time_group", "")),
                "split": frame.get("split", "unassigned"),
                "object_bbox_xyxy": [int(round(v)) for v in instance.bbox_xyxy],
                "contact_bbox_xyxy": None,
                "mask_path": str(mask_path.relative_to(args.output)).replace("\\", "/"),
                "crop_path": str(crop_path.relative_to(args.output)).replace("\\", "/"),
                "model_material": material["label"],
                "material_scores": material["scores"],
                "material_confidence": material["score"],
                "top2_gap": round(gap, 6),
                "detection_score": instance.det_score,
                "material_model_version": material["model_version"],
                "segmentation_model": prediction.model_info,
                "review_priority": _priority(instance.det_score, material["score"], gap,
                                               args.high_score, args.high_gap),
                "annotation_status": "needs_review",
                "final_material": None,
                "reviewer_id": None,
                "review_notes": "",
            })

    result = {
        "schema_version": "2.0-preannotation",
        "dataset_status": "model_preannotated_not_training_ready",
        "prompt": args.prompt,
        "source_pairs": str(args.pairs),
        "source_pairs_sha256": _sha256(args.pairs),
        "objects": objects,
        "review_rule": "Only human-confirmed objects may be exported to training_manifest.csv",
    }
    (args.output / "preannotations.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    template = Path(__file__).with_name("annotation_review.html")
    (args.output / "review.html").write_text(template.read_text(encoding="utf-8"), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--material-checkpoint", type=Path, required=True)
    parser.add_argument("--sam-weight", type=Path, default=Path("week1_model_selection/sam2.1_t.pt"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", default="box. bag. bottle. carton. packet.")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--mode", choices=("full", "tiled"), default="tiled")
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.20)
    parser.add_argument("--nms-threshold", type=float, default=0.50)
    parser.add_argument("--high-score", type=float, default=0.85)
    parser.add_argument("--high-gap", type=float, default=0.20)
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()
    result = build(args)
    print(json.dumps({"output": str(args.output.resolve()), "objects": len(result["objects"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()

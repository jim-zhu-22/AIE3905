"""Validate one frame of RGB-D instance annotations (schema v1)."""

import argparse
import json
from pathlib import Path

from PIL import Image


GEOMETRY = {"cuboid", "cylinder", "bottle", "vacuum_bag", "puffed_bag", "irregular", "unknown"}
MATERIAL = {"paper", "plastic", "metal", "wood", "fabric", "unknown"}
DEFORMABILITY = {"rigid", "deformable", "uncertain"}
STATUS = {"needs_review", "confirmed", "rejected"}


def validate(data: dict, base: Path, check_files: bool = True) -> list[str]:
    errors: list[str] = []

    def require(obj, key, where):
        value = obj.get(key)
        if value is None or value == "":
            errors.append(f"{where}.{key}: required")
        return value

    if data.get("schema_version") != "1.0":
        errors.append("schema_version: expected '1.0'")
    require(data, "frame_id", "frame")
    size = data.get("image_size")
    if not (isinstance(size, list) and len(size) == 2 and all(type(v) is int and v > 0 for v in size)):
        errors.append("image_size: expected [positive width, positive height]")
        size = None
    if data.get("alignment") not in {"unconfirmed", "aligned_to_rgb", "unaligned"}:
        errors.append("alignment: invalid value")
    if not isinstance(data.get("depth_unit"), str) or not data["depth_unit"]:
        errors.append("depth_unit: required string")
    for key in ("rgb_path", "depth_path"):
        value = require(data, key, "frame")
        if check_files and isinstance(value, str) and not (base / value).is_file():
            errors.append(f"{key}: file not found: {value}")
        elif check_files and isinstance(value, str):
            try:
                with Image.open(base / value) as im:
                    if key == "rgb_path" and size and list(im.size) != size:
                        errors.append("rgb_path: dimensions differ from image_size")
                    if key == "depth_path" and im.mode not in {"I;16", "I;16L", "I;16B", "I"}:
                        errors.append("depth_path: expected single-channel 16-bit depth")
            except (OSError, ValueError) as exc:
                errors.append(f"{key}: cannot decode ({exc})")

    instances = data.get("instances")
    targets = data.get("targets")
    if not isinstance(instances, list):
        errors.append("instances: expected list")
        instances = []
    if not isinstance(targets, list):
        errors.append("targets: expected list")
        targets = []

    ids = set()
    for i, obj in enumerate(instances):
        where = f"instances[{i}]"
        if not isinstance(obj, dict):
            errors.append(f"{where}: expected object")
            continue
        instance_id = require(obj, "instance_id", where)
        if not isinstance(instance_id, str):
            errors.append(f"{where}.instance_id: expected string")
            instance_id = None
        if instance_id in ids:
            errors.append(f"{where}.instance_id: duplicate")
        ids.add(instance_id)
        for key, allowed in (("geometry", GEOMETRY), ("material", MATERIAL),
                             ("deformability", DEFORMABILITY), ("annotation_status", STATUS)):
            if obj.get(key) not in allowed:
                errors.append(f"{where}.{key}: invalid value")
        box = obj.get("bbox_xyxy")
        valid_box = (size is not None and isinstance(box, list) and len(box) == 4
                     and all(type(v) is int for v in box)
                     and 0 <= box[0] < box[2] <= size[0]
                     and 0 <= box[1] < box[3] <= size[1])
        if not valid_box:
            errors.append(f"{where}.bbox_xyxy: invalid half-open box")
        contact = obj.get("contact_bbox_xyxy")
        if contact is not None:
            if not (valid_box and isinstance(contact, list) and len(contact) == 4
                    and all(type(v) is int for v in contact)
                    and box[0] <= contact[0] < contact[2] <= box[2]
                    and box[1] <= contact[1] < contact[3] <= box[3]):
                errors.append(f"{where}.contact_bbox_xyxy: must be inside object box")
        mask_path = obj.get("mask_path")
        if obj.get("annotation_status") == "confirmed" and not mask_path:
            errors.append(f"{where}.mask_path: required for confirmed instance")
        if mask_path and check_files:
            path = base / mask_path
            if not path.is_file():
                errors.append(f"{where}.mask_path: file not found: {mask_path}")
            else:
                try:
                    with Image.open(path) as im:
                        if im.mode != "L" or (size and list(im.size) != size):
                            errors.append(f"{where}.mask_path: expected 8-bit grayscale at RGB size")
                        else:
                            extrema = im.getextrema()
                            if extrema == (0, 0):
                                errors.append(f"{where}.mask_path: empty foreground")
                            elif set(im.getdata()) - {0, 255}:
                                errors.append(f"{where}.mask_path: values must be 0 or 255")
                            elif valid_box:
                                foreground_box = im.point(lambda p: 255 if p else 0).getbbox()
                                if foreground_box and not (box[0] <= foreground_box[0]
                                        and box[1] <= foreground_box[1]
                                        and foreground_box[2] <= box[2]
                                        and foreground_box[3] <= box[3]):
                                    errors.append(f"{where}.bbox_xyxy: does not contain mask")
                except (OSError, ValueError) as exc:
                    errors.append(f"{where}.mask_path: cannot decode ({exc})")

    prompt_ids = set()
    for i, target in enumerate(targets):
        where = f"targets[{i}]"
        if not isinstance(target, dict):
            errors.append(f"{where}: expected object")
            continue
        prompt_id = require(target, "prompt_id", where)
        require(target, "prompt", where)
        if not isinstance(prompt_id, str):
            errors.append(f"{where}.prompt_id: expected string")
            prompt_id = None
        if prompt_id in prompt_ids:
            errors.append(f"{where}.prompt_id: duplicate")
        prompt_ids.add(prompt_id)
        refs = target.get("instance_ids")
        if not isinstance(refs, list):
            errors.append(f"{where}.instance_ids: expected list")
        else:
            for ref in refs:
                if ref not in ids:
                    errors.append(f"{where}.instance_ids: unknown instance {ref}")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotation", type=Path)
    parser.add_argument("--skip-files", action="store_true", help="Check structure without referenced image files")
    args = parser.parse_args()
    data = json.loads(args.annotation.read_text(encoding="utf-8"))
    errors = validate(data, args.annotation.parent, not args.skip_files)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print("Annotation valid")


if __name__ == "__main__":
    main()

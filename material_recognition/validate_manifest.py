"""Check the published split and a future enterprise split for leakage."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from . import LABELS


def validate(path: Path, check_files: bool = True):
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError("Manifest has no rows")
    seen_ids = set()
    by_split = Counter()
    by_class = defaultdict(Counter)
    sku_splits = defaultdict(set)
    parent_splits = defaultdict(set)
    for row in rows:
        sample_id = row["sample_id"]
        if sample_id in seen_ids:
            raise ValueError(f"Duplicate sample_id: {sample_id}")
        seen_ids.add(sample_id)
        split = row["split"]
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Unknown split: {split}")
        label_id = int(row["label_id"])
        if not 0 <= label_id < len(LABELS) or LABELS[label_id] != row["label"]:
            raise ValueError(f"Invalid label mapping: {sample_id}")
        if check_files and not (path.parent / row["image_path"]).is_file():
            raise FileNotFoundError(path.parent / row["image_path"])
        by_split[split] += 1
        by_class[split][row["label"]] += 1
        sku = row.get("sku_id", "")
        if sku:
            sku_splits[sku].add(split)
        parent = row.get("parent_image_id", "")
        if parent:
            parent_splits[parent].add(split)
    leaks = {sku: sorted(splits) for sku, splits in sku_splits.items() if len(splits) > 1}
    if leaks:
        raise ValueError(f"SKU appears in multiple splits: {list(leaks.items())[:5]}")
    parent_leaks = {parent: sorted(splits) for parent, splits in parent_splits.items() if len(splits) > 1}
    if parent_leaks:
        raise ValueError(f"Source image appears in multiple splits: {list(parent_leaks.items())[:5]}")
    return {"rows": len(rows), "by_split": dict(by_split),
            "by_class": {split: dict(counts) for split, counts in by_class.items()},
            "unique_skus": len(sku_splits), "unique_source_images": len(parent_splits)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--skip-files", action="store_true")
    args = parser.parse_args()
    print(json.dumps(validate(args.manifest, check_files=not args.skip_files), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

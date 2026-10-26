"""Validate and copy an existing segmentation package into API storage."""

import argparse
import os
from pathlib import Path
import shutil

from .storage import contained_file, load_bundle, staged_bundle


def import_sample(source: Path, data_dir: Path, sample_id: str = "frame_211"):
    import re
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", sample_id):
        raise ValueError("Invalid sample_id")
    result = load_bundle(source, check_pixels=True)
    destination = data_dir / "samples" / sample_id
    paths = {"segmentation.json", "report.md", "overlay.jpg"} | {i.mask_path for i in result.instances}
    with staged_bundle(data_dir / "samples", sample_id) as stage:
        for relative in paths:
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            # Copy content without inheriting read-only permissions from extracted packages.
            shutil.copyfile(contained_file(source, relative), target)
    return destination, result.instance_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path(os.getenv("AIE_API_DATA_DIR", str(Path(__file__).resolve().parents[1] / "output" / "api"))))
    parser.add_argument("--sample-id", default="frame_211")
    args = parser.parse_args()
    path, count = import_sample(args.source.resolve(), args.data_dir.resolve(), args.sample_id)
    print(f"Imported {count} instances into {path}. This is an archived sample, not new inference.")


if __name__ == "__main__":
    main()

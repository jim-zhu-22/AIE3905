"""Download a small, balanced MINC-2500 sample via the public dataset viewer.

This is a quick pipeline check. It preserves the published train/validation
splits and does not stand in for the full dataset or an unseen-SKU evaluation.
"""

import argparse
import csv
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image

from . import LABELS, MINC_LABEL_IDS


DATASET = "mcimpoi/minc-2500_split_1"
VIEWER = "https://datasets-server.huggingface.co/rows"
CLASS_COUNTS = {"train": 2125, "validation": 125, "test": 250}


def fetch_rows(split: str, label: str, count: int, output_dir: Path) -> list[dict]:
    start = MINC_LABEL_IDS[label] * CLASS_COUNTS[split]
    rows = []
    for page in range(0, count, 100):
        cache = output_dir / "viewer_rows" / split / label / f"{page:04d}.json"
        if cache.exists() and time.time() - cache.stat().st_mtime < 1800:
            rows.extend(json.loads(cache.read_text(encoding="utf-8")))
            continue
        params = {"dataset": DATASET, "config": "default", "split": split,
                  "offset": start + page, "length": min(100, count - page)}
        for attempt in range(6):
            try:
                response = requests.get(VIEWER, params=params, timeout=60)
                if response.status_code in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(f"retryable status {response.status_code}")
                response.raise_for_status()
                page_rows = response.json()["rows"]
                break
            except (requests.RequestException, ValueError) as error:
                if attempt == 5:
                    raise
                print(f"Viewer error at {split}/{label}/{page}: {error}; waiting", flush=True)
                time.sleep(min(15 * (attempt + 1), 90))
        else:
            raise RuntimeError(f"Viewer unavailable: {split}/{label}/{page}")
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(page_rows), encoding="utf-8")
        rows.extend(page_rows)
        time.sleep(1)
    if len(rows) != count or any(int(row["row"]["label"]) != MINC_LABEL_IDS[label] for row in rows):
        raise ValueError(f"Viewer class ordering changed: {split}/{label}")
    return rows


def download_one(output_dir: Path, split: str, label: str, row: dict) -> dict:
    sample_id = f"{split}-{row['row_idx']:06d}"
    relative = Path("images") / split / label / f"{sample_id}.jpg"
    target = output_dir / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        for attempt in range(5):
            try:
                response = requests.get(row["row"]["image"]["src"], timeout=60)
                response.raise_for_status()
                with Image.open(io.BytesIO(response.content)) as image:
                    temporary = target.with_suffix(".tmp")
                    image.convert("RGB").save(temporary, "JPEG", quality=92)
                    temporary.replace(target)
                break
            except (requests.RequestException, OSError):
                if attempt == 4:
                    raise
                time.sleep(5 * (attempt + 1))
    return {"sample_id": sample_id, "split": split, "label": label,
            "label_id": LABELS.index(label), "image_path": relative.as_posix(),
            "source": "minc-2500_split_1_viewer_probe"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/minc5_probe"))
    parser.add_argument("--train-per-class", type=int, default=40)
    parser.add_argument("--validation-per-class", type=int, default=10)
    parser.add_argument("--full", action="store_true", help="Download all five classes in the published splits")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    counts = CLASS_COUNTS.items() if args.full else (("train", args.train_per_class),
                                                     ("validation", args.validation_per_class))
    rows = []
    expected = 5 * sum(count for _, count in counts)
    for split, count in counts:
        if not 0 < count <= CLASS_COUNTS[split]:
            raise ValueError(f"Invalid {split} sample count")
        for label in LABELS:
            page_rows = fetch_rows(split, label, count, args.output_dir)
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(download_one, args.output_dir, split, label, row) for row in page_rows]
                for future in as_completed(futures):
                    rows.append(future.result())
                    if len(rows) % 25 == 0:
                        print(f"Downloaded {len(rows)}/{expected}", flush=True)
    rows.sort(key=lambda row: (row["split"], row["label_id"], row["sample_id"]))
    if len(rows) != expected or len({row["sample_id"] for row in rows}) != expected:
        raise ValueError(f"Expected {expected} distinct images, found {len(rows)} rows")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("sample_id", "split", "label", "label_id", "image_path", "source"))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} images and manifest to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

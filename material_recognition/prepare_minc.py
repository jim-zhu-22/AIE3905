"""Extract the five POC classes while preserving the published split."""

import argparse
import csv
import io
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image, UnidentifiedImageError

from . import LABELS, MINC_LABEL_IDS


def prepare(raw_dir: Path, output_dir: Path) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_ids = {value: name for name, value in MINC_LABEL_IDS.items()}
    counts: dict[str, int] = {split: 0 for split in ("train", "validation", "test")}
    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=("sample_id", "split", "label", "label_id", "image_path", "source")
        )
        writer.writeheader()
        for split in counts:
            shards = sorted((raw_dir / "data").glob(f"{split}-*.parquet"))
            if not shards:
                raise FileNotFoundError(f"No {split} parquet file in {raw_dir / 'data'}")
            for shard_number, shard in enumerate(shards):
                parquet = pq.ParquetFile(shard)
                row_offset = 0
                for batch in parquet.iter_batches(batch_size=256, columns=["image", "label"]):
                    for local_row, sample in enumerate(batch.to_pylist()):
                        label = selected_ids.get(int(sample["label"]))
                        if label is None:
                            continue
                        payload = sample["image"]["bytes"]
                        if not payload:
                            raise ValueError(f"Missing image bytes in {shard.name} row {row_offset + local_row}")
                        sample_id = f"{split}-{shard_number:02d}-{row_offset + local_row:06d}"
                        relative = Path("images") / split / label / f"{sample_id}.jpg"
                        target = output_dir / relative
                        target.parent.mkdir(parents=True, exist_ok=True)
                        try:
                            with Image.open(io.BytesIO(payload)) as image:
                                image.convert("RGB").save(target, "JPEG", quality=95)
                        except (OSError, UnidentifiedImageError) as exc:
                            raise ValueError(f"Cannot decode {sample_id}") from exc
                        writer.writerow({
                            "sample_id": sample_id,
                            "split": split,
                            "label": label,
                            "label_id": LABELS.index(label),
                            "image_path": relative.as_posix(),
                            "source": "minc-2500_split_1",
                        })
                        counts[split] += 1
                    row_offset += batch.num_rows
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/minc-2500_split_1"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/minc5"))
    args = parser.parse_args()
    counts = prepare(args.raw_dir, args.output_dir)
    print(f"Prepared: {counts}; total={sum(counts.values())}")


if __name__ == "__main__":
    main()

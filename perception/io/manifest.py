import csv
import json
import re
from pathlib import Path


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def read_frames(path, default_prompt):
    path = Path(path).resolve()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Frame manifest is empty")
    seen = set()
    for index, row in enumerate(rows):
        frame_id = row.get("frame_id", "")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,119}", frame_id):
            raise ValueError(f"Unsafe frame_id at row {index + 2}")
        if frame_id.upper() in {"CON", "PRN", "AUX", "NUL", *[f"COM{i}" for i in range(1,10)], *[f"LPT{i}" for i in range(1,10)]}:
            raise ValueError("Reserved frame_id")
        if frame_id.casefold() in seen:
            raise ValueError(f"Duplicate frame_id: {frame_id}")
        seen.add(frame_id.casefold())
        if not row.get("rgb_path"):
            raise ValueError("Manifest requires rgb_path")
        for key in ("rgb_path", "depth_path"):
            row[key] = str((path.parent / row[key]).resolve()) if row.get(key) else None
        row["prompt"] = row.get("prompt") or default_prompt
        if not row["prompt"] or not row["prompt"].strip():
            raise ValueError("Each frame requires a prompt")
    return rows

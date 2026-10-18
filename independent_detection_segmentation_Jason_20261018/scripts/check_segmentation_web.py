"""Send a real image to the local POST /v1/segment endpoint and check artifacts."""

from __future__ import annotations

import argparse
import base64
from io import BytesIO
import json
from pathlib import Path
from urllib.request import Request, urlopen

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:7860")
    parser.add_argument("--image", type=Path, default=PROJECT_ROOT / "Chad_frames_002-005_20261013" / "images" / "2026_08_26_09_16_48_298_color.jpg")
    parser.add_argument("--mode", choices=["full", "tiled"], default="full")
    args = parser.parse_args()
    with urlopen(args.url + "/", timeout=30) as response:
        page = response.read().decode("utf-8")
    if 'id="file"' not in page or "/v1/segment" not in page:
        raise AssertionError("Web page does not expose image upload and model call")
    payload = {
        "image_base64": base64.b64encode(args.image.read_bytes()).decode("ascii"),
        "prompt": "box. bag. bottle. carton. packet.",
        "mode": args.mode,
    }
    request = Request(args.url + "/v1/segment", data=json.dumps(payload).encode("utf-8"),
                      headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=300) as response:
        result = json.load(response)
    with urlopen(args.url + result["overlay_ref"], timeout=30) as response:
        overlay = Image.open(BytesIO(response.read()))
        overlay.load()
    if overlay.size != (1280, 960):
        raise AssertionError(f"Wrong overlay size: {overlay.size}")
    if result["instances"]:
        with urlopen(args.url + result["instances"][0]["mask_ref"], timeout=30) as response:
            mask = Image.open(BytesIO(response.read()))
            mask.load()
        if mask.size != overlay.size:
            raise AssertionError(f"Wrong mask size: {mask.size}")
    target = ROOT / "results" / "segmentation_web_smoke.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"HTTP API OK: instances={result['instance_count']}, total_ms={result['latency_ms']['total']}, output={target}")


if __name__ == "__main__":
    main()

r"""Local on-demand web interface and POST /v1/segment JSON API.

PowerShell from this package: ..\.venv\Scripts\python.exe scripts\segmentation_web.py
Then open http://127.0.0.1:7860/ . Model inference happens after clicking Run.
"""

from __future__ import annotations

import argparse
import base64
import binascii
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
from pathlib import Path
import sys
import threading
from uuid import uuid4

from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
from segmentation_practice.pipeline import GroundedSegmenter, save_prediction

WEB_PAGE = Path(__file__).with_name("segmentation_web.html")
OUTPUT_ROOT = ROOT / "results" / "segmentation_web"
MODEL: GroundedSegmenter | None = None
MODEL_LOCK = threading.Lock()
MAX_POST_BYTES = 15 * 1024 * 1024


class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, value: dict) -> None:
        self._send(status, json.dumps(value, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self) -> None:
        if self.path == "/":
            self._send(200, WEB_PAGE.read_bytes(), "text/html; charset=utf-8")
            return
        if self.path.startswith("/artifacts/"):
            relative = self.path.removeprefix("/artifacts/")
            target = (OUTPUT_ROOT / relative).resolve()
            if not target.is_relative_to(OUTPUT_ROOT.resolve()) or not target.is_file():
                self._json(404, {"error": "artifact not found"})
                return
            content_type = "image/png" if target.suffix.lower() == ".png" else "image/jpeg"
            self._send(200, target.read_bytes(), content_type)
            return
        self._json(404, {"error": "route not found"})

    def do_POST(self) -> None:
        if self.path != "/v1/segment":
            self._json(404, {"error": "route not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_POST_BYTES:
                raise ValueError("Request body must be 1–15 MB")
            payload = json.loads(self.rfile.read(length))
            image_base64 = payload["image_base64"]
            if not isinstance(image_base64, str):
                raise ValueError("image_base64 must be a string")
            if image_base64.startswith("data:"):
                image_base64 = image_base64.split(",", 1)[1]
            image = Image.open(BytesIO(base64.b64decode(image_base64, validate=True)))
            image.verify()
            image = Image.open(BytesIO(base64.b64decode(image_base64, validate=True))).convert("RGB")
            if image.width * image.height > 20_000_000:
                raise ValueError("Image exceeds 20 megapixels")
            prompt = str(payload.get("prompt", "")).strip()
            mode = str(payload.get("mode", "full"))
            if not prompt or len(prompt) > 300:
                raise ValueError("Prompt must contain 1–300 characters")
            if mode not in {"full", "tiled"}:
                raise ValueError("mode must be full or tiled")
        except (KeyError, ValueError, TypeError, binascii.Error, UnidentifiedImageError) as exc:
            self._json(400, {"error": str(exc)})
            return
        try:
            assert MODEL is not None
            with MODEL_LOCK:
                prediction = MODEL.predict(image, prompt, mode)
            run_id = uuid4().hex
            result = save_prediction(prediction, OUTPUT_ROOT / run_id)
            result["run_id"] = run_id
            result["overlay_ref"] = f"/artifacts/{run_id}/overlay.jpg"
            for item in result["instances"]:
                item["mask_ref"] = f"/artifacts/{run_id}/{item['mask_ref']}"
            self._json(200, result)
        except Exception as exc:
            self._json(500, {"error": f"Model inference failed: {exc}"})


def main() -> None:
    global MODEL
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--sam-weight", default=str(PROJECT_ROOT / "sam2.1_t.pt"))
    args = parser.parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    MODEL = GroundedSegmenter(device=args.device, sam_weight=args.sam_weight)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Open http://{args.host}:{args.port}/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()

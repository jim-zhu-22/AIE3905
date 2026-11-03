"""Local, resumable image annotation queue. Predictions never overwrite reviews."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import Future
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import queue
import sqlite3
import threading
import time
from urllib.parse import parse_qs, unquote, urlsplit
import uuid
import zipfile

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
LABELS = {"paper", "plastic", "metal", "wood", "fabric", "unknown"}
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MAX_BYTES = 32 * 1024 * 1024
MAX_PIXELS = 20_000_000
PROMPT = "bottle. cup. box. bag. paper. book. chair. table. cabinet. cloth. container. object."


class WorkspaceLock:
    """OS-held lock: also prevents two different ports from using one queue."""
    def __init__(self, root):
        root.mkdir(parents=True, exist_ok=True)
        self.file = (root / "service.lock").open("a+b")
        if self.file.tell() == 0:
            self.file.write(b"0")
            self.file.flush()
        self.file.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.file.close()
            raise RuntimeError("该输出工作区已有服务运行，请打开原服务或使用不同的 --output")

    def close(self):
        self.file.close()


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, data):
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temp.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def read_image(data):
    with Image.open(io.BytesIO(data)) as im:
        if im.width * im.height > MAX_PIXELS:
            raise ValueError("图片超过 2000 万像素，请先缩小")
        return ImageOps.exif_transpose(im).convert("RGB")


def encode_mask(mask):
    flat = np.asarray(mask, dtype=np.uint8).ravel()
    boundaries = np.flatnonzero(np.diff(flat)) + 1
    counts = np.diff(np.r_[0, boundaries, flat.size]).tolist()
    if flat[0]:
        counts.insert(0, 0)
    return {"size": list(mask.shape), "counts": counts, "order": "C"}


def decode_mask(rle, width, height):
    if not isinstance(rle, dict) or rle.get("size") != [height, width] or rle.get("order") != "C":
        raise ValueError("mask 尺寸或编码错误")
    counts = rle.get("counts")
    if (not isinstance(counts, list) or not counts or len(counts) > width * height + 1
            or any(type(n) is not int or n < 0 for n in counts) or sum(counts) != width * height):
        raise ValueError("mask RLE 数据不完整")
    flat = np.zeros(width * height, dtype=np.uint8)
    pos = 0
    for i, count in enumerate(counts):
        if i % 2:
            flat[pos:pos + count] = 1
        pos += count
    return flat.reshape(height, width)


class Engine:
    """Selected Grounding DINO Tiny + SAM2.1 Tiny, plus existing material baseline."""
    def __init__(self, models, checkpoint, device="auto"):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".model_cache" / "ultralytics"))
        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        from ultralytics import SAM
        from .segmentation import MaterialPredictor, Segmenter

        torch.set_num_threads(8)
        self.torch = torch
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        # Reuse tested box clipping/NMS from the existing demo, not its SAM-v1 loader.
        self.detector = Segmenter.__new__(Segmenter)
        self.detector.torch = torch
        self.detector.device = self.device
        self.detector.detector_processor = AutoProcessor.from_pretrained(
            models / "grounding-dino-tiny", local_files_only=True)
        self.detector.detector = AutoModelForZeroShotObjectDetection.from_pretrained(
            models / "grounding-dino-tiny", local_files_only=True).to(self.device).eval()
        self.sam = SAM(str(models / "sam2.1_t.pt"))
        self.material = MaterialPredictor(checkpoint)
        self.info = {
            "detector": "Grounding DINO Tiny", "segmenter": "SAM2.1 Tiny",
            "material": str(checkpoint), "device": self.device, "prompt": PROMPT,
            "sam_sha256": hashlib.sha256((models / "sam2.1_t.pt").read_bytes()).hexdigest(),
            "material_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "box_threshold": .25, "text_threshold": .20, "schema_version": 1,
        }

    def predict(self, image, boxes=None):
        from .segmentation import mask_box
        if boxes is None:
            proposals, _ = self.detector.detect(image, PROMPT, .25, .20, .5, 40)
        else:
            proposals = [{"bbox": box, "label": "manual_box", "score": None} for box in boxes]
        if not proposals:
            return []
        # Ultralytics accepts numpy images in OpenCV BGR order.
        result = self.sam.predict(source=np.asarray(image)[:, :, ::-1].copy(),
                                  bboxes=[p["bbox"] for p in proposals],
                                  device=self.device, verbose=False)
        if not result or result[0].masks is None:
            raise RuntimeError("SAM2 没有返回 mask")
        raw_masks = result[0].masks.data.detach().cpu().numpy()
        if len(raw_masks) != len(proposals):
            raise RuntimeError("SAM2 mask 数量与检测框不一致")
        objects = []
        accepted = []
        for proposal, raw in zip(proposals, raw_masks):
            mask = raw > .5
            if mask.shape != (image.height, image.width):
                mask = np.asarray(Image.fromarray(mask).resize(image.size, Image.Resampling.NEAREST))
            bbox = mask_box(mask)
            if bbox is None:
                continue
            if any(np.logical_and(mask, old).sum() / max(1, np.logical_or(mask, old).sum()) > .9
                   for old in accepted):
                continue
            accepted.append(mask)
            material = self.material.predict(image.crop(bbox))
            objects.append({
                "id": uuid.uuid4().hex, "label": proposal["label"], "score": proposal["score"],
                "bbox": bbox, "mask": encode_mask(mask), "material": material["label"],
                "model_material": material["label"], "model_confidence": material["score"],
                "model_scores": material["scores"], "source": "auto" if boxes is None else "manual_box_sam2",
            })
        return objects


class Store:
    def __init__(self, output):
        self.root = Path(output)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "items").mkdir(exist_ok=True)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(self.root / "queue.sqlite3", check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("""CREATE TABLE IF NOT EXISTS jobs(
            id TEXT PRIMARY KEY, name TEXT NOT NULL, source TEXT NOT NULL,
            width INTEGER, height INTEGER, state TEXT, error TEXT, created TEXT, finished TEXT)""")
        # Recover interrupted workers. Committed predictions are never inferred again.
        for row in self.db.execute("SELECT id FROM jobs WHERE state='processing'").fetchall():
            state = "done" if (self.item(row["id"]) / "prediction.json").exists() else "pending"
            self.db.execute("UPDATE jobs SET state=?,error='' WHERE id=?", (state, row["id"]))
        self.db.commit()

    def item(self, key):
        if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
            raise ValueError("无效图片 ID")
        return self.root / "items" / key

    def add(self, data, name, source):
        if len(data) > MAX_BYTES:
            raise ValueError("单张图片不能超过 32 MB")
        key = hashlib.sha256(data).hexdigest()
        with self.lock:
            if self.db.execute("SELECT 1 FROM jobs WHERE id=?", (key,)).fetchone():
                return key, False
            image = read_image(data)
            folder = self.item(key)
            folder.mkdir(exist_ok=True)
            image.save(folder / "image.png")
            self.db.execute("INSERT INTO jobs VALUES(?,?,?,?,?,'pending','',?,NULL)",
                            (key, name, source, image.width, image.height, now()))
            self.db.commit()
        return key, True

    def row(self, key):
        self.item(key)
        with self.lock:
            row = self.db.execute("SELECT * FROM jobs WHERE id=?", (key,)).fetchone()
            if not row:
                raise KeyError("图片不存在")
            return dict(row)

    def state(self, key, state, error=""):
        with self.lock:
            self.db.execute("UPDATE jobs SET state=?,error=?,finished=? WHERE id=?",
                            (state, error, now() if state in ("done", "failed") else None, key))
            self.db.commit()

    def rows(self):
        with self.lock:
            return [dict(row) for row in self.db.execute("SELECT * FROM jobs ORDER BY created,id")]

    def document(self, key):
        row = self.row(key)
        with self.lock:
            folder = self.item(key)
            prediction_path = folder / "prediction.json"
            prediction = json.loads(prediction_path.read_text("utf-8")) if prediction_path.exists() else None
            review_path = folder / "review.json"
            review = json.loads(review_path.read_text("utf-8")) if review_path.exists() else None
            return {**row, "prediction": prediction, "review": review,
                    "revision": review["revision"] if review else 0}

    def save_review(self, key, data):
        row = self.row(key)
        if row["state"] != "done":
            raise ValueError("请等自动批注完成后再保存人工审核")
        objects = data.get("objects")
        if not isinstance(objects, list) or len(objects) > 200:
            raise ValueError("无效对象列表")
        reviewer = str(data.get("reviewer", "")).strip()[:100]
        confirmed = data.get("confirmed") is True
        if confirmed and not reviewer:
            raise ValueError("确认审核前请填写审核人")
        sanitized = []
        ids = set()
        for obj in objects:
            oid = obj.get("id")
            if not isinstance(oid, str) or not oid.isalnum() or len(oid) > 64 or oid in ids:
                raise ValueError("重复或无效对象 ID")
            ids.add(oid)
            mask = decode_mask(obj.get("mask"), row["width"], row["height"])
            if not mask.any():
                raise ValueError("对象 mask 为空；请删除该对象或补画 mask")
            material = obj.get("material")
            if material not in LABELS:
                raise ValueError("请选择有效材质（不确定可选 unknown）")
            from .segmentation import mask_box
            sanitized.append({**{k: obj[k] for k in (
                "id", "label", "score", "material", "model_material", "model_confidence",
                "model_scores", "source") if k in obj}, "bbox": mask_box(mask), "mask": encode_mask(mask)})
        with self.lock:
            existing = self.document(key)
            if type(data.get("revision")) is not int or data["revision"] != existing["revision"]:
                raise FileExistsError("另一页面已保存更新，请重新打开此图后编辑，避免覆盖")
            review = {"objects": sanitized, "reviewer": reviewer, "confirmed": confirmed,
                      "revision": existing["revision"] + 1, "updated": now()}
            atomic_json(self.item(key) / "review.json", review)
            return review

    def export(self, key=None):
        output = io.BytesIO()
        manifest = []
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
            for row in [self.row(key)] if key else self.rows():
                doc = self.document(row["id"])
                review = doc["review"]
                if not review or not review["confirmed"]:
                    continue
                entry = {"image_id": row["id"], "filename": row["name"],
                         "size": [row["width"], row["height"]], "masks": []}
                for index, obj in enumerate(review["objects"], 1):
                    mask = decode_mask(obj["mask"], row["width"], row["height"])
                    png = io.BytesIO()
                    Image.fromarray(mask * 255).save(png, "PNG")
                    name = f'{row["id"]}/mask_{index:03}.png'
                    archive.writestr(name, png.getvalue())
                    entry["masks"].append(name)
                manifest.append(entry)
            if not manifest:
                raise ValueError("暂无已确认审核的图片；先检查 mask 和材质并确认")
            # Identity mapping only; material predictions remain in the local audit record.
            archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        return output.getvalue()


class Service:
    def __init__(self, store, inbox, sources, engine_factory):
        self.store = store
        self.inbox = Path(inbox)
        self.inbox.mkdir(parents=True, exist_ok=True)
        self.sources = [self.inbox, *map(Path, sources)]
        self.engine_factory = engine_factory
        self.engine = None
        self.model_error = ""
        self.phase = "等待任务"
        self.current = ""
        self.seen = {}
        self.scan_errors = {}
        self.scan_lock = threading.Lock()
        self.stopping = threading.Event()
        self.manual = queue.Queue()
        control = self.store.root / "control.json"
        self.paused = json.loads(control.read_text("utf-8")).get("paused", False) if control.exists() else False

    def scan(self):
        with self.scan_lock:
            for root in self.sources:
                if not root.is_dir():
                    self.scan_errors[str(root)] = "文件夹不存在"
                    continue
                for path in sorted(root.rglob("*")):
                    if (not path.is_file() or path.suffix.lower() not in EXTENSIONS
                            or path.stem.lower().endswith(("_depth", "_mask"))
                            or path.is_symlink()):
                        continue
                    try:
                        stat = path.stat()
                        signature = (stat.st_size, stat.st_mtime_ns)
                        if self.seen.get(str(path)) == signature:
                            continue
                        if stat.st_size > MAX_BYTES:
                            raise ValueError("单张图片不能超过 32 MB")
                        data = path.read_bytes()
                        if path.stat().st_mtime_ns != stat.st_mtime_ns:
                            continue  # Still being copied; try next scan.
                        self.store.add(data, path.name, str(path))
                        self.seen[str(path)] = signature
                        self.scan_errors.pop(str(path), None)
                    except Exception as exc:
                        self.scan_errors[str(path)] = str(exc)

    def scanner(self):
        while not self.stopping.is_set():
            try:
                self.scan()
            except Exception as exc:
                self.scan_errors["scanner"] = str(exc)
            self.stopping.wait(5)

    def worker(self):
        while not self.stopping.is_set():
            try:
                manual = self.manual.get_nowait()
            except queue.Empty:
                manual = None
            pending = next((r for r in self.store.rows() if r["state"] == "pending"), None)
            if manual is None and (not pending or self.paused or self.model_error):
                if not self.model_error:
                    self.phase = "已暂停" if self.paused else "等待新图片"
                self.stopping.wait(.3)
                continue
            key, boxes, future = manual if manual else (pending["id"], None, None)
            try:
                self.current = self.store.row(key)["name"]
                if self.engine is None:
                    self.phase = "加载模型（首次启动需要等待）"
                    self.engine = self.engine_factory()
                    self.model_error = ""
                self.phase = "人工框选生成 mask" if manual else "自动检测 → 分割 → 材质预测"
                if not manual:
                    self.store.state(key, "processing")
                with Image.open(self.store.item(key) / "image.png") as im:
                    objects = self.engine.predict(im.convert("RGB"), boxes)
                if manual:
                    future.set_result(objects)
                else:
                    atomic_json(self.store.item(key) / "prediction.json", {
                        "objects": objects, "model": self.engine.info, "created": now()})
                    self.store.state(key, "done")
            except Exception as exc:
                if self.engine is None:
                    self.model_error = str(exc)
                    self.phase = "模型加载失败"
                if manual:
                    future.set_exception(exc)
                else:
                    self.store.state(key, "failed", str(exc))
                print(f"ERROR {key}: {exc}", flush=True)
            finally:
                self.current = ""

    def status(self):
        rows = self.store.rows()
        for row in rows:
            review = self.store.item(row["id"]) / "review.json"
            row["reviewed"] = review.exists() and json.loads(review.read_text("utf-8"))["confirmed"]
        return {"items": rows, "counts": dict(Counter(row["state"] for row in rows)),
                "total": len(rows), "phase": self.phase, "current": self.current,
                "model_error": self.model_error, "scan_errors": dict(self.scan_errors),
                "inbox": str(self.inbox), "sources": list(map(str, self.sources)),
                "output": str(self.store.root), "paused": self.paused,
                "model": self.engine.info if self.engine else None}


def make_handler(service):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send(self, code, data, kind="application/json; charset=utf-8", attachment=None):
            if isinstance(data, (dict, list)):
                data = json.dumps(data, ensure_ascii=False).encode("utf-8")
            elif isinstance(data, str):
                data = data.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            if attachment:
                self.send_header("Content-Disposition", f'attachment; filename="{attachment}"')
            self.end_headers()
            self.wfile.write(data)

        def body(self):
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_BYTES:
                raise ValueError("请求为空或超过 32 MB")
            return self.rfile.read(length)

        def dispatch(self, write=False):
            try:
                # Local-only service: reject cross-origin writes and DNS rebinding.
                host = self.headers.get("Host", "")
                if host not in {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}:
                    return self.send(403, {"error": "仅允许 localhost"})
                origin = self.headers.get("Origin")
                if write and origin and origin != "http://" + host:
                    return self.send(403, {"error": "不允许跨站请求"})
                if write and self.headers.get("X-Annotation-Client") != "local":
                    return self.send(403, {"error": "缺少本地客户端标识"})
                url = urlsplit(self.path)
                path = url.path
                query = parse_qs(url.query)
                store = service.store
                if not write:
                    if path in ("/", "/index.html"):
                        return self.send(200, Path(__file__).with_name("annotation_app.html").read_bytes(), "text/html; charset=utf-8")
                    if path == "/api/status":
                        return self.send(200, service.status())
                    if path == "/api/export":
                        return self.send(200, store.export(query.get("id", [None])[0]),
                                         "application/zip", "reviewed_masks.zip")
                    pieces = path.strip("/").split("/")
                    if len(pieces) == 3 and pieces[0] == "api":
                        key = pieces[2]
                        if pieces[1] == "item":
                            return self.send(200, store.document(key))
                        if pieces[1] == "image":
                            store.row(key)
                            return self.send(200, (store.item(key) / "image.png").read_bytes(), "image/png")
                else:
                    if path == "/api/upload":
                        name = query.get("name", ["upload.png"])[0]
                        # Only basenames; encoded separators and Windows drive paths are rejected.
                        if not name or any(c in name for c in "/\\:\x00") or name in (".", ".."):
                            raise ValueError("文件名不能包含路径")
                        if Path(name).suffix.lower() not in EXTENSIONS:
                            raise ValueError("仅支持 JPG/PNG/WEBP/BMP 图片")
                        data = self.body()
                        read_image(data)
                        key = hashlib.sha256(data).hexdigest()
                        target = service.inbox / (key + Path(name).suffix.lower())
                        if not target.exists():
                            temp = target.with_suffix("." + uuid.uuid4().hex + ".upload")
                            with temp.open("xb") as out:
                                out.write(data)
                            os.replace(temp, target)
                        key, added = store.add(data, name, str(target))
                        return self.send(200, {"id": key, "added": added})
                    data = json.loads(self.body())
                    if path == "/api/pause":
                        service.paused = data.get("paused") is True
                        atomic_json(store.root / "control.json", {"paused": service.paused})
                        return self.send(200, {"paused": service.paused})
                    if path == "/api/retry":
                        row = store.row(data["id"])
                        if row["state"] != "failed":
                            raise ValueError("只能重试失败图片，已完成图片不会再次自动批注")
                        store.state(row["id"], "pending")
                        service.model_error = ""
                        return self.send(200, {"ok": True})
                    if path.startswith("/api/review/"):
                        return self.send(200, store.save_review(path.rsplit("/", 1)[1], data))
                    if path.startswith("/api/box/"):
                        key = path.rsplit("/", 1)[1]
                        row = store.row(key)
                        if row["state"] != "done":
                            raise ValueError("请先等待本图自动批注完成")
                        from .segmentation import clean_box
                        box = clean_box(data["box"], (row["width"], row["height"]))
                        future = Future()
                        service.manual.put((key, [box], future))
                        return self.send(200, {"objects": future.result(timeout=180)})
                self.send(404, {"error": "接口不存在"})
            except FileExistsError as exc:
                self.send(409, {"error": str(exc)})
            except (ValueError, KeyError, TypeError) as exc:
                self.send(400, {"error": str(exc)})
            except Exception as exc:
                self.send(500, {"error": str(exc)})

        def do_GET(self):
            self.dispatch()

        def do_POST(self):
            self.dispatch(True)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inbox", type=Path, default=ROOT / "04-数据" / "自动标注输入")
    parser.add_argument("--source", action="append", default=[], help="Additional read-only source folders")
    parser.add_argument("--output", type=Path, default=ROOT / "06-实验结果" / "自动标注工作区")
    parser.add_argument("--models", type=Path, default=ROOT / "05-模型权重")
    parser.add_argument("--material", type=Path, default=ROOT / "06-实验结果" / "训练记录" / "week1_resnet18" / "best.pt")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--port", type=int, default=8769)
    parser.add_argument("--paused", action="store_true", help="Start with automatic queue paused")
    args = parser.parse_args()
    workspace_lock = WorkspaceLock(args.output)
    store = Store(args.output)
    service = Service(store, args.inbox, args.source,
                      lambda: Engine(args.models, args.material, args.device))
    if args.paused:
        service.paused = True
    # Bind before starting the worker to avoid duplicate inference if another demo is running.
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(service))
    for target in (service.scanner, service.worker):
        threading.Thread(target=target, daemon=True).start()
    print(f"Annotation demo: http://127.0.0.1:{args.port}/\nInbox: {args.inbox}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        service.stopping.set()
        server.server_close()
        # daemon worker ends with the process; keep the lock until then.
        _ = workspace_lock


if __name__ == "__main__":
    main()

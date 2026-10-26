"""Minimal standard-library client for submitting images from another program."""

import hashlib
import json
from pathlib import Path
import re
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4


class ApiError(RuntimeError):
    def __init__(self, status, body):
        error = body.get("error", {})
        super().__init__(error.get("message", f"HTTP {status}"))
        self.status, self.code = status, error.get("code")


class PerceptionClient:
    def __init__(self, base_url="http://127.0.0.1:8000", timeout=30):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _json(self, path, data=None, headers=None):
        request = Request(self.base_url + path, data=data, headers=headers or {})
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return json.load(response)
        except HTTPError as exc:
            try:
                body = json.load(exc)
            except (ValueError, OSError):
                body = {}
            raise ApiError(exc.code, body) from exc

    def ready(self):
        return self._json("/v1/app/ready")

    def submit_image(self, image_path, *, prompt, modules=None, key=None, frame_id=None, depth_path=None, meta=None, stable=False):
        image_path = Path(image_path)
        if image_path.stat().st_size > 15 * 1024 * 1024:
            raise ValueError("RGB image exceeds 15 MiB")
        image = image_path.read_bytes()
        depth = None
        if depth_path:
            depth_path = Path(depth_path)
            if depth_path.stat().st_size > 15 * 1024 * 1024:
                raise ValueError("Depth image exceeds 15 MiB")
            depth = depth_path.read_bytes()
        frame_id = frame_id or re.sub(r"[^A-Za-z0-9_-]", "_", image_path.stem)[:80] or "uploaded_frame"
        boundary = "aie_" + uuid4().hex
        parts = []
        for name, value in {"frame_id":frame_id, "prompt":prompt, "modules":json.dumps(modules or []),
                            "meta":json.dumps(meta or {})}.items():
            parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
        for field, raw, filename in (("rgb",image,"image.png"),("depth",depth,"depth.png")):
            if raw is None:
                continue
            parts.append((f'--{boundary}\r\nContent-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'
                          'Content-Type: application/octet-stream\r\n\r\n').encode() + raw + b"\r\n")
        parts.append(f"--{boundary}--\r\n".encode())
        body = b"".join(parts)
        if len(body) > 22 * 1024 * 1024:
            raise ValueError("Combined request exceeds 22 MiB")
        key = key or (stable_key(image_path,prompt,modules or [],image=image,depth=depth,meta=meta,frame_id=frame_id)
                      if stable else uuid4().hex)
        headers = {"Content-Type":f"multipart/form-data; boundary={boundary}","Idempotency-Key":key}
        # Only an ambiguous transport failure is retried, with the identical submission key.
        for attempt in range(2):
            try:
                return self._json("/v1/perceive",body,headers)
            except (URLError, TimeoutError, ConnectionError):
                if attempt:
                    raise

    def wait(self, job, timeout=600, interval=.5):
        job_id = job if isinstance(job,str) else job["id"]
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            current = self._json("/v1/jobs/" + job_id)
            if current["status"] not in {"queued","running"}:
                return current
            time.sleep(interval)
        raise TimeoutError(f"Task {job_id} is still running; query its id instead of resubmitting")

    def result(self, job):
        job_id = job if isinstance(job,str) else job["id"]
        return self._json("/v1/jobs/" + job_id + "/result")


def stable_key(path, prompt, modules, *, image=None, depth=None, meta=None, frame_id=None):
    """A restarted folder sender can safely find the original submission again."""
    path = Path(path)
    digest = hashlib.sha256(str(path.resolve()).encode())
    digest.update(image if image is not None else path.read_bytes())
    digest.update(depth or b"")
    digest.update(json.dumps([prompt,modules,meta,frame_id],ensure_ascii=False,sort_keys=True).encode())
    return "file_" + digest.hexdigest()

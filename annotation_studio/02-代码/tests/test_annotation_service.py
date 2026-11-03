import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from http.server import ThreadingHTTPServer
import zipfile

import numpy as np
from PIL import Image

from rgbd_perception.annotation_service import (
    Store, Service, WorkspaceLock, encode_mask, decode_mask, make_handler, atomic_json,
)


def png(color="red"):
    data = io.BytesIO()
    Image.new("RGB", (12, 8), color).save(data, "PNG")
    return data.getvalue()


class FakeEngine:
    info = {"test": True}
    calls = 0

    def predict(self, image, boxes=None):
        self.calls += 1
        mask = np.zeros((image.height, image.width), dtype=np.uint8)
        mask[1:5, 2:7] = 1
        return [{"id": "test1", "material": "paper", "model_material": "paper",
                 "model_confidence": .8, "bbox": [2, 1, 7, 5], "mask": encode_mask(mask)}]


class AnnotationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.store = Store(self.root / "out")

    def tearDown(self):
        self.store.db.close()
        self.tmp.cleanup()

    def complete(self):
        key, _ = self.store.add(png(), "你好.png", "test")
        objects = FakeEngine().predict(Image.new("RGB", (12, 8)))
        atomic_json(self.store.item(key) / "prediction.json", {"objects": objects})
        self.store.state(key, "done")
        return key, objects

    def test_rle_roundtrip_and_validation(self):
        for mask in [np.zeros((8, 12), np.uint8), np.ones((8, 12), np.uint8),
                     np.random.default_rng(3).integers(0, 2, (8, 12), dtype=np.uint8)]:
            np.testing.assert_array_equal(mask, decode_mask(encode_mask(mask), 12, 8))
        for counts in [[-1, 97], [True, 95], [1], [1.5, 94.5]]:
            with self.assertRaises(ValueError):
                decode_mask({"size": [8, 12], "order": "C", "counts": counts}, 12, 8)

    def test_workspace_lock(self):
        first = WorkspaceLock(self.root / "lock")
        try:
            with self.assertRaises(RuntimeError):
                WorkspaceLock(self.root / "lock")
        finally:
            first.close()
        second = WorkspaceLock(self.root / "lock")
        second.close()

    def test_dedup_and_restart_keeps_completed_review(self):
        key, objects = self.complete()
        review = self.store.save_review(key, {"objects": objects, "revision": 0, "reviewer": "测试", "confirmed": True})
        self.assertEqual(self.store.add(png(), "renamed.png", "other"), (key, False))
        self.store.db.close()
        self.store = Store(self.root / "out")
        self.assertEqual(self.store.row(key)["state"], "done")
        self.assertEqual(self.store.document(key)["review"], review)
        self.assertEqual(len(self.store.rows()), 1)

    def test_conflict_and_empty_mask_rejected(self):
        key, objects = self.complete()
        self.store.save_review(key, {"objects": objects, "revision": 0})
        with self.assertRaises(FileExistsError):
            self.store.save_review(key, {"objects": objects, "revision": 0})
        objects[0]["mask"] = encode_mask(np.zeros((8, 12), np.uint8))
        with self.assertRaises(ValueError):
            self.store.save_review(key, {"objects": objects, "revision": 1})

    def test_export_only_confirmed_full_resolution_masks(self):
        key, objects = self.complete()
        with self.assertRaises(ValueError):
            self.store.export()
        self.store.save_review(key, {"objects": objects, "revision": 0, "reviewer": "测试", "confirmed": True})
        with zipfile.ZipFile(io.BytesIO(self.store.export())) as z:
            manifest = json.loads(z.read("manifest.json"))
            self.assertEqual(manifest[0]["filename"], "你好.png")
            with Image.open(io.BytesIO(z.read(manifest[0]["masks"][0]))) as im:
                self.assertEqual(im.size, (12, 8))
                self.assertEqual(set(np.asarray(im).ravel()), {0, 255})
            self.assertNotIn("material", z.read("manifest.json").decode())
        self.store.save_review(key, {"objects": objects, "revision": 1, "confirmed": False})
        with self.assertRaises(ValueError):
            self.store.export()

    def test_interruption_recovery(self):
        key, _ = self.store.add(png(), "x.png", "test")
        self.store.state(key, "processing")
        self.store.db.close()
        self.store = Store(self.root / "out")
        self.assertEqual(self.store.row(key)["state"], "pending")
        atomic_json(self.store.item(key) / "prediction.json", {"objects": []})
        self.store.state(key, "processing")
        self.store.db.close()
        self.store = Store(self.root / "out")
        self.assertEqual(self.store.row(key)["state"], "done")

    def test_queue_scans_once_and_skips_depth(self):
        engine = FakeEngine()
        inbox = self.root / "inbox"
        inbox.mkdir()
        (inbox / "a_color.png").write_bytes(png())
        (inbox / "a_depth.png").write_bytes(png("blue"))
        service = Service(self.store, inbox, [], lambda: engine)
        service.scan()
        worker = threading.Thread(target=service.worker)
        worker.start()
        try:
            deadline = time.monotonic() + 5
            while self.store.rows()[0]["state"] != "done" and time.monotonic() < deadline:
                time.sleep(.02)
            service.scan()
            self.assertEqual(len(self.store.rows()), 1)
            self.assertEqual(self.store.rows()[0]["state"], "done")
            self.assertEqual(engine.calls, 1)
        finally:
            service.stopping.set()
            worker.join(5)

    def test_http_upload_duplicate_traversal_origin_and_utf8(self):
        service = Service(self.store, self.root / "inbox", [], FakeEngine)
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(service))
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            with urlopen(base) as response:
                self.assertIn("charset=utf-8", response.headers["Content-Type"])
                self.assertIn("本地自动标注", response.read().decode())
            def upload(name, headers=None):
                return urlopen(Request(base + "/api/upload?name=" + name, png(),
                                       {"X-Annotation-Client": "local", **(headers or {})}))
            with upload("sample.png") as response:
                self.assertTrue(json.load(response)["added"])
            with upload("same.png") as response:
                self.assertFalse(json.load(response)["added"])
            for name in ("..%2Fescape.png", "..%5Cescape.png", "C%3Aescape.png"):
                with self.assertRaises(HTTPError) as caught:
                    upload(name)
                self.assertEqual(caught.exception.code, 400)
            with self.assertRaises(HTTPError) as caught:
                upload("a.png", {"Origin": "https://evil.example"})
            self.assertEqual(caught.exception.code, 403)
            with self.assertRaises(HTTPError):
                urlopen(base + "/api/image/not-a-valid-id")
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()

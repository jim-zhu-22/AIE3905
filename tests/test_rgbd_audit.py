import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from scripts.inspect_rgbd import inspect


class RGBDAuditTest(unittest.TestCase):
    def test_single_pair_and_unpaired_rgb_are_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "raw"
            source.mkdir()
            frame = "2026_01_01_00_00_00_000"
            Image.new("RGB", (16, 12), "white").save(source / f"{frame}_color.jpg")
            Image.fromarray(np.full((12, 16), 1000, dtype=np.uint16)).save(source / f"{frame}_depth.png")
            missing = "2026_01_01_00_00_01_000"
            Image.new("RGB", (16, 12)).save(source / f"{missing}_color.jpg")
            output = root / "review"
            template = Path(__file__).resolve().parents[1] / "scripts/rgbd_viewer.html"
            with contextlib.redirect_stdout(io.StringIO()):
                inspect(source, output, template)
            result = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(result["readable_pairs"], 1)
            self.assertEqual(result["missing_depth"], [missing])
            self.assertEqual(result["raw_min"], 1000)
            self.assertIsNone(result["median_gap_seconds"])
            self.assertTrue((output / "rgb_depth_examples.jpg").is_file())
            page = (output / "index.html").read_text(encoding="utf-8")
            self.assertNotIn("__FRAMES__", page)
            self.assertIn(frame, page)

    def test_missing_source_fails_clearly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError):
                inspect(root/"absent", root/"out", root/"template")

    def test_output_cannot_modify_raw_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "outside"):
                inspect(root, root/"out", root/"template")

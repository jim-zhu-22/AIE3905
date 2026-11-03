import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
from perception.config import PipelineConfig, load_config
from perception.geometry.measure import measure
from perception.io.manifest import read_frames
from perception.pipeline.contracts import Instance, Prediction
from perception.pipeline.service import PerceptionPipeline, material_crop
from perception.runtime import resolve_device


class FakeSegmenter:
    def __init__(self, empty=False):
        self.empty = empty

    def predict(self, image, prompt, mode):
        mask = np.zeros((image.height, image.width), dtype=bool)
        mask[1:3, 1:3] = True
        instances = [] if self.empty else [Instance([0, 0, 4, 4], .9, "box", mask)]
        return Prediction(image, instances, mode, prompt, {}, len(instances), {"test_double": "true"})


class FakeMaterial:
    device = "cpu"

    def predict(self, image):
        return {"label": "paper", "score": .8, "scores": {"paper": .8}}


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.weight = self.root / "test.pt"
        self.weight.write_bytes(b"x" * 2048)
        self.rgb = self.root / "rgb.png"
        Image.new("RGB", (4, 4), (255, 0, 0)).save(self.rgb)
        self.config = PipelineConfig(str(self.weight), str(self.weight))

    def pipeline(self, empty=False):
        return PerceptionPipeline(self.config, segmenter=FakeSegmenter(empty), material=FakeMaterial())

    def test_serialized_result_and_mask_references(self):
        target = self.root / "out"
        result = self.pipeline().process(self.rgb, "box .", target)
        saved = json.loads((target / "perception.json").read_text(encoding="utf-8"))
        self.assertEqual(saved, result)
        item = saved["instances"][0]
        self.assertEqual(item["geometry"]["mask_area_px"], 4)
        self.assertEqual(item["material"]["label"], "paper")
        self.assertTrue((target / item["mask_ref"]).is_file())
        self.assertIsNone(item["physical_attributes"]["mass_kg"])
        with self.assertRaisesRegex(ValueError, "empty"):
            self.pipeline().process(self.rgb, "box", target)

    def test_no_detections_is_valid_empty_result(self):
        result = self.pipeline(True).process(self.rgb, "box", self.root / "empty")
        self.assertEqual(result["instances"], [])

    def test_depth_requires_registration_and_matching_dimensions(self):
        self.config.depth_processing = "statistics"
        depth = self.root / "depth.png"
        Image.fromarray(np.full((4, 4), 1000, dtype=np.uint16)).save(depth)
        with self.assertRaisesRegex(ValueError, "registration"):
            self.pipeline().process(self.rgb, "box", self.root / "unregistered", depth)
        self.config.aligned_depth = True
        self.config.depth_scale = .001
        result = self.pipeline().process(self.rgb, "box", self.root / "depth", depth)
        self.assertEqual(result["instances"][0]["geometry"]["depth_median_m"], 1.0)
        Image.fromarray(np.zeros((2, 2), dtype=np.uint16)).save(depth)
        with self.assertRaisesRegex(ValueError, "dimensions"):
            self.pipeline().process(self.rgb, "box", self.root / "bad_depth", depth)

    def test_invalid_and_missing_depth(self):
        mask = np.ones((2, 2), dtype=bool)
        result = measure(mask, np.array([[0, float("nan")], [1000, float("inf")]]), .001)
        self.assertEqual(result["valid_depth_fraction"], .25)
        self.assertEqual(result["depth_median_m"], 1.0)
        self.assertEqual(measure(mask, np.zeros((2, 2)), .001)["depth_status"], "no_valid_depth")
        with self.assertRaises(ValueError):
            measure(mask, depth_scale=float("nan"))

    def test_bbox_vs_masked_crops_and_empty_mask(self):
        image = Image.open(self.rgb)
        instance = FakeSegmenter().predict(image, "box", "full").instances[0]
        self.assertEqual(material_crop(image, instance, "bbox").getpixel((0, 0)), (255, 0, 0))
        self.assertEqual(material_crop(image, instance, "masked").getpixel((0, 0)), (127, 127, 127))
        instance.mask[:] = False
        self.assertIsNone(material_crop(image, instance, "masked"))
        image.close()

    def test_cpu_is_respected_when_cuda_available(self):
        with patch("torch.cuda.is_available", return_value=True):
            self.assertEqual(resolve_device("cpu"), "cpu")
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaises(RuntimeError):
                resolve_device("cuda")

    def test_config_relative_paths_and_reject_unknown_keys(self):
        path = self.root / "config.json"
        path.write_text(json.dumps({"sam_weight": "test.pt", "material_weight": "test.pt"}))
        self.assertEqual(load_config(path).material_weight, str(self.weight))
        path.write_text(json.dumps({"sam_weight": "test.pt", "material_weight": "test.pt", "typo": 1}))
        with self.assertRaises(TypeError):
            load_config(path)

    def test_manifest_rejects_duplicate_and_traversal_ids(self):
        path = self.root / "frames.csv"
        for ids in (["same", "SAME"], ["../escape"], ["CON"]):
            with path.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["frame_id", "rgb_path"])
                for id in ids:
                    writer.writerow([id, "rgb.png"])
            with self.assertRaises(ValueError):
                read_frames(path, "box")

    def test_batch_keeps_successes_and_returns_failure_for_missing_frame(self):
        from perception.pipeline.run import main
        manifest = self.root / "batch.csv"
        manifest.write_text("frame_id,rgb_path\nmissing,missing.png\nvalid,rgb.png\n")
        output = self.root / "batch"
        argv = ["run", "--manifest", str(manifest), "--sam-weight", str(self.weight),
                "--material-weight", str(self.weight), "--output", str(output)]
        with patch("sys.argv", argv), patch("perception.pipeline.service.PerceptionPipeline", return_value=self.pipeline()):
            with self.assertRaises(SystemExit) as error:
                main()
        self.assertEqual(error.exception.code, 1)
        summary = json.loads((output / "batch_summary.json").read_text())
        self.assertEqual([frame["status"] for frame in summary["frames"]], ["error", "ok"])
        self.assertTrue((output / "valid/perception.json").is_file())


if __name__ == "__main__":
    unittest.main()

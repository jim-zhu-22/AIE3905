import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from perception.config import PipelineConfig
from perception.io.pipeline_contract import SavedSegmenter
from perception.pipeline.contracts import Instance, Prediction
from perception.pipeline.service import PerceptionPipeline


class Material:
    device = "cpu"

    def predict(self, image):
        return {"label": "paper", "score": .8, "scores": {"paper": .8, "plastic": .2}}


class ContractTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.rgb = self.root / "rgb.png"
        Image.new("RGB", (6, 4), "red").save(self.rgb)
        self.weight = self.root / "test.pt"
        self.weight.write_bytes(b"x" * 2048)
        self.config = PipelineConfig("sam_not_needed", str(self.weight))
        self.source = self.root / "source"
        (self.source / "masks").mkdir(parents=True)
        self.data = {"schema_version": "pipeline-1.0", "stage": "segmentation", "status": "ok",
                     "run_id": "upstream_run", "frame_id": "upstream_frame", "prompt": "box.",
                     "model_version": "upstream_v1", "image_size": [6, 4], "instance_count": 2, "instances": [],
                     "depth": {"depth_path": "../missing.png", "depth_size": [6, 4], "depth_dtype": "uint16",
                               "depth_unit": "unconfirmed", "depth_scale_to_m": None,
                               "alignment": "unconfirmed", "depth_usable": False}}
        for ident, x in [("i014", 0), ("i002", 3)]:
            mask = np.zeros((4, 6), dtype=np.uint8)
            mask[1:3, x:x+2] = 255
            Image.fromarray(mask).save(self.source / "masks" / f"{ident}.png")
            self.data["instances"].append({"instance_id": ident, "object_label": "box", "bbox_xyxy": [x, 1, x+2, 3],
                                           "det_score": .8, "seg_score": None, "mask_path": f"masks/{ident}.png"})
        self.write()

    def write(self):
        self.path = self.source / "segmentation.json"
        self.path.write_text(json.dumps(self.data), encoding="utf-8")

    def run_pipeline(self, material=None, **kwargs):
        output = self.root / "output"
        pipeline = PerceptionPipeline(self.config, segmenter=SavedSegmenter(self.path), material=material or Material())
        pipeline.process(self.rgb, "ignored upstream prompt wins", output, **kwargs)
        return output, json.loads((output / "result.json").read_text(encoding="utf-8"))

    def test_import_preserves_identifiers_and_joinable_stages(self):
        output, result = self.run_pipeline()
        self.assertEqual([v["instance_id"] for v in result["instances"]], ["i014", "i002"])
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["prompt"], "box.")
        for name in ("preprocess", "segmentation", "material_input", "material_result", "result"):
            data = json.loads((output / f"{name}.json").read_text(encoding="utf-8"))
            self.assertEqual((data["run_id"], data["frame_id"]), ("upstream_run", "upstream_frame"))
        material = result["instances"][0]["material"]
        self.assertEqual((material["label"], material["top1_label"]), ("unknown", "paper"))
        self.assertFalse(result["depth"]["depth_usable"])
        self.assertIn("depth_file_missing", result["warnings"])
        for name in ("segmentation_raw.json", "perception.json", "report.md", "rgb/frame.png", "rois/i014_bbox_0.png"):
            self.assertTrue((output / name).is_file())
        np.testing.assert_array_equal(np.array(Image.open(output / "masks/i014.png")), np.array(Image.open(self.source / "masks/i014.png")))

    def test_unconfirmed_depth_is_reference_and_does_not_block_rgb(self):
        depth = self.root / "depth.png"
        Image.fromarray(np.full((4, 6), 1000, dtype=np.uint16)).save(depth)
        output, result = self.run_pipeline(depth_path=depth)
        self.assertEqual(result["depth"]["depth_size"], [6, 4])
        self.assertEqual(result["depth"]["processing_status"], "reference_only")
        self.assertTrue((output / result["depth"]["depth_path"]).is_file())
        legacy = json.loads((output / "perception.json").read_text(encoding="utf-8"))
        self.assertIsNone(legacy["instances"][0]["geometry"]["depth_median_m"])

    def test_no_match(self):
        self.data.update(status="no_match", instance_count=0, instances=[])
        self.write()
        _, result = self.run_pipeline()
        self.assertEqual(result["status"], "no_match")
        self.assertEqual(result["instances"], [])

    def test_single_material_failure_keeps_other_instances(self):
        class FailsOnce(Material):
            calls = 0
            def predict(self, image):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("test failure")
                return super().predict(image)
        _, result = self.run_pipeline(FailsOnce())
        self.assertEqual([v["material"]["status"] for v in result["instances"]], ["error", "unknown"])
        self.assertEqual(len(result["errors"]), 1)

    def test_rejects_unsafe_references_duplicate_ids_and_bad_boxes(self):
        original = json.loads(json.dumps(self.data))
        for key, value in [("mask_path", "../outside.png"), ("instance_id", "i002"),
                           ("instance_id", "CON"), ("bbox_xyxy", [0., 1, 2, 3]),
                           ("bbox_xyxy", [0, 0, 2, 3]), ("det_score", float("nan"))]:
            with self.subTest(key=key, value=value):
                self.data = json.loads(json.dumps(original))
                self.data["instances"][0][key] = value
                self.write()
                with self.assertRaises(ValueError):
                    SavedSegmenter(self.path)

    def test_rejects_nonbinary_empty_and_wrong_size_masks(self):
        for pixels in [np.full((4, 6), 1, dtype=np.uint8), np.zeros((4, 6), dtype=np.uint8), np.zeros((2, 3), dtype=np.uint8)]:
            Image.fromarray(pixels).save(self.source / "masks/i014.png")
            with self.assertRaises(ValueError):
                SavedSegmenter(self.path)

    def test_rejects_wrong_rgb_size(self):
        with self.assertRaisesRegex(ValueError, "RGB dimensions"):
            SavedSegmenter(self.path).predict(Image.new("RGB", (3, 2)), "box", "full")

    def test_comparison_matches_spatially_after_ids_are_swapped(self):
        from scripts.compare_saved_segmentations import compare
        other = json.loads(json.dumps(self.data))
        other["instances"][0]["instance_id"] = "i002"
        other["instances"][1]["instance_id"] = "i014"
        path = self.source / "other.json"
        path.write_text(json.dumps(other), encoding="utf-8")
        result = compare(self.path, path)
        self.assertEqual(result["matched_count"], 2)
        self.assertEqual(result["different_instance_id_count"], 2)
        self.assertEqual(result["mean_mask_iou"], 1.0)

    def test_normal_inference_exports_integer_mask_box(self):
        class Segmenter:
            def predict(self, image, prompt, mode):
                mask = np.zeros((4, 6), dtype=bool)
                mask[1:3, 1:3] = True
                return Prediction(image, [Instance([.2, .4, 4.8, 3.9], .9, "box", mask)], mode, prompt, {}, 1, {})
        output = self.root / "normal"
        PerceptionPipeline(self.config, segmenter=Segmenter(), material=Material()).process(self.rgb, "box", output)
        result = json.loads((output / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(result["instances"][0]["bbox_xyxy"], [1, 1, 3, 3])
        with Image.open(output / "rois/i000_bbox_0.png") as image:
            self.assertEqual(image.size, (2, 2))


if __name__ == "__main__":
    unittest.main()

import csv
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.approve_training_dataset import export_manifest
from scripts.build_training_dataset import _mask_box, _priority


class TrainingPipelineTests(unittest.TestCase):
    def test_mask_box_is_half_open(self):
        class FakeMask:
            def nonzero(self):
                return ([1, 2, 3], [2, 5, 3])
        mask = FakeMask()
        self.assertEqual(_mask_box(mask), (2, 1, 6, 4))

    def test_priority_never_claims_confirmed(self):
        self.assertEqual(_priority(.95, .95, .30, .85, .20), "quick_confirm")
        self.assertEqual(_priority(.95, .52, .05, .85, .20), "重点复核")

    def test_export_requires_human_confirmation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new("RGB", (10, 10)).save(root / "crop.png")
            source = {
                "schema_version": "2.0-preannotation",
                "objects": [
                    {"object_id": "a", "annotation_status": "needs_review", "crop_path": "crop.png"},
                    {"object_id": "b", "annotation_status": "confirmed", "final_material": "paper",
                     "split": "train", "crop_path": "crop.png", "frame_id": "f1"},
                ],
            }
            reviewed = root / "preannotations.json"
            reviewed.write_text(json.dumps(source), encoding="utf-8")
            summary = export_manifest(reviewed, root / "training_manifest.csv")
            self.assertEqual(summary["confirmed"], 1)
            with (root / "training_manifest.csv").open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["label"], "paper")


if __name__ == "__main__":
    unittest.main()

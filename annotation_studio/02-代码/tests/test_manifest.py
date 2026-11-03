import csv
import tempfile
import unittest
from pathlib import Path

from material_recognition.validate_manifest import validate


class ManifestValidationTest(unittest.TestCase):
    def test_rejects_sku_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.csv"
            with manifest.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=("sample_id", "split", "label", "label_id",
                                                          "image_path", "sku_id"))
                writer.writeheader()
                writer.writerows([
                    {"sample_id": "one", "split": "train", "label": "paper", "label_id": 0,
                     "image_path": "one.jpg", "sku_id": "sku-1"},
                    {"sample_id": "two", "split": "test", "label": "paper", "label_id": 0,
                     "image_path": "two.jpg", "sku_id": "sku-1"},
                ])
            with self.assertRaisesRegex(ValueError, "SKU appears in multiple splits"):
                validate(manifest, check_files=False)

    def test_rejects_source_image_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.csv"
            with manifest.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=("sample_id", "split", "label", "label_id",
                                                          "image_path", "parent_image_id"))
                writer.writeheader()
                writer.writerows([
                    {"sample_id": "one", "split": "train", "label": "paper", "label_id": 0,
                     "image_path": "one.jpg", "parent_image_id": "frame-1"},
                    {"sample_id": "two", "split": "validation", "label": "plastic", "label_id": 1,
                     "image_path": "two.jpg", "parent_image_id": "frame-1"},
                ])
            with self.assertRaisesRegex(ValueError, "Source image appears in multiple splits"):
                validate(manifest, check_files=False)


if __name__ == "__main__":
    unittest.main()

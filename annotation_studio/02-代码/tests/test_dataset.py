import csv
import tempfile
import unittest
from pathlib import Path

from material_recognition.data import ManifestDataset


class EnterpriseDatasetTest(unittest.TestCase):
    def test_unconfirmed_annotation_cannot_enter_training(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.csv"
            with manifest.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=("sample_id", "split", "label", "label_id",
                                                          "image_path", "annotation_status"))
                writer.writeheader()
                writer.writerow({"sample_id": "one", "split": "train", "label": "paper",
                                 "label_id": 0, "image_path": "one.jpg", "annotation_status": "needs_review"})
            with self.assertRaisesRegex(ValueError, "Unconfirmed annotations"):
                ManifestDataset(manifest, "train")


if __name__ == "__main__":
    unittest.main()

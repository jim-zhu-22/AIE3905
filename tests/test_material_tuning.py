import csv
from pathlib import Path
import tempfile
import unittest
import torch
from perception.models.resnet18.model import build_model
from perception.models.resnet18.tune import frame_folds, choose_replay, configure_scope, read_manifest


class TuningProtocolTests(unittest.TestCase):
    def test_folds_never_use_historical_test_or_heldout_frame(self):
        rows = [{"sample_id": f"{frame}-{label}", "parent_image_id": frame, "label": label,
                 "split": "test" if frame == "last" else "train"}
                for frame in ("a", "b", "c", "last") for label in ("paper", "plastic")]
        development, historical, folds = frame_folds(rows)
        self.assertEqual((len(development), len(historical), len(folds)), (6, 2, 3))
        for group, train, validation in folds:
            self.assertNotIn(group, {r["parent_image_id"] for r in train})
            self.assertNotIn("last", {r["parent_image_id"] for r in train + validation})

    def test_replay_uses_only_public_train_and_balances_classes(self):
        rows = [{"sample_id": f"{split}-{label}", "label": label, "split": split}
                for split in ("train", "validation", "test")
                for label in ("paper", "plastic", "metal", "wood", "fabric")]
        selected = choose_replay(rows, 1, 42)
        self.assertEqual(len(selected), 5)
        self.assertTrue(all(row["split"] == "train" for row in selected))

    def test_layer4_scope_preserves_earlier_weights_and_bn_statistics(self):
        model = build_model(False)
        configure_scope(model, "layer4")
        self.assertFalse(model.conv1.weight.requires_grad)
        self.assertTrue(model.layer4[0].conv1.weight.requires_grad)
        self.assertTrue(model.fc.weight.requires_grad)
        self.assertFalse(model.bn1.training)
        self.assertFalse(model.layer4[0].bn1.training)

    def test_unreviewed_material_cannot_enter_training(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.csv"
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["sample_id", "label", "label_id", "annotation_status", "image_path"])
                writer.writeheader()
                writer.writerow({"sample_id": "a", "label": "paper", "label_id": 0,
                                 "annotation_status": "needs_review", "image_path": "missing.png"})
            with self.assertRaisesRegex(ValueError, "confirmed"):
                read_manifest(path, "enterprise")


if __name__ == "__main__":
    unittest.main()

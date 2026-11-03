import csv
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights

from . import LABELS


def transform(training: bool):
    weights = ResNet18_Weights.IMAGENET1K_V1
    if not training:
        return weights.transforms()
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


class ManifestDataset(Dataset):
    def __init__(self, manifest: Path, split: str, training: bool = False, limit: int = 0):
        self.root = manifest.parent
        with manifest.open(newline="", encoding="utf-8") as file:
            self.rows = [row for row in csv.DictReader(file) if row["split"] == split]
        unconfirmed = [row["sample_id"] for row in self.rows
                       if "annotation_status" in row and row["annotation_status"] != "confirmed"]
        if unconfirmed:
            raise ValueError(f"Unconfirmed annotations in {split}: {unconfirmed[:5]}")
        if limit:
            # Fixed prefix only for a smoke run; the full experiment never uses this option.
            self.rows = self.rows[:limit]
        if not self.rows:
            raise ValueError(f"No rows for split={split} in {manifest}")
        self.transform = transform(training)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        label_id = int(row["label_id"])
        if LABELS[label_id] != row["label"]:
            raise ValueError(f"Label mapping mismatch for {row['sample_id']}")
        with Image.open(self.root / row["image_path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, label_id

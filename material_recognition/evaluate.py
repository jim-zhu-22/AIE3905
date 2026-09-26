import argparse
import csv
import json
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader

from . import LABELS
from .data import ManifestDataset
from .model import build_model


def evaluate(checkpoint_path: Path, manifest: Path, split: str, output_dir: Path, batch_size: int = 32):
    torch.set_num_threads(8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if tuple(checkpoint["labels"]) != LABELS:
        raise ValueError("Checkpoint label order does not match this project")
    model = build_model(pretrained=False)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    data = ManifestDataset(manifest, split)
    loader = DataLoader(data, batch_size=batch_size, num_workers=2, pin_memory=device.type == "cuda")
    true, pred, scores = [], [], []
    with torch.inference_mode():
        for images, targets in loader:
            probability = model(images.to(device)).softmax(1).cpu()
            true.extend(targets.tolist())
            pred.extend(probability.argmax(1).tolist())
            scores.extend(probability.max(1).values.tolist())
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "device": str(device),
        "checkpoint": str(checkpoint_path),
        "split": split,
        "samples": len(true),
        "accuracy": accuracy_score(true, pred),
        "classification": classification_report(true, pred, labels=list(range(len(LABELS))), target_names=LABELS,
                                                output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(true, pred, labels=list(range(len(LABELS)))).tolist(),
        "confusion_labels": LABELS,
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("sample_id", "true_label", "predicted_label", "max_score"))
        for row, actual, predicted, score in zip(data.rows, true, pred, scores):
            writer.writerow((row["sample_id"], LABELS[actual], LABELS[predicted], f"{score:.6f}"))
    print(json.dumps({"split": split, "samples": len(true), "accuracy": report["accuracy"],
                      "macro_f1": report["classification"]["macro avg"]["f1-score"]}, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/minc_resnet18/best.pt"))
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/minc5/manifest.csv"))
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/minc_resnet18/evaluation"))
    args = parser.parse_args()
    evaluate(args.checkpoint, args.manifest, args.split, args.output_dir)


if __name__ == "__main__":
    main()

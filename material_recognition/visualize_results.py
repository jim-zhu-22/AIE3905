"""Export validation curves, confusion matrix and representative predictions."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from . import LABELS


def generate(run_dir: Path, manifest: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "evaluation/metrics.json").read_text(encoding="utf-8"))
    with (run_dir / "evaluation/predictions.csv").open(encoding="utf-8") as file:
        predictions = list(csv.DictReader(file))
    with manifest.open(encoding="utf-8") as file:
        samples = {row["sample_id"]: row for row in csv.DictReader(file)}
    epochs = np.arange(1, len(history) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    for split in ("train", "validation"):
        axes[0].plot(epochs, [row[split]["loss"] for row in history], marker="o", label=split)
        axes[1].plot(epochs, [row[split]["macro_f1"] for row in history], marker="o", label=split)
    for ax in axes:
        ax.set_xlabel("Epoch"); ax.set_xticks(epochs); ax.grid(alpha=0.2); ax.legend()
        head_count = sum(row["stage"] == "head" for row in history)
        if head_count < len(history):
            ax.axvline(head_count + 0.5, color="gray", linestyle="--", alpha=0.7)
    axes[0].set_title("Cross-entropy loss"); axes[1].set_title("Macro-F1"); axes[1].set_ylim(0, 1)
    fig.tight_layout(); fig.savefig(output_dir / "training_curves.png", dpi=180); plt.close(fig)
    matrix = np.array(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set(xticks=range(5), yticks=range(5), xticklabels=LABELS, yticklabels=LABELS,
           xlabel="Predicted label", ylabel="True label", title=f"Validation confusion matrix (n={matrix.sum()})")
    for i in range(5):
        for j in range(5):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    fig.colorbar(image, ax=ax, shrink=0.8); fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=180); plt.close(fig)
    selected = []
    for label in LABELS:
        correct = [row for row in predictions if row["true_label"] == row["predicted_label"] == label]
        if correct:
            selected.append(max(correct, key=lambda row: float(row["max_score"])))
    errors = sorted((row for row in predictions if row["true_label"] != row["predicted_label"]),
                    key=lambda row: float(row["max_score"]), reverse=True)
    selected.extend(errors[:5])
    fig, axes = plt.subplots(2, 5, figsize=(12, 5.8))
    for ax in axes.flat:
        ax.axis("off")
    for ax, row in zip(axes.flat, selected):
        sample = samples[row["sample_id"]]
        with Image.open(manifest.parent / sample["image_path"]) as image:
            ax.imshow(image.convert("RGB"))
        ax.set_title(f"True: {row['true_label']}\nPred: {row['predicted_label']} ({float(row['max_score']):.2f})",
                     fontsize=9, color="darkgreen" if row["true_label"] == row["predicted_label"] else "firebrick")
    fig.suptitle("Selected validation examples: correct predictions and high-score errors", fontsize=12)
    fig.tight_layout(); fig.savefig(output_dir / "prediction_examples.png", dpi=180); plt.close(fig)
    (output_dir / "selected_examples.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    print(f"Saved result figures to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("runs/week1_resnet18"))
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/minc5/manifest.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs/week1_results"))
    args = parser.parse_args()
    generate(args.run_dir, args.manifest, args.output_dir)

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader

from . import LABELS
from .data import ManifestDataset
from .model import build_model


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    true, predicted = [], []
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, targets)
            if training:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * len(targets)
        true.extend(targets.cpu().tolist())
        predicted.extend(logits.argmax(1).cpu().tolist())
    return {
        "loss": total_loss / len(true),
        "accuracy": accuracy_score(true, predicted),
        "macro_f1": f1_score(true, predicted, labels=list(range(len(LABELS))), average="macro", zero_division=0),
    }


def train(args):
    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets = {
        "train": ManifestDataset(args.manifest, "train", training=True, limit=args.limit),
        "validation": ManifestDataset(args.manifest, "validation", limit=args.limit),
    }
    loaders = {
        split: DataLoader(data, batch_size=args.batch_size, shuffle=(split == "train"), num_workers=args.workers,
                          pin_memory=(device.type == "cuda"))
        for split, data in datasets.items()
    }
    model = build_model(pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    best_f1 = -1.0
    history = []
    for stage, epochs, rate in (("head", args.head_epochs, args.head_lr), ("full", args.full_epochs, args.full_lr)):
        for parameter in model.parameters():
            parameter.requires_grad = stage == "full"
        for parameter in model.fc.parameters():
            parameter.requires_grad = True
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=rate)
        for epoch in range(epochs):
            metrics = {
                "stage": stage,
                "epoch": epoch + 1,
                "train": run_epoch(model, loaders["train"], criterion, device, optimizer),
                "validation": run_epoch(model, loaders["validation"], criterion, device),
            }
            history.append(metrics)
            print(json.dumps(metrics, ensure_ascii=False), flush=True)
            if metrics["validation"]["macro_f1"] > best_f1:
                best_f1 = metrics["validation"]["macro_f1"]
                torch.save({"model": model.state_dict(), "labels": LABELS, "architecture": "resnet18",
                            "preprocessing": "ResNet18_Weights.IMAGENET1K_V1.transforms"}, args.output_dir / "best.pt")
            (args.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update({"device": str(device), "torch": torch.__version__, "labels": LABELS})
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return best_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/minc5/manifest.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/minc_resnet18"))
    parser.add_argument("--head-epochs", type=int, default=2)
    parser.add_argument("--full-epochs", type=int, default=3)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--full-lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0, help="Smoke runs only; fixed prefix per split")
    parser.add_argument("--no-pretrained", action="store_true", help="Testing only")
    print(f"Best validation Macro-F1: {train(parser.parse_args()):.4f}")


if __name__ == "__main__":
    main()

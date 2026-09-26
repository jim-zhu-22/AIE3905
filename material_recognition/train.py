import argparse
import json
import random
import time
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
    if training and not model.conv1.weight.requires_grad:
        # Frozen backbone also keeps its pretrained BatchNorm statistics.
        model.eval()
        model.fc.train()
    total_loss = 0.0
    true, predicted = [], []
    print(f"{'Training' if training else 'Validation'}: {len(loader)} batches", flush=True)
    for batch_number, (images, targets) in enumerate(loader, 1):
        if batch_number == 1:
            print("First batch loaded", flush=True)
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
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
        if batch_number == 1:
            print("First batch computation completed", flush=True)
        if training and batch_number % 100 == 0:
            print(f"Training batch {batch_number}/{len(loader)}", flush=True)
    return {
        "loss": total_loss / len(true),
        "accuracy": accuracy_score(true, predicted),
        "macro_f1": f1_score(true, predicted, labels=list(range(len(LABELS))), average="macro", zero_division=0),
    }


def train(args):
    if args.cpu_threads > 0:
        torch.set_num_threads(args.cpu_threads)
    seed_all(args.seed)
    requested = getattr(args, "device", "auto")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable in this Python environment")
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if requested == "auto" else requested)
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
    initial_checkpoint = None
    if args.init_checkpoint:
        initial_checkpoint = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        if tuple(initial_checkpoint["labels"]) != LABELS:
            raise ValueError("Initial checkpoint label order differs")
        model.load_state_dict(initial_checkpoint["model"])
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update({"device": str(device), "torch": torch.__version__, "labels": LABELS,
                   "cuda_runtime": torch.version.cuda,
                   "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                   "samples": {split: len(data) for split, data in datasets.items()}})
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps(config), flush=True)
    criterion = nn.CrossEntropyLoss()
    history = json.loads(args.prior_history.read_text(encoding="utf-8")) if args.prior_history else []
    if history and initial_checkpoint is None:
        raise ValueError("Prior history requires an initial checkpoint")
    best_f1 = max((row["validation"]["macro_f1"] for row in history), default=-1.0)
    if initial_checkpoint is not None:
        torch.save(initial_checkpoint, args.output_dir / "best.pt")
    for stage, epochs, rate in (("head", args.head_epochs, args.head_lr), ("full", args.full_epochs, args.full_lr)):
        for parameter in model.parameters():
            parameter.requires_grad = stage == "full"
        for parameter in model.fc.parameters():
            parameter.requires_grad = True
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=rate)
        for epoch in range(epochs):
            started = time.perf_counter()
            metrics = {
                "stage": stage,
                "epoch": epoch + 1,
                "train": run_epoch(model, loaders["train"], criterion, device, optimizer),
                "validation": run_epoch(model, loaders["validation"], criterion, device),
            }
            metrics["elapsed_seconds"] = time.perf_counter() - started
            history.append(metrics)
            print(json.dumps(metrics, ensure_ascii=False), flush=True)
            if metrics["validation"]["macro_f1"] > best_f1:
                best_f1 = metrics["validation"]["macro_f1"]
                torch.save({"model": model.state_dict(), "labels": LABELS, "architecture": "resnet18",
                            "preprocessing": "ResNet18_Weights.IMAGENET1K_V1.transforms"}, args.output_dir / "best.pt")
            (args.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
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
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--prior-history", type=Path, help="Earlier stages on the same validation split")
    parser.add_argument("--limit", type=int, default=0, help="Smoke runs only; fixed prefix per split")
    parser.add_argument("--no-pretrained", action="store_true", help="Testing only")
    print(f"Best validation Macro-F1: {train(parser.parse_args()):.4f}")


if __name__ == "__main__":
    main()

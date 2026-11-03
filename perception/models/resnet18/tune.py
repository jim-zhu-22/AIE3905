"""Frame-grouped tuning with public-data replay and a locked historical holdout.

Every CV fold restarts from the public MINC checkpoint, which has not seen the
enterprise frames. The already-adapted checkpoint is a comparison only.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import random
from time import perf_counter

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from .labels import LABELS
from .model import build_model
from .preprocess import transform
from perception.io.manifest import write_json
from perception.runtime import resolve_device, runtime_versions, sha256

RECIPES = {
    "head_replay": {"scope": "head", "head_lr": .001, "body_lr": .001},
    "layer4_replay": {"scope": "layer4", "head_lr": .001, "body_lr": .00003},
    "full_replay": {"scope": "full", "head_lr": .001, "body_lr": .00001},
}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_manifest(path, source):
    path = Path(path).resolve()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    seen = set()
    for row in rows:
        if row["sample_id"] in seen:
            raise ValueError("Duplicate sample identifier")
        seen.add(row["sample_id"])
        label_id = int(row["label_id"])
        if not 0 <= label_id < len(LABELS) or LABELS[label_id] != row["label"]:
            raise ValueError("Label mapping mismatch")
        if source == "enterprise" and row.get("annotation_status") != "confirmed":
            raise ValueError("Enterprise training requires confirmed annotations")
        row["path"] = str((path.parent / row["image_path"]).resolve())
        row["source"] = source
        if not Path(row["path"]).is_file():
            raise FileNotFoundError(row["path"])
    return rows


def frame_folds(rows):
    # Only the old training frames enter tuning; the historical test stays locked.
    development = [row for row in rows if row["split"] == "train"]
    heldout = [row for row in rows if row["split"] == "test"]
    groups = sorted({row.get("parent_image_id", "") for row in development})
    if len(groups) < 3 or "" in groups or not heldout:
        raise ValueError("Need >=3 training frames and a separate historical holdout")
    if set(groups) & {row.get("parent_image_id") for row in heldout}:
        raise ValueError("Frame leakage into historical holdout")
    folds = []
    for group in groups:
        train = [row for row in development if row["parent_image_id"] != group]
        validation = [row for row in development if row["parent_image_id"] == group]
        if {row["label"] for row in validation} != {row["label"] for row in development}:
            raise ValueError("Every held-out frame must cover all enterprise classes")
        folds.append((group, train, validation))
    return development, heldout, folds


def choose_replay(rows, per_class, seed):
    randomizer = random.Random(seed)
    selected = []
    for label in LABELS:
        candidates = sorted((r for r in rows if r["split"] == "train" and r["label"] == label),
                            key=lambda row: row["sample_id"])
        if len(candidates) < per_class:
            raise ValueError("Not enough public training samples")
        selected.extend(randomizer.sample(candidates, per_class))
    return selected


def pixel_hash(row):
    import hashlib
    with Image.open(row["path"]) as source:
        image = source.convert("RGB")
    return hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()


def audit_groups(partitions):
    hashes = {}
    for name, rows in partitions.items():
        hashes[name] = {pixel_hash(row) for row in rows}
    overlaps = {}
    names = list(hashes)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            count = len(hashes[left] & hashes[right])
            if count:
                overlaps[f"{left}/{right}"] = count
    if overlaps:
        raise ValueError(f"Exact image duplicates across partitions: {overlaps}")
    return {"exact_cross_partition_duplicates": overlaps,
            "unique_pixels": {name: len(values) for name, values in hashes.items()}}


class CachedImages(Dataset):
    def __init__(self, rows, training, cache):
        self.rows = rows
        self.transform = transform(training)
        self.cache = cache
        for row in rows:
            if row["path"] not in cache:
                with Image.open(row["path"]) as image:
                    cache[row["path"]] = image.convert("RGB")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return self.transform(self.cache[row["path"]]), int(row["label_id"])


def configure_scope(model, scope):
    for name, parameter in model.named_parameters():
        parameter.requires_grad = (scope == "full" or name.startswith("fc.") or
                                   (scope == "layer4" and name.startswith("layer4.")))
    # Small frame datasets are unstable for running BatchNorm statistics.
    model.train()
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def evaluate(model, rows, device, cache, batch_size):
    loader = DataLoader(CachedImages(rows, False, cache), batch_size=batch_size, shuffle=False)
    model.eval()
    truth, probabilities = [], []
    loss_sum = 0.0
    with torch.inference_mode():
        for images, targets in loader:
            logits = model(images.to(device))
            loss_sum += nn.functional.cross_entropy(logits, targets.to(device), reduction="sum").item()
            probabilities.extend(logits.softmax(1).cpu().tolist())
            truth.extend(targets.tolist())
    predicted = np.asarray(probabilities).argmax(1)
    covered = sorted(set(truth))
    metrics = {
        "samples": len(rows), "accuracy": float(accuracy_score(truth, predicted)),
        "macro_f1_covered": float(f1_score(truth, predicted, labels=covered, average="macro", zero_division=0)),
        "macro_f1_five": float(f1_score(truth, predicted, labels=list(range(5)), average="macro", zero_division=0)),
        "covered_labels": [LABELS[i] for i in covered], "cross_entropy": loss_sum / len(rows),
        "confusion_labels": list(LABELS), "confusion_matrix": confusion_matrix(truth, predicted, labels=list(range(5))).tolist(),
        "per_class_f1": dict(zip(LABELS, f1_score(truth, predicted, labels=list(range(5)), average=None, zero_division=0).tolist())),
    }
    predictions = [{"sample_id": row["sample_id"], "source": row["source"], "path": row["path"],
                    "true_label": row["label"], "predicted_label": LABELS[int(p)],
                    "scores": dict(zip(LABELS, scores))}
                   for row, p, scores in zip(rows, predicted, probabilities)]
    return metrics, predictions


def train_recipe(initial, recipe, rows, device, cache, args, log_path):
    seed_all(args.seed)
    model = build_model(False).to(device)
    model.load_state_dict(initial["model"])
    loader = DataLoader(CachedImages(rows, True, cache), batch_size=args.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(args.seed))
    history = []
    optimizer = None
    started = perf_counter()
    for epoch in range(args.epochs):
        scope = "head" if epoch == 0 else recipe["scope"]
        configure_scope(model, scope)
        if epoch in (0, 1):
            rate = recipe["head_lr"] if epoch == 0 else recipe["body_lr"]
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                          lr=rate, weight_decay=.01)
        loss_sum, correct, samples = 0.0, 0, 0
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, targets)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            samples += len(targets)
            loss_sum += loss.item() * len(targets)
            correct += int((logits.argmax(1) == targets).sum())
        history.append({"epoch": epoch + 1, "scope": scope, "loss": loss_sum / samples,
                        "training_accuracy": correct / samples, "elapsed_seconds": perf_counter() - started})
        write_json(log_path, history)
        print(f"{log_path.parent.name} epoch {epoch + 1}/{args.epochs}: loss={loss_sum / samples:.4f}", flush=True)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enterprise-manifest", type=Path, required=True)
    parser.add_argument("--public-manifest", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--annotation-workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--replay-per-class", type=int, default=100)
    parser.add_argument("--enterprise-repeat", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.epochs, args.replay_per_class, args.enterprise_repeat, args.batch_size) < 1:
        parser.error("Training counts must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = resolve_device(args.device)
    enterprise = read_manifest(args.enterprise_manifest, "enterprise")
    public = read_manifest(args.public_manifest, "minc")
    development, historical, folds = frame_folds(enterprise)
    replay = choose_replay(public, args.replay_per_class, args.seed)
    public_validation = [row for row in public if row["split"] == "validation"]
    audit = audit_groups({**{f"frame_{group}": validation for group, _, validation in folds},
                          "historical_test": historical, "public_replay": replay, "public_validation": public_validation})
    items = args.annotation_workspace / "items"
    reviews = [json.loads(path.read_text(encoding="utf-8")) for path in items.glob("*/review.json")]
    audit.update({"annotation_items": len(list(items.iterdir())),
                  "prediction_files": len(list(items.glob("*/prediction.json"))),
                  "review_files": len(reviews), "confirmed_reviews": sum(r.get("confirmed") is True for r in reviews),
                  "automatic_candidates_used_for_training": False,
                  "enterprise_counts": dict(Counter(row["label"] for row in development)),
                  "historical_counts": dict(Counter(row["label"] for row in historical)),
                  "public_replay_counts": dict(Counter(row["label"] for row in replay)),
                  "public_validation_count": len(public_validation),
                  "public_test_used": False,
                  "scope": "Adjacent-frame CV, SKU unknown; not cross-SKU or independent deployment evaluation"})
    write_json(args.output / "data_audit.json", audit)
    # Snapshot the complete protocol before looking at any new metrics.
    protocol = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    protocol.update({"recipes": RECIPES, "versions": runtime_versions(),
                     "initial_sha256": sha256(args.initial_checkpoint), "baseline_sha256": sha256(args.baseline_checkpoint),
                     "selection": "mean over frame folds of (enterprise covered macro-F1 + public five-class macro-F1)/2",
                     "tie_break": "lower mean validation cross-entropy, then recipe order",
                     "heldout_policy": "Historical test read for inference only after recipe and final weights are frozen",
                     "deployment_policy": "Candidate is exported separately; historical test does not trigger model selection"})
    write_json(args.output / "protocol.json", protocol)
    write_json(args.output / "dataset_snapshot.json", {"development": development, "historical_test": historical,
               "public_replay": replay, "public_validation": public_validation})
    initial = torch.load(args.initial_checkpoint, map_location="cpu", weights_only=True)
    baseline = torch.load(args.baseline_checkpoint, map_location="cpu", weights_only=True)
    for checkpoint in (initial, baseline):
        if tuple(checkpoint["labels"]) != LABELS:
            raise ValueError("Checkpoint label order differs")
    cache, trials = {}, []
    for name, recipe in RECIPES.items():
        results = []
        for group, train, validation in folds:
            folder = args.output / "cross_validation" / f"{name}_{group}"
            folder.mkdir(parents=True)
            model = train_recipe(initial, recipe, replay + train * args.enterprise_repeat,
                                 device, cache, args, folder / "history.json")
            enterprise_metrics, _ = evaluate(model, validation, device, cache, args.batch_size)
            public_metrics, _ = evaluate(model, public_validation, device, cache, args.batch_size)
            result = {"heldout_frame": group, "training_frames": sorted({r["parent_image_id"] for r in train}),
                      "enterprise": enterprise_metrics, "public": public_metrics}
            results.append(result)
            write_json(folder / "metrics.json", result)
            del model
        trial = {"name": name, "folds": results,
                 "selection_score": float(np.mean([(r["enterprise"]["macro_f1_covered"] + r["public"]["macro_f1_five"]) / 2 for r in results])),
                 "validation_loss": float(np.mean([(r["enterprise"]["cross_entropy"] + r["public"]["cross_entropy"]) / 2 for r in results]))}
        trials.append(trial)
        write_json(args.output / "trials.json", trials)
        print(f"RECIPE {name}: score={trial['selection_score']:.4f}", flush=True)
    selected = max(trials, key=lambda trial: (trial["selection_score"], -trial["validation_loss"]))
    write_json(args.output / "selection.json", {"selected": selected["name"], "score": selected["selection_score"],
               "historical_test_used_for_selection": False})
    model = train_recipe(initial, RECIPES[selected["name"]], replay + development * args.enterprise_repeat,
                         device, cache, args, args.output / "final_history.json")
    checkpoint_path = args.output / "candidate.pt"
    torch.save({"model": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                "labels": LABELS, "architecture": "resnet18", "preprocessing": "ResNet18_Weights.IMAGENET1K_V1.transforms",
                "training_protocol": protocol, "selected_recipe": selected["name"]}, checkpoint_path)
    comparison = {}
    for name, checkpoint in (("previous_minc", initial), ("previous_enterprise", baseline),
                             ("candidate", torch.load(checkpoint_path, map_location="cpu", weights_only=True))):
        model.load_state_dict(checkpoint["model"])
        comparison[name] = {}
        for domain, rows in (("public_validation", public_validation), ("historical_enterprise_test", historical)):
            metrics, predictions = evaluate(model, rows, device, cache, args.batch_size)
            comparison[name][domain] = metrics
            write_json(args.output / f"{name}_{domain}_predictions.json", predictions)
    # Exercise the actual deployment loader after training, not just an in-memory model.
    from .backend import Predictor
    predictor = Predictor(checkpoint_path, device=device)
    with Image.open(historical[0]["path"]) as source:
        prediction = predictor.predict(source)
    write_json(args.output / "summary.json", {"selected_recipe": selected["name"], "comparison": comparison,
               "candidate_sha256": sha256(checkpoint_path), "deployment_loader": "passed",
               "deployment_sample": prediction, "scope": audit["scope"], "default_model_replaced": False})
    print(json.dumps({"selected": selected["name"], "candidate": str(checkpoint_path), "comparison": comparison}), flush=True)


if __name__ == "__main__":
    main()

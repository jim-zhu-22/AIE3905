"""Fixed-budget adjacent-frame holdout experiment; no test-based model selection."""
import argparse
import csv
import hashlib
import html
import json
import shutil
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", type=Path, required=True)
    ap.add_argument("--project-code", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--finalize-only", action="store_true", help="Rebuild evaluation/report from saved final.pt without retraining")
    args = ap.parse_args()
    out = args.output.resolve()
    if args.finalize_only:
        assert (out / "final.pt").is_file()
    else:
        out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, out / "train_chad_holdout.py")
    shutil.copytree(args.project_code / "material_recognition", out / "source" / "material_recognition",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"), dirs_exist_ok=args.finalize_only)
    sys.path.insert(0, str(out / "source"))
    import numpy as np
    import torch
    import torchvision
    from PIL import Image
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
    from torch.utils.data import DataLoader
    from material_recognition import LABELS
    from material_recognition.data import ManifestDataset
    from material_recognition.model import build_model
    from material_recognition.train import run_epoch, seed_all

    torch.set_num_threads(8)
    seed_all(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = out / "dataset"
    data_root.mkdir(exist_ok=args.finalize_only)
    with zipfile.ZipFile(args.archive) as z:
        for entry in z.infolist():
            dest = (data_root / entry.filename).resolve()
            if not dest.is_relative_to(data_root):
                raise ValueError("Unsafe archive path")
        z.extractall(data_root)
    root = data_root / "Chad_frames_002-005_20261013"
    source = json.loads((root / "annotations.json").read_text(encoding="utf-8-sig"))
    rows = list(csv.DictReader((root / "metadata/material_crops.csv").open(encoding="utf-8-sig", newline="")))
    frames = sorted({r["frame_id"] for r in rows})
    assert len(frames) == 4 and len(rows) == 108
    heldout = frames[-1]
    manifest = root / "training_manifest.csv"
    fields = ["sample_id", "image_path", "label", "label_id", "split", "parent_image_id",
              "sku_id", "sequence_id", "annotation_status", "original_annotation_status", "confirmation_source"]
    prepared = []
    hashes = {"train": set(), "test": set()}
    for row in rows:
        path = root / row["crop_path"]
        with Image.open(path) as im:
            im.load()
            assert im.width > 0 and im.height > 0
            pixel_hash = hashlib.sha256(str(im.size).encode() + im.convert("RGB").tobytes()).hexdigest()
        split = "test" if row["frame_id"] == heldout else "train"
        hashes[split].add(pixel_hash)
        prepared.append(dict(sample_id=row["instance_id"], image_path=row["crop_path"], label=row["material"],
            label_id=LABELS.index(row["material"]), split=split, parent_image_id=row["frame_id"], sku_id="",
            sequence_id=row["sequence_group"], annotation_status="confirmed",
            original_annotation_status=row["annotation_status"], confirmation_source="user_chat_2026-10-13"))
    assert not (hashes["train"] & hashes["test"]), "Exact pixel duplicate across split"
    assert len({r["sample_id"] for r in prepared}) == 108
    with manifest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(prepared)
    provenance = {
        "user_confirmation": "这个材质是人为标注的没问题，我需要进行训练",
        "holdout_request": "可以用一张进行测试",
        "source_annotations_unchanged": True,
        "protocol": "First 3 chronological frames train; final frame test. Fixed 2 head + 3 full epochs; final checkpoint only.",
        "test_frame": heldout, "train_frames": frames[:-1],
        "counts": {s: dict(Counter(r["label"] for r in prepared if r["split"] == s)) for s in ["train", "test"]},
        "limitation": "Adjacent-frame exploratory holdout; same capture group, SKU unknown, possible repeated products. Not cross-SKU generalization.",
        "exact_pixel_duplicates_across_split": 0,
        "archive_sha256": sha(args.archive), "initial_checkpoint_sha256": sha(args.checkpoint),
    }
    dump(out / "data_provenance.json", provenance)
    config = {"architecture": "resnet18", "labels": list(LABELS), "seed": 42, "batch_size": 32,
              "epochs": {"head": 2, "full": 3}, "learning_rates": {"head": .001, "full": .0001},
              "weight_decay": .01, "device": str(device), "torch": str(torch.__version__),
              "torchvision": str(torchvision.__version__), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
              "selection": "fixed_final_epoch_no_validation_or_test_selection", "test_frame": heldout,
              "initial_checkpoint": str(args.checkpoint), "manifest": str(manifest),
              "preprocessing": "Reuse original project's data.transform without modification", "five_class_output_retained": True}
    dump(out / "config.json", config)
    print(json.dumps({"config": config, "counts": provenance["counts"]}, ensure_ascii=False), flush=True)
    model = build_model(False).to(device)
    initial = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    assert tuple(initial["labels"]) == LABELS
    model.load_state_dict(initial["model"])
    dataset = ManifestDataset(manifest, "train", training=True)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
    criterion = torch.nn.CrossEntropyLoss()
    history = json.loads((out / "history.json").read_text()) if args.finalize_only else []
    start = time.perf_counter()
    for stage, epochs, lr in ([] if args.finalize_only else [("head", 2, .001), ("full", 3, .0001)]):
        for p in model.parameters():
            p.requires_grad = stage == "full"
        for p in model.fc.parameters():
            p.requires_grad = True
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr)
        for epoch in range(epochs):
            tick = time.perf_counter()
            stats = run_epoch(model, loader, criterion, device, optimizer)
            assert np.isfinite(stats["loss"])
            if device.type == "cuda":
                torch.cuda.synchronize()
            history.append({"stage": stage, "epoch": epoch + 1, "training": stats, "seconds": time.perf_counter() - tick})
            dump(out / "history.json", history)
            print(json.dumps(history[-1]), flush=True)
        checkpoint = {"model": {k: v.cpu().clone() for k, v in model.state_dict().items()}, "labels": LABELS,
                      "architecture": "resnet18", "preprocessing": "ResNet18_Weights.IMAGENET1K_V1.transforms",
                      "training_protocol": config, "completed_stage": stage}
        torch.save(checkpoint, out / ("head.pt" if stage == "head" else "final.pt"))
    training_seconds = sum(row["seconds"] for row in history)
    # Evaluation occurs after the training budget is exhausted and never selects a checkpoint.
    def assess(state, split, name):
        model.load_state_dict(state["model"])
        model.eval()
        ds = ManifestDataset(manifest, split, training=False)
        dl = DataLoader(ds, batch_size=32, num_workers=0)
        true, probs = [], []
        with torch.inference_mode():
            for x, y in dl:
                p = model(x.to(device)).softmax(1).cpu().numpy()
                assert np.isfinite(p).all()
                probs.extend(p.tolist())
                true.extend(y.tolist())
        pred = np.asarray(probs).argmax(1).tolist()
        result = {"split": split, "samples": len(true), "accuracy": accuracy_score(true, pred),
                  "paper_plastic_macro_f1": f1_score(true, pred, labels=[0, 1], average="macro", zero_division=0),
                  "five_class_report": classification_report(true, pred, labels=list(range(5)), target_names=LABELS,
                                                               output_dict=True, zero_division=0),
                  "confusion_labels": list(LABELS), "confusion_matrix": confusion_matrix(true, pred, labels=list(range(5))).tolist(),
                  "predicted_other_three": sum(x not in [0, 1] for x in pred), "scope": provenance["limitation"]}
        dump(out / (name + "_metrics.json"), result)
        predictions = [dict(sample_id=r["sample_id"], crop=r["image_path"], true_label=LABELS[t],
                            predicted_label=LABELS[p], correct=t == p, scores=dict(zip(LABELS, scores)))
                       for r, t, p, scores in zip(ds.rows, true, pred, probs)]
        dump(out / (name + "_predictions.json"), predictions)
        return result, predictions
    final_state = torch.load(out / "final.pt", map_location="cpu", weights_only=True)
    after, after_rows = assess(final_state, "test", "finetuned_test")
    before, before_rows = assess(initial, "test", "baseline_test")
    train_fit, _ = assess(final_state, "train", "training_fit")
    # Check saved checkpoint against the existing deployment predictor.
    from material_recognition.infer import Predictor
    predictor = Predictor(out / "final.pt")
    with Image.open(root / after_rows[0]["crop"]) as im:
        smoke = predictor.predict(im)
        # Compare at identical batch size: CUDA kernels can differ slightly between batch 1 and 32.
        with torch.inference_mode():
            reference_scores = model(predictor.preprocess(im.convert("RGB")).unsqueeze(0).to(device)).softmax(1)[0].cpu().tolist()
    assert smoke["label"] == after_rows[0]["predicted_label"]
    max_delta = max(abs(smoke["scores"][k] - after_rows[0]["scores"][k]) for k in LABELS)
    single_delta = max(abs(smoke["scores"][k] - v) for k, v in zip(LABELS, reference_scores))
    assert single_delta < 1e-5
    summary = {"baseline_test": before, "finetuned_test": after, "training_fit": train_fit,
               "training_seconds": training_seconds, "final_sha256": sha(out / "final.pt"),
               "deployment_predictor_check": "passed", "single_vs_batch_prediction_max_delta": max_delta,
               "same_batch_size_prediction_max_delta": single_delta,
               "no_independent_validation": True, "test_used_for_model_selection": False}
    dump(out / "summary.json", summary)
    table = ""
    for old, new in zip(before_rows, after_rows):
        picture = (root / new["crop"]).relative_to(out).as_posix()
        table += f'<tr><td><img width="96" src="{html.escape(picture)}"></td><td>{html.escape(new["sample_id"])}</td><td>{new["true_label"]}</td><td>{old["predicted_label"]}</td><td style="color:{"green" if new["correct"] else "red"}">{new["predicted_label"]}</td></tr>'
    doc = f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>材质识别训练结果</title>
<style>body{{font-family:Arial,"Microsoft YaHei";max-width:1100px;margin:40px auto;padding:0 20px;color:#24334b}}table{{border-collapse:collapse;width:100%}}td,th{{padding:10px;border-bottom:1px solid #ddd;text-align:left}}.note{{background:#fff5dc;padding:16px;border-radius:8px}}img{{image-rendering:auto}}</style>
<h1>材质识别：3帧训练，1帧测试</h1><p>2026-10-13 · ResNet-18 · MINC权重 → 现场微调 · {html.escape(str(device))}</p>
<p>训练76个：纸32、塑料44；测试32个：纸14、塑料18。人工标签由用户在本次聊天中确认。</p>
<p class="note">相邻帧探索性测试：四帧同属一个连续采集组，SKU未知，可能重复出现商品。以下结果不代表未见SKU泛化。仅使用人工框裁图，未测试自动定位与分割。保留五类输出，现场测试仅覆盖纸和塑料。</p>
<table><tr><th>模型</th><th>测试准确率</th><th>纸/塑料 Macro-F1</th><th>误判为其他三类</th></tr>
<tr><td>原MINC基线</td><td>{before['accuracy']:.2%}</td><td>{before['paper_plastic_macro_f1']:.4f}</td><td>{before['predicted_other_three']}</td></tr>
<tr><td>本次微调</td><td>{after['accuracy']:.2%}</td><td>{after['paper_plastic_macro_f1']:.4f}</td><td>{after['predicted_other_three']}</td></tr></table>
<p>固定训练分类头2轮＋全网络3轮，不用测试成绩选择权重。训练耗时{training_seconds:.1f}秒。原有推理接口加载与预测检查通过。</p>
<p><a href="final.pt">模型权重</a> · <a href="summary.json">完整指标</a> · <a href="config.json">训练配置</a> · <a href="data_provenance.json">数据划分与来源</a></p>
<h2>32个测试商品逐图对比</h2><table><tr><th>原始裁图</th><th>样本</th><th>人工标签</th><th>微调前</th><th>微调后</th></tr>{table}</table></html>'''
    (out / "report.html").write_text(doc, encoding="utf-8")
    print(json.dumps({"output": str(out), "baseline_test_accuracy": before["accuracy"],
                      "finetuned_test_accuracy": after["accuracy"], "finetuned_macro_f1": after["paper_plastic_macro_f1"],
                      "training_seconds": training_seconds, "deployment_check": "passed"}), flush=True)


if __name__ == "__main__":
    main()

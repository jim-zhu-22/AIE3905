"""Reproduce the 24 manually selected ROI tests without training or label inference."""
import argparse
import csv
import hashlib
import json
import platform
import shutil
import time
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18

LABELS = ["paper", "plastic", "metal", "wood", "fabric"]
SELECTION = [
    dict(index=23, frame="2026_08_26_09_19_22_657", scene="Reference", scene_zh="原有人工复核样本",
         boxes=[[615,156,673,201],[709,221,758,282],[490,314,558,367],[810,383,863,451]],
         truth=["paper","plastic","plastic","paper"]),
    dict(index=45, frame="2026_08_26_09_22_21_794", scene="Human interaction", scene_zh="手部操作和局部遮挡",
         boxes=[[750,301,815,350],[679,104,724,155],[672,151,731,202],[560,495,618,549]]),
    dict(index=89, frame="2026_08_26_09_27_34_596", scene="Sparse A", scene_zh="稀疏摆放一",
         boxes=[[602,274,668,327],[568,349,635,411],[539,387,603,444],[690,277,754,347]]),
    dict(index=134, frame="2026_08_26_09_33_38_378", scene="Sparse B", scene_zh="稀疏摆放二",
         boxes=[[540,357,596,426],[586,357,643,425],[644,405,701,472],[735,293,803,366]]),
    dict(index=156, frame="2026_08_26_09_36_26_391", scene="Dense overlap", scene_zh="密集摆放和物体接触",
         boxes=[[480,317,549,375],[727,245,796,303],[614,351,672,406],[826,342,896,409]]),
    dict(index=245, frame="2026_08_26_09_54_15_753", scene="Later sequence", scene_zh="较晚采集时段密集场景",
         boxes=[[568,271,636,334],[667,243,739,310],[683,297,753,369],[785,349,830,407]])
]

def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--source", type=Path, help="RGBD directory; defaults to project data/映界_rgbd数据")
    parser.add_argument("--checkpoint", type=Path, help="Trusted project checkpoint; defaults to runs/week1_resnet18/best.pt")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    for name in ["frames", "crops", "figures", "code"]:
        (out / name).mkdir(parents=True, exist_ok=True)
    source = args.source or args.project_root / "data/映界_rgbd数据"
    checkpoint_path = args.checkpoint or args.project_root / "runs/week1_resnet18/best.pt"
    torch.set_num_threads(8)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert list(checkpoint["labels"]) == LABELS, "Checkpoint class order differs"
    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(LABELS))
    model.load_state_dict(checkpoint["model"])
    model.cpu().eval()
    preprocess = ResNet18_Weights.IMAGENET1K_V1.transforms()
    records = []
    crops = []
    times = []
    frame_images = []
    for s in SELECTION:
        image_path = source / (s["frame"] + "_color.jpg")
        depth_path = source / (s["frame"] + "_depth.png")
        image = Image.open(image_path).convert("RGB")
        depth = np.array(Image.open(depth_path))
        assert image.size == (1280, 960) and depth.shape == (960, 1280)
        shutil.copy2(image_path, out / "frames" / image_path.name)
        shutil.copy2(depth_path, out / "frames" / depth_path.name)
        frame_images.append(image.copy())
        for i, box in enumerate(s["boxes"]):
            x1,y1,x2,y2 = box
            assert 0 <= x1 < x2 <= image.width and 0 <= y1 < y2 <= image.height
            crop = image.crop(box)
            tensor = preprocess(crop).unsqueeze(0)
            start = time.perf_counter()
            with torch.inference_mode():
                scores = model(tensor).softmax(1)[0].tolist()
            times.append(time.perf_counter()-start)
            order = sorted(range(5), key=lambda k: scores[k], reverse=True)
            prediction = LABELS[order[0]]
            truth = s.get("truth", [None]*4)[i]
            object_id = f"f{s['index']:03d}_o{i+1}"
            crop.save(out / "crops" / (object_id + ".png"))
            records.append(dict(
                object_id=object_id, frame_index=s["index"], frame_id=s["frame"],
                scene=s["scene_zh"], bbox_xyxy=box, width=x2-x1, height=y2-y1,
                predicted_material=prediction, score=scores[order[0]],
                top2_margin=scores[order[0]]-scores[order[1]],
                scores=dict(zip(LABELS,scores)), true_material=truth,
                correct=(prediction == truth) if truth else None,
                annotation_status="confirmed_by_user" if truth else "needs_review",
                ground_truth_source="user_correction_of_original_four_rois" if truth else None,
                proposal_source="manual_visual_selection", sku_id=None,
                split="diagnostic_only", depth_used_for_inference=False,
                crop_file=f"crops/{object_id}.png"))
            crops.append(crop)
    for k,s in enumerate(SELECTION):
        subset = records[k*4:(k+1)*4]
        fig = plt.figure(figsize=(12, 6.8))
        gs = fig.add_gridspec(2,4)
        ax = fig.add_subplot(gs[:,:2])
        ax.imshow(frame_images[k]); ax.axis("off")
        ax.set_title(f"Frame {s['index']} | {s['scene']}\nManual boxes", fontsize=12)
        for i,r in enumerate(subset):
            x1,y1,x2,y2 = r["bbox_xyxy"]
            ax.add_patch(patches.Rectangle((x1,y1),x2-x1,y2-y1,fill=False,edgecolor="lime",lw=1.5))
            ax.text(x1,y1-4,str(i+1),color="lime",fontsize=11,weight="bold")
            cx = fig.add_subplot(gs[i//2,2+i%2])
            cx.imshow(crops[k*4+i]); cx.axis("off")
            gt = r["true_material"] or "unconfirmed"
            cx.set_title(f"{r['object_id']}: {r['predicted_material']} {r['score']:.1%}\nTrue material: {gt}",fontsize=10)
        fig.suptitle("Production RGB material probe | predictions are not verified labels",fontsize=13)
        fig.tight_layout()
        fig.savefig(out / "figures" / f"frame_{s['index']:03d}_results.png",dpi=150)
        plt.close(fig)
    fig, axes = plt.subplots(6,4,figsize=(12,15.4))
    for ax,crop,r in zip(axes.flat,crops,records):
        ax.imshow(crop); ax.axis("off")
        ax.set_title(f"{r['object_id']}: {r['predicted_material']} {r['score']:.1%}\nTrue: {r['true_material'] or 'unconfirmed'}",fontsize=10)
    fig.suptitle("24 manually selected ROIs | only the first row has reviewed labels",fontsize=14)
    fig.tight_layout(rect=(0,0,1,.98))
    fig.savefig(out / "figures/all_24_predictions.png",dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(3,2,figsize=(12,14))
    for ax,im,s in zip(axes.flat,frame_images,SELECTION):
        ax.imshow(im); ax.axis("off")
        ax.set_title(f"Frame {s['index']} | {s['scene']}")
        for i,box in enumerate(s["boxes"]):
            x1,y1,x2,y2 = box
            ax.add_patch(patches.Rectangle((x1,y1),x2-x1,y2-y1,fill=False,edgecolor="lime",lw=1.4))
            ax.text(x1,y1-5,str(i+1),color="lime",weight="bold")
    fig.tight_layout()
    fig.savefig(out / "figures/selected_frames.png",dpi=150)
    plt.close(fig)
    truth_records = [r for r in records if r["true_material"]]
    summary = dict(
        reference_experiment_date="2026-09-29", frame_count=6, roi_count=len(records), new_frame_count=5,
        new_unlabeled_roi_count=20, reviewed_count=len(truth_records),
        reviewed_correct=sum(r["correct"] for r in truth_records),
        reviewed_accuracy=sum(r["correct"] for r in truth_records)/len(truth_records),
        overall_accuracy=None, prediction_counts=dict(Counter(r["predicted_material"] for r in records)),
        new_prediction_counts=dict(Counter(r["predicted_material"] for r in records[4:])),
        score_min=min(r["score"] for r in records), score_max=max(r["score"] for r in records),
        score_median=float(np.median([r["score"] for r in records])),
        min_crop_side=min(min(r["width"],r["height"]) for r in records),
        max_crop_side=max(max(r["width"],r["height"]) for r in records),
        cpu_forward_seconds_total=sum(times),
        checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        checkpoint_path=str(checkpoint_path.resolve()),
        python=platform.python_version(),torch=torch.__version__,torchvision=torchvision.__version__,
        device="cpu",threads=8,preprocess=str(preprocess),
        scope="Manual RGB ROI classification only; no training, automatic segmentation or depth fusion",
        sampling="Purposeful scene sampling; repeated SKUs possible; not an independent test set",
        confidence="Uncalibrated softmax score, not probability of being correct")
    (out / "predictions.json").write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    (out / "code/selection.json").write_text(json.dumps(SELECTION,ensure_ascii=False,indent=2),encoding="utf-8")
    csv_rows=[]
    for r in records:
        flat = {k:v for k,v in r.items() if k!="scores"}
        flat["bbox_xyxy"]=json.dumps(flat["bbox_xyxy"])
        flat.update({"score_"+k:v for k,v in r["scores"].items()})
        csv_rows.append(flat)
    write_csv(out/"predictions.csv",csv_rows)
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__ == "__main__":
    main()

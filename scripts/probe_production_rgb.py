"""Classify manually selected RGB ROIs and save crops, scores and a figure."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from PIL import Image

from material_recognition import LABELS
from material_recognition.data import transform
from material_recognition.model import build_model

DEFAULT_FRAME = "2026_08_26_09_19_22_657"
DEFAULT_BOXES = [[615,156,673,201], [709,221,758,282],
                 [490,314,558,367], [810,383,863,451]]


def probe(source, output, checkpoint_path, frame, boxes):
    if not boxes or any(len(b) != 4 or any(type(v) is not int for v in b) for b in boxes):
        raise ValueError("Boxes must be a nonempty list of integer [x1,y1,x2,y2] coordinates")
    image = Image.open(source / f"{frame}_color.jpg").convert("RGB")
    depth = np.array(Image.open(source / f"{frame}_depth.png"))
    if depth.ndim != 2 or depth.dtype != np.uint16 or depth.shape != (image.height, image.width):
        raise ValueError("Depth must be uint16, single-channel and the same size as RGB")
    for x1,y1,x2,y2 in boxes:
        if not (0 <= x1 < x2 <= image.width and 0 <= y1 < y2 <= image.height):
            raise ValueError("Box is empty or outside the original RGB image")
    output = output.resolve()
    source = source.resolve()
    if output == source or source in output.parents:
        raise ValueError("Output must be outside the original data directory")
    torch.set_num_threads(8)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if tuple(checkpoint["labels"]) != LABELS:
        raise ValueError("Checkpoint label order does not match this project")
    model = build_model(pretrained=False).cpu().eval()
    model.load_state_dict(checkpoint["model"])
    preprocess = transform(False)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    rows = max(2, (len(boxes)+1)//2)
    fig = plt.figure(figsize=(12, 3.5*rows))
    gs = fig.add_gridspec(rows, 4)
    scene = fig.add_subplot(gs[:, :2])
    scene.imshow(image); scene.axis("off")
    scene.set_title("Manually selected object ROIs")
    for i,bbox in enumerate(boxes):
        crop = image.crop(bbox)
        with torch.inference_mode():
            scores = model(preprocess(crop).unsqueeze(0)).softmax(1)[0].tolist()
        best = max(range(len(LABELS)), key=scores.__getitem__)
        x1,y1,x2,y2 = bbox
        values = depth[y1:y2,x1:x2]
        records.append(dict(
            object_id=f"probe_{i+1}", frame_id=frame, bbox_xyxy=bbox,
            proposal_source="manual_visual_selection", predicted_material=LABELS[best],
            score=scores[best], scores=dict(zip(LABELS,scores)), true_material=None,
            annotation_status="needs_review", sku_id=None,
            same_pixel_depth_zero_fraction=float((values==0).mean()),
            depth_alignment="unconfirmed", depth_used_for_inference=False,
            device="cpu", model_version=checkpoint_path.parent.name))
        crop.save(output / f"probe_roi_{i+1}.jpg")
        scene.add_patch(patches.Rectangle((x1,y1),x2-x1,y2-y1,fill=False,edgecolor="lime",linewidth=1.5))
        scene.text(x1,y1-5,str(i+1),color="lime",fontsize=12)
        ax = fig.add_subplot(gs[i//2,2+i%2]); ax.imshow(crop); ax.axis("off")
        ax.set_title(f"ROI {i+1}: {LABELS[best]} ({scores[best]:.3f})\nGround truth: unconfirmed",fontsize=10)
    fig.suptitle("Production RGB exploratory predictions only",fontsize=14)
    fig.tight_layout(); fig.savefig(output / "material_probe.jpg",dpi=160); plt.close(fig)
    (output / "material_probe.json").write_text(json.dumps(records,indent=2),encoding="utf-8")
    print(json.dumps(records,indent=2))
    return records


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=root/"data/映界_rgbd数据")
    parser.add_argument("--output", type=Path, default=root/"output/production_probe")
    parser.add_argument("--checkpoint", type=Path, default=root/"runs/week1_resnet18/best.pt")
    parser.add_argument("--frame", default=DEFAULT_FRAME)
    parser.add_argument("--boxes-json", type=Path, help="JSON list of original-image half-open xyxy boxes")
    args = parser.parse_args()
    if args.frame != DEFAULT_FRAME and args.boxes_json is None:
        parser.error("A different frame requires --boxes-json; default boxes belong to the reference frame")
    boxes = json.loads(args.boxes_json.read_text(encoding="utf-8-sig")) if args.boxes_json else DEFAULT_BOXES
    probe(args.source, args.output, args.checkpoint, args.frame, boxes)


if __name__ == "__main__":
    main()

"""Legacy CLI/result adapter. Model implementations live in models/."""
import argparse
import json
import os
from functools import lru_cache
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from perception.pipeline.segmentation import GroundedSegmenter

ENGINE = "sam1"
ROOT = Path(__file__).resolve().parents[2]

@lru_cache(maxsize=4)
def _pipeline(engine, box_thresh, text_thresh):
    if engine == "sam1":
        from perception.models.sam1.backend import Segmenter
        weight = ROOT / "models/sam_b.pt"
    else:
        from perception.models.sam2.backend import Segmenter
        weight = ROOT / "week1_model_selection/sam2.1_t.pt"
    return GroundedSegmenter(sam_weight=str(weight), device="auto", box_threshold=box_thresh,
                             text_threshold=text_thresh, segmentation_factory=Segmenter)

def run_segmentation(image_path, prompt, box_thresh=.25, text_thresh=.20, *, engine=None):
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    prediction = _pipeline(engine or ENGINE, box_thresh, text_thresh).predict(image, prompt)
    return image, [{"instance_id": f"i{i:03d}", "bbox_xyxy": ins.bbox_xyxy,
                    "det_score": ins.det_score, "label": ins.label, "mask": ins.mask}
                   for i, ins in enumerate(prediction.instances)]

# ---------- 可视化 ----------
def draw_overlay(image, instances):
    img = np.array(image).copy()
    for ins in instances:
        mask = ins["mask"]
        color = np.array([0, 255, 0], dtype=np.uint8)
        img[mask] = (0.5 * img[mask] + 0.5 * color).astype(np.uint8)
        pil = Image.fromarray(img)
        draw = ImageDraw.Draw(pil)
        x1, y1, x2, y2 = ins["bbox_xyxy"]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, max(0, y1 - 14)),
                  f"{ins['label']} {ins['det_score']:.2f}", fill="red")
        img = np.array(pil)
    return Image.fromarray(img)


# ---------- CLI ----------
def main(engine="sam1"):
    global ENGINE
    ENGINE = engine
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="RGB 图路径")
    ap.add_argument("--prompt", required=True,
                    help='提示词，每个类别后加句号，例如 "bottle. box. bag. can."')
    ap.add_argument("--box-thresh", type=float, default=0.25, help="检测阈值，越小检出越多、误检也越多")
    ap.add_argument("--out-dir", default="output/seg_demo", help="输出目录")
    args = ap.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    masks_dir = os.path.join(out_dir, "masks")
    os.makedirs(masks_dir, exist_ok=True)

    print("=" * 60)
    image, instances = run_segmentation(args.image, args.prompt, args.box_thresh)

    overlay = draw_overlay(image, instances)
    overlay_path = os.path.join(out_dir, "overlay.jpg")
    overlay.save(overlay_path, quality=90)

    json_instances = []
    for ins in instances:
        mask_path = os.path.join(masks_dir, f"{ins['instance_id']}.png")
        Image.fromarray((ins["mask"] * 255).astype(np.uint8)).save(mask_path)
        json_instances.append({
            "instance_id": ins["instance_id"],
            "bbox_xyxy": ins["bbox_xyxy"],
            "det_score": ins["det_score"],
            "label": ins["label"],
            "mask_ref": os.path.relpath(mask_path, out_dir).replace("\\", "/"),
        })

    result = {
        "image": args.image,
        "prompt": args.prompt,
        "box_threshold": args.box_thresh,
        "device": _pipeline(ENGINE, args.box_thresh, .20).device,
        "num_instances": len(instances),
        "instances": json_instances,
    }
    json_path = os.path.join(out_dir, "result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print(f"完成！共检出 {len(instances)} 个物体。")
    print(f"叠加图: {os.path.abspath(overlay_path)}")
    print(f"结果 JSON: {os.path.abspath(json_path)}")
    print(f"Mask 目录: {os.path.abspath(masks_dir)}")

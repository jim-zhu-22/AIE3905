"""
Mask 质量对比：同一批图、同一个检测器（Grounding DINO），
分别用 SAM 1 和 SAM 2.1 分割，把两种 mask 叠在同一张图上对比。

颜色约定：
    红色 = 只有 SAM 1 画到的区域
    蓝色 = 只有 SAM 2.1 画到的区域
    绿色 = 两个模型都画到的区域（一致）
旁边标注每个框两模型 mask 的 IoU（越高表示轮廓越一致）和覆盖率。

用法（在项目根目录 D:/AIE3905 下）：
    python scripts/compare_masks.py
输出到 output/mask_compare/
"""
import json
import os
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_segmentation import run_segmentation as seg_sam1
from run_segmentation_sam2 import run_segmentation as seg_sam2

DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "data", "Chad_frames_002-005_20261013")
ANN_FILE = os.path.join(DATA_ROOT, "annotations", "instances_bbox_draft.coco.json")
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "output", "mask_compare")

PROMPT = "box. bag. bottle. carton. packet."


def mask_iou(m1, m2):
    inter = int((m1 & m2).sum())
    union = int((m1 | m2).sum())
    return inter / union if union > 0 else 0.0


def draw_pair(image, ins1, ins2, idx, font_scale_ok=True):
    """叠加两种 mask 到原图，返回 (图, iou, 覆盖率1, 覆盖率2)"""
    img = np.array(image).copy()
    m1 = ins1["mask"]
    m2 = ins2["mask"]

    only1 = m1 & ~m2
    only2 = m2 & ~m1
    both = m1 & m2
    img[only1] = (0.5 * img[only1] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)   # 红
    img[only2] = (0.5 * img[only2] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)   # 蓝
    img[both] = (0.5 * img[both] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)     # 绿

    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    x1, y1, x2, y2 = ins1["bbox_xyxy"]
    draw.rectangle([x1, y1, x2, y2], outline="white", width=1)

    iou = mask_iou(m1, m2)
    area = max(1, (x2 - x1) * (y2 - y1))
    cov1 = int(m1.sum()) / area
    cov2 = int(m2.sum()) / area
    draw.text((x1, max(0, y1 - 13)),
              f"#{idx} IoU={iou:.2f} cov1={cov1:.2f} cov2={cov2:.2f}",
              fill="white")
    return pil, iou, cov1, cov2


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(ANN_FILE, "r", encoding="utf-8") as f:
        coco = json.load(f)

    all_iou = []
    for im in coco["images"]:
        frame_id = im["frame_id"]
        img_path = os.path.join(DATA_ROOT, im["file_name"])
        print(f"\n===== {frame_id} =====")

        t0 = time.time()
        image1, inst1 = seg_sam1(img_path, PROMPT)
        t1 = time.time()
        image2, inst2 = seg_sam2(img_path, PROMPT)
        t2 = time.time()
        print(f"SAM1 推理 {t1-t0:.1f}s | SAM2.1 推理 {t2-t1:.1f}s | 检出 {len(inst1)}/{len(inst2)}")

        n = min(len(inst1), len(inst2))
        frame_ious = []
        for i in range(n):
            pil, iou, cov1, cov2 = draw_pair(image1, inst1[i], inst2[i], i)
            frame_ious.append(iou)
            all_iou.append(iou)
            if i < 6:  # 每帧前 6 个框出特写
                x1, y1, x2, y2 = inst1[i]["bbox_xyxy"]
                pad = 40
                crop = pil.crop((max(0, x1-pad), max(0, y1-pad),
                                 min(pil.width, x2+pad), min(pil.height, y2+pad)))
                crop = crop.resize((crop.width * 3, crop.height * 3), Image.LANCZOS)
                crop.save(os.path.join(OUT_DIR, f"{frame_id}_box{i:02d}.jpg"))

        # 整帧图（只叠加框提示文字少的版本：重新画不写字）
        pil_full = image1.copy()
        arr = np.array(pil_full)
        for i in range(n):
            m1, m2 = inst1[i]["mask"], inst2[i]["mask"]
            arr[m1 & ~m2] = (0.5 * arr[m1 & ~m2] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
            arr[m2 & ~m1] = (0.5 * arr[m2 & ~m1] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
            arr[m1 & m2] = (0.5 * arr[m1 & m2] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
        pil_full = Image.fromarray(arr)
        draw = ImageDraw.Draw(pil_full)
        for i in range(n):
            x1, y1, x2, y2 = inst1[i]["bbox_xyxy"]
            draw.rectangle([x1, y1, x2, y2], outline="white", width=1)
        full_path = os.path.join(OUT_DIR, f"{frame_id}_full.jpg")
        pil_full.save(full_path, quality=90)

        avg_iou = sum(frame_ious) / len(frame_ious) if frame_ious else 0
        print(f"本帧 {n} 个框 mask 平均 IoU: {avg_iou:.3f}（1.0=完全一致，越低差异越大）")
        print(f"整帧图: {full_path}")

    print("\n================ 汇总 ================")
    print(f"共对比 {len(all_iou)} 个框，两模型 mask 平均 IoU: {sum(all_iou)/len(all_iou):.3f}")
    print("颜色说明: 红=SAM1独有 | 蓝=SAM2.1独有 | 绿=一致")
    print(f"对比图目录: {os.path.abspath(OUT_DIR)}")
    print("提示：IoU 很高说明轮廓几乎一致；红/蓝面积大说明某模型明显多画/少画了区域。")


if __name__ == "__main__":
    main()

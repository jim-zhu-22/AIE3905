r"""
批量对照脚本：整图基线 vs ROI 限定 + 后处理，四帧一次跑完，输出对比表。

为什么单独写一个：
    直接调 CLI 每帧要重新加载一次 Grounding DINO + SAM（约 10s/次），
    8 次就白等一分多钟。这里在同一个进程里复用模型，只加载一次。

用法（项目根目录 D:\AIE3905 下）：
    :: 默认跑 data\Chad_frames_002-005_20261013 下全部帧
    python scripts\run_batch_compare.py

    :: 自定义 ROI 和输出目录
    python scripts\run_batch_compare.py --roi "450,0,900,690" --out-root output\compare_roi

产物：
    <out-root>\summary.md      对照表（可直接贴进汇报）
    <out-root>\summary.csv     同上，Excel 可打开
    <out-root>\<帧名>\baseline\   整图基线：segmentation.json / overlay.jpg / masks
    <out-root>\<帧名>\roi\        ROI 限定：同上
"""
import argparse
import csv
import glob
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_segmentation_sam2 as seg

# 两套配置：基线=什么都不加（等价旧版）；roi=组长反馈后的收紧版
PRESETS = {
    # 整图 + 完全不过滤（等价旧版，作为基线）
    "baseline": dict(roi=None, nms_iou=0.0, min_mask_area=0,
                     max_area_ratio=1.0, max_aspect=0.0, split_touching=False),
    # 整图推理 + ROI 中心点过滤（默认推荐，召回无损）
    "roi_filter": dict(roi=None, nms_iou=0.5, min_mask_area=100,
                       max_area_ratio=0.35, max_aspect=8.0, split_touching=True,
                       split_min_area=2000, split_min_seed=30, roi_crop=False),
    # 裁剪 ROI 后再推理（速度优先，贴边小目标召回会掉）
    "roi_crop": dict(roi=None, nms_iou=0.5, min_mask_area=100,
                     max_area_ratio=0.35, max_aspect=8.0, split_touching=True,
                     split_min_area=2000, split_min_seed=30, roi_crop=True),
}


def apply_preset(name, roi):
    cfg = dict(PRESETS[name])
    if name != "baseline":
        cfg["roi"] = roi
    for k, v in cfg.items():
        seg.CFG[k] = v


def count_outside(instances, roi):
    """按 bbox 中心点判断是否落在 ROI 外（框跨边界不算外，更贴近人工判断）。"""
    if not roi:
        return 0, []
    out = []
    for ins in instances:
        x1, y1, x2, y2 = ins["bbox_xyxy"]
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        if not (roi[0] <= cx <= roi[2] and roi[1] <= cy <= roi[3]):
            out.append(ins["instance_id"])
    return len(out), out


def depth_path_for(image_path, depth_dir):
    r"""xxx_color.jpg -> depth_dir\xxx_depth.png"""
    base = os.path.basename(image_path).replace("_color", "_depth")
    base = os.path.splitext(base)[0] + ".png"
    p = os.path.join(depth_dir, base)
    return p if os.path.exists(p) else None


def run_one(image_path, preset_name, roi, out_dir, prompt, depth_dir, override=None):
    frame = os.path.splitext(os.path.basename(image_path))[0]
    apply_preset(preset_name, roi)
    if override:
        for k, v in override.items():
            seg.CFG[k] = v
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    img, instances, filtered = seg.run_segmentation(image_path, prompt)
    elapsed = time.time() - t0

    overlay = seg.draw_overlay(img, instances, filtered, roi, grid=False)
    overlay.save(os.path.join(out_dir, "overlay.jpg"), quality=90)

    depth_info, warns = seg.build_depth_info(depth_path_for(image_path, depth_dir), out_dir)
    args = type("A", (), {"run_id": f"batch_{preset_name}", "frame_id": frame,
                          "prompt": prompt, "preprocess": None})()
    seg.write_outputs(out_dir, img, instances, args, elapsed, depth_info, warns, filtered)

    n_out, out_ids = count_outside(instances, roi)
    n_split = sum(1 for i in instances if i.get("split_from"))
    return {
        "frame": frame,
        "preset": preset_name,
        "instances": len(instances),
        "filtered": len(filtered),
        "outside_roi": n_out,
        "outside_ids": ",".join(out_ids) if out_ids else "-",
        "split_instances": n_split,
        "seconds": round(elapsed, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir",
                    default=os.path.join("data", "Chad_frames_002-005_20261013", "images"))
    ap.add_argument("--depth-dir",
                    default=os.path.join("data", "Chad_frames_002-005_20261013", "depth"))
    ap.add_argument("--out-root", default=os.path.join("output", "compare_roi"))
    ap.add_argument("--roi", default="450,0,900,690", help="x1,y1,x2,y2；传 none 关闭 ROI")
    ap.add_argument("--prompt", default="box. bag. bottle. carton. packet.")
    ap.add_argument("--box-thresh", type=float, default=0.25)
    ap.add_argument("--presets", default="baseline,roi_filter",
                    help="要跑的配置，逗号分隔：baseline / roi_filter / roi_crop")
    ap.add_argument("--roi-margin", type=int, default=60,
                    help="ROI 裁剪外扩像素，防贴边物体被切断，默认 60")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 帧，0=全部")
    ap.add_argument("--stride", type=int, default=1, help="每隔 N 帧取一帧，1=不间隔")
    ap.add_argument("--pattern", default="*color*", help="图片名匹配模式，默认 *color*；新数据命名不同时传 *")
    ap.add_argument("--max-area-ratio", type=float, default=0.0,
                    help="覆盖面积占比上限，0=用预设值（roi_filter 默认 0.35）")
    args = ap.parse_args()

    roi = None
    if args.roi and args.roi.lower() != "none":
        roi = tuple(int(float(v)) for v in args.roi.replace("，", ",").split(","))

    seg.CFG["roi_margin"] = args.roi_margin
    images = sorted(glob.glob(os.path.join(args.images_dir, args.pattern + ".jpg"))
                    + glob.glob(os.path.join(args.images_dir, args.pattern + ".png")))
    if not images:
        print(f"[error] 在 {args.images_dir} 下没找到 {args.pattern}.jpg 图片，换 --pattern 试试")
        sys.exit(1)
    if args.stride > 1:
        images = images[::args.stride]
    if args.limit > 0:
        images = images[:args.limit]
    print(f"[info] 共 {len(images)} 帧，ROI={roi}，输出到 {args.out_root}")

    presets = [p.strip() for p in args.presets.split(",") if p.strip()]
    bad = [p for p in presets if p not in PRESETS]
    if bad:
        print(f"[error] 未知配置 {bad}，可选：{list(PRESETS)}")
        sys.exit(1)
    rows = []
    override = {}
    if args.max_area_ratio > 0:
        override["max_area_ratio"] = args.max_area_ratio
    for idx, img_path in enumerate(images, 1):
        frame = os.path.splitext(os.path.basename(img_path))[0]
        print(f"\n{'=' * 60}\n[{idx}/{len(images)}] {frame}")
        for preset in presets:
            sub = os.path.join(args.out_root, frame, preset)
            r = run_one(img_path, preset, roi, sub, args.prompt, args.depth_dir, override)
            rows.append(r)
            print(f"  {preset:9s} 实例={r['instances']:3d}  过滤={r['filtered']:2d}  "
                  f"ROI外={r['outside_roi']:2d}  分离={r['split_instances']:2d}  {r['seconds']}s")

    # 汇总表
    os.makedirs(args.out_root, exist_ok=True)
    md = ["# 整图基线 vs ROI 限定 · 对照汇总", "",
          f"- ROI: {roi if roi else '整图'}",
          f"- prompt: {args.prompt}",
          f"- box_thresh: {args.box_thresh}",
          f"- ROI 裁剪外扩: {args.roi_margin}px（归属判定按原 ROI 中心点）",
          "",
          "| 帧 | 配置 | 实例数 | 被过滤 | ROI 外 | 分离实例 | 耗时(s) |",
          "|---|---|---|---|---|---|---|"]
    by_frame = {}
    for r in rows:
        by_frame.setdefault(r["frame"], {})[r["preset"]] = r
        md.append(f"| {r['frame']} | {r['preset']} | {r['instances']} | {r['filtered']} "
                  f"| {r['outside_roi']} | {r['split_instances']} | {r['seconds']} |")

    md += ["", "## 合计", ""]
    tot = {}
    for preset in presets:
        tot[preset] = {
            "inst": sum(v.get(preset, {}).get("instances", 0) for v in by_frame.values()),
            "out": sum(v.get(preset, {}).get("outside_roi", 0) for v in by_frame.values()),
            "sec": sum(v.get(preset, {}).get("seconds", 0) for v in by_frame.values()),
            "split": sum(v.get(preset, {}).get("split_instances", 0) for v in by_frame.values()),
        }
        md.append(f"- {preset}: 实例 {tot[preset]['inst']}，ROI 外 {tot[preset]['out']}，"
                  f"分离 {tot[preset]['split']}，总耗时 {tot[preset]['sec']:.1f}s")

    md_path = os.path.join(args.out_root, "summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    csv_path = os.path.join(args.out_root, "summary.csv")
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["frame", "preset", "instances", "filtered",
                                          "outside_roi", "outside_ids",
                                          "split_instances", "seconds"])
        w.writeheader()
        w.writerows(rows)

    print(f"\n{'=' * 60}\n汇总: {os.path.abspath(md_path)}\nCSV : {os.path.abspath(csv_path)}")
    for preset in presets:
        t = tot[preset]
        print(f"{preset}: 实例 {t['inst']}，ROI 外 {t['out']}，总耗时 {t['sec']:.1f}s")


if __name__ == "__main__":
    main()

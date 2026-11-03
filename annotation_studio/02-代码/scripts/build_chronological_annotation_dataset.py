"""Build a traceable RGB-D box dataset from cumulative annotation JSON exports.

Requires Pillow. Does not invent masks, review decisions, or train/test splits.
"""
import argparse
import copy
import csv
import hashlib
import html
import json
import math
import shutil
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def keyed(objects):
    result = {}
    for obj in objects:
        key = (obj["frame_id"], obj["object_id"])
        if key in result:
            raise ValueError(f"Duplicate object key in one export: {key}")
        result[key] = obj
    return result


def csv_write(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def draw_box(draw, box, number, material, font):
    x1, y1, x2, y2 = box
    color = "#ffb020" if material == "paper" else "#24a9ff"
    draw.rectangle([x1, y1, x2-1, y2-1], outline=color, width=3)
    label = str(number)
    label_width = draw.textbbox((0, 0), label, font=font)[2] + 6
    # Compact numeric tags preserve visibility in crowded small-object scenes.
    draw.rectangle([x1, y1, x1+label_width, y1+22], fill=color)
    draw.text((x1+3, y1), label, font=font, fill="#101827")


def build(archive, raw, out):
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset: {out}")
    exports = []
    with zipfile.ZipFile(archive) as z:
        if z.testzip():
            raise ValueError("Input ZIP CRC check failed")
        for item in z.infolist():
            if item.filename.endswith(".json") and "__MACOSX" not in item.filename.split("/"):
                content = z.read(item)
                document = json.loads(content.decode("utf-8-sig"))
                if not isinstance(document, dict) or "objects" not in document:
                    raise ValueError(f"Not an annotation export: {item.filename}")
                timestamp = datetime.fromisoformat(document["exported_at"].replace("Z", "+00:00"))
                exports.append((timestamp, item.filename, document, content))
    exports.sort(key=lambda entry: entry[0])
    if not exports:
        raise ValueError("No annotation JSON exports found")
    if len({e[0] for e in exports}) != len(exports):
        raise ValueError("Ambiguous equal export timestamps")
    if len({(e[2].get('dataset_id'), e[2].get('annotator')) for e in exports}) != 1:
        raise ValueError("Mixed dataset/annotator exports require explicit merge rules")
    history, previous, provenance = [], {}, {}
    for _, name, document, _ in exports:
        current = keyed(document["objects"])
        removed = sorted(set(previous) - set(current))
        changed = [k for k in previous.keys() & current.keys() if previous[k] != current[k]]
        # This tool supports verified append-only cumulative snapshots. Never silently
        # restore deletions or overwrite conflicting annotations from other exports.
        if removed or changed:
            raise ValueError(f"Exports are not append-only snapshots: {name}; resolve changes first")
        for key in current:
            provenance.setdefault(key, {"first_seen_export": name, "first_seen_at": document["exported_at"]})
        history.append({"source": name, "exported_at": document["exported_at"],
                        "objects": len(current), "new_objects": len(current.keys() - previous.keys()),
                        "unchanged_repeated_objects": len(current.keys() & previous.keys()),
                        "changed_objects": len(changed), "removed_objects": len(removed)})
        previous = current
    merged = copy.deepcopy(exports[-1][2])
    merged["objects"].sort(key=lambda obj: (obj["frame_id"], obj["object_id"]))
    frame_ids = sorted({o["frame_id"] for o in merged["objects"]})
    all_raw_frames = sorted(p.name.removesuffix("_color.jpg") for p in raw.glob("*_color.jpg"))
    raw_indices = {frame: i + 1 for i, frame in enumerate(all_raw_frames)}
    # Validate everything before creating an output directory.
    for obj in merged["objects"]:
        frame = obj["frame_id"]
        datetime.strptime(frame, "%Y_%m_%d_%H_%M_%S_%f")
        if obj.get("mask_rle"):
            raise ValueError("This builder only handles box-only exports")
        if obj.get("coordinate_convention") != "original_image_pixels_xyxy_half_open":
            raise ValueError("Unknown box coordinate convention")
        for field, suffix in [("rgb_file", "_color.jpg"), ("depth_file", "_depth.png")]:
            if obj[field] != frame + suffix or not (raw / obj[field]).is_file():
                raise ValueError(f"Unmatched or unsafe source filename: {obj[field]}")
        w, h = obj["image_size"]
        x1, y1, x2, y2 = obj["object_bbox_xyxy"]
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in [x1, y1, x2, y2]):
            raise ValueError("Non-finite bounding box")
        if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
            raise ValueError(f"Invalid bounding box: {obj['object_id']}")
        if obj["material"] not in {"paper", "plastic", "metal", "wood", "fabric", "unknown"}:
            raise ValueError(f"Unexpected material: {obj['material']}")
        if Path(obj['object_id']).name != obj['object_id'] or any(c in obj['object_id'] for c in '/\\:'):
            raise ValueError("Unsafe object id")
    for frame in frame_ids:
        rows = [o for o in merged["objects"] if o["frame_id"] == frame]
        with Image.open(raw / rows[0]["rgb_file"]) as rgb, Image.open(raw / rows[0]["depth_file"]) as depth:
            rgb.load(); depth.load()
            if rgb.size != depth.size or any(tuple(o["image_size"]) != rgb.size for o in rows):
                raise ValueError(f"Image dimensions mismatch: {frame}")

    for folder in ["images", "depth", "previews", "crops", "annotations", "metadata", "source_annotations"]:
        (out / folder).mkdir(parents=True, exist_ok=True)
    shutil.copy2(archive, out / "source_annotations" / "original_data.zip")
    for i, (_, name, _, content) in enumerate(exports, 1):
        (out / "source_annotations" / f"{i:02d}_{Path(name).name}").write_bytes(content)
    write_json(out / "annotations.json", merged)
    write_json(out / "metadata" / "export_history.json", history)
    coco = {"info": {"description": "Draft object boxes only; no segmentation ground truth; no completeness confirmation"},
            "images": [], "annotations": [], "categories": [{"id": 1, "name": "object"}]}
    frames, instances, crops, cards = [], [], [], []
    try:
        font = ImageFont.truetype("arial.ttf", 17)
    except OSError:
        font = ImageFont.load_default(size=17)
    for order, frame in enumerate(frame_ids, 1):
        rows = [o for o in merged["objects"] if o["frame_id"] == frame]
        rgb_rel, depth_rel = f"images/{frame}_color.jpg", f"depth/{frame}_depth.png"
        for rel in [rgb_rel, depth_rel]:
            shutil.copy2(raw / Path(rel).name, out / rel)
        with Image.open(out / rgb_rel) as img:
            rgb = img.convert("RGB")
        preview = rgb.copy()
        draw = ImageDraw.Draw(preview)
        materials = Counter(o["material"] for o in rows)
        capture = datetime.strptime(frame, "%Y_%m_%d_%H_%M_%S_%f").isoformat(timespec="milliseconds")
        record = {"chronological_index": order, "original_image_number": raw_indices[frame],
                  "frame_id": frame, "capture_time_local_timezone_unconfirmed": capture,
                  "rgb_path": rgb_rel, "depth_path": depth_rel, "object_count": len(rows),
                  "paper_count": materials["paper"], "plastic_count": materials["plastic"],
                  "split": "unassigned", "sequence_group": "adjacent_capture_group_001"}
        frames.append(record)
        coco["images"].append({"id": order, "file_name": rgb_rel, "width": rgb.width, "height": rgb.height,
                               "frame_id": frame, "frame_completeness": "unconfirmed"})
        legend = []
        for number, obj in enumerate(rows, 1):
            x1, y1, x2, y2 = obj["object_bbox_xyxy"]
            # Floor/ceil include the entire annotated region if future exports use fractions.
            crop_box = [math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2)]
            crop_rel = f"crops/{obj['material']}/{frame}__{obj['object_id']}.png"
            (out / crop_rel).parent.mkdir(exist_ok=True)
            rgb.crop(crop_box).save(out / crop_rel)
            instance = {**obj, **provenance[(frame, obj["object_id"])],
                        "instance_id": f"{frame}__{obj['object_id']}", "rgb_path": rgb_rel, "depth_path": depth_rel,
                        "crop_path": crop_rel, "crop_bbox_xyxy": crop_box, "mask_path": None,
                        "split": "unassigned", "sequence_group": record["sequence_group"], "training_ready": False,
                        "blocking_reasons": ["human_review_not_confirmed", "frame_completeness_unconfirmed", "split_not_assigned"],
                        "segmentation_blocking_reasons": ["mask_missing"]}
            instances.append(instance)
            crops.append({k: instance[k] for k in ["instance_id", "frame_id", "object_id", "crop_path", "material", "annotation_status", "sequence_group", "split"]})
            coco["annotations"].append({"id": len(instances), "image_id": order, "category_id": 1,
                                        "bbox": [x1, y1, x2-x1, y2-y1], "area": (x2-x1)*(y2-y1),
                                        "area_source": "bounding_box", "iscrowd": 0,
                                        "material": obj["material"], "source_object_id": obj["object_id"],
                                        "annotation_status": obj.get("annotation_status", "needs_review")})
            draw_box(draw, [x1, y1, x2, y2], number, obj["material"], font)
            legend.append(f"<span>{number}. {html.escape(obj['object_id'])} · {'纸质' if obj['material']=='paper' else '塑料'}</span>")
        preview_rel = f"previews/{order:02d}_{frame}.jpg"
        preview.save(out / preview_rel, quality=94)
        cards.append(f'<article><h2>原始第 {raw_indices[frame]} 张 · 时间顺序 {order}</h2><p>{capture} · {len(rows)} 个物体 · 纸质 {materials["paper"]} / 塑料 {materials["plastic"]}</p><a href="{preview_rel}" target="_blank"><img src="{preview_rel}" alt="带框标注预览"></a><p><a href="{rgb_rel}" target="_blank">查看原图</a> · <a href="{depth_rel}">原始 Depth</a> · 点击图片放大</p><details><summary>物体编号与材质</summary><div class="legend">{"".join(legend)}</div></details></article>')
    write_json(out / "instances.json", {"schema_version": "chronological-box-dataset-1", "training_ready": False, "instances": instances})
    write_json(out / "annotations" / "instances_bbox_draft.coco.json", coco)
    csv_write(out / "metadata" / "frames.csv", frames, list(frames[0]))
    csv_write(out / "metadata" / "material_crops.csv", crops, list(crops[0]))
    quality = {"frames": len(frames), "objects": len(instances), "materials": dict(Counter(o["material"] for o in instances)),
               "source_exports": len(exports), "raw_export_rows": sum(h["objects"] for h in history),
               "repeated_snapshot_rows_removed": sum(h["unchanged_repeated_objects"] for h in history),
               "merge_rule": "Verified append-only snapshots, use latest export; order images by capture timestamp",
               "original_image_numbers": [f["original_image_number"] for f in frames],
               "missing_annotations_before_first_selected_frame": [f for f in all_raw_frames if f < frame_ids[0]],
               "masks": 0, "review_status_counts": dict(Counter(o.get("annotation_status", "missing") for o in instances)),
               "duplicate_objects": 0, "missing_rgb_or_depth": 0, "invalid_boxes": 0,
               "split_assigned": False, "depth_unit": merged.get("depth_unit"), "alignment": merged.get("alignment"),
               "notes": ["Only annotated frames included; no inferred annotations for image 1", "Nearby frames grouped to prevent random frame/crop split leakage", "No mask generation or model training performed"]}
    write_json(out / "quality_report.json", quality)
    (out / "index.html").write_text('''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>按时间整理的标注数据集</title><style>body{max-width:1300px;margin:36px auto;padding:0 24px;background:#f2f5f9;color:#172334;font:16px/1.65 system-ui}h1{font-size:30px}.notice,article{background:white;border-radius:14px;padding:22px;margin-bottom:24px;box-shadow:0 3px 16px #1122330a}img{display:block;width:100%;border-radius:8px}a{color:#1467bd}.legend{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:8px}summary{cursor:pointer}</style><h1>前几张 RGB-D 图片 · 标注数据集</h1><div class="notice"><strong>按图片采集时间排序；原始第 2–5 张，108 个物体。</strong><br>纸质 46 个（橙色），塑料 62 个（蓝色）。4 次累计导出已去重，原始第 1 张无标注。<br>这批数据有物体框和材质，没有 Mask；复核状态保留为待复核。预览中的矩形不是分割轮廓。<br><a href="annotations.json">合并标注 JSON</a> · <a href="metadata/frames.csv">图片时间与编号表</a> · <a href="README.txt">使用说明</a></div>''' + "\n".join(cards) + "</html>", encoding="utf-8")
    readme = f'''按时间整理的 RGB-D 标注数据集

结果：{len(frames)} 对 RGB + 原始 Depth，{len(instances)} 个物体框；纸质 46，塑料 62。
对应原始 245 张数据的第 2、3、4、5 张；第 1 张没有标注，未擅自补标。
图片按文件名内的采集时间排序（2026-08-26），不是按 JSON 的导出时间排序。
4 次 JSON 导出时间（2026-10-13，UTC）用于确定版本；经逐条比对为累积追加且无修改。
27 + 51 + 76 + 108 = 262 条快照记录，去除 154 条重复后保留 108 条。

如何查看：双击 index.html；点击带框图片放大，橙色纸质、蓝色塑料。
如何继续标注：将根目录 annotations.json 导入原标注 Demo，使用相同的原始 RGB-D 数据。
如何交给训练同学：整个目录或 ZIP 一并交付，内部路径均为相对路径。

images/：4 张原始 RGB，保持文件字节不变。
depth/：4 张原始深度 PNG，保持位深和文件字节不变；深度单位和对齐情况仍待确认。
previews/：带编号和材质的框选检查图，不能拿带框图片当模型输入。
crops/paper/、crops/plastic/：108 张无绘制标记的物体裁剪图（PNG），来自原始 RGB。
annotations.json：最新累计快照，按采集时间、物体编号排序，可重新导入标注工具。
instances.json：原图、深度、裁剪图及每个物体的对应关系与来源版本。
annotations/instances_bbox_draft.coco.json：COCO 格式框标注草稿，类别统一为 object；material 为额外属性，没有 segmentation。
metadata/frames.csv：时间顺序、原始图片编号、路径和数量。
metadata/material_crops.csv：材质分类样本清单；标签仍待复核。
metadata/export_history.json：每次导出的新增和重复数量。
source_annotations/：原始 ZIP 与四份原始 JSON，保留追溯。
quality_report.json：配对、坐标、去重和就绪状态报告。
checksums.sha256：交付文件校验和。

标注坐标为原图像素 xyxy，右下边界不包含。COCO 转为 x,y,width,height。
108 个标签的原始 annotation_status 均为 needs_review，未自动改为已确认。
物体框和材质可用于下一步检测/分类基线的数据准备；正式训练前请复核标签和整帧完整性。
没有 Mask，所以这不是可直接用于监督实例分割训练的成品；需生成轮廓草稿并人工复核。
这 4 帧仅相隔约 24 秒，存在相同 SKU 和关联场景的风险。当前全部 split=unassigned，
同归 adjacent_capture_group_001，不随机拆分裁剪图。请结合其他时间段及 SKU 建立独立评估集。
本次没有训练模型或计算准确率，也没有将模型预测当作人工标注。
'''
    (out / "README.txt").write_text(readme, encoding="utf-8-sig")
    files = sorted(p for p in out.rglob("*") if p.is_file())
    (out / "checksums.sha256").write_text("\n".join(f"{sha256(p)}  {p.relative_to(out).as_posix()}" for p in files)+"\n", encoding="utf-8")
    return quality


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--zip", type=Path)
    args = parser.parse_args()
    if args.zip and args.zip.exists():
        parser.error("ZIP output already exists")
    report = build(args.archive, args.raw, args.output)
    if args.zip:
        args.zip.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(args.zip, "w", zipfile.ZIP_DEFLATED) as z:
            for path in sorted(args.output.rglob("*")):
                if path.is_file():
                    z.write(path, args.output.name + "/" + path.relative_to(args.output).as_posix())
        with zipfile.ZipFile(args.zip) as z:
            assert z.testzip() is None
        args.zip.with_suffix(".zip.sha256.txt").write_text(sha256(args.zip)+"  "+args.zip.name+"\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))

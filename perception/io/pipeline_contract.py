"""pipeline-1.0 interchange; external predictions are data, never executable code."""
from copy import deepcopy
from pathlib import Path
import json
import re
import shutil
import uuid

import numpy as np
from PIL import Image

from perception.io.manifest import write_json
from perception.pipeline.contracts import Instance, Prediction
from perception.runtime import sha256


def mask_box(mask):
    ys, xs = np.asarray(mask, dtype=bool).nonzero()
    if not len(xs):
        raise ValueError("pipeline-1.0 requires nonempty instance masks")
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


class SavedSegmenter:
    """Validate a saved segmentation and resume without loading DINO or SAM."""
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.data = json.loads(self.path.read_text(encoding="utf-8-sig"))
        data = self.data
        if data.get("schema_version") != "pipeline-1.0" or data.get("stage") != "segmentation":
            raise ValueError("Expected pipeline-1.0 segmentation")
        for key in ("run_id", "frame_id", "prompt", "model_version"):
            if not isinstance(data.get(key), str) or not data[key].strip():
                raise ValueError(f"Missing segmentation {key}")
        size = data.get("image_size")
        if not isinstance(size, list) or len(size) != 2 or any(type(v) is not int or v <= 0 for v in size):
            raise ValueError("Invalid image_size")
        rows = data.get("instances")
        if not isinstance(rows, list) or type(data.get("instance_count")) is not int or data["instance_count"] != len(rows):
            raise ValueError("instance_count mismatch")
        if data.get("status") != ("ok" if rows else "no_match"):
            raise ValueError("Invalid segmentation status/count combination")
        self.instances = []
        seen = set()
        for item in rows:
            ident = item.get("instance_id", "")
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,119}", ident) or ident.casefold() in seen:
                raise ValueError("Unsafe or duplicate instance_id")
            if ident.upper() in {"CON", "PRN", "AUX", "NUL", *[f"COM{i}" for i in range(1,10)], *[f"LPT{i}" for i in range(1,10)]}:
                raise ValueError("Reserved instance_id")
            seen.add(ident.casefold())
            reference = item.get("mask_path")
            if not isinstance(reference, str) or not reference or Path(reference).is_absolute():
                raise ValueError("mask_path must be bundle-relative")
            path = (self.path.parent / reference).resolve()
            if not path.is_relative_to(self.path.parent):
                raise ValueError("mask_path escapes bundle")
            with Image.open(path) as source:
                pixels = np.array(source)
            if pixels.ndim != 2 or list(pixels.shape[::-1]) != size:
                raise ValueError("Mask dimensions must match RGB")
            if not np.isin(pixels, [0, 255]).all():
                raise ValueError("Mask must contain only binary 0/255 pixels")
            mask = pixels > 0
            box = item.get("bbox_xyxy")
            if not isinstance(box, list) or len(box) != 4 or any(type(v) is not int for v in box) or box != mask_box(mask):
                raise ValueError("bbox_xyxy must equal integer half-open mask bounds")
            if not isinstance(item.get("object_label"), str) or not item["object_label"]:
                raise ValueError("Missing object_label")
            for key in ("det_score", "seg_score"):
                value = item.get(key)
                if value is None and key == "seg_score":
                    continue
                if type(value) not in (int, float) or not np.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError(f"Invalid {key}")
            self.instances.append(Instance(box, item["det_score"], item["object_label"], mask,
                                           instance_id=ident, seg_score=item.get("seg_score")))

    def predict(self, image, prompt, mode):
        if list(image.size) != self.data["image_size"]:
            raise ValueError("RGB dimensions do not match segmentation")
        data = deepcopy(self.data)
        data["import_provenance"] = {"segmentation_path": str(self.path), "sha256": sha256(self.path),
                                     "rgb_binding": "caller_supplied_same_frame_not_hash_verified"}
        depth = data.get("depth", {})
        if depth.get("depth_path"):
            depth["source_depth_path"] = depth["depth_path"]
            depth["depth_path"] = str((self.path.parent / depth["depth_path"]).resolve())
        return Prediction(image, self.instances, "imported", data["prompt"], {}, len(self.instances),
                          {"model_version": data["model_version"], "execution": "imported_not_rerun"}, data)


def export_contract(prediction, output, rgb_path, depth_path, config, legacy, rois, *, frame_id=None, run_id=None):
    """Write the five linked stages, keeping legacy perception.json separately."""
    output = Path(output)
    source = prediction.source_contract or {}
    common = {"schema_version": "pipeline-1.0", "run_id": source.get("run_id") or run_id or f"run_{uuid.uuid4().hex}",
              "frame_id": source.get("frame_id") or frame_id or Path(rgb_path).stem}
    warnings = list(source.get("warnings", []))
    warnings.extend(["contact_roi_not_verified", "material_rejection_not_validated"] if prediction.instances else [])
    depth = deepcopy(source.get("depth")) or {
        "depth_path": None, "depth_size": None, "depth_dtype": None, "depth_unit": "unconfirmed",
        "depth_scale_to_m": None, "invalid_depth_values": None, "alignment": "unconfirmed",
        "intrinsics": None, "depth_usable": False}
    if depth_path:
        path = Path(depth_path).resolve()
        depth["depth_path"] = str(path)
        if path.is_file():
            (output / "depth").mkdir(exist_ok=True)
            shutil.copy2(path, output / "depth" / path.name)
            depth["depth_path"] = f"depth/{path.name}"
            try:
                with Image.open(path) as im:
                    depth["depth_size"] = list(im.size)
                    depth["depth_dtype"] = str(np.array(im).dtype)
            except (OSError, ValueError):
                warnings.append("depth_unreadable")
        else:
            warnings.append("depth_file_missing")
        depth["alignment"] = "aligned_to_rgb" if config.aligned_depth else "unconfirmed"
        depth["depth_scale_to_m"] = config.depth_scale
        depth["depth_unit"] = "m" if config.depth_scale == 1 else "mm" if config.depth_scale == .001 else "unconfirmed"
    elif depth.get("depth_path") and not Path(depth["depth_path"]).is_file():
        warnings.append("depth_file_missing")
    # Reference-only contract never implies that quality, invalid codes or registration were verified.
    depth["depth_usable"] = False
    depth["processing_status"] = "reference_only" if config.depth_processing == "reference" else "legacy_statistics_only"
    if depth.get("depth_path") and (depth.get("alignment") == "unconfirmed" or depth.get("depth_scale_to_m") is None):
        warnings.append("depth_metadata_unconfirmed")
    warnings = list(dict.fromkeys(warnings))
    (output / "rgb").mkdir(exist_ok=True)
    prediction.image.save(output / "rgb/frame.png")
    preprocess = {**common, "stage": "preprocess", "status": "ok", "prompt": prediction.prompt,
                  "rgb_path": "rgb/frame.png", "image_size": list(prediction.image.size), **depth,
                  "preprocess_version": "rgb_native_v1", "warnings": warnings,
                  "source_rgb_sha256": sha256(rgb_path)}
    items = []
    roi_items = []
    material_items = []
    final_items = []
    errors = []
    (output / "rois").mkdir(exist_ok=True)
    for index, (instance, old, roi) in enumerate(zip(prediction.instances, legacy["instances"], rois)):
        ident = old["instance_id"]
        box = mask_box(instance.mask)
        item = {"instance_id": ident, "object_label": instance.label, "bbox_xyxy": box,
                "mask_path": old["mask_ref"], "det_score": instance.det_score, "seg_score": instance.seg_score}
        items.append(item)
        roi_id = f"{ident}_bbox_0"
        crop_path = f"rois/{roi_id}.png"
        mask_path = f"rois/{roi_id}_mask.png"
        x1, y1, x2, y2 = box
        crop_mask = instance.mask[y1:y2, x1:x2]
        roi_error = old.get("processing_error")
        if roi is not None:
            roi.save(output / crop_path)
            Image.fromarray(crop_mask.astype(np.uint8) * 255).save(output / mask_path)
        roi_items.append({"instance_id": ident, "roi_id": roi_id, "status": "ok" if roi is not None else "error",
                          "contact_bbox_xyxy": box, "crop_path": crop_path if roi is not None else None,
                          "crop_mask_path": mask_path if roi is not None else None,
                          "crop_size": list(roi.size) if roi is not None else None,
                          "mask_coverage": float(crop_mask.mean()), "selection_method": "instance_bbox_fallback",
                          "contact_verified": False, "background_mode": "original" if config.crop_mode == "bbox" else "gray_127"})
        raw = old["material"]
        material = {"instance_id": ident, "roi_id": roi_id, "status": "error" if roi_error else "unknown",
                    "label": "unknown", "top1_label": raw.get("label"), "score": raw.get("score"),
                    "scores": raw.get("scores") or None, "accept_threshold": None,
                    "reason": "instance_processing_failed" if roi_error else "decision_policy_not_validated",
                    "decision_policy_version": "unvalidated_no_accept_v1"}
        material_items.append(material)
        if roi_error:
            errors.append({"stage": "material", "instance_id": ident, "code": "instance_processing_failed", "message": roi_error})
        final_items.append({**item, "segmentation": {"status": "ok", "det_score": instance.det_score, "seg_score": instance.seg_score},
                            "contact_roi": {"roi_id": roi_id, "bbox_xyxy": box, "crop_path": crop_path,
                                            "selection_method": "instance_bbox_fallback", "contact_verified": False} if roi is not None else None,
                            "material": material, "geometry": {"status": "not_run", "label": "unknown", "mask_area_px": int(instance.mask.sum())},
                            "deformability": {"status": "not_run", "label": "uncertain"},
                            "physical_attributes": old["physical_attributes"]})
    segmentation = {**common, "stage": "segmentation", "status": "ok" if items else "no_match",
                    "prompt": prediction.prompt, "preprocess_ref": "preprocess.json", "image_size": list(prediction.image.size),
                    "model_version": source.get("model_version", "seg_dino_tiny_sam21_v1"),
                    "instance_count": len(items), "instances": items, "depth": depth, "warnings": warnings, "errors": [],
                    "model_info": prediction.model_info, "latency_ms": prediction.latency_ms}
    if source:
        segmentation["import_provenance"] = source["import_provenance"]
        segmentation["source_preprocess_ref"] = source.get("preprocess_ref")
    material_input = {**common, "stage": "material_roi", "status": "partial" if errors else "ok" if items else "no_match",
                      "segmentation_ref": "segmentation.json", "roi_policy_version": "instance_bbox_fallback_v1",
                      "items": roi_items, "warnings": ["contact_roi_not_verified"] if items else []}
    material_result = {**common, "stage": "material", "status": "partial" if items else "no_match",
                       "input_ref": "material_input.json", "model_version": legacy["material_model"]["sha256"],
                       "preprocess_version": "resnet18_imagenet1k_v1", "decision_policy_version": "unvalidated_no_accept_v1",
                       "items": material_items, "errors": errors}
    result = {**common, "stage": "result", "status": "partial" if items else "no_match", "prompt": prediction.prompt,
              "rgb_path": "rgb/frame.png", "image_size": list(prediction.image.size), "depth": depth,
              "preprocess_ref": "preprocess.json", "segmentation_ref": "segmentation.json",
              "material_input_ref": "material_input.json", "material_result_ref": "material_result.json",
              "instance_count": len(items), "instances": final_items, "warnings": warnings, "errors": errors,
              "versions": {"pipeline": "0.4.0", "segmentation": segmentation["model_version"],
                           "material": material_result["model_version"], "roi_policy": material_input["roi_policy_version"]},
              "provenance": {"source": legacy["source"], "material_model": legacy["material_model"],
                             "segmentation_import": source.get("import_provenance"), "config": legacy["config"]},
              "timing": {"model_load_ms": legacy["model_load_ms"], "processing_before_contract_export_ms": legacy["frame_processing_ms"]}}
    for name, value in [("preprocess", preprocess), ("segmentation", segmentation), ("material_input", material_input),
                        ("material_result", material_result), ("result", result)]:
        write_json(output / f"{name}.json", value)
    lines = [f"# 合并链路结果 · {common['frame_id']}", "", f"- run_id: {common['run_id']}",
             f"- 实例数：{len(items)}；分割状态：{segmentation['status']}；最终状态：{result['status']}",
             "- bbox 为原图 mask 非零像素的整数半开区间。", "- 材质运行使用实例 bbox 回退区域，尚未验证接触面选择及拒识策略；top1 为模型候选，最终 label 保留 unknown。",
             f"- 深度：{depth['processing_status']}；等高线、三维尺寸和物理属性未实现。",
             "- imported 模式使用已有 mask，未重跑检测/分割；RGB 由调用者提供，原包无 RGB 哈希，无法自动证明同帧。" if source else "- 本次运行检测、分割和材质模型。",
             "", "| 实例 | 物体 | 整数框 | 检测分数 | 材质候选 | 材质分数 | 状态 |", "|---|---|---|---:|---|---:|---|"]
    for item in final_items:
        m = item["material"]
        lines.append(f"| {item['instance_id']} | {item['object_label']} | {item['bbox_xyxy']} | {item['det_score']} | {m['top1_label']} | {m['score']} | {m['status']} |")
    lines += ["", "警告：" + ", ".join(warnings)]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result

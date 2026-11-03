"""Portable pipeline-1.0 files alongside the stable perception-1.0 UI response.

Uses Pillow only so archive deployments do not acquire neural-model dependencies.
"""
import hashlib
import json
from PIL import Image


STAGE_FILES = {"preprocess.json", "segmentation.json", "material_input.json", "material_result.json", "result.json"}


def artifact_names(directory):
    names = set(STAGE_FILES)
    if (directory / "material_input.json").is_file():
        data = json.loads((directory / "material_input.json").read_text(encoding="utf-8"))
        for roi in data["items"]:
            names.update(v for v in (roi.get("crop_path"), roi.get("crop_mask_path")) if v)
    return names


def export_pipeline(directory, segmentation, perception):
    common = {"schema_version": "pipeline-1.0", "run_id": segmentation.run_id, "frame_id": segmentation.frame_id}
    rgb_path = "rgb.png" if (directory / "rgb.png").is_file() else None
    depth = segmentation.depth.model_dump()
    warnings = list(segmentation.warnings)
    if rgb_path is None:
        warnings.append("original_rgb_not_provided")
    if depth["depth_path"] and not (directory / "depth.png").is_file():
        warnings.append("depth_file_missing")
    if depth["alignment"] == "unconfirmed" or depth["depth_scale_to_m"] is None:
        warnings.append("depth_metadata_unconfirmed")
    rois, materials, instances, errors = [], [], [], []
    if rgb_path:
        with Image.open(directory / rgb_path) as source:
            rgb = source.convert("RGB")
    else:
        rgb = None
    for instance in perception.instances:
        ident = instance.instance_id
        roi_id = f"{ident}_bbox_0"
        crop_path = f"rois/{roi_id}.png" if rgb else None
        crop_mask_path = f"rois/{roi_id}_mask.png" if rgb else None
        roi = {"instance_id": ident, "roi_id": roi_id, "status": "ok" if rgb else "skipped",
               "contact_bbox_xyxy": instance.bbox_xyxy, "crop_path": crop_path, "crop_mask_path": crop_mask_path,
               "crop_size": None, "mask_coverage": None, "selection_method": "instance_bbox_fallback",
               "background_mode": "original", "contact_verified": False}
        if rgb:
            (directory / "rois").mkdir(exist_ok=True)
            crop = rgb.crop(instance.bbox_xyxy)
            crop.save(directory / crop_path)
            with Image.open(directory / instance.mask_path) as mask:
                crop_mask = mask.convert("L").crop(instance.bbox_xyxy)
            crop_mask.save(directory / crop_mask_path)
            roi.update(crop_size=list(crop.size), mask_coverage=crop_mask.histogram()[255] / (crop.width * crop.height))
        else:
            roi["reason"] = "original_rgb_not_provided"
        attr = instance.attributes.get("material")
        raw = attr.model_dump() if attr else {}
        values = raw.get("values", {})
        status = raw.get("status", "skipped")
        if status == "ok":
            status = "unknown"  # Adapter success does not establish an acceptance policy.
        material = {"instance_id": ident, "roi_id": roi_id, "status": status, "label": "unknown",
                    "top1_label": values.get("top1_label") or (raw.get("label") if raw.get("label") != "unknown" else None),
                    "score": raw.get("score"), "scores": values.get("scores") or None, "accept_threshold": None,
                    "reason": raw.get("reason") or ("decision_policy_not_validated" if attr else "material_not_requested"),
                    "decision_policy_version": "unvalidated_no_accept_v1",
                    "model_version": raw.get("model_version"), "checkpoint_sha256": values.get("checkpoint_sha256")}
        if status == "error":
            errors.append({"stage": "material", "instance_id": ident, "code": "module_error", "message": material["reason"]})
        rois.append(roi)
        materials.append(material)
        item = {k: getattr(instance, k) for k in ("instance_id", "object_label", "bbox_xyxy", "mask_path")}
        item.update(segmentation={"status": "ok", "det_score": instance.det_score, "seg_score": instance.seg_score},
                    contact_roi={"roi_id": roi_id, "bbox_xyxy": instance.bbox_xyxy, "crop_path": crop_path,
                                 "selection_method": "instance_bbox_fallback", "contact_verified": False} if rgb else None,
                    material=material, geometry={"status": "not_run", "label": "unknown"},
                    deformability={"status": "not_run", "label": "uncertain"},
                    attributes={k: v.model_dump() for k, v in instance.attributes.items()})
        instances.append(item)
    if instances:
        warnings.extend(["contact_roi_not_verified", "material_rejection_not_validated"])
    warnings = list(dict.fromkeys(warnings))
    pre = {**common, "stage": "preprocess", "status": "ok" if rgb else "partial", "rgb_path": rgb_path,
           "image_size": segmentation.image_size, "prompt": segmentation.prompt, **depth,
           "preprocess_version": "rgb_native_v1", "warnings": warnings,
           "rgb_sha256": hashlib.sha256((directory / rgb_path).read_bytes()).hexdigest() if rgb else None}
    mi = {**common, "stage": "material_roi", "status": "no_match" if not instances else "ok" if rgb else "partial",
          "segmentation_ref": "segmentation.json", "roi_policy_version": "instance_bbox_fallback_v1", "items": rois}
    mr = {**common, "stage": "material", "status": "partial" if instances else "no_match", "items": materials,
          "input_ref": "material_input.json", "model_version": perception.model_versions.get("material"),
          "preprocess_version": "resnet18_imagenet1k_v1", "decision_policy_version": "unvalidated_no_accept_v1", "errors": errors}
    final = {**common, "stage": "result", "status": "partial" if instances else "no_match", "prompt": segmentation.prompt,
             "rgb_path": rgb_path, "image_size": segmentation.image_size, "instance_count": len(instances), "instances": instances,
             "depth": depth, "preprocess_ref": "preprocess.json", "segmentation_ref": "segmentation.json",
             "material_input_ref": "material_input.json", "material_result_ref": "material_result.json",
             "versions": {"pipeline": "0.5.0", "segmentation": segmentation.model_version,
                          "material": perception.model_versions.get("material"), "roi_policy": "instance_bbox_fallback_v1"},
             "execution": perception.execution.model_dump(), "warnings": warnings, "errors": errors}
    # File references are portable; HTTP addresses remain only in perception.json.
    seg = segmentation.model_dump()
    for key in ("overlay_ref", "report_ref", "segmentation_ref"):
        seg.pop(key, None)
    for item in seg["instances"]:
        item.pop("mask_ref", None)
    for name, data in [("preprocess", pre), ("segmentation", seg), ("material_input", mi), ("material_result", mr), ("result", final)]:
        (directory / f"{name}.json").write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    with (directory / "report.md").open("a", encoding="utf-8") as stream:
        stream.write(f"\n- 标准结果：result.json（pipeline-1.0，状态 {final['status']}）\n")
        stream.write(f"- 分割新推理：{perception.execution.new_inference}；属性新推理：{perception.execution.new_attribute_inference}\n")
        stream.write("- 材质候选见 material.top1_label；接触面与接受策略未验收，正式标签保持 unknown。\n")

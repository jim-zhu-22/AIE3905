"""Reusable orchestration. Load models once, then process many independent frames."""
from pathlib import Path
from time import perf_counter
import numpy as np
from PIL import Image
from perception.geometry.measure import measure
from perception.attributes.estimate import estimate
from perception.io.export import save_prediction
from perception.runtime import sha256


def material_crop(image, instance, mode):
    if instance.mask.shape != (image.height, image.width):
        raise ValueError("Instance mask shape does not match RGB")
    if not np.any(instance.mask):
        return None
    box = np.asarray(instance.bbox_xyxy, dtype=float)
    if box.shape != (4,) or not np.isfinite(box).all():
        raise ValueError("Invalid detected ROI")
    x1, y1, x2, y2 = box
    box = (max(0, int(np.floor(x1))), max(0, int(np.floor(y1))),
           min(image.width, int(np.ceil(x2))), min(image.height, int(np.ceil(y2))))
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("Detected ROI is outside RGB")
    roi = image.crop(box)
    if mode == "masked":
        pixels = np.array(roi)
        pixels[~instance.mask[box[1]:box[3], box[0]:box[2]].astype(bool)] = 127
        roi = Image.fromarray(pixels)
    elif mode != "bbox":
        raise ValueError("Unknown crop mode")
    return roi


class PerceptionPipeline:
    def __init__(self, config, *, segmenter=None, material=None):
        started = perf_counter()
        self.config = config.validate(require_sam=segmenter is None)
        if segmenter is None:
            from perception.pipeline.segmentation import GroundedSegmenter
            segmenter = GroundedSegmenter(**{key: getattr(config, key) for key in (
                "dino_model", "dino_revision", "sam_weight", "device",
                "box_threshold", "text_threshold", "nms_threshold")})
        if material is None:
            from perception.models.resnet18.backend import Predictor
            material = Predictor(Path(config.material_weight), device=config.device)
        self.segmenter, self.material = segmenter, material
        self.material_info = {"checkpoint": str(config.material_weight),
                              "sha256": sha256(config.material_weight),
                              "device": str(material.device), "crop_mode": config.crop_mode}
        self.load_ms = (perf_counter() - started) * 1000

    def process(self, rgb_path, prompt, output_dir, depth_path=None, *, frame_id=None, run_id=None):
        output_dir = Path(output_dir)
        if output_dir.exists() and any(output_dir.iterdir()):
            raise ValueError(f"Output directory must be empty: {output_dir}")
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        started = perf_counter()
        with Image.open(rgb_path) as source:
            image = source.convert("RGB")
        depth = None
        if depth_path and self.config.depth_processing == "statistics":
            if not self.config.aligned_depth:
                raise ValueError("Depth registration must be confirmed via aligned_depth")
            with Image.open(depth_path) as source:
                depth = np.array(source)
            if depth.ndim != 2 or depth.shape != (image.height, image.width):
                raise ValueError("Depth dimensions must match RGB")
        prediction = self.segmenter.predict(image, prompt, self.config.mode)
        from perception.io.pipeline_contract import mask_box, export_contract
        # All downstream crops use the same integer, mask-derived bounds as the interchange file.
        for instance in prediction.instances:
            instance.bbox_xyxy = mask_box(instance.mask)
        material_started = perf_counter()
        extras = []
        rois = []
        for instance in prediction.instances:
            roi = None
            error = None
            try:
                roi = material_crop(image, instance, self.config.crop_mode)
                material = self.material.predict(roi) if roi is not None else {
                    "label": None, "score": None, "scores": {}, "status": "empty_mask"}
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                material = {"label": None, "score": None, "scores": {}, "status": "error"}
            rois.append(roi)
            extras.append({"material": material, "processing_error": error,
                           "geometry": measure(instance.mask, depth, self.config.depth_scale),
                           "physical_attributes": estimate()})
        material_ms = (perf_counter() - material_started) * 1000
        result = save_prediction(prediction, output_dir)
        (output_dir / "result.json").replace(output_dir / "segmentation_raw.json")
        for item, extra in zip(result["instances"], extras):
            item.update(extra)
        result.update({"schema_version": "1.1", "source": {
            "rgb": str(Path(rgb_path).resolve()),
            "depth": str(Path(depth_path).resolve()) if depth_path else None},
            "config": self.config.to_dict(), "material_model": self.material_info,
            "model_load_ms": round(self.load_ms, 2),
            "frame_processing_ms": round((perf_counter() - started) * 1000, 2),
            "material_and_geometry_ms": round(material_ms, 2)})
        from perception.io.manifest import write_json
        write_json(output_dir / "perception.json", result)
        export_contract(prediction, output_dir, rgb_path, depth_path, self.config, result, rois,
                        frame_id=frame_id, run_id=run_id)
        return result

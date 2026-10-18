"""Prompt detection, fixed-tile detection, and full-image instance masks.

RGB is the only model input in this baseline. Depth is retained in the dataset
but is not silently fed to either pretrained model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any
import inspect
import json
import os
import hashlib
import platform

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DINO_REVISION = "a2bb814dd30d776dcf7e30523b00659f4f141c71"


@dataclass
class Instance:
    bbox_xyxy: list[float]
    det_score: float
    label: str
    mask: np.ndarray
    source_tiles: list[str] = field(default_factory=list)


@dataclass
class Prediction:
    image: Image.Image
    instances: list[Instance]
    mode: str
    prompt: str
    latency_ms: dict[str, float]
    raw_detections: int
    model_info: dict[str, str]


def box_iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def fixed_tiles(width: int, height: int) -> list[tuple[str, tuple[int, int, int, int]]]:
    """For 1280x960: 704x528 tiles at x=0/576, y=0/432."""
    tw, th = min(704, width), min(528, height)
    xs = sorted({0, width - tw})
    ys = sorted({0, height - th})
    return [(f"x{x}_y{y}", (x, y, x + tw, y + th)) for y in ys for x in xs]


def deduplicate(detections: list[dict[str, Any]], threshold: float = 0.5) -> list[dict[str, Any]]:
    """Score-ordered box NMS after restoring full-image coordinates."""
    kept: list[dict[str, Any]] = []
    for candidate in sorted(detections, key=lambda x: x["score"], reverse=True):
        duplicate = next(
            (old for old in kept if box_iou(candidate["box"], old["box"]) >= threshold),
            None,
        )
        if duplicate is None:
            kept.append(candidate)
        else:
            duplicate["source_tiles"].extend(candidate["source_tiles"])
    return kept


class GroundedSegmenter:
    def __init__(
        self,
        *,
        dino_model: str = "IDEA-Research/grounding-dino-tiny",
        dino_revision: str = DINO_REVISION,
        sam_weight: str = str(PROJECT_ROOT / "sam2.1_t.pt"),
        device: str = "cpu",
        box_threshold: float = 0.25,
        text_threshold: float = 0.20,
        nms_threshold: float = 0.5,
    ) -> None:
        os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".model_cache" / "huggingface"))
        os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / ".model_cache" / "ultralytics"))
        import torch
        import transformers
        import ultralytics
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        from ultralytics import SAM

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but this PyTorch build has no CUDA device")
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.nms_threshold = nms_threshold
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(dino_model, revision=dino_revision)
        self.detector = AutoModelForZeroShotObjectDetection.from_pretrained(
            dino_model, revision=dino_revision
        ).to(self.device)
        self.detector.eval()
        self.segmenter = SAM(sam_weight)
        checkpoint = Path(sam_weight)
        checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest() if checkpoint.is_file() else "unavailable"
        self.model_info = {
            "detector": dino_model,
            "detector_revision": dino_revision,
            "segmenter": sam_weight,
            "segmenter_sha256": checkpoint_hash,
            "device": self.device,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "ultralytics": ultralytics.__version__,
            "python": platform.python_version(),
        }

    def _detect(self, image: Image.Image, prompt: str) -> list[dict[str, Any]]:
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            outputs = self.detector(**inputs)
        post = self.processor.post_process_grounded_object_detection
        params = inspect.signature(post).parameters
        kwargs: dict[str, Any] = {"target_sizes": [(image.height, image.width)]}
        if "threshold" in params:
            kwargs["threshold"] = self.box_threshold
        if "box_threshold" in params:
            kwargs["box_threshold"] = self.box_threshold
        if "text_threshold" in params:
            kwargs["text_threshold"] = self.text_threshold
        args = [outputs]
        if "input_ids" in params:
            args.append(inputs.input_ids)
        result = post(*args, **kwargs)[0]
        labels = result.get("text_labels", result.get("labels", []))
        return [
            {
                "box": [float(v) for v in box.detach().cpu().tolist()],
                "score": float(score.detach().cpu()),
                "label": str(label),
                "source_tiles": [],
            }
            for box, score, label in zip(result["boxes"], result["scores"], labels)
        ]

    def predict(self, image: Image.Image, prompt: str, mode: str = "full") -> Prediction:
        if mode not in {"full", "tiled"}:
            raise ValueError("mode must be full or tiled")
        image = image.convert("RGB")
        started = perf_counter()
        detections: list[dict[str, Any]] = []
        if mode == "full":
            detections = self._detect(image, prompt)
            for item in detections:
                item["source_tiles"] = ["full"]
        else:
            for tile_id, (x1, y1, x2, y2) in fixed_tiles(image.width, image.height):
                for item in self._detect(image.crop((x1, y1, x2, y2)), prompt):
                    box = item["box"]
                    item["box"] = [box[0] + x1, box[1] + y1, box[2] + x1, box[3] + y1]
                    item["source_tiles"] = [tile_id]
                    detections.append(item)
        detected = perf_counter()
        raw_count = len(detections)
        detections = deduplicate(detections, self.nms_threshold)
        merged = perf_counter()

        masks: list[np.ndarray] = []
        if detections:
            boxes = [item["box"] for item in detections]
            results = self.segmenter.predict(
                source=np.array(image), bboxes=boxes, device=self.device, verbose=False
            )
            if not results or results[0].masks is None:
                raise RuntimeError("SAM returned no masks for nonempty detection boxes")
            raw_masks = results[0].masks.data.detach().cpu().numpy()
            if len(raw_masks) != len(boxes):
                raise RuntimeError(f"SAM returned {len(raw_masks)} masks for {len(boxes)} boxes")
            for raw_mask in raw_masks:
                if raw_mask.shape != (image.height, image.width):
                    mask_image = Image.fromarray((raw_mask > 0.5).astype(np.uint8) * 255)
                    raw_mask = np.asarray(mask_image.resize(image.size, Image.Resampling.NEAREST))
                masks.append(raw_mask > 0.5)
        segmented = perf_counter()
        instances = [
            Instance(
                bbox_xyxy=[round(v, 2) for v in item["box"]],
                det_score=round(item["score"], 4),
                label=item["label"],
                mask=mask,
                source_tiles=sorted(set(item["source_tiles"])),
            )
            for item, mask in zip(detections, masks)
        ]
        return Prediction(
            image=image,
            instances=instances,
            mode=mode,
            prompt=prompt,
            latency_ms={
                "detection": round((detected - started) * 1000, 2),
                "merge": round((merged - detected) * 1000, 2),
                "segmentation": round((segmented - merged) * 1000, 2),
                "total": round((segmented - started) * 1000, 2),
            },
            raw_detections=raw_count,
            model_info=self.model_info,
        )


def overlay(prediction: Prediction) -> Image.Image:
    canvas = np.asarray(prediction.image).copy()
    for i, instance in enumerate(prediction.instances):
        color = np.array([36, 220, 87] if i % 2 == 0 else [36, 170, 245])
        canvas[instance.mask] = (0.55 * canvas[instance.mask] + 0.45 * color).astype(np.uint8)
    output = Image.fromarray(canvas)
    draw = ImageDraw.Draw(output)
    for i, instance in enumerate(prediction.instances):
        draw.rectangle(instance.bbox_xyxy, outline="red", width=2)
        draw.text((instance.bbox_xyxy[0], max(0, instance.bbox_xyxy[1] - 13)),
                  f"{i}: {instance.label} {instance.det_score:.2f}", fill="red")
    return output


def save_prediction(prediction: Prediction, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = output_dir / "masks"
    mask_dir.mkdir(exist_ok=True)
    items: list[dict[str, Any]] = []
    for index, instance in enumerate(prediction.instances):
        mask_path = mask_dir / f"i{index:03d}.png"
        Image.fromarray(instance.mask.astype(np.uint8) * 255).save(mask_path)
        items.append({
            "instance_id": f"i{index:03d}",
            "bbox_xyxy": instance.bbox_xyxy,
            "det_score": instance.det_score,
            "label": instance.label,
            "mask_ref": f"masks/{mask_path.name}",
            "source_tiles": instance.source_tiles,
        })
    overlay(prediction).save(output_dir / "overlay.jpg", quality=92)
    result = {
        "mode": prediction.mode,
        "prompt": prediction.prompt,
        "model_info": prediction.model_info,
        "raw_detections": prediction.raw_detections,
        "instance_count": len(items),
        "latency_ms": prediction.latency_ms,
        "instances": items,
    }
    (output_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result

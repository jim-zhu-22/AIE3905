from __future__ import annotations
from pathlib import Path
from time import perf_counter
from typing import Any
import os
from PIL import Image
from perception.models.grounding_dino.backend import Detector, DINO_REVISION
from perception.models.sam2.backend import Segmenter
from perception.pipeline.contracts import Instance, Prediction
from perception.geometry.boxes import fixed_tiles, deduplicate
PROJECT_ROOT = Path(__file__).resolve().parents[2]

class GroundedSegmenter:
    def __init__(
        self,
        *,
        dino_model: str = "IDEA-Research/grounding-dino-tiny",
        dino_revision: str = DINO_REVISION,
        sam_weight: str = str(PROJECT_ROOT / "week1_model_selection" / "sam2.1_t.pt"),
        device: str = "cpu",
        box_threshold: float = 0.25,
        text_threshold: float = 0.20,
        nms_threshold: float = 0.5,
        segmentation_factory=Segmenter,
    ) -> None:
        os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".model_cache" / "huggingface"))
        os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / ".model_cache" / "ultralytics"))
        Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
        from perception.runtime import resolve_device, runtime_versions
        self.device = resolve_device(device)
        self.nms_threshold = nms_threshold
        self.detection_backend = Detector(dino_model, dino_revision, self.device, box_threshold, text_threshold)
        self.segmentation_backend = segmentation_factory(sam_weight, self.device)
        self.model_info = {**runtime_versions(), **self.detection_backend.metadata,
                           **self.segmentation_backend.metadata, "device": self.device}

    def _detect(self, image, prompt):
        return self.detection_backend.predict(image, prompt)

    def predict(self, image: Image.Image, prompt: str, mode: str = "full") -> Prediction:
        if mode not in {"full", "tiled"}:
            raise ValueError("mode must be full or tiled")
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")
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

        masks = self.segmentation_backend.predict(image, [item["box"] for item in detections])
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

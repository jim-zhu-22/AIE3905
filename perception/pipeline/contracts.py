from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from PIL import Image

@dataclass
class Instance:
    bbox_xyxy: list[float]
    det_score: float
    label: str
    mask: np.ndarray
    source_tiles: list[str] = field(default_factory=list)
    instance_id: str | None = None
    seg_score: float | None = None


@dataclass
class Prediction:
    image: Image.Image
    instances: list[Instance]
    mode: str
    prompt: str
    latency_ms: dict[str, float]
    raw_detections: int
    model_info: dict[str, str]
    source_contract: dict | None = None

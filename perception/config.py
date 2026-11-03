from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

@dataclass
class PipelineConfig:
    sam_weight: str
    material_weight: str
    device: str = "cpu"
    dino_model: str = "IDEA-Research/grounding-dino-tiny"
    dino_revision: str = "a2bb814dd30d776dcf7e30523b00659f4f141c71"
    box_threshold: float = .25
    text_threshold: float = .20
    nms_threshold: float = .5
    mode: str = "full"
    crop_mode: str = "bbox"
    aligned_depth: bool = False
    depth_scale: float | None = None
    depth_processing: str = "reference"

    def validate(self, *, require_sam=True):
        if self.device not in {"cpu", "cuda", "auto"}:
            raise ValueError("Invalid device")
        if self.mode not in {"full", "tiled"} or self.crop_mode not in {"bbox", "masked"}:
            raise ValueError("Invalid mode or crop_mode")
        if not isinstance(self.aligned_depth, bool):
            raise ValueError("aligned_depth must be a boolean")
        if self.depth_processing not in {"reference", "statistics"}:
            raise ValueError("depth_processing must be reference or statistics")
        for name in ("box_threshold", "text_threshold", "nms_threshold"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        if self.depth_scale is not None and (not math.isfinite(self.depth_scale) or self.depth_scale <= 0):
            raise ValueError("depth_scale must be positive and finite")
        for name in ("sam_weight", "material_weight"):
            if name == "sam_weight" and not require_sam:
                continue
            path = Path(getattr(self, name))
            if not path.is_file() or path.stat().st_size < 1024:
                raise ValueError(f"{name} must reference a complete local checkpoint: {path}")
        return self

    def to_dict(self):
        return asdict(self)

def load_config(path, overrides=None, *, require_sam=True):
    path = Path(path).resolve()
    values = json.loads(path.read_text(encoding="utf-8-sig"))
    for key in ("sam_weight", "material_weight"):
        if key in values:
            values[key] = str((path.parent / values[key]).resolve())
    values.update({k: v for k, v in (overrides or {}).items() if v is not None})
    return PipelineConfig(**values).validate(require_sam=require_sam)

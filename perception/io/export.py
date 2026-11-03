from __future__ import annotations
from pathlib import Path
from typing import Any
import json
import numpy as np
from PIL import Image, ImageDraw
from perception.pipeline.contracts import Prediction

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
        instance_id = instance.instance_id or f"i{index:03d}"
        mask_path = mask_dir / f"{instance_id}.png"
        Image.fromarray(instance.mask.astype(np.uint8) * 255).save(mask_path)
        items.append({
            "instance_id": instance_id,
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

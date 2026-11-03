from __future__ import annotations
from typing import Any
from PIL import Image
DINO_REVISION = "a2bb814dd30d776dcf7e30523b00659f4f141c71"
import inspect

class Detector:
    def __init__(self, model, revision, device, box_threshold, text_threshold):
        import torch
        self.processor, self.detector = load_detector(model, revision, device)
        self.torch = torch
        self.device, self.box_threshold, self.text_threshold = device, box_threshold, text_threshold
        self.metadata = {"detector": model, "detector_revision": revision}

    def predict(self, image: Image.Image, prompt: str) -> list[dict[str, Any]]:
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


def load_detector(model, revision, device):
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    processor = AutoProcessor.from_pretrained(model, revision=revision)
    detector = AutoModelForZeroShotObjectDetection.from_pretrained(model, revision=revision).to(device)
    detector.eval()
    return processor, detector

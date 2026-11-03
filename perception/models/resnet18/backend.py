from pathlib import Path

import torch
from PIL import Image

from .labels import LABELS
from .preprocess import transform
from .model import build_model


class Predictor:
    def __init__(self, checkpoint_path: Path, device="auto"):
        from perception.runtime import resolve_device
        checkpoint_path = Path(checkpoint_path)
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(resolve_device(device))
        if self.device.type == "cpu":
            torch.set_num_threads(8)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if tuple(checkpoint["labels"]) != LABELS:
            raise ValueError("Checkpoint label order does not match this project")
        self.model = build_model(pretrained=False)
        self.model.load_state_dict(checkpoint["model"])
        self.model.to(self.device).eval()
        self.preprocess = transform(False)

    def predict(self, image: Image.Image):
        tensor = self.preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            probabilities = self.model(tensor).softmax(1)[0].cpu().tolist()
        best = max(range(len(LABELS)), key=probabilities.__getitem__)
        return {
            "label": LABELS[best],
            "score": probabilities[best],
            "scores": dict(zip(LABELS, probabilities)),
            "model_version": self.checkpoint_path.parent.name,
        }

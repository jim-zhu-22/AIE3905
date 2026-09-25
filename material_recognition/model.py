import os
from pathlib import Path

from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from . import LABELS


def build_model(pretrained: bool = True):
    # Keep downloaded ImageNet weights inside this project unless explicitly overridden.
    os.environ.setdefault("TORCH_HOME", str(Path(__file__).resolve().parents[1] / "models" / "pretrained"))
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(LABELS))
    return model

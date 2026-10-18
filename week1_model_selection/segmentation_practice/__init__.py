"""Independent Grounding DINO + SAM experiment for the RGB-D project."""

from .pipeline import GroundedSegmenter, Prediction, Instance

__all__ = ["GroundedSegmenter", "Prediction", "Instance"]

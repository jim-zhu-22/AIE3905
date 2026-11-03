"""Compatibility entry point for the optional YOLOE comparison model."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from perception.models.yoloe.backend import run_segmentation

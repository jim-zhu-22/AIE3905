"""Compatibility imports; implementations live in perception."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from perception.pipeline.segmentation import GroundedSegmenter, PROJECT_ROOT, DINO_REVISION
from perception.pipeline.contracts import Instance, Prediction
from perception.geometry.boxes import box_iou, fixed_tiles, deduplicate
from perception.io.export import overlay, save_prediction

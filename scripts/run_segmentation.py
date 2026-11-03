"""Compatibility entry point for sam1."""
import sys
from pathlib import Path
from functools import partial
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from perception.pipeline.legacy import run_segmentation as _run, draw_overlay, main
run_segmentation = partial(_run, engine="sam1")
if __name__ == "__main__":
    main("sam1")

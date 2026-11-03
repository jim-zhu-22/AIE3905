"""Read-only runtime and checkpoint checks, without loading model networks."""
import argparse
import json
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {"python_executable": sys.executable, "checks": {}, "ok": True}
    try:
        cache = Path(__file__).resolve().parents[1] / ".model_cache/ultralytics"
        os.environ.setdefault("YOLO_CONFIG_DIR", str(cache))
        Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
        from perception.runtime import runtime_versions
        import numpy as np
        import torch
        import torchvision
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        from ultralytics import SAM
        from perception.models.resnet18.backend import Predictor
        report["versions"] = runtime_versions()
        report["checks"]["numpy_operation"] = int(np.array([1, 2]).sum()) == 3
        report["checks"]["torch_operation"] = int(torch.tensor([1, 2]).sum()) == 3
        report["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            report["gpu"] = torch.cuda.get_device_name(0)
        if args.config:
            from perception.config import load_config
            from perception.runtime import resolve_device, sha256
            config = load_config(args.config)
            report["device"] = resolve_device(config.device)
            report["checkpoint_sha256"] = {name: sha256(getattr(config, name))
                                           for name in ("sam_weight", "material_weight")}
        report["ok"] = all(report["checks"].values())
    except Exception as error:
        report["ok"] = False
        report["error"] = f"{type(error).__name__}: {error}"
    if args.output:
        from perception.io.manifest import write_json
        write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

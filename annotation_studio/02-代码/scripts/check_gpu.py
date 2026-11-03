"""Check actual CUDA forward/backward work in a time-limited child process."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def probe(mode, batch_size):
    import torch
    import torchvision
    from torchvision.models import resnet18
    info = {"python": sys.version.split()[0], "torch": torch.__version__,
            "torchvision": torchvision.__version__, "cuda_runtime": torch.version.cuda,
            "available": torch.cuda.is_available()}
    print(json.dumps(info), flush=True)
    if not info["available"]:
        raise RuntimeError("CUDA unavailable")
    print(json.dumps({"gpu": torch.cuda.get_device_name(0)}), flush=True)
    model = resnet18(weights=None).cuda()
    model.train(mode == "train")
    images = torch.ones(batch_size, 3, 224, 224, device="cuda")
    labels = torch.zeros(batch_size, dtype=torch.long, device="cuda")
    started = time.perf_counter()
    if mode == "train":
        loss = torch.nn.functional.cross_entropy(model(images), labels)
        loss.backward()
    else:
        with torch.inference_mode():
            model(images)
    torch.cuda.synchronize()
    print(json.dumps({"status": "passed", "mode": mode, "batch_size": batch_size,
                      "compute_seconds": time.perf_counter() - started}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mode", choices=("train", "eval"), default="train")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    if args.probe:
        probe(args.mode, args.batch_size)
        return
    try:
        result = subprocess.run([sys.executable, "-u", __file__, "--probe", "--mode", args.mode,
                                 "--batch-size", str(args.batch_size)], capture_output=True,
                                text=True, timeout=args.timeout)
        report = {"status": "passed" if result.returncode == 0 else "failed", "stdout": result.stdout,
                  "stderr": result.stderr, "exit_code": result.returncode}
    except subprocess.TimeoutExpired as error:
        def decode(value):
            return value.decode(errors="replace") if isinstance(value, bytes) else value or ""
        report = {"status": "timeout", "timeout_seconds": args.timeout,
                  "stdout": decode(error.stdout), "stderr": decode(error.stderr)}
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

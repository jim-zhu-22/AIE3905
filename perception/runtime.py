import hashlib
import importlib.metadata
import platform
from pathlib import Path

def resolve_device(device):
    import torch
    if device not in {"cpu", "cuda", "auto"}:
        raise ValueError("device must be cpu, cuda or auto")
    available = torch.cuda.is_available()
    if device == "cuda" and not available:
        raise RuntimeError("CUDA requested but unavailable")
    return ("cuda" if available else "cpu") if device == "auto" else device

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def runtime_versions():
    return {"python": platform.python_version(), **{
        name: importlib.metadata.version(name)
        for name in ("numpy", "Pillow", "torch", "torchvision", "transformers", "ultralytics")}}

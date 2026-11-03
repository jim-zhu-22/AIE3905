"""Publish complete bundles atomically; never expose half-written output."""

from contextlib import contextmanager
import os
import re
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
import tempfile
import json

from PIL import Image, ImageChops

from .schemas import SegmentationResult


def bundle_inputs(directory, result):
    """Only declared, contained original inputs are eligible for downstream inference."""
    paths = set()
    rgb = result.rgb_path
    if result.preprocess_ref:
        preprocess = contained_file(directory, result.preprocess_ref)
        data = json.loads(preprocess.read_text(encoding="utf-8"))
        if data.get("frame_id") != result.frame_id or data.get("image_size") != result.image_size:
            raise ValueError("Preprocess frame does not match segmentation")
        rgb = rgb or data.get("rgb_path")
        paths.add(result.preprocess_ref)
    if rgb:
        if rgb == "overlay.jpg":
            raise ValueError("Overlay cannot be used as original RGB")
        path = contained_file(directory, rgb)
        with Image.open(path) as image:
            if image.format not in {"PNG", "JPEG"} or list(image.size) != result.image_size:
                raise ValueError("Original RGB dimensions/format do not match segmentation")
            image.verify()
        paths.add(rgb)
    depth = result.depth.depth_path
    if depth:
        try:
            path = contained_file(directory, depth)
        except ValueError:
            # Historical packages can contain an external, unresolved depth reference.
            depth = None
        else:
            with Image.open(path) as image:
                if image.format != "PNG" or image.mode not in {"I", "I;16", "I;16L", "I;16B"} or list(image.size) != result.image_size:
                    raise ValueError("Invalid referenced depth")
                image.verify()
            paths.add(depth)
    return rgb, depth, paths


def contained_file(directory: Path, relative: str) -> Path:
    posix = PurePosixPath(relative)
    if (not relative or posix.is_absolute() or PureWindowsPath(relative).drive or "\\" in relative
            or ".." in posix.parts or str(posix) != relative):
        raise ValueError("Artifact paths must be portable relative paths")
    path = (directory / relative).resolve()
    if not path.is_relative_to(directory.resolve()) or not path.is_file():
        raise ValueError(f"Missing file or unsafe path: {relative}")
    return path


def load_bundle(directory: Path, *, check_pixels: bool = False) -> SegmentationResult:
    data = SegmentationResult.model_validate_json(contained_file(directory, "segmentation.json").read_text(encoding="utf-8"))
    width, height = data.image_size
    for item in data.instances:
        if not item.mask_path.startswith("masks/") or not item.mask_path.endswith(".png"):
            raise ValueError("Masks must be PNG files inside masks/")
        path = contained_file(directory, item.mask_path)
        if check_pixels:
            with Image.open(path) as source:
                if source.format != "PNG" or source.size != (width, height):
                    raise ValueError(f"Invalid mask format or size: {item.instance_id}")
                r, g, b = source.convert("RGB").split()
                if ImageChops.difference(r, g).getbbox() or ImageChops.difference(r, b).getbbox():
                    raise ValueError("Mask channels must agree")
                if "A" in source.getbands() and source.getchannel("A").getextrema() != (255, 255):
                    raise ValueError("Masks must be opaque")
                mask = source.convert("L")
                if any(mask.histogram()[1:255]):
                    raise ValueError(f"Mask is not binary: {item.instance_id}")
                if mask.getbbox() != tuple(item.bbox_xyxy):
                    raise ValueError(f"Mask and bbox disagree: {item.instance_id}")
    contained_file(directory, "report.md")
    overlay = contained_file(directory, "overlay.jpg")
    if check_pixels:
        with Image.open(overlay) as image:
            if image.format != "JPEG" or image.size != (width, height):
                raise ValueError("Overlay must be a JPEG matching image_size")
            image.verify()
    return data


@contextmanager
def staged_bundle(parent: Path, bundle_id: str):
    """Reserve a name and move validated output into view on the same filesystem."""
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", bundle_id):
        raise ValueError("Invalid bundle_id")
    parent = parent.resolve()
    parent.mkdir(parents=True, exist_ok=True)
    final = parent / bundle_id
    reservation = parent / f".{bundle_id}.lock"
    lock_fd = os.open(reservation, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    stage = None
    try:
        os.close(lock_fd)
        if final.exists():
            raise FileExistsError(f"Target already exists: {final}")
        stage = Path(tempfile.mkdtemp(prefix=f".{bundle_id}-", dir=parent))
        yield stage
        load_bundle(stage, check_pixels=True)
        stage.rename(final)
        stage = None
    finally:
        if stage is not None and stage.exists():
            if not stage.resolve().is_relative_to(parent) or not stage.name.startswith(f".{bundle_id}-"):
                raise ValueError("Refusing cleanup outside staging directory")
            shutil.rmtree(stage)
        reservation.unlink(missing_ok=True)

"""Validate actual image content, not client filenames or MIME claims."""

import base64
import binascii
from io import BytesIO
import warnings

from PIL import Image, UnidentifiedImageError

from .errors import fail
from .schemas import MAX_IMAGE_BYTES, MAX_PIXELS


def decode_bytes(raw: bytes) -> Image.Image:
    if len(raw) > MAX_IMAGE_BYTES:
        fail(413, "image_too_large", "图片不能超过 15 MiB。")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(raw)) as image:
                if image.width * image.height > MAX_PIXELS:
                    fail(413, "image_too_large", "图片不能超过 2000 万像素。")
                if image.format not in {"PNG", "JPEG"}:
                    fail(415, "unsupported_image_format", "请上传 PNG 或 JPEG 图片。")
                if getattr(image, "n_frames", 1) != 1:
                    fail(415, "animated_image_not_supported", "请上传单帧 PNG 或 JPEG 图片。")
                image.verify()
            with Image.open(BytesIO(raw)) as image:
                # Do not rotate using EXIF: mask and bbox refer to stored pixel coordinates.
                return image.convert("RGB")
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        fail(413, "image_too_large", "图片像素数量超过限制。")
    except (ValueError, UnidentifiedImageError, OSError):
        fail(400, "invalid_image", "图片损坏或无法解码。")


def decode_image(value: str) -> Image.Image:
    if value.startswith("data:"):
        value = value.partition(",")[2]
    try:
        raw = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error):
        fail(400, "invalid_image", "图片必须是有效的 Base64 图片。")
    return decode_bytes(raw)

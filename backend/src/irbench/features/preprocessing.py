"""Deterministic image loading, cropping and resizing."""

from __future__ import annotations

import io
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

RESAMPLE = Image.Resampling.BICUBIC


def load_image(path: str | Path) -> Image.Image:
    """Load an image from disk as RGB, applying EXIF orientation."""
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        return img.convert("RGB")


def decode_image_bytes(data: bytes, max_pixels: int | None = None) -> Image.Image:
    """Decode uploaded bytes into an RGB image.

    Raises:
        ValueError: if the bytes are not a decodable image or exceed ``max_pixels``.
    """
    try:
        with Image.open(io.BytesIO(data)) as img:
            if max_pixels is not None and img.width * img.height > max_pixels:
                raise ValueError(f"image has {img.width * img.height} pixels, limit is {max_pixels}")
            img = ImageOps.exif_transpose(img)
            img.load()
            return img.convert("RGB")
    except ValueError:
        raise
    except Exception as exc:  # PIL raises many exception types on corrupt input
        raise ValueError(f"cannot decode image: {exc}") from exc


def crop_bbox(img: Image.Image, bbox: Sequence[float] | None) -> Image.Image:
    """Crop ``[x1, y1, x2, y2]`` (clamped to the image, rounded to whole pixels)."""
    if bbox is None:
        return img
    x1, y1, x2, y2 = (float(v) for v in bbox)
    x1 = int(np.clip(np.floor(x1), 0, img.width - 1))
    y1 = int(np.clip(np.floor(y1), 0, img.height - 1))
    x2 = int(np.clip(np.ceil(x2), x1 + 1, img.width))
    y2 = int(np.clip(np.ceil(y2), y1 + 1, img.height))
    return img.crop((x1, y1, x2, y2))


def resize_max_side(img: Image.Image, max_side: int | None, multiple_of: int = 1) -> Image.Image:
    """Resize so the longer side equals ``max_side`` (aspect preserved).

    When ``multiple_of > 1`` both sides are rounded down to a multiple of it (minimum one
    multiple) - required for ViTs whose patch size must tile the input exactly.
    """
    if max_side is None and multiple_of <= 1:
        return img
    w, h = img.size
    if max_side is not None:
        scale = max_side / float(max(w, h))
        w, h = max(1, round(w * scale)), max(1, round(h * scale))
    if multiple_of > 1:
        w = max(multiple_of, (w // multiple_of) * multiple_of)
        h = max(multiple_of, (h // multiple_of) * multiple_of)
    if (w, h) == img.size:
        return img
    return img.resize((w, h), RESAMPLE)


def make_thumbnail(img: Image.Image, size: int = 256) -> Image.Image:
    """Aspect-preserving thumbnail whose longer side is ``size``."""
    thumb = img.copy()
    thumb.thumbnail((size, size), RESAMPLE)
    return thumb


def verify_image(path: str | Path) -> bool:
    """Return ``True`` if the file fully decodes as an image."""
    try:
        with Image.open(path) as img:
            img.load()
        return True
    except Exception:
        return False

"""Pure image decode, bbox validation, and crop helpers."""
from __future__ import annotations

import base64
import io
from numbers import Real
from typing import Any

from PIL import Image

from backend.services.image_limits import ImageSizeLimitError, ensure_image_within_limits


def decode_base64_image(
    value: str,
    *,
    max_pixels: int | None = None,
    max_dimension: int | None = None,
) -> Image.Image:
    raw = base64.b64decode(value, validate=True)
    try:
        with Image.open(io.BytesIO(raw)) as probe:
            ensure_image_within_limits(probe, max_pixels=max_pixels, max_dimension=max_dimension)
            probe.verify()
        with Image.open(io.BytesIO(raw)) as image:
            ensure_image_within_limits(image, max_pixels=max_pixels, max_dimension=max_dimension)
            rgb = image.convert("RGB")
            rgb.load()
            return rgb
    except (Image.DecompressionBombError, ImageSizeLimitError, OSError, ValueError) as exc:
        raise ValueError(str(exc)) from exc


def decode_data_url(
    value: str,
    *,
    max_pixels: int | None = None,
    max_dimension: int | None = None,
) -> Image.Image:
    if value.startswith("data:"):
        comma = value.find(",")
        if comma == -1:
            raise ValueError("malformed data URL: no comma found")
        value = value[comma + 1 :]
    return decode_base64_image(value, max_pixels=max_pixels, max_dimension=max_dimension)


def validate_bbox(bbox: Any, image_size: tuple[int, int], *, require_positive: bool = False) -> list[int | float]:
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError("bbox must be [x, y, w, h]")
    x, y, w, h = bbox
    if any(isinstance(value, bool) or not isinstance(value, Real) for value in bbox):
        raise ValueError("bbox values must be numeric")
    iw, ih = image_size
    if x < 0 or y < 0 or x + w > iw or y + h > ih:
        raise ValueError("bbox out of image bounds")
    if require_positive and (w <= 0 or h <= 0):
        raise ValueError("bbox out of image bounds")
    return bbox


def crop_bbox(image: Image.Image, bbox: list[int | float]) -> Image.Image:
    x, y, w, h = bbox
    return image.crop((x, y, x + w, y + h))

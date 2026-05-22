"""Pure image decode, bbox validation, and crop helpers."""
from __future__ import annotations

import base64
import io
from numbers import Real
from typing import Any

from PIL import Image


def decode_base64_image(value: str) -> Image.Image:
    raw = base64.b64decode(value)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def decode_data_url(value: str) -> Image.Image:
    if value.startswith("data:"):
        comma = value.find(",")
        if comma == -1:
            raise ValueError("malformed data URL: no comma found")
        value = value[comma + 1 :]
    return decode_base64_image(value)


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

"""Shared image size limits for upload and base64/data-url decoding."""
from __future__ import annotations

import os
from typing import Protocol

DEFAULT_MAX_IMAGE_PIXELS = 80_000_000
DEFAULT_MAX_IMAGE_DIMENSION = 10_000


class SizedImage(Protocol):
    size: tuple[int, int]


class ImageSizeLimitError(ValueError):
    """Raised when an image exceeds configured Phase 1 decode limits."""


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def default_max_image_pixels() -> int:
    return _env_int("MAX_IMAGE_PIXELS", DEFAULT_MAX_IMAGE_PIXELS)


def default_max_image_dimension() -> int:
    return _env_int("MAX_IMAGE_DIMENSION", DEFAULT_MAX_IMAGE_DIMENSION)


def image_limits_message(max_pixels: int, max_dimension: int) -> str:
    return f"image exceeds maximum size ({max_pixels} pixels, {max_dimension}px per side)"


def ensure_image_within_limits(
    image: SizedImage,
    *,
    max_pixels: int | None = None,
    max_dimension: int | None = None,
) -> None:
    limit_pixels = max_pixels or default_max_image_pixels()
    limit_dimension = max_dimension or default_max_image_dimension()
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ImageSizeLimitError("image dimensions must be positive")
    if width > limit_dimension or height > limit_dimension or width * height > limit_pixels:
        raise ImageSizeLimitError(image_limits_message(limit_pixels, limit_dimension))

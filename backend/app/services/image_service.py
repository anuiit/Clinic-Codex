"""Upload image decoding helpers for API routes."""
from __future__ import annotations

import io
from typing import BinaryIO, Protocol

from PIL import Image, UnidentifiedImageError


class ReadableUpload(Protocol):
    def read(self, *args, **kwargs) -> bytes: ...


class InvalidImageError(ValueError):
    """Raised when uploaded bytes cannot be decoded as an image."""


INVALID_IMAGE_CODE = "INVALID_IMAGE"
INVALID_IMAGE_MESSAGE = "uploaded file is not a valid image"


def invalid_image_payload() -> dict[str, dict[str, str]]:
    return {"error": {"code": INVALID_IMAGE_CODE, "message": INVALID_IMAGE_MESSAGE}}


def decode_uploaded_image(file: ReadableUpload | BinaryIO) -> Image.Image:
    """Decode an uploaded image as RGB and fail before route services run.

    The upload stream is consumed once into memory, then PIL validation is
    forced with ``verify()``. Because ``verify()`` invalidates the PIL image
    object, the bytes are reopened from a fresh ``BytesIO`` before converting
    to RGB and forcing pixel load.
    """
    raw = file.read()
    try:
        with Image.open(io.BytesIO(raw)) as probe:
            probe.verify()

        with Image.open(io.BytesIO(raw)) as image:
            rgb = image.convert("RGB")
            rgb.load()
            return rgb
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError(INVALID_IMAGE_MESSAGE) from exc

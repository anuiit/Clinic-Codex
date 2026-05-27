"""Single source of truth: backend/annotations/<analysis_id>/. No writes to training_data/."""
from __future__ import annotations

import base64
import errno
import io
import json
import os
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from PIL import Image


class AnnotationStorageError(Exception):
    """Base class for annotation storage errors."""


class AnnotationPermissionError(AnnotationStorageError, PermissionError):
    """Raised when the annotations directory is not writable."""


class AnnotationDiskFullError(AnnotationStorageError, OSError):
    """Raised when the disk has no space left."""


def sanitize_class_name(name: str) -> str:
    name = name.strip()
    if not name:
        raise ValueError("class_name is empty after strip")
    if "/" in name:
        raise ValueError("class_name contains '/'")
    if "\\" in name:
        raise ValueError("class_name contains '\\'")
    if ".." in name:
        raise ValueError("class_name contains '..'")
    if "\x00" in name:
        raise ValueError("class_name contains null byte")
    return name


def decode_image_data_url(data_url: str) -> Image.Image:
    if data_url.startswith("data:"):
        comma = data_url.find(",")
        if comma == -1:
            raise ValueError("malformed data URL: no comma found")
        b64 = data_url[comma + 1:]
    else:
        b64 = data_url
    try:
        raw = base64.b64decode(b64, validate=True)
    except Exception as exc:
        raise ValueError(f"base64 decode failed: {exc}") from exc
    try:
        return Image.open(io.BytesIO(raw))
    except Exception as exc:
        raise ValueError(f"PIL could not open image: {exc}") from exc


def clamp_bbox(
    bbox: tuple[int, int, int, int],
    img_w: int,
    img_h: int,
) -> tuple[int, int, int, int]:
    x, y, w, h = bbox
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = min(w, img_w - x)
    h = min(h, img_h - y)
    if w <= 0:
        raise ValueError(f"clamped width is {w} <= 0")
    if h <= 0:
        raise ValueError(f"clamped height is {h} <= 0")
    return (x, y, w, h)


def normalize_bbox_to_int_pixels(bbox: Sequence[int | float]) -> tuple[int, int, int, int]:
    """Normalize frontend bbox values to integer pixels before crop/save.

    Phase 0 intentionally uses Python's documented ``round()`` behavior for
    every coordinate and dimension, including banker-rounding ties.
    """
    if len(bbox) != 4:
        raise ValueError("bbox must be [x, y, w, h]")
    x, y, w, h = bbox
    return (int(round(x)), int(round(y)), int(round(w)), int(round(h)))


_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")


def save_annotation(
    analysis_id: str,
    image: Image.Image,
    annotations: list[dict],
    base_dir: Path,
    elements_dir: Path,  # kept for backward compat — not used
) -> dict:
    if not _SAFE_ID.match(analysis_id):
        raise ValueError(
            f"analysis_id '{analysis_id}' must be alphanumeric/dash/underscore only"
        )

    target_dir = base_dir / analysis_id
    tmp_dir = base_dir / f".tmp-{analysis_id}-{os.getpid()}-{uuid.uuid4().hex}"

    try:
        (tmp_dir / "elements").mkdir(parents=True)

        img_w, img_h = image.size
        image.save(tmp_dir / "image.png", format="PNG")

        saved_annotations = []
        classes_seen: set[str] = set()

        for ann in annotations:
            idx = ann["index"]
            raw_class = ann["class_name"]
            bbox_raw = ann["bbox"]

            cls = sanitize_class_name(raw_class)
            normalized_bbox = normalize_bbox_to_int_pixels(bbox_raw)
            x, y, w, h = clamp_bbox(normalized_bbox, img_w, img_h)
            crop = image.crop((x, y, x + w, y + h))
            crop_filename = f"{idx}.png"
            crop.save(tmp_dir / "elements" / crop_filename, format="PNG")

            classes_seen.add(cls)
            saved_annotations.append(
                {
                    "index": idx,
                    "class_name": cls,
                    "bbox": [x, y, w, h],
                    "crop_path": str(target_dir / "elements" / crop_filename),
                }
            )

        metadata = {
            "analysis_id": analysis_id,
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
            "annotations": saved_annotations,
        }
        (tmp_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.move(str(tmp_dir), str(target_dir))

    except PermissionError as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise AnnotationPermissionError(str(e)) from e
    except OSError as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        if e.errno == errno.ENOSPC:
            raise AnnotationDiskFullError(str(e)) from e
        raise
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    return {
        "status": "ok",
        "analysis_id": analysis_id,
        "saved_count": len(saved_annotations),
        "classes": sorted(classes_seen),
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }

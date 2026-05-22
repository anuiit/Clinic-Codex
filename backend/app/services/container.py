"""Lazy injectable services for route handlers."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from backend.app.config import Settings

try:
    from backend.services.annotation_storage import decode_image_data_url, save_annotation
except ImportError:  # pragma: no cover - compatibility when backend dir is sys.path root
    from services.annotation_storage import decode_image_data_url, save_annotation  # type: ignore


def _ensure_backend_root_on_path(settings: Settings) -> None:
    backend_root = str(settings.backend_root)
    if backend_root not in sys.path:
        sys.path.insert(0, backend_root)


class DefaultServices:
    """Default production services.

    Heavy ML imports are intentionally inside lazy providers so importing the
    app factory and constructing the default Flask app remain cheap.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._classifier = None
        self._segmenter = None
        self._sample_index: dict[str, list[dict[str, str]]] | None = None

    def get_classifier(self):
        if self._classifier is None:
            _ensure_backend_root_on_path(self.settings)
            from codex_model import CodexClassifier

            if self.settings.model_dir:
                self._classifier = CodexClassifier(model_dir=self.settings.model_dir)
            else:
                self._classifier = CodexClassifier()
        return self._classifier

    def classify(self, image: Image.Image | np.ndarray, **kwargs):
        return self.get_classifier().classify(image, **kwargs)

    def classify_batch(self, images: list[Image.Image | np.ndarray]):
        return self.get_classifier().classify_batch(images)

    def get_segmenter(self):
        if self._segmenter is None:
            _ensure_backend_root_on_path(self.settings)
            from codex_pipeline.segmentation import MobileSAMSegmenter

            self._segmenter = MobileSAMSegmenter(points_per_side=16)
        return self._segmenter

    def segment_page(self, image: np.ndarray):
        segmenter = self.get_segmenter()
        proposals = segmenter.segment_page(image)
        return segmenter.extract_crops(image, proposals)

    def load_classes(self) -> dict[str, Any]:
        with self.settings.class_config_path.open() as f:
            return json.load(f)

    def save_annotation(self, analysis_id: str, image, annotations: list[dict]) -> dict[str, Any]:
        return save_annotation(
            analysis_id,
            image,
            annotations,
            base_dir=self.settings.annotations_dir,
            elements_dir=self.settings.elements_dir,
        )

    def decode_annotation_image(self, data_url: str):
        return decode_image_data_url(data_url)

    def sample_index(self) -> dict[str, list[dict[str, str]]]:
        if self._sample_index is None:
            self._sample_index = build_sample_index(self.settings.data_dir)
        return self._sample_index


def sample_class_name(class_dir_name: str) -> str:
    if class_dir_name.endswith("-glyph"):
        return class_dir_name[: -len("-glyph")]
    if "-" in class_dir_name:
        prefix, remainder = class_dir_name.split("-", 1)
        if prefix.isdigit() and remainder:
            return remainder
    return class_dir_name


def build_sample_index(data_dir: Path) -> dict[str, list[dict[str, str]]]:
    sample_index: dict[str, list[dict[str, str]]] = {}
    valid_suffixes = {".jpg", ".jpeg", ".png", ".bmp"}
    for subdir in ["elements_sample", "glyphs_sample"]:
        sample_dir = data_dir / subdir
        if not sample_dir.exists():
            continue
        for class_dir in sample_dir.iterdir():
            if not class_dir.is_dir():
                continue
            class_name = sample_class_name(class_dir.name)
            sample_index.setdefault(class_name, [])
            for image_path in sorted(class_dir.iterdir()):
                if image_path.is_file() and image_path.suffix.lower() in valid_suffixes:
                    sample_index[class_name].append({"path": str(image_path), "class_name": class_name})
    return sample_index

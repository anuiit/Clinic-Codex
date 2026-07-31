"""Lazy injectable services for route handlers."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from backend.app.config import Settings
from backend.app.errors import ModelAssetUnavailable

try:
    from backend.services.annotation_storage import decode_image_data_url, save_annotation
    from backend.services.annotation_review import AnnotationReviewStore
    from backend.services.training_jobs import AdminTrainingService, RequestLaunchContext
except ImportError:  # pragma: no cover - compatibility when backend dir is sys.path root
    from services.annotation_storage import decode_image_data_url, save_annotation  # type: ignore
    from services.annotation_review import AnnotationReviewStore  # type: ignore
    from services.training_jobs import AdminTrainingService, RequestLaunchContext  # type: ignore


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
        self._annotation_review_store: AnnotationReviewStore | None = None
        self._admin_training_service: AdminTrainingService | None = None
        self._sample_index: dict[str, list[dict[str, str]]] | None = None

    def get_classifier(self):
        self._raise_for_missing_classifier_assets()
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
        self._raise_for_missing_mobile_sam_checkpoint()
        if self._segmenter is None:
            _ensure_backend_root_on_path(self.settings)
            from codex_pipeline.segmentation import MobileSAMSegmenter

            self._segmenter = MobileSAMSegmenter(
                checkpoint_path=str(self.settings.mobile_sam_checkpoint_path),
                points_per_side=16,
            )
        return self._segmenter

    def segment_page(self, image: np.ndarray):
        segmenter = self.get_segmenter()
        proposals = segmenter.segment_page(image)
        return segmenter.extract_crops(image, proposals)

    def load_classes(self) -> dict[str, Any]:
        with self.settings.class_config_path.open() as f:
            return json.load(f)

    def save_annotation(self, analysis_id: str, image, annotations: list[dict], *, author_id: str | None = None) -> dict[str, Any]:
        return save_annotation(
            analysis_id,
            image,
            annotations,
            base_dir=self.settings.annotations_dir,
            elements_dir=self.settings.elements_dir,
            author_id=author_id,
        )

    def decode_annotation_image(self, data_url: str):
        return decode_image_data_url(
            data_url,
            max_pixels=self.settings.max_image_pixels,
            max_dimension=self.settings.max_image_dimension,
        )

    def annotation_review_store(self) -> AnnotationReviewStore:
        if self._annotation_review_store is None:
            self._annotation_review_store = AnnotationReviewStore(self.settings.annotations_dir)
        return self._annotation_review_store

    def list_annotation_reviews(self) -> dict[str, Any]:
        return self.annotation_review_store().list_queue()

    def set_annotation_review_status(
        self,
        analysis_id: str,
        index: int,
        status: str,
        *, reviewer_id: str | None = None,
    ) -> dict[str, Any]:
        return self.annotation_review_store().set_status(analysis_id, index, status, reviewer_id=reviewer_id)

    def modify_annotation_review_element(
        self,
        analysis_id: str,
        index: int,
        *,
        class_name: str,
        bbox: list[int | float],
        status: str = "pending",
        reviewer_id: str | None = None,
    ) -> dict[str, Any]:
        return self.annotation_review_store().modify_element(
            analysis_id,
            index,
            class_name=class_name,
            bbox=bbox,
            status=status,
            reviewer_id=reviewer_id,
        )

    def iter_approved_annotations(self):
        return self.annotation_review_store().iter_approved_annotations()

    def annotation_review_image_path(self, analysis_id: str):
        return self.annotation_review_store().image_path_for(analysis_id)

    def annotation_review_crop_path(self, analysis_id: str, index: int):
        return self.annotation_review_store().crop_path_for(analysis_id, index)

    def admin_training_service(self) -> AdminTrainingService:
        if self._admin_training_service is None:
            self._admin_training_service = AdminTrainingService(
                self.settings,
                self.annotation_review_store(),
            )
        return self._admin_training_service

    def admin_training_summary(self, context: RequestLaunchContext):
        return self.admin_training_service().summary(context)

    def latest_admin_training_job(self):
        return self.admin_training_service().latest_job()

    def get_admin_training_job(self, run_id: str):
        return self.admin_training_service().get_job(run_id)

    def start_admin_training_job(self, payload: dict[str, Any], context: RequestLaunchContext, *, actor_id: str | None = None):
        return self.admin_training_service().start_job(payload, context, actor_id=actor_id)

    def sample_index(self) -> dict[str, list[dict[str, str]]]:
        if self._sample_index is None:
            self._sample_index = build_sample_index(self.settings.data_dir)
        return self._sample_index

    def readiness(self) -> dict[str, Any]:
        return readiness_report(self.settings)

    def _raise_for_missing_classifier_assets(self) -> None:
        for check in readiness_report(self.settings)["checks"]:
            if check["name"].startswith("classifier_") and not check["available"]:
                raise ModelAssetUnavailable(
                    asset=check["name"],
                    path=check["path"],
                    hint=check["hint"],
                )

    def _raise_for_missing_mobile_sam_checkpoint(self) -> None:
        check = mobile_sam_checkpoint_check(self.settings)
        if not check["available"]:
            raise ModelAssetUnavailable(
                asset=check["name"],
                path=check["path"],
                hint=check["hint"],
            )


def _asset_check(name: str, path: Path, hint: str) -> dict[str, Any]:
    expanded = path.expanduser()
    return {
        "name": name,
        "available": expanded.is_file(),
        "path": str(expanded),
        "hint": hint,
    }


def classifier_asset_checks(settings: Settings) -> list[dict[str, Any]]:
    weights_dir = settings.classifier_weights_dir
    hint = "Prepare classifier weights with prototypes.pt and projection.pt before starting ML requests."
    return [
        _asset_check("classifier_prototypes", weights_dir / "prototypes.pt", hint),
        _asset_check("classifier_projection", weights_dir / "projection.pt", hint),
    ]


def mobile_sam_checkpoint_check(settings: Settings) -> dict[str, Any]:
    return _asset_check(
        "mobile_sam_checkpoint",
        settings.mobile_sam_checkpoint_path,
        "Place mobile_sam.pt at this path or set MOBILE_SAM_CHECKPOINT to a local checkpoint before segmentation.",
    )


def readiness_report(settings: Settings) -> dict[str, Any]:
    checks = [*classifier_asset_checks(settings), mobile_sam_checkpoint_check(settings)]
    ready = all(check["available"] for check in checks)
    return {"status": "ready" if ready else "not_ready", "ready": ready, "checks": checks}


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

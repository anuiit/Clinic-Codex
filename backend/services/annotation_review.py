"""Local admin review state for submitted annotations.

Submitted annotations under ``backend/annotations/<analysis_id>/`` remain the
canonical source for crops, labels, and boxes.  This module keeps the separate
operator review decision manifest that gates classifier retraining eligibility.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from PIL import Image

try:
    from backend.services.annotation_storage import (
        clamp_bbox,
        normalize_bbox_to_int_pixels,
        sanitize_class_name,
    )
except ImportError:  # pragma: no cover - compatibility when backend dir is sys.path root
    from services.annotation_storage import (  # type: ignore
        clamp_bbox,
        normalize_bbox_to_int_pixels,
        sanitize_class_name,
    )


REVIEW_STATUSES = {"pending", "approved", "rejected"}
TRAINABLE_STATUS = "approved"
DATASET_SPLITS = {"train", "val", "test", "excluded"}
TRAINABLE_DATASET_SPLITS = {"train", "val", "test"}
SPLIT_REASONS = {
    "trainable_hash_80_10_10",
    "pending_review",
    "rejected_review",
    "missing_crop",
    "stale_decision",
}
MANIFEST_SCHEMA_VERSION = 1
MANIFEST_FILENAME = "review-index.json"
LOCAL_ONLY_WARNING = (
    "Local/dev-only annotation review endpoint. It is not production-secured; "
    "only use it with locally bound backend/frontend services."
)

_SAFE_ANALYSIS_ID = re.compile(r"^[A-Za-z0-9_-]+$")


class AnnotationReviewError(Exception):
    """Base class for annotation review errors."""


class AnnotationReviewValidationError(AnnotationReviewError, ValueError):
    """Raised for invalid review mutation input."""


class AnnotationReviewNotFoundError(AnnotationReviewError, LookupError):
    """Raised when a canonical submitted annotation cannot be found."""


def review_key(analysis_id: str, index: int) -> str:
    return f"{analysis_id}:{index}"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _empty_manifest() -> dict[str, Any]:
    return {"schema_version": MANIFEST_SCHEMA_VERSION, "decisions": {}}


def _diagnostic(
    code: str,
    message: str,
    *,
    analysis_id: str | None = None,
    index: int | None = None,
    key: str | None = None,
) -> dict[str, Any]:
    data: dict[str, Any] = {"code": code, "message": message}
    if analysis_id is not None:
        data["analysis_id"] = analysis_id
    if index is not None:
        data["index"] = index
    if key is not None:
        data["key"] = key
    return data


def _validate_analysis_id(analysis_id: str) -> None:
    if not isinstance(analysis_id, str) or not analysis_id:
        raise AnnotationReviewValidationError("analysis_id must be a non-empty string")
    if not _SAFE_ANALYSIS_ID.match(analysis_id):
        raise AnnotationReviewValidationError(
            "analysis_id must be alphanumeric/dash/underscore only"
        )


def _validate_index(index: int) -> None:
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise AnnotationReviewValidationError("index must be a non-negative integer")


def _validate_status(status: str) -> None:
    if status not in REVIEW_STATUSES:
        allowed = ", ".join(sorted(REVIEW_STATUSES))
        raise AnnotationReviewValidationError(f"status must be one of: {allowed}")


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _resolve_under_analysis_dir(path_value: object, analysis_dir: Path, fallback: Path) -> Path:
    if isinstance(path_value, str) and path_value:
        candidate = Path(path_value)
        if candidate.is_absolute() and candidate.exists() and _is_relative_to(candidate, analysis_dir):
            return candidate
        if not candidate.is_absolute():
            relative = analysis_dir / candidate
            if relative.exists() and _is_relative_to(relative, analysis_dir):
                return relative
    return fallback


def _path_payload(path: Path) -> dict[str, Any]:
    exists = path.is_file()
    payload: dict[str, Any] = {"path": str(path), "exists": exists}
    if exists:
        stat = path.stat()
        payload["size"] = stat.st_size
        payload["mtime_ns"] = stat.st_mtime_ns
    return payload


def _source_fingerprint(
    *,
    analysis_id: str,
    uploaded_at: str | None,
    annotation: dict[str, Any],
    crop_path: Path,
) -> str:
    payload = {
        "analysis_id": analysis_id,
        "index": annotation.get("index"),
        "uploaded_at": uploaded_at,
        "class_name": annotation.get("class_name"),
        "bbox": annotation.get("bbox"),
        "crop": _path_payload(crop_path),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _dataset_split_for_key(key: str) -> str:
    """Return the deterministic 80/10/10 split for an immutable review key."""
    bucket = int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "val"
    return "test"


def _excluded_split_reason(
    *,
    review_status: str,
    crop_exists: bool,
    stale_decision: bool,
) -> str:
    if stale_decision:
        return "stale_decision"
    if not crop_exists:
        return "missing_crop"
    if review_status == "rejected":
        return "rejected_review"
    return "pending_review"


def _split_payload_for_element(
    *,
    key: str,
    review_status: str,
    crop_exists: bool,
    stale_decision: bool,
    decision: dict[str, Any] | None = None,
) -> dict[str, str]:
    if review_status == TRAINABLE_STATUS and crop_exists and not stale_decision:
        persisted_split = (decision or {}).get("dataset_split")
        dataset_split = (
            persisted_split
            if persisted_split in TRAINABLE_DATASET_SPLITS
            else _dataset_split_for_key(key)
        )
        return {
            "dataset_split": dataset_split,
            "split_reason": "trainable_hash_80_10_10",
        }
    return {
        "dataset_split": "excluded",
        "split_reason": _excluded_split_reason(
            review_status=review_status,
            crop_exists=crop_exists,
            stale_decision=stale_decision,
        ),
    }


class AnnotationReviewStore:
    """File-backed element-level review manifest over submitted annotations."""

    def __init__(self, annotations_dir: Path, manifest_path: Path | None = None):
        self.annotations_dir = Path(annotations_dir)
        self.manifest_path = Path(manifest_path) if manifest_path else self.annotations_dir / MANIFEST_FILENAME

    def list_queue(self) -> dict[str, Any]:
        manifest = self._load_manifest()
        decisions = manifest["decisions"]
        analyses: list[dict[str, Any]] = []
        diagnostics: list[dict[str, Any]] = []
        seen_keys: set[str] = set()
        counts = {"total": 0, "pending": 0, "approved": 0, "rejected": 0, "trainable": 0}

        if not self.annotations_dir.exists():
            self.annotations_dir.mkdir(parents=True, exist_ok=True)

        for analysis_dir in sorted(self.annotations_dir.iterdir(), key=lambda p: p.name):
            if not analysis_dir.is_dir() or analysis_dir.name.startswith("."):
                continue
            analysis = self._read_analysis(analysis_dir, decisions, diagnostics, seen_keys)
            if analysis is None:
                continue
            analyses.append(analysis)
            for element in analysis["elements"]:
                counts["total"] += 1
                counts[element["review_status"]] += 1
                if element["trainable"]:
                    counts["trainable"] += 1

        for key, decision in sorted(decisions.items()):
            if key not in seen_keys:
                diagnostics.append(
                    _diagnostic(
                        "orphan_decision",
                        "Review decision has no matching canonical annotation and is not trainable.",
                        analysis_id=decision.get("analysis_id"),
                        index=decision.get("index"),
                        key=key,
                    )
                )

        return {
            "status": "ok",
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "local_only": True,
            "warning": LOCAL_ONLY_WARNING,
            "counts": counts,
            "analyses": analyses,
            "diagnostics": diagnostics,
        }

    def set_status(self, analysis_id: str, index: int, status: str) -> dict[str, Any]:
        _validate_analysis_id(analysis_id)
        _validate_index(index)
        _validate_status(status)

        queue = self.list_queue()
        element = self._find_element(queue["analyses"], analysis_id, index)
        if element is None:
            raise AnnotationReviewNotFoundError(
                f"annotation element not found: {analysis_id}:{index}"
            )

        manifest = self._load_manifest()
        key = review_key(analysis_id, index)
        split_payload = _split_payload_for_element(
            key=key,
            review_status=status,
            crop_exists=bool(element.get("crop_exists")),
            stale_decision=False,
            decision=manifest["decisions"].get(key)
            if isinstance(manifest["decisions"].get(key), dict)
            else None,
        )
        manifest["decisions"][key] = {
            "analysis_id": analysis_id,
            "index": index,
            "status": status,
            "reviewed_at": utc_now_iso(),
            "source_fingerprint": element["source_fingerprint"],
            "class_name": element["class_name"],
            "bbox": element["bbox"],
            **split_payload,
        }
        self._save_manifest(manifest)
        refreshed = self._find_element(self.list_queue()["analyses"], analysis_id, index)
        return {
            "status": "ok",
            "local_only": True,
            "warning": LOCAL_ONLY_WARNING,
            "element": refreshed,
        }

    def modify_element(
        self,
        analysis_id: str,
        index: int,
        *,
        class_name: str,
        bbox: Sequence[int | float],
        status: str = "pending",
    ) -> dict[str, Any]:
        """Modify canonical class/bbox data and record a fresh review decision.

        The submitted annotation metadata remains the source of truth.  A modified
        crop is written to a new filename before metadata is atomically replaced,
        so an interrupted write cannot leave metadata pointing at a missing or
        half-written crop.  Modified elements default to pending; callers may ask
        for an immediate approved/rejected final status only after the fresh crop
        is available.
        """
        _validate_analysis_id(analysis_id)
        _validate_index(index)
        _validate_status(status)

        if not isinstance(class_name, str):
            raise AnnotationReviewValidationError("class_name must be a string")

        try:
            sanitized_class = sanitize_class_name(class_name)
            normalized_bbox = normalize_bbox_to_int_pixels(bbox)
        except Exception as exc:
            raise AnnotationReviewValidationError(str(exc)) from exc

        analysis_dir = self.annotations_dir / analysis_id
        metadata_path = analysis_dir / "metadata.json"
        image_path = analysis_dir / "image.png"
        if not metadata_path.is_file():
            raise AnnotationReviewNotFoundError(f"annotation metadata not found: {analysis_id}")
        if not image_path.is_file():
            raise AnnotationReviewNotFoundError(f"annotation image not found: {analysis_id}")

        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise AnnotationReviewValidationError(f"metadata.json is invalid JSON: {exc}") from exc

        metadata_analysis_id = metadata.get("analysis_id", analysis_id)
        if metadata_analysis_id != analysis_id:
            raise AnnotationReviewValidationError("metadata analysis_id does not match requested analysis_id")
        annotations = metadata.get("annotations")
        if not isinstance(annotations, list):
            raise AnnotationReviewValidationError("metadata annotations must be a list")

        target_position: int | None = None
        existing_annotation: dict[str, Any] | None = None
        for position, annotation in enumerate(annotations):
            if isinstance(annotation, dict) and annotation.get("index") == index:
                target_position = position
                existing_annotation = dict(annotation)
                break
        if target_position is None or existing_annotation is None:
            raise AnnotationReviewNotFoundError(f"annotation element not found: {analysis_id}:{index}")

        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                x, y, w, h = clamp_bbox(normalized_bbox, *image.size)
                crop = image.crop((x, y, x + w, y + h))
                elements_dir = analysis_dir / "elements"
                elements_dir.mkdir(parents=True, exist_ok=True)
                crop_filename = f"{index}-{uuid.uuid4().hex}.png"
                final_crop_path = elements_dir / crop_filename
                tmp_crop_path = elements_dir / f".{crop_filename}.tmp-{os.getpid()}"
                crop.save(tmp_crop_path, format="PNG")
                os.replace(tmp_crop_path, final_crop_path)
        except ValueError as exc:
            raise AnnotationReviewValidationError(str(exc)) from exc

        old_crop_path = _resolve_under_analysis_dir(
            existing_annotation.get("crop_path"),
            analysis_dir,
            analysis_dir / "elements" / f"{index}.png",
        )
        updated_annotation = {
            **existing_annotation,
            "class_name": sanitized_class,
            "bbox": [x, y, w, h],
            "crop_path": str(final_crop_path),
        }
        updated_annotations = list(annotations)
        updated_annotations[target_position] = updated_annotation
        updated_metadata = {**metadata, "annotations": updated_annotations}
        tmp_metadata_path = metadata_path.with_name(
            f".{metadata_path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
        )
        tmp_metadata_path.write_text(
            json.dumps(updated_metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp_metadata_path, metadata_path)

        if old_crop_path != final_crop_path and old_crop_path.is_file() and _is_relative_to(old_crop_path, analysis_dir):
            try:
                old_crop_path.unlink()
            except OSError:
                pass

        refreshed_queue = self.list_queue()
        element = self._find_element(refreshed_queue["analyses"], analysis_id, index)
        if element is None:
            raise AnnotationReviewNotFoundError(f"annotation element not found after modify: {analysis_id}:{index}")

        manifest = self._load_manifest()
        key = review_key(analysis_id, index)
        split_payload = _split_payload_for_element(
            key=key,
            review_status=status,
            crop_exists=bool(element.get("crop_exists")),
            stale_decision=False,
            decision=manifest["decisions"].get(key)
            if isinstance(manifest["decisions"].get(key), dict)
            else None,
        )
        manifest["decisions"][key] = {
            "analysis_id": analysis_id,
            "index": index,
            "status": status,
            "reviewed_at": utc_now_iso(),
            "source_fingerprint": element["source_fingerprint"],
            "class_name": element["class_name"],
            "bbox": element["bbox"],
            **split_payload,
        }
        self._save_manifest(manifest)
        final_queue = self.list_queue()
        refreshed = self._find_element(final_queue["analyses"], analysis_id, index)
        return {
            "status": "ok",
            "local_only": True,
            "warning": LOCAL_ONLY_WARNING,
            "element": refreshed,
            "counts": final_queue["counts"],
        }

    def iter_approved_annotations(self) -> Iterable[dict[str, Any]]:
        """Yield canonical annotations that are exactly approved and trainable."""
        queue = self.list_queue()
        for analysis in queue["analyses"]:
            for element in analysis["elements"]:
                if element["trainable"]:
                    yield {
                        "analysis_id": element["analysis_id"],
                        "index": element["index"],
                        "class_name": element["class_name"],
                        "bbox": element["bbox"],
                        "crop_path": element["crop_path"],
                        "image_path": analysis["image_path"],
                        "uploaded_at": analysis.get("uploaded_at"),
                        "source_fingerprint": element["source_fingerprint"],
                        "dataset_split": element["dataset_split"],
                    }

    def image_path_for(self, analysis_id: str) -> Path:
        _validate_analysis_id(analysis_id)
        queue = self.list_queue()
        for analysis in queue["analyses"]:
            if analysis.get("analysis_id") == analysis_id and analysis.get("image_exists"):
                return Path(analysis["image_path"])
        raise AnnotationReviewNotFoundError(f"annotation image not found: {analysis_id}")

    def crop_path_for(self, analysis_id: str, index: int) -> Path:
        _validate_analysis_id(analysis_id)
        _validate_index(index)
        queue = self.list_queue()
        element = self._find_element(queue["analyses"], analysis_id, index)
        if element is not None and element.get("crop_exists"):
            return Path(element["crop_path"])
        raise AnnotationReviewNotFoundError(f"annotation crop not found: {analysis_id}:{index}")

    def _load_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            return _empty_manifest()
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise AnnotationReviewValidationError("review manifest must be a JSON object")
        decisions = manifest.get("decisions")
        if not isinstance(decisions, dict):
            decisions = {}
        return {
            "schema_version": manifest.get("schema_version", MANIFEST_SCHEMA_VERSION),
            "decisions": decisions,
        }

    def _save_manifest(self, manifest: dict[str, Any]) -> None:
        self.annotations_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "decisions": manifest.get("decisions", {}),
        }
        tmp_path = self.manifest_path.with_name(
            f".{self.manifest_path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
        )
        tmp_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp_path, self.manifest_path)

    def _read_analysis(
        self,
        analysis_dir: Path,
        decisions: dict[str, Any],
        diagnostics: list[dict[str, Any]],
        seen_keys: set[str],
    ) -> dict[str, Any] | None:
        metadata_path = analysis_dir / "metadata.json"
        if not metadata_path.is_file():
            diagnostics.append(
                _diagnostic(
                    "missing_metadata",
                    "Annotation folder has no metadata.json and cannot be reviewed.",
                    analysis_id=analysis_dir.name,
                )
            )
            return None

        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            diagnostics.append(
                _diagnostic(
                    "invalid_metadata",
                    f"metadata.json is invalid JSON: {exc}",
                    analysis_id=analysis_dir.name,
                )
            )
            return None

        analysis_id = str(metadata.get("analysis_id") or analysis_dir.name)
        if not _SAFE_ANALYSIS_ID.match(analysis_id):
            diagnostics.append(
                _diagnostic(
                    "invalid_metadata",
                    "metadata.json analysis_id must be alphanumeric/dash/underscore only.",
                    analysis_id=analysis_dir.name,
                )
            )
            return None
        uploaded_at = metadata.get("uploaded_at")
        image_path = analysis_dir / "image.png"
        annotations = metadata.get("annotations", [])
        if not isinstance(annotations, list):
            diagnostics.append(
                _diagnostic(
                    "invalid_metadata",
                    "metadata.json annotations must be a list.",
                    analysis_id=analysis_id,
                )
            )
            return None

        elements: list[dict[str, Any]] = []
        seen_indexes: set[int] = set()
        for position, annotation in enumerate(annotations):
            if not isinstance(annotation, dict):
                diagnostics.append(
                    _diagnostic(
                        "invalid_annotation",
                        f"annotations[{position}] must be an object.",
                        analysis_id=analysis_id,
                    )
                )
                continue
            index = annotation.get("index")
            if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                diagnostics.append(
                    _diagnostic(
                        "invalid_annotation",
                        f"annotations[{position}].index must be a non-negative integer.",
                        analysis_id=analysis_id,
                    )
                )
                continue
            if index in seen_indexes:
                diagnostics.append(
                    _diagnostic(
                        "duplicate_index",
                        "Duplicate annotation index is ignored for review/training.",
                        analysis_id=analysis_id,
                        index=index,
                        key=review_key(analysis_id, index),
                    )
                )
                continue
            seen_indexes.add(index)

            key = review_key(analysis_id, index)
            seen_keys.add(key)
            fallback_crop = analysis_dir / "elements" / f"{index}.png"
            crop_path = _resolve_under_analysis_dir(
                annotation.get("crop_path"),
                analysis_dir,
                fallback_crop,
            )
            fingerprint = _source_fingerprint(
                analysis_id=analysis_id,
                uploaded_at=uploaded_at if isinstance(uploaded_at, str) else None,
                annotation=annotation,
                crop_path=crop_path,
            )
            decision = decisions.get(key) if isinstance(decisions.get(key), dict) else None
            review_status = "pending"
            stale_decision = False
            if decision:
                decision_status = decision.get("status")
                if decision_status in REVIEW_STATUSES:
                    if decision.get("source_fingerprint") == fingerprint:
                        review_status = decision_status
                    else:
                        stale_decision = True
                        diagnostics.append(
                            _diagnostic(
                                "stale_decision",
                                "Review decision source fingerprint no longer matches canonical annotation; treating as pending and not trainable.",
                                analysis_id=analysis_id,
                                index=index,
                                key=key,
                            )
                        )

            crop_exists = crop_path.is_file()
            if not crop_exists:
                diagnostics.append(
                    _diagnostic(
                        "missing_crop",
                        "Canonical crop file is missing; element is not trainable.",
                        analysis_id=analysis_id,
                        index=index,
                        key=key,
                    )
                )

            trainable = review_status == TRAINABLE_STATUS and crop_exists and not stale_decision
            split_payload = _split_payload_for_element(
                key=key,
                review_status=review_status,
                crop_exists=crop_exists,
                stale_decision=stale_decision,
                decision=decision,
            )
            elements.append(
                {
                    "key": key,
                    "analysis_id": analysis_id,
                    "index": index,
                    "class_name": annotation.get("class_name", ""),
                    "bbox": annotation.get("bbox", []),
                    "crop_path": str(crop_path),
                    "crop_url": f"/admin/annotations/{analysis_id}/{index}/crop",
                    "crop_exists": crop_exists,
                    "review_status": review_status,
                    "trainable": trainable,
                    "source_fingerprint": fingerprint,
                    "stale_decision": stale_decision,
                    **split_payload,
                }
            )

        return {
            "analysis_id": analysis_id,
            "uploaded_at": uploaded_at,
            "image_path": str(image_path),
            "image_url": f"/admin/annotations/{analysis_id}/image",
            "image_exists": image_path.is_file(),
            "elements": elements,
        }

    @staticmethod
    def _find_element(
        analyses: list[dict[str, Any]],
        analysis_id: str,
        index: int,
    ) -> dict[str, Any] | None:
        for analysis in analyses:
            if analysis.get("analysis_id") != analysis_id:
                continue
            for element in analysis.get("elements", []):
                if element.get("index") == index:
                    return element
        return None

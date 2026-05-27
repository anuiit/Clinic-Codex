from __future__ import annotations

from flask import Blueprint, current_app, jsonify, request, send_file

from backend.services.annotation_review import (
    AnnotationReviewNotFoundError,
    AnnotationReviewValidationError,
)

bp = Blueprint("admin_annotations", __name__)


def _services():
    return current_app.extensions["clinic_services"]


def _error(message: str, status_code: int):
    return jsonify({"status": "error", "error": message}), status_code


@bp.get("/admin/annotations")
def list_admin_annotations():
    """Local/dev-only admin review queue.

    This endpoint intentionally exposes no auth claims and no retrain controls;
    it is for local operator review while services are bound locally.
    """
    return jsonify(_services().list_annotation_reviews()), 200


@bp.get("/admin/annotations/<analysis_id>/image")
def get_admin_annotation_image(analysis_id: str):
    try:
        return send_file(_services().annotation_review_image_path(analysis_id))
    except AnnotationReviewValidationError as exc:
        return _error(str(exc), 400)
    except AnnotationReviewNotFoundError as exc:
        return _error(str(exc), 404)


@bp.get("/admin/annotations/<analysis_id>/<int:index>/crop")
def get_admin_annotation_crop(analysis_id: str, index: int):
    try:
        return send_file(_services().annotation_review_crop_path(analysis_id, index))
    except AnnotationReviewValidationError as exc:
        return _error(str(exc), 400)
    except AnnotationReviewNotFoundError as exc:
        return _error(str(exc), 404)


@bp.post("/admin/annotations/<analysis_id>/<int:index>/review")
def set_admin_annotation_review(analysis_id: str, index: int):
    data = request.get_json(force=True, silent=True)
    if data is None:
        return _error("invalid JSON", 400)

    status = data.get("status")
    if not isinstance(status, str):
        return _error("missing field: status", 400)

    try:
        result = _services().set_annotation_review_status(analysis_id, index, status)
    except AnnotationReviewValidationError as exc:
        return _error(str(exc), 400)
    except AnnotationReviewNotFoundError as exc:
        return _error(str(exc), 404)

    return jsonify(result), 200


@bp.post("/admin/annotations/<analysis_id>/<int:index>/modify")
def modify_admin_annotation_element(analysis_id: str, index: int):
    data = request.get_json(force=True, silent=True)
    if data is None:
        return _error("invalid JSON", 400)

    class_name = data.get("class_name")
    bbox = data.get("bbox")
    if not isinstance(class_name, str):
        return _error("missing field: class_name", 400)
    if not isinstance(bbox, list):
        return _error("missing field: bbox", 400)

    approve_after_save = data.get("approve_after_save") is True
    requested_status = data.get("status")
    if requested_status is not None and not isinstance(requested_status, str):
        return _error("status must be a string when provided", 400)
    status = requested_status or ("approved" if approve_after_save else "pending")
    if requested_status and approve_after_save and requested_status != "approved":
        return _error("approve_after_save conflicts with status", 400)

    try:
        result = _services().modify_annotation_review_element(
            analysis_id,
            index,
            class_name=class_name,
            bbox=bbox,
            status=status,
        )
    except AnnotationReviewValidationError as exc:
        return _error(str(exc), 400)
    except AnnotationReviewNotFoundError as exc:
        return _error(str(exc), 404)

    return jsonify(result), 200

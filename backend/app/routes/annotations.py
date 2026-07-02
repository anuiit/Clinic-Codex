from __future__ import annotations

from flask import Blueprint, current_app, jsonify, request

from backend.app.errors import annotation_error_response
from backend.services.annotation_storage import sanitize_class_name

bp = Blueprint("annotations", __name__)


def _services():
    return current_app.extensions["clinic_services"]


def _validation_error(message: str):
    return jsonify({"status": "error", "error_code": "VALIDATION_ERROR", "message": message, "error": message}), 400


def _validate_annotation_payload(annotations: list[object]) -> str | None:
    seen_indexes: set[int] = set()

    for position, annotation in enumerate(annotations):
        prefix = f"annotations[{position}]"
        if not isinstance(annotation, dict):
            return f"{prefix} must be an object"

        if "index" not in annotation:
            return f"{prefix}.index required"
        index = annotation["index"]
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            return f"{prefix}.index must be a non-negative integer"
        if index in seen_indexes:
            return f"{prefix}.index duplicates {index}"
        seen_indexes.add(index)

        if "class_name" not in annotation:
            return f"{prefix}.class_name required"
        class_name = annotation["class_name"]
        if not isinstance(class_name, str) or not class_name.strip():
            return f"{prefix}.class_name must be a non-empty string"
        try:
            sanitize_class_name(class_name)
        except ValueError as exc:
            return f"{prefix}.class_name invalid: {exc}"

        if "bbox" not in annotation:
            return f"{prefix}.bbox required"
        bbox = annotation["bbox"]
        if not isinstance(bbox, list) or len(bbox) != 4:
            return f"{prefix}.bbox must be [x, y, w, h]"
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in bbox):
            return f"{prefix}.bbox values must be numeric"
        if bbox[2] <= 0 or bbox[3] <= 0:
            return f"{prefix}.bbox width and height must be positive"

    return None


@bp.post("/save-annotation")
def save_annotation_route():
    data = request.get_json(force=True, silent=True)
    if data is None:
        return _validation_error("invalid JSON")

    for field in ("analysis_id", "image_data_url", "annotations"):
        if field not in data:
            return _validation_error(f"missing field: {field}")

    if not isinstance(data["annotations"], list) or len(data["annotations"]) == 0:
        return _validation_error("annotations must be a non-empty list")

    if not isinstance(data["analysis_id"], str) or not data["analysis_id"].strip():
        return _validation_error("missing field: analysis_id")

    annotation_error = _validate_annotation_payload(data["annotations"])
    if annotation_error:
        return _validation_error(annotation_error)

    services = _services()
    try:
        image = services.decode_annotation_image(data["image_data_url"])
    except ValueError as exc:
        return _validation_error(str(exc))

    try:
        result = services.save_annotation(data["analysis_id"], image, data["annotations"])
        return jsonify(result), 200
    except Exception as exc:  # Preserve Phase 0 storage/internal mappings.
        return annotation_error_response(exc)

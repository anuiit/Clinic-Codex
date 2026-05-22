from __future__ import annotations

from flask import Blueprint, current_app, jsonify, request

from backend.app.errors import annotation_error_response

bp = Blueprint("annotations", __name__)


def _services():
    return current_app.extensions["clinic_services"]


@bp.post("/save-annotation")
def save_annotation_route():
    data = request.get_json(force=True, silent=True)
    if data is None:
        return jsonify({"status": "error", "error": "invalid JSON"}), 400

    for field in ("analysis_id", "image_data_url", "annotations"):
        if field not in data:
            return jsonify({"status": "error", "error": f"missing field: {field}"}), 400

    if not isinstance(data["annotations"], list) or len(data["annotations"]) == 0:
        return jsonify({"status": "error", "error": "annotations must be a non-empty list"}), 400

    if not isinstance(data["analysis_id"], str) or not data["analysis_id"].strip():
        return jsonify({"status": "error", "error": "missing field: analysis_id"}), 400

    services = _services()
    try:
        image = services.decode_annotation_image(data["image_data_url"])
    except ValueError as exc:
        return jsonify({"status": "error", "error": str(exc)}), 400

    try:
        result = services.save_annotation(data["analysis_id"], image, data["annotations"])
        return jsonify(result), 200
    except Exception as exc:  # Preserve Phase 0 storage/internal mappings.
        return annotation_error_response(exc)

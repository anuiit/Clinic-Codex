from __future__ import annotations

import math
import os
from pathlib import Path

from flask import Blueprint, current_app, jsonify, request, send_file

from backend.app.services.crop_service import crop_bbox, decode_base64_image, validate_bbox

bp = Blueprint("similarity", __name__)
legacy_bp = Blueprint("legacy_similarity", __name__)


def _services():
    return current_app.extensions["clinic_services"]


def _settings():
    return current_app.config["CLINIC_SETTINGS"]


def _invalid_request():
    return jsonify({"error": {"code": "INVALID_REQUEST", "message": "image_base64 and bbox required"}}), 400


def _invalid_parameter(message: str):
    return jsonify({"error": {"code": "INVALID_REQUEST", "message": message}}), 400


def _bounded_positive_int(data, field: str, default: int, *, max_value: int = 50):
    value = default if data is None else data.get(field, default)
    if isinstance(value, bool):
        return None, _invalid_parameter(f"{field} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None, _invalid_parameter(f"{field} must be an integer")
    if parsed < 1 or parsed > max_value:
        return None, _invalid_parameter(f"{field} must be between 1 and {max_value}")
    return parsed, None


def _load_image_and_bbox(data, *, require_positive_bbox: bool = False):
    if not data or "image_base64" not in data or "bbox" not in data:
        return None, None, _invalid_request()
    try:
        img = decode_base64_image(data["image_base64"])
    except Exception as exc:
        return None, None, (jsonify({"error": {"code": "INVALID_IMAGE", "message": str(exc)}}), 400)
    try:
        bbox = validate_bbox(data["bbox"], img.size, require_positive=require_positive_bbox)
    except ValueError as exc:
        return None, None, (jsonify({"error": {"code": "INVALID_BBOX", "message": str(exc)}}), 400)
    return img, bbox, None


def _band(sim):
    if sim >= 0.60:
        return "high"
    if sim >= 0.35:
        return "moderate"
    return "low"


@bp.post("/similar")
def similar():
    data = request.get_json()
    img, bbox, error = _load_image_and_bbox(data, require_positive_bbox=True)
    if error:
        return error

    crop = crop_bbox(img, bbox)
    limit, limit_error = _bounded_positive_int(data, "limit", 5)
    if limit_error:
        return limit_error
    result = _services().classify(crop, top_k=limit)

    results = []
    for rank, item in enumerate(result.get("top_k", []), 1):
        results.append(
            {
                "rank": rank,
                "match_type": "class_prototype",
                "class_name": item["class_name"],
                "class_label": item.get("class_label"),
                "similarity": item["confidence"],
                "band": _band(item["confidence"]),
                "asset": None,
            }
        )

    return jsonify(
        {
            "query": {"bbox": bbox, "mode": "prototype"},
            "best_match": {
                "class_name": result["class_name"],
                "similarity": result["confidence"],
                "rejected": result["rejected"],
            },
            "results": results,
        }
    )


@bp.post("/trust")
def trust():
    data = request.get_json()
    img, bbox, error = _load_image_and_bbox(data, require_positive_bbox=True)
    if error:
        return error

    crop = crop_bbox(img, bbox)
    top_k, top_k_error = _bounded_positive_int(data, "top_k", 10)
    if top_k_error:
        return top_k_error
    result = _services().classify(crop, top_k=top_k)

    predicted_class = data.get("predicted_class", result["class_name"])
    top_k_list = result.get("top_k", [])

    predicted_rank = None
    predicted_sim = None
    for rank, item in enumerate(top_k_list, 1):
        if item["class_name"] == predicted_class:
            predicted_rank = rank
            predicted_sim = item["confidence"]
            break

    if predicted_rank is None:
        predicted_rank = -1
        predicted_sim = 0.0

    margin = 0.0
    if len(top_k_list) >= 2:
        margin = top_k_list[0]["confidence"] - top_k_list[1]["confidence"]

    entropy = 0.0
    if top_k_list:
        total = sum(item["confidence"] for item in top_k_list)
        if total > 0:
            probs = [item["confidence"] / total for item in top_k_list]
            entropy = -sum(p * math.log(p + 1e-10) for p in probs)

    return jsonify(
        {
            "query": {"bbox": bbox, "predicted_class": predicted_class},
            "trust": {
                "predicted_class_rank": predicted_rank,
                "predicted_class_similarity": predicted_sim,
                "top1_class": result["class_name"],
                "top1_similarity": result["confidence"],
                "margin_to_second": round(margin, 4),
                "above_rejection_threshold": result["confidence"] >= 0.35,
                "rejection_threshold": 0.35,
                "ambiguous": margin < 0.05,
                "entropy": round(entropy, 4),
                "top_k": top_k_list,
            },
        }
    )


@legacy_bp.get("/sample-image")
def sample_image():
    path = request.args.get("path")
    if not path:
        return jsonify({"error": "Image not found"}), 404

    data_dir = _settings().data_dir.resolve()
    resolved_path = Path(path).resolve()
    try:
        resolved_path.relative_to(data_dir)
    except ValueError:
        return jsonify({"error": "Image not found"}), 404

    if not os.path.exists(resolved_path):
        return jsonify({"error": "Image not found"}), 404

    return send_file(resolved_path)


@legacy_bp.post("/similar-samples")
def similar_samples():
    data = request.get_json()
    img, bbox, error = _load_image_and_bbox(data, require_positive_bbox=True)
    if error:
        return error

    crop = crop_bbox(img, bbox)
    result = _services().classify(crop, top_k=3)
    predicted_class = result["class_name"]

    limit = data.get("limit", 4)
    try:
        limit = max(1, int(limit))
    except (TypeError, ValueError):
        return jsonify({"error": {"code": "INVALID_REQUEST", "message": "limit must be an integer"}}), 400

    samples = _services().sample_index().get(predicted_class, [])[:limit]
    exemplars = [
        {
            "image_url": f"/sample-image?path={sample['path']}",
            "class_name": sample["class_name"],
            "source": "sample",
        }
        for sample in samples
    ]

    return jsonify(
        {
            "query": {"bbox": bbox, "predicted_class": predicted_class},
            "exemplars": exemplars,
            "has_samples": len(exemplars) > 0,
        }
    )

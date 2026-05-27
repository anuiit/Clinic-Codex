from __future__ import annotations

import numpy as np
from flask import Blueprint, current_app, jsonify, request

from backend.app.services.image_service import InvalidImageError, decode_uploaded_image, invalid_image_payload

bp = Blueprint("classify", __name__)


def _services():
    return current_app.extensions["clinic_services"]


def _invalid_image_response():
    return jsonify(invalid_image_payload()), 400


@bp.post("/classify")
def classify():
    if "image" not in request.files:
        return jsonify({"error": "No 'image' file in request"}), 400
    try:
        img = decode_uploaded_image(request.files["image"])
    except InvalidImageError:
        return _invalid_image_response()
    return jsonify(_services().classify(img))


@bp.post("/classify-batch")
def classify_batch():
    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "No 'images' files in request"}), 400
    try:
        images = [np.array(decode_uploaded_image(file)) for file in files]
    except InvalidImageError:
        return _invalid_image_response()
    return jsonify(_services().classify_batch(images))

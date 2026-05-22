from __future__ import annotations

import io

import numpy as np
from flask import Blueprint, current_app, jsonify, request
from PIL import Image

bp = Blueprint("classify", __name__)


def _services():
    return current_app.extensions["clinic_services"]


@bp.post("/classify")
def classify():
    if "image" not in request.files:
        return jsonify({"error": "No 'image' file in request"}), 400
    img = Image.open(io.BytesIO(request.files["image"].read())).convert("RGB")
    return jsonify(_services().classify(img))


@bp.post("/classify-batch")
def classify_batch():
    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "No 'images' files in request"}), 400
    images = [np.array(Image.open(io.BytesIO(f.read())).convert("RGB")) for f in files]
    return jsonify(_services().classify_batch(images))

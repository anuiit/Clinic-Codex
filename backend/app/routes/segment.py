from __future__ import annotations

import io

import numpy as np
from flask import Blueprint, current_app, jsonify, request
from PIL import Image

bp = Blueprint("segment", __name__)


def _services():
    return current_app.extensions["clinic_services"]


@bp.post("/segment")
def segment():
    if "image" not in request.files:
        return jsonify({"error": "No 'image' file in request"}), 400

    img = np.array(Image.open(io.BytesIO(request.files["image"].read())).convert("RGB"))
    h, w = img.shape[:2]

    elements = []
    for proposal in _services().segment_page(img):
        if proposal.crop is None:
            continue
        x, y, bw, bh = proposal.bbox
        result = _services().classify(proposal.crop)
        elements.append({"bbox": [int(x), int(y), int(bw), int(bh)], **result})

    return jsonify({"num_elements": len(elements), "image_size": [int(w), int(h)], "elements": elements})

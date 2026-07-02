from __future__ import annotations

import numpy as np
from flask import Blueprint, current_app, jsonify, request

from backend.app.errors import ApiError
from backend.app.services.image_service import InvalidImageError, decode_uploaded_image, invalid_image_payload

bp = Blueprint("segment", __name__)


def _services():
    return current_app.extensions["clinic_services"]


def _invalid_image_response():
    return jsonify(invalid_image_payload()), 400


def _image_decode_kwargs():
    settings = current_app.config["CLINIC_SETTINGS"]
    return {
        "max_pixels": settings.max_image_pixels,
        "max_dimension": settings.max_image_dimension,
    }


@bp.post("/segment")
def segment():
    if "image" not in request.files:
        return jsonify({"error": "No 'image' file in request"}), 400

    try:
        img = np.array(decode_uploaded_image(request.files["image"], **_image_decode_kwargs()))
    except InvalidImageError:
        return _invalid_image_response()
    h, w = img.shape[:2]

    valid_proposals = []
    crops = []
    for proposal in _services().segment_page(img):
        if proposal.crop is None:
            continue

        valid_proposals.append(proposal)
        crops.append(proposal.crop)

    elements = []
    if crops:
        results = _services().classify_batch(crops)
        if len(results) != len(valid_proposals):
            raise ApiError(
                code="SEGMENT_CLASSIFICATION_MISMATCH",
                message="segment classifier result count did not match proposal count",
                status_code=500,
            )
        for proposal, result in zip(valid_proposals, results):
            x, y, bw, bh = proposal.bbox
            elements.append({"bbox": [int(x), int(y), int(bw), int(bh)], **result})

    return jsonify({"num_elements": len(elements), "image_size": [int(w), int(h)], "elements": elements})

from __future__ import annotations

from flask import Blueprint, current_app, jsonify

bp = Blueprint("classes", __name__)


def _services():
    return current_app.extensions["clinic_services"]


@bp.get("/classes")
def get_classes():
    config = _services().load_classes()
    return jsonify({"num_classes": config["num_classes"], "class_names": config["class_names"]})

from __future__ import annotations

from flask import Blueprint, current_app, jsonify

from backend.app.services.container import readiness_report

bp = Blueprint("health", __name__)


@bp.get("/health")
def health():
    return jsonify({"status": "ok"})


@bp.get("/ready")
def ready():
    services = current_app.extensions.get("clinic_services")
    if hasattr(services, "readiness"):
        report = services.readiness()
    else:
        report = readiness_report(current_app.config["CLINIC_SETTINGS"])
    return jsonify(report), 200 if report["ready"] else 503

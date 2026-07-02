from __future__ import annotations

from flask import Blueprint, current_app, jsonify, request

from backend.services.training_jobs import (
    AdminTrainingConflictError,
    AdminTrainingForbiddenError,
    AdminTrainingValidationError,
    RequestLaunchContext,
)
from backend.security.local_guard import request_launch_context, require_local_request

bp = Blueprint("admin_training", __name__)


def _services():
    return current_app.extensions["clinic_services"]


def _error(message: str, status_code: int):
    return jsonify({"status": "error", "error": message}), status_code


def _request_context() -> RequestLaunchContext:
    return request_launch_context()


@bp.get("/admin/training/summary")
@require_local_request
def get_admin_training_summary():
    return jsonify(_services().admin_training_summary(_request_context())), 200


@bp.get("/admin/training/jobs/latest")
@require_local_request
def get_latest_admin_training_job():
    job = _services().latest_admin_training_job()
    if job is None:
        return jsonify({"status": "ok", "local_only": True, "job": None}), 200
    return jsonify({"status": "ok", "local_only": True, "job": job}), 200


@bp.get("/admin/training/jobs/<run_id>")
@require_local_request
def get_admin_training_job(run_id: str):
    job = _services().get_admin_training_job(run_id)
    if job is None:
        return _error(f"training job not found: {run_id}", 404)
    return jsonify({"status": "ok", "local_only": True, "job": job}), 200


@bp.post("/admin/training/jobs")
def start_admin_training_job():
    data = request.get_json(force=True, silent=True)
    if data is None:
        return _error("invalid JSON", 400)
    try:
        job = _services().start_admin_training_job(data, _request_context())
    except AdminTrainingForbiddenError as exc:
        return _error(str(exc), 403)
    except AdminTrainingValidationError as exc:
        return _error(str(exc), 400)
    except AdminTrainingConflictError as exc:
        return _error(str(exc), 409)
    return jsonify({"status": "ok", "local_only": True, "job": job}), 202

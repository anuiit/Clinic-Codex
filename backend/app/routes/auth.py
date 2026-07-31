from __future__ import annotations

from flask import Blueprint, current_app, jsonify, request

from backend.security.auth import (
    BootstrapUnavailableError,
    BootstrapValidationError,
    COOKIE_NAME,
    auth_enabled,
    current_user,
    require_csrf,
)
from backend.security.local_guard import require_local_request

bp = Blueprint("auth", __name__, url_prefix="/auth")


def _auth_store():
    return current_app.extensions["clinic_auth_store"]


def _login_limiter():
    return current_app.extensions["clinic_login_limiter"]


def _session_response(user, *, status_code: int = 200):
    session_id, csrf_token = _auth_store().create_session(user["id"])
    response = jsonify(
        {
            "status": "ok",
            "auth_enabled": True,
            "user": user,
            "csrf_token": csrf_token,
        }
    )
    response.set_cookie(
        COOKIE_NAME,
        session_id,
        httponly=True,
        secure=current_app.config["CLINIC_SETTINGS"].auth_cookie_secure,
        samesite="Lax",
        path="/",
    )
    response.headers["Cache-Control"] = "no-store"
    return response, status_code


@bp.get("/bootstrap/status")
@require_local_request
def bootstrap_status():
    enabled = auth_enabled()
    available = enabled and _auth_store().bootstrap_available()
    response = jsonify(
        {
            "status": "ok",
            "auth_enabled": enabled,
            "bootstrap_available": available,
        }
    )
    response.headers["Cache-Control"] = "no-store"
    return response


@bp.post("/bootstrap")
@require_local_request
def bootstrap_first_admin():
    if not auth_enabled():
        return jsonify(
            {
                "status": "error",
                "error_code": "AUTH_DISABLED",
                "error": "authentication is disabled",
            }
        ), 409
    data = request.get_json(silent=True) or {}
    try:
        user = _auth_store().create_first_admin(data.get("email"), data.get("password"))
    except BootstrapValidationError as error:
        return jsonify(
            {
                "status": "error",
                "error_code": error.error_code,
                "error": str(error),
            }
        ), 400
    except BootstrapUnavailableError:
        return jsonify(
            {
                "status": "error",
                "error_code": "BOOTSTRAP_UNAVAILABLE",
                "error": "The first administrator has already been created.",
            }
        ), 409
    return _session_response(user, status_code=201)


@bp.post("/login")
def login():
    data = request.get_json(silent=True) or {}
    email, password = data.get("email"), data.get("password")
    if not isinstance(email, str) or not isinstance(password, str):
        return jsonify({"status": "error", "error": "email and password are required"}), 400
    limiter = _login_limiter()
    if not limiter.allow(email, request.remote_addr):
        return jsonify({"status": "error", "error_code": "LOGIN_RATE_LIMITED", "error": "too many failed login attempts"}), 429
    user = _auth_store().authenticate(email, password)
    if user is None:
        if not limiter.record_failure(email, request.remote_addr):
            return jsonify({"status": "error", "error_code": "LOGIN_RATE_LIMITED", "error": "too many failed login attempts"}), 429
        return jsonify({"status": "error", "error_code": "INVALID_CREDENTIALS", "error": "invalid credentials"}), 401
    limiter.clear(email, request.remote_addr)
    return _session_response(user)


@bp.get("/me")
def me():
    if not auth_enabled():
        return jsonify({"status": "ok", "auth_enabled": False, "user": None})
    user = current_user()
    if user is None:
        return jsonify({"status": "error", "error_code": "AUTH_REQUIRED", "error": "authentication required"}), 401
    csrf_token = _auth_store().csrf_token_for_session(request.cookies.get(COOKIE_NAME))
    if csrf_token is None:
        return jsonify({"status": "error", "error_code": "AUTH_REQUIRED", "error": "authentication required"}), 401
    return jsonify({"status": "ok", "auth_enabled": True, "user": user, "csrf_token": csrf_token})


@bp.post("/logout")
@require_csrf
def logout():
    _auth_store().revoke_session(request.cookies.get(COOKIE_NAME))
    response = jsonify({"status": "ok"})
    response.delete_cookie(
        COOKIE_NAME,
        path="/",
        secure=current_app.config["CLINIC_SETTINGS"].auth_cookie_secure,
        samesite="Lax",
    )
    return response

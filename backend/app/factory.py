"""Flask application factory."""
from __future__ import annotations

from flask import Flask, request

from backend.app.config import Settings
from backend.app.errors import ApiError, ModelAssetUnavailable
from backend.app.routes import register_routes
from backend.app.services.container import DefaultServices
from backend.security.auth import AuthStore, LoginAttemptLimiter, load_current_user


def create_app(settings: Settings | None = None, services=None) -> Flask:
    settings = settings or Settings.from_env()
    if settings.authentication_enabled and not settings.auth_secret_key:
        raise RuntimeError("AUTH_SECRET_KEY must be configured when authentication is enabled")
    app = Flask(__name__)
    app.config.update(
        TESTING=settings.testing,
        MAX_CONTENT_LENGTH=settings.max_content_length,
        CLINIC_SETTINGS=settings,
        ENABLE_LEGACY_ENDPOINTS=settings.enable_legacy_endpoints,
        SECRET_KEY=settings.auth_secret_key or "testing-only-secret",
    )
    app.extensions["clinic_services"] = services or DefaultServices(settings)

    if settings.authentication_enabled:
        app.extensions["clinic_auth_store"] = AuthStore(
            settings.auth_database_path, session_ttl_hours=settings.auth_session_hours
        )
        app.extensions["clinic_login_limiter"] = LoginAttemptLimiter()
        app.extensions["clinic_auth_store"].ensure_bootstrap_user(
            settings.auth_bootstrap_email, settings.auth_bootstrap_password, settings.auth_bootstrap_role
        )

    @app.before_request
    def load_identity():
        load_current_user()
    @app.after_request
    def add_cors_headers(response):
        origin = request.headers.get("Origin", "")
        if origin and origin in settings.cors_origins:
            response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-CSRF-Token"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        if origin and origin in settings.cors_origins:
            response.headers["Access-Control-Allow-Credentials"] = "true"
            response.headers["Vary"] = "Origin"
        return response

    @app.errorhandler(ApiError)
    def handle_api_error(error: ApiError):
        return error.to_response()

    @app.errorhandler(ModelAssetUnavailable)
    def handle_model_asset_unavailable(error: ModelAssetUnavailable):
        return error.to_response()

    register_routes(app, settings, app.extensions["clinic_services"])
    return app

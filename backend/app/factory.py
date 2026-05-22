"""Flask application factory."""
from __future__ import annotations

from flask import Flask, request

from backend.app.config import Settings
from backend.app.errors import ApiError, ModelAssetUnavailable
from backend.app.routes import register_routes
from backend.app.services.container import DefaultServices


def create_app(settings: Settings | None = None, services=None) -> Flask:
    settings = settings or Settings.from_env()
    app = Flask(__name__)
    app.config.update(
        TESTING=settings.testing,
        MAX_CONTENT_LENGTH=settings.max_content_length,
        CLINIC_SETTINGS=settings,
        ENABLE_LEGACY_ENDPOINTS=settings.enable_legacy_endpoints,
    )
    app.extensions["clinic_services"] = services or DefaultServices(settings)

    @app.after_request
    def add_cors_headers(response):
        origin = request.headers.get("Origin", "")
        if origin and origin in settings.cors_origins:
            response.headers["Access-Control-Allow-Origin"] = origin
        else:
            response.headers["Access-Control-Allow-Origin"] = settings.cors_origins[0] if settings.cors_origins else ""
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        return response

    @app.errorhandler(ApiError)
    def handle_api_error(error: ApiError):
        return error.to_response()

    @app.errorhandler(ModelAssetUnavailable)
    def handle_model_asset_unavailable(error: ModelAssetUnavailable):
        return error.to_response()

    register_routes(app, settings, app.extensions["clinic_services"])
    return app

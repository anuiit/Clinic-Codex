"""Route registration for the modular Flask backend."""
from __future__ import annotations

from flask import Flask

from backend.app.config import Settings


def register_routes(app: Flask, settings: Settings, services) -> None:
    from .admin_annotations import bp as admin_annotations_bp
    from .annotations import bp as annotations_bp
    from .admin_training import bp as admin_training_bp
    from .classes import bp as classes_bp
    from .classify import bp as classify_bp
    from .health import bp as health_bp
    from .segment import bp as segment_bp
    from .similarity import legacy_bp, bp as similarity_bp

    app.register_blueprint(health_bp)
    app.register_blueprint(classes_bp)
    app.register_blueprint(annotations_bp)
    app.register_blueprint(admin_annotations_bp)
    app.register_blueprint(admin_training_bp)
    app.register_blueprint(similarity_bp)
    app.register_blueprint(classify_bp)
    app.register_blueprint(segment_bp)
    if settings.enable_legacy_endpoints:
        app.register_blueprint(legacy_bp)

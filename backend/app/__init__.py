"""Modular Flask application package for Clinic Codex."""

def create_app(*args, **kwargs):
    # Loading settings must not eagerly import routes and their services.
    from .factory import create_app as factory
    return factory(*args, **kwargs)

__all__ = ["create_app"]

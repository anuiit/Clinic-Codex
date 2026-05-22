"""WSGI entrypoint for the Clinic Codex backend."""

from backend.app.factory import create_app

app = create_app()

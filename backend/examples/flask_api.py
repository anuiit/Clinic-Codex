"""Compatibility runner for the modular Clinic Codex Flask API.

The route implementation now lives under `backend.app`. This module remains so
existing commands such as `python backend/examples/flask_api.py` and imports of
`examples.flask_api.app` continue to work during Phase 1.
"""
from __future__ import annotations

import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent
for path in (str(REPO_ROOT), str(BACKEND_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from backend.app.config import Settings
from backend.app.factory import create_app

settings = Settings.from_env()
app = create_app(settings=settings)

# Compatibility aliases for older smoke tests/scripts that introspect module
# configuration. Route handlers do not read these globals anymore.
HOST = settings.host
PORT = settings.port
ALLOWED_CORS_ORIGINS = list(settings.cors_origins)
MODEL_DIR = settings.model_dir

if __name__ == "__main__":
    print(f"Codex Classifier API - http://{HOST}:{PORT}")
    print("  POST /classify, /classify-batch, /segment, /similar, /trust")
    print("  GET  /classes")
    app.run(host=HOST, port=PORT, debug=False)

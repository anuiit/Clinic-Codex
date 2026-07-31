"""Release and active-model version metadata."""
from __future__ import annotations

import json
from typing import Any

from backend.app.config import Settings

APP_NAME = "Clinic Codex"
DEVELOPMENT_VERSION = "dev"


def release_version(settings: Settings) -> str:
    version_path = settings.backend_root.parent / "VERSION"
    try:
        version = version_path.read_text(encoding="utf-8").strip()
    except OSError:
        return DEVELOPMENT_VERSION
    return version or DEVELOPMENT_VERSION


def active_model_version(settings: Settings) -> str | None:
    try:
        payload: Any = json.loads(
            settings.class_config_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(payload, dict):
        return None
    value = payload.get("model_version")
    if not isinstance(value, (str, int, float)):
        return None
    normalized = str(value).strip()
    return normalized or None


def runtime_version_info(settings: Settings) -> dict[str, str | None]:
    return {
        "app_name": APP_NAME,
        "app_version": release_version(settings),
        "model_version": active_model_version(settings),
    }

"""Lightweight settings for the Clinic Codex Flask app."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _truthy(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


@dataclass(frozen=True)
class Settings:
    backend_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])
    host: str = "0.0.0.0"
    port: int = 7117
    cors_origins: tuple[str, ...] = ("http://localhost:7118",)
    model_dir: str = ""
    max_content_length: int = 50 * 1024 * 1024
    enable_legacy_endpoints: bool = True
    testing: bool = False

    @property
    def class_config_path(self) -> Path:
        return self.backend_root / "codex_model" / "config.json"

    @property
    def annotations_dir(self) -> Path:
        return self.backend_root / "annotations"

    @property
    def elements_dir(self) -> Path:
        return self.backend_root / "training_data" / "Elements"

    @property
    def data_dir(self) -> Path:
        return self.backend_root / "data"

    @classmethod
    def from_env(cls) -> "Settings":
        raw_origins = os.environ.get("CORS_ORIGINS", "http://localhost:7118")
        origins = tuple(o.strip() for o in raw_origins.split(",") if o.strip())
        return cls(
            host=os.environ.get("HOST", "0.0.0.0"),
            port=int(os.environ.get("PORT", "7117")),
            cors_origins=origins,
            model_dir=os.environ.get("MODEL_DIR", ""),
            enable_legacy_endpoints=_truthy(os.environ.get("ENABLE_LEGACY_ENDPOINTS"), True),
            testing=_truthy(os.environ.get("FLASK_TESTING"), False),
        )

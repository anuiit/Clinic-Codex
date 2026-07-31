"""Lightweight settings for the Clinic Codex Flask app."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _truthy(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def _positive_int(value: str | None, default: int) -> int:
    if value is None or value.strip() == "":
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


@dataclass(frozen=True)
class Settings:
    backend_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])
    host: str = "127.0.0.1"
    port: int = 7117
    cors_origins: tuple[str, ...] = ("http://localhost:7118",)
    model_dir: str = ""
    max_content_length: int = 50 * 1024 * 1024
    max_image_pixels: int = 80_000_000
    max_image_dimension: int = 10_000
    enable_legacy_endpoints: bool = True
    testing: bool = False
    mobile_sam_checkpoint: str = ""
    enable_admin_training_jobs: bool = False
    admin_training_log_tail_lines: int = 80
    admin_training_max_batch_size: int = 256
    admin_training_allowed_devices: tuple[str, ...] = ("auto", "cpu", "mps", "cuda")

    auth_required: bool | None = None
    auth_secret_key: str = ""
    auth_session_hours: int = 24 * 7
    auth_cookie_secure: bool = True
    auth_bootstrap_email: str = ""
    auth_bootstrap_password: str = ""
    auth_bootstrap_role: str = "org_admin"

    @property
    def authentication_enabled(self) -> bool:
        return self.auth_required if self.auth_required is not None else not self.testing

    @property
    def auth_database_path(self) -> Path:
        return self.backend_root / "clinic_auth.sqlite3"
    @property
    def class_config_path(self) -> Path:
        return self.backend_root / "codex_model" / "config.json"

    @property
    def classifier_weights_dir(self) -> Path:
        return Path(self.model_dir).expanduser() if self.model_dir else self.backend_root / "codex_model" / "weights"

    @property
    def mobile_sam_checkpoint_path(self) -> Path:
        if self.mobile_sam_checkpoint:
            return Path(self.mobile_sam_checkpoint).expanduser()
        return Path.home() / ".cache" / "mobile_sam" / "mobile_sam.pt"

    @property
    def annotations_dir(self) -> Path:
        return self.backend_root / "annotations"

    @property
    def elements_dir(self) -> Path:
        return self.backend_root / "training_data" / "Elements"

    @property
    def data_dir(self) -> Path:
        return self.backend_root / "data"

    @property
    def admin_training_runs_dir(self) -> Path:
        return self.backend_root / "training_runs"

    @property
    def model_registry_dir(self) -> Path:
        return self.backend_root / "model_registry"

    @property
    def admin_training_script_path(self) -> Path:
        return self.backend_root.parent / "scripts" / "retrain.sh"

    @classmethod
    def from_env(cls) -> "Settings":
        raw_origins = os.environ.get("CORS_ORIGINS", "http://localhost:7118")
        origins = tuple(o.strip() for o in raw_origins.split(",") if o.strip())
        return cls(
            host=os.environ.get("HOST", "127.0.0.1"),
            port=int(os.environ.get("PORT", "7117")),
            cors_origins=origins,
            model_dir=os.environ.get("MODEL_DIR", ""),
            mobile_sam_checkpoint=os.environ.get("MOBILE_SAM_CHECKPOINT", ""),
            max_image_pixels=_positive_int(os.environ.get("MAX_IMAGE_PIXELS"), 80_000_000),
            max_image_dimension=_positive_int(os.environ.get("MAX_IMAGE_DIMENSION"), 10_000),
            enable_legacy_endpoints=_truthy(os.environ.get("ENABLE_LEGACY_ENDPOINTS"), True),
            enable_admin_training_jobs=_truthy(os.environ.get("ENABLE_ADMIN_TRAINING_JOBS"), False),
            auth_required=None if "AUTH_REQUIRED" not in os.environ else _truthy(os.environ.get("AUTH_REQUIRED"), False),
            auth_secret_key=os.environ.get("AUTH_SECRET_KEY", ""),
            auth_session_hours=_positive_int(os.environ.get("AUTH_SESSION_HOURS"), 24 * 7),
            auth_cookie_secure=_truthy(os.environ.get("AUTH_COOKIE_SECURE"), not _truthy(os.environ.get("FLASK_DEBUG"), False)),
            auth_bootstrap_email=os.environ.get("AUTH_BOOTSTRAP_EMAIL", ""),
            auth_bootstrap_password=os.environ.get("AUTH_BOOTSTRAP_PASSWORD", ""),
            auth_bootstrap_role=os.environ.get("AUTH_BOOTSTRAP_ROLE", "org_admin"),
            testing=_truthy(os.environ.get("FLASK_TESTING"), False),
        )

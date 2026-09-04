"""Lightweight settings for the Clinic Codex Flask app."""
from __future__ import annotations

import math
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


def _nonnegative_float(
    value: str | None, default: float, *, maximum: float | None = None
) -> float:
    if value is None or value.strip() == "":
        return default
    try:
        parsed = float(value)
    except ValueError:
        return default
    if not math.isfinite(parsed) or parsed < 0 or (maximum is not None and parsed > maximum):
        return default
    return parsed


def _rollout_mode(value: str | None) -> str:
    mode = (value or "off").strip().lower()
    if mode not in {"off", "shadow", "canary"}:
        raise ValueError("CLASSIFIER_ROLLOUT_MODE must be one of: off, shadow, canary")
    return mode


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
    allow_local_admin_self_review: bool = False
    admin_training_log_tail_lines: int = 80
    admin_training_max_batch_size: int = 256
    admin_training_allowed_devices: tuple[str, ...] = ("auto", "cpu", "mps", "cuda")
    admin_training_snapshot_dir: str = ""
    admin_training_backbone_manifest: str = ""

    classifier_rollout_mode: str = "off"
    classifier_candidate_reference: str = "candidate"
    classifier_candidate_device: str = "cpu"
    classifier_canary_percent: float = 0.0
    classifier_shadow_sample_percent: float = 100.0
    classifier_candidate_timeout_seconds: float = 120.0
    classifier_candidate_batch_size: int = 16
    classifier_candidate_max_images: int = 512
    classifier_shadow_queue_size: int = 2
    classifier_shadow_max_inflight: int = 1
    classifier_offline_report: str = ""
    classifier_promotion_spec: str = ""

    auth_required: bool | None = None
    auth_secret_key: str = ""
    auth_session_hours: int = 24 * 7
    auth_cookie_secure: bool = True
    auth_bootstrap_email: str = ""
    auth_bootstrap_password: str = ""
    auth_bootstrap_role: str = "org_admin"

    def __post_init__(self) -> None:
        if self.classifier_rollout_mode not in {"off", "shadow", "canary"}:
            raise ValueError("classifier_rollout_mode must be off, shadow, or canary")
        for name in ("classifier_canary_percent", "classifier_shadow_sample_percent"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 100.0:
                raise ValueError(f"{name} must be between 0 and 100")
        if (
            not math.isfinite(self.classifier_candidate_timeout_seconds)
            or self.classifier_candidate_timeout_seconds <= 0
        ):
            raise ValueError("classifier_candidate_timeout_seconds must be positive")
        for name in (
            "classifier_candidate_batch_size",
            "classifier_candidate_max_images",
            "classifier_shadow_queue_size",
            "classifier_shadow_max_inflight",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")

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
        if self.model_dir:
            return Path(self.model_dir).expanduser()
        return self.backend_root / "codex_model" / "weights"

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
    def classifier_offline_report_path(self) -> Path | None:
        if not self.classifier_offline_report:
            return None
        return Path(self.classifier_offline_report).expanduser()

    @property
    def classifier_promotion_spec_path(self) -> Path:
        if self.classifier_promotion_spec:
            return Path(self.classifier_promotion_spec).expanduser()
        return self.model_registry_dir / "specs" / "r2-e2e-promotion-v1.json"

    @property
    def admin_training_script_path(self) -> Path:
        return self.backend_root.parent / "scripts" / ("retrain.ps1" if os.name == "nt" else "retrain.sh")

    @property
    def admin_training_snapshot_path(self) -> Path | None:
        if not self.admin_training_snapshot_dir:
            return None
        return Path(self.admin_training_snapshot_dir).expanduser().resolve()

    @property
    def admin_training_backbone_manifest_path(self) -> Path:
        if self.admin_training_backbone_manifest:
            return Path(self.admin_training_backbone_manifest).expanduser().resolve()
        return self.backend_root / "training_corpus" / "backbone-pins" / "dinov2-vits14-local.json"

    @property
    def admin_training_config_path(self) -> Path:
        return self.backend_root / "codex_pipeline" / "config" / "snapshot-warmstart.yaml"

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
            allow_local_admin_self_review=_truthy(os.environ.get("ALLOW_LOCAL_ADMIN_SELF_REVIEW"), False),
            admin_training_snapshot_dir=os.environ.get("ADMIN_TRAINING_SNAPSHOT_DIR", "").strip(),
            admin_training_backbone_manifest=os.environ.get(
                "ADMIN_TRAINING_BACKBONE_MANIFEST", ""
            ).strip(),
            classifier_rollout_mode=_rollout_mode(os.environ.get("CLASSIFIER_ROLLOUT_MODE")),
            classifier_candidate_reference=os.environ.get("CLASSIFIER_CANDIDATE_REFERENCE", "candidate").strip(),
            classifier_candidate_device=os.environ.get("CLASSIFIER_CANDIDATE_DEVICE", "cpu").strip() or "cpu",
            classifier_canary_percent=_nonnegative_float(
                os.environ.get("CLASSIFIER_CANARY_PERCENT"), 0.0, maximum=100.0
            ),
            classifier_shadow_sample_percent=_nonnegative_float(
                os.environ.get("CLASSIFIER_SHADOW_SAMPLE_PERCENT"), 100.0, maximum=100.0
            ),
            classifier_candidate_timeout_seconds=_nonnegative_float(
                os.environ.get("CLASSIFIER_CANDIDATE_TIMEOUT_SECONDS"), 120.0
            ),
            classifier_candidate_batch_size=_positive_int(
                os.environ.get("CLASSIFIER_CANDIDATE_BATCH_SIZE"), 16
            ),
            classifier_candidate_max_images=_positive_int(
                os.environ.get("CLASSIFIER_CANDIDATE_MAX_IMAGES"), 512
            ),
            classifier_shadow_queue_size=_positive_int(
                os.environ.get("CLASSIFIER_SHADOW_QUEUE_SIZE"), 2
            ),
            classifier_shadow_max_inflight=_positive_int(
                os.environ.get("CLASSIFIER_SHADOW_MAX_INFLIGHT"), 1
            ),
            classifier_offline_report=os.environ.get("CLASSIFIER_OFFLINE_REPORT", "").strip(),
            classifier_promotion_spec=os.environ.get("CLASSIFIER_PROMOTION_SPEC", "").strip(),
            auth_required=None if "AUTH_REQUIRED" not in os.environ else _truthy(os.environ.get("AUTH_REQUIRED"), False),
            auth_secret_key=os.environ.get("AUTH_SECRET_KEY", ""),
            auth_session_hours=_positive_int(os.environ.get("AUTH_SESSION_HOURS"), 24 * 7),
            auth_cookie_secure=_truthy(os.environ.get("AUTH_COOKIE_SECURE"), not _truthy(os.environ.get("FLASK_DEBUG"), False)),
            auth_bootstrap_email=os.environ.get("AUTH_BOOTSTRAP_EMAIL", ""),
            auth_bootstrap_password=os.environ.get("AUTH_BOOTSTRAP_PASSWORD", ""),
            auth_bootstrap_role=os.environ.get("AUTH_BOOTSTRAP_ROLE", "org_admin"),
            testing=_truthy(os.environ.get("FLASK_TESTING"), False),
        )

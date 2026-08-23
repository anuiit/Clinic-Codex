"""Guarded local admin training summary and job runner.

This module intentionally exposes only a small allowlisted wrapper around the
existing approved-only classifier retrain script.  It is not an auth system and
is disabled by default.
"""
from __future__ import annotations

import hashlib
import errno
import json
import os
import subprocess
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import yaml

try:
    from backend.app.config import Settings
    from backend.security.local_guard import RequestLaunchContext, is_local_origin, is_loopback_address
    from backend.services.model_registry import SCHEMA_VERSION, ModelRegistry, ModelRegistryValidationError, safe_relative_path, sha256_file
    from backend.services.annotation_review import LOCAL_ONLY_WARNING, AnnotationReviewStore
except ImportError:  # pragma: no cover - compatibility when backend dir is sys.path root
    from app.config import Settings  # type: ignore
    from security.local_guard import RequestLaunchContext, is_local_origin, is_loopback_address  # type: ignore
    from services.model_registry import SCHEMA_VERSION, ModelRegistry, ModelRegistryValidationError, safe_relative_path, sha256_file  # type: ignore
    from services.annotation_review import LOCAL_ONLY_WARNING, AnnotationReviewStore  # type: ignore

ALLOWED_JOB_FIELDS = {"dry_run", "device", "batch_size", "notes"}
DEFAULT_ALLOWED_DEVICES = ("auto", "cpu", "mps", "cuda")
TERMINAL_STATUSES = {"succeeded", "failed", "disabled", "rejected"}
LAUNCH_GUARD_STALE_SECONDS = 3600
_LAUNCH_GUARD_LOCK = threading.Lock()


class AdminTrainingError(Exception):
    """Base class for admin training errors."""


class AdminTrainingValidationError(AdminTrainingError, ValueError):
    """Raised for invalid training job input."""


class AdminTrainingForbiddenError(AdminTrainingError, PermissionError):
    """Raised when local training launch is disabled or nonlocal."""


class AdminTrainingConflictError(AdminTrainingError, RuntimeError):
    """Raised when another training job is already running."""


@contextmanager
def _atomic_launch_guard(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LAUNCH_GUARD_LOCK:
        try:
            age_seconds = datetime.now(timezone.utc).timestamp() - path.stat().st_mtime
        except FileNotFoundError:
            age_seconds = 0
        if age_seconds > LAUNCH_GUARD_STALE_SECONDS:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise AdminTrainingConflictError("training job launch already in progress") from exc
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps({"pid": os.getpid(), "created_at": utc_now_iso()}) + "\n")
    try:
        yield
    finally:
        with _LAUNCH_GUARD_LOCK:
            try:
                path.unlink()
            except FileNotFoundError:
                pass


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _live_annotations_sha256(records: list[dict[str, Any]]) -> str:
    ordered = sorted(
        records,
        key=lambda item: json.dumps(
            item, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ),
    )
    payload = json.dumps(
        ordered, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _manifest_live_annotation_records(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in ("rows", "duplicates"):
        value = manifest.get(key)
        if not isinstance(value, list):
            raise ValueError(f"manifest {key} must be a list")
        rows.extend(row for row in value if isinstance(row, dict))
    conflicts = manifest.get("conflicts")
    if not isinstance(conflicts, list):
        raise ValueError("manifest conflicts must be a list")
    for conflict in conflicts:
        if not isinstance(conflict, dict) or not isinstance(conflict.get("rows"), list):
            raise ValueError("manifest conflict rows must be a list")
        rows.extend(row for row in conflict["rows"] if isinstance(row, dict))
    return [
        {
            "source_id": row.get("source_id"),
            "class_name": row.get("class_name"),
            "bbox": row.get("bbox"),
            "source_fingerprint_v1": row.get("source_fingerprint_v1"),
            "source_sha256": row.get("source_sha256"),
            "source_image_sha256": row.get("source_image_sha256"),
        }
        for row in rows
        if row.get("source_kind") == "live_annotation"
    ]


def _file_info(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
    if path.is_file():
        stat = path.stat()
        info.update({"size": stat.st_size, "mtime": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()})
    return info


def _tail_lines(path: Path, limit: int) -> list[str]:
    if not path.is_file():
        return []
    return path.read_text(encoding="utf-8", errors="replace").splitlines()[-limit:]


def _windows_process_is_alive(pid: int) -> bool | None:
    try:
        import ctypes
        from ctypes import wintypes
    except Exception:
        return None

    process_query_limited_information = 0x1000
    error_invalid_parameter = 87
    still_active = 259
    try:
        kernel32 = ctypes.windll.kernel32
    except AttributeError:
        return None
    _configure_kernel32_process_signatures(kernel32, ctypes, wintypes)
    handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
    if not handle:
        error = kernel32.GetLastError()
        if error == error_invalid_parameter:
            return False
        if error == 5:  # ERROR_ACCESS_DENIED means a process exists but cannot be queried.
            return True
        return None
    try:
        exit_code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return None
        return exit_code.value == still_active
    finally:
        kernel32.CloseHandle(handle)


def _configure_kernel32_process_signatures(kernel32: Any, ctypes_module: Any, wintypes: Any) -> None:
    kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.GetLastError.argtypes = ()
    kernel32.GetLastError.restype = wintypes.DWORD
    kernel32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes_module.POINTER(wintypes.DWORD))
    kernel32.GetExitCodeProcess.restype = wintypes.BOOL
    kernel32.GetProcessTimes.argtypes = (
        wintypes.HANDLE,
        ctypes_module.POINTER(wintypes.FILETIME),
        ctypes_module.POINTER(wintypes.FILETIME),
        ctypes_module.POINTER(wintypes.FILETIME),
        ctypes_module.POINTER(wintypes.FILETIME),
    )
    kernel32.GetProcessTimes.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL


def _parse_positive_pid(pid: Any) -> int | None:
    if isinstance(pid, bool):
        return None
    try:
        parsed_pid = int(pid)
    except (TypeError, ValueError):
        return None
    return parsed_pid if parsed_pid > 0 else None


def _process_is_alive(pid: Any) -> bool | None:
    """Return process liveness when it can be determined without owning Popen."""
    parsed_pid = _parse_positive_pid(pid)
    if parsed_pid is None:
        return None
    if os.name == "nt":
        return _windows_process_is_alive(parsed_pid)
    if os.name != "posix":
        return None
    try:
        os.kill(parsed_pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as exc:
        if exc.errno == errno.ESRCH:
            return False
        if exc.errno == errno.EPERM:
            return True
        return None
    return True


def _process_identity(pid: Any) -> str | None:
    """Return a stable process identity token when the platform can provide one.

    PID liveness alone cannot distinguish the original retrain process from an
    unrelated process that later reused the same PID after a backend restart.
    """
    parsed_pid = _parse_positive_pid(pid)
    if parsed_pid is None:
        return None
    if os.name == "nt":
        return _windows_process_identity(parsed_pid)
    if os.name == "posix":
        return _posix_process_identity(parsed_pid)
    return None


def _posix_process_identity(pid: int) -> str | None:
    stat_path = Path("/proc") / str(pid) / "stat"
    try:
        raw = stat_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    marker = raw.rfind(")")
    if marker == -1:
        return None
    fields = raw[marker + 1 :].strip().split()
    if len(fields) < 20:
        return None
    start_ticks = fields[19]
    return f"linux-proc-start:{start_ticks}" if start_ticks else None


def _windows_process_identity(pid: int) -> str | None:
    try:
        import ctypes
        from ctypes import wintypes
    except Exception:
        return None

    process_query_limited_information = 0x1000
    try:
        kernel32 = ctypes.windll.kernel32
    except AttributeError:
        return None
    _configure_kernel32_process_signatures(kernel32, ctypes, wintypes)
    handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
    if not handle:
        return None
    try:
        creation_time = wintypes.FILETIME()
        exit_time = wintypes.FILETIME()
        kernel_time = wintypes.FILETIME()
        user_time = wintypes.FILETIME()
        ok = kernel32.GetProcessTimes(
            handle,
            ctypes.byref(creation_time),
            ctypes.byref(exit_time),
            ctypes.byref(kernel_time),
            ctypes.byref(user_time),
        )
        if not ok:
            return None
        created = (creation_time.dwHighDateTime << 32) | creation_time.dwLowDateTime
        return f"windows-filetime:{created}" if created else None
    finally:
        kernel32.CloseHandle(handle)


def _read_lock_pid(path: Path) -> int | None:
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    try:
        pid = int(raw)
    except ValueError:
        return None
    return pid if pid > 0 else None


class AdminTrainingService:
    def __init__(self, settings: Settings, review_store: AnnotationReviewStore):
        self.settings = settings
        self.review_store = review_store
        self.runs_dir = settings.admin_training_runs_dir
        self.script_path = settings.admin_training_script_path
        self.repo_root = settings.backend_root.parent
        self.model_registry = ModelRegistry(
            settings.model_registry_dir,
            repo_root=self.repo_root,
            runtime_model_dir=settings.backend_root / "codex_model",
        )
        self._processes: dict[str, subprocess.Popen] = {}

    def summary(self, context: RequestLaunchContext | None = None) -> dict[str, Any]:
        queue = self.review_store.list_queue()
        rows = [element for analysis in queue["analyses"] for element in analysis["elements"]]
        trainable = [element for element in rows if element.get("trainable")]
        per_class: dict[str, int] = {}
        split_counts = {"train": 0, "val": 0, "test": 0, "excluded": 0}
        for element in rows:
            split = element.get("dataset_split")
            if split not in split_counts:
                split = "excluded"
            split_counts[split] += 1
        for element in trainable:
            class_name = element.get("class_name") or "Unnamed"
            per_class[class_name] = per_class.get(class_name, 0) + 1

        snapshot = self._training_snapshot_info()
        allowed = self.launch_allowed(
            context,
            trainable_count=queue["counts"]["trainable"],
            snapshot=snapshot,
        )
        return {
            "status": "ok",
            "local_only": True,
            "warning": LOCAL_ONLY_WARNING,
            "training_jobs_enabled": self.settings.enable_admin_training_jobs,
            "launch_allowed_for_request": allowed["allowed"],
            "launch_disabled_reasons": allowed["reasons"],
            "data": {
                "total": queue["counts"]["total"],
                "pending": queue["counts"]["pending"],
                "approved": queue["counts"]["approved"],
                "rejected": queue["counts"]["rejected"],
                "trainable": queue["counts"]["trainable"],
                "classes": sorted(per_class),
                "per_class": dict(sorted(per_class.items())),
                "split_counts": split_counts,
                "diagnostics": queue["diagnostics"],
            },
            "parameters": self._parameters(),
            "paths": self._paths(),
            "artifacts": self._artifacts(),
            "training_snapshot": snapshot,
            "latest_job": self.latest_job(),
        }

    def launch_allowed(
        self,
        context: RequestLaunchContext | None,
        *,
        trainable_count: int | None = None,
        snapshot: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        reasons: list[str] = []
        if not self.settings.enable_admin_training_jobs:
            reasons.append("disabled_by_default: set ENABLE_ADMIN_TRAINING_JOBS=1 to allow local launches")
        else:
            snapshot = snapshot or self._training_snapshot_info()
            reasons.extend(snapshot["errors"])
            if self.settings.model_dir:
                reasons.append(
                    "training_model_dir_override_unsupported: unset MODEL_DIR before admin retraining"
                )
        if trainable_count is None:
            try:
                trainable_count = int(self.review_store.list_queue()["counts"]["trainable"])
            except Exception as exc:
                reasons.append(f"review_store_unavailable: {exc}")
                trainable_count = None
        if trainable_count is not None and trainable_count <= 0:
            reasons.append("no_trainable_annotations: approve at least one current annotation before launching retraining")
        if context is not None:
            if not is_loopback_address(context.remote_addr):
                reasons.append("non_loopback_remote_addr")
            if not is_loopback_address(context.host):
                reasons.append("nonlocal_host")
            if not is_local_origin(context.origin):
                reasons.append("nonlocal_origin")
        if not self.script_path.is_file():
            reasons.append(f"missing_retrain_script: {self.script_path}")
        return {"allowed": not reasons, "reasons": reasons}

    def latest_job(self) -> dict[str, Any] | None:
        if not self.runs_dir.exists():
            return None
        candidates = sorted(self.runs_dir.glob("*/status.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            return None
        return self._hydrate_job(candidates[0])

    def get_job(self, run_id: str) -> dict[str, Any] | None:
        if not run_id or "/" in run_id or "\\" in run_id or ".." in run_id:
            return None
        return self._hydrate_job(self.runs_dir / run_id / "status.json")

    def start_job(self, payload: dict[str, Any], context: RequestLaunchContext, *, actor_id: str | None = None) -> dict[str, Any]:
        allowed = self.launch_allowed(context)
        if not allowed["allowed"]:
            raise AdminTrainingForbiddenError("; ".join(allowed["reasons"]))

        request = self._validate_payload(payload)
        with _atomic_launch_guard(self.runs_dir / ".launch.lock"):
            snapshot = self._training_snapshot_info()
            if not snapshot["valid"]:
                raise AdminTrainingForbiddenError("; ".join(snapshot["errors"]))
            latest = self.latest_job()
            if latest and latest.get("status") == "running":
                raise AdminTrainingConflictError(f"training job already running: {latest.get('run_id')}")
            return self._start_job_unlocked(
                request, snapshot=snapshot, actor_id=actor_id
            )

    def _start_job_unlocked(
        self,
        request: dict[str, Any],
        *,
        snapshot: dict[str, Any],
        actor_id: str | None = None,
    ) -> dict[str, Any]:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
        model_version_id = self.model_registry.build_version_id(run_id=run_id.rsplit("-", 1)[-1])
        candidate_version_dir = self.model_registry.version_dir(model_version_id)
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        log_path = run_dir / "train.log"
        status_path = run_dir / "status.json"
        command = [
            "bash",
            str(self.script_path),
            "--elements-dir",
            snapshot["paths"]["elements"],
            "--approved-manifest",
            snapshot["paths"]["manifest"],
            "--metadata-csv",
            snapshot["paths"]["metadata"],
            "--backbone-manifest",
            str(self.settings.admin_training_backbone_manifest_path),
            "--config",
            str(self.settings.admin_training_config_path),
            "--init-projection",
            str(self.settings.classifier_weights_dir / "projection.pt"),
        ] + (["--dry-run"] if request["dry_run"] else [])
        env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": os.environ.get("HOME", ""),
            "PYTHONUNBUFFERED": "1",
            "BATCH_SIZE": str(request["batch_size"]),
            "DEVICE": request["device"],
            "MODEL_VERSION_ID": model_version_id,
            "MODEL_REGISTRY_DIR": str(self.settings.model_registry_dir),
        }
        status = {
            "run_id": run_id,
            "model_version_id": model_version_id,
            "model_registry_dir": str(self.settings.model_registry_dir),
            "candidate_version_dir": str(candidate_version_dir),
            "candidate_manifest_path": str(candidate_version_dir / "manifest.json"),
            "status": "running",
            "local_only": True,
            "dry_run": request["dry_run"],
            "device": request["device"],
            "batch_size": request["batch_size"],
            "notes": request["notes"],
            "requested_by": actor_id,
            "started_at": utc_now_iso(),
            "finished_at": None,
            "exit_code": None,
            "pid": None,
            "process_identity": None,
            "command": command,
            "cwd": str(self.repo_root),
            "env": {
                "BATCH_SIZE": env["BATCH_SIZE"],
                "DEVICE": env["DEVICE"],
                "PYTHONUNBUFFERED": env["PYTHONUNBUFFERED"],
                "MODEL_VERSION_ID": env["MODEL_VERSION_ID"],
                "MODEL_REGISTRY_DIR": env["MODEL_REGISTRY_DIR"],
            },
            "lock_path": str(self.settings.backend_root / ".retrain.lock"),
            "log_path": str(log_path),
            "training_snapshot": snapshot,
            "training_snapshot_manifest_hash": snapshot["snapshot_manifest_sha256"],
        }
        _write_json_atomic(status_path, status)
        try:
            with log_path.open("a", encoding="utf-8") as log_handle:
                process = subprocess.Popen(
                    command,
                    cwd=str(self.repo_root),
                    env=env,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
        except Exception as exc:
            status.update({"status": "failed", "finished_at": utc_now_iso(), "exit_code": None, "error": str(exc)})
            _write_json_atomic(status_path, status)
            raise AdminTrainingValidationError(f"could not start retrain script: {exc}") from exc
        status["pid"] = process.pid
        status["process_identity"] = _process_identity(process.pid)
        self._processes[run_id] = process
        _write_json_atomic(status_path, status)
        return self._hydrate_job(status_path) or status

    def _validate_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise AdminTrainingValidationError("payload must be a JSON object")
        unknown = sorted(set(payload) - ALLOWED_JOB_FIELDS)
        if unknown:
            raise AdminTrainingValidationError(f"unknown field(s): {', '.join(unknown)}")
        dry_run = payload.get("dry_run", False)
        if not isinstance(dry_run, bool):
            raise AdminTrainingValidationError("dry_run must be a boolean")
        device = payload.get("device", "auto")
        if device not in self.settings.admin_training_allowed_devices:
            allowed = ", ".join(self.settings.admin_training_allowed_devices)
            raise AdminTrainingValidationError(f"device must be one of: {allowed}")
        batch_size = payload.get("batch_size", 16)
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise AdminTrainingValidationError("batch_size must be an integer")
        if batch_size < 1 or batch_size > self.settings.admin_training_max_batch_size:
            raise AdminTrainingValidationError(
                f"batch_size must be between 1 and {self.settings.admin_training_max_batch_size}"
            )
        notes = payload.get("notes", "")
        if notes is None:
            notes = ""
        if not isinstance(notes, str):
            raise AdminTrainingValidationError("notes must be a string")
        if len(notes) > 200:
            raise AdminTrainingValidationError("notes must be 200 characters or fewer")
        return {"dry_run": dry_run, "device": device, "batch_size": batch_size, "notes": notes}

    def _hydrate_job(self, status_path: Path) -> dict[str, Any] | None:
        status = _read_json(status_path)
        if status is None:
            return None
        run_id = str(status.get("run_id") or "")
        if status.get("status") == "running":
            process = self._processes.get(run_id)
            poll = process.poll() if process is not None and hasattr(process, "poll") else None
            if poll is not None:
                status["status"] = "succeeded" if poll == 0 else "failed"
                status["exit_code"] = poll
                status["finished_at"] = utc_now_iso()
                self._processes.pop(run_id, None)
                _write_json_atomic(status_path, status)
            elif process is None:
                pid = status.get("pid")
                pid_alive = _process_is_alive(pid)
                status_pid = _parse_positive_pid(pid)
                lock_pid = _read_lock_pid(Path(status.get("lock_path") or self.settings.backend_root / ".retrain.lock"))
                expected_identity = status.get("process_identity")
                current_identity = _process_identity(status_pid) if status_pid else None
                dry_run_without_handle = bool(status.get("dry_run"))
                full_run_lock_missing = not bool(status.get("dry_run")) and (status_pid is None or lock_pid != status_pid)
                full_run_identity_unverified = not bool(status.get("dry_run")) and (
                    not expected_identity or current_identity is None or current_identity != expected_identity
                )
                if (
                    dry_run_without_handle
                    or pid_alive is False
                    or status_pid is None
                    or full_run_lock_missing
                    or full_run_identity_unverified
                ):
                    status["status"] = "failed"
                    status["exit_code"] = None
                    status["finished_at"] = utc_now_iso()
                    status["error"] = "training process is no longer running; marked failed after backend lost process handle"
                    _write_json_atomic(status_path, status)
        log_path = Path(status.get("log_path") or status_path.parent / "train.log")
        status["log_tail"] = _tail_lines(log_path, self.settings.admin_training_log_tail_lines)
        status["artifacts"] = self._artifacts()
        return status

    def _parameters(self) -> dict[str, Any]:
        config_path = self.settings.admin_training_config_path
        config: dict[str, Any] = {}
        if config_path.is_file():
            loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                config = loaded
        return {
            "editable": {
                "dry_run": True,
                "device": list(self.settings.admin_training_allowed_devices),
                "batch_size": {"default": 16, "min": 1, "max": self.settings.admin_training_max_batch_size},
            },
            "script_env_defaults": {"BATCH_SIZE": "16", "DEVICE": "auto"},
            "config": {key: config.get(key, {}) for key in ["data", "model", "training", "evaluation", "inference"]},
        }

    def _paths(self) -> dict[str, Any]:
        approved_root = self.settings.backend_root / "training_data" / "approved"
        registry = self._registry_summary()
        candidate_version = registry["aliases"].get("candidate")
        return {
            "script": str(self.script_path),
            "promote_script": str(self.repo_root / "scripts" / "promote_model.py"),
            "repo_root": str(self.repo_root),
            "annotations_dir": str(self.settings.annotations_dir),
            "training_snapshot_dir": (
                str(self.settings.admin_training_snapshot_path)
                if self.settings.admin_training_snapshot_path
                else None
            ),
            "training_backbone_manifest": str(
                self.settings.admin_training_backbone_manifest_path
            ),
            "training_config": str(self.settings.admin_training_config_path),
            "approved_elements_dir": str(approved_root / "Elements"),
            "metadata_csv": str(approved_root / "metadata.csv"),
            "features_file": str(approved_root / "precomputed" / "features.pt"),
            "checkpoint_dir": str(self.settings.backend_root / "checkpoints"),
            "prototype_file": str(self.settings.backend_root / "prototypes" / "prototypes.pt"),
            "weights_dir": str(self.settings.classifier_weights_dir),
            "effective_classifier_weights_dir": str(self.settings.classifier_weights_dir),
            "model_dir_override_active": bool(self.settings.model_dir),
            "runs_dir": str(self.runs_dir),
            "model_registry_dir": str(self.settings.model_registry_dir),
            "model_registry_versions_dir": str(self.model_registry.versions_dir),
            "model_registry_snapshots_dir": str(self.model_registry.snapshots_dir),
            "candidate_version_dir": str(self.model_registry.version_dir(candidate_version)) if candidate_version else None,
            "current_promoted_version": registry["promoted_version"],
            "original_version": registry["original_version"],
        }

    def _approved_manifest_path(self) -> Path:
        return self.settings.backend_root / "training_data" / "approved" / "Elements" / "_approved_export_manifest.json"

    def _training_snapshot_info(self) -> dict[str, Any]:
        snapshot_dir = self.settings.admin_training_snapshot_path
        paths = {
            "manifest": str(snapshot_dir / "snapshot_manifest.json") if snapshot_dir else None,
            "metadata": str(snapshot_dir / "metadata.csv") if snapshot_dir else None,
            "checksums": str(snapshot_dir / "checksums.json") if snapshot_dir else None,
            "elements": str(snapshot_dir / "Elements") if snapshot_dir else None,
        }
        info: dict[str, Any] = {
            "configured": snapshot_dir is not None,
            "valid": False,
            "snapshot_id": None,
            "snapshot_manifest_sha256": None,
            "row_count": None,
            "class_count": None,
            "live_annotation_count": None,
            "live_annotations_sha256": None,
            "ready_for_training": False,
            "promotion_evaluation_ready": False,
            "split_counts": None,
            "paths": paths,
            "errors": [],
        }
        errors: list[str] = info["errors"]
        if snapshot_dir is None:
            errors.append(
                "training_snapshot_not_configured: set ADMIN_TRAINING_SNAPSHOT_DIR"
            )
            return info

        manifest_path = snapshot_dir / "snapshot_manifest.json"
        metadata_path = snapshot_dir / "metadata.csv"
        checksums_path = snapshot_dir / "checksums.json"
        elements_path = snapshot_dir / "Elements"
        required = {
            "training_snapshot_manifest_missing": manifest_path.is_file(),
            "training_snapshot_metadata_missing": metadata_path.is_file(),
            "training_snapshot_checksums_missing": checksums_path.is_file(),
            "training_snapshot_elements_missing": elements_path.is_dir(),
            "training_backbone_manifest_missing": self.settings.admin_training_backbone_manifest_path.is_file(),
            "training_snapshot_config_missing": self.settings.admin_training_config_path.is_file(),
            "training_warmstart_projection_missing": (
                self.settings.classifier_weights_dir / "projection.pt"
            ).is_file(),
        }
        errors.extend(name for name, present in required.items() if not present)
        if not manifest_path.is_file():
            return info

        manifest = _read_json(manifest_path)
        if manifest is None or manifest.get("schema_version") != "training-snapshot.v2":
            errors.append("training_snapshot_manifest_invalid")
            return info

        info.update(
            {
                "snapshot_id": manifest.get("snapshot_id"),
                "snapshot_manifest_sha256": _sha256_file(manifest_path),
                "row_count": manifest.get("row_count"),
                "class_count": manifest.get("class_count"),
                "live_annotation_count": manifest.get("live_annotation_count"),
                "live_annotations_sha256": manifest.get(
                    "live_annotations_sha256"
                ),
                "ready_for_training": manifest.get("ready_for_training") is True,
                "promotion_evaluation_ready": (
                    manifest.get("promotion_evaluation_ready") is True
                ),
                "split_counts": manifest.get("split_counts"),
            }
        )
        if not info["ready_for_training"]:
            errors.append("training_snapshot_not_ready")

        try:
            manifest_live_records = _manifest_live_annotation_records(manifest)
        except ValueError:
            errors.append("training_snapshot_live_annotations_invalid")
            manifest_live_records = []
        if (
            not isinstance(info["live_annotation_count"], int)
            or info["live_annotation_count"] != len(manifest_live_records)
            or not isinstance(info["live_annotations_sha256"], str)
            or info["live_annotations_sha256"]
            != _live_annotations_sha256(manifest_live_records)
        ):
            errors.append("training_snapshot_live_annotations_invalid")

        runtime_config = _read_json(self.settings.class_config_path)
        runtime_classes = runtime_config.get("class_names") if runtime_config else None
        if not isinstance(runtime_classes, list) or manifest.get("class_order") != runtime_classes:
            errors.append("training_snapshot_runtime_class_order_mismatch")

        if metadata_path.is_file() and checksums_path.is_file():
            checksums = _read_json(checksums_path)
            if (
                checksums is None
                or checksums.get("schema_version") != "training-snapshot-checksums.v1"
                or checksums.get("snapshot_id") != info["snapshot_id"]
                or checksums.get("snapshot_manifest_sha256")
                != info["snapshot_manifest_sha256"]
                or checksums.get("metadata_csv_sha256") != _sha256_file(metadata_path)
            ):
                errors.append("training_snapshot_checksums_invalid")

        review_sha = _sha256_file(self.review_store.manifest_path)
        sources = manifest.get("source_manifests")
        review_sources = (
            [
                source
                for source in sources
                if isinstance(source, dict)
                and source.get("kind") == "annotation-review-index.v1"
            ]
            if isinstance(sources, list)
            else []
        )
        empty_review_store = (
            review_sha is None
            and not review_sources
            and info["live_annotation_count"] == 0
        )
        if not empty_review_store and (
            review_sha is None
            or not any(source.get("sha256") == review_sha for source in review_sources)
        ):
            errors.append(
                "training_snapshot_stale: rebuild it from the current approved annotations"
            )

        try:
            current_live_records = []
            for item in self.review_store.iter_approved_annotations():
                crop_sha = _sha256_file(Path(str(item.get("crop_path") or "")))
                image_sha = _sha256_file(Path(str(item.get("image_path") or "")))
                if crop_sha is None or image_sha is None:
                    raise OSError("approved annotation source is missing")
                current_live_records.append(
                    {
                        "source_id": f"{item.get('analysis_id')}:{item.get('index')}",
                        "class_name": item.get("class_name"),
                        "bbox": item.get("bbox"),
                        "source_fingerprint_v1": item.get("source_fingerprint"),
                        "source_sha256": crop_sha,
                        "source_image_sha256": image_sha,
                    }
                )
            live_set_matches = (
                info["live_annotation_count"] == len(current_live_records)
                and info["live_annotations_sha256"]
                == _live_annotations_sha256(current_live_records)
            )
        except Exception as exc:
            errors.append(f"review_store_unavailable: {exc}")
            live_set_matches = False
        if not live_set_matches and not any(
            reason.startswith("training_snapshot_stale:") for reason in errors
        ):
            errors.append(
                "training_snapshot_stale: rebuild it from the current approved annotations"
            )

        info["errors"] = list(dict.fromkeys(errors))
        info["valid"] = not info["errors"]
        return info

    def _artifacts(self) -> dict[str, Any]:
        return {
            "approved_export_manifest": {**_file_info(self._approved_manifest_path()), "sha256": _sha256_file(self._approved_manifest_path())},
            "training_snapshot": self._training_snapshot_info(),
            "prototypes": _file_info(self.settings.backend_root / "prototypes" / "prototypes.pt"),
            "classifier_prototypes": _file_info(self.settings.classifier_weights_dir / "prototypes.pt"),
            "classifier_projection": _file_info(self.settings.classifier_weights_dir / "projection.pt"),
            "classifier_config": _file_info(self.settings.backend_root / "codex_model" / "config.json"),
            "model_registry": self._registry_summary(),
        }

    def _registry_summary(self) -> dict[str, Any]:
        index_exists = self.model_registry.index_path.is_file()
        try:
            index = self.model_registry.read_index()
        except ModelRegistryValidationError as exc:
            return {
                "index_exists": index_exists,
                "index_path": str(self.model_registry.index_path),
                "status": "unhealthy",
                "error": str(exc),
                "aliases": {"original": None, "candidate": None, "promoted": None, "previous": None},
                "promoted_version": None,
                "original_version": None,
                "latest_candidate": None,
                "promoted": None,
                "original": None,
            }
        aliases = index.get("aliases", {})
        candidate = aliases.get("candidate")
        promoted = index.get("promoted_version") or aliases.get("promoted")
        original = index.get("original_version") or aliases.get("original")
        promotion_marker = _read_json(self.model_registry.root / "promotion_in_progress.json")
        status = "promotion_in_progress" if promotion_marker else ("ok" if index_exists else "not_initialized")
        return {
            "index_exists": index_exists,
            "index_path": str(self.model_registry.index_path),
            "status": status,
            "promotion_in_progress": promotion_marker,
            "aliases": aliases,
            "promoted_version": promoted,
            "original_version": original,
            "version_count": len(index.get("versions", {})),
            "latest_candidate": self._version_summary(candidate) if candidate else None,
            "promoted": self._version_summary(promoted) if promoted else None,
            "original": self._version_summary(original) if original else None,
        }

    def _version_summary(self, version_id: str) -> dict[str, Any]:
        version_dir = self.model_registry.version_dir(version_id)
        manifest_path = version_dir / "manifest.json"
        checksums_path = version_dir / "checksums.sha256"
        health = self._manifest_health(version_id)
        return {
            "version_id": version_id,
            "version_dir": str(version_dir),
            "manifest": _file_info(manifest_path),
            "checksums": _file_info(checksums_path),
            "manifest_health": health,
        }

    def _manifest_health(self, version_id: str) -> dict[str, Any]:
        version_dir = self.model_registry.version_dir(version_id)
        manifest_path = version_dir / "manifest.json"
        manifest = _read_json(manifest_path)
        if manifest is None:
            return {"status": "missing", "errors": [f"missing manifest: {manifest_path}"]}
        errors: list[str] = []
        if manifest.get("schema_version") != SCHEMA_VERSION:
            errors.append(f"unsupported manifest schema_version: {manifest.get('schema_version')}")
        if manifest.get("model_id") != self.model_registry.model_id:
            errors.append(f"manifest model_id mismatch: {manifest.get('model_id')}")
        index = self.model_registry.read_index()
        version_meta = index.get("versions", {}).get(version_id)
        if isinstance(version_meta, dict):
            expected_manifest_sha = version_meta.get("manifest_sha256")
            expected_checksums_sha = version_meta.get("checksums_sha256")
            if expected_manifest_sha:
                actual_manifest_sha = sha256_file(manifest_path)
                if actual_manifest_sha != expected_manifest_sha:
                    errors.append("index manifest_sha256 mismatch")
            else:
                errors.append("index missing manifest_sha256")
            checksums_path = version_dir / "checksums.sha256"
            if expected_checksums_sha and checksums_path.is_file():
                actual_checksums_sha = sha256_file(checksums_path)
                if actual_checksums_sha != expected_checksums_sha:
                    errors.append("index checksums_sha256 mismatch")
            elif not expected_checksums_sha:
                errors.append("index missing checksums_sha256")
        else:
            errors.append("version missing from index")
        checksums_records: dict[str, str] = {}
        checksums_path = version_dir / "checksums.sha256"
        if checksums_path.is_file():
            for line_number, raw_line in enumerate(checksums_path.read_text(encoding="utf-8").splitlines(), start=1):
                line = raw_line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    errors.append(f"invalid checksums.sha256 line {line_number}")
                    continue
                try:
                    rel = safe_relative_path(parts[1].strip())
                except ModelRegistryValidationError as exc:
                    errors.append(str(exc))
                    continue
                checksums_records[rel.as_posix()] = parts[0]
        else:
            errors.append("missing checksums.sha256")
        artifacts = manifest.get("artifacts")
        manifest_paths: set[str] = set()
        if not isinstance(artifacts, list):
            errors.append("manifest artifacts is not a list")
        else:
            for artifact in artifacts:
                if not isinstance(artifact, dict):
                    errors.append("manifest artifact entry is not an object")
                    continue
                try:
                    rel = safe_relative_path(str(artifact.get("path", "")))
                except ModelRegistryValidationError as exc:
                    errors.append(str(exc))
                    continue
                manifest_paths.add(rel.as_posix())
                artifact_path = version_dir / rel
                if not artifact_path.is_file():
                    errors.append(f"missing artifact: {rel.as_posix()}")
                    continue
                expected = str(artifact.get("sha256") or "")
                actual = sha256_file(artifact_path)
                if expected and expected != actual:
                    errors.append(f"checksum mismatch: {rel.as_posix()}")
                checksum_expected = checksums_records.get(rel.as_posix())
                if checksum_expected != expected:
                    errors.append(f"checksums.sha256 mismatch: {rel.as_posix()}")
        checksum_only = sorted(set(checksums_records) - manifest_paths)
        if checksum_only:
            errors.append("checksums.sha256 contains artifact(s) missing from manifest: " + ", ".join(checksum_only))
        return {"status": "healthy" if not errors else "unhealthy", "errors": errors}

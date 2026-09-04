#!/usr/bin/env python3
"""Promote or roll back immutable local classifier versions.

This is the only sanctioned writer to ``backend/codex_model`` for the versioned
retraining workflow.  Training/export produces immutable candidate packages
under ``backend/model_registry/versions/<version_id>/``; this script validates a
package, snapshots the current runtime files, atomically replaces runtime files,
verifies checksums after copy, and records the pointer update in the registry
index.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.services.model_registry import (  # noqa: E402
    ModelRegistry,
    ModelRegistryValidationError,
    SCHEMA_VERSION,
    atomic_write_json,
    read_json_object,
    safe_relative_path,
    sha256_file,
    utc_now_iso,
)
from scripts.evaluate_r2_e2e import E2EEvaluationError, validate_promotion_report  # noqa: E402

REQUIRED_RUNTIME_FILES = {
    "runtime/weights/prototypes.pt": ("weights", "prototypes.pt"),
    "runtime/weights/projection.pt": ("weights", "projection.pt"),
    "runtime/config.json": ("config.json",),
}
PROMOTION_MARKER = "promotion_in_progress.json"


class PromotionError(RuntimeError):
    """Raised when promotion/rollback cannot safely proceed."""


@contextmanager
def promotion_lock(registry: ModelRegistry):
    lock_dir = registry.root / ".locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / "promotion.lock"
    fd: int | None = None
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"pid={os.getpid()} at={utc_now_iso()}\n".encode("utf-8"))
        yield lock_path
    except FileExistsError as exc:
        raise PromotionError(f"promotion lock already exists: {lock_path}") from exc
    finally:
        if fd is not None:
            os.close(fd)
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass


def _load_manifest(registry: ModelRegistry, version_id: str) -> dict[str, Any]:
    safe_relative_path(version_id)
    index = registry.read_index()
    version_meta = index["versions"].get(version_id)
    if not isinstance(version_meta, dict):
        raise PromotionError(f"version is not registered in index.json: {version_id}")
    manifest_path = registry.version_dir(version_id) / "manifest.json"
    expected_manifest_sha = version_meta.get("manifest_sha256")
    if not expected_manifest_sha:
        raise PromotionError(f"index is missing manifest_sha256 for {version_id}")
    actual_manifest_sha = sha256_file(manifest_path)
    if expected_manifest_sha != actual_manifest_sha:
        raise PromotionError(
            f"manifest digest mismatch for {version_id}: expected {expected_manifest_sha}, got {actual_manifest_sha}"
        )
    checksums_path = registry.version_dir(version_id) / "checksums.sha256"
    expected_checksums_sha = version_meta.get("checksums_sha256")
    if not expected_checksums_sha:
        raise PromotionError(f"index is missing checksums_sha256 for {version_id}")
    actual_checksums_sha = sha256_file(checksums_path)
    if expected_checksums_sha != actual_checksums_sha:
        raise PromotionError(
            f"checksums digest mismatch for {version_id}: expected {expected_checksums_sha}, got {actual_checksums_sha}"
        )
    manifest = read_json_object(manifest_path)
    if manifest is None:
        raise PromotionError(f"manifest not found: {manifest_path}")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise PromotionError(
            f"unsupported manifest schema_version for {version_id}: {manifest.get('schema_version')}"
        )
    if manifest.get("model_id") != registry.model_id:
        raise PromotionError(
            f"manifest model_id mismatch for {version_id}: expected {registry.model_id}, got {manifest.get('model_id')}"
        )
    if manifest.get("version_id") != version_id:
        raise PromotionError(f"manifest version_id mismatch for {version_id}")
    return manifest


def _read_checksums_file(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    if not path.is_file():
        raise PromotionError(f"checksums.sha256 not found: {path}")
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise PromotionError(f"invalid checksums.sha256 line {line_number}: {raw_line!r}")
        digest, rel_path = parts
        rel = safe_relative_path(rel_path.strip())
        records[rel.as_posix()] = digest
    return records


def _validate_manifest_artifacts(registry: ModelRegistry, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    version_id = str(manifest["version_id"])
    version_dir = registry.version_dir(version_id)
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise PromotionError("manifest artifacts must be a list")
    records: list[dict[str, Any]] = []
    seen = set()
    checksums_records = _read_checksums_file(version_dir / "checksums.sha256")
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise PromotionError("manifest artifact entries must be objects")
        rel = safe_relative_path(str(artifact.get("path", "")))
        path = version_dir / rel
        resolved_version_dir = version_dir.resolve()
        try:
            resolved_path = path.resolve(strict=True)
            resolved_path.relative_to(resolved_version_dir)
        except (FileNotFoundError, OSError, ValueError) as exc:
            raise PromotionError(f"manifest artifact escapes version directory: {rel.as_posix()}") from exc
        cursor = path
        while cursor != version_dir:
            if cursor.is_symlink():
                raise PromotionError(f"manifest artifact path contains a symlink: {rel.as_posix()}")
            cursor = cursor.parent
        if not path.is_file():
            raise PromotionError(f"manifest artifact missing: {rel.as_posix()}")
        expected = str(artifact.get("sha256") or "")
        actual = sha256_file(path)
        if expected != actual:
            raise PromotionError(f"checksum mismatch for {rel.as_posix()}: expected {expected}, got {actual}")
        checksum_expected = checksums_records.get(rel.as_posix())
        if checksum_expected != expected:
            raise PromotionError(
                f"checksums.sha256 mismatch for {rel.as_posix()}: expected {expected}, got {checksum_expected}"
            )
        records.append({"path": rel.as_posix(), "sha256": actual, "size": path.stat().st_size})
        seen.add(rel.as_posix())
    checksum_only = sorted(set(checksums_records) - seen)
    if checksum_only:
        raise PromotionError(f"checksums.sha256 contains artifact(s) missing from manifest: {', '.join(checksum_only)}")
    for required in REQUIRED_RUNTIME_FILES:
        required_path = version_dir / safe_relative_path(required)
        if not required_path.is_file():
            raise PromotionError(f"required runtime artifact missing: {required}")
        if required not in seen:
            raise PromotionError(f"required runtime artifact missing from manifest: {required}")
    return records


def _read_class_order(config_path: Path, *, label: str) -> list[str]:
    config = read_json_object(config_path)
    if config is None:
        raise PromotionError(f"{label} runtime config is missing or invalid JSON: {config_path}")
    class_names = config.get("class_names")
    if not isinstance(class_names, list) or not class_names:
        raise PromotionError(f"{label} runtime config must contain a non-empty class_names list")
    if any(not isinstance(name, str) or not name for name in class_names):
        raise PromotionError(f"{label} runtime class_names must contain non-empty strings")
    if len(set(class_names)) != len(class_names):
        raise PromotionError(f"{label} runtime class_names contains duplicates")
    num_classes = config.get("num_classes", len(class_names))
    if num_classes != len(class_names):
        raise PromotionError(
            f"{label} runtime num_classes mismatch: {num_classes} != {len(class_names)}"
        )
    return class_names


def _read_numeric_label_contract(config_path: Path, prototypes_path: Path, *, label: str) -> dict[str, int]:
    class_names = _read_class_order(config_path, label=label)
    try:
        data = torch.load(prototypes_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise PromotionError(f"{label} prototype artifact cannot be safely loaded: {prototypes_path}") from exc
    if not isinstance(data, dict):
        raise PromotionError(f"{label} prototype artifact must be a mapping")
    raw_labels, raw_names = data.get("class_labels"), data.get("class_names")
    if not isinstance(raw_labels, torch.Tensor) or raw_labels.ndim != 1 or raw_labels.numel() != len(class_names):
        raise PromotionError(f"{label} prototype class_labels do not match config")
    if raw_labels.dtype not in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}:
        raise PromotionError(f"{label} prototype class_labels must be integers")
    labels = [int(value) for value in raw_labels.tolist()]
    if labels != sorted(labels) or len(set(labels)) != len(labels):
        raise PromotionError(f"{label} prototype class_labels must be unique and sorted")
    if not isinstance(raw_names, dict) or set(raw_names) != set(labels):
        raise PromotionError(f"{label} prototype class_names and class_labels disagree")
    if any(
        isinstance(key, bool) or not isinstance(key, int) or not isinstance(value, str) or not value
        for key, value in raw_names.items()
    ):
        raise PromotionError(f"{label} prototype class_names mapping is invalid")
    names = [raw_names[numeric_label] for numeric_label in labels]
    if names != class_names:
        raise PromotionError(f"{label} prototype taxonomy does not match config class_names")
    return dict(zip(names, labels))


def _validate_runtime_class_order(registry: ModelRegistry, version_id: str) -> None:
    target_path = registry.version_dir(version_id) / "runtime" / "config.json"
    target_order = _read_class_order(target_path, label="candidate")
    index = registry.read_index()
    original_version = index["aliases"].get("original")
    if original_version:
        reference_path = registry.version_dir(original_version) / "runtime" / "config.json"
        reference_prototypes_path = registry.version_dir(original_version) / "runtime" / "weights" / "prototypes.pt"
    else:
        missing = [
            str(registry.runtime_model_dir.joinpath(*runtime_parts))
            for runtime_parts in REQUIRED_RUNTIME_FILES.values()
            if not registry.runtime_model_dir.joinpath(*runtime_parts).is_file()
        ]
        if missing:
            raise PromotionError("missing required artifact(s) in current runtime: " + ", ".join(missing))
        reference_path = registry.runtime_model_dir / "config.json"
        reference_prototypes_path = registry.runtime_model_dir / "weights" / "prototypes.pt"
    reference_order = _read_class_order(reference_path, label="reference")
    if target_order != reference_order:
        raise PromotionError(
            "candidate runtime class order differs from the historical ABI "
            f"({len(target_order)} candidate classes vs {len(reference_order)} reference classes)"
        )
    target_contract = _read_numeric_label_contract(
        target_path,
        registry.version_dir(version_id) / "runtime" / "weights" / "prototypes.pt",
        label="candidate",
    )
    reference_contract = _read_numeric_label_contract(
        reference_path,
        reference_prototypes_path,
        label="reference",
    )
    if target_contract != reference_contract:
        raise PromotionError(
            "candidate numeric class label mapping differs from the historical ABI"
        )


def _validate_promotion_policy(manifest: dict[str, Any], *, action: str) -> bool:
    """Reject explicit gates and return whether the existing R2 E2E guard applies."""
    if action != "promote":
        return False
    promotion = manifest.get("promotion")
    if not isinstance(promotion, dict):
        return False
    if "blocked" in promotion and promotion.get("blocked") is not False:
        reason = promotion.get("reason")
        detail = reason if isinstance(reason, str) and reason.strip() else "no unblock contract is recorded"
        raise PromotionError(f"candidate promotion is explicitly blocked: {detail}")
    return bool(promotion.get("e2e_report_required"))


def _copy_atomic(source: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        shutil.copy2(source, tmp)
        os.replace(tmp, dest)
    finally:
        if tmp.exists():
            tmp.unlink()


def _snapshot_runtime(registry: ModelRegistry, *, label: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    snapshot_dir = registry.snapshots_dir / f"{timestamp}-{label}"
    counter = 1
    while snapshot_dir.exists():
        counter += 1
        snapshot_dir = registry.snapshots_dir / f"{timestamp}-{label}-{counter}"
    runtime = registry.runtime_model_dir
    files: list[dict[str, Any]] = []
    missing: list[str] = []
    for _rel, runtime_parts in REQUIRED_RUNTIME_FILES.items():
        source = runtime.joinpath(*runtime_parts)
        runtime_rel = Path(*runtime_parts)
        if source.is_file():
            dest = snapshot_dir / "runtime" / runtime_rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
            files.append({"path": runtime_rel.as_posix(), "existed": True, "sha256": sha256_file(source)})
        else:
            missing.append(str(source))
            files.append({"path": runtime_rel.as_posix(), "existed": False, "sha256": None})
    if missing:
        raise PromotionError("cannot snapshot incomplete runtime; missing: " + ", ".join(missing))
    atomic_write_json(
        snapshot_dir / "snapshot-manifest.json",
        {"created_at": utc_now_iso(), "runtime_dir": str(runtime), "files": files},
    )
    return snapshot_dir


def _restore_snapshot(registry: ModelRegistry, snapshot_dir: Path) -> None:
    snapshot_manifest = read_json_object(snapshot_dir / "snapshot-manifest.json")
    manifest_files = snapshot_manifest.get("files") if isinstance(snapshot_manifest, dict) else None
    if not isinstance(manifest_files, list):
        raise PromotionError(f"snapshot manifest missing or invalid: {snapshot_dir / 'snapshot-manifest.json'}")
    for _rel, runtime_parts in REQUIRED_RUNTIME_FILES.items():
        runtime_rel = Path(*runtime_parts)
        dest = registry.runtime_model_dir / runtime_rel
        record = next(
            (item for item in manifest_files if isinstance(item, dict) and item.get("path") == runtime_rel.as_posix()),
            None,
        )
        if record is None:
            raise PromotionError(f"snapshot manifest lacks runtime file entry: {runtime_rel.as_posix()}")
        source = snapshot_dir / "runtime" / runtime_rel
        if record.get("existed"):
            _copy_atomic(source, dest)
            expected = record.get("sha256")
            if expected and sha256_file(dest) != expected:
                raise PromotionError(f"snapshot restore checksum mismatch: {dest}")
        elif dest.exists():
            dest.unlink()


def _write_promotion_marker(
    registry: ModelRegistry,
    *,
    action: str,
    version_id: str,
    previous_version: str | None,
) -> Path:
    marker_path = registry.root / PROMOTION_MARKER
    atomic_write_json(
        marker_path,
        {
            "action": action,
            "version_id": version_id,
            "previous_version": previous_version,
            "runtime_dir": str(registry.runtime_model_dir),
            "started_at": utc_now_iso(),
            "recovery": (
                "A previous promotion was interrupted. Verify runtime files against "
                "the promoted_version in index.json or rerun scripts/promote_model.py "
                "for the intended version."
            ),
        },
    )
    return marker_path


def _clear_promotion_marker(registry: ModelRegistry) -> None:
    try:
        (registry.root / PROMOTION_MARKER).unlink()
    except FileNotFoundError:
        pass


def _runtime_copy_plan(registry: ModelRegistry, version_id: str) -> list[dict[str, str]]:
    version_dir = registry.version_dir(version_id)
    plan = []
    for rel, runtime_parts in REQUIRED_RUNTIME_FILES.items():
        source = version_dir / safe_relative_path(rel)
        dest = registry.runtime_model_dir.joinpath(*runtime_parts)
        plan.append({"source": str(source), "dest": str(dest), "relative_path": rel})
    return plan


def _append_history(
    registry: ModelRegistry,
    *,
    action: str,
    status: str,
    version_id: str,
    previous_version: str | None,
    snapshot_dir: Path | None,
    error: str | None = None,
) -> dict[str, Any]:
    index = registry.read_index()
    event = {
        "action": action,
        "status": status,
        "version_id": version_id,
        "previous_version": previous_version,
        "snapshot_path": str(snapshot_dir) if snapshot_dir else None,
        "created_at": utc_now_iso(),
        "recovery": (
            "If promotion failed after a partial copy, runtime files were restored from "
            "snapshot_path. Verify checksums before retrying."
        ),
    }
    if error:
        event["error"] = error
    index["promotion_history"].append(event)
    return registry.write_index(index)


def _record_success(
    registry: ModelRegistry,
    *,
    action: str,
    version_id: str,
    previous_version: str | None,
    snapshot_dir: Path,
) -> dict[str, Any]:
    index = registry.read_index()
    if previous_version and previous_version != version_id:
        index["aliases"]["previous"] = previous_version
    index["aliases"]["promoted"] = version_id
    if action == "promote":
        index["aliases"]["candidate"] = None
    index["promoted_version"] = version_id
    version_status = "original" if version_id == index.get("original_version") else "promoted"
    version = index["versions"].get(version_id, {})
    version["status"] = version_status
    version["promoted_at"] = utc_now_iso()
    version["previous_version"] = previous_version
    version["snapshot_path"] = str(snapshot_dir)
    index["versions"][version_id] = version
    index["promotion_history"].append(
        {
            "action": action,
            "status": "succeeded",
            "version_id": version_id,
            "previous_version": previous_version,
            "snapshot_path": str(snapshot_dir),
            "created_at": utc_now_iso(),
        }
    )
    manifest_path = registry.version_dir(version_id) / "manifest.json"
    manifest = read_json_object(manifest_path)
    if manifest is not None:
        manifest["status"] = version_status
        promotion = manifest.get("promotion") if isinstance(manifest.get("promotion"), dict) else {}
        promotion.update(
            {
                "action": action,
                "promoted_at": utc_now_iso(),
                "previous_version": previous_version,
                "backup_snapshot": str(snapshot_dir),
            }
        )
        manifest["promotion"] = promotion
        atomic_write_json(manifest_path, manifest)
    version["manifest_sha256"] = sha256_file(manifest_path)
    version["checksums_sha256"] = sha256_file(registry.version_dir(version_id) / "checksums.sha256")
    index["versions"][version_id] = version
    return registry.write_index(index)


def resolve_rollback_target(registry: ModelRegistry, requested: str | None) -> str:
    if requested:
        return requested
    index = registry.read_index()
    return index["aliases"].get("previous") or index["aliases"].get("original") or ""


def promote_or_rollback(
    *,
    registry: ModelRegistry,
    version_id: str,
    dry_run: bool,
    action: str,
    e2e_report_path: Path | None = None,
) -> dict[str, Any]:
    if not version_id:
        raise PromotionError("no target version available")
    manifest = _load_manifest(registry, version_id)
    artifacts = _validate_manifest_artifacts(registry, manifest)
    _validate_runtime_class_order(registry, version_id)
    e2e_report: dict[str, Any] | None = None
    e2e_required = _validate_promotion_policy(manifest, action=action)
    if action == "promote" and e2e_required:
        if e2e_report_path is None:
            raise PromotionError(
                "candidate requires a passing E2E report; pass --e2e-report before promotion"
            )
        try:
            e2e_report = validate_promotion_report(
                e2e_report_path,
                manifest=manifest,
                manifest_path=registry.version_dir(version_id) / "manifest.json",
                repo_root=registry.repo_root,
            )
        except E2EEvaluationError as exc:
            raise PromotionError(f"invalid E2E promotion report: {exc}") from exc
    index = registry.read_index()
    previous_version = index.get("promoted_version") or index["aliases"].get("promoted")
    plan = _runtime_copy_plan(registry, version_id)

    result: dict[str, Any] = {
        "ok": True,
        "action": action,
        "dry_run": dry_run,
        "version_id": version_id,
        "previous_version": previous_version,
        "copy_plan": plan,
        "artifact_count": len(artifacts),
        "runtime_dir": str(registry.runtime_model_dir),
    }
    if e2e_report is not None:
        result["e2e_report"] = {
            "path": str(e2e_report_path),
            "schema_version": e2e_report["schema_version"],
            "report_payload_sha256": e2e_report["integrity"]["report_payload_sha256"],
            "overall_pass": True,
        }
    if dry_run:
        result["would_import_original"] = not bool(index["aliases"].get("original"))
        result["would_snapshot_runtime"] = True
        result["message"] = "dry run only; runtime files and index were not modified"
        return result

    with promotion_lock(registry):
        # Re-validate every immutable binding inside the lock. The manifest,
        # its artifacts, and the E2E report were read before the lock was
        # acquired; an operator or concurrent process could have mutated them
        # in between. Re-loading here closes that TOCTOU window before any
        # runtime file is copied.
        manifest = _load_manifest(registry, version_id)
        artifacts = _validate_manifest_artifacts(registry, manifest)
        _validate_runtime_class_order(registry, version_id)
        e2e_required = _validate_promotion_policy(manifest, action=action)
        if action == "promote" and e2e_required:
            if e2e_report_path is None:
                raise PromotionError(
                    "candidate requires a passing E2E report; pass --e2e-report before promotion"
                )
            try:
                e2e_report = validate_promotion_report(
                    e2e_report_path,
                    manifest=manifest,
                    manifest_path=registry.version_dir(version_id) / "manifest.json",
                    repo_root=registry.repo_root,
                )
            except E2EEvaluationError as exc:
                raise PromotionError(f"invalid E2E promotion report: {exc}") from exc
        plan = _runtime_copy_plan(registry, version_id)
        result["copy_plan"] = plan
        result["artifact_count"] = len(artifacts)
        if not registry.read_index()["aliases"].get("original"):
            registry.import_current_runtime()
            previous_version = registry.read_index().get("promoted_version") or previous_version
            result["previous_version"] = previous_version

        marker_path = _write_promotion_marker(
            registry,
            action=action,
            version_id=version_id,
            previous_version=previous_version,
        )
        snapshot_dir: Path | None = None
        try:
            snapshot_dir = _snapshot_runtime(registry, label=f"pre-{action}")
            for item in plan:
                source = Path(item["source"])
                dest = Path(item["dest"])
                _copy_atomic(source, dest)
                if sha256_file(dest) != sha256_file(source):
                    raise PromotionError(f"runtime checksum mismatch after copy: {dest}")
            _record_success(
                registry,
                action=action,
                version_id=version_id,
                previous_version=previous_version,
                snapshot_dir=snapshot_dir,
            )
        except Exception as exc:
            if snapshot_dir is not None:
                _restore_snapshot(registry, snapshot_dir)
            _append_history(
                registry,
                action=action,
                status="failed",
                version_id=version_id,
                previous_version=previous_version,
                snapshot_dir=snapshot_dir,
                error=str(exc),
            )
            _clear_promotion_marker(registry)
            raise PromotionError(
                f"{action} failed; restored runtime files from {snapshot_dir or marker_path}. "
                "Verify checksums before retrying."
            ) from exc
        _clear_promotion_marker(registry)
        result["snapshot_path"] = str(snapshot_dir)
        result["message"] = f"{action} succeeded; restart backend to load {version_id}"
        return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Promote or roll back a local immutable classifier version.")
    parser.add_argument("version_id", nargs="?", help="Version to promote, or rollback target when --rollback is set.")
    parser.add_argument("--rollback", action="store_true", help="Restore previous/original or the supplied version.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the copy plan without mutating runtime files.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON output.")
    parser.add_argument("--registry-dir", default=str(REPO_ROOT / "backend" / "model_registry"))
    parser.add_argument("--runtime-dir", default=None)
    parser.add_argument(
        "--e2e-report",
        type=Path,
        default=None,
        help="Passing, hash-bound E2E report required by guarded candidates such as R2.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if os.environ.get("MODEL_DIR") and args.runtime_dir is None:
        message = (
            "MODEL_DIR is set, so the backend may load classifier weights outside "
            "backend/codex_model. Unset MODEL_DIR before promotion or pass "
            "--runtime-dir explicitly only for a matching custom runtime package."
        )
        if args.json:
            print(json.dumps({"ok": False, "action": "rollback" if args.rollback else "promote", "error": message}, indent=2, sort_keys=True))
        else:
            print(f"ERROR: {message}", file=sys.stderr)
        return 2
    runtime_dir = Path(args.runtime_dir) if args.runtime_dir else REPO_ROOT / "backend" / "codex_model"
    registry = ModelRegistry(
        Path(args.registry_dir),
        repo_root=REPO_ROOT,
        runtime_model_dir=runtime_dir,
    )
    action = "rollback" if args.rollback else "promote"
    version_id = resolve_rollback_target(registry, args.version_id) if args.rollback else (args.version_id or "")
    try:
        result = promote_or_rollback(
            registry=registry,
            version_id=version_id,
            dry_run=args.dry_run,
            action=action,
            e2e_report_path=args.e2e_report,
        )
    except (PromotionError, ModelRegistryValidationError) as exc:
        if args.json:
            print(json.dumps({"ok": False, "action": action, "error": str(exc)}, indent=2, sort_keys=True))
        else:
            print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(result["message"])
        for item in result["copy_plan"]:
            print(f"  {item['source']} -> {item['dest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

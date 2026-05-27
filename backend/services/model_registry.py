"""Local immutable model registry helpers.

The registry is intentionally small and dependency-light.  Training/export
code writes candidate packages under ``backend/model_registry/versions`` and
runtime files remain unchanged until the promotion tool explicitly copies a
validated candidate into ``backend/codex_model``.
"""
from __future__ import annotations

import getpass
import hashlib
import json
import os
import re
import shutil
import subprocess
import uuid
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

MODEL_ID = "codex_classifier"
SCHEMA_VERSION = 1
DEFAULT_ALIASES = ("original", "candidate", "promoted", "previous")
RUNTIME_ARTIFACTS = (
    ("runtime/weights/prototypes.pt", ("weights", "prototypes.pt")),
    ("runtime/weights/projection.pt", ("weights", "projection.pt")),
    ("runtime/config.json", ("config.json",)),
)
_VERSION_PART_RE = re.compile(r"[^A-Za-z0-9_.-]+")


class ModelRegistryError(RuntimeError):
    """Base class for local registry errors."""


class ModelRegistryValidationError(ModelRegistryError, ValueError):
    """Raised when registry metadata or paths are unsafe."""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def atomic_write_json(path: Path, data: Mapping[str, Any]) -> None:
    """Write JSON through a same-directory temporary file and atomic replace."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        tmp.write_text(
            json.dumps(data, indent=2, sort_keys=True, default=_json_default) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def read_json_object(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ModelRegistryValidationError(f"invalid JSON: {path}") from exc
    if not isinstance(data, dict):
        raise ModelRegistryValidationError(f"expected JSON object: {path}")
    return data


def write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        tmp.write_text(content, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def sha256_file(path: Path) -> str:
    if not path.is_file():
        raise ModelRegistryValidationError(f"missing artifact: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_info(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "sha256": sha256_file(path),
        "size": stat.st_size,
        "mtime": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
    }


def safe_relative_path(value: str | Path) -> Path:
    """Return a safe relative path for manifest/checksum artifact entries."""

    raw = str(value).replace("\\", "/")
    pure = PurePosixPath(raw)
    if pure.is_absolute() or not raw or ".." in pure.parts:
        raise ModelRegistryValidationError(f"unsafe registry artifact path: {value}")
    return Path(*pure.parts)


def _sanitize_version_part(value: str, *, fallback: str) -> str:
    sanitized = _VERSION_PART_RE.sub("-", value.strip()).strip(".-")
    return sanitized or fallback


def build_version_id(
    *,
    created_at: datetime | None = None,
    git_commit: str | None = None,
    run_id: str | None = None,
) -> str:
    """Build sortable local version IDs: YYYYMMDDTHHMMSSZ-<gitshort>-<run8>."""

    now = created_at or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    timestamp = now.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    git_short = _sanitize_version_part((git_commit or "nogit")[:12], fallback="nogit")
    run8 = _sanitize_version_part((run_id or uuid.uuid4().hex)[:8], fallback=uuid.uuid4().hex[:8])
    return f"{timestamp}-{git_short}-{run8}"


def _merge_dict(base: dict[str, Any], overlay: Mapping[str, Any] | None) -> dict[str, Any]:
    if not overlay:
        return base
    for key, value in overlay.items():
        if isinstance(base.get(key), dict) and isinstance(value, Mapping):
            base[key] = _merge_dict(dict(base[key]), value)
        else:
            base[key] = value
    return base


class ModelRegistry:
    """Repo-local immutable classifier registry."""

    def __init__(
        self,
        registry_root: Path,
        *,
        repo_root: Path | None = None,
        runtime_model_dir: Path | None = None,
        model_id: str = MODEL_ID,
    ) -> None:
        self.root = Path(registry_root)
        self.repo_root = Path(repo_root) if repo_root is not None else self.root.parents[1]
        self.runtime_model_dir = (
            Path(runtime_model_dir) if runtime_model_dir is not None else self.root.parent / "codex_model"
        )
        self.model_id = model_id
        self.index_path = self.root / "index.json"
        self.versions_dir = self.root / "versions"
        self.snapshots_dir = self.root / "snapshots"

    def empty_index(self) -> dict[str, Any]:
        now = utc_now_iso()
        return {
            "schema_version": SCHEMA_VERSION,
            "model_id": self.model_id,
            "created_at": now,
            "updated_at": now,
            "promoted_version": None,
            "original_version": None,
            "aliases": {alias: None for alias in DEFAULT_ALIASES},
            "versions": {},
            "promotion_history": [],
        }

    def read_index(self) -> dict[str, Any]:
        index = read_json_object(self.index_path) or self.empty_index()
        return self._normalize_index(index)

    def write_index(self, index: Mapping[str, Any]) -> dict[str, Any]:
        normalized = self._normalize_index(dict(index))
        normalized["updated_at"] = utc_now_iso()
        atomic_write_json(self.index_path, normalized)
        return normalized

    def _normalize_index(self, index: dict[str, Any]) -> dict[str, Any]:
        schema_version = index.get("schema_version", SCHEMA_VERSION)
        if schema_version != SCHEMA_VERSION:
            raise ModelRegistryValidationError(
                f"unsupported model registry schema_version: {schema_version}"
            )
        index["schema_version"] = SCHEMA_VERSION
        model_id = index.get("model_id", self.model_id)
        if model_id != self.model_id:
            raise ModelRegistryValidationError(f"unexpected model registry model_id: {model_id}")
        index["model_id"] = self.model_id
        index.setdefault("created_at", utc_now_iso())
        index.setdefault("updated_at", index["created_at"])
        aliases = index.get("aliases")
        if not isinstance(aliases, dict):
            raise ModelRegistryValidationError("model registry aliases must be an object")
        index["aliases"] = {alias: aliases.get(alias) for alias in DEFAULT_ALIASES}
        index.setdefault("promoted_version", index["aliases"].get("promoted"))
        index.setdefault("original_version", index["aliases"].get("original"))
        if not isinstance(index.get("versions"), dict):
            raise ModelRegistryValidationError("model registry versions must be an object")
        if not isinstance(index.get("promotion_history"), list):
            raise ModelRegistryValidationError("model registry promotion_history must be a list")
        return index

    def version_dir(self, version_id: str) -> Path:
        version = safe_relative_path(version_id)
        if len(version.parts) != 1:
            raise ModelRegistryValidationError(f"invalid version id: {version_id}")
        return self.versions_dir / version

    def git_short(self) -> str | None:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short=8", "HEAD"],
                cwd=self.repo_root,
                check=True,
                text=True,
                capture_output=True,
            )
        except (OSError, subprocess.CalledProcessError, TypeError):
            return None
        value = result.stdout.strip()
        return value or None

    def build_version_id(self, *, run_id: str | None = None, created_at: datetime | None = None) -> str:
        return build_version_id(created_at=created_at, git_commit=self.git_short(), run_id=run_id)

    def artifact_record(self, version_dir: Path, artifact_path: Path) -> dict[str, Any]:
        try:
            rel = artifact_path.relative_to(version_dir)
        except ValueError as exc:
            raise ModelRegistryValidationError(f"artifact must live under version dir: {artifact_path}") from exc
        rel = safe_relative_path(rel)
        info = file_info(version_dir / rel)
        return {"path": rel.as_posix(), **info}

    def copy_artifacts(self, version_id: str, artifact_sources: Mapping[str, Path]) -> list[Path]:
        """Copy source files into a version package and return copied paths."""

        version_dir = self.version_dir(version_id)
        copied: list[Path] = []
        for rel_path, source_path in artifact_sources.items():
            rel = safe_relative_path(rel_path)
            source = Path(source_path)
            if not source.is_file():
                raise ModelRegistryValidationError(f"missing artifact: {source}")
            dest = version_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            if source.resolve() != dest.resolve():
                shutil.copy2(source, dest)
            copied.append(dest)
        return copied

    def collect_artifact_records(self, version_id: str, artifact_paths: Iterable[Path]) -> list[dict[str, Any]]:
        version_dir = self.version_dir(version_id)
        records = [self.artifact_record(version_dir, Path(path)) for path in artifact_paths]
        return sorted(records, key=lambda item: item["path"])

    def write_checksums(self, version_id: str, artifact_records: Iterable[Mapping[str, Any]]) -> Path:
        version_dir = self.version_dir(version_id)
        lines = []
        for record in sorted(artifact_records, key=lambda item: str(item["path"])):
            rel = safe_relative_path(str(record["path"]))
            digest = str(record["sha256"])
            lines.append(f"{digest}  {rel.as_posix()}")
        checksums_path = version_dir / "checksums.sha256"
        write_text_atomic(checksums_path, "\n".join(lines) + ("\n" if lines else ""))
        return checksums_path

    def write_model_card(self, manifest: Mapping[str, Any]) -> Path:
        version_id = str(manifest["version_id"])
        version_dir = self.version_dir(version_id)
        artifacts = manifest.get("artifacts", [])
        lines = [
            f"# Model Card — {version_id}",
            "",
            f"- Model ID: `{manifest.get('model_id', self.model_id)}`",
            f"- Status: `{manifest.get('status')}`",
            f"- Created: `{manifest.get('created_at')}`",
            f"- Created by: `{manifest.get('created_by')}`",
            "",
            "## Provenance",
            "",
            f"- Source: `{json.dumps(manifest.get('source', {}), sort_keys=True, default=_json_default)}`",
            f"- Data: `{json.dumps(manifest.get('data', {}), sort_keys=True, default=_json_default)}`",
            f"- Training: `{json.dumps(manifest.get('training', {}), sort_keys=True, default=_json_default)}`",
            "",
            "## Artifacts",
            "",
            "| Path | SHA-256 | Size |",
            "| --- | --- | ---: |",
        ]
        if isinstance(artifacts, list):
            for artifact in artifacts:
                if isinstance(artifact, Mapping):
                    lines.append(
                        f"| `{artifact.get('path')}` | `{artifact.get('sha256')}` | {artifact.get('size', 0)} |"
                    )
        card_path = version_dir / "model-card.md"
        write_text_atomic(card_path, "\n".join(lines) + "\n")
        return card_path

    def write_manifest(
        self,
        version_id: str,
        *,
        status: str = "candidate",
        artifact_paths: Iterable[Path] = (),
        metadata: Mapping[str, Any] | None = None,
        register: bool = True,
    ) -> dict[str, Any]:
        version_dir = self.version_dir(version_id)
        version_dir.mkdir(parents=True, exist_ok=True)
        records = self.collect_artifact_records(version_id, artifact_paths)
        self.write_checksums(version_id, records)
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "model_id": self.model_id,
            "version_id": version_id,
            "created_at": utc_now_iso(),
            "created_by": getpass.getuser(),
            "status": status,
            "source": {},
            "data": {},
            "training": {},
            "base_models": {},
            "metrics": {},
            "artifacts": records,
            "promotion": {},
        }
        manifest = _merge_dict(manifest, metadata)
        manifest["artifacts"] = records
        manifest["schema_version"] = SCHEMA_VERSION
        manifest["model_id"] = self.model_id
        manifest["version_id"] = version_id
        manifest["status"] = status

        manifest_path = version_dir / "manifest.json"
        atomic_write_json(manifest_path, manifest)
        self.write_model_card(manifest)
        if register:
            self.register_version(manifest)
        return manifest

    def register_version(self, manifest: Mapping[str, Any]) -> dict[str, Any]:
        version_id = str(manifest["version_id"])
        status = str(manifest.get("status") or "candidate")
        safe_relative_path(version_id)
        version_dir = self.version_dir(version_id)
        index = self.read_index()
        index["versions"][version_id] = {
            "version_id": version_id,
            "status": status,
            "created_at": manifest.get("created_at"),
            "manifest_path": (Path("versions") / version_id / "manifest.json").as_posix(),
            "model_card_path": (Path("versions") / version_id / "model-card.md").as_posix(),
            "checksums_path": (Path("versions") / version_id / "checksums.sha256").as_posix(),
            "manifest_sha256": sha256_file(version_dir / "manifest.json"),
            "checksums_sha256": sha256_file(version_dir / "checksums.sha256"),
            "runtime_path": (Path("versions") / version_id / "runtime").as_posix(),
            "artifact_count": len(manifest.get("artifacts", [])) if isinstance(manifest.get("artifacts"), list) else 0,
        }
        if status == "candidate":
            index["aliases"]["candidate"] = version_id
        elif status == "original":
            index["aliases"]["original"] = version_id
            index["aliases"]["promoted"] = index["aliases"].get("promoted") or version_id
            index["original_version"] = version_id
            index["promoted_version"] = index.get("promoted_version") or version_id
        elif status == "promoted":
            previous = index.get("promoted_version")
            if previous and previous != version_id:
                index["aliases"]["previous"] = previous
            index["aliases"]["promoted"] = version_id
            index["promoted_version"] = version_id
        if not version_dir.is_dir():
            raise ModelRegistryValidationError(f"missing version directory: {version_dir}")
        return self.write_index(index)

    def create_version_from_artifacts(
        self,
        version_id: str,
        *,
        artifact_sources: Mapping[str, Path],
        status: str = "candidate",
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        copied = self.copy_artifacts(version_id, artifact_sources)
        return self.write_manifest(version_id, status=status, artifact_paths=copied, metadata=metadata)

    def import_current_runtime(self, *, version_id: str | None = None, force: bool = False) -> dict[str, Any]:
        """Copy current runtime files into an immutable ``original`` package.

        Runtime files in ``backend/codex_model`` are copied, never moved or
        modified.  The first imported original becomes the promoted pointer until
        an explicit promotion changes it.
        """

        index = self.read_index()
        if index["aliases"].get("original") and not force:
            existing_id = str(index["aliases"]["original"])
            manifest = read_json_object(self.version_dir(existing_id) / "manifest.json")
            if manifest is None:
                raise ModelRegistryValidationError(f"original manifest missing for {existing_id}")
            return manifest

        resolved_id = version_id or self.build_version_id(run_id="original")
        sources: dict[str, Path] = {}
        missing: list[str] = []
        for rel_dest, runtime_parts in RUNTIME_ARTIFACTS:
            source = self.runtime_model_dir.joinpath(*runtime_parts)
            if source.is_file():
                sources[rel_dest] = source
            else:
                missing.append(str(source))
        if missing:
            raise ModelRegistryValidationError(
                "cannot import original runtime; missing required artifact(s): " + ", ".join(missing)
            )

        metadata = {
            "source": {
                "kind": "runtime_import",
                "runtime_model_dir": str(self.runtime_model_dir),
                "git_commit": self.git_short(),
            },
            "training": {"command": "import_current_runtime"},
            "promotion": {
                "imported_as_original": True,
                "runtime_model_dir": str(self.runtime_model_dir),
            },
        }
        return self.create_version_from_artifacts(
            resolved_id,
            artifact_sources=sources,
            status="original",
            metadata=metadata,
        )

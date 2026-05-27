from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.services.model_registry import ModelRegistry, atomic_write_json, sha256_file  # noqa: E402


def _write_runtime(runtime_dir: Path, label: str) -> None:
    (runtime_dir / "weights").mkdir(parents=True, exist_ok=True)
    (runtime_dir / "weights" / "prototypes.pt").write_bytes(f"{label}-prototypes".encode("utf-8"))
    (runtime_dir / "weights" / "projection.pt").write_bytes(f"{label}-projection".encode("utf-8"))
    (runtime_dir / "config.json").write_text(
        json.dumps({"model_version": label, "class_names": [label]}),
        encoding="utf-8",
    )


def _create_candidate(registry: ModelRegistry, tmp_path: Path, version_id: str, label: str = "candidate") -> None:
    source = tmp_path / "candidate-source" / version_id
    _write_runtime(source, label)
    registry.create_version_from_artifacts(
        version_id,
        artifact_sources={
            "runtime/weights/prototypes.pt": source / "weights" / "prototypes.pt",
            "runtime/weights/projection.pt": source / "weights" / "projection.pt",
            "runtime/config.json": source / "config.json",
        },
        metadata={"training": {"command": ["test"]}},
    )


def _run_promote(
    tmp_path: Path,
    *args: str,
    env: dict[str, str] | None = None,
    include_runtime_dir: bool = True,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        "scripts/promote_model.py",
        *args,
        "--registry-dir",
        str(tmp_path / "backend" / "model_registry"),
        "--json",
    ]
    if include_runtime_dir:
        command.extend(["--runtime-dir", str(tmp_path / "backend" / "codex_model")])
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        env=env,
    )


def test_promote_model_dry_run_validates_without_mutating_runtime(tmp_path):
    runtime = tmp_path / "backend" / "codex_model"
    _write_runtime(runtime, "original")
    registry = ModelRegistry(tmp_path / "backend" / "model_registry", repo_root=tmp_path, runtime_model_dir=runtime)
    _create_candidate(registry, tmp_path, "20260527T010203Z-test-candidate")
    before = (runtime / "config.json").read_text(encoding="utf-8")

    result = _run_promote(tmp_path, "20260527T010203Z-test-candidate", "--dry-run")

    assert result.returncode == 0, result.stderr
    body = json.loads(result.stdout)
    assert body["dry_run"] is True
    assert body["would_import_original"] is True
    assert body["would_snapshot_runtime"] is True
    assert body["copy_plan"][0]["dest"].startswith(str(runtime))
    assert (runtime / "config.json").read_text(encoding="utf-8") == before
    index = json.loads(registry.index_path.read_text(encoding="utf-8"))
    assert index["aliases"]["promoted"] is None


def test_promote_model_promotes_and_rolls_back_with_snapshots(tmp_path):
    runtime = tmp_path / "backend" / "codex_model"
    _write_runtime(runtime, "original")
    registry = ModelRegistry(tmp_path / "backend" / "model_registry", repo_root=tmp_path, runtime_model_dir=runtime)
    _create_candidate(registry, tmp_path, "20260527T010203Z-test-candidate")

    promoted = _run_promote(tmp_path, "20260527T010203Z-test-candidate")

    assert promoted.returncode == 0, promoted.stderr
    promoted_body = json.loads(promoted.stdout)
    assert promoted_body["action"] == "promote"
    assert Path(promoted_body["snapshot_path"]).is_dir()
    assert json.loads((runtime / "config.json").read_text(encoding="utf-8"))["model_version"] == "candidate"
    index_after_promote = json.loads(registry.index_path.read_text(encoding="utf-8"))
    original_version = index_after_promote["aliases"]["original"]
    assert index_after_promote["aliases"]["promoted"] == "20260527T010203Z-test-candidate"
    assert index_after_promote["aliases"]["previous"] == original_version

    rolled_back = _run_promote(tmp_path, "--rollback")

    assert rolled_back.returncode == 0, rolled_back.stderr
    rollback_body = json.loads(rolled_back.stdout)
    assert rollback_body["action"] == "rollback"
    assert json.loads((runtime / "config.json").read_text(encoding="utf-8"))["model_version"] == "original"
    index_after_rollback = json.loads(registry.index_path.read_text(encoding="utf-8"))
    assert index_after_rollback["aliases"]["promoted"] == original_version
    assert index_after_rollback["aliases"]["previous"] == "20260527T010203Z-test-candidate"
    assert len(index_after_rollback["promotion_history"]) >= 2
    assert not (settings_marker := (tmp_path / "backend" / "model_registry" / "promotion_in_progress.json")).exists(), settings_marker


def test_promote_model_rejects_partial_runtime_before_promotion(tmp_path):
    runtime = tmp_path / "backend" / "codex_model"
    (runtime / "weights").mkdir(parents=True, exist_ok=True)
    (runtime / "config.json").write_text('{"model_version":"partial"}', encoding="utf-8")
    registry = ModelRegistry(tmp_path / "backend" / "model_registry", repo_root=tmp_path, runtime_model_dir=runtime)
    _create_candidate(registry, tmp_path, "20260527T010203Z-test-candidate")

    result = _run_promote(tmp_path, "20260527T010203Z-test-candidate")

    assert result.returncode == 2
    assert "missing required artifact" in json.loads(result.stdout)["error"]
    assert json.loads((runtime / "config.json").read_text(encoding="utf-8"))["model_version"] == "partial"


def test_promote_model_rejects_checksum_file_drift(tmp_path):
    runtime = tmp_path / "backend" / "codex_model"
    _write_runtime(runtime, "original")
    registry = ModelRegistry(tmp_path / "backend" / "model_registry", repo_root=tmp_path, runtime_model_dir=runtime)
    version_id = "20260527T010203Z-test-candidate"
    _create_candidate(registry, tmp_path, version_id)
    checksums_path = registry.version_dir(version_id) / "checksums.sha256"
    checksums_path.write_text("0" * 64 + "  runtime/config.json\n", encoding="utf-8")

    result = _run_promote(tmp_path, version_id, "--dry-run")

    assert result.returncode == 2
    assert "checksums digest mismatch" in json.loads(result.stdout)["error"]


def test_promote_model_rejects_incompatible_manifest_schema_or_model(tmp_path):
    runtime = tmp_path / "backend" / "codex_model"
    _write_runtime(runtime, "original")
    registry = ModelRegistry(tmp_path / "backend" / "model_registry", repo_root=tmp_path, runtime_model_dir=runtime)
    version_id = "20260527T010203Z-test-candidate"
    _create_candidate(registry, tmp_path, version_id)
    manifest_path = registry.version_dir(version_id) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = 999
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    index = registry.read_index()
    index["versions"][version_id]["manifest_sha256"] = sha256_file(manifest_path)
    registry.write_index(index)

    schema_result = _run_promote(tmp_path, version_id, "--dry-run")

    assert schema_result.returncode == 2
    assert "unsupported manifest schema_version" in json.loads(schema_result.stdout)["error"]

    manifest["schema_version"] = 1
    manifest["model_id"] = "not_codex_classifier"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    index = registry.read_index()
    index["versions"][version_id]["manifest_sha256"] = sha256_file(manifest_path)
    registry.write_index(index)

    model_result = _run_promote(tmp_path, version_id, "--dry-run")

    assert model_result.returncode == 2
    assert "manifest model_id mismatch" in json.loads(model_result.stdout)["error"]


def test_promote_model_rejects_ambient_model_dir_override(tmp_path):
    runtime = tmp_path / "backend" / "codex_model"
    _write_runtime(runtime, "original")
    registry = ModelRegistry(tmp_path / "backend" / "model_registry", repo_root=tmp_path, runtime_model_dir=runtime)
    version_id = "20260527T010203Z-test-candidate"
    _create_candidate(registry, tmp_path, version_id)

    result = _run_promote(
        tmp_path,
        version_id,
        "--dry-run",
        env={**os.environ, "MODEL_DIR": str(tmp_path / "other")},
        include_runtime_dir=False,
    )

    assert result.returncode == 2
    assert "MODEL_DIR is set" in json.loads(result.stdout)["error"]


def test_promote_model_rejects_manifest_path_traversal(tmp_path):
    runtime = tmp_path / "backend" / "codex_model"
    _write_runtime(runtime, "original")
    registry = ModelRegistry(tmp_path / "backend" / "model_registry", repo_root=tmp_path, runtime_model_dir=runtime)
    version_id = "20260527T010203Z-test-badpath"
    version_dir = registry.version_dir(version_id)
    version_dir.mkdir(parents=True)
    atomic_write_json(
        version_dir / "manifest.json",
        {
            "schema_version": 1,
            "model_id": "codex_classifier",
            "version_id": version_id,
            "status": "candidate",
            "artifacts": [{"path": "../escape.pt", "sha256": "bad"}],
        },
    )
    (version_dir / "checksums.sha256").write_text("bad  ../escape.pt\n", encoding="utf-8")
    index = registry.empty_index()
    index["versions"][version_id] = {
        "version_id": version_id,
        "status": "candidate",
        "manifest_path": f"versions/{version_id}/manifest.json",
        "manifest_sha256": sha256_file(version_dir / "manifest.json"),
        "checksums_sha256": sha256_file(version_dir / "checksums.sha256"),
    }
    registry.write_index(index)

    result = _run_promote(tmp_path, version_id, "--dry-run")

    assert result.returncode == 2
    assert "unsafe registry artifact path" in json.loads(result.stdout)["error"]

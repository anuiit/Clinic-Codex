# type: ignore
# pyright: reportMissingImports=false
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from backend.services.model_registry import (
    ModelRegistry,
    ModelRegistryValidationError,
    build_version_id,
    sha256_file,
)


def test_build_version_id_is_sortable_and_sanitized():
    version_id = build_version_id(
        created_at=datetime(2026, 5, 27, 1, 2, 3, tzinfo=timezone.utc),
        git_commit="abc1234/dirty",
        run_id="run id with spaces",
    )

    assert version_id == "20260527T010203Z-abc1234-dirt-run-id-w"


def test_registry_package_writes_manifest_checksums_model_card_and_index(tmp_path):
    registry = ModelRegistry(tmp_path / "backend" / "model_registry", repo_root=tmp_path)
    source_dir = tmp_path / "source"
    (source_dir / "weights").mkdir(parents=True)
    (source_dir / "weights" / "prototypes.pt").write_bytes(b"prototype-bytes")
    (source_dir / "weights" / "projection.pt").write_bytes(b"projection-bytes")
    (source_dir / "config.json").write_text('{"model_version":"candidate"}', encoding="utf-8")

    manifest = registry.create_version_from_artifacts(
        "20260527T010203Z-nogit-deadbeef",
        artifact_sources={
            "runtime/weights/prototypes.pt": source_dir / "weights" / "prototypes.pt",
            "runtime/weights/projection.pt": source_dir / "weights" / "projection.pt",
            "runtime/config.json": source_dir / "config.json",
        },
        metadata={
            "source": {"git_commit": "abc12345", "dirty": False},
            "data": {"metadata_csv_sha256": "metadata-hash"},
            "training": {"command": ["bash", "scripts/retrain.sh", "--dry-run"]},
            "metrics": {"prototype_export": "pending"},
        },
    )

    version_dir = registry.version_dir("20260527T010203Z-nogit-deadbeef")
    manifest_path = version_dir / "manifest.json"
    checksums_path = version_dir / "checksums.sha256"
    card_path = version_dir / "model-card.md"
    index = json.loads(registry.index_path.read_text(encoding="utf-8"))

    assert manifest["status"] == "candidate"
    assert manifest["data"]["metadata_csv_sha256"] == "metadata-hash"
    assert sorted(item["path"] for item in manifest["artifacts"]) == [
        "runtime/config.json",
        "runtime/weights/projection.pt",
        "runtime/weights/prototypes.pt",
    ]
    assert manifest_path.is_file()
    assert checksums_path.is_file()
    assert card_path.is_file()
    assert "runtime/weights/prototypes.pt" in checksums_path.read_text(encoding="utf-8")
    assert "Model Card" in card_path.read_text(encoding="utf-8")
    assert index["aliases"]["candidate"] == "20260527T010203Z-nogit-deadbeef"
    assert index["versions"]["20260527T010203Z-nogit-deadbeef"]["artifact_count"] == 3
    assert not list(registry.root.glob(".*.tmp-*"))


def test_import_current_runtime_copies_original_without_mutating_runtime(tmp_path):
    backend_root = tmp_path / "backend"
    runtime = backend_root / "codex_model"
    (runtime / "weights").mkdir(parents=True)
    config_path = runtime / "config.json"
    prototypes_path = runtime / "weights" / "prototypes.pt"
    projection_path = runtime / "weights" / "projection.pt"
    config_path.write_text('{"model_version":"original"}', encoding="utf-8")
    prototypes_path.write_bytes(b"runtime-prototypes")
    projection_path.write_bytes(b"runtime-projection")
    before = {
        path: (path.read_bytes(), sha256_file(path))
        for path in [config_path, prototypes_path, projection_path]
    }
    registry = ModelRegistry(backend_root / "model_registry", repo_root=tmp_path, runtime_model_dir=runtime)

    manifest = registry.import_current_runtime(version_id="20260527T010203Z-nogit-original")
    second = registry.import_current_runtime(version_id="ignored-unless-force")
    index = json.loads(registry.index_path.read_text(encoding="utf-8"))

    assert manifest["status"] == "original"
    assert second["version_id"] == manifest["version_id"]
    assert index["aliases"]["original"] == "20260527T010203Z-nogit-original"
    assert index["aliases"]["promoted"] == "20260527T010203Z-nogit-original"
    assert index["original_version"] == "20260527T010203Z-nogit-original"
    assert index["promoted_version"] == "20260527T010203Z-nogit-original"
    copied_config = registry.version_dir("20260527T010203Z-nogit-original") / "runtime" / "config.json"
    assert copied_config.read_bytes() == before[config_path][0]
    for path, (content, digest) in before.items():
        assert path.read_bytes() == content
        assert sha256_file(path) == digest


def test_registry_rejects_path_traversal_artifact_entries(tmp_path):
    registry = ModelRegistry(tmp_path / "backend" / "model_registry", repo_root=tmp_path)
    source = tmp_path / "source.pt"
    source.write_bytes(b"payload")

    with pytest.raises(ModelRegistryValidationError):
        registry.create_version_from_artifacts(
            "20260527T010203Z-nogit-deadbeef",
            artifact_sources={"../escape.pt": source},
        )

    with pytest.raises(ModelRegistryValidationError):
        registry.version_dir("../bad-version")


def test_registry_rejects_unsupported_index_schema(tmp_path):
    registry = ModelRegistry(tmp_path / "backend" / "model_registry", repo_root=tmp_path)
    registry.index_path.parent.mkdir(parents=True)
    registry.index_path.write_text(
        json.dumps(
            {
                "schema_version": 999,
                "model_id": "codex_classifier",
                "aliases": {},
                "versions": {},
                "promotion_history": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ModelRegistryValidationError, match="unsupported model registry schema_version"):
        registry.read_index()

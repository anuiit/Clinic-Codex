from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch

from backend.codex_pipeline.models.projection_head import ProjectionHead


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "export_elements_refit.py"
SPEC = importlib.util.spec_from_file_location("export_elements_refit", SCRIPT)
assert SPEC is not None
assert SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def _sparse_runtime_labels() -> torch.Tensor:
    values = list(range(14)) + list(range(15, 287))
    labels = torch.tensor(values[:286], dtype=torch.long)
    assert len(labels) == 286
    assert not torch.equal(labels, torch.arange(286))
    return labels


def _write_runtime_config(path: Path) -> Path:
    class_names = [f"class_{index}" for index in range(286)]
    payload = {
        "model_version": "1.0.0",
        "backbone": "dinov2_vits14",
        "embedding_dim": 128,
        "hidden_dim": 384,
        "image_size": 224,
        "rejection_threshold": 0.35,
        "num_classes": 286,
        "class_names": class_names,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _write_runtime_prototypes(path: Path) -> Path:
    class_labels = _sparse_runtime_labels()
    class_names = {int(label): f"class_{index}" for index, label in enumerate(class_labels.tolist())}
    prototypes = torch.zeros(286, 128, dtype=torch.float32)
    torch.save(
        {
            "prototypes": prototypes,
            "class_names": class_names,
            "class_labels": class_labels,
            "embedding_dim": 128,
        },
        path,
    )
    return path


def _write_feature_cache(path: Path) -> Path:
    features = torch.eye(286, 384, dtype=torch.float32)
    labels = torch.arange(286, dtype=torch.long)
    class_names = {index: f"class_{index}" for index in range(286)}
    torch.save(
        {
            "features": features,
            "labels": labels,
            "class_names": class_names,
            "backbone": "dinov2_vits14",
            "hidden_dim": 384,
            "image_size": 224,
        },
        path,
    )
    return path


def _write_checkpoint(path: Path) -> Path:
    state_dict = ProjectionHead(384, 128).state_dict()
    torch.save(
        {
            "model_state_dict": state_dict,
            "spec": {
                "objective": "proxy_distillation",
                "initialization": "random",
                "seed": 7,
                "teacher_weight": 1.0,
                "hidden_teacher_weight": 0.5,
                "teacher_temperature": 0.2,
            },
            "train_top1": 0.9,
            "train_macro_top1": 0.8,
            "train_top3": 0.95,
        },
        path,
    )
    return path


def test_export_elements_refit_writes_registry_package_and_preserves_abi(tmp_path: Path, monkeypatch) -> None:
    registry_root = tmp_path / "model_registry"
    runtime_config = _write_runtime_config(tmp_path / "runtime.json")
    runtime_prototypes = _write_runtime_prototypes(tmp_path / "prototypes.pt")
    feature_cache = _write_feature_cache(tmp_path / "cache.pt")
    checkpoint = _write_checkpoint(tmp_path / "checkpoint.pt")

    monkeypatch.setattr(
        module,
        "compute_cache_metrics",
        lambda *args, **kwargs: {
            "prototype_top1": 0.5,
            "prototype_macro_top1": 0.6,
            "prototype_top3": 0.7,
        },
    )

    result = module.export_elements_refit(
        checkpoint,
        feature_cache,
        registry_root,
        runtime_config,
        runtime_prototypes,
        "20260728T120000Z-elements-refit-test",
        "cpu",
    )

    version_dir = registry_root / "versions" / "20260728T120000Z-elements-refit-test"
    prototypes = torch.load(version_dir / "runtime" / "weights" / "prototypes.pt", map_location="cpu", weights_only=False)
    projection = torch.load(version_dir / "runtime" / "weights" / "projection.pt", map_location="cpu", weights_only=False)
    runtime = json.loads((version_dir / "runtime" / "config.json").read_text(encoding="utf-8"))
    provenance = json.loads((version_dir / "provenance.json").read_text(encoding="utf-8"))
    manifest = json.loads((version_dir / "manifest.json").read_text(encoding="utf-8"))
    index = json.loads((registry_root / "index.json").read_text(encoding="utf-8"))

    expected_sparse_labels = _sparse_runtime_labels()
    assert result["runtime_compatible"] is True
    assert result["class_count"] == 286
    assert result["prototype_count"] == 286
    assert runtime["num_classes"] == 286
    assert runtime["class_names"] == [f"class_{index}" for index in range(286)]
    assert tuple(prototypes["prototypes"].shape) == (286, 128)
    assert prototypes["embedding_dim"] == 128
    assert torch.equal(prototypes["class_labels"], expected_sparse_labels)
    expected_class_names = {int(label): f"class_{index}" for index, label in enumerate(expected_sparse_labels.tolist())}
    assert prototypes["class_names"] == expected_class_names
    assert set(projection) == set(ProjectionHead(384, 128).state_dict())
    assert provenance["feature_cache"]["feature_count"] == 286
    assert provenance["metrics"]["prototype_top1"] == 0.5
    assert provenance["training"]["teacher_assisted"] is True
    assert provenance["training"]["teacher_weight"] == 1.0
    assert provenance["training"]["hidden_teacher_weight"] == 0.5
    assert provenance["training"]["teacher_temperature"] == 0.2
    assert provenance["training"]["teacher_projection_path"] == str((module.RUNTIME_MODEL_DIR / "weights" / "projection.pt").resolve())
    assert provenance["training"]["teacher_projection_sha256"] == module.sha256_file(module.RUNTIME_MODEL_DIR / "weights" / "projection.pt")
    assert provenance["runtime"]["compatible"] is True
    assert provenance["runtime"]["prototype_label_order_sha256"]
    assert provenance["training"]["teacher_temperature"] == 0.2
    assert manifest["model_id"] == "codex_classifier"
    assert manifest["training"]["teacher_assisted"] is True
    assert manifest["training"]["teacher_weight"] == 1.0
    assert manifest["training"]["hidden_teacher_weight"] == 0.5
    assert manifest["training"]["teacher_temperature"] == 0.2
    assert manifest["training"]["teacher_projection_path"] == str((module.RUNTIME_MODEL_DIR / "weights" / "projection.pt").resolve())
    assert manifest["training"]["teacher_projection_sha256"] == module.sha256_file(module.RUNTIME_MODEL_DIR / "weights" / "projection.pt")
    assert any(item["path"] == "runtime/weights/prototypes.pt" for item in manifest["artifacts"])
    assert any(item["path"] == "runtime/weights/projection.pt" for item in manifest["artifacts"])
    assert any(item["path"] == "runtime/config.json" for item in manifest["artifacts"])
    assert any(item["path"] == "provenance.json" for item in manifest["artifacts"])
    assert "20260728T120000Z-elements-refit-test" in index["versions"]


def test_main_exports_with_explicit_arguments(tmp_path: Path, monkeypatch) -> None:
    runtime_config = _write_runtime_config(tmp_path / "runtime.json")
    runtime_prototypes = _write_runtime_prototypes(tmp_path / "prototypes.pt")
    feature_cache = _write_feature_cache(tmp_path / "cache.pt")
    checkpoint = _write_checkpoint(tmp_path / "checkpoint.pt")
    registry_root = tmp_path / "registry"

    captured: dict[str, object] = {}

    def fake_export(*args):
        captured["args"] = args
        return {"ok": True}

    monkeypatch.setattr(module, "export_elements_refit", fake_export)

    exit_code = module.main(
        [
            "--checkpoint",
            str(checkpoint),
            "--feature-cache",
            str(feature_cache),
            "--registry-root",
            str(registry_root),
            "--runtime-config",
            str(runtime_config),
            "--runtime-prototypes",
            str(runtime_prototypes),
            "--version-id",
            "version-123",
            "--device",
            "cpu",
        ]
    )

    assert exit_code == 0
    assert captured["args"] == (
        checkpoint,
        feature_cache,
        registry_root,
        runtime_config,
        runtime_prototypes,
        "version-123",
        "cpu",
    )


def test_export_elements_refit_refuses_backend_codex_model(tmp_path: Path) -> None:
    runtime_config = _write_runtime_config(tmp_path / "runtime.json")
    runtime_prototypes = _write_runtime_prototypes(tmp_path / "prototypes.pt")
    feature_cache = _write_feature_cache(tmp_path / "cache.pt")
    checkpoint = _write_checkpoint(tmp_path / "checkpoint.pt")

    with pytest.raises(ValueError, match="backend/codex_model"):
        module.export_elements_refit(
            checkpoint,
            feature_cache,
            ROOT / "backend" / "codex_model",
            runtime_config,
            runtime_prototypes,
            "version-123",
            "cpu",
        )

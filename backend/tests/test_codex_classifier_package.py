# type: ignore
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from backend.codex_model.classifier import (
    CodexClassifier,
    ModelPackageValidationError,
    _ProjectionHead,
)
from backend.services.model_registry import ModelRegistry


class _FakeBackbone(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.ones((inputs.shape[0], self.hidden_dim), dtype=torch.float32)


def _write_package(
    tmp_path: Path,
    *,
    version_id: str,
    backbone: str,
    hidden_dim: int,
    class_names: list[str] | None = None,
) -> tuple[ModelRegistry, Path]:
    names = class_names or ["atl", "calli"]
    source = tmp_path / f"source-{version_id}"
    weights = source / "weights"
    weights.mkdir(parents=True)
    prototypes = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    torch.save(
        {
            "prototypes": prototypes,
            "class_names": {4: "atl", 9: "calli"},
            "class_labels": torch.tensor([4, 9], dtype=torch.int64),
            "embedding_dim": 2,
        },
        weights / "prototypes.pt",
    )
    torch.save(
        _ProjectionHead(hidden_dim=hidden_dim, embedding_dim=2).state_dict(),
        weights / "projection.pt",
    )
    (source / "config.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_version": version_id,
                "backbone": backbone,
                "hidden_dim": hidden_dim,
                "embedding_dim": 2,
                "image_size": 224,
                "rejection_threshold": 0.35,
                "num_classes": 2,
                "class_names": names,
            }
        ),
        encoding="utf-8",
    )
    registry = ModelRegistry(tmp_path / "backend" / "model_registry", repo_root=tmp_path)
    registry.create_version_from_artifacts(
        version_id,
        artifact_sources={
            "runtime/config.json": source / "config.json",
            "runtime/weights/prototypes.pt": weights / "prototypes.pt",
            "runtime/weights/projection.pt": weights / "projection.pt",
        },
    )
    return registry, registry.resolve_runtime_package(version_id)


@pytest.mark.parametrize(
    ("backbone", "hidden_dim"),
    [("dinov2_vits14", 384), ("dinov2_vitb14", 768)],
)
def test_classifier_loads_validated_runtime_packages(
    tmp_path, monkeypatch, backbone, hidden_dim
):
    registry, runtime_dir = _write_package(
        tmp_path,
        version_id=f"valid-{hidden_dim}",
        backbone=backbone,
        hidden_dim=hidden_dim,
    )
    loaded: list[str] = []

    def fake_load(_repository, requested_backbone, *, pretrained):
        assert pretrained is True
        loaded.append(requested_backbone)
        return _FakeBackbone(hidden_dim)

    monkeypatch.setattr(torch.hub, "load", fake_load)

    classifier = CodexClassifier(
        model_dir=registry.resolve_runtime_package("candidate"),
        device="cpu",
    )
    result = classifier.classify(np.zeros((12, 8, 3), dtype=np.uint8), top_k=2)

    assert runtime_dir.name == "runtime"
    assert classifier.num_classes == 2
    assert result["class_label"] in {4, 9}
    assert loaded == [backbone]


def test_classifier_rejects_taxonomy_before_loading_backbone(tmp_path, monkeypatch):
    _registry, runtime_dir = _write_package(
        tmp_path,
        version_id="invalid-taxonomy",
        backbone="dinov2_vits14",
        hidden_dim=384,
        class_names=["calli", "atl"],
    )
    monkeypatch.setattr(
        torch.hub,
        "load",
        lambda *_args, **_kwargs: pytest.fail("backbone loaded before taxonomy validation"),
    )

    with pytest.raises(ModelPackageValidationError, match="taxonomy"):
        CodexClassifier(model_dir=runtime_dir, device="cpu")


def test_classifier_preserves_legacy_weights_directory_fallback(tmp_path, monkeypatch):
    _, runtime_dir = _write_package(tmp_path, version_id="legacy", backbone="dinov2_vits14", hidden_dim=384)
    monkeypatch.setattr(sys.modules[CodexClassifier.__module__], "__file__", str(runtime_dir / "classifier.py"))
    monkeypatch.setattr(
        torch.hub,
        "load",
        lambda *_args, **_kwargs: _FakeBackbone(384),
    )

    classifier = CodexClassifier(device="cpu")

    assert classifier.num_classes == 2
    assert classifier.config["backbone"] == "dinov2_vits14"

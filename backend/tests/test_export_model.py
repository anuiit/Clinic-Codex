# type: ignore
# pyright: reportMissingImports=false
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from codex_model.classifier import _ProjectionHead  # noqa: E402
from codex_pipeline.scripts.export_model import RuntimeWriteRefusedError, export_model  # noqa: E402


def _write_prototype_fixture(prototype_src: Path) -> None:
    prototype_src.parent.mkdir(parents=True, exist_ok=True)
    projection = _ProjectionHead(hidden_dim=4, embedding_dim=2)
    torch.save(
        {
            "prototypes": torch.eye(2),
            "class_names": {0: "atl", 1: "calli"},
            "class_labels": torch.tensor([0, 1]),
            "embedding_dim": 2,
            "hidden_dim": 4,
            "model_state_dict": projection.state_dict(),
        },
        prototype_src,
    )


def _write_config(config_path: Path, model_version: str = "test") -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(
            {
                "model_version": model_version,
                "backbone": "dinov2_vits14",
                "image_size": 224,
                "rejection_threshold": 0.35,
            }
        ),
        encoding="utf-8",
    )


def test_export_model_writes_backend_loadable_classifier_artifacts(tmp_path):
    prototype_src = tmp_path / "prototypes" / "prototypes.pt"
    weights_dir = tmp_path / "codex_model" / "weights"
    config_path = tmp_path / "codex_model" / "config.json"
    _write_prototype_fixture(prototype_src)
    _write_config(config_path)

    outputs = export_model(prototype_src, weights_dir, config_path)

    assert outputs["prototypes"].is_file()
    assert outputs["projection"].is_file()
    assert outputs["config"].is_file()

    exported_prototypes = torch.load(weights_dir / "prototypes.pt", map_location="cpu", weights_only=False)
    assert exported_prototypes["class_names"] == {0: "atl", 1: "calli"}
    assert exported_prototypes["prototypes"].shape == (2, 2)

    smoke_projection = _ProjectionHead(hidden_dim=4, embedding_dim=2)
    smoke_projection.load_state_dict(torch.load(weights_dir / "projection.pt", map_location="cpu", weights_only=False))

    config = json.loads(config_path.read_text())
    assert config["num_classes"] == 2
    assert config["class_names"] == ["atl", "calli"]
    assert config["embedding_dim"] == 2
    assert config["hidden_dim"] == 4


def test_export_model_candidate_config_out_does_not_mutate_runtime_template(tmp_path):
    prototype_src = tmp_path / "prototypes" / "prototypes.pt"
    runtime_dir = tmp_path / "backend" / "codex_model"
    runtime_config = runtime_dir / "config.json"
    candidate_dir = tmp_path / "backend" / "model_registry" / "versions" / "v1" / "runtime"
    candidate_config = candidate_dir / "config.json"
    manifest_out = tmp_path / "backend" / "model_registry" / "versions" / "v1" / "export_model_manifest.json"
    _write_prototype_fixture(prototype_src)
    _write_config(runtime_config, model_version="runtime-original")
    before_runtime_config = runtime_config.read_text(encoding="utf-8")

    outputs = export_model(
        prototype_src,
        candidate_dir / "weights",
        runtime_config,
        config_out_path=candidate_config,
        runtime_model_dir=runtime_dir,
        manifest_out_path=manifest_out,
    )

    assert outputs["config"] == candidate_config
    assert outputs["manifest"] == manifest_out
    assert runtime_config.read_text(encoding="utf-8") == before_runtime_config
    candidate = json.loads(candidate_config.read_text(encoding="utf-8"))
    assert candidate["model_version"] == "runtime-original"
    assert candidate["num_classes"] == 2
    assert (candidate_dir / "weights" / "prototypes.pt").is_file()
    manifest = json.loads(manifest_out.read_text(encoding="utf-8"))
    assert manifest["allow_runtime_write"] is False
    assert manifest["config_template"] == str(runtime_config)


def test_export_model_can_register_candidate_manifest(tmp_path):
    prototype_src = tmp_path / "backend" / "model_registry" / "versions" / "v1" / "prototypes" / "prototypes.pt"
    runtime_dir = tmp_path / "backend" / "codex_model"
    runtime_config = runtime_dir / "config.json"
    version_dir = tmp_path / "backend" / "model_registry" / "versions" / "v1"
    _write_prototype_fixture(prototype_src)
    _write_config(runtime_config, model_version="runtime-original")

    outputs = export_model(
        prototype_src,
        version_dir / "runtime" / "weights",
        runtime_config,
        config_out_path=version_dir / "runtime" / "config.json",
        runtime_model_dir=runtime_dir,
        registry_dir=tmp_path / "backend" / "model_registry",
        version_id="v1",
    )

    assert outputs["registry_manifest"] == version_dir / "manifest.json"
    assert (version_dir / "checksums.sha256").is_file()
    assert (version_dir / "model-card.md").is_file()
    index = json.loads((tmp_path / "backend" / "model_registry" / "index.json").read_text(encoding="utf-8"))
    assert index["aliases"]["candidate"] == "v1"


def test_export_model_refuses_runtime_write_without_explicit_opt_in(tmp_path):
    prototype_src = tmp_path / "prototypes" / "prototypes.pt"
    runtime_dir = tmp_path / "backend" / "codex_model"
    runtime_config = runtime_dir / "config.json"
    _write_prototype_fixture(prototype_src)
    _write_config(runtime_config)

    with pytest.raises(RuntimeWriteRefusedError):
        export_model(
            prototype_src,
            runtime_dir / "weights",
            runtime_config,
            runtime_model_dir=runtime_dir,
        )

    outputs = export_model(
        prototype_src,
        runtime_dir / "weights",
        runtime_config,
        runtime_model_dir=runtime_dir,
        allow_runtime_write=True,
    )
    assert outputs["prototypes"].is_file()

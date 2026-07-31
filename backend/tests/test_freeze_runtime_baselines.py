from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/freeze_runtime_baselines.py"
SPEC = importlib.util.spec_from_file_location("freeze_runtime_baselines", SCRIPT)
assert SPEC and SPEC.loader
freeze = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(freeze)


def _projection(offset: float = 0.0) -> dict[str, torch.Tensor]:
    return {
        "net.0.weight": torch.full((384, 384), offset),
        "net.0.bias": torch.full((384,), offset),
        "net.3.weight": torch.full((128, 384), offset),
        "net.3.bias": torch.full((128,), offset),
    }


def _write_runtime(root: Path, *, offset: float = 0.0, reverse: bool = False) -> None:
    root.mkdir(parents=True)
    state = _projection(offset)
    if reverse:
        state = dict(reversed(list(state.items())))
    torch.save(state, root / "projection.pt")
    names = {index: f"class-{index}" for index in range(286)}
    torch.save(
        {
            "class_names": names,
            "class_labels": torch.arange(286, dtype=torch.long),
            "prototypes": torch.arange(286 * 128, dtype=torch.float32).reshape(
                286, 128
            ),
        },
        root / "prototypes.pt",
    )


def test_manifest_proves_tensor_equality_despite_serialization_difference(
    tmp_path: Path,
) -> None:
    current = tmp_path / "current"
    legacy = tmp_path / "legacy"
    _write_runtime(current)
    _write_runtime(legacy, reverse=True)
    current_classifier = tmp_path / "current.py"
    legacy_classifier = tmp_path / "legacy.py"
    current_classifier.write_text("padding = (255, 255, 255)\n", encoding="utf-8")
    legacy_classifier.write_text("padding = (128, 128, 128)\n", encoding="utf-8")

    manifest = freeze.build_manifest(
        current, legacy, current_classifier, legacy_classifier
    )

    assert manifest["runtime_weights_tensor_equal"] is True
    assert manifest["baselines"]["deployed_white"]["padding_rgb"] == [255, 255, 255]
    assert manifest["baselines"]["legacy_gray"]["padding_rgb"] == [128, 128, 128]
    assert (
        manifest["baselines"]["deployed_white"]["projection_file_sha256"]
        != manifest["baselines"]["legacy_gray"]["projection_file_sha256"]
    )
    assert len(manifest["manifest_sha256"]) == 64


def test_manifest_rejects_tensor_drift(tmp_path: Path) -> None:
    current = tmp_path / "current"
    legacy = tmp_path / "legacy"
    _write_runtime(current)
    _write_runtime(legacy, offset=1.0)
    current_classifier = tmp_path / "current.py"
    legacy_classifier = tmp_path / "legacy.py"
    current_classifier.write_text("(255, 255, 255)", encoding="utf-8")
    legacy_classifier.write_text("(128, 128, 128)", encoding="utf-8")

    with pytest.raises(ValueError, match="projection tensors differ"):
        freeze.build_manifest(
            current, legacy, current_classifier, legacy_classifier
        )


def test_manifest_is_stable(tmp_path: Path) -> None:
    current = tmp_path / "current"
    legacy = tmp_path / "legacy"
    _write_runtime(current)
    _write_runtime(legacy)
    current_classifier = tmp_path / "current.py"
    legacy_classifier = tmp_path / "legacy.py"
    current_classifier.write_text("(255, 255, 255)", encoding="utf-8")
    legacy_classifier.write_text("(128, 128, 128)", encoding="utf-8")

    first = freeze.build_manifest(
        current, legacy, current_classifier, legacy_classifier
    )
    second = freeze.build_manifest(
        current, legacy, current_classifier, legacy_classifier
    )
    assert first == second

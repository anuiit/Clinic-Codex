from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
import pytest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/autoresearch_vicreg_backbone_feasibility_v19.py"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "autoresearch_vicreg_backbone_feasibility_v19_tested", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[2] / ".omx/argos-council/elements-model-improvement/turns/043/synthesis.md").exists(),
    reason="requires unshipped research artifact: 043/synthesis.md",
)
def test_frozen_history_runtime_and_b14_pin_are_unchanged() -> None:
    module = load_module()

    verified = module.verify_file_hashes()
    pin = module.backbone_screen.validate_backbone_manifest(
        module.DEFAULT_BACKBONE_MANIFEST
    )

    assert set(verified) == set(module.EXPECTED_HASHES)
    assert pin["manifest_sha256"] == module.EXPECTED_HASHES["b14_manifest"][1]
    assert pin["weights_sha256_verified"] == module.backbone_screen.EXPECTED_WEIGHTS_SHA256
    assert pin["weights_bytes_verified"] == 346_378_731


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[2] / ".omc/autoresearch/elements-baseline-replacement/runs/20260804-discriminative-readout-v18/specs/iteration-0003-source-inventory.jsonl").exists(),
    reason="requires unshipped research artifact: specs/iteration-0003-source-inventory.jsonl",
)
def test_inventory_cache_estimate_matches_frozen_five_fold_geometry() -> None:
    module = load_module()

    estimate = module.inventory_cache_estimate(module.DEFAULT_SOURCE_INVENTORY)

    assert estimate["inventory_rows"] == 9990
    assert estimate["total_fold_local_train_rows"] == 39960
    assert estimate["estimated_cache_bytes"] == 552_407_040
    assert estimate["estimated_cache_bytes"] < 2 * 1024**3
    assert estimate["pass"] is True
    assert {
        fold: values["train_rows"]
        for fold, values in estimate["per_fold"].items()
    } == {
        "1": 8091,
        "2": 8139,
        "3": 8021,
        "4": 8041,
        "5": 7668,
    }


def test_projection_architecture_is_dimension_forced_v9_head() -> None:
    module = load_module()

    architecture = module.projection_architecture()

    assert architecture["architecture"] == "768->768->128"
    assert architecture["pass"] is True


def test_batch_selection_requires_fifteen_percent_free_margin() -> None:
    module = load_module()
    probes = [
        {
            "batch_size": 8,
            "status": "pass",
            "free_memory_fraction_during_forward": 0.50,
        },
        {
            "batch_size": 16,
            "status": "pass",
            "free_memory_fraction_during_forward": 0.15,
        },
        {
            "batch_size": 32,
            "status": "pass",
            "free_memory_fraction_during_forward": 0.149,
        },
    ]

    assert module.choose_batch(probes) == 16
    assert module.choose_batch([probes[-1]]) is None


def test_synthetic_mini_cache_is_sealed_and_byte_identical(
    tmp_path: Path, monkeypatch
) -> None:
    module = load_module()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    values = np.arange(24, dtype=np.float16).reshape(3, 8)
    path = tmp_path / "run/mini.npy"

    audit = module.seal_mini_cache(path, values)

    assert audit["byte_identical_readback"] is True
    assert audit["value_identical_readback"] is True
    assert audit["synthetic_only"] is True
    assert audit["oof_rows_accessed"] == 0
    assert path.is_file()
    assert not path.with_suffix(".npy.partial").exists()

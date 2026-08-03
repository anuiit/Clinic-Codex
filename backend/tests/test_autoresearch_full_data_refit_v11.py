from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/autoresearch_full_data_refit_v11.py"
SPEC = ROOT / ".omc/autoresearch/elements-baseline-replacement/runs/20260803-full-data-v11/specs/iteration-0001.json"
EVALUATOR = ROOT / ".omc/autoresearch/elements-baseline-replacement/runs/20260803-full-data-v11/evaluator.json"
spec = importlib.util.spec_from_file_location("autoresearch_full_data_refit_v11", SCRIPT)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def _synthetic_cache(path: Path) -> Path:
    generator = torch.Generator().manual_seed(91)
    rows = 32
    labels = torch.tensor([label for label in range(4) for _ in range(8)])
    payload = {
        "schema_version": "autoresearch-full-data-v11.cache",
        "features": torch.randn(rows, 384, generator=generator).half(),
        "view_features": torch.randn(rows, 2, 384, generator=generator).half(),
        "labels": labels,
        "class_names": {label: f"class-{label}" for label in range(4)},
        "views": 2,
        "view_seed": 3,
        "image_size": 224,
        "row_id": [f"row-{index}" for index in range(rows)],
        "component_id": [f"component-{index // 4}" for index in range(rows)],
        "decoded_pixel_sha256": [f"pixel-{index}" for index in range(rows)],
        "class_label": labels.tolist(),
        "class_name": [f"class-{int(label)}" for label in labels],
        "fold": [1] * rows,
        "provenance": {
            "corpus_validation": {},
            "labels_used_by_ssl": False,
            "heldout_rows_consumed": 0,
            "external_holdout_consumed": False,
            "final_test_read": False,
        },
    }
    torch.save(payload, path)
    module.v9.write_json(
        module.cache_sidecar_path(path),
        {
            "schema_version": "autoresearch-full-data-v11.cache.provenance",
            "cache_sha256": module.sha256_file(path),
        },
    )
    return path


def test_v11_contract_is_hash_pinned() -> None:
    contract = module.validate_contract(SPEC, EVALUATOR)
    assert contract["spec_sha256"] == module.EXPECTED_SPEC_SHA256
    assert contract["evaluator_sha256"] == module.EXPECTED_EVALUATOR_SHA256
    assert contract["evaluator"]["promotion_eligible"] is False


def test_full_data_seed_is_fresh_and_deterministic() -> None:
    assert module.FULL_DATA_SEED == module.v9.stable_seed("v11-full-data-refit")
    assert module.FULL_DATA_SEED not in {17, 42, 73}


def test_cache_hash_tampering_is_rejected(tmp_path: Path) -> None:
    path = _synthetic_cache(tmp_path / "cache.pt")
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="hash"):
        module.load_full_cache(path, strict=False)


def test_two_small_refits_are_byte_identical(tmp_path: Path) -> None:
    path = _synthetic_cache(tmp_path / "cache.pt")
    kwargs = {
        "cache_path": path,
        "seed": 101,
        "ssl_epochs": 1,
        "ssl_batch_size": 16,
        "supervised_epochs": 1,
        "episodes_per_epoch": 2,
        "n_way": 2,
        "k_shot": 1,
        "q_queries": 1,
        "device": torch.device("cpu"),
        "supervised_device": torch.device("cpu"),
        "strict_cache": False,
    }
    first = module.train_replica(output_dir=tmp_path / "first", **kwargs)
    second = module.train_replica(output_dir=tmp_path / "second", **kwargs)
    assert first["state_dict_sha256"] == second["state_dict_sha256"]
    assert first["prototypes_sha256"] == second["prototypes_sha256"]
    result = module.verify_replicas(
        tmp_path / "first/checkpoint.pt",
        tmp_path / "second/checkpoint.pt",
        output_path=tmp_path / "reproducibility.json",
    )
    assert result["pass"] is True
    assert all(result["gates"].values())


def test_runner_exposes_build_only_commands() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    for command in ("precompute", "refit", "verify", "export", "all"):
        assert command in completed.stdout

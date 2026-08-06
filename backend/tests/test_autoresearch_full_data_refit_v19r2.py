from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/autoresearch_full_data_refit_v19r2.py"
BUILD_SPEC = ROOT / "backend/model_registry/specs/r2-full-data-production-v1.json"
E2E_SPEC = ROOT / "backend/model_registry/specs/r2-e2e-promotion-v1.json"
SPEC = importlib.util.spec_from_file_location("autoresearch_full_data_refit_v19r2", SCRIPT)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def _synthetic_cache(
    path: Path,
    *,
    replica: int,
    feature_delta: float = 0.0,
) -> Path:
    generator = torch.Generator().manual_seed(91)
    rows = 32
    labels = torch.tensor([label for label in range(4) for _ in range(8)])
    features = torch.randn(rows, module.HIDDEN_DIM, generator=generator).half()
    if feature_delta:
        features[0, 0] += feature_delta
    payload = {
        "schema_version": module.CACHE_SCHEMA,
        "build_spec_sha256": module.EXPECTED_BUILD_SPEC_SHA256,
        "extraction_replica": replica,
        "backbone": module.BACKBONE,
        "hidden_dim": module.HIDDEN_DIM,
        "features": features,
        "view_features": torch.randn(
            rows,
            2,
            module.HIDDEN_DIM,
            generator=generator,
        ).half(),
        "labels": labels,
        "class_names": {label: f"class-{label}" for label in range(4)},
        "views": 2,
        "view_seed": 3,
        "image_size": 224,
        "row_id": [f"row-{index:03d}" for index in range(rows)],
        "component_id": [f"component-{index // 4}" for index in range(rows)],
        "decoded_pixel_sha256": [f"pixel-{index}" for index in range(rows)],
        "class_label": labels.tolist(),
        "class_name": [f"class-{int(label)}" for label in labels],
        "fold": [1] * rows,
        "provenance": {
            "corpus_validation": {},
            "environment": {"replica": replica},
            "labels_used_by_ssl": False,
            "heldout_rows_consumed": 0,
            "external_holdout_consumed": False,
            "final_test_read": False,
            "runtime_write": False,
        },
    }
    torch.save(payload, path)
    module.v9.write_json(
        module.cache_sidecar_path(path),
        {
            "schema_version": module.CACHE_SIDECAR_SCHEMA,
            "cache_sha256": module.sha256_file(path),
            "cache_semantic_sha256": module.cache_semantic_sha256(payload),
        },
    )
    return path


def _write_fold_cache(path: Path, cache_path: Path) -> Path:
    cache = module.load_full_cache(cache_path, strict=False)
    midpoint = len(cache["row_id"]) // 2

    def split(start: int, end: int, *, views: bool) -> dict[str, object]:
        payload: dict[str, object] = {
            "row_id": cache["row_id"][start:end],
            "base_features": cache["features"][start:end].clone(),
        }
        if views:
            payload["view_features"] = cache["view_features"][start:end].clone()
        return payload

    torch.save(
        {
            "train": split(0, midpoint, views=True),
            "oof": split(midpoint, len(cache["row_id"]), views=False),
        },
        path,
    )
    return path


def test_r2_contract_is_hash_pinned() -> None:
    contract = module.validate_contract(BUILD_SPEC, E2E_SPEC)
    assert contract["build_spec_sha256"] == module.EXPECTED_BUILD_SPEC_SHA256
    assert contract["e2e_spec_sha256"] == module.EXPECTED_E2E_SPEC_SHA256
    assert contract["build_spec"]["runtime_write_allowed"] is False
    assert contract["e2e_spec"]["model_version_id"] == module.DEFAULT_VERSION_ID


def test_r2_geometry_and_seed_match_the_frozen_spec() -> None:
    assert module.BACKBONE == "dinov2_vitb14"
    assert module.HIDDEN_DIM == 768
    assert module.EMBEDDING_DIM == 128
    assert module.IMAGE_BATCH_SIZE == 4
    assert module.VIEWS == 8
    assert module.FULL_DATA_SEED == 1706651702


def test_cache_hash_tampering_is_rejected(tmp_path: Path) -> None:
    path = _synthetic_cache(tmp_path / "cache.pt", replica=1)
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="hash"):
        module.load_full_cache(path, strict=False)


def test_cache_replicas_use_semantic_equality_and_fold_cosine_gate(tmp_path: Path) -> None:
    left = _synthetic_cache(tmp_path / "left.pt", replica=1)
    right = _synthetic_cache(tmp_path / "right.pt", replica=2)
    fold_dir = tmp_path / "folds"
    fold_dir.mkdir()
    _write_fold_cache(fold_dir / "fold-01-views08.pt", left)

    result = module.verify_cache_replicas(
        left,
        right,
        fold_cache_dir=fold_dir,
        output_path=None,
        strict=False,
    )

    assert result["pass"] is True
    assert result["gates"]["semantic_hash_exact"] is True
    assert result["fold_reference_cosine"]["pass"] is True
    assert result["fold_reference_cosine"]["base_cosine_min"] >= module.COSINE_MIN
    assert result["fold_reference_cosine"]["view_cosine_min"] >= module.COSINE_MIN


def test_cache_replica_tensor_drift_is_rejected(tmp_path: Path) -> None:
    left = _synthetic_cache(tmp_path / "left.pt", replica=1)
    right = _synthetic_cache(tmp_path / "right.pt", replica=2, feature_delta=0.5)
    with pytest.raises(RuntimeError, match="cache reproducibility"):
        module.verify_cache_replicas(
            left,
            right,
            fold_cache_dir=None,
            output_path=None,
            strict=False,
        )


def _precompute_args(cache: Path, tmp_path: Path) -> dict:
    unused = tmp_path / "unused.json"
    return {
        "cache_path_value": cache,
        "extraction_replica": 1,
        "manifest_path": unused,
        "audit_path": unused,
        "backbone_manifest": unused,
        "views": module.VIEWS,
        "view_seed": module.VIEW_SEED,
        "image_size": module.IMAGE_SIZE,
        "image_batch_size": module.IMAGE_BATCH_SIZE,
        "num_workers": module.NUM_WORKERS,
        "device": torch.device("cpu"),
        "force": False,
    }


def test_precompute_reuse_requires_optin(tmp_path: Path, monkeypatch) -> None:
    cache = _synthetic_cache(tmp_path / "cache.pt", replica=1)
    payload = module.load_full_cache(cache, strict=False)
    monkeypatch.setattr(module, "load_full_cache", lambda *a, **k: payload)

    args = _precompute_args(cache, tmp_path)

    # Default: implicit reuse is forbidden for the auditable final build.
    with pytest.raises(ValueError, match="implicit reuse is forbidden"):
        module.precompute_full_cache(**args)

    # Explicit opt-in reuses after provenance validation.
    reused = module.precompute_full_cache(**{**args, "allow_reuse": True})
    assert reused["reused"] is True
    assert reused["cache_semantic_sha256"] == module.cache_semantic_sha256(payload)


def test_two_small_r2_refits_are_semantically_exact(tmp_path: Path, monkeypatch) -> None:
    left_cache = _synthetic_cache(tmp_path / "left.pt", replica=1)
    right_cache = _synthetic_cache(tmp_path / "right.pt", replica=2)
    monkeypatch.setattr(
        module,
        "fit_metrics",
        lambda *args: {
            "context": module.METRIC_CONTEXT,
            "values": {"prototype_top1": 0.5},
        },
    )
    kwargs = {
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
    first = module.train_replica(
        cache_path_value=left_cache,
        output_dir=tmp_path / "first",
        **kwargs,
    )
    second = module.train_replica(
        cache_path_value=right_cache,
        output_dir=tmp_path / "second",
        **kwargs,
    )
    assert first["state_dict_sha256"] == second["state_dict_sha256"]
    assert first["prototypes_sha256"] == second["prototypes_sha256"]

    second_path = tmp_path / "second/checkpoint.pt"
    second_payload = torch.load(second_path, map_location="cpu", weights_only=False)
    second_payload["serialization_note"] = "raw bytes are not a semantic gate"
    torch.save(second_payload, second_path)

    result = module.verify_refit_replicas(
        tmp_path / "first/checkpoint.pt",
        second_path,
        output_path=None,
        strict=False,
    )
    assert result["pass"] is True
    assert result["checkpoint_files_byte_exact"] is False
    assert all(result["gates"].values())


def test_checkpoint_carries_frozen_promotion_contract(tmp_path: Path, monkeypatch) -> None:
    cache = _synthetic_cache(tmp_path / "cache.pt", replica=1)
    monkeypatch.setattr(
        module,
        "fit_metrics",
        lambda *args: {"context": module.METRIC_CONTEXT, "values": {}},
    )
    module.train_replica(
        cache_path_value=cache,
        output_dir=tmp_path / "result",
        seed=101,
        ssl_epochs=1,
        ssl_batch_size=16,
        supervised_epochs=1,
        episodes_per_epoch=1,
        n_way=2,
        k_shot=1,
        q_queries=1,
        device=torch.device("cpu"),
        supervised_device=torch.device("cpu"),
        strict_cache=False,
    )
    checkpoint = torch.load(
        tmp_path / "result/checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint["promotion_eligible"] is False
    assert checkpoint["promotion_contract"] == {
        "e2e_report_required": True,
        "e2e_spec_path": "backend/model_registry/specs/r2-e2e-promotion-v1.json",
        "e2e_spec_sha256": module.EXPECTED_E2E_SPEC_SHA256,
        "build_spec_path": "backend/model_registry/specs/r2-full-data-production-v1.json",
        "build_spec_sha256": module.EXPECTED_BUILD_SPEC_SHA256,
    }


def test_runner_exposes_build_only_commands() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    for command in (
        "precompute",
        "verify-cache",
        "refit",
        "verify-refit",
        "export",
        "all",
    ):
        assert command in completed.stdout

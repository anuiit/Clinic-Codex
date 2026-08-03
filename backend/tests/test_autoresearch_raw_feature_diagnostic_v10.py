from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
V9_SCRIPT = ROOT / "scripts" / "autoresearch_self_supervised_v9.py"
V9_SPEC = importlib.util.spec_from_file_location("autoresearch_self_supervised_v9", V9_SCRIPT)
assert V9_SPEC is not None and V9_SPEC.loader is not None
v9 = importlib.util.module_from_spec(V9_SPEC)
sys.modules[V9_SPEC.name] = v9
V9_SPEC.loader.exec_module(v9)

SCRIPT = ROOT / "scripts" / "autoresearch_raw_feature_diagnostic_v10.py"
SPEC = importlib.util.spec_from_file_location("autoresearch_raw_feature_diagnostic_v10", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def _baseline_row(seed: int, *, cache_hash: str = "cache-1") -> dict[str, object]:
    return {
        "row_id": "row-1",
        "label": 7,
        "class_name": "class-7",
        "provenance_component": "component-1",
        "decoded_pixel_sha256": "pixel-1",
        "outer_fold": 1,
        "seed": seed,
        "baseline_topk": [7, 2, 3],
        "data_sha256": cache_hash,
        "code_sha256": "baseline-code",
        "folds_sha256": "folds-hash",
    }


def test_classification_uses_strict_preregistered_interval_rules() -> None:
    assert module.classify_interval([0.001, 0.2]) == "raw_superior"
    assert module.classify_interval([-0.2, -0.001]) == "learned_projection_helpful"
    assert module.classify_interval([0.0, 0.2]) == "neutral_or_inconclusive"
    assert module.classify_interval([-0.2, 0.0]) == "neutral_or_inconclusive"
    with pytest.raises(ValueError, match="finite"):
        module.classify_interval([float("nan"), 0.2])


def test_raw_predictions_are_duplicated_but_identical_across_seeds() -> None:
    records = [
        {"outer_fold": 1, "row_id": "row-1", "seed": seed, "candidate_topk": [4, 5, 6]}
        for seed in (17, 42, 73)
    ]

    report = module.raw_seed_variant_report(records, expected_seeds=[17, 42, 73])

    assert report["oof_row_groups"] == 1
    assert report["maximum_unique_raw_prediction_variants_per_oof_row"] == 1
    assert report["raw_predictions_identical_across_seeds"] is True
    assert report["missing_or_extra_seed_groups"] == []


def test_raw_seed_variant_report_detects_seed_dependent_candidate() -> None:
    records = [
        {"outer_fold": 1, "row_id": "row-1", "seed": 17, "candidate_topk": [4, 5, 6]},
        {"outer_fold": 1, "row_id": "row-1", "seed": 42, "candidate_topk": [4, 5, 6]},
        {"outer_fold": 1, "row_id": "row-1", "seed": 73, "candidate_topk": [9, 5, 6]},
    ]

    report = module.raw_seed_variant_report(records, expected_seeds=[17, 42, 73])

    assert report["maximum_unique_raw_prediction_variants_per_oof_row"] == 2
    assert report["raw_predictions_identical_across_seeds"] is False


def test_build_paired_records_replays_b0_without_retraining() -> None:
    baseline = [_baseline_row(seed) for seed in (17, 42, 73)]
    raw = {(1, "row-1"): [4, 5, 6]}
    metadata = {
        (1, "row-1"): {
            "label": 7,
            "class_name": "class-7",
            "provenance_component": "component-1",
            "decoded_pixel_sha256": "pixel-1",
        }
    }

    records, mismatches = module.build_paired_records(
        baseline,
        raw,
        metadata,
        cache_hashes={"1": "cache-1"},
        baseline_predictions_sha256="baseline-predictions",
        diagnostic_code_sha256="diagnostic-code",
    )

    assert mismatches == 0
    assert len(records) == 3
    assert {tuple(row["candidate_topk"]) for row in records} == {(4, 5, 6)}
    assert all(row["baseline_retrained"] is False for row in records)
    assert all(row["raw_prediction_seed_dependent"] is False for row in records)


def test_build_paired_records_counts_source_cache_hash_mismatch() -> None:
    baseline = [_baseline_row(17, cache_hash="wrong-cache")]
    raw = {(1, "row-1"): [4, 5, 6]}
    metadata = {
        (1, "row-1"): {
            "label": 7,
            "class_name": "class-7",
            "provenance_component": "component-1",
            "decoded_pixel_sha256": "pixel-1",
        }
    }

    _, mismatches = module.build_paired_records(
        baseline,
        raw,
        metadata,
        cache_hashes={"1": "cache-1"},
        baseline_predictions_sha256="baseline-predictions",
        diagnostic_code_sha256="diagnostic-code",
    )

    assert mismatches == 1

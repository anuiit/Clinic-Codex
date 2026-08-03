from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/autoresearch_coverage_power_audit_v10.py"
SPEC = importlib.util.spec_from_file_location("autoresearch_coverage_power_audit_v10", SCRIPT)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def _row(row_id: str, label: int, component: str, fold: int, family: str = "family") -> dict[str, object]:
    return {
        "row_id": row_id,
        "class_label": label,
        "class_name": f"class-{label}",
        "component_id": component,
        "fold": fold,
        "source_family": family,
    }


def test_coverage_report_counts_classes_components_and_transfer() -> None:
    report = module.coverage_report(
        [
            _row("a", 0, "c0", 1),
            _row("b", 0, "c1", 2),
            _row("c", 1, "c2", 3),
        ]
    )
    assert report["rows"] == 3
    assert report["classes"] == 2
    assert report["provenance_components"] == 3
    assert report["classes_with_one_component"] == 1
    assert report["classes_with_two_or_more_components"] == 1
    assert report["classes_with_components_in_two_or_more_folds"] == 1
    assert report["per_class"][0]["inter_component_transfer_evaluable"] is True


def test_coverage_report_fails_when_component_crosses_folds() -> None:
    with pytest.raises(ValueError, match="components cross folds"):
        module.coverage_report([_row("a", 0, "shared", 1), _row("b", 0, "shared", 2)])


def test_interval_sensitivity_is_explicitly_conditional() -> None:
    report = module.interval_sensitivity([-0.01, 0.02], current_components=25)
    assert report["half_width"] == pytest.approx(0.015)
    assert report["conditional_required_components"]["half_width_0.010"] == 57
    assert report["conditional_required_components"]["half_width_0.005"] == 225
    assert "not a prospective power guarantee" in report["assumption"]


def test_dead_fold_report_flags_disjoint_source_family() -> None:
    manifest = [
        _row("train", 0, "train-component", 1, family="scanner-a"),
        _row("oof", 0, "oof-component", 3, family="scanner-b"),
    ]
    predictions = []
    for seed in module.EXPECTED_SEEDS:
        predictions.append(
            {
                "row_id": "oof",
                "outer_fold": 3,
                "seed": seed,
                "label": 0,
                "class_name": "class-0",
                "provenance_component": "oof-component",
                "baseline_topk": [1, 2, 3],
                "candidate_topk": [1, 2, 3],
            }
        )
    report = module.dead_fold_report(manifest, predictions, folds=(3,))["3"]
    assert report["both_arms_zero_top3"] is True
    assert report["metadata_shift_present"] is True
    assert report["classes"][0]["source_family_overlap"] == []


def test_iteration_4_contract_is_descriptive_and_non_promotable() -> None:
    contract = module.validate_contract(module.DEFAULT_SPEC, module.DEFAULT_EVALUATOR)
    spec = module.read_json(module.DEFAULT_SPEC)
    evaluator = module.read_json(module.DEFAULT_EVALUATOR)
    assert contract["spec_sha256"]
    assert spec["single_diagnostic_factor"]["model_parameters_changed"] is False
    assert evaluator["outcome_is_descriptive_not_a_pass_gate"] is True
    assert evaluator["promotion_eligible"] is False


def test_require_hash_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "input.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        module.require_hash(path, "0" * 64, "fixture")

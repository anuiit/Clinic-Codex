from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/autoresearch_support_aware_readout_v10.py"
SPEC = ROOT / ".omc/autoresearch/elements-baseline-replacement/runs/20260803-self-supervised-v10/specs/iteration-0005.json"
EVALUATOR = ROOT / ".omc/autoresearch/elements-baseline-replacement/runs/20260803-self-supervised-v10/evaluator-iteration-0005.json"
spec = importlib.util.spec_from_file_location("autoresearch_support_aware_readout_v10", SCRIPT)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_iteration_5_contract_is_hash_pinned_and_valid() -> None:
    contract = module.validate_contract(SPEC, EVALUATOR)
    assert contract["spec_sha256"] == module.EXPECTED_SPEC_SHA256
    assert contract["evaluator_sha256"] == module.EXPECTED_EVALUATOR_SHA256


def test_support_aware_plan_is_deterministic_fixed_shape_and_disjoint() -> None:
    labels = torch.tensor([10] * 2 + [20] * 3 + [30] * 7 + [40] * 8, dtype=torch.long)
    kwargs = dict(n_way=4, k_shot=3, q_queries=5, epochs=2, episodes_per_epoch=3, seed=19)
    first, first_hash, first_diag = module.build_support_aware_episode_plan(labels, **kwargs)
    second, second_hash, second_diag = module.build_support_aware_episode_plan(labels, **kwargs)
    assert first == second
    assert first_hash == second_hash
    assert first_diag == second_diag
    assert first_diag["eligible_classes"] == 4
    assert first_diag["support_query_source_row_overlap"] == 0
    assert first_diag["scarce_class_slots"] > 0
    for epoch in first:
        for support, support_labels, query, query_labels in epoch:
            assert len(support) == 4 * 3
            assert len(query) == 4 * 5
            assert len(support_labels) == len(support)
            assert len(query_labels) == len(query)
            for local_label in range(4):
                local_support = {index for index, label in zip(support, support_labels) if label == local_label}
                local_query = {index for index, label in zip(query, query_labels) if label == local_label}
                assert local_support
                assert local_query
                assert local_support.isdisjoint(local_query)


def test_support_aware_plan_preserves_historical_plan_when_all_classes_have_eight_rows() -> None:
    labels = torch.tensor([label for label in range(5) for _ in range(8)], dtype=torch.long)
    kwargs = dict(n_way=5, k_shot=3, q_queries=5, epochs=2, episodes_per_epoch=2, seed=23)
    historical, historical_hash = module.v9.build_episode_plan(labels, **kwargs)
    candidate, candidate_hash, diagnostics = module.build_support_aware_episode_plan(labels, **kwargs)
    assert candidate == historical
    assert candidate_hash == historical_hash
    assert diagnostics["scarce_class_slots"] == 0
    assert diagnostics["support_query_source_row_overlap"] == 0


def test_plan_audit_detects_oof_exposure_and_invalid_indices() -> None:
    plan = [[([0, 0, 0], [0, 0, 0], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0])]]
    clean = module.audit_plan_train_only(plan, ["train-a", "train-b"], ["oof-a"])
    assert clean == {"candidate_episode_oof_row_exposure": 0, "candidate_episode_index_out_of_range": 0}
    exposed = module.audit_plan_train_only(plan, ["train-a", "oof-a"], ["oof-a"])
    assert exposed["candidate_episode_oof_row_exposure"] == 5
    invalid_plan = [[([2], [0], [1], [0])]]
    invalid = module.audit_plan_train_only(invalid_plan, ["train-a", "train-b"], [])
    assert invalid["candidate_episode_index_out_of_range"] == 1


def test_efficacy_gate_semantics_require_practical_effect_and_both_rank_guards() -> None:
    passing = {
        "delta_top1": 0.02,
        "positive_seed_count": 2,
        "delta_top1_component_bootstrap_lower_95": 0.001,
        "delta_macro_top1": -0.004,
        "affected_oof_delta_top1": 0.01,
        "minimum_candidate_over_control_effective_rank_ratio": 0.95,
        "minimum_candidate_over_persisted_B0_effective_rank_ratio": 0.91,
    }
    assert all(module.efficacy_gate_passes(passing).values())
    for key in (
        "delta_top1",
        "delta_top1_component_bootstrap_lower_95",
        "affected_oof_delta_top1",
        "minimum_candidate_over_control_effective_rank_ratio",
        "minimum_candidate_over_persisted_B0_effective_rank_ratio",
    ):
        failing = dict(passing)
        failing[key] = 0.0
        assert not all(module.efficacy_gate_passes(failing).values())


def test_scarce_class_resampling_uses_duplicates_only_within_disjoint_sides() -> None:
    labels = torch.tensor([1] * 2 + [2] * 8, dtype=torch.long)
    plan, _digest, diagnostics = module.build_support_aware_episode_plan(
        labels,
        n_way=2,
        k_shot=3,
        q_queries=5,
        epochs=1,
        episodes_per_epoch=1,
        seed=3,
    )
    support, support_labels, query, query_labels = plan[0][0]
    scarce_local = next(
        local
        for local in set(support_labels)
        if {int(labels[index]) for index, value in zip(support, support_labels) if value == local} == {1}
    )
    scarce_support = [index for index, value in zip(support, support_labels) if value == scarce_local]
    scarce_query = [index for index, value in zip(query, query_labels) if value == scarce_local]
    assert len(set(scarce_support)) == 1
    assert len(set(scarce_query)) == 1
    assert set(scarce_support).isdisjoint(scarce_query)
    assert diagnostics["support_query_source_row_overlap"] == 0

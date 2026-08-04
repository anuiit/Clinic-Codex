from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/autoresearch_supervised_multiview_v17.py"
spec = importlib.util.spec_from_file_location(
    "autoresearch_supervised_multiview_v17", SCRIPT
)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_contract_is_hash_pinned_and_keeps_runtime_sealed() -> None:
    contract = module.validate_contract(module.DEFAULT_SPEC, module.DEFAULT_EVALUATOR)
    assert contract == {
        "spec_sha256": module.EXPECTED_SPEC_SHA256,
        "evaluator_sha256": module.EXPECTED_EVALUATOR_SHA256,
    }
    spec_payload = module.read_json(module.DEFAULT_SPEC)
    assert spec_payload["changed_factor"]["only_change"] == (
        "supervised support/query input feature source"
    )
    assert spec_payload["hard_boundaries"]["final_test_read"] is False
    assert spec_payload["hard_boundaries"]["runtime_write"] is False
    assert spec_payload["hard_boundaries"]["promotion"] is False


def test_view_index_is_exact_stable_seed_formula_with_row_identity() -> None:
    expected = (
        module.v9.stable_seed(
            "v17-multiview", 3, 42, 7, 11, "query", 19, "row:abc"
        )
        % 8
    )
    assert module.deterministic_view_index(
        fold=3,
        seed=42,
        epoch=7,
        episode_index=11,
        role="query",
        slot_position=19,
        row_id="row:abc",
        views=8,
    ) == expected
    assert module.deterministic_view_index(
        fold=3,
        seed=42,
        epoch=7,
        episode_index=11,
        role="query",
        slot_position=19,
        row_id="row:def",
        views=8,
    ) == (
        module.v9.stable_seed(
            "v17-multiview", 3, 42, 7, 11, "query", 19, "row:def"
        )
        % 8
    )


def test_view_plan_is_deterministic_losslessly_audited_and_oof_free() -> None:
    episode_plan = [
        [
            ([0, 1], [0, 1], [2, 3], [0, 1]),
            ([3, 2], [0, 1], [1, 0], [0, 1]),
        ]
    ]
    row_ids = ["train:a", "train:b", "train:c", "train:d"]
    first_plan, first_audit = module.build_view_plan(
        episode_plan,
        row_ids,
        fold=1,
        seed=17,
        views=8,
        oof_row_ids=["oof:z"],
    )
    second_plan, second_audit = module.build_view_plan(
        episode_plan,
        row_ids,
        fold=1,
        seed=17,
        views=8,
        oof_row_ids=["oof:z"],
    )
    assert first_plan == second_plan
    assert first_audit == second_audit
    assert first_audit["occurrence_count"] == 8
    assert sum(first_audit["per_view_usage_counts"]) == 8
    assert set(first_audit["per_row_view_usage_counts"]) == set(row_ids)
    assert all(
        len(counts) == 8
        for counts in first_audit["per_row_view_usage_counts"].values()
    )
    assert sum(
        sum(counts)
        for counts in first_audit["per_row_view_usage_counts"].values()
    ) == 8
    assert first_audit["candidate_episode_oof_row_exposure"] == 0
    assert len(first_audit["view_plan_sha256"]) == 64


def test_usage_balance_uses_relative_deviation_from_uniform() -> None:
    passed, maximum = module.usage_within_balance(
        [1000, 1001, 999, 1000, 1000, 1000, 1000, 1000], tolerance=0.05
    )
    assert passed is True
    assert maximum == 0.001
    failed, maximum = module.usage_within_balance(
        [1200, 1000, 1000, 1000, 1000, 1000, 1000, 800], tolerance=0.05
    )
    assert failed is False
    assert maximum == 0.2


def test_verdict_tiers_follow_frozen_council_order() -> None:
    common = {
        "integrity_pass": True,
        "engagement_stability_pass": True,
        "b0_anchor_pass": True,
        "c1_noninferiority_pass": True,
    }
    assert module.classify_verdict(
        **common,
        candidate_minus_c1_delta_top1=0.011,
        candidate_minus_c1_bootstrap_lower=0.001,
        candidate_minus_c1_positive_seed_count=2,
    ) == ("supported_strong", "component_level_supported")
    assert module.classify_verdict(
        **common,
        candidate_minus_c1_delta_top1=0.006,
        candidate_minus_c1_bootstrap_lower=-0.002,
        candidate_minus_c1_positive_seed_count=2,
    ) == ("supported_reference", "unresolved_at_component_level_power")
    assert module.classify_verdict(
        **common,
        candidate_minus_c1_delta_top1=0.001,
        candidate_minus_c1_bootstrap_lower=-0.002,
        candidate_minus_c1_positive_seed_count=1,
    ) == ("neutral_preservation", "not_evidence_of_a_better_model")
    assert module.classify_verdict(
        **{**common, "c1_noninferiority_pass": False},
        candidate_minus_c1_delta_top1=-0.02,
        candidate_minus_c1_bootstrap_lower=-0.03,
        candidate_minus_c1_positive_seed_count=0,
    ) == ("not_supported", "valid_but_gate_failure")
    assert module.classify_verdict(
        **{**common, "integrity_pass": False},
        candidate_minus_c1_delta_top1=0.02,
        candidate_minus_c1_bootstrap_lower=0.01,
        candidate_minus_c1_positive_seed_count=3,
    ) == ("invalid", "candidate_results_uninterpretable")


def test_multiview_training_is_deterministic_on_the_frozen_view_plan() -> None:
    episode_plan = [
        [([0, 1], [0, 1], [2, 3], [0, 1])],
        [([2, 3], [0, 1], [0, 1], [0, 1])],
    ]
    view_plan, audit = module.build_view_plan(
        episode_plan,
        ["row:a", "row:b", "row:c", "row:d"],
        fold=2,
        seed=42,
        views=8,
    )
    view_features = torch.arange(4 * 8 * 4, dtype=torch.float32).reshape(4, 8, 4)
    view_features = view_features / view_features.abs().max()
    module.v9.configure_determinism(909)
    initial = module.v9.ProjectionHead(input_dim=4, embedding_dim=2)
    first = copy.deepcopy(initial)
    second = copy.deepcopy(initial)
    kwargs = {
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "temperature": 0.1,
        "warmup_epochs": 1,
        "rng_seed": 12345,
        "device": torch.device("cpu"),
    }
    first_result = module.train_supervised_multiview(
        first,
        view_features,
        episode_plan,
        view_plan,
        **kwargs,
    )
    second_result = module.train_supervised_multiview(
        second,
        view_features,
        episode_plan,
        view_plan,
        **kwargs,
    )
    assert module.v9.state_dict_sha256(first) == module.v9.state_dict_sha256(second)
    assert first_result == second_result
    assert first_result["epochs"] == 2
    assert first_result["episodes_per_epoch"] == 1
    assert first_result["episode_total"] == 2
    assert first_result["minimum_post_l2_query_batch_mean_std"] > 0.0

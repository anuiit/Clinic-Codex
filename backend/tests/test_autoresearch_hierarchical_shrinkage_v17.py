from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/autoresearch_hierarchical_shrinkage_v17.py"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "autoresearch_hierarchical_shrinkage_v17_tested", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_contract_and_canonical_cache_pins_are_frozen() -> None:
    module = load_module()
    contract = module.validate_contract(
        module.DEFAULT_SPEC,
        module.DEFAULT_EVALUATOR,
        module.DEFAULT_ERRATUM,
    )

    declared = {
        int(key): value
        for key, value in contract["spec"]["frozen_inputs"][
            "cache_sha256_by_fold"
        ].items()
    }
    assert declared == module.i5.EXPECTED_CACHE_SHA256
    assert contract["spec_sha256"] == module.EXPECTED_SPEC_SHA256
    assert contract["evaluator_sha256"] == module.EXPECTED_EVALUATOR_SHA256
    assert contract["erratum_sha256"] == module.EXPECTED_ERRATUM_SHA256


def test_replay_normalization_contract_is_identity_and_hash_pinned() -> None:
    module = load_module()

    assert module.REPLAY_NORMALIZATION_CONTRACT["summary_removed_keys"] == []
    assert module.REPLAY_NORMALIZATION_CONTRACT["pair_removed_keys"] == []
    assert (
        module.replay_normalization_sha256()
        == module.EXPECTED_REPLAY_NORMALIZATION_SHA256
    )


def test_build_prototypes_matches_frozen_geometry() -> None:
    module = load_module()
    embeddings = F.normalize(
        torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.8, 0.6, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.8, 0.6],
            ]
        ),
        dim=1,
    )
    labels = torch.tensor([2, 2, 7, 7])

    prototype_labels, control, candidate, diagnostics = module.build_prototypes(
        embeddings, labels, shrinkage_lambda=8.0
    )

    global_mean = embeddings.mean(dim=0)
    expected_control_2 = F.normalize(embeddings[:2].mean(dim=0), dim=0)
    expected_candidate_2 = F.normalize(
        (2 * embeddings[:2].mean(dim=0) + 8 * global_mean) / 10,
        dim=0,
    )
    assert prototype_labels.tolist() == [2, 7]
    assert torch.equal(control[0], expected_control_2)
    assert torch.equal(candidate[0], expected_candidate_2)
    assert [item["train_support"] for item in diagnostics] == [2, 2]
    assert all(item["support_bin_three"] == "n_y_lte_8" for item in diagnostics)
    assert all(item["prototype_changed"] for item in diagnostics)


def test_control_formula_is_exact_chordal_mean_normalization() -> None:
    module = load_module()
    embeddings = F.normalize(torch.arange(1, 25, dtype=torch.float32).reshape(6, 4), dim=1)
    labels = torch.tensor([0, 0, 1, 1, 1, 2])

    prototype_labels, controls, _candidates, _diagnostics = module.build_prototypes(
        embeddings, labels
    )

    for index, label in enumerate(prototype_labels.tolist()):
        expected = F.normalize(embeddings[labels == label].mean(dim=0), dim=0)
        assert torch.equal(controls[index], expected)


def test_predict_topk_uses_sorted_prototype_labels() -> None:
    module = load_module()
    labels = torch.tensor([3, 9, 12])
    prototypes = torch.eye(3)
    queries = torch.tensor([[0.1, 0.9, 0.2], [0.8, 0.2, 0.1]])

    assert module.predict_topk(queries, labels, prototypes) == [
        [9, 12, 3],
        [3, 9, 12],
    ]


@pytest.mark.parametrize(
    ("count", "three", "binary"),
    [
        (2, "n_y_lte_8", "n_y_lte_8"),
        (8, "n_y_lte_8", "n_y_lte_8"),
        (9, "n_y_9_to_31", "n_y_gt_8"),
        (31, "n_y_9_to_31", "n_y_gt_8"),
        (32, "n_y_gte_32", "n_y_gt_8"),
    ],
)
def test_support_bins_are_frozen(count: int, three: str, binary: str) -> None:
    module = load_module()

    assert module.support_bin_three(count) == three
    assert module.support_bin_binary(count) == binary


@pytest.mark.parametrize(
    ("kwargs", "decision", "claim"),
    [
        (
            dict(
                integrity_pass=False,
                engagement_pass=True,
                b0_anchor_pass=True,
                c1_noninferiority_pass=True,
                c1_delta_top1=0.02,
                c1_bootstrap_lower=0.01,
                c1_positive_seed_count=3,
            ),
            "invalid",
            "candidate_results_uninterpretable",
        ),
        (
            dict(
                integrity_pass=True,
                engagement_pass=True,
                b0_anchor_pass=True,
                c1_noninferiority_pass=True,
                c1_delta_top1=0.01,
                c1_bootstrap_lower=0.0001,
                c1_positive_seed_count=3,
            ),
            "supported_strong",
            "component_level_supported",
        ),
        (
            dict(
                integrity_pass=True,
                engagement_pass=True,
                b0_anchor_pass=True,
                c1_noninferiority_pass=True,
                c1_delta_top1=0.005,
                c1_bootstrap_lower=-0.001,
                c1_positive_seed_count=2,
            ),
            "supported_reference",
            "unresolved_at_component_level_power",
        ),
        (
            dict(
                integrity_pass=True,
                engagement_pass=True,
                b0_anchor_pass=True,
                c1_noninferiority_pass=True,
                c1_delta_top1=0.0049,
                c1_bootstrap_lower=-0.001,
                c1_positive_seed_count=3,
            ),
            "neutral_preservation",
            "not_evidence_of_a_better_model",
        ),
    ],
)
def test_verdict_tiers_match_frozen_evaluator(
    kwargs: dict[str, object], decision: str, claim: str
) -> None:
    module = load_module()

    assert module.classify_verdict(**kwargs) == (decision, claim)

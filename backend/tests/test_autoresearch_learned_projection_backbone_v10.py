from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/autoresearch_learned_projection_backbone_v10.py"
SPEC = importlib.util.spec_from_file_location("autoresearch_learned_projection_backbone_v10", SCRIPT)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_fixed_hidden_projection_has_preregistered_abi_and_unit_norm() -> None:
    model = module.FixedHiddenProjection(input_dim=768, hidden_dim=384, embedding_dim=128)
    assert tuple(model.net[0].weight.shape) == (384, 768)
    assert tuple(model.net[3].weight.shape) == (128, 384)
    model.eval()
    output = model(torch.randn(4, 768))
    assert tuple(output.shape) == (4, 128)
    assert torch.allclose(output.norm(dim=1), torch.ones(4), atol=1e-6)


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[2] / ".omc/autoresearch/elements-baseline-replacement/runs/20260803-self-supervised-v9/iteration-0001/results/paired_seed_summary.json").exists(),
    reason="requires unshipped research artifact: results/paired_seed_summary.json",
)
def test_B14_initialization_reconstructs_B0_and_reuses_common_output() -> None:
    summary = module.read_json(module.DEFAULT_V9_SUMMARY)
    diagnostic = next(item for item in summary["diagnostics"] if item["fold"] == 1 and item["seed"] == 17)
    model, report = module.build_b14_projection(17, diagnostic["initial_state_sha256"])
    assert report["architecture"] == "768-384-128"
    assert report["common_output_initialization_matches_reconstructed_v9_B0"] is True
    assert tuple(model.net[0].weight.shape) == (384, 768)
    assert tuple(model.net[3].weight.shape) == (128, 384)


def test_require_sha256_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        module.require_sha256(path, "0" * 64, "fixture")


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[2] / ".omc/autoresearch/elements-baseline-replacement/runs/20260803-self-supervised-v10/specs/iteration-0003.json").exists(),
    reason="requires unshipped research artifact: specs/iteration-0003.json",
)
def test_iteration_contract_keeps_original_gates() -> None:
    contract = module.validate_iteration_contract(
        module.DEFAULT_SPEC,
        module.DEFAULT_ADDENDUM,
        module.DEFAULT_EVALUATOR,
    )
    assert contract["spec"]["evaluation"]["efficacy_gates"] == {
        "positive_seed_count": 2,
        "delta_top1_component_bootstrap_lower_95_gt": 0.0,
        "delta_macro_top1_component_bootstrap_lower_95_gte": 0.0,
        "B14_effective_rank_ratio_gte": 0.9,
    }
    assert contract["addendum"]["locked_before_any_iteration_0003_B14_prediction"] is True


def _strict_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for oof_index in range(653):
        fold = oof_index % 5 + 1
        label = oof_index % 37
        for seed in module.EXPECTED_SEEDS:
            records.append(
                {
                    "row_id": f"row-{oof_index}",
                    "provenance_component": f"component-{oof_index}",
                    "outer_fold": fold,
                    "seed": seed,
                    "label": label,
                    "baseline_topk": [(label + 1) % 37, (label + 2) % 37, (label + 3) % 37],
                    "candidate_topk": [label, (label + 1) % 37, (label + 2) % 37],
                    "coverage_stratum": (
                        "single_train_component" if oof_index % 2 == 0 else "multiple_train_components"
                    ),
                }
            )
    assert len(records) == module.EXPECTED_PAIRED_ROWS
    return records


def _strict_diagnostics(rank_ratio: float = 1.0) -> list[dict[str, object]]:
    return [
        {
            "fold": fold,
            "seed": seed,
            "S14_effective_rank": 20.0,
            "B14_effective_rank": 20.0 * rank_ratio,
            "B14_over_S14_effective_rank_ratio": rank_ratio,
            "initialization": {"common_output_initialization_matches_reconstructed_v9_B0": True},
            "episode_plan_matches_persisted_B0": True,
            "supervised_budget_and_optimizer_match_persisted_B0": True,
        }
        for fold in module.EXPECTED_FOLDS
        for seed in module.EXPECTED_SEEDS
    ]


def _isolation() -> dict[str, int]:
    return {
        "provenance_component_overlap_across_folds": 0,
        "decoded_pixel_hash_overlap_across_folds": 0,
    }


def test_evaluator_passes_only_when_integrity_and_efficacy_pass() -> None:
    result = module.evaluate_records(
        _strict_records(),
        _strict_diagnostics(),
        runtime_unchanged=True,
        bootstrap_replicates=100,
        B0_provenance_exact=True,
        B14_cache_valid=True,
        fold_isolation=_isolation(),
    )
    assert result["integrity_pass"] is True
    assert result["efficacy_pass"] is True
    assert result["pass"] is True
    assert result["promotion_eligible"] is False
    assert result["runtime_promotion"] is False
    assert result["final_test_read"] is False
    assert result["descriptive_breakdowns"]["non_gating"] is True


def test_effective_rank_gate_is_not_relaxed() -> None:
    result = module.evaluate_records(
        _strict_records(),
        _strict_diagnostics(rank_ratio=0.89),
        runtime_unchanged=True,
        bootstrap_replicates=100,
        B0_provenance_exact=True,
        B14_cache_valid=True,
        fold_isolation=_isolation(),
    )
    assert result["integrity_pass"] is True
    assert result["gate_passes"]["minimum_B14_over_S14_effective_rank_ratio_gte"] is False
    assert result["efficacy_pass"] is False
    assert result["pass"] is False
    assert result["decision"] == "stop_frozen_backbone_scaling_in_this_run"


def test_parser_defaults_to_exact_preregistered_grid() -> None:
    args = module.build_parser().parse_args(["run"])
    assert args.folds == module.EXPECTED_FOLDS
    assert args.seeds == module.EXPECTED_SEEDS
    assert args.bootstrap_replicates == 2000
    assert args.supervised_device == "cpu"

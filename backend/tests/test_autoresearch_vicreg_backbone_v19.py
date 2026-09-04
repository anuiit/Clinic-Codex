from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/autoresearch_vicreg_backbone_v19.py"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "autoresearch_vicreg_backbone_v19_tested", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def default_args(module):
    return module.build_parser().parse_args(["validate-contract"])


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[2] / ".omc/autoresearch/elements-baseline-replacement/runs/20260803-self-supervised-v9/test-spec.json").exists(),
    reason="requires unshipped research artifact: 20260803-self-supervised-v9/test-spec.json",
)
def test_static_contract_hashes_every_frozen_input_without_data_operations() -> None:
    module = load_module()
    result = module.validate_static_contract(default_args(module))

    assert result["pass"] is True
    assert result["canonical_checkpoint_hash_match_count"] == 30
    assert result["canonical_diagnostic_count"] == 15
    assert result["canonical_prediction_row_count"] == 1959
    assert set(result["canonical_cache_pins"]) == {"1", "2", "3", "4", "5"}
    assert set(result["operation_counts"].values()) == {0}
    assert result["final_test_read"] is False
    assert result["runtime_unchanged"] is True


def test_phase_authorization_is_hash_bound_to_frozen_runner(tmp_path: Path) -> None:
    module = load_module()
    args = default_args(module)
    freeze = {
        "frozen_artifacts": {
            "runner_sha256": module.v9.sha256_file(module.Path(module.__file__)),
            "spec_sha256": module.EXPECTED_SPEC_SHA256,
            "evaluator_sha256": module.EXPECTED_EVALUATOR_SHA256,
        }
    }
    args.freeze_audit = tmp_path / "freeze.json"
    module.write_json(args.freeze_audit, freeze)
    args.authorization = tmp_path / "authorization.json"
    module.write_json(
        args.authorization,
        {
            "schema_version": "autoresearch-v19.phase-authorization-v1",
            "authorized": True,
            "phase": "phase_2_extraction_and_c1_control",
            "council_session_id": "adv_20260803T072824_f903d4d6",
            "contract_freeze_audit_sha256": module.v9.sha256_file(args.freeze_audit),
            "published_synthesis_sha256": "council-hash",
        },
    )

    result = module.validate_phase_authorization(
        args, expected_phase="phase_2_extraction_and_c1_control"
    )

    assert result["phase"] == "phase_2_extraction_and_c1_control"
    module.write_json(args.authorization, {**module.read_json(args.authorization), "authorized": False})
    with pytest.raises(ValueError, match="not affirmative"):
        module.validate_phase_authorization(
            args, expected_phase="phase_2_extraction_and_c1_control"
        )


def test_commands_refuse_data_phases_before_authorization() -> None:
    module = load_module()
    args = module.build_parser().parse_args(["precompute"])

    with pytest.raises(ValueError, match="requires --authorization"):
        module.command_precompute(args)


def test_view_plan_is_deterministic_order_sensitive_and_v9_seeded() -> None:
    module = load_module()
    rows = ["row-a", "row-b"]

    first = module.view_plan_sha256(rows)
    second = module.view_plan_sha256(rows)
    reversed_rows = module.view_plan_sha256(list(reversed(rows)))

    assert first == second
    assert first != reversed_rows
    assert module.v9.stable_seed(
        "ssl-view", module.VIEW_SEED, "row-a", 0
    ) != module.v9.stable_seed("ssl-view", module.VIEW_SEED, "row-a", 1)


def test_candidate_rng_is_restored_to_canonical_v9_post_head_state() -> None:
    module = load_module()

    candidate, audit = module.aligned_candidate_initialization(17)
    module.configure_determinism(17)
    canonical = module.ProjectionHead(input_dim=384, embedding_dim=128)

    assert candidate.net[0].in_features == 768
    assert candidate.net[-1].out_features == 128
    assert module.v9.state_dict_sha256(canonical) == audit["canonical_c1_initial_state_sha256"]
    assert module.torch_rng_sha256() == audit["rng_restored_to_canonical_post_head_sha256"]
    assert audit["rng_after_candidate_head_sha256"] != audit[
        "rng_restored_to_canonical_post_head_sha256"
    ]


def test_loss_trajectory_requires_exact_epoch_partition() -> None:
    module = load_module()
    losses = [float(value) for value in range(12)]

    assert module.loss_trajectory(
        losses, train_rows=8, epochs=3, batch_size=2
    ) == [1.5, 5.5, 9.5]
    with pytest.raises(ValueError, match="observer count mismatch"):
        module.loss_trajectory(losses[:-1], train_rows=8, epochs=3, batch_size=2)


def synthetic_cache(module, path: Path) -> tuple[dict, dict]:
    train = {
        "row_id": ["train-a", "train-b"],
        "class_label": [1, 1],
        "class_name": ["one", "one"],
        "component_id": ["component-train", "component-train"],
        "decoded_pixel_sha256": ["pixel-train-a", "pixel-train-b"],
        "fold": [2, 2],
        "base_features": torch.zeros((2, 768), dtype=torch.float16),
        "view_features": torch.zeros((2, 8, 768), dtype=torch.float16),
    }
    oof = {
        "row_id": ["oof-a"],
        "class_label": [1],
        "class_name": ["one"],
        "component_id": ["component-oof"],
        "decoded_pixel_sha256": ["pixel-oof"],
        "fold": [1],
        "base_features": torch.zeros((1, 768), dtype=torch.float16),
    }
    provenance = {
        "corpus_validation": {
            "source_rows": 10039,
            "retained_rows": 9990,
            "quarantined_rows": 49,
            "conflicting_rgb_hashes": 22,
            "components_after_quarantine": 300,
        },
        "backbone": {"backbone": "synthetic-b14"},
        "safe_view_operations": list(module.v9.SAFE_VIEW_OPERATIONS),
        "forbidden_view_operations": sorted(module.v9.FORBIDDEN_VIEW_OPERATIONS),
        "ssl_outer_fold_row_exposure": 0,
        "learned_statistics_outer_fold_exposure": 0,
        "labels_used_by_ssl": False,
        "oof_view_features_persisted": False,
    }
    payload = {
        "schema_version": "autoresearch-self-supervised-v9.fold-cache",
        "fold": 1,
        "views": 8,
        "view_seed": module.VIEW_SEED,
        "image_size": module.IMAGE_SIZE,
        "max_rows_per_class": None,
        "train": train,
        "oof": oof,
        "provenance": provenance,
    }
    torch.save(payload, path)
    sidecar = {
        "cache_sha256": module.v9.sha256_file(path),
        "fold": 1,
        "views": 8,
        "train_row_ids_sha256": module.v9.sha256_json(train["row_id"]),
        "oof_row_ids_sha256": module.v9.sha256_json(oof["row_id"]),
        "ssl_outer_fold_row_exposure": 0,
        "learned_statistics_outer_fold_exposure": 0,
    }
    path.with_suffix(path.suffix + ".prov.json").write_text(
        json.dumps(sidecar), encoding="utf-8"
    )
    return payload, provenance


def test_synthetic_b14_cache_enforces_shape_dtype_order_and_no_oof_views(
    tmp_path: Path,
) -> None:
    module = load_module()
    path = tmp_path / "fold-01-views08.pt"
    canonical, provenance = synthetic_cache(module, path)

    loaded, audit = module.validate_b14_cache(
        path,
        fold=1,
        canonical=canonical,
        backbone_provenance=provenance["backbone"],
    )

    assert loaded["oof"].get("view_features") is None
    assert audit["train_shape"] == [2, 768]
    assert audit["view_shape"] == [2, 8, 768]
    assert audit["oof_shape"] == [1, 768]
    assert audit["dtype"] == "torch.float16"
    assert audit["final_test_read"] is False


def make_evaluation_fixture(module):
    records = []
    for seed in module.EXPECTED_SEEDS:
        for index in range(module.EXPECTED_UNIQUE_OOF_ROWS):
            label = index % 7
            wrong = (label + 1) % 7
            c1_top1 = label if index % 4 else wrong
            records.append(
                {
                    "row_id": f"row-{index}",
                    "provenance_component": f"component-{index % 5}",
                    "decoded_pixel_sha256": f"pixel-{index}",
                    "label": label,
                    "class_name": str(label),
                    "outer_fold": 1 + index % 5,
                    "seed": seed,
                    "b0_topk": [wrong, (wrong + 1) % 7, (wrong + 2) % 7],
                    "c1_topk": [c1_top1, label, (label + 2) % 7],
                    "candidate_topk": [label, (label + 1) % 7, (label + 2) % 7],
                    "train_support": 16,
                    "support_bin_three": "n_y_9_to_31",
                    "support_bin_binary": "n_y_gt_8",
                }
            )
    diagnostics = [
        {
            "fold": fold,
            "seed": seed,
            "checkpoint": {"sha256": f"sha-{fold}-{seed}"},
            "episode_plan_matches_c1": True,
            "ssl": {
                "plan_matches_c1": True,
                "epoch_loss_mean": [2.0, 1.0],
                "end_epoch_loss_below_start": True,
            },
            "candidate_final_state_differs_from_initial": True,
            "embedding_movement_mean_absolute": 0.1,
            "finite_gradients": True,
            "finite_weights": True,
            "candidate_to_c1_effective_rank_ratio": 1.0,
        }
        for fold in module.EXPECTED_FOLDS
        for seed in module.EXPECTED_SEEDS
    ]
    caches = [
        {
            "train_row_ids_sha256": f"train-{fold}",
            "canonical_train_row_ids_sha256": f"train-{fold}",
            "oof_row_ids_sha256": f"oof-{fold}",
            "canonical_oof_row_ids_sha256": f"oof-{fold}",
            "view_plan_sha256": f"view-{fold}",
            "byte_identical_readback": True,
        }
        for fold in module.EXPECTED_FOLDS
    ]
    control = {"exact_topk_match_count": module.EXPECTED_PAIRED_ROWS}
    return records, diagnostics, caches, control


def test_evaluator_reaches_strong_tier_only_after_all_nonreplay_gates() -> None:
    module = load_module()
    records, diagnostics, caches, control = make_evaluation_fixture(module)

    result = module.evaluate_candidate(
        records, diagnostics, caches, control_audit=control
    )

    assert result["integrity_pass"] is True
    assert result["engagement_pass"] is True
    assert result["b0_mission_anchor_pass"] is True
    assert result["c1_noninferiority_pass"] is True
    assert result["provisional_verdict_before_replay"] == "supported_strong"
    assert result["replay_pending"] is True
    assert result["promotion_eligible"] is False


def test_replay_normalization_removes_only_path_dependent_fields() -> None:
    module = load_module()
    left = {
        "diagnostics_sha256": "left",
        "artifact_paths": {"summary": "left"},
        "diagnostics": [{"checkpoint": {"path": "left", "sha256": "same"}}],
        "score": 0.1,
    }
    right = {
        "diagnostics_sha256": "right",
        "artifact_paths": {"summary": "right"},
        "diagnostics": [{"checkpoint": {"path": "right", "sha256": "same"}}],
        "score": 0.1,
    }

    assert module.normalized_replay_sha256(left) == module.normalized_replay_sha256(right)
    right["score"] = 0.2
    assert module.normalized_replay_sha256(left) != module.normalized_replay_sha256(right)


@pytest.mark.skipif(sys.platform != "linux", reason="historical Linux ext4 workspace audit")
def test_workspace_cache_target_is_real_ext4() -> None:
    module = load_module()

    audit = module.assert_ext4_workspace_path(module.DEFAULT_B14_CACHE_DIR)

    assert audit["filesystem"] == "ext4"
    assert audit["resolved_path"].startswith(str(module.ROOT.resolve()))


def test_candidate_checkpoint_serialization_is_byte_deterministic(
    tmp_path: Path,
) -> None:
    module = load_module()
    model, initialization = module.aligned_candidate_initialization(42)
    state_sha256 = module.v9.state_dict_sha256(model)

    first = module.atomic_save_checkpoint(
        tmp_path / "first.pt",
        fold=1,
        seed=42,
        model=model,
        initial_state_sha256=initialization["candidate_initial_state_sha256"],
        final_state_sha256=state_sha256,
    )
    second = module.atomic_save_checkpoint(
        tmp_path / "second.pt",
        fold=1,
        seed=42,
        model=model,
        initial_state_sha256=initialization["candidate_initial_state_sha256"],
        final_state_sha256=state_sha256,
    )

    assert first["sha256"] == second["sha256"]
    assert first["bytes"] == second["bytes"]

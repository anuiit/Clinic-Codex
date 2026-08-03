from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "autoresearch_self_supervised_v9.py"
SPEC = importlib.util.spec_from_file_location("autoresearch_self_supervised_v9", SCRIPT)
assert SPEC is not None
assert SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def _row(
    row_id: str,
    *,
    fold: int,
    class_label: int,
    component_id: str,
    decoded_pixel_sha256: str,
    class_name: str = "class",
) -> dict[str, object]:
    return {
        "row_id": row_id,
        "image_path": f"{row_id}.png",
        "class_label": class_label,
        "class_name": class_name,
        "source_family": f"family-{class_label}",
        "decoded_pixel_sha256": decoded_pixel_sha256,
        "fold": fold,
        "component_id": component_id,
    }


def _synthetic_cache(train_rows: int = 2, oof_rows: int = 1) -> dict[str, object]:
    train_payload = {
        "row_id": [f"train-{index}" for index in range(train_rows)],
        "class_label": [0 for _ in range(train_rows)],
        "class_name": ["class-0" for _ in range(train_rows)],
        "component_id": [f"component-train-{index}" for index in range(train_rows)],
        "decoded_pixel_sha256": [f"train-hash-{index}" for index in range(train_rows)],
        "fold": [2 for _ in range(train_rows)],
        "base_features": torch.zeros(train_rows, 384, dtype=torch.float16),
        "view_features": torch.zeros(train_rows, 2, 384, dtype=torch.float16),
    }
    oof_payload = {
        "row_id": [f"oof-{index}" for index in range(oof_rows)],
        "class_label": [0 for _ in range(oof_rows)],
        "class_name": ["class-0" for _ in range(oof_rows)],
        "component_id": [f"component-oof-{index}" for index in range(oof_rows)],
        "decoded_pixel_sha256": [f"oof-hash-{index}" for index in range(oof_rows)],
        "fold": [1 for _ in range(oof_rows)],
        "base_features": torch.zeros(oof_rows, 384, dtype=torch.float16),
    }
    return {"fold": 1, "train": train_payload, "oof": oof_payload}


def test_safe_ssl_view_is_deterministic_and_uses_only_safe_transforms() -> None:
    base = Image.new("RGB", (24, 16), (128, 64, 32))

    first = module.safe_ssl_view(base, "row-1", 0, 7)
    second = module.safe_ssl_view(base, "row-1", 0, 7)
    different = module.safe_ssl_view(base, "row-1", 1, 7)

    assert first.size == (24, 16)
    assert first.tobytes() == second.tobytes()
    assert first.tobytes() != different.tobytes()
    assert module.FORBIDDEN_VIEW_OPERATIONS.isdisjoint(module.SAFE_VIEW_OPERATIONS)
    assert not set(module.FORBIDDEN_VIEW_OPERATIONS) & set(module.SAFE_VIEW_OPERATIONS)


def test_precompute_fold_rejects_cross_fold_hash_overlap(tmp_path: Path) -> None:
    rows = [
        _row("train", fold=2, class_label=0, component_id="component-a", decoded_pixel_sha256="shared-hash"),
        _row("oof", fold=1, class_label=0, component_id="component-a", decoded_pixel_sha256="shared-hash"),
    ]

    def fake_extract_features(*args, **kwargs):
        count = len(args[1])
        base = torch.zeros(count, 384, dtype=torch.float16)
        views = torch.zeros(count, 2, 384, dtype=torch.float16)
        return base, views if kwargs.get("views", 0) else None

    with pytest.raises(ValueError, match="leakage"):
        module.precompute_fold(
            rows,
            {"schema_version": "test"},
            fold=1,
            views=2,
            view_seed=13,
            image_size=32,
            batch_size=2,
            num_workers=0,
            device=torch.device("cpu"),
            backbone=object(),
            backbone_provenance={"schema_version": "test"},
            cache_dir=tmp_path,
            max_rows_per_class=None,
            force=True,
        )


def test_precompute_fold_writes_isolated_cache_provenance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        _row("train-0", fold=2, class_label=0, component_id="component-train-0", decoded_pixel_sha256="train-hash-0"),
        _row("train-1", fold=2, class_label=1, component_id="component-train-1", decoded_pixel_sha256="train-hash-1"),
        _row("oof-0", fold=1, class_label=0, component_id="component-oof-0", decoded_pixel_sha256="oof-hash-0"),
    ]

    def fake_extract_features(backbone, selected_rows, *, views, **kwargs):
        count = len(selected_rows)
        base = torch.arange(count * 384, dtype=torch.float32).reshape(count, 384).to(torch.float16)
        if views:
            return base, torch.ones(count, views, 384, dtype=torch.float16)
        return base, None

    monkeypatch.setattr(module, "extract_features", fake_extract_features)

    cache_path = module.precompute_fold(
        rows,
        {"schema_version": "test"},
        fold=1,
        views=2,
        view_seed=13,
        image_size=32,
        batch_size=2,
        num_workers=0,
        device=torch.device("cpu"),
        backbone=object(),
        backbone_provenance={"schema_version": "test"},
        cache_dir=tmp_path,
        max_rows_per_class=None,
        force=True,
    )

    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    assert payload["schema_version"] == "autoresearch-self-supervised-v9.fold-cache"
    assert payload["provenance"]["ssl_outer_fold_row_exposure"] == 0
    assert payload["provenance"]["safe_view_operations"] == list(module.SAFE_VIEW_OPERATIONS)
    assert payload["provenance"]["forbidden_view_operations"] == sorted(module.FORBIDDEN_VIEW_OPERATIONS)

    provenance = json.loads((cache_path.with_suffix(cache_path.suffix + ".prov.json")).read_text(encoding="utf-8"))
    assert provenance["ssl_outer_fold_row_exposure"] == 0
    assert provenance["train_rows"] == 2
    assert provenance["oof_rows"] == 1
    assert provenance["train_row_ids_sha256"]
    assert provenance["oof_row_ids_sha256"]


def test_episode_plan_maps_sparse_global_labels_to_local_class_indices() -> None:
    labels = torch.tensor([10] * 8 + [42] * 8 + [99] * 8, dtype=torch.long)

    plan, plan_sha256 = module.build_episode_plan(
        labels,
        n_way=3,
        k_shot=2,
        q_queries=2,
        epochs=1,
        episodes_per_epoch=1,
        seed=7,
    )

    support_indices, support_labels, query_indices, query_labels = plan[0][0]
    assert set(support_labels) == {0, 1, 2}
    assert set(query_labels) == {0, 1, 2}
    assert max(support_labels + query_labels) < 3
    global_labels_per_local: dict[int, set[int]] = {}
    for index, local_label in zip(
        support_indices + query_indices,
        support_labels + query_labels,
    ):
        global_labels_per_local.setdefault(local_label, set()).add(int(labels[index]))
    assert all(len(values) == 1 for values in global_labels_per_local.values())
    assert len(plan_sha256) == 64


def test_run_fold_seed_records_checkpoint_hashes_and_episode_provenance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache = _synthetic_cache()
    checkpoint_dir = tmp_path / "checkpoints"

    monkeypatch.setattr(module, "configure_determinism", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        module,
        "pretrain_vicreg",
        lambda *args, **kwargs: {
            "epochs": kwargs["epochs"],
            "samples_per_epoch": len(cache["train"]["row_id"]),
            "plan_sha256": "ssl-plan-sha",
            "last_loss": 0.1,
            "last_invariance": 0.01,
            "last_variance": 0.02,
            "last_covariance": 0.03,
            "minimum_batch_std": 0.5,
            "effective_rank": 4.0,
        },
    )
    monkeypatch.setattr(module, "build_episode_plan", lambda *args, **kwargs: ([[([0], [0], [0], [0])]], "episode-plan-sha"))
    monkeypatch.setattr(
        module,
        "train_supervised",
        lambda *args, **kwargs: {"epochs": 1, "episodes_per_epoch": 1, "last_loss": 0.1, "last_accuracy": 1.0},
    )
    monkeypatch.setattr(
        module,
        "predict_arm",
        lambda model, train, oof, device: ([[0, 1, 2] for _ in range(len(oof["row_id"]))], 1.0),
    )

    records, diagnostics = module.run_fold_seed(
        cache,
        seed=17,
        ssl_epochs=1,
        ssl_batch_size=2,
        supervised_epochs=1,
        episodes_per_epoch=1,
        n_way=1,
        k_shot=1,
        q_queries=1,
        device=torch.device("cpu"),
        checkpoint_dir=checkpoint_dir,
    )

    baseline_path = checkpoint_dir / "fold-01-seed-17-B0.pt"
    candidate_path = checkpoint_dir / "fold-01-seed-17-C1.pt"

    assert len(records) == len(cache["oof"]["row_id"])
    assert records[0]["episode_plan_sha256"] == "episode-plan-sha"
    assert diagnostics["checkpoints"]["baseline"]["sha256"] == module.sha256_file(baseline_path)
    assert diagnostics["checkpoints"]["candidate"]["sha256"] == module.sha256_file(candidate_path)


def test_paired_component_bootstrap_preserves_resampled_multiplicity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = [
        {"provenance_component": component, "seed": 17}
        for component in ("component-a", "component-b")
    ]
    sampled_components: list[list[str]] = []

    def capture_sample(sample: list[dict[str, object]]) -> dict[str, float]:
        sampled_components.append([str(row["provenance_component"]) for row in sample])
        return {"delta_top1": 0.0, "delta_macro_top1": 0.0}

    monkeypatch.setattr(module, "accuracy_metrics", capture_sample)

    result = module.paired_component_bootstrap(records, replicates=100, seed=0)

    assert result["replicates"] == 100
    assert len(sampled_components) == 100
    assert all(len(sample) == 2 for sample in sampled_components)
    assert any(len(set(sample)) == 1 for sample in sampled_components)


def test_aggregate_results_exposes_actual_gate_values_and_pass_map() -> None:
    records = []
    for seed in (17, 42, 73):
        for component in ("component-a", "component-b"):
            records.append(
                {
                    "row_id": f"{component}-{seed}",
                    "provenance_component": component,
                    "decoded_pixel_sha256": f"{component}-hash",
                    "label": 0,
                    "class_name": "class-0",
                    "outer_fold": 1,
                    "seed": seed,
                    "recipe": "B0-vs-C1-vicreg-projection",
                    "baseline_topk": [1, 2, 3],
                    "candidate_topk": [0, 2, 3],
                    "episode_plan_sha256": "episode-plan-sha",
                }
            )

    diagnostics = [
        {
            "fold": 1,
            "seed": 17,
            "initial_state_sha256": "a",
            "baseline_candidate_initial_state_equal": True,
            "episode_plan_sha256": "episode-plan-sha",
            "baseline_candidate_supervised_episode_trace_equal": True,
            "baseline_candidate_supervised_budget_equal": True,
            "ssl": {"epochs": 1},
            "baseline_training": {"epochs": 1},
            "candidate_training": {"epochs": 1},
            "baseline_effective_rank": 1.0,
            "candidate_effective_rank": 1.0,
            "effective_rank_ratio": 0.95,
            "checkpoints": {"baseline": {"sha256": "b"}, "candidate": {"sha256": "c"}},
        },
        {
            "fold": 2,
            "seed": 42,
            "initial_state_sha256": "a",
            "baseline_candidate_initial_state_equal": True,
            "episode_plan_sha256": "episode-plan-sha",
            "baseline_candidate_supervised_episode_trace_equal": True,
            "baseline_candidate_supervised_budget_equal": True,
            "ssl": {"epochs": 1},
            "baseline_training": {"epochs": 1},
            "candidate_training": {"epochs": 1},
            "baseline_effective_rank": 1.0,
            "candidate_effective_rank": 1.0,
            "effective_rank_ratio": 0.92,
            "checkpoints": {"baseline": {"sha256": "b"}, "candidate": {"sha256": "c"}},
        },
    ]

    cache_validations = [
        {
            "v8_conflicting_rgb_hashes_quarantined": True,
            "provenance_component_overlap_across_folds": 0,
            "decoded_pixel_hash_overlap_across_folds": 0,
            "ssl_outer_fold_row_exposure": 0,
            "learned_statistics_outer_fold_exposure": 0,
        }
    ]
    result = module.aggregate_results(
        records,
        diagnostics,
        cache_validations,
        bootstrap_replicates=100,
        expected_seed_count=3,
        expected_fold_count=1,
        runtime_unchanged=True,
    )

    assert result["pass"] is True
    assert result["gate_passes"]["provenance_component_overlap_across_folds"] is True
    assert result["gate_passes"]["decoded_pixel_hash_overlap_across_folds"] is True
    assert result["gate_passes"]["ssl_outer_fold_row_exposure"] is True
    assert result["gate_passes"]["learned_statistics_outer_fold_exposure"] is True
    assert result["gate_passes"]["nan_or_collapse_detected"] is True
    assert result["gate_passes"]["final_test_read"] is True
    assert result["gate_passes"]["automatic_promotion"] is True
    assert result["gates"]["provenance_component_overlap_across_folds"] == 0
    assert result["gates"]["decoded_pixel_hash_overlap_across_folds"] == 0
    assert result["gates"]["ssl_outer_fold_row_exposure"] == 0
    assert result["gates"]["learned_statistics_outer_fold_exposure"] == 0
    assert result["gates"]["nan_or_collapse_detected"] is False
    assert result["gates"]["final_test_read"] is False
    assert result["gates"]["automatic_promotion"] is False
    assert result["positive_seed_count"] == 3
    assert result["minimum_effective_rank_ratio"] >= 0.9
    assert result["overall_metrics"]["delta_top1"] > 0.0


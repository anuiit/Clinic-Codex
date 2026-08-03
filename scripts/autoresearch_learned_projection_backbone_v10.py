#!/usr/bin/env python3
"""Compare learned 128-d readouts on frozen DINOv2-S/14 and B/14 features."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import autoresearch_backbone_screen_v10 as backbone_screen  # noqa: E402
import autoresearch_self_supervised_v9 as v9  # noqa: E402


V9_RUN = ROOT / ".omc/autoresearch/elements-baseline-replacement/runs/20260803-self-supervised-v9"
V10_RUN = ROOT / ".omc/autoresearch/elements-baseline-replacement/runs/20260803-self-supervised-v10"
DEFAULT_V9_SUMMARY = V9_RUN / "iteration-0001/results/paired_seed_summary.json"
DEFAULT_V9_PREDICTIONS = V9_RUN / "iteration-0001/results/prediction_rows.jsonl"
DEFAULT_V9_RUNNER = ROOT / "scripts/autoresearch_self_supervised_v9.py"
DEFAULT_V9_CACHE_DIR = V9_RUN / "iteration-0001/caches"
DEFAULT_B14_MANIFEST = V10_RUN / "inputs/dinov2-vitb14-local-manifest.json"
DEFAULT_B14_CACHE = V10_RUN / "iteration-0002/cache/dinov2-vitb14-base-features.pt"
DEFAULT_SPEC = V10_RUN / "specs/iteration-0003.json"
DEFAULT_ADDENDUM = V10_RUN / "specs/iteration-0003-interpretation-addendum.json"
DEFAULT_EVALUATOR = V10_RUN / "evaluator-iteration-0003.json"
DEFAULT_OUTPUT_DIR = V10_RUN / "iteration-0003/results"
DEFAULT_RUNTIME_PROJECTION = ROOT / "backend/codex_model/weights/projection.pt"
DEFAULT_RUNTIME_PROTOTYPES = ROOT / "backend/codex_model/weights/prototypes.pt"

EXPECTED_V9_SUMMARY_SHA256 = "d6094bc9887748ba9bc89b3f6d691483b803be806cf0623e450450132387a22d"
EXPECTED_V9_PREDICTIONS_SHA256 = "37a15fb66ef883d15df5db8e1cadb152f9460876b60ace832c434375a3cdf34f"
EXPECTED_V9_RUNNER_SHA256 = "e03e1e29340a1a165ab2f82001236f5aaae4450afcee862dff8cb27ef26c48a9"
EXPECTED_V9_DATA_SHA256 = "45745373a1feac6731acb50ec87de699f2cf613314c6285279a1f551a893b4ac"
EXPECTED_FOLDS_SHA256 = "18b4222d265631051dde5df809d2eff4a088a0ef90b420940eb94a47a437805c"
EXPECTED_B14_CACHE_SHA256 = "fc9ab8bf5773ccd53e3ec74f8a8028f865848a0df2a2c16b3ee60e93bcea8c41"
EXPECTED_FOLDS = [1, 2, 3, 4, 5]
EXPECTED_SEEDS = [17, 42, 73]
EXPECTED_PAIRED_ROWS = 1959
EXPECTED_B14_ROWS = 9990
EXPECTED_B14_DIM = 768


class FixedHiddenProjection(nn.Module):
    """ProjectionHead semantics with a fixed 384-wide hidden layer."""

    def __init__(self, input_dim: int, hidden_dim: int = 384, embedding_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(features), p=2, dim=-1)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def require_sha256(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing: {path}")
    actual = v9.sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: expected {expected}, found {actual}")
    return actual


def validate_iteration_contract(spec_path: Path, addendum_path: Path, evaluator_path: Path) -> dict[str, Any]:
    spec = read_json(spec_path)
    addendum = read_json(addendum_path)
    evaluator = read_json(evaluator_path)
    if spec.get("iteration") != 3 or evaluator.get("iteration") != 3:
        raise ValueError("iteration-3 spec/evaluator mismatch")
    if "preregistered_before_any_iteration_0002_or_0003_B14_prediction" != spec.get("status"):
        raise ValueError("iteration-3 spec was not preregistered before B/14 predictions")
    if spec.get("single_factor", {}).get("name") != "frozen_backbone":
        raise ValueError("iteration-3 does not isolate the frozen backbone")
    gates = spec.get("evaluation", {}).get("efficacy_gates", {})
    expected_gates = {
        "positive_seed_count": 2,
        "delta_top1_component_bootstrap_lower_95_gt": 0.0,
        "delta_macro_top1_component_bootstrap_lower_95_gte": 0.0,
        "B14_effective_rank_ratio_gte": 0.9,
    }
    if gates != expected_gates:
        raise ValueError("iteration-3 efficacy gates changed after preregistration")
    if addendum.get("locked_before_any_iteration_0003_B14_prediction") is not True:
        raise ValueError("iteration-3 interpretation addendum is not locked")
    if addendum.get("locked_gate_policy", {}).get("preserve_original_iteration_0003_efficacy_gates") is not True:
        raise ValueError("iteration-3 addendum does not preserve original gates")
    return {
        "spec": spec,
        "addendum": addendum,
        "evaluator": evaluator,
        "spec_sha256": v9.sha256_file(spec_path),
        "addendum_sha256": v9.sha256_file(addendum_path),
        "evaluator_sha256": v9.sha256_file(evaluator_path),
    }


def _diagnostics_by_key(summary: dict[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    diagnostics = {
        (int(item["fold"]), int(item["seed"])): item
        for item in summary.get("diagnostics", [])
    }
    expected = {(fold, seed) for fold in EXPECTED_FOLDS for seed in EXPECTED_SEEDS}
    if set(diagnostics) != expected:
        raise ValueError("persisted B0 diagnostics do not cover the exact fold/seed grid")
    return diagnostics


def validate_persisted_b0(
    summary_path: Path,
    predictions_path: Path,
    runner_path: Path,
    cache_dir: Path,
) -> dict[str, Any]:
    require_sha256(summary_path, EXPECTED_V9_SUMMARY_SHA256, "v9 B0 summary")
    require_sha256(predictions_path, EXPECTED_V9_PREDICTIONS_SHA256, "v9 B0 predictions")
    require_sha256(runner_path, EXPECTED_V9_RUNNER_SHA256, "v9 runner")
    summary = read_json(summary_path)
    records = read_jsonl(predictions_path)
    if summary.get("code_sha256") != EXPECTED_V9_RUNNER_SHA256:
        raise ValueError("v9 summary runner provenance mismatch")
    if summary.get("data_sha256") != EXPECTED_V9_DATA_SHA256:
        raise ValueError("v9 summary data provenance mismatch")
    if summary.get("folds_sha256") != EXPECTED_FOLDS_SHA256:
        raise ValueError("v9 fold assignment hash mismatch")
    if summary.get("paired_predictions_path") != predictions_path.name:
        raise ValueError("v9 paired prediction filename mismatch")
    if len(records) != EXPECTED_PAIRED_ROWS:
        raise ValueError(f"expected {EXPECTED_PAIRED_ROWS} persisted B0 rows, found {len(records)}")
    diagnostics = _diagnostics_by_key(summary)
    seen: set[tuple[int, int, str]] = set()
    for row in records:
        key = (int(row["outer_fold"]), int(row["seed"]), str(row["row_id"]))
        if key in seen:
            raise ValueError(f"duplicate persisted B0 prediction: {key}")
        seen.add(key)
        fold_seed = key[:2]
        diagnostic = diagnostics.get(fold_seed)
        if diagnostic is None:
            raise ValueError(f"persisted B0 row outside fold/seed grid: {key}")
        if row.get("code_sha256") != EXPECTED_V9_RUNNER_SHA256:
            raise ValueError(f"persisted B0 row runner mismatch: {key}")
        if row.get("folds_sha256") != EXPECTED_FOLDS_SHA256:
            raise ValueError(f"persisted B0 row fold hash mismatch: {key}")
        if row.get("episode_plan_sha256") != diagnostic.get("episode_plan_sha256"):
            raise ValueError(f"persisted B0 episode plan mismatch: {key}")
        expected_checkpoint = diagnostic.get("checkpoints", {}).get("baseline", {}).get("sha256")
        if row.get("baseline_checkpoint_sha256") != expected_checkpoint:
            raise ValueError(f"persisted B0 checkpoint provenance mismatch: {key}")

    cache_hashes = {str(key): str(value) for key, value in summary.get("cache_sha256", {}).items()}
    for fold in EXPECTED_FOLDS:
        path = v9.cache_path(cache_dir, fold, 8, None)
        expected_hash = cache_hashes.get(str(fold))
        if expected_hash is None:
            raise ValueError(f"v9 summary lacks fold-{fold} cache provenance")
        require_sha256(path, expected_hash, f"v9 fold-{fold} cache")
    for key, diagnostic in diagnostics.items():
        checkpoint = diagnostic.get("checkpoints", {}).get("baseline", {})
        checkpoint_path = ROOT / str(checkpoint.get("path", ""))
        require_sha256(checkpoint_path, str(checkpoint.get("sha256")), f"v9 B0 checkpoint {key}")
        training = diagnostic.get("baseline_training", {})
        if training.get("epochs") != 30 or training.get("episodes_per_epoch") != 100:
            raise ValueError(f"v9 B0 supervised budget mismatch: {key}")
    if sorted({int(row["outer_fold"]) for row in records}) != EXPECTED_FOLDS:
        raise ValueError("persisted B0 fold coverage mismatch")
    if sorted({int(row["seed"]) for row in records}) != EXPECTED_SEEDS:
        raise ValueError("persisted B0 seed coverage mismatch")
    return {
        "summary": summary,
        "records": records,
        "diagnostics": diagnostics,
        "summary_sha256": EXPECTED_V9_SUMMARY_SHA256,
        "predictions_sha256": EXPECTED_V9_PREDICTIONS_SHA256,
        "runner_sha256": EXPECTED_V9_RUNNER_SHA256,
        "cache_sha256": cache_hashes,
        "exact": True,
    }


def build_b14_projection(seed: int, expected_s14_initial_sha256: str) -> tuple[FixedHiddenProjection, dict[str, Any]]:
    v9.configure_determinism(seed)
    reference = v9.ProjectionHead(input_dim=384, embedding_dim=128)
    reference_hash = v9.state_dict_sha256(reference)
    if reference_hash != expected_s14_initial_sha256:
        raise ValueError(
            f"reconstructed S/14 B0 initialization mismatch for seed {seed}: "
            f"expected {expected_s14_initial_sha256}, found {reference_hash}"
        )
    v9.configure_determinism(seed)
    candidate = FixedHiddenProjection(input_dim=768, hidden_dim=384, embedding_dim=128)
    candidate.net[3].load_state_dict(reference.net[3].state_dict())
    reference_output_hash = v9.state_dict_sha256(reference.net[3])
    candidate_output_hash = v9.state_dict_sha256(candidate.net[3])
    if reference_output_hash != candidate_output_hash:
        raise AssertionError("common 384-to-128 output initialization mismatch")
    return candidate, {
        "S14_reconstructed_initial_state_sha256": reference_hash,
        "B14_initial_state_sha256": v9.state_dict_sha256(candidate),
        "common_output_initialization_sha256": candidate_output_hash,
        "common_output_initialization_matches_reconstructed_v9_B0": True,
        "architecture": "768-384-128",
    }


def _fold_indices(cache: dict[str, Any], fold: int) -> tuple[list[int], list[int]]:
    train_indices = [index for index, value in enumerate(cache["fold"]) if int(value) != fold]
    train_labels = {int(cache["class_label"][index]) for index in train_indices}
    oof_indices = [
        index
        for index, value in enumerate(cache["fold"])
        if int(value) == fold and int(cache["class_label"][index]) in train_labels
    ]
    return train_indices, oof_indices


def _class_component_counts(cache: dict[str, Any], train_indices: Sequence[int]) -> dict[int, int]:
    components: dict[int, set[str]] = defaultdict(set)
    for index in train_indices:
        components[int(cache["class_label"][index])].add(str(cache["component_id"][index]))
    return {label: len(values) for label, values in components.items()}


def train_or_resume_fold_seed(
    cache: dict[str, Any],
    *,
    fold: int,
    seed: int,
    source_diagnostic: dict[str, Any],
    code_sha256: str,
    checkpoint_dir: Path,
    prediction_device: torch.device,
    supervised_device: torch.device,
) -> tuple[dict[str, list[int]], dict[str, Any]]:
    train_indices, oof_indices = _fold_indices(cache, fold)
    train_labels = torch.tensor([cache["class_label"][index] for index in train_indices], dtype=torch.long)
    episode_plan, episode_hash = v9.build_episode_plan(
        train_labels,
        n_way=20,
        k_shot=3,
        q_queries=5,
        epochs=30,
        episodes_per_epoch=100,
        seed=v9.stable_seed("supervised-episodes", fold, seed),
    )
    expected_episode_hash = str(source_diagnostic["episode_plan_sha256"])
    if episode_hash != expected_episode_hash:
        raise ValueError(f"B/14 episode plan does not replay B0 for fold={fold}, seed={seed}")
    model, init_diagnostics = build_b14_projection(seed, str(source_diagnostic["initial_state_sha256"]))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"fold-{fold:02d}-seed-{seed}-B14.pt"
    resumed = False
    if checkpoint_path.exists():
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        expected_metadata = {
            "schema_version": "autoresearch-self-supervised-v10.B14-projection-checkpoint",
            "fold": fold,
            "seed": seed,
            "runner_sha256": code_sha256,
            "episode_plan_sha256": episode_hash,
            "B14_feature_cache_sha256": EXPECTED_B14_CACHE_SHA256,
            "architecture": "768-384-128",
        }
        for key, value in expected_metadata.items():
            if payload.get(key) != value:
                raise ValueError(f"existing B/14 checkpoint metadata mismatch: {checkpoint_path} ({key})")
        model.load_state_dict(payload["model_state_dict"])
        training = dict(payload["training"])
        resumed = True
    else:
        model = model.to(supervised_device)
        training = v9.train_supervised(
            model,
            cache["base_features"][train_indices],
            episode_plan,
            learning_rate=1e-3,
            weight_decay=1e-4,
            temperature=0.1,
            warmup_epochs=5,
            rng_seed=v9.stable_seed("supervised-rng", fold, seed),
            device=supervised_device,
        )
        payload = {
            "schema_version": "autoresearch-self-supervised-v10.B14-projection-checkpoint",
            "fold": fold,
            "seed": seed,
            "arm": "B14_projection",
            "architecture": "768-384-128",
            "runner_sha256": code_sha256,
            "episode_plan_sha256": episode_hash,
            "B14_feature_cache_sha256": EXPECTED_B14_CACHE_SHA256,
            "initialization": init_diagnostics,
            "training": training,
            "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        }
        temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
        torch.save(payload, temporary_path)
        temporary_path.replace(checkpoint_path)
    if training.get("epochs") != 30 or training.get("episodes_per_epoch") != 100:
        raise ValueError("B/14 supervised training budget mismatch")
    model = model.to(prediction_device)
    train = {
        "base_features": cache["base_features"][train_indices],
        "class_label": [cache["class_label"][index] for index in train_indices],
    }
    oof = {"base_features": cache["base_features"][oof_indices]}
    topk, rank = v9.predict_arm(model, train, oof, device=prediction_device)
    predictions = {str(cache["row_id"][index]): topk[offset] for offset, index in enumerate(oof_indices)}
    baseline_rank = float(source_diagnostic["baseline_effective_rank"])
    diagnostics = {
        "fold": fold,
        "seed": seed,
        "architecture": "768-384-128",
        "episode_plan_sha256": episode_hash,
        "episode_plan_matches_persisted_B0": True,
        "supervised_budget_and_optimizer_match_persisted_B0": True,
        "initialization": init_diagnostics,
        "training": training,
        "S14_effective_rank": baseline_rank,
        "B14_effective_rank": float(rank),
        "B14_over_S14_effective_rank_ratio": float(rank / baseline_rank) if baseline_rank else 0.0,
        "checkpoint": {"path": str(checkpoint_path), "sha256": v9.sha256_file(checkpoint_path)},
        "resumed_from_verified_checkpoint": resumed,
        "prediction_device": str(prediction_device),
        "supervised_device": str(supervised_device),
        "nan_or_nonfinite_detected": not math.isfinite(float(rank)),
    }
    return predictions, diagnostics


def validate_b14_to_b0_metadata(
    cache: dict[str, Any],
    baseline_records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    row_index = {str(row_id): index for index, row_id in enumerate(cache["row_id"])}
    if len(row_index) != EXPECTED_B14_ROWS:
        raise ValueError("B/14 cache row IDs are not unique and complete")
    for row in baseline_records:
        row_id = str(row["row_id"])
        index = row_index.get(row_id)
        if index is None or int(cache["fold"][index]) != int(row["outer_fold"]):
            raise ValueError(f"B/14 cache fold/row mismatch: {row_id}")
        comparisons = (
            (int(row["label"]), int(cache["class_label"][index])),
            (str(row["class_name"]), str(cache["class_name"][index])),
            (str(row["provenance_component"]), str(cache["component_id"][index])),
            (str(row["decoded_pixel_sha256"]), str(cache["decoded_pixel_sha256"][index])),
        )
        if any(left != right for left, right in comparisons):
            raise ValueError(f"B/14 cache metadata mismatch: {row_id}")
    component_folds: dict[str, set[int]] = defaultdict(set)
    pixel_folds: dict[str, set[int]] = defaultdict(set)
    for component, pixel_hash, fold in zip(cache["component_id"], cache["decoded_pixel_sha256"], cache["fold"]):
        component_folds[str(component)].add(int(fold))
        pixel_folds[str(pixel_hash)].add(int(fold))
    component_overlap = sum(len(values) > 1 for values in component_folds.values())
    pixel_overlap = sum(len(values) > 1 for values in pixel_folds.values())
    if component_overlap or pixel_overlap:
        raise ValueError("B/14 cache violates fold isolation")
    return {
        "row_index": row_index,
        "provenance_component_overlap_across_folds": component_overlap,
        "decoded_pixel_hash_overlap_across_folds": pixel_overlap,
    }


def validate_fold_cache_row_replay(
    B14_cache: dict[str, Any],
    cache_dir: Path,
    expected_cache_hashes: dict[str, str],
) -> dict[str, Any]:
    """Prove the global B/14 cache replays each pinned v9 fold row order exactly."""
    reports: dict[str, Any] = {}
    for fold in EXPECTED_FOLDS:
        path = v9.cache_path(cache_dir, fold, 8, None)
        require_sha256(path, expected_cache_hashes[str(fold)], f"v9 fold-{fold} cache")
        payload = v9.load_fold_cache(
            path,
            expected_fold=fold,
            expected_views=8,
            expected_max_rows_per_class=None,
        )
        train_indices, oof_indices = _fold_indices(B14_cache, fold)
        expected_train_row_ids = [str(B14_cache["row_id"][index]) for index in train_indices]
        expected_oof_row_ids = [str(B14_cache["row_id"][index]) for index in oof_indices]
        actual_train_row_ids = [str(value) for value in payload["train"]["row_id"]]
        actual_oof_row_ids = [str(value) for value in payload["oof"]["row_id"]]
        if actual_train_row_ids != expected_train_row_ids:
            raise ValueError(f"B/14 train row order does not replay pinned v9 fold-{fold} cache")
        if actual_oof_row_ids != expected_oof_row_ids:
            raise ValueError(f"B/14 OOF row order does not replay pinned v9 fold-{fold} cache")
        for section, indices in (("train", train_indices), ("oof", oof_indices)):
            expected_labels = [int(B14_cache["class_label"][index]) for index in indices]
            expected_components = [str(B14_cache["component_id"][index]) for index in indices]
            expected_pixels = [str(B14_cache["decoded_pixel_sha256"][index]) for index in indices]
            if [int(value) for value in payload[section]["class_label"]] != expected_labels:
                raise ValueError(f"B/14 {section} labels do not replay pinned v9 fold-{fold} cache")
            if [str(value) for value in payload[section]["component_id"]] != expected_components:
                raise ValueError(f"B/14 {section} components do not replay pinned v9 fold-{fold} cache")
            if [str(value) for value in payload[section]["decoded_pixel_sha256"]] != expected_pixels:
                raise ValueError(f"B/14 {section} pixel hashes do not replay pinned v9 fold-{fold} cache")
        reports[str(fold)] = {
            "cache_sha256": expected_cache_hashes[str(fold)],
            "train_rows": len(actual_train_row_ids),
            "oof_rows": len(actual_oof_row_ids),
            "train_row_ids_sha256": v9.sha256_json(actual_train_row_ids),
            "oof_row_ids_sha256": v9.sha256_json(actual_oof_row_ids),
            "exact_order_and_metadata_match": True,
        }
        del payload
    return {"exact": True, "folds": reports}


def _renamed_metrics(records: Sequence[dict[str, Any]]) -> dict[str, float]:
    raw = v9.accuracy_metrics(records)
    return {
        "S14_projection_top1": raw["baseline_top1"],
        "S14_projection_top3": raw["baseline_top3"],
        "S14_projection_macro_top1": raw["baseline_macro_top1"],
        "B14_projection_top1": raw["candidate_top1"],
        "B14_projection_top3": raw["candidate_top3"],
        "B14_projection_macro_top1": raw["candidate_macro_top1"],
        "delta_top1_B14_projection_minus_S14_projection": raw["delta_top1"],
        "delta_top3_B14_projection_minus_S14_projection": raw["delta_top3"],
        "delta_macro_top1_B14_projection_minus_S14_projection": raw["delta_macro_top1"],
    }


def descriptive_breakdowns(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    folds = sorted({int(row["outer_fold"]) for row in records})
    per_fold = {
        str(fold): _renamed_metrics([row for row in records if int(row["outer_fold"]) == fold])
        for fold in folds
    }
    strata: dict[str, Any] = {}
    for stratum in ("single_train_component", "multiple_train_components"):
        selected = [row for row in records if row.get("coverage_stratum") == stratum]
        strata[stratum] = {
            "rows": len(selected),
            "unique_oof_rows": len({(int(row["outer_fold"]), str(row["row_id"])) for row in selected}),
            "labels": len({int(row["label"]) for row in selected}),
            "metrics": _renamed_metrics(selected) if selected else None,
        }
    return {"non_gating": True, "per_fold": per_fold, "coverage_strata": strata}


def evaluate_records(
    records: Sequence[dict[str, Any]],
    diagnostics: Sequence[dict[str, Any]],
    *,
    runtime_unchanged: bool,
    bootstrap_replicates: int,
    B0_provenance_exact: bool,
    B14_cache_valid: bool,
    fold_isolation: dict[str, Any],
) -> dict[str, Any]:
    folds = sorted({int(row["outer_fold"]) for row in records})
    seeds = sorted({int(row["seed"]) for row in records})
    overall = _renamed_metrics(records)
    seed_metrics = [
        {"seed": seed, **_renamed_metrics([row for row in records if int(row["seed"]) == seed])}
        for seed in seeds
    ]
    intervals = v9.paired_component_bootstrap(records, replicates=bootstrap_replicates, seed=20260803)
    positive_seed_count = sum(
        item["delta_top1_B14_projection_minus_S14_projection"] > 0.0 for item in seed_metrics
    )
    minimum_rank_ratio = min(
        float(item["B14_over_S14_effective_rank_ratio"]) for item in diagnostics
    )
    finite_values: list[float] = list(overall.values()) + [
        float(item[key])
        for item in diagnostics
        for key in ("S14_effective_rank", "B14_effective_rank", "B14_over_S14_effective_rank_ratio")
    ]
    gates: dict[str, Any] = {
        "paired_prediction_rows": len(records),
        "paired_seed_count": len(seeds),
        "outer_fold_count": len(folds),
        "B14_checkpoint_count": len(diagnostics),
        "B0_provenance_exact": B0_provenance_exact,
        "B14_feature_rows": EXPECTED_B14_ROWS,
        "B14_embedding_dim": EXPECTED_B14_DIM,
        "B14_projection_architecture": "768-384-128",
        "common_output_initialization_matches_reconstructed_v9_B0": all(
            item["initialization"]["common_output_initialization_matches_reconstructed_v9_B0"]
            for item in diagnostics
        ),
        "episode_plan_hashes_match_persisted_B0": all(
            item["episode_plan_matches_persisted_B0"] for item in diagnostics
        ),
        "supervised_budget_and_optimizer_match_persisted_B0": all(
            item["supervised_budget_and_optimizer_match_persisted_B0"] for item in diagnostics
        ),
        "B14_cache_valid": B14_cache_valid,
        "backbone_updates": 0,
        "provenance_component_overlap_across_folds": fold_isolation[
            "provenance_component_overlap_across_folds"
        ],
        "decoded_pixel_hash_overlap_across_folds": fold_isolation["decoded_pixel_hash_overlap_across_folds"],
        "nan_or_nonfinite_detected": not all(math.isfinite(value) for value in finite_values),
        "runtime_unchanged": runtime_unchanged,
        "final_test_read": False,
        "automatic_promotion": False,
        "positive_seed_count_gte": positive_seed_count,
        "delta_top1_component_bootstrap_lower_95_gt": intervals["delta_top1_95"][0],
        "delta_macro_top1_component_bootstrap_lower_95_gte": intervals["delta_macro_top1_95"][0],
        "minimum_B14_over_S14_effective_rank_ratio_gte": minimum_rank_ratio,
    }
    integrity_keys = (
        "paired_prediction_rows",
        "paired_seed_count",
        "outer_fold_count",
        "B14_checkpoint_count",
        "B0_provenance_exact",
        "B14_feature_rows",
        "B14_embedding_dim",
        "B14_projection_architecture",
        "common_output_initialization_matches_reconstructed_v9_B0",
        "episode_plan_hashes_match_persisted_B0",
        "supervised_budget_and_optimizer_match_persisted_B0",
        "B14_cache_valid",
        "backbone_updates",
        "provenance_component_overlap_across_folds",
        "decoded_pixel_hash_overlap_across_folds",
        "nan_or_nonfinite_detected",
        "runtime_unchanged",
        "final_test_read",
        "automatic_promotion",
    )
    gate_passes = {
        "paired_prediction_rows": gates["paired_prediction_rows"] == EXPECTED_PAIRED_ROWS,
        "paired_seed_count": gates["paired_seed_count"] == len(EXPECTED_SEEDS),
        "outer_fold_count": gates["outer_fold_count"] == len(EXPECTED_FOLDS),
        "B14_checkpoint_count": gates["B14_checkpoint_count"] == 15,
        "B0_provenance_exact": gates["B0_provenance_exact"] is True,
        "B14_feature_rows": gates["B14_feature_rows"] == EXPECTED_B14_ROWS,
        "B14_embedding_dim": gates["B14_embedding_dim"] == EXPECTED_B14_DIM,
        "B14_projection_architecture": gates["B14_projection_architecture"] == "768-384-128",
        "common_output_initialization_matches_reconstructed_v9_B0": gates[
            "common_output_initialization_matches_reconstructed_v9_B0"
        ] is True,
        "episode_plan_hashes_match_persisted_B0": gates["episode_plan_hashes_match_persisted_B0"] is True,
        "supervised_budget_and_optimizer_match_persisted_B0": gates[
            "supervised_budget_and_optimizer_match_persisted_B0"
        ] is True,
        "B14_cache_valid": gates["B14_cache_valid"] is True,
        "backbone_updates": gates["backbone_updates"] == 0,
        "provenance_component_overlap_across_folds": gates[
            "provenance_component_overlap_across_folds"
        ] == 0,
        "decoded_pixel_hash_overlap_across_folds": gates["decoded_pixel_hash_overlap_across_folds"] == 0,
        "nan_or_nonfinite_detected": gates["nan_or_nonfinite_detected"] is False,
        "runtime_unchanged": gates["runtime_unchanged"] is True,
        "final_test_read": gates["final_test_read"] is False,
        "automatic_promotion": gates["automatic_promotion"] is False,
        "positive_seed_count_gte": gates["positive_seed_count_gte"] >= 2,
        "delta_top1_component_bootstrap_lower_95_gt": gates[
            "delta_top1_component_bootstrap_lower_95_gt"
        ] > 0.0,
        "delta_macro_top1_component_bootstrap_lower_95_gte": gates[
            "delta_macro_top1_component_bootstrap_lower_95_gte"
        ] >= 0.0,
        "minimum_B14_over_S14_effective_rank_ratio_gte": gates[
            "minimum_B14_over_S14_effective_rank_ratio_gte"
        ] >= 0.9,
    }
    integrity_pass = all(gate_passes[key] for key in integrity_keys)
    efficacy_keys = (
        "positive_seed_count_gte",
        "delta_top1_component_bootstrap_lower_95_gt",
        "delta_macro_top1_component_bootstrap_lower_95_gte",
        "minimum_B14_over_S14_effective_rank_ratio_gte",
    )
    efficacy_pass = all(gate_passes[key] for key in efficacy_keys)
    top1_interval = intervals["delta_top1_95"]
    if efficacy_pass:
        classification = "learned_B14_supported_as_research_reference"
    elif top1_interval[1] <= 0.0:
        classification = "learned_B14_inferior"
    else:
        classification = "neutral_or_not_supported_at_current_power"
    return {
        "pass": integrity_pass and efficacy_pass,
        "integrity_pass": integrity_pass,
        "efficacy_pass": efficacy_pass,
        "score": overall["delta_top1_B14_projection_minus_S14_projection"],
        "diagnostic_classification": classification,
        "promotion_eligible": False,
        "overall_metrics": overall,
        "seed_metrics": seed_metrics,
        "bootstrap": {
            "method": intervals["method"],
            "replicates": intervals["replicates"],
            "delta_top1_B14_projection_minus_S14_projection_95": intervals["delta_top1_95"],
            "delta_macro_top1_B14_projection_minus_S14_projection_95": intervals[
                "delta_macro_top1_95"
            ],
        },
        "mcnemar": v9.exact_mcnemar(records),
        "positive_seed_count": positive_seed_count,
        "minimum_B14_over_S14_effective_rank_ratio": minimum_rank_ratio,
        "gates": gates,
        "gate_passes": gate_passes,
        "descriptive_breakdowns": descriptive_breakdowns(records),
        "decision": (
            "B14_research_reference_only_no_promotion"
            if efficacy_pass
            else "stop_frozen_backbone_scaling_in_this_run"
        ),
        "claim_limit": "neutral means unresolved at current component-level power, not equivalence",
        "final_test_read": False,
        "runtime_promotion": False,
    }


def command_validate(args: argparse.Namespace) -> dict[str, Any]:
    contract = validate_iteration_contract(args.spec, args.addendum, args.evaluator)
    baseline = validate_persisted_b0(
        args.v9_summary,
        args.v9_predictions,
        args.v9_runner,
        args.v9_cache_dir,
    )
    pin = backbone_screen.validate_backbone_manifest(args.B14_manifest)
    cache = backbone_screen.load_feature_cache(
        args.B14_cache,
        expected_manifest_sha256=pin["manifest_sha256"],
    )
    if cache["_cache_validation"]["cache_sha256"] != EXPECTED_B14_CACHE_SHA256:
        raise ValueError("B/14 feature cache does not match the preregistered iteration-2 artifact")
    isolation = validate_b14_to_b0_metadata(cache, baseline["records"])
    row_replay = validate_fold_cache_row_replay(cache, args.v9_cache_dir, baseline["cache_sha256"])
    result = {
        "pass": True,
        "contract": {key: value for key, value in contract.items() if key.endswith("sha256")},
        "B0_provenance": {key: value for key, value in baseline.items() if key.endswith("sha256") or key == "exact"},
        "B14_cache": cache["_cache_validation"],
        "B14_to_B0_fold_cache_row_replay": row_replay,
        "fold_isolation": {key: value for key, value in isolation.items() if key != "row_index"},
        "final_test_read": False,
        "runtime_unchanged": True,
    }
    print(v9.canonical_json(result), end="")
    return result


def command_run(args: argparse.Namespace) -> dict[str, Any]:
    if args.folds != EXPECTED_FOLDS or args.seeds != EXPECTED_SEEDS:
        raise ValueError("strict run requires the preregistered five folds and three seeds")
    if args.bootstrap_replicates != 2000:
        raise ValueError("strict run requires 2,000 bootstrap replicates")
    contract = validate_iteration_contract(args.spec, args.addendum, args.evaluator)
    baseline = validate_persisted_b0(
        args.v9_summary,
        args.v9_predictions,
        args.v9_runner,
        args.v9_cache_dir,
    )
    pin = backbone_screen.validate_backbone_manifest(args.B14_manifest)
    cache = backbone_screen.load_feature_cache(
        args.B14_cache,
        expected_manifest_sha256=pin["manifest_sha256"],
    )
    if cache["_cache_validation"]["cache_sha256"] != EXPECTED_B14_CACHE_SHA256:
        raise ValueError("B/14 feature cache SHA-256 differs from the locked iteration-2 artifact")
    isolation = validate_b14_to_b0_metadata(cache, baseline["records"])
    row_replay = validate_fold_cache_row_replay(cache, args.v9_cache_dir, baseline["cache_sha256"])
    before_runtime = v9.runtime_hashes(args.runtime_projection, args.runtime_prototypes)
    if before_runtime != baseline["summary"].get("runtime_checkpoint_sha256"):
        raise ValueError("runtime hashes no longer match the persisted B0 research source")
    prediction_device = v9.resolve_device(args.device)
    supervised_device = v9.resolve_device(args.supervised_device)
    if supervised_device.type == "cpu":
        if args.supervised_cpu_threads < 1:
            raise ValueError("supervised CPU thread count must be positive")
        torch.set_num_threads(args.supervised_cpu_threads)
    code_sha256 = v9.sha256_file(Path(__file__))
    candidate_predictions: dict[tuple[int, int, str], list[int]] = {}
    diagnostics: list[dict[str, Any]] = []
    train_component_counts: dict[int, dict[int, int]] = {}
    for fold in args.folds:
        train_indices, _ = _fold_indices(cache, fold)
        train_component_counts[fold] = _class_component_counts(cache, train_indices)
        for seed in args.seeds:
            predictions, diagnostic = train_or_resume_fold_seed(
                cache,
                fold=fold,
                seed=seed,
                source_diagnostic=baseline["diagnostics"][(fold, seed)],
                code_sha256=code_sha256,
                checkpoint_dir=args.output_dir / "checkpoints",
                prediction_device=prediction_device,
                supervised_device=supervised_device,
            )
            for row_id, topk in predictions.items():
                candidate_predictions[(fold, seed, row_id)] = topk
            diagnostics.append(diagnostic)

    records: list[dict[str, Any]] = []
    for source in baseline["records"]:
        fold = int(source["outer_fold"])
        seed = int(source["seed"])
        row_id = str(source["row_id"])
        key = (fold, seed, row_id)
        candidate_topk = candidate_predictions.get(key)
        if candidate_topk is None:
            raise ValueError(f"B/14 learned prediction missing: {key}")
        component_count = train_component_counts[fold][int(source["label"])]
        records.append(
            {
                "row_id": row_id,
                "label": int(source["label"]),
                "class_name": str(source["class_name"]),
                "provenance_component": str(source["provenance_component"]),
                "decoded_pixel_sha256": str(source["decoded_pixel_sha256"]),
                "outer_fold": fold,
                "seed": seed,
                "baseline_topk": [int(value) for value in source["baseline_topk"]],
                "candidate_topk": [int(value) for value in candidate_topk],
                "recipe": "S14-B0-projection-vs-B14-fixed-hidden-projection",
                "episode_plan_sha256": str(source["episode_plan_sha256"]),
                "folds_sha256": EXPECTED_FOLDS_SHA256,
                "S14_B0_predictions_sha256": EXPECTED_V9_PREDICTIONS_SHA256,
                "S14_B0_checkpoint_sha256": str(source["baseline_checkpoint_sha256"]),
                "B14_checkpoint_sha256": next(
                    item["checkpoint"]["sha256"]
                    for item in diagnostics
                    if item["fold"] == fold and item["seed"] == seed
                ),
                "B14_feature_cache_sha256": EXPECTED_B14_CACHE_SHA256,
                "runner_sha256": code_sha256,
                "train_provenance_component_count_for_label": component_count,
                "coverage_stratum": (
                    "single_train_component" if component_count == 1 else "multiple_train_components"
                ),
            }
        )
    after_runtime = v9.runtime_hashes(args.runtime_projection, args.runtime_prototypes)
    runtime_unchanged = before_runtime == after_runtime
    if not runtime_unchanged:
        raise RuntimeError("runtime projection/prototype artifacts changed during iteration 3")
    evaluation = evaluate_records(
        records,
        diagnostics,
        runtime_unchanged=runtime_unchanged,
        bootstrap_replicates=args.bootstrap_replicates,
        B0_provenance_exact=baseline["exact"] and row_replay["exact"],
        B14_cache_valid=True,
        fold_isolation=isolation,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "prediction_rows.jsonl"
    v9.write_jsonl(predictions_path, records)
    summary = {
        "schema_version": "autoresearch-self-supervised-v10.learned-backbone-evaluation",
        "iteration": 3,
        "name": "frozen-backbone-supervised-projection-comparison",
        **evaluation,
        "folds": args.folds,
        "seeds": args.seeds,
        "folds_sha256": EXPECTED_FOLDS_SHA256,
        "diagnostics": diagnostics,
        "paired_predictions_path": predictions_path.name,
        "paired_predictions_sha256": v9.sha256_file(predictions_path),
        "B0_provenance": {
            "summary_sha256": baseline["summary_sha256"],
            "predictions_sha256": baseline["predictions_sha256"],
            "runner_sha256": baseline["runner_sha256"],
            "data_sha256": EXPECTED_V9_DATA_SHA256,
            "cache_sha256": baseline["cache_sha256"],
            "fold_cache_row_replay": row_replay,
            "exact": baseline["exact"] and row_replay["exact"],
        },
        "B14_feature_cache": cache["_cache_validation"],
        "B14_backbone_pin": pin,
        "runner_sha256": code_sha256,
        "spec_sha256": contract["spec_sha256"],
        "interpretation_addendum_sha256": contract["addendum_sha256"],
        "evaluator_sha256": contract["evaluator_sha256"],
        "runtime_checkpoint_sha256": before_runtime,
        "runtime_unchanged": runtime_unchanged,
        "backbone_updates": 0,
        "final_test_read": False,
        "runtime_promotion": False,
        "automatic_promotion": False,
    }
    v9.write_json(args.output_dir / "summary.json", summary)
    v9.write_json(
        args.output_dir / "per_arm_metrics.json",
        {
            "overall": summary["overall_metrics"],
            "seeds": summary["seed_metrics"],
            "descriptive_breakdowns": summary["descriptive_breakdowns"],
        },
    )
    print(v9.canonical_json(summary), end="")
    return summary


def parse_int_csv(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def add_shared_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--v9-summary", type=Path, default=DEFAULT_V9_SUMMARY)
    parser.add_argument("--v9-predictions", type=Path, default=DEFAULT_V9_PREDICTIONS)
    parser.add_argument("--v9-runner", type=Path, default=DEFAULT_V9_RUNNER)
    parser.add_argument("--v9-cache-dir", type=Path, default=DEFAULT_V9_CACHE_DIR)
    parser.add_argument("--B14-manifest", type=Path, default=DEFAULT_B14_MANIFEST)
    parser.add_argument("--B14-cache", type=Path, default=DEFAULT_B14_CACHE)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--addendum", type=Path, default=DEFAULT_ADDENDUM)
    parser.add_argument("--evaluator", type=Path, default=DEFAULT_EVALUATOR)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate", help="verify all persisted inputs without training")
    add_shared_args(validate)
    run = subparsers.add_parser("run", help="run the strict learned projection comparison")
    add_shared_args(run)
    run.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    run.add_argument("--folds", type=parse_int_csv, default=EXPECTED_FOLDS)
    run.add_argument("--seeds", type=parse_int_csv, default=EXPECTED_SEEDS)
    run.add_argument("--bootstrap-replicates", type=int, default=2000)
    run.add_argument("--device", default="auto")
    run.add_argument("--supervised-device", default="cpu")
    run.add_argument("--supervised-cpu-threads", type=int, default=1)
    run.add_argument("--runtime-projection", type=Path, default=DEFAULT_RUNTIME_PROJECTION)
    run.add_argument("--runtime-prototypes", type=Path, default=DEFAULT_RUNTIME_PROTOTYPES)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "validate":
        command_validate(args)
    elif args.command == "run":
        command_run(args)
    else:  # pragma: no cover
        raise AssertionError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    main()

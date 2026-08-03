#!/usr/bin/env python3
"""Compare B0 and exact C1 against a covariance-regularized C1 readout."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import autoresearch_self_supervised_v9 as v9  # noqa: E402
import autoresearch_support_aware_readout_v10 as i5  # noqa: E402


V9_RUN = ROOT / ".omc/autoresearch/elements-baseline-replacement/runs/20260803-self-supervised-v9"
V10_RUN = ROOT / ".omc/autoresearch/elements-baseline-replacement/runs/20260803-self-supervised-v10"
DEFAULT_V9_SUMMARY = V9_RUN / "iteration-0001/results/paired_seed_summary.json"
DEFAULT_V9_PREDICTIONS = V9_RUN / "iteration-0001/results/prediction_rows.jsonl"
DEFAULT_SPEC = V10_RUN / "specs/iteration-0007.json"
DEFAULT_EVALUATOR = V10_RUN / "evaluator-iteration-0007.json"
DEFAULT_OUTPUT_DIR = V10_RUN / "iteration-0007/results"
DEFAULT_CACHE_DIR = V9_RUN / "iteration-0001/caches"
DEFAULT_RUNTIME_PROJECTION = ROOT / "backend/codex_model/weights/projection.pt"
DEFAULT_RUNTIME_PROTOTYPES = ROOT / "backend/codex_model/weights/prototypes.pt"

EXPECTED_SPEC_SHA256 = "da2b5f8c0cb7fdefe407cd6de10cc19bd3e944d39a2298b32608d887f08b9b39"
EXPECTED_EVALUATOR_SHA256 = "5e6e1f56131e01be944e3b5cfef21aeaeec3385436bf708538b01c33d43ae379"
EXPECTED_I5_RUNNER_SHA256 = "acd06503d801bdaf30e1b3d8569fd6713d9b3b83e545561fc26677998a8de0c6"
EXPECTED_V9_RUNNER_SHA256 = i5.EXPECTED_V9_RUNNER_SHA256
EXPECTED_V9_SUMMARY_SHA256 = i5.EXPECTED_V9_SUMMARY_SHA256
EXPECTED_V9_PREDICTIONS_SHA256 = i5.EXPECTED_V9_PREDICTIONS_SHA256
EXPECTED_CACHE_SHA256 = i5.EXPECTED_CACHE_SHA256
EXPECTED_RUNTIME_SHA256 = i5.EXPECTED_RUNTIME_SHA256
EXPECTED_FOLDS = i5.EXPECTED_FOLDS
EXPECTED_SEEDS = i5.EXPECTED_SEEDS
EXPECTED_PAIRED_ROWS = i5.EXPECTED_PAIRED_ROWS
COVARIANCE_COEFFICIENT = 0.04


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    v9.write_json(path, value)


def validate_contract(spec_path: Path, evaluator_path: Path) -> dict[str, str]:
    spec_sha256 = v9.sha256_file(spec_path)
    evaluator_sha256 = v9.sha256_file(evaluator_path)
    if spec_sha256 != EXPECTED_SPEC_SHA256:
        raise ValueError("iteration-7 spec SHA mismatch")
    if evaluator_sha256 != EXPECTED_EVALUATOR_SHA256:
        raise ValueError("iteration-7 evaluator SHA mismatch")
    spec = read_json(spec_path)
    evaluator = read_json(evaluator_path)
    if spec.get("iteration") != 7 or evaluator.get("iteration") != 7:
        raise ValueError("iteration-7 contract mismatch")
    if spec.get("status") != (
        "preregistered_revision_after_council_scale_review_before_any_iteration_0007_candidate_prediction"
    ):
        raise ValueError("iteration-7 experiment was not preregistered")
    factor = spec.get("single_factor", {})
    if factor.get("coefficient") != COVARIANCE_COEFFICIENT:
        raise ValueError("iteration-7 covariance coefficient changed")
    if factor.get("penalty_tensor") != (
        "the exact 100 post-L2 query embeddings per episode shared with the prototypical classification loss"
    ):
        raise ValueError("iteration-7 penalty tensor changed")
    if spec.get("evaluation", {}).get("rank_policy", {}).get("hard_gate") is not False:
        raise ValueError("iteration-7 rank policy changed")
    return {
        "spec_sha256": spec_sha256,
        "evaluator_sha256": evaluator_sha256,
    }


def validate_dependencies() -> dict[str, str]:
    i5_runner_sha256 = v9.sha256_file(Path(i5.__file__).resolve())
    if i5_runner_sha256 != EXPECTED_I5_RUNNER_SHA256:
        raise ValueError("imported iteration-5 helper runner SHA mismatch")
    return {
        "i5_runner_sha256": i5_runner_sha256,
        "v9_runner_sha256": i5.validate_v9_runner(),
    }


def runtime_hashes(projection: Path, prototypes: Path) -> dict[str, str]:
    hashes = {
        "projection": v9.sha256_file(projection),
        "prototypes": v9.sha256_file(prototypes),
    }
    if hashes != EXPECTED_RUNTIME_SHA256:
        raise ValueError("runtime projection/prototype SHA mismatch")
    return hashes


def off_diagonal_covariance_penalty(values: torch.Tensor) -> torch.Tensor:
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("covariance penalty requires a 2D batch with at least two rows")
    centered = values - values.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / (values.shape[0] - 1)
    dimension = covariance.shape[0]
    off_diagonal = covariance.flatten()[:-1].view(dimension - 1, dimension + 1)[:, 1:].flatten()
    return off_diagonal.square().sum() / dimension


def train_supervised_covariance(
    model: v9.ProjectionHead,
    features: torch.Tensor,
    episode_plan: Sequence[Sequence[v9.Episode]],
    *,
    learning_rate: float,
    weight_decay: float,
    temperature: float,
    warmup_epochs: int,
    rng_seed: int,
    device: torch.device,
    covariance_coefficient: float,
) -> dict[str, Any]:
    if covariance_coefficient < 0.0:
        raise ValueError("covariance coefficient must be non-negative")
    v9.reset_training_rng(rng_seed)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    criterion = v9.PrototypicalLoss(temperature=temperature)
    features_on_device = features.to(device=device, dtype=torch.float32)
    device_episode_plan = [
        tuple(
            torch.tensor(
                [episode[field_index] for episode in episodes],
                dtype=torch.long,
                device=device,
            )
            for field_index in range(4)
        )
        for episodes in episode_plan
    ]
    total_epochs = len(device_episode_plan)
    last_classification_loss = 0.0
    last_covariance_penalty = 0.0
    last_weighted_covariance_penalty = 0.0
    last_total_loss = 0.0
    last_accuracy = 0.0
    penalty_sum = 0.0
    weighted_penalty_sum = 0.0
    classification_loss_sum = 0.0
    episode_total = 0
    minimum_post_l2_query_batch_mean_std = math.inf
    for epoch, episode_tensors in enumerate(device_episode_plan):
        if epoch < warmup_epochs:
            factor = 0.01 + 0.99 * (epoch + 1) / max(1, warmup_epochs)
        else:
            progress = (epoch - warmup_epochs) / max(
                1,
                total_epochs - warmup_epochs - 1,
            )
            factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = learning_rate * factor
        epoch_classification = torch.zeros((), device=device)
        epoch_penalty = torch.zeros((), device=device)
        epoch_total = torch.zeros((), device=device)
        epoch_accuracy = torch.zeros((), device=device)
        episode_count = episode_tensors[0].shape[0]
        for episode_index in range(episode_count):
            support_indices, support_labels, query_indices, query_labels = (
                values[episode_index] for values in episode_tensors
            )
            support_raw = model.net(features_on_device[support_indices])
            query_raw = model.net(features_on_device[query_indices])
            support_normalized = F.normalize(support_raw, p=2, dim=-1)
            query_normalized = F.normalize(query_raw, p=2, dim=-1)
            result = criterion(
                support_normalized,
                support_labels,
                query_normalized,
                query_labels,
            )
            covariance_penalty = off_diagonal_covariance_penalty(query_normalized)
            weighted_penalty = covariance_coefficient * covariance_penalty
            total_loss = (
                result["loss"]
                if covariance_coefficient == 0.0
                else result["loss"] + weighted_penalty
            )
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            query_mean_std = float(
                torch.sqrt(
                    query_normalized.detach().var(dim=0, unbiased=False) + 1e-4
                )
                .mean()
                .cpu()
            )
            minimum_post_l2_query_batch_mean_std = min(
                minimum_post_l2_query_batch_mean_std,
                query_mean_std,
            )
            penalty_value = float(covariance_penalty.detach().cpu())
            weighted_penalty_value = float(weighted_penalty.detach().cpu())
            classification_loss_value = float(result["loss"].detach().cpu())
            penalty_sum += penalty_value
            weighted_penalty_sum += weighted_penalty_value
            classification_loss_sum += classification_loss_value
            episode_total += 1
            epoch_classification += result["loss"].detach()
            epoch_penalty += covariance_penalty.detach()
            epoch_total += total_loss.detach()
            epoch_accuracy += result["accuracy"].detach()
        last_classification_loss = float((epoch_classification / episode_count).cpu())
        last_covariance_penalty = float((epoch_penalty / episode_count).cpu())
        last_weighted_covariance_penalty = (
            covariance_coefficient * last_covariance_penalty
        )
        last_total_loss = float((epoch_total / episode_count).cpu())
        last_accuracy = float((epoch_accuracy / episode_count).cpu())
    return {
        "epochs": total_epochs,
        "episodes_per_epoch": device_episode_plan[0][0].shape[0],
        "covariance_coefficient": covariance_coefficient,
        "last_classification_loss": last_classification_loss,
        "last_covariance_penalty": last_covariance_penalty,
        "last_weighted_covariance_penalty": last_weighted_covariance_penalty,
        "last_weighted_covariance_penalty_over_classification_loss": (
            last_weighted_covariance_penalty / last_classification_loss
            if last_classification_loss > 0.0
            else 0.0
        ),
        "last_total_loss": last_total_loss,
        "last_accuracy": last_accuracy,
        "mean_classification_loss": classification_loss_sum / episode_total,
        "mean_covariance_penalty": penalty_sum / episode_total,
        "mean_weighted_covariance_penalty": weighted_penalty_sum / episode_total,
        "mean_weighted_covariance_penalty_over_classification_loss": (
            weighted_penalty_sum / classification_loss_sum
            if classification_loss_sum > 0.0
            else 0.0
        ),
        "minimum_post_l2_query_batch_mean_std": (
            minimum_post_l2_query_batch_mean_std
        ),
        "penalty_tensor": "query_post_l2_shared_with_classification",
        "penalty_normalization": "off_diagonal_squared_sum_div_embedding_dim",
    }


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    *,
    fold: int,
    seed: int,
    runner_sha256: str,
    contract: dict[str, str],
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "autoresearch-self-supervised-v10.covariance-checkpoint",
        "fold": fold,
        "seed": seed,
        "arm": "C1-query-covariance-0.04",
        "runner_sha256": runner_sha256,
        "spec_sha256": contract["spec_sha256"],
        "evaluator_sha256": contract["evaluator_sha256"],
        "model_state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return v9.sha256_file(path)


def _load_resumable_pair(
    pair_path: Path,
    checkpoint_path: Path,
    *,
    fold: int,
    seed: int,
    expected_rows: int,
    runner_sha256: str,
    cache_sha256: str,
    contract: dict[str, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
    if not pair_path.is_file() or not checkpoint_path.is_file():
        return None
    try:
        payload = read_json(pair_path)
        expected = {
            "fold": fold,
            "seed": seed,
            "runner_sha256": runner_sha256,
            "cache_sha256": cache_sha256,
            "spec_sha256": contract["spec_sha256"],
            "evaluator_sha256": contract["evaluator_sha256"],
        }
        if any(payload.get(key) != value for key, value in expected.items()):
            return None
        if len(payload["records"]) != expected_rows:
            return None
        diagnostics = dict(payload["diagnostics"])
        if diagnostics["candidate_checkpoint_sha256"] != v9.sha256_file(
            checkpoint_path
        ):
            return None
        if not diagnostics["control_state_dict_matches_persisted_v9_C1"]:
            return None
        if int(diagnostics["control_topk_match_count"]) != expected_rows:
            return None
        diagnostics["resumed_from_pair_artifact"] = True
        return payload["records"], diagnostics
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _persist_pair(
    pair_path: Path,
    records: Sequence[dict[str, Any]],
    diagnostics: dict[str, Any],
    *,
    fold: int,
    seed: int,
    runner_sha256: str,
    cache_sha256: str,
    contract: dict[str, str],
) -> None:
    pair_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "autoresearch-self-supervised-v10.covariance-pair",
        "fold": fold,
        "seed": seed,
        "runner_sha256": runner_sha256,
        "cache_sha256": cache_sha256,
        "spec_sha256": contract["spec_sha256"],
        "evaluator_sha256": contract["evaluator_sha256"],
        "records": list(records),
        "diagnostics": diagnostics,
    }
    temporary = pair_path.with_suffix(pair_path.suffix + ".tmp")
    write_json(temporary, payload)
    temporary.replace(pair_path)


def run_fold_seed(
    cache: dict[str, Any],
    *,
    fold: int,
    seed: int,
    persisted: dict[str, Any],
    checkpoint_dir: Path,
    pair_dir: Path,
    device: torch.device,
    supervised_device: torch.device,
    runner_sha256: str,
    cache_sha256: str,
    contract: dict[str, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_diagnostic = persisted["diagnostics"][(fold, seed)]
    source_rows = {
        str(row["row_id"]): row
        for row in persisted["records"]
        if int(row["outer_fold"]) == fold and int(row["seed"]) == seed
    }
    checkpoint_path = checkpoint_dir / f"fold-{fold:02d}-seed-{seed}-covariance.pt"
    pair_path = pair_dir / f"fold-{fold:02d}-seed-{seed}.json"
    resumed = _load_resumable_pair(
        pair_path,
        checkpoint_path,
        fold=fold,
        seed=seed,
        expected_rows=len(source_rows),
        runner_sha256=runner_sha256,
        cache_sha256=cache_sha256,
        contract=contract,
    )
    if resumed is not None:
        return resumed

    v9.configure_determinism(seed)
    initial_model = v9.ProjectionHead(input_dim=384, embedding_dim=128)
    unused_b0 = copy.deepcopy(initial_model).to(device)
    pretrained = copy.deepcopy(initial_model).to(device)
    initial_hash = v9.state_dict_sha256(initial_model)
    if initial_hash != str(source_diagnostic["initial_state_sha256"]):
        raise ValueError(f"initial state mismatch for fold={fold}, seed={seed}")
    ssl = v9.pretrain_vicreg(
        pretrained,
        cache["train"],
        epochs=30,
        batch_size=256,
        learning_rate=3e-4,
        weight_decay=1e-4,
        seed=seed,
        device=device,
    )
    if ssl["plan_sha256"] != source_diagnostic["ssl"]["plan_sha256"]:
        raise ValueError(f"VICReg plan mismatch for fold={fold}, seed={seed}")
    control = copy.deepcopy(pretrained).to(supervised_device)
    candidate = copy.deepcopy(pretrained).to(supervised_device)
    shared_pretrained_hash = v9.state_dict_sha256(pretrained)
    if len(
        {
            shared_pretrained_hash,
            v9.state_dict_sha256(control),
            v9.state_dict_sha256(candidate),
        }
    ) != 1:
        raise AssertionError("control/candidate pretrained state mismatch")

    train_labels = torch.tensor(cache["train"]["class_label"], dtype=torch.long)
    episode_seed = v9.stable_seed("supervised-episodes", fold, seed)
    plan, plan_hash = v9.build_episode_plan(
        train_labels,
        n_way=20,
        k_shot=3,
        q_queries=5,
        epochs=30,
        episodes_per_epoch=100,
        seed=episode_seed,
    )
    if plan_hash != str(source_diagnostic["episode_plan_sha256"]):
        raise ValueError(f"episode plan mismatch for fold={fold}, seed={seed}")
    plan_audit = i5.audit_plan_train_only(
        plan,
        cache["train"]["row_id"],
        cache["oof"]["row_id"],
    )
    if any(plan_audit.values()):
        raise ValueError(f"historical episode plan integrity failure: {plan_audit}")
    rng_seed = v9.stable_seed("supervised-rng", fold, seed)
    control_training = v9.train_supervised(
        control,
        cache["train"]["base_features"],
        plan,
        learning_rate=1e-3,
        weight_decay=1e-4,
        temperature=0.1,
        warmup_epochs=5,
        rng_seed=rng_seed,
        device=supervised_device,
    )
    candidate_training = train_supervised_covariance(
        candidate,
        cache["train"]["base_features"],
        plan,
        learning_rate=1e-3,
        weight_decay=1e-4,
        temperature=0.1,
        warmup_epochs=5,
        rng_seed=rng_seed,
        device=supervised_device,
        covariance_coefficient=COVARIANCE_COEFFICIENT,
    )
    control = control.to(device)
    candidate = candidate.to(device)
    control_topk, control_rank = v9.predict_arm(
        control,
        cache["train"],
        cache["oof"],
        device=device,
    )
    candidate_topk, candidate_rank = v9.predict_arm(
        candidate,
        cache["train"],
        cache["oof"],
        device=device,
    )

    source_checkpoint_path = i5._resolve_source_checkpoint(source_diagnostic)
    if v9.sha256_file(source_checkpoint_path) != str(
        source_diagnostic["checkpoints"]["candidate"]["sha256"]
    ):
        raise ValueError(f"persisted C1 checkpoint SHA mismatch for fold={fold}, seed={seed}")
    _source_state, source_state_hash = i5._load_projection_state(
        source_checkpoint_path
    )
    control_state_hash = v9.state_dict_sha256(control)
    if control_state_hash != source_state_hash:
        raise ValueError(f"control state_dict mismatch for fold={fold}, seed={seed}")

    records: list[dict[str, Any]] = []
    control_topk_match_count = 0
    for index, row_id_value in enumerate(cache["oof"]["row_id"]):
        row_id = str(row_id_value)
        scaffold = source_rows[row_id]
        actual_control = [int(value) for value in control_topk[index]]
        expected_control = [int(value) for value in scaffold["candidate_topk"]]
        if actual_control != expected_control:
            raise ValueError(f"control topk mismatch for persisted C1 row {row_id}")
        control_topk_match_count += 1
        records.append(
            {
                "row_id": row_id,
                "provenance_component": str(cache["oof"]["component_id"][index]),
                "decoded_pixel_sha256": str(
                    cache["oof"]["decoded_pixel_sha256"][index]
                ),
                "label": int(cache["oof"]["class_label"][index]),
                "class_name": str(cache["oof"]["class_name"][index]),
                "outer_fold": fold,
                "seed": seed,
                "baseline_topk": [int(value) for value in scaffold["baseline_topk"]],
                "control_topk": actual_control,
                "candidate_topk": [int(value) for value in candidate_topk[index]],
                "episode_plan_sha256": plan_hash,
                "data_sha256": cache_sha256,
            }
        )

    checkpoint_sha256 = _save_checkpoint(
        checkpoint_path,
        candidate,
        fold=fold,
        seed=seed,
        runner_sha256=runner_sha256,
        contract=contract,
    )
    numeric = [
        float(control_rank),
        float(candidate_rank),
        float(candidate_training["last_total_loss"]),
        float(candidate_training["mean_covariance_penalty"]),
        float(candidate_training["minimum_post_l2_query_batch_mean_std"]),
        float(
            candidate_training[
                "mean_weighted_covariance_penalty_over_classification_loss"
            ]
        ),
    ]
    b0_rank = float(source_diagnostic["baseline_effective_rank"])
    diagnostics = {
        "fold": fold,
        "seed": seed,
        "devices": {
            "ssl_and_prediction": str(device),
            "supervised": str(supervised_device),
        },
        "initial_state_sha256": initial_hash,
        "shared_pretrained_state_sha256": shared_pretrained_hash,
        "candidate_and_control_shared_pretrained_state": True,
        "episode_plan_sha256": plan_hash,
        "episode_plan_matches_persisted_v9_C1": True,
        "control_state_dict_sha256": control_state_hash,
        "persisted_v9_C1_state_dict_sha256": source_state_hash,
        "control_state_dict_matches_persisted_v9_C1": True,
        "control_topk_match_count": control_topk_match_count,
        "control_topk_matches_persisted_v9_C1": (
            control_topk_match_count == len(records)
        ),
        "candidate_and_control_budget_equal": (
            control_training["epochs"] == candidate_training["epochs"]
            and control_training["episodes_per_epoch"]
            == candidate_training["episodes_per_epoch"]
        ),
        "persisted_v9_B0_effective_rank": b0_rank,
        "control_C1_effective_rank": float(control_rank),
        "candidate_effective_rank": float(candidate_rank),
        "candidate_over_control_C1_effective_rank_ratio": (
            float(candidate_rank / control_rank) if control_rank else 0.0
        ),
        "candidate_over_persisted_B0_effective_rank_ratio": (
            float(candidate_rank / b0_rank) if b0_rank else 0.0
        ),
        "candidate_checkpoint_path": str(checkpoint_path),
        "candidate_checkpoint_sha256": checkpoint_sha256,
        "candidate_checkpoint_exists": checkpoint_path.is_file(),
        "ssl": ssl,
        "control_training": control_training,
        "candidate_training": candidate_training,
        **plan_audit,
        "nan_or_nonfinite_detected": not all(math.isfinite(value) for value in numeric),
        "resumed_from_pair_artifact": False,
    }
    _persist_pair(
        pair_path,
        records,
        diagnostics,
        fold=fold,
        seed=seed,
        runner_sha256=runner_sha256,
        cache_sha256=cache_sha256,
        contract=contract,
    )
    del unused_b0
    return records, diagnostics


def comparison_records(
    records: Sequence[dict[str, Any]],
    *,
    baseline_key: str,
    candidate_key: str,
) -> list[dict[str, Any]]:
    return [
        {
            **row,
            "baseline_topk": list(row[baseline_key]),
            "candidate_topk": list(row[candidate_key]),
        }
        for row in records
    ]


def comparison_summary(
    records: Sequence[dict[str, Any]],
    *,
    bootstrap_seed: int,
) -> dict[str, Any]:
    seeds = sorted({int(row["seed"]) for row in records})
    folds = sorted({int(row["outer_fold"]) for row in records})
    seed_metrics = [
        {
            "seed": seed,
            **v9.accuracy_metrics(
                [row for row in records if int(row["seed"]) == seed]
            ),
        }
        for seed in seeds
    ]
    fold_metrics = [
        {
            "fold": fold,
            **v9.accuracy_metrics(
                [row for row in records if int(row["outer_fold"]) == fold]
            ),
        }
        for fold in folds
    ]
    return {
        "overall": v9.accuracy_metrics(records),
        "seeds": seed_metrics,
        "folds": fold_metrics,
        "positive_seed_count": sum(
            item["delta_top1"] > 0.0 for item in seed_metrics
        ),
        "minimum_fold_delta_top1": min(
            item["delta_top1"] for item in fold_metrics
        ),
        "bootstrap": v9.paired_component_bootstrap(
            records,
            replicates=2000,
            seed=bootstrap_seed,
        ),
        "mcnemar": v9.exact_mcnemar(records),
    }


def aggregate_results(
    records: Sequence[dict[str, Any]],
    diagnostics: Sequence[dict[str, Any]],
    cache_validations: Sequence[dict[str, Any]],
    *,
    runtime_unchanged: bool,
) -> dict[str, Any]:
    b0_records = comparison_records(
        records,
        baseline_key="baseline_topk",
        candidate_key="candidate_topk",
    )
    c1_records = comparison_records(
        records,
        baseline_key="control_topk",
        candidate_key="candidate_topk",
    )
    vs_b0 = comparison_summary(b0_records, bootstrap_seed=20260803)
    vs_c1 = comparison_summary(c1_records, bootstrap_seed=20260804)
    seeds = sorted({int(row["seed"]) for row in records})
    folds = sorted({int(row["outer_fold"]) for row in records})
    integrity_values = {
        "paired_prediction_rows": len(records),
        "paired_seed_count": len(seeds),
        "outer_fold_count": len(folds),
        "candidate_checkpoint_count": sum(
            bool(item["candidate_checkpoint_exists"]) for item in diagnostics
        ),
        "control_state_dict_matches_persisted_v9_C1_count": sum(
            bool(item["control_state_dict_matches_persisted_v9_C1"])
            for item in diagnostics
        ),
        "control_topk_matches_persisted_v9_C1_rows": sum(
            int(item["control_topk_match_count"]) for item in diagnostics
        ),
        "episode_plan_matches_persisted_v9_C1_count": sum(
            bool(item["episode_plan_matches_persisted_v9_C1"])
            for item in diagnostics
        ),
        "v9_summary_and_prediction_hashes_match": True,
        "v9_cache_hashes_match_count": sum(
            bool(item["cache_sha256_matches"]) for item in cache_validations
        ),
        "candidate_and_control_shared_pretrained_state": all(
            bool(item["candidate_and_control_shared_pretrained_state"])
            for item in diagnostics
        ),
        "candidate_and_control_budget_equal": all(
            bool(item["candidate_and_control_budget_equal"])
            for item in diagnostics
        ),
        "candidate_episode_oof_row_exposure": sum(
            int(item["candidate_episode_oof_row_exposure"])
            for item in diagnostics
        ),
        "runtime_unchanged": runtime_unchanged,
        "final_test_read": False,
        "automatic_promotion": False,
    }
    integrity_passes = {
        "paired_prediction_rows": integrity_values["paired_prediction_rows"]
        == EXPECTED_PAIRED_ROWS,
        "paired_seed_count": integrity_values["paired_seed_count"]
        == len(EXPECTED_SEEDS),
        "outer_fold_count": integrity_values["outer_fold_count"]
        == len(EXPECTED_FOLDS),
        "candidate_checkpoint_count": integrity_values["candidate_checkpoint_count"]
        == 15,
        "control_state_dict_matches_persisted_v9_C1_count": integrity_values[
            "control_state_dict_matches_persisted_v9_C1_count"
        ]
        == 15,
        "control_topk_matches_persisted_v9_C1_rows": integrity_values[
            "control_topk_matches_persisted_v9_C1_rows"
        ]
        == EXPECTED_PAIRED_ROWS,
        "episode_plan_matches_persisted_v9_C1_count": integrity_values[
            "episode_plan_matches_persisted_v9_C1_count"
        ]
        == 15,
        "v9_summary_and_prediction_hashes_match": integrity_values[
            "v9_summary_and_prediction_hashes_match"
        ]
        is True,
        "v9_cache_hashes_match_count": integrity_values[
            "v9_cache_hashes_match_count"
        ]
        == 5,
        "candidate_and_control_shared_pretrained_state": integrity_values[
            "candidate_and_control_shared_pretrained_state"
        ]
        is True,
        "candidate_and_control_budget_equal": integrity_values[
            "candidate_and_control_budget_equal"
        ]
        is True,
        "candidate_episode_oof_row_exposure": integrity_values[
            "candidate_episode_oof_row_exposure"
        ]
        == 0,
        "runtime_unchanged": integrity_values["runtime_unchanged"] is True,
        "final_test_read": integrity_values["final_test_read"] is False,
        "automatic_promotion": integrity_values["automatic_promotion"] is False,
    }
    efficacy_values = {
        "delta_top1": float(vs_b0["overall"]["delta_top1"]),
        "positive_seed_count": int(vs_b0["positive_seed_count"]),
        "delta_top1_component_bootstrap_lower_95": float(
            vs_b0["bootstrap"]["delta_top1_95"][0]
        ),
        "delta_macro_top1": float(vs_b0["overall"]["delta_macro_top1"]),
        "delta_top3": float(vs_b0["overall"]["delta_top3"]),
        "minimum_fold_delta_top1": float(vs_b0["minimum_fold_delta_top1"]),
    }
    efficacy_passes = {
        "delta_top1_gte": efficacy_values["delta_top1"] >= 0.01,
        "positive_seed_count_gte": efficacy_values["positive_seed_count"] >= 2,
        "delta_top1_component_bootstrap_lower_95_gt": efficacy_values[
            "delta_top1_component_bootstrap_lower_95"
        ]
        > 0.0,
        "delta_macro_top1_gte": efficacy_values["delta_macro_top1"] >= -0.005,
        "delta_top3_gte": efficacy_values["delta_top3"] >= 0.0,
        "minimum_fold_delta_top1_gte": efficacy_values[
            "minimum_fold_delta_top1"
        ]
        >= -0.05,
    }
    noninferiority_values = {
        "delta_top1": float(vs_c1["overall"]["delta_top1"]),
        "delta_top3": float(vs_c1["overall"]["delta_top3"]),
        "delta_macro_top1": float(vs_c1["overall"]["delta_macro_top1"]),
    }
    noninferiority_passes = {
        "delta_top1_gte": noninferiority_values["delta_top1"] >= -0.005,
        "delta_top3_gte": noninferiority_values["delta_top3"] >= -0.01,
        "delta_macro_top1_gte": noninferiority_values["delta_macro_top1"]
        >= -0.01,
    }
    mean_penalty = float(
        np.mean(
            [
                item["candidate_training"]["mean_covariance_penalty"]
                for item in diagnostics
            ]
        )
    )
    mean_weighted_penalty = float(
        np.mean(
            [
                item["candidate_training"]["mean_weighted_covariance_penalty"]
                for item in diagnostics
            ]
        )
    )
    mean_penalty_ratio = float(
        np.mean(
            [
                item["candidate_training"][
                    "mean_weighted_covariance_penalty_over_classification_loss"
                ]
                for item in diagnostics
            ]
        )
    )
    stability_values = {
        "minimum_post_l2_query_batch_mean_std": min(
            float(
                item["candidate_training"]["minimum_post_l2_query_batch_mean_std"]
            )
            for item in diagnostics
        ),
        "nan_or_nonfinite_detected": any(
            bool(item["nan_or_nonfinite_detected"]) for item in diagnostics
        ),
        "mean_covariance_penalty": mean_penalty,
        "mean_weighted_covariance_penalty": mean_weighted_penalty,
        "mean_weighted_covariance_penalty_over_classification_loss": mean_penalty_ratio,
    }
    stability_passes = {
        "minimum_post_l2_query_batch_mean_std_gte": stability_values[
            "minimum_post_l2_query_batch_mean_std"
        ]
        >= 0.01,
        "nan_or_nonfinite_detected": stability_values[
            "nan_or_nonfinite_detected"
        ]
        is False,
        "mean_covariance_penalty_gt": mean_penalty > 0.0,
        "mean_weighted_covariance_penalty_gt": mean_weighted_penalty > 0.0,
    }
    minimum_rank_b0 = min(
        float(item["candidate_over_persisted_B0_effective_rank_ratio"])
        for item in diagnostics
    )
    minimum_rank_c1 = min(
        float(item["candidate_over_control_C1_effective_rank_ratio"])
        for item in diagnostics
    )
    integrity_ok = all(integrity_passes.values())
    engaged = (
        math.isfinite(mean_penalty)
        and math.isfinite(mean_weighted_penalty)
        and math.isfinite(mean_penalty_ratio)
        and mean_penalty > 0.0
        and mean_weighted_penalty > 0.0
    )
    supported = (
        integrity_ok
        and all(efficacy_passes.values())
        and all(noninferiority_passes.values())
        and all(stability_passes.values())
    )
    decision = (
        "invalid"
        if not integrity_ok
        else "intervention_not_engaged"
        if not engaged
        else "supported"
        if supported
        else "neutral_or_not_supported"
    )
    return {
        "pass": supported,
        "score": efficacy_values["delta_top1"],
        "hypothesis_supported": supported,
        "decision": decision,
        "promotion_eligible": False,
        "comparison_vs_persisted_B0": vs_b0,
        "comparison_vs_exact_C1": vs_c1,
        "integrity_gates": integrity_values,
        "integrity_gate_passes": integrity_passes,
        "efficacy_vs_B0_gates": efficacy_values,
        "efficacy_vs_B0_gate_passes": efficacy_passes,
        "noninferiority_vs_C1_gates": noninferiority_values,
        "noninferiority_vs_C1_gate_passes": noninferiority_passes,
        "direct_stability_gates": stability_values,
        "direct_stability_gate_passes": stability_passes,
        "rank_diagnostic": {
            "minimum_candidate_over_persisted_B0": minimum_rank_b0,
            "minimum_candidate_over_exact_C1": minimum_rank_c1,
            "council_escalation_threshold": 0.8,
            "council_escalation_required": minimum_rank_b0 < 0.8,
            "hard_gate": False,
        },
        "final_test_read": False,
        "runtime_promotion": False,
    }


def command_run(args: argparse.Namespace) -> dict[str, Any]:
    contract = validate_contract(args.spec, args.evaluator)
    dependencies = validate_dependencies()
    persisted = i5.validate_inputs(args.v9_summary, args.v9_predictions)
    before_runtime = runtime_hashes(
        args.runtime_projection,
        args.runtime_prototypes,
    )
    device = v9.resolve_device(args.device)
    supervised_device = v9.resolve_device(args.supervised_device)
    if supervised_device.type == "cpu":
        if args.supervised_cpu_threads < 1:
            raise ValueError("supervised CPU threads must be positive")
        torch.set_num_threads(args.supervised_cpu_threads)

    caches: dict[int, dict[str, Any]] = {}
    cache_validations: list[dict[str, Any]] = []
    for fold in EXPECTED_FOLDS:
        cache_path = v9.cache_path(args.cache_dir, fold, 8, None)
        cache, validation = i5.load_validated_cache(
            cache_path,
            fold=fold,
            views=8,
        )
        caches[fold] = cache
        cache_validations.append(validation)

    runner_sha256 = v9.sha256_file(Path(__file__).resolve())
    all_records: list[dict[str, Any]] = []
    all_diagnostics: list[dict[str, Any]] = []
    for fold in EXPECTED_FOLDS:
        cache = caches[fold]
        cache_sha256 = str(cache["_cache_validation"]["cache_sha256"])
        for seed in EXPECTED_SEEDS:
            records, diagnostics = run_fold_seed(
                cache,
                fold=fold,
                seed=seed,
                persisted=persisted,
                checkpoint_dir=args.output_dir / "checkpoints",
                pair_dir=args.output_dir / "pairs",
                device=device,
                supervised_device=supervised_device,
                runner_sha256=runner_sha256,
                cache_sha256=cache_sha256,
                contract=contract,
            )
            provenance = {
                "runner_sha256": runner_sha256,
                "spec_sha256": contract["spec_sha256"],
                "evaluator_sha256": contract["evaluator_sha256"],
                "v9_summary_sha256": EXPECTED_V9_SUMMARY_SHA256,
                "v9_predictions_sha256": EXPECTED_V9_PREDICTIONS_SHA256,
                "candidate_checkpoint_sha256": diagnostics[
                    "candidate_checkpoint_sha256"
                ],
            }
            for record in records:
                record.update(provenance)
            all_records.extend(records)
            all_diagnostics.append(diagnostics)

    after_runtime = runtime_hashes(
        args.runtime_projection,
        args.runtime_prototypes,
    )
    runtime_unchanged = before_runtime == after_runtime
    if not runtime_unchanged:
        raise RuntimeError("runtime artifacts changed during iteration 7")
    fold_assignments = sorted(
        {
            (
                str(row["row_id"]),
                int(row["outer_fold"]),
                str(row["provenance_component"]),
                str(row["decoded_pixel_sha256"]),
            )
            for row in all_records
        }
    )
    folds_sha256 = v9.sha256_json(fold_assignments)
    for row in all_records:
        row["folds_sha256"] = folds_sha256

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "prediction_rows.jsonl"
    v9.write_jsonl(predictions_path, all_records)
    result = aggregate_results(
        all_records,
        all_diagnostics,
        cache_validations,
        runtime_unchanged=runtime_unchanged,
    )
    result.update(
        {
            "schema_version": "autoresearch-self-supervised-v10.covariance-evaluation",
            "iteration": 7,
            "name": "c1-readout-query-covariance-regularization",
            "paired_predictions_path": predictions_path.name,
            "folds": EXPECTED_FOLDS,
            "seeds": EXPECTED_SEEDS,
            "folds_sha256": folds_sha256,
            "cache_sha256": {
                str(item["fold"]): item["cache_sha256"]
                for item in cache_validations
            },
            "runner_sha256": runner_sha256,
            **dependencies,
            "spec_sha256": contract["spec_sha256"],
            "evaluator_sha256": contract["evaluator_sha256"],
            "runtime_checkpoint_sha256": before_runtime,
            "runtime_unchanged": runtime_unchanged,
            "cache_validations": cache_validations,
            "diagnostics": all_diagnostics,
        }
    )
    write_json(args.output_dir / "summary.json", result)
    write_json(args.output_dir / "evaluation.json", result)
    print(v9.canonical_json(result), end="")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser(
        "run",
        help="run exact C1 and covariance-regularized C1 against persisted B0",
    )
    run.add_argument("--v9-summary", type=Path, default=DEFAULT_V9_SUMMARY)
    run.add_argument("--v9-predictions", type=Path, default=DEFAULT_V9_PREDICTIONS)
    run.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    run.add_argument("--evaluator", type=Path, default=DEFAULT_EVALUATOR)
    run.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    run.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    run.add_argument("--device", default="auto")
    run.add_argument("--supervised-device", default="cpu")
    run.add_argument("--supervised-cpu-threads", type=int, default=1)
    run.add_argument(
        "--runtime-projection",
        type=Path,
        default=DEFAULT_RUNTIME_PROJECTION,
    )
    run.add_argument(
        "--runtime-prototypes",
        type=Path,
        default=DEFAULT_RUNTIME_PROTOTYPES,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "run":
        command_run(args)
    else:  # pragma: no cover
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()

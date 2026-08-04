#!/usr/bin/env python3
"""Evaluate deterministic supervised multi-view readout on exact VICReg-C1."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import autoresearch_covariance_readout_v10 as i7  # noqa: E402
import autoresearch_self_supervised_v9 as v9  # noqa: E402
import autoresearch_support_aware_readout_v10 as i5  # noqa: E402


V9_RUN = ROOT / ".omc/autoresearch/elements-baseline-replacement/runs/20260803-self-supervised-v9"
V17_RUN = ROOT / ".omc/autoresearch/elements-baseline-replacement/runs/20260803-model-decision-audit-v17"
DEFAULT_V9_SUMMARY = V9_RUN / "iteration-0001/results/paired_seed_summary.json"
DEFAULT_V9_PREDICTIONS = V9_RUN / "iteration-0001/results/prediction_rows.jsonl"
DEFAULT_SPEC = V17_RUN / "specs/iteration-0002.json"
DEFAULT_EVALUATOR = V17_RUN / "evaluator-iteration-0002.json"
DEFAULT_CACHE_DIR = V9_RUN / "iteration-0001/caches"
DEFAULT_OUTPUT_DIR = V17_RUN / "iteration-0002"
DEFAULT_RUNTIME_PROJECTION = ROOT / "backend/codex_model/weights/projection.pt"
DEFAULT_RUNTIME_PROTOTYPES = ROOT / "backend/codex_model/weights/prototypes.pt"
DEFAULT_RUNTIME_CONFIG = ROOT / "backend/codex_model/config.json"

EXPECTED_SPEC_SHA256 = "51548b999afaffaa75c2d3b4fb84670aa283151a407f1c4b3ef1079899286dae"
EXPECTED_EVALUATOR_SHA256 = "2e71acc2f6d5fc0ec389d27b7a7be40d5d0ef4f78b80ad181d714e9930e6ca62"
EXPECTED_V9_RUNNER_SHA256 = "e03e1e29340a1a165ab2f82001236f5aaae4450afcee862dff8cb27ef26c48a9"
EXPECTED_I5_RUNNER_SHA256 = "acd06503d801bdaf30e1b3d8569fd6713d9b3b83e545561fc26677998a8de0c6"
EXPECTED_I7_RUNNER_SHA256 = "f8b51e9f5473a3a5306432809236680f778ecaaa6be316e17fcfae084d3bc453"
EXPECTED_RUNTIME_SHA256 = {
    "projection": "0ad6ce348eaf44c8844415a769b048b8f155017befa93f16254b1263b9241210",
    "prototypes": "ffc52f03696f03780a01302708b53e319dc24069187a6fd1a2927ea537b1cc54",
    "config": "f2150a97ec7eee50a42ac1a108ae4bc092d46a6b8e4a2d91de59349560fbf18b",
}
EXPECTED_FOLDS = [1, 2, 3, 4, 5]
EXPECTED_SEEDS = [17, 42, 73]
EXPECTED_PAIRED_ROWS = 1959
EXPECTED_CACHE_VIEWS = 8
VIEW_BALANCE_TOLERANCE = 0.05


ViewEpisode = tuple[list[int], list[int]]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    v9.write_json(path, value)


def validate_contract(spec_path: Path, evaluator_path: Path) -> dict[str, str]:
    spec_sha256 = v9.sha256_file(spec_path)
    evaluator_sha256 = v9.sha256_file(evaluator_path)
    if spec_sha256 != EXPECTED_SPEC_SHA256:
        raise ValueError("v17.2 spec SHA mismatch")
    if evaluator_sha256 != EXPECTED_EVALUATOR_SHA256:
        raise ValueError("v17.2 evaluator SHA mismatch")
    spec = read_json(spec_path)
    evaluator = read_json(evaluator_path)
    if spec.get("iteration") != 2 or evaluator.get("iteration") != 2:
        raise ValueError("v17.2 iteration contract mismatch")
    if spec.get("status") != (
        "preregistered_before_runner_implementation_and_candidate_output"
    ):
        raise ValueError("v17.2 was not prospectively preregistered")
    factor = spec.get("changed_factor", {})
    if factor.get("view_index_formula") != (
        'stable_seed("v17-multiview", fold, seed, epoch, episode_index, role, slot_position, row_id) mod 8'
    ):
        raise ValueError("v17.2 view-index formula changed")
    if factor.get("only_change") != "supervised support/query input feature source":
        raise ValueError("v17.2 changed-factor declaration changed")
    if spec.get("inference_domain", {}).get("final_class_prototypes") != (
        "base train features only"
    ):
        raise ValueError("v17.2 inference domain changed")
    boundaries = spec.get("hard_boundaries", {})
    if any(
        boundaries.get(key) is not False
        for key in (
            "new_feature_extraction",
            "new_view_generation",
            "hyperparameter_or_view_strength_sweep",
            "second_candidate",
            "final_test_read",
            "runtime_write",
            "promotion",
        )
    ):
        raise ValueError("v17.2 hard boundary changed")
    return {
        "spec_sha256": spec_sha256,
        "evaluator_sha256": evaluator_sha256,
    }


def validate_dependencies() -> dict[str, str]:
    actual = {
        "v9_runner_sha256": v9.sha256_file(Path(v9.__file__).resolve()),
        "i5_runner_sha256": v9.sha256_file(Path(i5.__file__).resolve()),
        "i7_runner_sha256": v9.sha256_file(Path(i7.__file__).resolve()),
    }
    expected = {
        "v9_runner_sha256": EXPECTED_V9_RUNNER_SHA256,
        "i5_runner_sha256": EXPECTED_I5_RUNNER_SHA256,
        "i7_runner_sha256": EXPECTED_I7_RUNNER_SHA256,
    }
    if actual != expected:
        raise ValueError(f"v17.2 dependency hash mismatch: {actual}")
    return actual


def runtime_hashes(
    projection: Path,
    prototypes: Path,
    config: Path,
) -> dict[str, str]:
    hashes = {
        "projection": v9.sha256_file(projection),
        "prototypes": v9.sha256_file(prototypes),
        "config": v9.sha256_file(config),
    }
    if hashes != EXPECTED_RUNTIME_SHA256:
        raise ValueError("runtime projection/prototype/config SHA mismatch")
    return hashes


def deterministic_view_index(
    *,
    fold: int,
    seed: int,
    epoch: int,
    episode_index: int,
    role: str,
    slot_position: int,
    row_id: str,
    views: int = EXPECTED_CACHE_VIEWS,
) -> int:
    if role not in {"support", "query"}:
        raise ValueError(f"invalid multi-view role: {role}")
    if views < 1:
        raise ValueError("view count must be positive")
    return (
        v9.stable_seed(
            "v17-multiview",
            fold,
            seed,
            epoch,
            episode_index,
            role,
            slot_position,
            row_id,
        )
        % views
    )


def usage_within_balance(
    counts: Sequence[int],
    *,
    tolerance: float = VIEW_BALANCE_TOLERANCE,
) -> tuple[bool, float]:
    if not counts or any(int(value) < 0 for value in counts):
        raise ValueError("view usage counts must be a non-empty non-negative sequence")
    total = sum(int(value) for value in counts)
    if total == 0:
        return False, math.inf
    uniform = total / len(counts)
    maximum = max(abs(int(value) - uniform) / uniform for value in counts)
    return maximum <= tolerance, float(maximum)


def build_view_plan(
    episode_plan: Sequence[Sequence[v9.Episode]],
    train_row_ids: Sequence[str],
    *,
    fold: int,
    seed: int,
    views: int = EXPECTED_CACHE_VIEWS,
    oof_row_ids: Sequence[str] = (),
) -> tuple[list[list[ViewEpisode]], dict[str, Any]]:
    row_ids = [str(value) for value in train_row_ids]
    if len(row_ids) != len(set(row_ids)):
        raise ValueError("train row IDs must be unique for deterministic view planning")
    oof_ids = {str(value) for value in oof_row_ids}
    per_view = [0] * views
    per_row = {row_id: [0] * views for row_id in row_ids}
    exposure = 0
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"fold": fold, "seed": seed, "views": views},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    plan: list[list[ViewEpisode]] = []
    for epoch, episodes in enumerate(episode_plan):
        epoch_plan: list[ViewEpisode] = []
        for episode_index, episode in enumerate(episodes):
            support_indices, _support_labels, query_indices, _query_labels = episode
            role_views: list[list[int]] = []
            for role, indices in (
                ("support", support_indices),
                ("query", query_indices),
            ):
                selected: list[int] = []
                for slot_position, row_index in enumerate(indices):
                    if row_index < 0 or row_index >= len(row_ids):
                        raise IndexError("episode row index outside train cache")
                    row_id = row_ids[row_index]
                    view_index = deterministic_view_index(
                        fold=fold,
                        seed=seed,
                        epoch=epoch,
                        episode_index=episode_index,
                        role=role,
                        slot_position=slot_position,
                        row_id=row_id,
                        views=views,
                    )
                    selected.append(view_index)
                    per_view[view_index] += 1
                    per_row[row_id][view_index] += 1
                    exposure += int(row_id in oof_ids)
                role_views.append(selected)
                digest.update(role.encode("ascii"))
                digest.update(np.asarray(selected, dtype=np.uint8).tobytes())
            epoch_plan.append((role_views[0], role_views[1]))
        plan.append(epoch_plan)
    balanced, maximum_deviation = usage_within_balance(per_view)
    return plan, {
        "view_plan_sha256": digest.hexdigest(),
        "occurrence_count": sum(per_view),
        "per_view_usage_counts": per_view,
        "per_row_view_usage_counts": per_row,
        "all_eight_views_consumed": views == 8 and all(value > 0 for value in per_view),
        "view_usage_within_five_percent_of_uniform": balanced,
        "maximum_view_usage_relative_deviation_from_uniform": maximum_deviation,
        "candidate_episode_oof_row_exposure": exposure,
    }


def train_supervised_multiview(
    model: v9.ProjectionHead,
    view_features: torch.Tensor,
    episode_plan: Sequence[Sequence[v9.Episode]],
    view_plan: Sequence[Sequence[ViewEpisode]],
    *,
    learning_rate: float,
    weight_decay: float,
    temperature: float,
    warmup_epochs: int,
    rng_seed: int,
    device: torch.device,
) -> dict[str, Any]:
    if view_features.ndim != 3 or view_features.shape[1] != EXPECTED_CACHE_VIEWS:
        raise ValueError("candidate requires an N x 8 x D cached view tensor")
    if len(episode_plan) != len(view_plan):
        raise ValueError("row episode plan and view plan epoch counts differ")
    v9.reset_training_rng(rng_seed)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    criterion = v9.PrototypicalLoss(temperature=temperature)
    features_on_device = view_features.to(device=device, dtype=torch.float32)
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
    device_view_plan = [
        tuple(
            torch.tensor(
                [episode[field_index] for episode in episodes],
                dtype=torch.long,
                device=device,
            )
            for field_index in range(2)
        )
        for episodes in view_plan
    ]
    if any(
        row_tensors[0].shape != view_tensors[0].shape
        or row_tensors[2].shape != view_tensors[1].shape
        for row_tensors, view_tensors in zip(device_episode_plan, device_view_plan)
    ):
        raise ValueError("row episode slots and view-plan slots differ")

    total_epochs = len(device_episode_plan)
    last_loss = 0.0
    last_accuracy = 0.0
    minimum_post_l2_query_batch_mean_std = math.inf
    episode_total = 0
    for epoch, (episode_tensors, view_tensors) in enumerate(
        zip(device_episode_plan, device_view_plan)
    ):
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
        epoch_loss = torch.zeros((), device=device)
        epoch_accuracy = torch.zeros((), device=device)
        episode_count = episode_tensors[0].shape[0]
        for episode_index in range(episode_count):
            support_indices, support_labels, query_indices, query_labels = (
                values[episode_index] for values in episode_tensors
            )
            support_view_indices, query_view_indices = (
                values[episode_index] for values in view_tensors
            )
            support_embeddings = model(
                features_on_device[support_indices, support_view_indices]
            )
            query_embeddings = model(
                features_on_device[query_indices, query_view_indices]
            )
            result = criterion(
                support_embeddings,
                support_labels,
                query_embeddings,
                query_labels,
            )
            optimizer.zero_grad(set_to_none=True)
            result["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            query_mean_std = float(
                torch.sqrt(
                    query_embeddings.detach().var(dim=0, unbiased=False) + 1e-4
                )
                .mean()
                .cpu()
            )
            minimum_post_l2_query_batch_mean_std = min(
                minimum_post_l2_query_batch_mean_std,
                query_mean_std,
            )
            epoch_loss += result["loss"].detach()
            epoch_accuracy += result["accuracy"].detach()
            episode_total += 1
        last_loss = float((epoch_loss / episode_count).cpu())
        last_accuracy = float((epoch_accuracy / episode_count).cpu())
    return {
        "epochs": total_epochs,
        "episodes_per_epoch": device_episode_plan[0][0].shape[0],
        "episode_total": episode_total,
        "last_loss": last_loss,
        "last_accuracy": last_accuracy,
        "minimum_post_l2_query_batch_mean_std": (
            minimum_post_l2_query_batch_mean_std
        ),
    }


def classify_verdict(
    *,
    integrity_pass: bool,
    engagement_stability_pass: bool,
    b0_anchor_pass: bool,
    c1_noninferiority_pass: bool,
    candidate_minus_c1_delta_top1: float,
    candidate_minus_c1_bootstrap_lower: float,
    candidate_minus_c1_positive_seed_count: int,
) -> tuple[str, str]:
    if not integrity_pass:
        return "invalid", "candidate_results_uninterpretable"
    if not (engagement_stability_pass and b0_anchor_pass and c1_noninferiority_pass):
        return "not_supported", "valid_but_gate_failure"
    if (
        candidate_minus_c1_delta_top1 >= 0.01
        and candidate_minus_c1_bootstrap_lower > 0.0
    ):
        return "supported_strong", "component_level_supported"
    if (
        candidate_minus_c1_delta_top1 >= 0.005
        and candidate_minus_c1_positive_seed_count >= 2
    ):
        return "supported_reference", "unresolved_at_component_level_power"
    return "neutral_preservation", "not_evidence_of_a_better_model"


def _relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


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
        "schema_version": "autoresearch-model-decision-audit-v17.multiview-checkpoint",
        "fold": fold,
        "seed": seed,
        "arm": "exact-C1-supervised-multiview",
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


def _persist_pair(
    path: Path,
    *,
    fold: int,
    seed: int,
    records: Sequence[dict[str, Any]],
    diagnostics: dict[str, Any],
    runner_sha256: str,
    cache_sha256: str,
    contract: dict[str, str],
) -> None:
    payload = {
        "schema_version": "autoresearch-model-decision-audit-v17.multiview-pair",
        "fold": fold,
        "seed": seed,
        "runner_sha256": runner_sha256,
        "cache_sha256": cache_sha256,
        "spec_sha256": contract["spec_sha256"],
        "evaluator_sha256": contract["evaluator_sha256"],
        "records": list(records),
        "diagnostics": diagnostics,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_json(temporary, payload)
    temporary.replace(path)


def _load_resumable_pair(
    path: Path,
    checkpoint_path: Path,
    view_audit_path: Path,
    *,
    fold: int,
    seed: int,
    expected_rows: int,
    runner_sha256: str,
    cache_sha256: str,
    contract: dict[str, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
    if not path.is_file() or not checkpoint_path.is_file() or not view_audit_path.is_file():
        return None
    try:
        payload = read_json(path)
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
        records = payload["records"]
        diagnostics = dict(payload["diagnostics"])
        if len(records) != expected_rows:
            return None
        if diagnostics.get("candidate_checkpoint_sha256") != v9.sha256_file(
            checkpoint_path
        ):
            return None
        if diagnostics.get("view_plan_audit_file_sha256") != v9.sha256_file(
            view_audit_path
        ):
            return None
        if not diagnostics.get("control_state_dict_matches_persisted_v9_C1"):
            return None
        if int(diagnostics.get("control_topk_match_count", -1)) != expected_rows:
            return None
        if not diagnostics.get("view_plan", {}).get(
            "view_usage_within_five_percent_of_uniform"
        ):
            return None
        diagnostics["resumed_from_pair_artifact"] = True
        return records, diagnostics
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def run_fold_seed(
    cache: dict[str, Any],
    *,
    fold: int,
    seed: int,
    persisted: dict[str, Any],
    output_dir: Path,
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
    if len(source_rows) != len(cache["oof"]["row_id"]):
        raise ValueError(f"persisted v9 OOF row count mismatch for fold={fold}, seed={seed}")

    checkpoint_path = (
        output_dir / "checkpoints" / f"fold-{fold:02d}-seed-{seed}-multiview.pt"
    )
    view_audit_path = output_dir / "view-plans" / f"fold-{fold:02d}-seed-{seed}.json"
    pair_path = output_dir / "pairs" / f"fold-{fold:02d}-seed-{seed}.json"
    resumed = _load_resumable_pair(
        pair_path,
        checkpoint_path,
        view_audit_path,
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
    expected_initial_hash = str(source_diagnostic["initial_state_sha256"])
    if initial_hash != expected_initial_hash:
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
    if ssl["plan_sha256"] != str(source_diagnostic["ssl"]["plan_sha256"]):
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
    episode_plan, episode_plan_hash = v9.build_episode_plan(
        train_labels,
        n_way=20,
        k_shot=3,
        q_queries=5,
        epochs=30,
        episodes_per_epoch=100,
        seed=episode_seed,
    )
    if episode_plan_hash != str(source_diagnostic["episode_plan_sha256"]):
        raise ValueError(f"episode plan mismatch for fold={fold}, seed={seed}")
    episode_audit = i5.audit_plan_train_only(
        episode_plan,
        cache["train"]["row_id"],
        cache["oof"]["row_id"],
    )
    if any(episode_audit.values()):
        raise ValueError(f"episode plan integrity failure: {episode_audit}")
    view_plan, view_audit = build_view_plan(
        episode_plan,
        cache["train"]["row_id"],
        fold=fold,
        seed=seed,
        views=EXPECTED_CACHE_VIEWS,
        oof_row_ids=cache["oof"]["row_id"],
    )
    if not view_audit["all_eight_views_consumed"]:
        raise ValueError(f"not all cached views consumed for fold={fold}, seed={seed}")
    if not view_audit["view_usage_within_five_percent_of_uniform"]:
        raise ValueError(f"view usage imbalance for fold={fold}, seed={seed}")
    if view_audit["candidate_episode_oof_row_exposure"] != 0:
        raise ValueError(f"OOF exposure in view plan for fold={fold}, seed={seed}")

    write_json(
        view_audit_path,
        {
            "schema_version": "autoresearch-model-decision-audit-v17.view-plan-audit",
            "fold": fold,
            "seed": seed,
            "episode_plan_sha256": episode_plan_hash,
            **view_audit,
        },
    )
    view_audit_file_sha256 = v9.sha256_file(view_audit_path)

    rng_seed = v9.stable_seed("supervised-rng", fold, seed)
    control_training = v9.train_supervised(
        control,
        cache["train"]["base_features"],
        episode_plan,
        learning_rate=1e-3,
        weight_decay=1e-4,
        temperature=0.1,
        warmup_epochs=5,
        rng_seed=rng_seed,
        device=supervised_device,
    )
    candidate_training = train_supervised_multiview(
        candidate,
        cache["train"]["view_features"],
        episode_plan,
        view_plan,
        learning_rate=1e-3,
        weight_decay=1e-4,
        temperature=0.1,
        warmup_epochs=5,
        rng_seed=rng_seed,
        device=supervised_device,
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
    source_checkpoint_file_sha256 = str(
        source_diagnostic["checkpoints"]["candidate"]["sha256"]
    )
    if v9.sha256_file(source_checkpoint_path) != source_checkpoint_file_sha256:
        raise ValueError(f"persisted C1 checkpoint SHA mismatch for fold={fold}, seed={seed}")
    _source_state, source_state_hash = i5._load_projection_state(
        source_checkpoint_path
    )
    control_state_hash = v9.state_dict_sha256(control)
    if control_state_hash != source_state_hash:
        raise ValueError(f"control state mismatch for fold={fold}, seed={seed}")

    records: list[dict[str, Any]] = []
    control_topk_match_count = 0
    for index, row_id_value in enumerate(cache["oof"]["row_id"]):
        row_id = str(row_id_value)
        scaffold = source_rows.get(row_id)
        if scaffold is None:
            raise ValueError(f"OOF row absent from persisted v9 predictions: {row_id}")
        metadata_match = (
            int(scaffold["label"]) == int(cache["oof"]["class_label"][index])
            and str(scaffold["provenance_component"])
            == str(cache["oof"]["component_id"][index])
            and str(scaffold["decoded_pixel_sha256"])
            == str(cache["oof"]["decoded_pixel_sha256"][index])
        )
        if not metadata_match:
            raise ValueError(f"OOF metadata mismatch for {row_id}")
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
                "recipe": "persisted-B0-vs-exact-C1-vs-C1-supervised-multiview",
                "baseline_topk": [int(value) for value in scaffold["baseline_topk"]],
                "control_topk": actual_control,
                "candidate_topk": [int(value) for value in candidate_topk[index]],
                "episode_plan_sha256": episode_plan_hash,
                "view_plan_sha256": view_audit["view_plan_sha256"],
                "data_sha256": cache_sha256,
            }
        )

    checkpoint_path = (
        output_dir / "checkpoints" / f"fold-{fold:02d}-seed-{seed}-multiview.pt"
    )
    checkpoint_sha256 = _save_checkpoint(
        checkpoint_path,
        candidate,
        fold=fold,
        seed=seed,
        runner_sha256=runner_sha256,
        contract=contract,
    )
    b0_rank = float(source_diagnostic["baseline_effective_rank"])
    numeric = [
        float(control_rank),
        float(candidate_rank),
        float(control_training["last_loss"]),
        float(candidate_training["last_loss"]),
        float(candidate_training["minimum_post_l2_query_batch_mean_std"]),
        float(ssl["last_loss"]),
    ]
    diagnostics = {
        "fold": fold,
        "seed": seed,
        "devices": {
            "ssl_and_prediction": str(device),
            "supervised": str(supervised_device),
            "supervised_cpu_threads": torch.get_num_threads(),
        },
        "initial_state_sha256": initial_hash,
        "persisted_v9_C1_initial_state_sha256": expected_initial_hash,
        "initial_state_matches_persisted_v9_C1": True,
        "shared_pretrained_state_sha256": shared_pretrained_hash,
        "candidate_and_control_shared_pretrained_state": True,
        "episode_plan_sha256": episode_plan_hash,
        "episode_plan_matches_persisted_v9_C1": True,
        "candidate_and_control_shared_row_episode_plan": True,
        "control_state_dict_sha256": control_state_hash,
        "persisted_v9_C1_state_dict_sha256": source_state_hash,
        "control_state_dict_matches_persisted_v9_C1": True,
        "control_topk_match_count": control_topk_match_count,
        "control_topk_matches_persisted_v9_C1": (
            control_topk_match_count == len(records)
        ),
        "candidate_and_control_equal_supervised_budget": (
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
        "candidate_checkpoint_path": _relative_path(checkpoint_path),
        "candidate_checkpoint_sha256": checkpoint_sha256,
        "candidate_checkpoint_exists": checkpoint_path.is_file(),
        "view_plan_audit_path": _relative_path(view_audit_path),
        "view_plan_audit_file_sha256": view_audit_file_sha256,
        "view_plan_persisted": view_audit_path.is_file(),
        "view_plan": view_audit,
        "ssl": ssl,
        "ssl_plan_matches_persisted_v9_C1": True,
        "control_training": control_training,
        "candidate_training": candidate_training,
        **episode_audit,
        "nan_or_nonfinite_detected": not all(math.isfinite(value) for value in numeric),
    }
    pair_path = output_dir / "pairs" / f"fold-{fold:02d}-seed-{seed}.json"
    _persist_pair(
        pair_path,
        fold=fold,
        seed=seed,
        records=records,
        diagnostics=diagnostics,
        runner_sha256=runner_sha256,
        cache_sha256=cache_sha256,
        contract=contract,
    )
    del unused_b0
    return records, diagnostics


def aggregate_results(
    records: Sequence[dict[str, Any]],
    diagnostics: Sequence[dict[str, Any]],
    cache_validations: Sequence[dict[str, Any]],
    *,
    runtime_unchanged: bool,
) -> dict[str, Any]:
    if not records or not diagnostics:
        raise ValueError("v17.2 aggregation requires records and diagnostics")
    vs_b0_records = i7.comparison_records(
        records,
        baseline_key="baseline_topk",
        candidate_key="candidate_topk",
    )
    vs_c1_records = i7.comparison_records(
        records,
        baseline_key="control_topk",
        candidate_key="candidate_topk",
    )
    vs_b0 = i7.comparison_summary(vs_b0_records, bootstrap_seed=20260803)
    vs_c1 = i7.comparison_summary(vs_c1_records, bootstrap_seed=20260804)
    seeds = sorted({int(row["seed"]) for row in records})
    folds = sorted({int(row["outer_fold"]) for row in records})

    integrity_values = {
        "contract_and_frozen_input_hashes_match": True,
        "paired_prediction_rows": len(records),
        "paired_seed_count": len(seeds),
        "outer_fold_count": len(folds),
        "candidate_checkpoint_count": sum(
            bool(item["candidate_checkpoint_exists"]) for item in diagnostics
        ),
        "initial_state_matches_persisted_c1_count": sum(
            bool(item["initial_state_matches_persisted_v9_C1"])
            for item in diagnostics
        ),
        "ssl_plan_matches_persisted_c1_count": sum(
            bool(item["ssl_plan_matches_persisted_v9_C1"])
            for item in diagnostics
        ),
        "episode_plan_matches_persisted_c1_count": sum(
            bool(item["episode_plan_matches_persisted_v9_C1"])
            for item in diagnostics
        ),
        "control_state_matches_persisted_c1_count": sum(
            bool(item["control_state_dict_matches_persisted_v9_C1"])
            for item in diagnostics
        ),
        "control_topk_matches_persisted_c1_rows": sum(
            int(item["control_topk_match_count"]) for item in diagnostics
        ),
        "candidate_and_control_shared_pretrained_state": all(
            bool(item["candidate_and_control_shared_pretrained_state"])
            for item in diagnostics
        ),
        "candidate_and_control_shared_row_episode_plan": all(
            bool(item["candidate_and_control_shared_row_episode_plan"])
            for item in diagnostics
        ),
        "candidate_and_control_equal_supervised_budget": all(
            bool(item["candidate_and_control_equal_supervised_budget"])
            for item in diagnostics
        ),
        "persisted_view_plan_count": sum(
            bool(item["view_plan_persisted"]) for item in diagnostics
        ),
        "all_eight_views_consumed_count": sum(
            bool(item["view_plan"]["all_eight_views_consumed"])
            for item in diagnostics
        ),
        "maximum_view_usage_relative_deviation_from_uniform": max(
            float(
                item["view_plan"][
                    "maximum_view_usage_relative_deviation_from_uniform"
                ]
            )
            for item in diagnostics
        ),
        "candidate_episode_oof_row_exposure": sum(
            int(item["candidate_episode_oof_row_exposure"])
            + int(item["view_plan"]["candidate_episode_oof_row_exposure"])
            for item in diagnostics
        ),
        "v9_cache_hashes_match_count": sum(
            bool(item["cache_sha256_matches"]) for item in cache_validations
        ),
        "nan_or_nonfinite_detected": any(
            bool(item["nan_or_nonfinite_detected"]) for item in diagnostics
        ),
        "runtime_unchanged": runtime_unchanged,
        "final_test_read": False,
        "automatic_promotion": False,
    }
    integrity_passes = {
        "contract_and_frozen_input_hashes_match": integrity_values[
            "contract_and_frozen_input_hashes_match"
        ]
        is True,
        "paired_prediction_rows": integrity_values["paired_prediction_rows"]
        == EXPECTED_PAIRED_ROWS,
        "paired_seed_count": integrity_values["paired_seed_count"] == 3,
        "outer_fold_count": integrity_values["outer_fold_count"] == 5,
        "candidate_checkpoint_count": integrity_values[
            "candidate_checkpoint_count"
        ]
        == 15,
        "initial_state_matches_persisted_c1_count": integrity_values[
            "initial_state_matches_persisted_c1_count"
        ]
        == 15,
        "ssl_plan_matches_persisted_c1_count": integrity_values[
            "ssl_plan_matches_persisted_c1_count"
        ]
        == 15,
        "episode_plan_matches_persisted_c1_count": integrity_values[
            "episode_plan_matches_persisted_c1_count"
        ]
        == 15,
        "control_state_matches_persisted_c1_count": integrity_values[
            "control_state_matches_persisted_c1_count"
        ]
        == 15,
        "control_topk_matches_persisted_c1_rows": integrity_values[
            "control_topk_matches_persisted_c1_rows"
        ]
        == EXPECTED_PAIRED_ROWS,
        "candidate_and_control_shared_pretrained_state": integrity_values[
            "candidate_and_control_shared_pretrained_state"
        ]
        is True,
        "candidate_and_control_shared_row_episode_plan": integrity_values[
            "candidate_and_control_shared_row_episode_plan"
        ]
        is True,
        "candidate_and_control_equal_supervised_budget": integrity_values[
            "candidate_and_control_equal_supervised_budget"
        ]
        is True,
        "persisted_view_plan_count": integrity_values["persisted_view_plan_count"]
        == 15,
        "all_eight_views_consumed_count": integrity_values[
            "all_eight_views_consumed_count"
        ]
        == 15,
        "per_view_usage_relative_deviation_from_uniform_lte": integrity_values[
            "maximum_view_usage_relative_deviation_from_uniform"
        ]
        <= VIEW_BALANCE_TOLERANCE,
        "candidate_episode_oof_row_exposure": integrity_values[
            "candidate_episode_oof_row_exposure"
        ]
        == 0,
        "v9_cache_hashes_match_count": integrity_values[
            "v9_cache_hashes_match_count"
        ]
        == 5,
        "nan_or_nonfinite_detected": integrity_values[
            "nan_or_nonfinite_detected"
        ]
        is False,
        "runtime_unchanged": integrity_values["runtime_unchanged"] is True,
        "final_test_read": integrity_values["final_test_read"] is False,
        "automatic_promotion": integrity_values["automatic_promotion"] is False,
    }

    minimum_query_std = min(
        float(item["candidate_training"]["minimum_post_l2_query_batch_mean_std"])
        for item in diagnostics
    )
    engagement_values = {
        "all_eight_views_consumed_per_fold_seed": all(
            bool(item["view_plan"]["all_eight_views_consumed"])
            for item in diagnostics
        ),
        "minimum_post_l2_query_batch_mean_std": minimum_query_std,
        "nan_or_nonfinite_detected": integrity_values["nan_or_nonfinite_detected"],
    }
    engagement_passes = {
        "all_eight_views_consumed_per_fold_seed": engagement_values[
            "all_eight_views_consumed_per_fold_seed"
        ]
        is True,
        "minimum_post_l2_query_batch_mean_std_gte": minimum_query_std >= 0.01,
        "nan_or_nonfinite_detected": engagement_values[
            "nan_or_nonfinite_detected"
        ]
        is False,
    }

    b0_values = {
        "delta_top1": float(vs_b0["overall"]["delta_top1"]),
        "delta_top1_component_bootstrap_lower_95": float(
            vs_b0["bootstrap"]["delta_top1_95"][0]
        ),
        "positive_seed_count": int(vs_b0["positive_seed_count"]),
        "delta_macro_top1": float(vs_b0["overall"]["delta_macro_top1"]),
        "delta_top3": float(vs_b0["overall"]["delta_top3"]),
        "minimum_fold_delta_top1": float(vs_b0["minimum_fold_delta_top1"]),
    }
    b0_passes = {
        "delta_top1_gte": b0_values["delta_top1"] >= 0.01,
        "delta_top1_component_bootstrap_lower_95_gt": b0_values[
            "delta_top1_component_bootstrap_lower_95"
        ]
        > 0.0,
        "positive_seed_count_gte": b0_values["positive_seed_count"] >= 2,
        "delta_macro_top1_gte": b0_values["delta_macro_top1"] >= -0.005,
        "delta_top3_gte": b0_values["delta_top3"] >= 0.0,
        "minimum_fold_delta_top1_gte": b0_values["minimum_fold_delta_top1"]
        >= -0.05,
    }

    c1_values = {
        "delta_top1": float(vs_c1["overall"]["delta_top1"]),
        "delta_top1_component_bootstrap_lower_95": float(
            vs_c1["bootstrap"]["delta_top1_95"][0]
        ),
        "positive_seed_count": int(vs_c1["positive_seed_count"]),
        "delta_top3": float(vs_c1["overall"]["delta_top3"]),
        "delta_macro_top1": float(vs_c1["overall"]["delta_macro_top1"]),
    }
    c1_passes = {
        "delta_top1_gte": c1_values["delta_top1"] >= -0.005,
        "delta_top1_component_bootstrap_lower_95_gt": c1_values[
            "delta_top1_component_bootstrap_lower_95"
        ]
        > -0.01,
        "delta_top3_gte": c1_values["delta_top3"] >= -0.01,
        "delta_macro_top1_gte": c1_values["delta_macro_top1"] >= -0.01,
    }
    integrity_ok = all(integrity_passes.values())
    engagement_ok = all(engagement_passes.values())
    b0_ok = all(b0_passes.values())
    c1_ok = all(c1_passes.values())
    verdict, claim = classify_verdict(
        integrity_pass=integrity_ok,
        engagement_stability_pass=engagement_ok,
        b0_anchor_pass=b0_ok,
        c1_noninferiority_pass=c1_ok,
        candidate_minus_c1_delta_top1=c1_values["delta_top1"],
        candidate_minus_c1_bootstrap_lower=c1_values[
            "delta_top1_component_bootstrap_lower_95"
        ],
        candidate_minus_c1_positive_seed_count=c1_values["positive_seed_count"],
    )
    minimum_rank_b0 = min(
        float(item["candidate_over_persisted_B0_effective_rank_ratio"])
        for item in diagnostics
    )
    minimum_rank_c1 = min(
        float(item["candidate_over_control_C1_effective_rank_ratio"])
        for item in diagnostics
    )
    return {
        "pass": verdict in {"supported_strong", "supported_reference"},
        "score": c1_values["delta_top1"],
        "hypothesis_supported": verdict
        in {"supported_strong", "supported_reference"},
        "decision": verdict,
        "claim": claim,
        "promotion_eligible": False,
        "comparison_vs_persisted_B0": vs_b0,
        "comparison_vs_exact_C1": vs_c1,
        "integrity_gates": integrity_values,
        "integrity_gate_passes": integrity_passes,
        "engagement_and_stability_gates": engagement_values,
        "engagement_and_stability_gate_passes": engagement_passes,
        "b0_mission_anchor_gates": b0_values,
        "b0_mission_anchor_gate_passes": b0_passes,
        "c1_noninferiority_gates": c1_values,
        "c1_noninferiority_gate_passes": c1_passes,
        "rank_diagnostic": {
            "minimum_candidate_over_persisted_B0": minimum_rank_b0,
            "minimum_candidate_over_exact_C1": minimum_rank_c1,
            "council_escalation_threshold": 0.8,
            "council_escalation_required": (
                minimum_rank_b0 < 0.8 or minimum_rank_c1 < 0.8
            ),
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
        args.runtime_config,
    )
    device = v9.resolve_device(args.device)
    supervised_device = v9.resolve_device(args.supervised_device)
    if device.type != "cuda":
        raise ValueError("v17.2 requires CUDA for SSL and prediction")
    if supervised_device.type != "cpu":
        raise ValueError("v17.2 requires CPU for supervised replay")
    if args.supervised_cpu_threads != 1:
        raise ValueError("v17.2 requires exactly one supervised CPU thread")
    torch.set_num_threads(1)

    caches: dict[int, dict[str, Any]] = {}
    cache_validations: list[dict[str, Any]] = []
    for fold in EXPECTED_FOLDS:
        cache_path = v9.cache_path(args.cache_dir, fold, EXPECTED_CACHE_VIEWS, None)
        cache, validation = i5.load_validated_cache(
            cache_path,
            fold=fold,
            views=EXPECTED_CACHE_VIEWS,
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
                output_dir=args.output_dir,
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
                "v9_summary_sha256": i5.EXPECTED_V9_SUMMARY_SHA256,
                "v9_predictions_sha256": i5.EXPECTED_V9_PREDICTIONS_SHA256,
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
        args.runtime_config,
    )
    runtime_unchanged = before_runtime == after_runtime
    if not runtime_unchanged:
        raise RuntimeError("runtime artifacts changed during v17.2")
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
            "schema_version": "autoresearch-model-decision-audit-v17.multiview-evaluation",
            "iteration": 2,
            "name": "exact-c1-supervised-multiview-readout",
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
    summary_path = args.output_dir / "summary.json"
    evaluation_path = args.output_dir / "evaluation.json"
    write_json(summary_path, result)
    write_json(evaluation_path, result)
    audit = {
        "schema_version": "autoresearch-model-decision-audit-v17.multiview-audit",
        "iteration": 2,
        "decision": result["decision"],
        "claim": result["claim"],
        "execution_integrity_pass": all(result["integrity_gate_passes"].values()),
        "summary_sha256": v9.sha256_file(summary_path),
        "evaluation_sha256": v9.sha256_file(evaluation_path),
        "prediction_rows_sha256": v9.sha256_file(predictions_path),
        "runtime_unchanged": runtime_unchanged,
        "final_test_read": False,
        "runtime_promotion": False,
    }
    write_json(args.output_dir / "audit.json", audit)
    print(
        v9.canonical_json(
            {
                "decision": result["decision"],
                "claim": result["claim"],
                "score": result["score"],
                "integrity_pass": audit["execution_integrity_pass"],
                "summary_sha256": audit["summary_sha256"],
                "prediction_rows_sha256": audit["prediction_rows_sha256"],
            }
        ),
        end="",
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser(
        "run",
        help="run exact C1 and deterministic supervised multi-view C1",
    )
    run.add_argument("--v9-summary", type=Path, default=DEFAULT_V9_SUMMARY)
    run.add_argument("--v9-predictions", type=Path, default=DEFAULT_V9_PREDICTIONS)
    run.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    run.add_argument("--evaluator", type=Path, default=DEFAULT_EVALUATOR)
    run.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    run.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    run.add_argument("--device", default="cuda")
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
    run.add_argument(
        "--runtime-config",
        type=Path,
        default=DEFAULT_RUNTIME_CONFIG,
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

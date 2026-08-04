#!/usr/bin/env python3
"""Evaluate one prototype-initialized discriminative cosine head on exact C1."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import autoresearch_covariance_readout_v10 as i7  # noqa: E402
import autoresearch_hierarchical_shrinkage_v17 as v17  # noqa: E402
import autoresearch_self_supervised_v9 as v9  # noqa: E402
import autoresearch_support_aware_readout_v10 as i5  # noqa: E402


V9_RUN = ROOT / ".omc/autoresearch/elements-baseline-replacement/runs/20260803-self-supervised-v9"
V17_RUN = ROOT / ".omc/autoresearch/elements-baseline-replacement/runs/20260803-model-decision-audit-v17"
V18_RUN = ROOT / ".omc/autoresearch/elements-baseline-replacement/runs/20260804-discriminative-readout-v18"
DEFAULT_V9_SUMMARY = V9_RUN / "iteration-0001/results/paired_seed_summary.json"
DEFAULT_V9_PREDICTIONS = V9_RUN / "iteration-0001/results/prediction_rows.jsonl"
DEFAULT_CACHE_DIR = V9_RUN / "iteration-0001/caches"
DEFAULT_SPEC = V18_RUN / "specs/iteration-0001.json"
DEFAULT_EVALUATOR = V18_RUN / "evaluator-iteration-0001.json"
DEFAULT_AMENDMENT = V17_RUN / "iteration-0003-amendment-log.json"
DEFAULT_OUTPUT_DIR = V18_RUN / "iteration-0001"
DEFAULT_RUNTIME_PROJECTION = ROOT / "backend/codex_model/weights/projection.pt"
DEFAULT_RUNTIME_PROTOTYPES = ROOT / "backend/codex_model/weights/prototypes.pt"
DEFAULT_RUNTIME_CONFIG = ROOT / "backend/codex_model/config.json"

EXPECTED_SPEC_SHA256 = "9d6d85a0fef6cdae479238f25ca464daea427af7a0c547604cd09a388403e1e6"
EXPECTED_EVALUATOR_SHA256 = "70b0093737d29c555ef30e50d26ae7badd0aac29a3553a2585f2e923a16d2ae2"
EXPECTED_AMENDMENT_SHA256 = "54c1da9b1d4fdeaac94fdf79c989237b7b78e420caba473cafc01120f52c9311"
EXPECTED_V9_RUNNER_SHA256 = "e03e1e29340a1a165ab2f82001236f5aaae4450afcee862dff8cb27ef26c48a9"
EXPECTED_I5_RUNNER_SHA256 = "acd06503d801bdaf30e1b3d8569fd6713d9b3b83e545561fc26677998a8de0c6"
EXPECTED_I7_RUNNER_SHA256 = "f8b51e9f5473a3a5306432809236680f778ecaaa6be316e17fcfae084d3bc453"
EXPECTED_V17_RUNNER_SHA256 = "a1f0fb9e0fae18b0f5fb39dbbfa93361f041980447403eeb6a12f31630bafb08"
EXPECTED_RUNTIME_SHA256 = v17.EXPECTED_RUNTIME_SHA256
EXPECTED_FOLDS = [1, 2, 3, 4, 5]
EXPECTED_SEEDS = [17, 42, 73]
EXPECTED_PAIRED_ROWS = 1959
EXPECTED_UNIQUE_OOF_ROWS = 653
EXPECTED_CACHE_VIEWS = 8
LOGIT_SCALE = 16.0
LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.0
TRAINING_STEPS = 100
REPLAY_NORMALIZATION_CONTRACT = {
    "pair_removed_keys": [],
    "schema_version": "autoresearch-discriminative-readout-v18.replay-normalization-v1",
    "summary_removed_keys": [],
}
EXPECTED_REPLAY_NORMALIZATION_SHA256 = (
    "58335c98ef3cad6e3fafbff36fcfcafc36d1d700e45865b1e36708cbde126e82"
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def replay_normalization_sha256() -> str:
    return v9.sha256_json(REPLAY_NORMALIZATION_CONTRACT)


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(v9.canonical_json(list(value.shape)).encode("utf-8"))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def validate_contract(
    spec_path: Path,
    evaluator_path: Path,
    amendment_path: Path,
) -> dict[str, Any]:
    hashes = {
        "spec_sha256": v9.sha256_file(spec_path),
        "evaluator_sha256": v9.sha256_file(evaluator_path),
        "amendment_sha256": v9.sha256_file(amendment_path),
    }
    expected = {
        "spec_sha256": EXPECTED_SPEC_SHA256,
        "evaluator_sha256": EXPECTED_EVALUATOR_SHA256,
        "amendment_sha256": EXPECTED_AMENDMENT_SHA256,
    }
    if hashes != expected:
        raise ValueError(f"v18.1 contract SHA mismatch: {hashes}")
    spec = read_json(spec_path)
    evaluator = read_json(evaluator_path)
    if spec.get("iteration") != 1 or evaluator.get("iteration") != 1:
        raise ValueError("v18.1 iteration contract mismatch")
    if spec.get("status") != (
        "preregistered_before_runner_implementation_and_any_candidate_training_or_prediction"
    ):
        raise ValueError("v18.1 was not prospectively preregistered")
    if evaluator.get("status") != (
        "frozen_before_runner_implementation_and_any_candidate_training_or_prediction"
    ):
        raise ValueError("v18.1 evaluator was not prospectively frozen")
    declared_caches = {
        int(key): str(value)
        for key, value in spec.get("frozen_inputs", {})
        .get("cache_sha256_by_fold", {})
        .items()
    }
    if declared_caches != i5.EXPECTED_CACHE_SHA256:
        raise ValueError("v18.1 cache pins do not equal canonical i5 constants")
    replay = dict(spec.get("replay_normalization", {}))
    declared_replay_sha = replay.pop("sha256", None)
    if replay != REPLAY_NORMALIZATION_CONTRACT:
        raise ValueError("v18.1 replay normalization contract changed")
    if declared_replay_sha != EXPECTED_REPLAY_NORMALIZATION_SHA256:
        raise ValueError("v18.1 replay normalization declaration changed")
    if replay_normalization_sha256() != EXPECTED_REPLAY_NORMALIZATION_SHA256:
        raise ValueError("v18.1 replay normalization implementation changed")
    if evaluator.get("integrity_gates", {}).get(
        "replay_normalization_routine_sha256"
    ) != EXPECTED_REPLAY_NORMALIZATION_SHA256:
        raise ValueError("v18.1 evaluator replay normalization hash changed")
    factor = spec.get("changed_factor", {})
    optimizer = factor.get("optimizer", {})
    expected_optimizer = {
        "name": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "betas": [0.9, 0.999],
        "epsilon": 1e-8,
        "amsgrad": False,
        "maximize": False,
        "foreach": False,
        "fused": False,
    }
    if (
        float(factor.get("scale", math.nan)) != LOGIT_SCALE
        or factor.get("bias") is not False
        or int(factor.get("steps", -1)) != TRAINING_STEPS
        or optimizer != expected_optimizer
    ):
        raise ValueError("v18.1 frozen head hyperparameters changed")
    boundaries = spec.get("hard_boundaries", {})
    forbidden_true = (
        "new_feature_extraction",
        "projection_or_backbone_training",
        "closed_factor_combination",
        "hyperparameter_sweep",
        "second_candidate",
        "final_test_read",
        "runtime_write",
        "promotion",
    )
    if any(boundaries.get(key) is not False for key in forbidden_true):
        raise ValueError("v18.1 hard boundary changed")
    return {**hashes, "spec": spec, "evaluator": evaluator}


def validate_dependencies() -> dict[str, str]:
    actual = {
        "v9_runner_sha256": v9.sha256_file(Path(v9.__file__).resolve()),
        "i5_runner_sha256": v9.sha256_file(Path(i5.__file__).resolve()),
        "i7_runner_sha256": v9.sha256_file(Path(i7.__file__).resolve()),
        "v17_runner_sha256": v9.sha256_file(Path(v17.__file__).resolve()),
    }
    expected = {
        "v9_runner_sha256": EXPECTED_V9_RUNNER_SHA256,
        "i5_runner_sha256": EXPECTED_I5_RUNNER_SHA256,
        "i7_runner_sha256": EXPECTED_I7_RUNNER_SHA256,
        "v17_runner_sha256": EXPECTED_V17_RUNNER_SHA256,
    }
    if actual != expected:
        raise ValueError(f"v18.1 dependency SHA mismatch: {actual}")
    return actual


def runtime_hashes(projection: Path, prototypes: Path, config: Path) -> dict[str, str]:
    actual = {
        "projection": v9.sha256_file(projection),
        "prototypes": v9.sha256_file(prototypes),
        "config": v9.sha256_file(config),
    }
    if actual != EXPECTED_RUNTIME_SHA256:
        raise ValueError("runtime projection/prototype/config SHA mismatch")
    return actual


def build_control_prototypes(
    train_embeddings: torch.Tensor,
    train_labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    if train_embeddings.ndim != 2 or len(train_embeddings) != len(train_labels):
        raise ValueError("train embedding/label shape mismatch")
    if len(train_embeddings) == 0 or not torch.isfinite(train_embeddings).all():
        raise ValueError("prototype construction requires finite rows")
    prototype_labels = train_labels.unique(sorted=True)
    prototypes: list[torch.Tensor] = []
    rows: list[dict[str, Any]] = []
    for label in prototype_labels.tolist():
        selected = train_embeddings[train_labels == label]
        count = int(len(selected))
        prototype = F.normalize(selected.mean(dim=0), p=2, dim=0)
        prototypes.append(prototype)
        rows.append(
            {
                "class_label": int(label),
                "train_support": count,
                "support_bin_three": v17.support_bin_three(count),
                "support_bin_binary": v17.support_bin_binary(count),
            }
        )
    result = torch.stack(prototypes)
    if not torch.isfinite(result).all():
        raise FloatingPointError("non-finite control prototype")
    return prototype_labels, result, rows


def class_balanced_row_weights(
    train_labels: torch.Tensor,
    prototype_labels: torch.Tensor,
) -> torch.Tensor:
    if train_labels.ndim != 1 or prototype_labels.ndim != 1 or len(train_labels) == 0:
        raise ValueError("class weighting requires one-dimensional non-empty labels")
    counts = Counter(int(value) for value in train_labels.tolist())
    if sorted(counts) != [int(value) for value in prototype_labels.tolist()]:
        raise ValueError("prototype labels do not cover train labels")
    row_count = len(train_labels)
    class_count = len(prototype_labels)
    weights = torch.tensor(
        [row_count / (class_count * counts[int(label)]) for label in train_labels.tolist()],
        dtype=torch.float32,
    )
    if not torch.isfinite(weights).all() or not torch.allclose(
        weights.mean(), torch.tensor(1.0), atol=1e-6, rtol=0.0
    ):
        raise FloatingPointError("class-balanced row weights are not finite mean-one")
    return weights


def target_indices(
    train_labels: torch.Tensor,
    prototype_labels: torch.Tensor,
) -> torch.Tensor:
    index = {int(label): position for position, label in enumerate(prototype_labels.tolist())}
    try:
        return torch.tensor([index[int(label)] for label in train_labels.tolist()], dtype=torch.long)
    except KeyError as error:
        raise ValueError("train target absent from prototype labels") from error


def cosine_logits(embeddings: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return LOGIT_SCALE * F.normalize(embeddings, p=2, dim=1) @ F.normalize(
        weights, p=2, dim=1
    ).T


def weighted_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    row_weights: torch.Tensor,
) -> torch.Tensor:
    losses = F.cross_entropy(logits, targets, reduction="none")
    return (losses * row_weights).mean()


def train_cosine_head(
    train_embeddings: torch.Tensor,
    train_labels: torch.Tensor,
    prototype_labels: torch.Tensor,
    initial_weights: torch.Tensor,
    *,
    device: torch.device,
    steps: int = TRAINING_STEPS,
) -> tuple[torch.Tensor, list[float], dict[str, Any]]:
    if steps != TRAINING_STEPS:
        raise ValueError("v18.1 training step count changed")
    embeddings = F.normalize(train_embeddings.float(), p=2, dim=1).to(device)
    targets = target_indices(train_labels, prototype_labels).to(device)
    row_weights = class_balanced_row_weights(train_labels, prototype_labels).to(device)
    weight = torch.nn.Parameter(initial_weights.detach().clone().float().to(device))
    initial_hash = tensor_sha256(initial_weights)
    if tensor_sha256(weight) != initial_hash:
        raise ValueError("step-zero head tensor differs from control prototypes")
    optimizer = torch.optim.AdamW(
        [weight],
        lr=LEARNING_RATE,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=WEIGHT_DECAY,
        amsgrad=False,
        maximize=False,
        foreach=False,
        fused=False,
    )
    finite = True

    def current_loss() -> torch.Tensor:
        nonlocal finite
        logits = cosine_logits(embeddings, weight)
        loss = weighted_cross_entropy(logits, targets, row_weights)
        finite = finite and bool(torch.isfinite(logits).all()) and bool(torch.isfinite(loss))
        return loss

    with torch.no_grad():
        trajectory = [float(current_loss().detach().cpu())]
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = current_loss()
        loss.backward()
        if weight.grad is None or not torch.isfinite(weight.grad).all():
            finite = False
        optimizer.step()
        if not torch.isfinite(weight).all():
            finite = False
        with torch.no_grad():
            trajectory.append(float(current_loss().detach().cpu()))
    final = weight.detach().cpu().contiguous()
    if not finite or not all(math.isfinite(value) for value in trajectory):
        raise FloatingPointError("non-finite v18.1 optimization state")
    final_hash = tensor_sha256(final)
    return final, trajectory, {
        "optimizer_step_count": steps,
        "loss_trajectory_length": len(trajectory),
        "start_loss": trajectory[0],
        "end_loss": trajectory[-1],
        "end_loss_below_start_loss": trajectory[-1] < trajectory[0],
        "initial_head_tensor_sha256": initial_hash,
        "final_head_tensor_sha256": final_hash,
        "final_head_hash_differs_from_initial": final_hash != initial_hash,
        "finite_losses_logits_gradients_and_weights": finite,
    }


def full_batch_row_order_sha256(cache: dict[str, Any]) -> str:
    rows = [
        [
            index,
            str(row_id),
            int(cache["train"]["class_label"][index]),
            str(cache["train"]["component_id"][index]),
            str(cache["train"]["decoded_pixel_sha256"][index]),
        ]
        for index, row_id in enumerate(cache["train"]["row_id"])
    ]
    return v9.sha256_json(rows)


def write_head_artifact(
    path: Path,
    *,
    fold: int,
    seed: int,
    prototype_labels: torch.Tensor,
    initial_sha256: str,
    final_weights: torch.Tensor,
) -> str:
    value = final_weights.detach().cpu().contiguous().float()
    metadata = {
        "schema_version": "autoresearch-discriminative-readout-v18.head-v1",
        "fold": fold,
        "seed": seed,
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "prototype_labels": [int(item) for item in prototype_labels.tolist()],
        "initial_head_tensor_sha256": initial_sha256,
        "final_head_tensor_sha256": tensor_sha256(value),
    }
    payload = v9.canonical_json(metadata).encode("utf-8") + value.numpy().tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return v9.sha256_file(path)


def _movement_rows(
    prototype_rows: Sequence[dict[str, Any]],
    initial_weights: torch.Tensor,
    final_weights: torch.Tensor,
) -> list[dict[str, Any]]:
    initial = F.normalize(initial_weights.float(), p=2, dim=1)
    final = F.normalize(final_weights.float(), p=2, dim=1)
    cosines = (initial * final).sum(dim=1).clamp(-1.0, 1.0)
    return [
        {
            **dict(row),
            "initial_final_cosine": float(cosines[index]),
            "angular_displacement": float(1.0 - cosines[index]),
        }
        for index, row in enumerate(prototype_rows)
    ]


def run_fold_seed(
    cache: dict[str, Any],
    *,
    fold: int,
    seed: int,
    source_diagnostic: dict[str, Any],
    source_rows: Sequence[dict[str, Any]],
    device: torch.device,
    cache_sha256: str,
    runner_sha256: str,
    contract: dict[str, Any],
    head_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    checkpoint_path, checkpoint_sha256 = v17._source_checkpoint(source_diagnostic)
    state, source_state_sha256 = i5._load_projection_state(checkpoint_path)
    model = v9.ProjectionHead(input_dim=384, embedding_dim=128)
    model.load_state_dict(state)
    model = model.to(device)
    if v9.state_dict_sha256(model) != source_state_sha256:
        raise ValueError("loaded C1 state hash mismatch")
    train_embeddings = v9.embed_in_batches(model, cache["train"]["base_features"], device)
    oof_embeddings = v9.embed_in_batches(model, cache["oof"]["base_features"], device)
    train_labels = torch.tensor(cache["train"]["class_label"], dtype=torch.long)
    labels, control_prototypes, prototype_rows = build_control_prototypes(
        train_embeddings, train_labels
    )
    control_topk = v17.predict_topk(oof_embeddings, labels, control_prototypes)
    initial_hash = tensor_sha256(control_prototypes)
    step_zero_hash = tensor_sha256(control_prototypes.detach().clone())
    if step_zero_hash != initial_hash:
        raise ValueError("step-zero head tensor hash mismatch")
    step_zero_topk = v17.predict_topk(oof_embeddings, labels, control_prototypes.detach().clone())
    if step_zero_topk != control_topk:
        raise ValueError("step-zero head predictions differ from C1 control")
    final_weights, loss_trajectory, training = train_cosine_head(
        train_embeddings,
        train_labels,
        labels,
        control_prototypes,
        device=device,
    )
    candidate_topk = v17.predict_topk(
        oof_embeddings, labels, F.normalize(final_weights, p=2, dim=1)
    )
    movement_rows = _movement_rows(prototype_rows, control_prototypes, final_weights)
    artifact_sha256 = write_head_artifact(
        head_path,
        fold=fold,
        seed=seed,
        prototype_labels=labels,
        initial_sha256=initial_hash,
        final_weights=final_weights,
    )
    source_by_row_id = {str(row["row_id"]): row for row in source_rows}
    if len(source_by_row_id) != len(source_rows):
        raise ValueError(f"duplicate source row ID for fold={fold}, seed={seed}")
    support_by_label = {
        int(item["class_label"]): int(item["train_support"]) for item in prototype_rows
    }
    records: list[dict[str, Any]] = []
    control_matches = 0
    step_zero_matches = 0
    for index, row_id_value in enumerate(cache["oof"]["row_id"]):
        row_id = str(row_id_value)
        source = source_by_row_id.get(row_id)
        if source is None:
            raise ValueError(f"persisted C1 row missing: {row_id}")
        label = int(cache["oof"]["class_label"][index])
        if (
            int(source["label"]) != label
            or str(source["provenance_component"])
            != str(cache["oof"]["component_id"][index])
            or str(source["decoded_pixel_sha256"])
            != str(cache["oof"]["decoded_pixel_sha256"][index])
        ):
            raise ValueError(f"persisted C1 metadata mismatch: {row_id}")
        actual_control = [int(value) for value in control_topk[index]]
        if actual_control != [int(value) for value in source["candidate_topk"]]:
            raise ValueError(f"exact C1 control top-k mismatch: {row_id}")
        if [int(value) for value in step_zero_topk[index]] != actual_control:
            raise ValueError(f"step-zero top-k mismatch: {row_id}")
        control_matches += 1
        step_zero_matches += 1
        train_support = support_by_label[label]
        records.append(
            {
                "row_id": row_id,
                "provenance_component": str(cache["oof"]["component_id"][index]),
                "decoded_pixel_sha256": str(cache["oof"]["decoded_pixel_sha256"][index]),
                "label": label,
                "class_name": str(cache["oof"]["class_name"][index]),
                "outer_fold": fold,
                "seed": seed,
                "train_support": train_support,
                "support_bin_three": v17.support_bin_three(train_support),
                "support_bin_binary": v17.support_bin_binary(train_support),
                "recipe": "persisted-B0-vs-exact-C1-vs-C1-prototype-initialized-class-balanced-cosine-head",
                "baseline_topk": [int(value) for value in source["baseline_topk"]],
                "control_topk": actual_control,
                "candidate_topk": [int(value) for value in candidate_topk[index]],
                "data_sha256": cache_sha256,
                "runner_sha256": runner_sha256,
                "spec_sha256": contract["spec_sha256"],
                "evaluator_sha256": contract["evaluator_sha256"],
            }
        )
    if len(records) != len(source_rows):
        raise ValueError(f"source/control row count mismatch for fold={fold}, seed={seed}")
    control_rank = v9.effective_rank(oof_embeddings)
    expected_control_rank = float(source_diagnostic["candidate_effective_rank"])
    control_rank_delta = abs(control_rank - expected_control_rank)
    if control_rank_delta > 1e-3:
        raise ValueError(f"C1 effective-rank diagnostic drift: {control_rank_delta}")
    control_head_rank = v9.effective_rank(control_prototypes)
    learned_head_rank = v9.effective_rank(F.normalize(final_weights, p=2, dim=1))
    numeric = [
        control_rank,
        control_head_rank,
        learned_head_rank,
        *loss_trajectory,
        *[float(item["initial_final_cosine"]) for item in movement_rows],
    ]
    diagnostics = {
        "fold": fold,
        "seed": seed,
        "device": str(device),
        "source_checkpoint_path": str(checkpoint_path.resolve().relative_to(ROOT)),
        "source_checkpoint_sha256": checkpoint_sha256,
        "source_checkpoint_hash_matches": True,
        "source_state_dict_sha256": source_state_sha256,
        "control_topk_match_count": control_matches,
        "control_topk_matches_persisted_v9_C1": control_matches == len(records),
        "step_zero_head_tensor_sha256": step_zero_hash,
        "step_zero_head_tensor_hash_matches_control": step_zero_hash == initial_hash,
        "step_zero_topk_match_count": step_zero_matches,
        "step_zero_topk_matches_control": step_zero_matches == len(records),
        "full_batch_row_order_sha256": full_batch_row_order_sha256(cache),
        "full_batch_row_order_hash_present": True,
        "training_operation_scope": "head_weights_only",
        "training_operation_count": TRAINING_STEPS,
        "ssl_operation_count": 0,
        "new_feature_extraction": False,
        "candidate_count": 1,
        "prototype_class_count": len(prototype_rows),
        "loss_trajectory": loss_trajectory,
        **training,
        "head_artifact": f"heads/{head_path.name}",
        "head_artifact_format": "canonical-json-header-plus-contiguous-float32-bytes-v1",
        "head_artifact_sha256": artifact_sha256,
        "head_artifact_exists": head_path.is_file(),
        "movement_rows": movement_rows,
        "nonzero_angular_head_movement": any(
            float(item["angular_displacement"]) > 0.0 for item in movement_rows
        ),
        "persisted_v9_B0_effective_rank": float(source_diagnostic["baseline_effective_rank"]),
        "control_C1_effective_rank": control_rank,
        "candidate_effective_rank": control_rank,
        "persisted_C1_effective_rank": expected_control_rank,
        "control_rank_absolute_delta_from_persisted_C1": control_rank_delta,
        "candidate_over_control_C1_effective_rank_ratio": 1.0,
        "candidate_over_persisted_B0_effective_rank_ratio": (
            control_rank / float(source_diagnostic["baseline_effective_rank"])
        ),
        "control_head_effective_rank": control_head_rank,
        "learned_head_effective_rank": learned_head_rank,
        "learned_over_control_head_effective_rank_ratio": (
            learned_head_rank / control_head_rank if control_head_rank else 0.0
        ),
        "nan_or_nonfinite_detected": not all(math.isfinite(value) for value in numeric),
    }
    return records, diagnostics


def _movement_summary(
    diagnostics: Sequence[dict[str, Any]],
    *,
    bin_key: str,
) -> dict[str, Any]:
    groups: dict[str, list[float]] = defaultdict(list)
    for item in diagnostics:
        for row in item["movement_rows"]:
            groups[str(row[bin_key])].append(float(row["initial_final_cosine"]))
    return {
        key: {
            "head_class_instances": len(values),
            "mean_initial_final_cosine": sum(values) / len(values),
            "minimum_initial_final_cosine": min(values),
            "maximum_initial_final_cosine": max(values),
            "mean_angular_displacement": sum(1.0 - value for value in values) / len(values),
        }
        for key, values in sorted(groups.items())
    }


def aggregate_results(
    records: Sequence[dict[str, Any]],
    diagnostics: Sequence[dict[str, Any]],
    cache_validations: Sequence[dict[str, Any]],
    *,
    runtime_unchanged: bool,
) -> dict[str, Any]:
    if not records or not diagnostics:
        raise ValueError("v18.1 aggregation requires records and diagnostics")
    vs_b0 = i7.comparison_summary(
        i7.comparison_records(records, baseline_key="baseline_topk", candidate_key="candidate_topk"),
        bootstrap_seed=20260804,
    )
    vs_c1 = i7.comparison_summary(
        i7.comparison_records(records, baseline_key="control_topk", candidate_key="candidate_topk"),
        bootstrap_seed=20260805,
    )
    seeds = sorted({int(row["seed"]) for row in records})
    folds = sorted({int(row["outer_fold"]) for row in records})
    integrity_values = {
        "contract_and_frozen_input_hashes_match": True,
        "v17_3_amendment_hash_matches": True,
        "spec_cache_pins_equal_canonical_i5_constants": True,
        "paired_prediction_rows": len(records),
        "unique_oof_rows": len({str(row["row_id"]) for row in records}),
        "paired_seed_count": len(seeds),
        "outer_fold_count": len(folds),
        "persisted_c1_checkpoint_hash_match_count": sum(bool(item["source_checkpoint_hash_matches"]) for item in diagnostics),
        "control_topk_matches_persisted_c1_rows": sum(int(item["control_topk_match_count"]) for item in diagnostics),
        "step_zero_head_tensor_hash_matches_control_count": sum(bool(item["step_zero_head_tensor_hash_matches_control"]) for item in diagnostics),
        "step_zero_topk_matches_control_rows": sum(int(item["step_zero_topk_match_count"]) for item in diagnostics),
        "full_batch_row_order_hash_count": sum(bool(item["full_batch_row_order_hash_present"]) for item in diagnostics),
        "optimizer_step_count": sum(int(item["optimizer_step_count"]) for item in diagnostics),
        "loss_trajectory_101_value_count": sum(int(item["loss_trajectory_length"]) == 101 for item in diagnostics),
        "learned_head_artifact_count": sum(bool(item["head_artifact_exists"]) for item in diagnostics),
        "training_operation_scope": sorted({str(item["training_operation_scope"]) for item in diagnostics}),
        "v9_cache_hashes_match_count": sum(bool(item["cache_sha256_matches"]) for item in cache_validations),
        "ssl_operation_count": sum(int(item["ssl_operation_count"]) for item in diagnostics),
        "new_feature_extraction": any(bool(item["new_feature_extraction"]) for item in diagnostics),
        "candidate_count": max(int(item["candidate_count"]) for item in diagnostics),
        "finite_losses_logits_gradients_and_weights": all(bool(item["finite_losses_logits_gradients_and_weights"]) for item in diagnostics),
        "nan_or_nonfinite_detected": any(bool(item["nan_or_nonfinite_detected"]) for item in diagnostics),
        "runtime_unchanged": runtime_unchanged,
        "final_test_read": False,
        "automatic_promotion": False,
        "replay_normalization_routine_sha256": replay_normalization_sha256(),
    }
    integrity_passes = {
        "contract_and_frozen_input_hashes_match": True,
        "v17_3_amendment_hash_matches": integrity_values["v17_3_amendment_hash_matches"],
        "spec_cache_pins_equal_canonical_i5_constants": integrity_values["spec_cache_pins_equal_canonical_i5_constants"],
        "paired_prediction_rows": integrity_values["paired_prediction_rows"] == EXPECTED_PAIRED_ROWS,
        "unique_oof_rows": integrity_values["unique_oof_rows"] == EXPECTED_UNIQUE_OOF_ROWS,
        "paired_seed_count": integrity_values["paired_seed_count"] == 3,
        "outer_fold_count": integrity_values["outer_fold_count"] == 5,
        "persisted_c1_checkpoint_hash_match_count": integrity_values["persisted_c1_checkpoint_hash_match_count"] == 15,
        "control_topk_matches_persisted_c1_rows": integrity_values["control_topk_matches_persisted_c1_rows"] == EXPECTED_PAIRED_ROWS,
        "step_zero_head_tensor_hash_matches_control_count": integrity_values["step_zero_head_tensor_hash_matches_control_count"] == 15,
        "step_zero_topk_matches_control_rows": integrity_values["step_zero_topk_matches_control_rows"] == EXPECTED_PAIRED_ROWS,
        "full_batch_row_order_hash_count": integrity_values["full_batch_row_order_hash_count"] == 15,
        "optimizer_step_count": integrity_values["optimizer_step_count"] == 1500,
        "loss_trajectory_101_value_count": integrity_values["loss_trajectory_101_value_count"] == 15,
        "learned_head_artifact_count": integrity_values["learned_head_artifact_count"] == 15,
        "training_operation_scope": integrity_values["training_operation_scope"] == ["head_weights_only"],
        "v9_cache_hashes_match_count": integrity_values["v9_cache_hashes_match_count"] == 5,
        "ssl_operation_count": integrity_values["ssl_operation_count"] == 0,
        "new_feature_extraction": integrity_values["new_feature_extraction"] is False,
        "candidate_count": integrity_values["candidate_count"] == 1,
        "finite_losses_logits_gradients_and_weights": integrity_values["finite_losses_logits_gradients_and_weights"] is True,
        "nan_or_nonfinite_detected": integrity_values["nan_or_nonfinite_detected"] is False,
        "runtime_unchanged": integrity_values["runtime_unchanged"] is True,
        "final_test_read": integrity_values["final_test_read"] is False,
        "automatic_promotion": integrity_values["automatic_promotion"] is False,
        "replay_normalization_routine_sha256_pinned": integrity_values["replay_normalization_routine_sha256"] == EXPECTED_REPLAY_NORMALIZATION_SHA256,
    }
    discordant = sum(int(row["control_topk"][0]) != int(row["candidate_topk"][0]) for row in records)
    engagement_values = {
        "end_loss_below_start_loss_count": sum(bool(item["end_loss_below_start_loss"]) for item in diagnostics),
        "final_head_hash_differs_from_initial_count": sum(bool(item["final_head_hash_differs_from_initial"]) for item in diagnostics),
        "nonzero_angular_head_movement_count": sum(bool(item["nonzero_angular_head_movement"]) for item in diagnostics),
        "candidate_vs_c1_discordant_prediction_rows": discordant,
    }
    engagement_passes = {
        "end_loss_below_start_loss_count": engagement_values["end_loss_below_start_loss_count"] == 15,
        "final_head_hash_differs_from_initial_count": engagement_values["final_head_hash_differs_from_initial_count"] == 15,
        "nonzero_angular_head_movement_count": engagement_values["nonzero_angular_head_movement_count"] == 15,
        "candidate_vs_c1_discordant_prediction_rows_gt": discordant > 0,
    }
    b0_values = {
        "delta_top1": float(vs_b0["overall"]["delta_top1"]),
        "delta_top1_component_bootstrap_lower_95": float(vs_b0["bootstrap"]["delta_top1_95"][0]),
        "positive_seed_count": int(vs_b0["positive_seed_count"]),
        "delta_macro_top1": float(vs_b0["overall"]["delta_macro_top1"]),
        "delta_top3": float(vs_b0["overall"]["delta_top3"]),
        "minimum_fold_delta_top1": float(vs_b0["minimum_fold_delta_top1"]),
    }
    b0_passes = {
        "delta_top1_gte": b0_values["delta_top1"] >= 0.01,
        "delta_top1_component_bootstrap_lower_95_gt": b0_values["delta_top1_component_bootstrap_lower_95"] > 0.0,
        "positive_seed_count_gte": b0_values["positive_seed_count"] >= 2,
        "delta_macro_top1_gte": b0_values["delta_macro_top1"] >= -0.005,
        "delta_top3_gte": b0_values["delta_top3"] >= 0.0,
        "minimum_fold_delta_top1_gte": b0_values["minimum_fold_delta_top1"] >= -0.05,
    }
    c1_values = {
        "delta_top1": float(vs_c1["overall"]["delta_top1"]),
        "delta_top1_component_bootstrap_lower_95": float(vs_c1["bootstrap"]["delta_top1_95"][0]),
        "positive_seed_count": int(vs_c1["positive_seed_count"]),
        "delta_top3": float(vs_c1["overall"]["delta_top3"]),
        "delta_macro_top1": float(vs_c1["overall"]["delta_macro_top1"]),
    }
    c1_passes = {
        "delta_top1_gte": c1_values["delta_top1"] >= -0.005,
        "delta_top1_component_bootstrap_lower_95_gt": c1_values["delta_top1_component_bootstrap_lower_95"] > -0.01,
        "delta_top3_gte": c1_values["delta_top3"] >= -0.01,
        "delta_macro_top1_gte": c1_values["delta_macro_top1"] >= -0.01,
    }
    verdict, claim = v17.classify_verdict(
        integrity_pass=all(integrity_passes.values()),
        engagement_pass=all(engagement_passes.values()),
        b0_anchor_pass=all(b0_passes.values()),
        c1_noninferiority_pass=all(c1_passes.values()),
        c1_delta_top1=c1_values["delta_top1"],
        c1_bootstrap_lower=c1_values["delta_top1_component_bootstrap_lower_95"],
        c1_positive_seed_count=c1_values["positive_seed_count"],
    )
    minimum_rank_b0 = min(float(item["candidate_over_persisted_B0_effective_rank_ratio"]) for item in diagnostics)
    minimum_rank_c1 = min(float(item["candidate_over_control_C1_effective_rank_ratio"]) for item in diagnostics)
    minimum_head_rank = min(float(item["learned_over_control_head_effective_rank_ratio"]) for item in diagnostics)
    return {
        "pass": verdict in {"supported_strong", "supported_reference"},
        "score": c1_values["delta_top1"],
        "hypothesis_supported": verdict in {"supported_strong", "supported_reference"},
        "decision": verdict,
        "claim": claim,
        "failure_interpretation": (
            "discriminative head-only adaptation did not establish a better model; close this branch"
            if verdict in {"neutral_preservation", "not_supported"}
            else None
        ),
        "promotion_eligible": False,
        "comparison_vs_persisted_B0": vs_b0,
        "comparison_vs_exact_C1": vs_c1,
        "integrity_gates": integrity_values,
        "integrity_gate_passes": integrity_passes,
        "engagement_gates": engagement_values,
        "engagement_gate_passes": engagement_passes,
        "b0_mission_anchor_gates": b0_values,
        "b0_mission_anchor_gate_passes": b0_passes,
        "c1_noninferiority_gates": c1_values,
        "c1_noninferiority_gate_passes": c1_passes,
        "support_diagnostics": {
            "three_bin_candidate_vs_c1": v17._metrics_by_group(records, group_key="support_bin_three"),
            "binary_candidate_vs_c1": v17._metrics_by_group(records, group_key="support_bin_binary"),
            "head_movement_three_bin": _movement_summary(diagnostics, bin_key="support_bin_three"),
            "head_movement_binary": _movement_summary(diagnostics, bin_key="support_bin_binary"),
            "per_class_candidate_vs_c1": v17._per_class_metrics(records),
        },
        "rank_diagnostic": {
            "minimum_candidate_over_persisted_B0_embedding_rank": minimum_rank_b0,
            "minimum_candidate_over_exact_C1_embedding_rank": minimum_rank_c1,
            "minimum_learned_over_control_head_rank": minimum_head_rank,
            "council_escalation_threshold": 0.8,
            "council_escalation_required": min(minimum_rank_b0, minimum_rank_c1, minimum_head_rank) < 0.8,
            "hard_gate": False,
        },
        "final_test_read": False,
        "runtime_promotion": False,
    }


def command_run(args: argparse.Namespace) -> dict[str, Any]:
    contract = validate_contract(args.spec, args.evaluator, args.amendment)
    dependencies = validate_dependencies()
    persisted = i5.validate_inputs(args.v9_summary, args.v9_predictions)
    before_runtime = runtime_hashes(args.runtime_projection, args.runtime_prototypes, args.runtime_config)
    device = v9.resolve_device(args.device)
    if device.type != "cuda":
        raise ValueError("v18.1 requires CUDA for exact persisted C1 training and prediction")
    v9.configure_determinism(0)
    caches: dict[int, dict[str, Any]] = {}
    cache_validations: list[dict[str, Any]] = []
    for fold in EXPECTED_FOLDS:
        cache_path = v9.cache_path(args.cache_dir, fold, EXPECTED_CACHE_VIEWS, None)
        cache, validation = i5.load_validated_cache(cache_path, fold=fold, views=EXPECTED_CACHE_VIEWS)
        caches[fold] = cache
        cache_validations.append(validation)
    runner_sha256 = v9.sha256_file(Path(__file__).resolve())
    all_records: list[dict[str, Any]] = []
    all_diagnostics: list[dict[str, Any]] = []
    for fold in EXPECTED_FOLDS:
        for seed in EXPECTED_SEEDS:
            source_rows = [
                row for row in persisted["records"]
                if int(row["outer_fold"]) == fold and int(row["seed"]) == seed
            ]
            head_path = args.output_dir / "heads" / f"fold-{fold:02d}-seed-{seed:02d}.bin"
            records, diagnostics = run_fold_seed(
                caches[fold],
                fold=fold,
                seed=seed,
                source_diagnostic=persisted["diagnostics"][(fold, seed)],
                source_rows=source_rows,
                device=device,
                cache_sha256=str(caches[fold]["_cache_validation"]["cache_sha256"]),
                runner_sha256=runner_sha256,
                contract=contract,
                head_path=head_path,
            )
            all_records.extend(records)
            all_diagnostics.append(diagnostics)
    after_runtime = runtime_hashes(args.runtime_projection, args.runtime_prototypes, args.runtime_config)
    runtime_unchanged = before_runtime == after_runtime
    if not runtime_unchanged:
        raise RuntimeError("runtime artifacts changed during v18.1")
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
            "schema_version": "autoresearch-discriminative-readout-v18.cosine-head-evaluation",
            "iteration": 1,
            "name": "prototype-initialized-class-balanced-normalized-cosine-head",
            "paired_predictions_path": predictions_path.name,
            "head_artifact_directory": "heads",
            "folds": EXPECTED_FOLDS,
            "seeds": EXPECTED_SEEDS,
            "folds_sha256": folds_sha256,
            "cache_sha256": {
                str(item["fold"]): str(item["cache_sha256"])
                for item in cache_validations
            },
            "runner_sha256": runner_sha256,
            **dependencies,
            "spec_sha256": contract["spec_sha256"],
            "evaluator_sha256": contract["evaluator_sha256"],
            "amendment_sha256": contract["amendment_sha256"],
            "runtime_checkpoint_sha256": before_runtime,
            "runtime_unchanged": runtime_unchanged,
            "replay_normalization_sha256": replay_normalization_sha256(),
            "cache_validations": cache_validations,
            "diagnostics": all_diagnostics,
        }
    )
    summary_path = args.output_dir / "summary.json"
    evaluation_path = args.output_dir / "evaluation.json"
    v9.write_json(summary_path, result)
    v9.write_json(evaluation_path, result)
    audit = {
        "schema_version": "autoresearch-discriminative-readout-v18.cosine-head-audit",
        "iteration": 1,
        "decision": result["decision"],
        "claim": result["claim"],
        "execution_integrity_pass": all(result["integrity_gate_passes"].values()),
        "engagement_pass": all(result["engagement_gate_passes"].values()),
        "summary_sha256": v9.sha256_file(summary_path),
        "evaluation_sha256": v9.sha256_file(evaluation_path),
        "prediction_rows_sha256": v9.sha256_file(predictions_path),
        "head_artifact_sha256": {
            str(item["head_artifact"]): str(item["head_artifact_sha256"])
            for item in all_diagnostics
        },
        "replay_normalization_sha256": replay_normalization_sha256(),
        "runtime_unchanged": runtime_unchanged,
        "final_test_read": False,
        "runtime_promotion": False,
    }
    v9.write_json(args.output_dir / "audit.json", audit)
    print(
        v9.canonical_json(
            {
                "decision": result["decision"],
                "claim": result["claim"],
                "score": result["score"],
                "integrity_pass": audit["execution_integrity_pass"],
                "engagement_pass": audit["engagement_pass"],
                "summary_sha256": audit["summary_sha256"],
                "prediction_rows_sha256": audit["prediction_rows_sha256"],
            }
        ),
        end="",
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--evaluator", type=Path, default=DEFAULT_EVALUATOR)
    parser.add_argument("--amendment", type=Path, default=DEFAULT_AMENDMENT)
    parser.add_argument("--v9-summary", type=Path, default=DEFAULT_V9_SUMMARY)
    parser.add_argument("--v9-predictions", type=Path, default=DEFAULT_V9_PREDICTIONS)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--runtime-projection", type=Path, default=DEFAULT_RUNTIME_PROJECTION)
    parser.add_argument("--runtime-prototypes", type=Path, default=DEFAULT_RUNTIME_PROTOTYPES)
    parser.add_argument("--runtime-config", type=Path, default=DEFAULT_RUNTIME_CONFIG)
    return parser



def main() -> None:
    command_run(build_parser().parse_args())


if __name__ == "__main__":
    main()

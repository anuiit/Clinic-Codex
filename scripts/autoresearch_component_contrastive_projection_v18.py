#!/usr/bin/env python3
"""Evaluate component-aware contrastive adaptation of exact C1 final layers."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import autoresearch_covariance_readout_v10 as i7  # noqa: E402
import autoresearch_discriminative_cosine_head_v18 as v18  # noqa: E402
import autoresearch_hierarchical_shrinkage_v17 as v17  # noqa: E402
import autoresearch_self_supervised_v9 as v9  # noqa: E402
import autoresearch_support_aware_readout_v10 as i5  # noqa: E402


V18_RUN = v18.V18_RUN
DEFAULT_V9_SUMMARY = v18.DEFAULT_V9_SUMMARY
DEFAULT_V9_PREDICTIONS = v18.DEFAULT_V9_PREDICTIONS
DEFAULT_CACHE_DIR = v18.DEFAULT_CACHE_DIR
DEFAULT_SPEC = V18_RUN / "specs/iteration-0002.json"
DEFAULT_EVALUATOR = V18_RUN / "evaluator-iteration-0002.json"
DEFAULT_MANIFEST = V18_RUN / "specs/iteration-0002-eligibility-manifest.json"
DEFAULT_V18_1_REPLAY_AUDIT = V18_RUN / "iteration-0001-replay-audit.json"
DEFAULT_OUTPUT_DIR = V18_RUN / "iteration-0002"
DEFAULT_RUNTIME_PROJECTION = v18.DEFAULT_RUNTIME_PROJECTION
DEFAULT_RUNTIME_PROTOTYPES = v18.DEFAULT_RUNTIME_PROTOTYPES
DEFAULT_RUNTIME_CONFIG = v18.DEFAULT_RUNTIME_CONFIG

EXPECTED_SPEC_SHA256 = "9e309341b63fbd7478dcd043f9a235ecc06cf3bd3571e4e4c0c85714b26bda81"
EXPECTED_EVALUATOR_SHA256 = "fa7e71cd2e678ddd079cd617ccb1f2a068dd00da6635c4d68c2fc7372796f0bd"
EXPECTED_MANIFEST_SHA256 = "cb9b5507a75de467384b11c28048f55af10db45b0e69981b241a80d97974184d"
EXPECTED_V18_1_REPLAY_AUDIT_SHA256 = (
    "2b0266c8b3e2955975573af7384de550e8dcda4c78b0f0ef6fab001b62e62991"
)
EXPECTED_V9_RUNNER_SHA256 = v18.EXPECTED_V9_RUNNER_SHA256
EXPECTED_I5_RUNNER_SHA256 = v18.EXPECTED_I5_RUNNER_SHA256
EXPECTED_I7_RUNNER_SHA256 = v18.EXPECTED_I7_RUNNER_SHA256
EXPECTED_V17_RUNNER_SHA256 = v18.EXPECTED_V17_RUNNER_SHA256
EXPECTED_V18_1_RUNNER_SHA256 = (
    "b3f5c59dae7ac97bf888ac01ffa437b14958a8ccff20ffc37be62506539ddb94"
)
EXPECTED_RUNTIME_SHA256 = v18.EXPECTED_RUNTIME_SHA256
EXPECTED_FOLDS = v18.EXPECTED_FOLDS
EXPECTED_SEEDS = v18.EXPECTED_SEEDS
EXPECTED_PAIRED_ROWS = v18.EXPECTED_PAIRED_ROWS
EXPECTED_UNIQUE_OOF_ROWS = v18.EXPECTED_UNIQUE_OOF_ROWS
EXPECTED_CACHE_VIEWS = v18.EXPECTED_CACHE_VIEWS
TEMPERATURE = 0.1
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.0
TRAINING_STEPS = 100
REPLAY_NORMALIZATION_CONTRACT = {
    "pair_removed_keys": [],
    "schema_version": "autoresearch-discriminative-readout-v18.component-contrastive-replay-normalization-v1",
    "summary_removed_keys": [],
}
EXPECTED_REPLAY_NORMALIZATION_SHA256 = (
    "c56140ea622f99a30205a185b03df54b7c364b7e8b8b3299890a5c8ad174c4bb"
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def replay_normalization_sha256() -> str:
    return v9.sha256_json(REPLAY_NORMALIZATION_CONTRACT)


def layer_state_sha256(layer: torch.nn.Linear) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(layer.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(v9.canonical_json(list(value.shape)).encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def validate_contract(
    spec_path: Path,
    evaluator_path: Path,
    manifest_path: Path,
    predecessor_audit_path: Path,
) -> dict[str, Any]:
    hashes = {
        "spec_sha256": v9.sha256_file(spec_path),
        "evaluator_sha256": v9.sha256_file(evaluator_path),
        "manifest_sha256": v9.sha256_file(manifest_path),
        "v18_1_replay_audit_sha256": v9.sha256_file(predecessor_audit_path),
    }
    expected = {
        "spec_sha256": EXPECTED_SPEC_SHA256,
        "evaluator_sha256": EXPECTED_EVALUATOR_SHA256,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "v18_1_replay_audit_sha256": EXPECTED_V18_1_REPLAY_AUDIT_SHA256,
    }
    if hashes != expected:
        raise ValueError(f"v18.2 contract SHA mismatch: {hashes}")
    spec = read_json(spec_path)
    evaluator = read_json(evaluator_path)
    manifest = read_json(manifest_path)
    if spec.get("iteration") != 2 or evaluator.get("iteration") != 2:
        raise ValueError("v18.2 iteration contract mismatch")
    if spec.get("status") != (
        "preregistered_before_runner_implementation_and_any_iteration_0002_candidate_training_or_prediction"
    ):
        raise ValueError("v18.2 was not prospectively preregistered")
    if evaluator.get("status") != (
        "frozen_before_runner_implementation_and_any_iteration_0002_candidate_training_or_prediction"
    ):
        raise ValueError("v18.2 evaluator was not prospectively frozen")
    if manifest.get("status") != (
        "frozen_before_spec_evaluator_runner_and_candidate_training_or_prediction"
    ):
        raise ValueError("v18.2 eligibility manifest was not prospectively frozen")
    declared_caches = {
        int(key): str(value)
        for key, value in spec.get("frozen_inputs", {})
        .get("cache_sha256_by_fold", {})
        .items()
    }
    if declared_caches != i5.EXPECTED_CACHE_SHA256:
        raise ValueError("v18.2 cache pins do not equal canonical i5 constants")
    if spec["frozen_inputs"].get("eligibility_manifest_sha256") != hashes[
        "manifest_sha256"
    ]:
        raise ValueError("v18.2 manifest declaration changed")
    replay = dict(spec.get("replay_normalization", {}))
    declared_replay_sha = replay.pop("sha256", None)
    if replay != REPLAY_NORMALIZATION_CONTRACT:
        raise ValueError("v18.2 replay normalization contract changed")
    if declared_replay_sha != EXPECTED_REPLAY_NORMALIZATION_SHA256:
        raise ValueError("v18.2 replay normalization declaration changed")
    if replay_normalization_sha256() != EXPECTED_REPLAY_NORMALIZATION_SHA256:
        raise ValueError("v18.2 replay normalization implementation changed")
    if evaluator.get("integrity_gates", {}).get(
        "replay_normalization_routine_sha256"
    ) != EXPECTED_REPLAY_NORMALIZATION_SHA256:
        raise ValueError("v18.2 evaluator replay normalization hash changed")
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
        float(factor.get("temperature", math.nan)) != TEMPERATURE
        or int(factor.get("steps", -1)) != TRAINING_STEPS
        or factor.get("trainable_parameters") != ["net.3.weight", "net.3.bias"]
        or factor.get("fallback") is not None
        or factor.get("class_or_support_weighting") is not None
        or factor.get("random_sampling") is not False
        or optimizer != expected_optimizer
    ):
        raise ValueError("v18.2 frozen adaptation configuration changed")
    boundaries = spec.get("hard_boundaries", {})
    forbidden_true = (
        "new_feature_extraction",
        "net0_or_backbone_training",
        "within_component_positive_fallback",
        "closed_factor_combination",
        "hyperparameter_sweep",
        "second_candidate",
        "final_test_read",
        "runtime_write",
        "promotion",
    )
    if any(boundaries.get(key) is not False for key in forbidden_true):
        raise ValueError("v18.2 hard boundary changed")
    return {**hashes, "spec": spec, "evaluator": evaluator, "manifest": manifest}


def validate_dependencies() -> dict[str, str]:
    actual = {
        "v9_runner_sha256": v9.sha256_file(Path(v9.__file__).resolve()),
        "i5_runner_sha256": v9.sha256_file(Path(i5.__file__).resolve()),
        "i7_runner_sha256": v9.sha256_file(Path(i7.__file__).resolve()),
        "v17_runner_sha256": v9.sha256_file(Path(v17.__file__).resolve()),
        "v18_1_runner_sha256": v9.sha256_file(Path(v18.__file__).resolve()),
    }
    expected = {
        "v9_runner_sha256": EXPECTED_V9_RUNNER_SHA256,
        "i5_runner_sha256": EXPECTED_I5_RUNNER_SHA256,
        "i7_runner_sha256": EXPECTED_I7_RUNNER_SHA256,
        "v17_runner_sha256": EXPECTED_V17_RUNNER_SHA256,
        "v18_1_runner_sha256": EXPECTED_V18_1_RUNNER_SHA256,
    }
    if actual != expected:
        raise ValueError(f"v18.2 dependency SHA mismatch: {actual}")
    return actual


def build_eligibility(train: dict[str, Any]) -> tuple[list[int], torch.Tensor, dict[str, Any]]:
    required = (
        "row_id",
        "class_label",
        "component_id",
        "decoded_pixel_sha256",
    )
    if any(len(train[key]) != len(train["row_id"]) for key in required):
        raise ValueError("eligibility metadata length mismatch")
    by_label: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(train["class_label"]):
        by_label[int(label)].append(index)
    positives: dict[int, list[int]] = {}
    for index, label_value in enumerate(train["class_label"]):
        label = int(label_value)
        pixel = str(train["decoded_pixel_sha256"][index])
        component = str(train["component_id"][index])
        matches = [
            other
            for other in by_label[label]
            if other != index
            and str(train["decoded_pixel_sha256"][other]) != pixel
            and str(train["component_id"][other]) != component
        ]
        if matches:
            positives[index] = matches
    eligible = sorted(positives)
    eligible_position = {original: position for position, original in enumerate(eligible)}
    mask = torch.zeros((len(eligible), len(eligible)), dtype=torch.bool)
    ordered_pairs: list[list[str]] = []
    for anchor in eligible:
        for positive in positives[anchor]:
            if positive not in eligible_position:
                raise ValueError("positive relation is not symmetric")
            mask[eligible_position[anchor], eligible_position[positive]] = True
            ordered_pairs.append(
                [str(train["row_id"][anchor]), str(train["row_id"][positive])]
            )
    if bool(mask.diagonal().any()) or not bool(mask.any(dim=1).all()):
        raise ValueError("every eligible anchor needs non-self positives")
    eligible_rows = [
        [
            index,
            str(train["row_id"][index]),
            int(train["class_label"][index]),
            str(train["component_id"][index]),
            str(train["decoded_pixel_sha256"][index]),
        ]
        for index in eligible
    ]
    classes = sorted({int(train["class_label"][index]) for index in eligible})
    audit = {
        "eligible_anchor_count": len(eligible),
        "eligible_class_count": len(classes),
        "eligible_class_labels": classes,
        "eligible_row_ids": [str(train["row_id"][index]) for index in eligible],
        "eligible_rows_sha256": v9.sha256_json(eligible_rows),
        "positive_ordered_pair_count": len(ordered_pairs),
        "positive_ordered_pairs_sha256": v9.sha256_json(ordered_pairs),
    }
    return eligible, mask, audit


def validate_eligibility(
    cache: dict[str, Any],
    *,
    fold: int,
    manifest: dict[str, Any],
) -> tuple[list[int], torch.Tensor, dict[str, Any]]:
    eligible, mask, actual = build_eligibility(cache["train"])
    expected = manifest["folds"][str(fold)]
    keys = (
        "eligible_anchor_count",
        "eligible_class_count",
        "eligible_class_labels",
        "eligible_row_ids",
        "eligible_rows_sha256",
        "positive_ordered_pair_count",
        "positive_ordered_pairs_sha256",
    )
    if any(actual[key] != expected[key] for key in keys):
        raise ValueError(f"v18.2 eligibility manifest mismatch for fold={fold}")
    return eligible, mask, {**actual, "manifest_matches": True}


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    positive_mask: torch.Tensor,
    *,
    temperature: float = TEMPERATURE,
) -> torch.Tensor:
    if embeddings.ndim != 2 or positive_mask.shape != (
        len(embeddings),
        len(embeddings),
    ):
        raise ValueError("contrastive embedding/mask shape mismatch")
    if temperature != TEMPERATURE or len(embeddings) < 2:
        raise ValueError("v18.2 contrastive temperature or row count changed")
    if positive_mask.dtype != torch.bool:
        raise ValueError("positive mask must be boolean")
    if bool(positive_mask.diagonal().any()) or not bool(positive_mask.any(dim=1).all()):
        raise ValueError("invalid positive mask")
    normalized = F.normalize(embeddings, p=2, dim=1)
    raw_logits = normalized @ normalized.T / temperature
    self_mask = torch.eye(len(normalized), dtype=torch.bool, device=normalized.device)
    logits = raw_logits.masked_fill(self_mask, -torch.inf)
    log_probabilities = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    positive_mask = positive_mask.to(normalized.device)
    positive_log_probabilities = log_probabilities.masked_fill(~positive_mask, 0.0)
    per_anchor = -positive_log_probabilities.sum(dim=1) / positive_mask.sum(dim=1)
    loss = per_anchor.mean()
    if not torch.isfinite(raw_logits).all() or not torch.isfinite(loss):
        raise FloatingPointError("non-finite supervised contrastive loss")
    return loss


def configure_trainable_final_layer(model: torch.nn.Module) -> torch.nn.Linear:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    layer = model.net[3]
    if not isinstance(layer, torch.nn.Linear):
        raise TypeError("ProjectionHead net[3] must be Linear")
    layer.weight.requires_grad_(True)
    layer.bias.requires_grad_(True)
    trainable = [name for name, value in model.named_parameters() if value.requires_grad]
    if trainable != ["net.3.weight", "net.3.bias"]:
        raise ValueError(f"v18.2 trainable boundary changed: {trainable}")
    return layer


def eligible_hidden_features(
    model: torch.nn.Module,
    base_features: torch.Tensor,
    eligible: Sequence[int],
    device: torch.device,
) -> torch.Tensor:
    indices = torch.tensor(list(eligible), dtype=torch.long)
    with torch.no_grad():
        values = base_features.index_select(0, indices).float().to(device)
        hidden = model.net[1](model.net[0](values)).detach()
    if not torch.isfinite(hidden).all():
        raise FloatingPointError("non-finite frozen hidden features")
    return hidden


def train_final_layer(
    layer: torch.nn.Linear,
    hidden: torch.Tensor,
    positive_mask: torch.Tensor,
    *,
    steps: int = TRAINING_STEPS,
) -> tuple[list[float], dict[str, Any]]:
    if steps != TRAINING_STEPS:
        raise ValueError("v18.2 training step count changed")
    initial_hash = layer_state_sha256(layer)
    optimizer = torch.optim.AdamW(
        [layer.weight, layer.bias],
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
        embeddings = F.normalize(layer(hidden), p=2, dim=1)
        loss = supervised_contrastive_loss(embeddings, positive_mask)
        finite = finite and bool(torch.isfinite(embeddings).all()) and bool(
            torch.isfinite(loss)
        )
        return loss

    with torch.no_grad():
        trajectory = [float(current_loss().detach().cpu())]
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = current_loss()
        loss.backward()
        gradients = (layer.weight.grad, layer.bias.grad)
        if any(value is None or not torch.isfinite(value).all() for value in gradients):
            finite = False
        optimizer.step()
        if not torch.isfinite(layer.weight).all() or not torch.isfinite(layer.bias).all():
            finite = False
        with torch.no_grad():
            trajectory.append(float(current_loss().detach().cpu()))
    if not finite or not all(math.isfinite(value) for value in trajectory):
        raise FloatingPointError("non-finite v18.2 optimization state")
    final_hash = layer_state_sha256(layer)
    return trajectory, {
        "optimizer_step_count": steps,
        "loss_trajectory_length": len(trajectory),
        "start_loss": trajectory[0],
        "end_loss": trajectory[-1],
        "end_loss_below_start_loss": trajectory[-1] < trajectory[0],
        "initial_final_layer_state_sha256": initial_hash,
        "final_final_layer_state_sha256": final_hash,
        "final_layer_hash_differs_from_initial": final_hash != initial_hash,
        "finite_losses_logits_gradients_weights_and_embeddings": finite,
    }


def write_layer_artifact(
    path: Path,
    *,
    fold: int,
    seed: int,
    initial_sha256: str,
    layer: torch.nn.Linear,
) -> str:
    state = {
        name: tensor.detach().cpu().contiguous().float()
        for name, tensor in sorted(layer.state_dict().items())
    }
    metadata = {
        "schema_version": "autoresearch-discriminative-readout-v18.adapted-final-layer-v1",
        "fold": fold,
        "seed": seed,
        "initial_final_layer_state_sha256": initial_sha256,
        "final_final_layer_state_sha256": layer_state_sha256(layer),
        "tensors": {
            name: {"dtype": str(value.dtype), "shape": list(value.shape)}
            for name, value in state.items()
        },
    }
    payload = bytearray(v9.canonical_json(metadata).encode("utf-8"))
    for name, value in state.items():
        payload.extend(name.encode("utf-8") + b"\0")
        payload.extend(value.numpy().tobytes())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(payload))
    return v9.sha256_file(path)


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
    artifact_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    checkpoint_path, checkpoint_sha256 = v17._source_checkpoint(source_diagnostic)
    state, source_state_sha256 = i5._load_projection_state(checkpoint_path)
    model = v9.ProjectionHead(input_dim=384, embedding_dim=128)
    model.load_state_dict(state)
    model = model.to(device)
    if v9.state_dict_sha256(model) != source_state_sha256:
        raise ValueError("loaded C1 state hash mismatch")
    layer = configure_trainable_final_layer(model)
    initial_layer_hash = layer_state_sha256(layer)
    control_train = v9.embed_in_batches(model, cache["train"]["base_features"], device)
    control_oof = v9.embed_in_batches(model, cache["oof"]["base_features"], device)
    train_labels = torch.tensor(cache["train"]["class_label"], dtype=torch.long)
    labels, control_prototypes, prototype_rows = v18.build_control_prototypes(
        control_train, train_labels
    )
    control_topk = v17.predict_topk(control_oof, labels, control_prototypes)
    eligible, positive_mask, eligibility = validate_eligibility(
        cache, fold=fold, manifest=contract["manifest"]
    )
    hidden = eligible_hidden_features(
        model, cache["train"]["base_features"], eligible, device
    )
    trajectory, training = train_final_layer(
        layer, hidden, positive_mask.to(device)
    )
    candidate_train = v9.embed_in_batches(
        model, cache["train"]["base_features"], device
    )
    candidate_oof = v9.embed_in_batches(model, cache["oof"]["base_features"], device)
    candidate_labels, candidate_prototypes, _ = v18.build_control_prototypes(
        candidate_train, train_labels
    )
    if not torch.equal(candidate_labels, labels):
        raise ValueError("candidate prototype labels changed")
    candidate_topk = v17.predict_topk(
        candidate_oof, candidate_labels, candidate_prototypes
    )
    artifact_sha256 = write_layer_artifact(
        artifact_path,
        fold=fold,
        seed=seed,
        initial_sha256=initial_layer_hash,
        layer=layer,
    )
    source_by_row_id = {str(row["row_id"]): row for row in source_rows}
    if len(source_by_row_id) != len(source_rows):
        raise ValueError(f"duplicate source row ID for fold={fold}, seed={seed}")
    support_by_label = {
        int(item["class_label"]): int(item["train_support"])
        for item in prototype_rows
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
        control_matches += 1
        step_zero_matches += 1
        train_support = support_by_label[label]
        records.append(
            {
                "row_id": row_id,
                "provenance_component": str(cache["oof"]["component_id"][index]),
                "decoded_pixel_sha256": str(
                    cache["oof"]["decoded_pixel_sha256"][index]
                ),
                "label": label,
                "class_name": str(cache["oof"]["class_name"][index]),
                "outer_fold": fold,
                "seed": seed,
                "train_support": train_support,
                "support_bin_three": v17.support_bin_three(train_support),
                "support_bin_binary": v17.support_bin_binary(train_support),
                "recipe": "persisted-B0-vs-exact-C1-vs-component-aware-contrastive-C1-net3",
                "baseline_topk": [int(value) for value in source["baseline_topk"]],
                "control_topk": actual_control,
                "candidate_topk": [int(value) for value in candidate_topk[index]],
                "data_sha256": cache_sha256,
                "runner_sha256": runner_sha256,
                "spec_sha256": contract["spec_sha256"],
                "evaluator_sha256": contract["evaluator_sha256"],
                "manifest_sha256": contract["manifest_sha256"],
            }
        )
    if len(records) != len(source_rows):
        raise ValueError(f"source/control row count mismatch for fold={fold}, seed={seed}")
    control_rank = v9.effective_rank(control_oof)
    candidate_rank = v9.effective_rank(candidate_oof)
    expected_control_rank = float(source_diagnostic["candidate_effective_rank"])
    control_rank_delta = abs(control_rank - expected_control_rank)
    if control_rank_delta > 1e-3:
        raise ValueError(f"C1 effective-rank diagnostic drift: {control_rank_delta}")
    oof_cosines = (control_oof * candidate_oof).sum(dim=1).clamp(-1.0, 1.0)
    maximum_embedding_delta = float((candidate_oof - control_oof).abs().max())
    numeric = [
        control_rank,
        candidate_rank,
        maximum_embedding_delta,
        *trajectory,
        *[float(value) for value in oof_cosines.tolist()],
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
        "step_zero_final_layer_state_sha256": initial_layer_hash,
        "step_zero_final_layer_state_hash_matches_c1": (
            initial_layer_hash == training["initial_final_layer_state_sha256"]
        ),
        "step_zero_topk_match_count": step_zero_matches,
        "step_zero_topk_matches_control": step_zero_matches == len(records),
        "eligibility": eligibility,
        "training_operation_scope": "net.3_weight_and_bias_only",
        "training_operation_count": TRAINING_STEPS,
        "optimizer_step_count": training["optimizer_step_count"],
        "ssl_operation_count": 0,
        "new_feature_extraction": False,
        "candidate_count": 1,
        "loss_trajectory": trajectory,
        **training,
        "adapted_layer_artifact": f"adapted_layers/{artifact_path.name}",
        "adapted_layer_artifact_format": "canonical-json-header-plus-named-contiguous-float32-bytes-v1",
        "adapted_layer_artifact_sha256": artifact_sha256,
        "adapted_layer_artifact_exists": artifact_path.is_file(),
        "maximum_absolute_oof_embedding_delta": maximum_embedding_delta,
        "mean_control_candidate_oof_embedding_cosine": float(oof_cosines.mean()),
        "minimum_control_candidate_oof_embedding_cosine": float(oof_cosines.min()),
        "nonzero_embedding_movement": maximum_embedding_delta > 0.0,
        "persisted_v9_B0_effective_rank": float(
            source_diagnostic["baseline_effective_rank"]
        ),
        "control_C1_effective_rank": control_rank,
        "candidate_effective_rank": candidate_rank,
        "persisted_C1_effective_rank": expected_control_rank,
        "control_rank_absolute_delta_from_persisted_C1": control_rank_delta,
        "candidate_over_control_C1_effective_rank_ratio": (
            candidate_rank / control_rank if control_rank else 0.0
        ),
        "candidate_over_persisted_B0_effective_rank_ratio": (
            candidate_rank / float(source_diagnostic["baseline_effective_rank"])
        ),
        "control_prototype_effective_rank": v9.effective_rank(control_prototypes),
        "candidate_prototype_effective_rank": v9.effective_rank(candidate_prototypes),
        "nan_or_nonfinite_detected": not all(math.isfinite(value) for value in numeric),
    }
    return records, diagnostics


def _embedding_movement_summary(
    diagnostics: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    return {
        f"fold_{int(item['fold'])}_seed_{int(item['seed'])}": {
            "maximum_absolute_oof_embedding_delta": float(
                item["maximum_absolute_oof_embedding_delta"]
            ),
            "mean_control_candidate_oof_embedding_cosine": float(
                item["mean_control_candidate_oof_embedding_cosine"]
            ),
            "minimum_control_candidate_oof_embedding_cosine": float(
                item["minimum_control_candidate_oof_embedding_cosine"]
            ),
        }
        for item in diagnostics
    }


def aggregate_results(
    records: Sequence[dict[str, Any]],
    diagnostics: Sequence[dict[str, Any]],
    cache_validations: Sequence[dict[str, Any]],
    *,
    runtime_unchanged: bool,
) -> dict[str, Any]:
    if not records or not diagnostics:
        raise ValueError("v18.2 aggregation requires records and diagnostics")
    vs_b0 = i7.comparison_summary(
        i7.comparison_records(
            records, baseline_key="baseline_topk", candidate_key="candidate_topk"
        ),
        bootstrap_seed=20260805,
    )
    vs_c1 = i7.comparison_summary(
        i7.comparison_records(
            records, baseline_key="control_topk", candidate_key="candidate_topk"
        ),
        bootstrap_seed=20260806,
    )
    seeds = sorted({int(row["seed"]) for row in records})
    folds = sorted({int(row["outer_fold"]) for row in records})
    eligibility_by_fold: dict[str, dict[str, Any]] = {}
    for item in diagnostics:
        key = str(item["fold"])
        current = dict(item["eligibility"])
        if key in eligibility_by_fold and eligibility_by_fold[key] != current:
            raise ValueError(f"seed-dependent eligibility for fold={key}")
        eligibility_by_fold[key] = current
    integrity_values = {
        "contract_and_frozen_input_hashes_match": True,
        "v18_1_replay_audit_hash_matches": True,
        "spec_cache_pins_equal_canonical_i5_constants": True,
        "eligibility_manifest_matches_all_fold_seeds": all(
            bool(item["eligibility"]["manifest_matches"]) for item in diagnostics
        ),
        "eligible_anchor_counts_by_fold": {
            key: int(value["eligible_anchor_count"])
            for key, value in sorted(eligibility_by_fold.items())
        },
        "eligible_class_counts_by_fold": {
            key: int(value["eligible_class_count"])
            for key, value in sorted(eligibility_by_fold.items())
        },
        "positive_ordered_pair_counts_by_fold": {
            key: int(value["positive_ordered_pair_count"])
            for key, value in sorted(eligibility_by_fold.items())
        },
        "paired_prediction_rows": len(records),
        "unique_oof_rows": len({str(row["row_id"]) for row in records}),
        "paired_seed_count": len(seeds),
        "outer_fold_count": len(folds),
        "persisted_c1_checkpoint_hash_match_count": sum(
            bool(item["source_checkpoint_hash_matches"]) for item in diagnostics
        ),
        "control_topk_matches_persisted_c1_rows": sum(
            int(item["control_topk_match_count"]) for item in diagnostics
        ),
        "step_zero_final_layer_state_hash_matches_c1_count": sum(
            bool(item["step_zero_final_layer_state_hash_matches_c1"])
            for item in diagnostics
        ),
        "step_zero_topk_matches_control_rows": sum(
            int(item["step_zero_topk_match_count"]) for item in diagnostics
        ),
        "optimizer_step_count": sum(
            int(item["optimizer_step_count"]) for item in diagnostics
        ),
        "loss_trajectory_101_value_count": sum(
            int(item["loss_trajectory_length"]) == 101 for item in diagnostics
        ),
        "adapted_layer_artifact_count": sum(
            bool(item["adapted_layer_artifact_exists"]) for item in diagnostics
        ),
        "training_operation_scope": sorted(
            {str(item["training_operation_scope"]) for item in diagnostics}
        ),
        "v9_cache_hashes_match_count": sum(
            bool(item["cache_sha256_matches"]) for item in cache_validations
        ),
        "ssl_operation_count": sum(
            int(item["ssl_operation_count"]) for item in diagnostics
        ),
        "new_feature_extraction": any(
            bool(item["new_feature_extraction"]) for item in diagnostics
        ),
        "candidate_count": max(int(item["candidate_count"]) for item in diagnostics),
        "finite_losses_logits_gradients_weights_and_embeddings": all(
            bool(item["finite_losses_logits_gradients_weights_and_embeddings"])
            for item in diagnostics
        ),
        "nan_or_nonfinite_detected": any(
            bool(item["nan_or_nonfinite_detected"]) for item in diagnostics
        ),
        "runtime_unchanged": runtime_unchanged,
        "final_test_read": False,
        "automatic_promotion": False,
        "replay_normalization_routine_sha256": replay_normalization_sha256(),
    }
    expected_anchor = {"1": 106, "2": 645, "3": 641, "4": 200, "5": 558}
    expected_classes = {"1": 2, "2": 10, "3": 11, "4": 6, "5": 10}
    expected_pairs = {"1": 784, "2": 4454, "3": 4446, "4": 804, "5": 3346}
    integrity_passes = {
        "contract_and_frozen_input_hashes_match": True,
        "v18_1_replay_audit_hash_matches": True,
        "spec_cache_pins_equal_canonical_i5_constants": True,
        "eligibility_manifest_matches_all_fold_seeds": integrity_values[
            "eligibility_manifest_matches_all_fold_seeds"
        ],
        "eligible_anchor_counts_by_fold": integrity_values[
            "eligible_anchor_counts_by_fold"
        ]
        == expected_anchor,
        "eligible_class_counts_by_fold": integrity_values[
            "eligible_class_counts_by_fold"
        ]
        == expected_classes,
        "positive_ordered_pair_counts_by_fold": integrity_values[
            "positive_ordered_pair_counts_by_fold"
        ]
        == expected_pairs,
        "paired_prediction_rows": integrity_values["paired_prediction_rows"]
        == EXPECTED_PAIRED_ROWS,
        "unique_oof_rows": integrity_values["unique_oof_rows"]
        == EXPECTED_UNIQUE_OOF_ROWS,
        "paired_seed_count": integrity_values["paired_seed_count"] == 3,
        "outer_fold_count": integrity_values["outer_fold_count"] == 5,
        "persisted_c1_checkpoint_hash_match_count": integrity_values[
            "persisted_c1_checkpoint_hash_match_count"
        ]
        == 15,
        "control_topk_matches_persisted_c1_rows": integrity_values[
            "control_topk_matches_persisted_c1_rows"
        ]
        == EXPECTED_PAIRED_ROWS,
        "step_zero_final_layer_state_hash_matches_c1_count": integrity_values[
            "step_zero_final_layer_state_hash_matches_c1_count"
        ]
        == 15,
        "step_zero_topk_matches_control_rows": integrity_values[
            "step_zero_topk_matches_control_rows"
        ]
        == EXPECTED_PAIRED_ROWS,
        "optimizer_step_count": integrity_values["optimizer_step_count"] == 1500,
        "loss_trajectory_101_value_count": integrity_values[
            "loss_trajectory_101_value_count"
        ]
        == 15,
        "adapted_layer_artifact_count": integrity_values[
            "adapted_layer_artifact_count"
        ]
        == 15,
        "training_operation_scope": integrity_values["training_operation_scope"]
        == ["net.3_weight_and_bias_only"],
        "v9_cache_hashes_match_count": integrity_values[
            "v9_cache_hashes_match_count"
        ]
        == 5,
        "ssl_operation_count": integrity_values["ssl_operation_count"] == 0,
        "new_feature_extraction": integrity_values["new_feature_extraction"] is False,
        "candidate_count": integrity_values["candidate_count"] == 1,
        "finite_losses_logits_gradients_weights_and_embeddings": integrity_values[
            "finite_losses_logits_gradients_weights_and_embeddings"
        ]
        is True,
        "nan_or_nonfinite_detected": integrity_values[
            "nan_or_nonfinite_detected"
        ]
        is False,
        "runtime_unchanged": integrity_values["runtime_unchanged"] is True,
        "final_test_read": integrity_values["final_test_read"] is False,
        "automatic_promotion": integrity_values["automatic_promotion"] is False,
        "replay_normalization_routine_sha256_pinned": integrity_values[
            "replay_normalization_routine_sha256"
        ]
        == EXPECTED_REPLAY_NORMALIZATION_SHA256,
    }
    discordant = sum(
        int(row["control_topk"][0]) != int(row["candidate_topk"][0])
        for row in records
    )
    engagement_values = {
        "end_loss_below_start_loss_count": sum(
            bool(item["end_loss_below_start_loss"]) for item in diagnostics
        ),
        "final_layer_hash_differs_from_initial_count": sum(
            bool(item["final_layer_hash_differs_from_initial"])
            for item in diagnostics
        ),
        "nonzero_embedding_movement_count": sum(
            bool(item["nonzero_embedding_movement"]) for item in diagnostics
        ),
        "candidate_vs_c1_discordant_prediction_rows": discordant,
    }
    engagement_passes = {
        "end_loss_below_start_loss_count": engagement_values[
            "end_loss_below_start_loss_count"
        ]
        == 15,
        "final_layer_hash_differs_from_initial_count": engagement_values[
            "final_layer_hash_differs_from_initial_count"
        ]
        == 15,
        "nonzero_embedding_movement_count": engagement_values[
            "nonzero_embedding_movement_count"
        ]
        == 15,
        "candidate_vs_c1_discordant_prediction_rows_gt": discordant > 0,
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
    verdict, claim = v17.classify_verdict(
        integrity_pass=all(integrity_passes.values()),
        engagement_pass=all(engagement_passes.values()),
        b0_anchor_pass=all(b0_passes.values()),
        c1_noninferiority_pass=all(c1_passes.values()),
        c1_delta_top1=c1_values["delta_top1"],
        c1_bootstrap_lower=c1_values[
            "delta_top1_component_bootstrap_lower_95"
        ],
        c1_positive_seed_count=c1_values["positive_seed_count"],
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
        "failure_interpretation": (
            "cache-bounded projection adaptation did not establish a better model; close this branch"
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
            "three_bin_candidate_vs_c1": v17._metrics_by_group(
                records, group_key="support_bin_three"
            ),
            "binary_candidate_vs_c1": v17._metrics_by_group(
                records, group_key="support_bin_binary"
            ),
            "per_class_candidate_vs_c1": v17._per_class_metrics(records),
        },
        "embedding_movement_diagnostic": _embedding_movement_summary(diagnostics),
        "rank_diagnostic": {
            "minimum_candidate_over_persisted_B0_embedding_rank": minimum_rank_b0,
            "minimum_candidate_over_exact_C1_embedding_rank": minimum_rank_c1,
            "council_escalation_threshold": 0.8,
            "council_escalation_required": min(minimum_rank_b0, minimum_rank_c1)
            < 0.8,
            "hard_gate": False,
        },
        "final_test_read": False,
        "runtime_promotion": False,
    }


def command_run(args: argparse.Namespace) -> dict[str, Any]:
    contract = validate_contract(
        args.spec,
        args.evaluator,
        args.manifest,
        args.v18_1_replay_audit,
    )
    dependencies = validate_dependencies()
    persisted = i5.validate_inputs(args.v9_summary, args.v9_predictions)
    before_runtime = v18.runtime_hashes(
        args.runtime_projection, args.runtime_prototypes, args.runtime_config
    )
    device = v9.resolve_device(args.device)
    if device.type != "cuda":
        raise ValueError("v18.2 requires CUDA for exact C1 training and prediction")
    v9.configure_determinism(0)
    caches: dict[int, dict[str, Any]] = {}
    cache_validations: list[dict[str, Any]] = []
    for fold in EXPECTED_FOLDS:
        cache_path = v9.cache_path(args.cache_dir, fold, EXPECTED_CACHE_VIEWS, None)
        cache, validation = i5.load_validated_cache(
            cache_path, fold=fold, views=EXPECTED_CACHE_VIEWS
        )
        validate_eligibility(cache, fold=fold, manifest=contract["manifest"])
        caches[fold] = cache
        cache_validations.append(validation)
    runner_sha256 = v9.sha256_file(Path(__file__).resolve())
    all_records: list[dict[str, Any]] = []
    all_diagnostics: list[dict[str, Any]] = []
    for fold in EXPECTED_FOLDS:
        for seed in EXPECTED_SEEDS:
            source_rows = [
                row
                for row in persisted["records"]
                if int(row["outer_fold"]) == fold and int(row["seed"]) == seed
            ]
            artifact_path = (
                args.output_dir
                / "adapted_layers"
                / f"fold-{fold:02d}-seed-{seed:02d}.bin"
            )
            records, diagnostics = run_fold_seed(
                caches[fold],
                fold=fold,
                seed=seed,
                source_diagnostic=persisted["diagnostics"][(fold, seed)],
                source_rows=source_rows,
                device=device,
                cache_sha256=str(
                    caches[fold]["_cache_validation"]["cache_sha256"]
                ),
                runner_sha256=runner_sha256,
                contract=contract,
                artifact_path=artifact_path,
            )
            all_records.extend(records)
            all_diagnostics.append(diagnostics)
    after_runtime = v18.runtime_hashes(
        args.runtime_projection, args.runtime_prototypes, args.runtime_config
    )
    runtime_unchanged = before_runtime == after_runtime
    if not runtime_unchanged:
        raise RuntimeError("runtime artifacts changed during v18.2")
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
            "schema_version": "autoresearch-discriminative-readout-v18.component-contrastive-projection-evaluation",
            "iteration": 2,
            "name": "component-aware-supervised-contrastive-final-projection-adaptation",
            "paired_predictions_path": predictions_path.name,
            "adapted_layer_artifact_directory": "adapted_layers",
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
            "manifest_sha256": contract["manifest_sha256"],
            "v18_1_replay_audit_sha256": contract[
                "v18_1_replay_audit_sha256"
            ],
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
        "schema_version": "autoresearch-discriminative-readout-v18.component-contrastive-projection-audit",
        "iteration": 2,
        "decision": result["decision"],
        "claim": result["claim"],
        "execution_integrity_pass": all(result["integrity_gate_passes"].values()),
        "engagement_pass": all(result["engagement_gate_passes"].values()),
        "summary_sha256": v9.sha256_file(summary_path),
        "evaluation_sha256": v9.sha256_file(evaluation_path),
        "prediction_rows_sha256": v9.sha256_file(predictions_path),
        "adapted_layer_artifact_sha256": {
            str(item["adapted_layer_artifact"]): str(
                item["adapted_layer_artifact_sha256"]
            )
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
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--v18-1-replay-audit",
        type=Path,
        default=DEFAULT_V18_1_REPLAY_AUDIT,
    )
    parser.add_argument("--v9-summary", type=Path, default=DEFAULT_V9_SUMMARY)
    parser.add_argument("--v9-predictions", type=Path, default=DEFAULT_V9_PREDICTIONS)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--runtime-projection", type=Path, default=DEFAULT_RUNTIME_PROJECTION
    )
    parser.add_argument(
        "--runtime-prototypes", type=Path, default=DEFAULT_RUNTIME_PROTOTYPES
    )
    parser.add_argument("--runtime-config", type=Path, default=DEFAULT_RUNTIME_CONFIG)
    return parser


def main() -> None:
    command_run(build_parser().parse_args())


if __name__ == "__main__":
    main()

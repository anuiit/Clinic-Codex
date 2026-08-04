#!/usr/bin/env python3
"""Evaluate fixed spherical prototype shrinkage on exact persisted VICReg-C1."""

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
import autoresearch_self_supervised_v9 as v9  # noqa: E402
import autoresearch_support_aware_readout_v10 as i5  # noqa: E402


V9_RUN = ROOT / ".omc/autoresearch/elements-baseline-replacement/runs/20260803-self-supervised-v9"
V17_RUN = ROOT / ".omc/autoresearch/elements-baseline-replacement/runs/20260803-model-decision-audit-v17"
DEFAULT_V9_SUMMARY = V9_RUN / "iteration-0001/results/paired_seed_summary.json"
DEFAULT_V9_PREDICTIONS = V9_RUN / "iteration-0001/results/prediction_rows.jsonl"
DEFAULT_CACHE_DIR = V9_RUN / "iteration-0001/caches"
DEFAULT_SPEC = V17_RUN / "specs/iteration-0003.json"
DEFAULT_EVALUATOR = V17_RUN / "evaluator-iteration-0003.json"
DEFAULT_ERRATUM = V17_RUN / "specs/iteration-0002-erratum.json"
DEFAULT_OUTPUT_DIR = V17_RUN / "iteration-0003"
DEFAULT_RUNTIME_PROJECTION = ROOT / "backend/codex_model/weights/projection.pt"
DEFAULT_RUNTIME_PROTOTYPES = ROOT / "backend/codex_model/weights/prototypes.pt"
DEFAULT_RUNTIME_CONFIG = ROOT / "backend/codex_model/config.json"

EXPECTED_SPEC_SHA256 = "ff1ef196554f4417f7f42ef6f0c1fe53271c22d93c199c530f5ed9742ba4e03c"
EXPECTED_EVALUATOR_SHA256 = "42482502d498ce8026c485193bba5d3db46aa73dd8cc4042756096bc00c91469"
EXPECTED_ERRATUM_SHA256 = "acd62f0692d8d5bc6315c87ed85d7b826aa0e50a55530268acd189ff57769fa3"
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
EXPECTED_UNIQUE_OOF_ROWS = 653
EXPECTED_CACHE_VIEWS = 8
SHRINKAGE_LAMBDA = 8.0
REPLAY_NORMALIZATION_CONTRACT = {
    "pair_removed_keys": [],
    "schema_version": "autoresearch-model-decision-audit-v17.replay-normalization-v1",
    "summary_removed_keys": [],
}
EXPECTED_REPLAY_NORMALIZATION_SHA256 = (
    "03b6b1b3161ea4ec0455c1bd7bc80ec30038fa1e1947941a710d472b6f62238a"
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    v9.write_json(path, value)


def replay_normalization_sha256() -> str:
    return v9.sha256_json(REPLAY_NORMALIZATION_CONTRACT)


def validate_contract(
    spec_path: Path,
    evaluator_path: Path,
    erratum_path: Path,
) -> dict[str, Any]:
    hashes = {
        "spec_sha256": v9.sha256_file(spec_path),
        "evaluator_sha256": v9.sha256_file(evaluator_path),
        "erratum_sha256": v9.sha256_file(erratum_path),
    }
    expected = {
        "spec_sha256": EXPECTED_SPEC_SHA256,
        "evaluator_sha256": EXPECTED_EVALUATOR_SHA256,
        "erratum_sha256": EXPECTED_ERRATUM_SHA256,
    }
    if hashes != expected:
        raise ValueError(f"v17.3 contract SHA mismatch: {hashes}")
    spec = read_json(spec_path)
    evaluator = read_json(evaluator_path)
    if spec.get("iteration") != 3 or evaluator.get("iteration") != 3:
        raise ValueError("v17.3 iteration contract mismatch")
    if spec.get("status") != "preregistered_before_any_iteration_0003_candidate_prediction":
        raise ValueError("v17.3 was not prospectively preregistered")
    factor = spec.get("changed_factor", {})
    if float(factor.get("lambda", math.nan)) != SHRINKAGE_LAMBDA:
        raise ValueError("v17.3 lambda changed")
    if factor.get("control_formula") != "normalize(mu_y)":
        raise ValueError("v17.3 control geometry changed")
    if factor.get("candidate_formula") != (
        "normalize((n_y * mu_y + 8 * mu_G) / (n_y + 8))"
    ):
        raise ValueError("v17.3 candidate geometry changed")
    declared_caches = {
        int(key): str(value)
        for key, value in spec.get("frozen_inputs", {})
        .get("cache_sha256_by_fold", {})
        .items()
    }
    if declared_caches != i5.EXPECTED_CACHE_SHA256:
        raise ValueError("v17.3 spec cache pins do not equal canonical i5 constants")
    replay = dict(spec.get("replay_normalization", {}))
    declared_replay_sha = replay.pop("sha256", None)
    if replay != REPLAY_NORMALIZATION_CONTRACT:
        raise ValueError("v17.3 replay normalization contract changed")
    if declared_replay_sha != EXPECTED_REPLAY_NORMALIZATION_SHA256:
        raise ValueError("v17.3 replay normalization hash declaration changed")
    if replay_normalization_sha256() != EXPECTED_REPLAY_NORMALIZATION_SHA256:
        raise ValueError("v17.3 replay normalization implementation changed")
    if evaluator.get("integrity_gates", {}).get(
        "replay_normalization_routine_sha256"
    ) != EXPECTED_REPLAY_NORMALIZATION_SHA256:
        raise ValueError("v17.3 evaluator replay hash changed")
    boundaries = spec.get("hard_boundaries", {})
    forbidden_true = (
        "new_feature_extraction",
        "supervised_or_ssl_training",
        "multiview_checkpoint_or_factor_combination",
        "architecture_change",
        "lambda_or_geometry_sweep",
        "second_candidate",
        "final_test_read",
        "runtime_write",
        "promotion",
    )
    if any(boundaries.get(key) is not False for key in forbidden_true):
        raise ValueError("v17.3 hard boundary changed")
    return {**hashes, "spec": spec, "evaluator": evaluator}


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
        raise ValueError(f"v17.3 dependency hash mismatch: {actual}")
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


def support_bin_three(count: int) -> str:
    if count <= 8:
        return "n_y_lte_8"
    if count <= 31:
        return "n_y_9_to_31"
    return "n_y_gte_32"


def support_bin_binary(count: int) -> str:
    return "n_y_lte_8" if count <= 8 else "n_y_gt_8"


def build_prototypes(
    train_embeddings: torch.Tensor,
    train_labels: torch.Tensor,
    *,
    shrinkage_lambda: float = SHRINKAGE_LAMBDA,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    if train_embeddings.ndim != 2 or len(train_embeddings) != len(train_labels):
        raise ValueError("train embedding/label shape mismatch")
    if len(train_embeddings) == 0 or shrinkage_lambda <= 0.0:
        raise ValueError("prototype construction requires rows and positive lambda")
    if not torch.isfinite(train_embeddings).all():
        raise FloatingPointError("non-finite train embedding")
    prototype_labels = train_labels.unique(sorted=True)
    global_mean = train_embeddings.mean(dim=0)
    controls: list[torch.Tensor] = []
    candidates: list[torch.Tensor] = []
    diagnostics: list[dict[str, Any]] = []
    for label in prototype_labels.tolist():
        selected = train_embeddings[train_labels == label]
        count = int(len(selected))
        class_mean = selected.mean(dim=0)
        control = F.normalize(class_mean, p=2, dim=0)
        shrunk_mean = (
            count * class_mean + shrinkage_lambda * global_mean
        ) / (count + shrinkage_lambda)
        candidate = F.normalize(shrunk_mean, p=2, dim=0)
        cosine = float(torch.dot(control, candidate).clamp(-1.0, 1.0))
        max_abs_delta = float((candidate - control).abs().max())
        controls.append(control)
        candidates.append(candidate)
        diagnostics.append(
            {
                "class_label": int(label),
                "train_support": count,
                "support_bin_three": support_bin_three(count),
                "support_bin_binary": support_bin_binary(count),
                "control_candidate_cosine": cosine,
                "maximum_absolute_prototype_delta": max_abs_delta,
                "prototype_changed": max_abs_delta > 0.0,
            }
        )
    control_tensor = torch.stack(controls)
    candidate_tensor = torch.stack(candidates)
    if not torch.isfinite(control_tensor).all() or not torch.isfinite(candidate_tensor).all():
        raise FloatingPointError("non-finite prototype")
    return prototype_labels, control_tensor, candidate_tensor, diagnostics


def predict_topk(
    oof_embeddings: torch.Tensor,
    prototype_labels: torch.Tensor,
    prototypes: torch.Tensor,
) -> list[list[int]]:
    if not torch.isfinite(oof_embeddings).all():
        raise FloatingPointError("non-finite OOF embedding")
    logits = oof_embeddings @ prototypes.T
    top_indices = logits.topk(k=min(3, len(prototype_labels)), dim=1).indices
    top_labels = prototype_labels[top_indices]
    return [[int(value) for value in row] for row in top_labels.tolist()]


def classify_verdict(
    *,
    integrity_pass: bool,
    engagement_pass: bool,
    b0_anchor_pass: bool,
    c1_noninferiority_pass: bool,
    c1_delta_top1: float,
    c1_bootstrap_lower: float,
    c1_positive_seed_count: int,
) -> tuple[str, str]:
    if not integrity_pass:
        return "invalid", "candidate_results_uninterpretable"
    if not (engagement_pass and b0_anchor_pass and c1_noninferiority_pass):
        return "not_supported", "valid_but_gate_failure"
    if c1_delta_top1 >= 0.01 and c1_bootstrap_lower > 0.0:
        return "supported_strong", "component_level_supported"
    if c1_delta_top1 >= 0.005 and c1_positive_seed_count >= 2:
        return "supported_reference", "unresolved_at_component_level_power"
    return "neutral_preservation", "not_evidence_of_a_better_model"


def _source_checkpoint(source_diagnostic: dict[str, Any]) -> tuple[Path, str]:
    path = i5._resolve_source_checkpoint(source_diagnostic)
    expected_sha = str(source_diagnostic["checkpoints"]["candidate"]["sha256"])
    if v9.sha256_file(path) != expected_sha:
        raise ValueError(f"persisted C1 checkpoint SHA mismatch: {path}")
    return path, expected_sha


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
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    checkpoint_path, checkpoint_sha256 = _source_checkpoint(source_diagnostic)
    state, source_state_sha256 = i5._load_projection_state(checkpoint_path)
    model = v9.ProjectionHead(input_dim=384, embedding_dim=128)
    model.load_state_dict(state)
    model = model.to(device)
    if v9.state_dict_sha256(model) != source_state_sha256:
        raise ValueError("loaded C1 state hash mismatch")
    train_embeddings = v9.embed_in_batches(
        model, cache["train"]["base_features"], device
    )
    oof_embeddings = v9.embed_in_batches(model, cache["oof"]["base_features"], device)
    train_labels = torch.tensor(cache["train"]["class_label"], dtype=torch.long)
    labels, control_prototypes, candidate_prototypes, prototype_rows = (
        build_prototypes(train_embeddings, train_labels)
    )
    control_topk = predict_topk(oof_embeddings, labels, control_prototypes)
    candidate_topk = predict_topk(oof_embeddings, labels, candidate_prototypes)
    source_by_row_id = {str(row["row_id"]): row for row in source_rows}
    if len(source_by_row_id) != len(source_rows):
        raise ValueError(f"duplicate source row ID for fold={fold}, seed={seed}")
    support_by_label = {
        int(item["class_label"]): int(item["train_support"])
        for item in prototype_rows
    }
    records: list[dict[str, Any]] = []
    control_matches = 0
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
                "support_bin_three": support_bin_three(train_support),
                "support_bin_binary": support_bin_binary(train_support),
                "recipe": "persisted-B0-vs-exact-C1-vs-C1-lambda8-global-centroid-shrinkage",
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
    control_rank_absolute_delta = abs(control_rank - expected_control_rank)
    if control_rank_absolute_delta > 1e-3:
        raise ValueError(
            f"C1 effective-rank diagnostic drift: {control_rank_absolute_delta}"
        )
    control_prototype_rank = v9.effective_rank(control_prototypes)
    candidate_prototype_rank = v9.effective_rank(candidate_prototypes)
    numeric = [
        control_rank,
        control_prototype_rank,
        candidate_prototype_rank,
        *[float(item["control_candidate_cosine"]) for item in prototype_rows],
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
        "training_operation_count": 0,
        "ssl_operation_count": 0,
        "new_feature_extraction": False,
        "candidate_count": 1,
        "lambda": SHRINKAGE_LAMBDA,
        "prototype_class_count": len(prototype_rows),
        "candidate_prototype_changed_class_count": sum(
            bool(item["prototype_changed"]) for item in prototype_rows
        ),
        "prototype_rows": prototype_rows,
        "persisted_v9_B0_effective_rank": float(
            source_diagnostic["baseline_effective_rank"]
        ),
        "control_C1_effective_rank": control_rank,
        "candidate_effective_rank": control_rank,
        "persisted_C1_effective_rank": expected_control_rank,
        "control_rank_absolute_delta_from_persisted_C1": control_rank_absolute_delta,
        "candidate_over_control_C1_effective_rank_ratio": 1.0,
        "candidate_over_persisted_B0_effective_rank_ratio": (
            control_rank / float(source_diagnostic["baseline_effective_rank"])
        ),
        "control_prototype_effective_rank": control_prototype_rank,
        "candidate_prototype_effective_rank": candidate_prototype_rank,
        "candidate_over_control_prototype_effective_rank_ratio": (
            candidate_prototype_rank / control_prototype_rank
            if control_prototype_rank
            else 0.0
        ),
        "nan_or_nonfinite_detected": not all(math.isfinite(value) for value in numeric),
    }
    return records, diagnostics


def _metrics_by_group(
    records: Sequence[dict[str, Any]],
    *,
    group_key: str,
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[str(row[group_key])].append(row)
    return {
        key: {
            "records": len(values),
            **v9.accuracy_metrics(
                i7.comparison_records(
                    values,
                    baseline_key="control_topk",
                    candidate_key="candidate_topk",
                )
            ),
        }
        for key, values in sorted(groups.items())
    }


def _per_class_metrics(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[(int(row["label"]), str(row["class_name"]))].append(row)
    result = []
    for (label, name), values in sorted(groups.items()):
        supports = sorted({int(row["train_support"]) for row in values})
        comparison = i7.comparison_records(
            values,
            baseline_key="control_topk",
            candidate_key="candidate_topk",
        )
        result.append(
            {
                "class_label": label,
                "class_name": name,
                "records": len(values),
                "train_support_values": supports,
                **v9.accuracy_metrics(comparison),
            }
        )
    return result


def _prototype_shift_summary(
    diagnostics: Sequence[dict[str, Any]],
    *,
    bin_key: str,
) -> dict[str, Any]:
    groups: dict[str, list[float]] = defaultdict(list)
    for item in diagnostics:
        for row in item["prototype_rows"]:
            groups[str(row[bin_key])].append(float(row["control_candidate_cosine"]))
    return {
        key: {
            "prototype_instances": len(values),
            "mean_control_candidate_cosine": sum(values) / len(values),
            "minimum_control_candidate_cosine": min(values),
            "maximum_control_candidate_cosine": max(values),
            "mean_angular_displacement": sum(1.0 - value for value in values)
            / len(values),
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
        raise ValueError("v17.3 aggregation requires records and diagnostics")
    vs_b0 = i7.comparison_summary(
        i7.comparison_records(
            records, baseline_key="baseline_topk", candidate_key="candidate_topk"
        ),
        bootstrap_seed=20260803,
    )
    vs_c1 = i7.comparison_summary(
        i7.comparison_records(
            records, baseline_key="control_topk", candidate_key="candidate_topk"
        ),
        bootstrap_seed=20260804,
    )
    seeds = sorted({int(row["seed"]) for row in records})
    folds = sorted({int(row["outer_fold"]) for row in records})
    integrity_values = {
        "contract_and_frozen_input_hashes_match": True,
        "v17_2_erratum_hash_matches": True,
        "spec_cache_pins_equal_canonical_i5_constants": True,
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
        "v9_cache_hashes_match_count": sum(
            bool(item["cache_sha256_matches"]) for item in cache_validations
        ),
        "training_operation_count": sum(
            int(item["training_operation_count"]) for item in diagnostics
        ),
        "ssl_operation_count": sum(
            int(item["ssl_operation_count"]) for item in diagnostics
        ),
        "new_feature_extraction": any(
            bool(item["new_feature_extraction"]) for item in diagnostics
        ),
        "candidate_count": max(int(item["candidate_count"]) for item in diagnostics),
        "lambda": max(float(item["lambda"]) for item in diagnostics),
        "geometry_matches_spec": True,
        "nan_or_nonfinite_detected": any(
            bool(item["nan_or_nonfinite_detected"]) for item in diagnostics
        ),
        "runtime_unchanged": runtime_unchanged,
        "final_test_read": False,
        "automatic_promotion": False,
        "replay_normalization_routine_sha256": replay_normalization_sha256(),
    }
    integrity_passes = {
        "contract_and_frozen_input_hashes_match": True,
        "v17_2_erratum_hash_matches": integrity_values[
            "v17_2_erratum_hash_matches"
        ],
        "spec_cache_pins_equal_canonical_i5_constants": integrity_values[
            "spec_cache_pins_equal_canonical_i5_constants"
        ],
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
        "v9_cache_hashes_match_count": integrity_values[
            "v9_cache_hashes_match_count"
        ]
        == 5,
        "training_operation_count": integrity_values["training_operation_count"] == 0,
        "ssl_operation_count": integrity_values["ssl_operation_count"] == 0,
        "new_feature_extraction": integrity_values["new_feature_extraction"] is False,
        "candidate_count": integrity_values["candidate_count"] == 1,
        "lambda": integrity_values["lambda"] == SHRINKAGE_LAMBDA,
        "geometry_matches_spec": integrity_values["geometry_matches_spec"] is True,
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
    candidate_changed = sum(
        int(item["candidate_prototype_changed_class_count"])
        for item in diagnostics
    )
    top1_discordant = sum(
        int(row["control_topk"][0]) != int(row["candidate_topk"][0])
        for row in records
    )
    engagement_values = {
        "candidate_prototype_changed_class_count": candidate_changed,
        "candidate_vs_c1_discordant_prediction_rows": top1_discordant,
    }
    engagement_passes = {
        "candidate_prototype_changed_class_count_gt": candidate_changed > 0,
        "candidate_vs_c1_discordant_prediction_rows_gt": top1_discordant > 0,
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
    verdict, claim = classify_verdict(
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
    three_bin_metrics = _metrics_by_group(records, group_key="support_bin_three")
    binary_metrics = _metrics_by_group(records, group_key="support_bin_binary")
    low_support_regressed = (
        three_bin_metrics.get("n_y_lte_8", {}).get("delta_top1", 0.0) < 0.0
    )
    if low_support_regressed and verdict not in {"supported_strong", "supported_reference"}:
        failure_interpretation = (
            "shrinkage cost exceeded its variance benefit on minimal-support classes"
        )
    else:
        failure_interpretation = None
    minimum_rank_b0 = min(
        float(item["candidate_over_persisted_B0_effective_rank_ratio"])
        for item in diagnostics
    )
    minimum_rank_c1 = min(
        float(item["candidate_over_control_C1_effective_rank_ratio"])
        for item in diagnostics
    )
    minimum_prototype_rank = min(
        float(item["candidate_over_control_prototype_effective_rank_ratio"])
        for item in diagnostics
    )
    return {
        "pass": verdict in {"supported_strong", "supported_reference"},
        "score": c1_values["delta_top1"],
        "hypothesis_supported": verdict
        in {"supported_strong", "supported_reference"},
        "decision": verdict,
        "claim": claim,
        "failure_interpretation": failure_interpretation,
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
            "three_bin_candidate_vs_c1": three_bin_metrics,
            "binary_candidate_vs_c1": binary_metrics,
            "prototype_shift_three_bin": _prototype_shift_summary(
                diagnostics, bin_key="support_bin_three"
            ),
            "prototype_shift_binary": _prototype_shift_summary(
                diagnostics, bin_key="support_bin_binary"
            ),
            "per_class_candidate_vs_c1": _per_class_metrics(records),
        },
        "rank_diagnostic": {
            "minimum_candidate_over_persisted_B0_embedding_rank": minimum_rank_b0,
            "minimum_candidate_over_exact_C1_embedding_rank": minimum_rank_c1,
            "minimum_candidate_over_control_prototype_rank": minimum_prototype_rank,
            "council_escalation_threshold": 0.8,
            "council_escalation_required": (
                minimum_rank_b0 < 0.8
                or minimum_rank_c1 < 0.8
                or minimum_prototype_rank < 0.8
            ),
            "hard_gate": False,
        },
        "final_test_read": False,
        "runtime_promotion": False,
    }


def command_run(args: argparse.Namespace) -> dict[str, Any]:
    contract = validate_contract(args.spec, args.evaluator, args.erratum)
    dependencies = validate_dependencies()
    persisted = i5.validate_inputs(args.v9_summary, args.v9_predictions)
    before_runtime = runtime_hashes(
        args.runtime_projection, args.runtime_prototypes, args.runtime_config
    )
    device = v9.resolve_device(args.device)
    if device.type != "cuda":
        raise ValueError("v17.3 requires CUDA for exact persisted C1 prediction")
    v9.configure_determinism(0)
    caches: dict[int, dict[str, Any]] = {}
    cache_validations: list[dict[str, Any]] = []
    for fold in EXPECTED_FOLDS:
        cache_path = v9.cache_path(args.cache_dir, fold, EXPECTED_CACHE_VIEWS, None)
        cache, validation = i5.load_validated_cache(
            cache_path, fold=fold, views=EXPECTED_CACHE_VIEWS
        )
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
            )
            all_records.extend(records)
            all_diagnostics.append(diagnostics)
    after_runtime = runtime_hashes(
        args.runtime_projection, args.runtime_prototypes, args.runtime_config
    )
    runtime_unchanged = before_runtime == after_runtime
    if not runtime_unchanged:
        raise RuntimeError("runtime artifacts changed during v17.3")
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
            "schema_version": "autoresearch-model-decision-audit-v17.hierarchical-shrinkage-evaluation",
            "iteration": 3,
            "name": "exact-c1-lambda8-global-centroid-prototype-shrinkage",
            "paired_predictions_path": predictions_path.name,
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
            "erratum_sha256": contract["erratum_sha256"],
            "runtime_checkpoint_sha256": before_runtime,
            "runtime_unchanged": runtime_unchanged,
            "replay_normalization_sha256": replay_normalization_sha256(),
            "cache_validations": cache_validations,
            "diagnostics": all_diagnostics,
        }
    )
    summary_path = args.output_dir / "summary.json"
    evaluation_path = args.output_dir / "evaluation.json"
    write_json(summary_path, result)
    write_json(evaluation_path, result)
    audit = {
        "schema_version": "autoresearch-model-decision-audit-v17.hierarchical-shrinkage-audit",
        "iteration": 3,
        "decision": result["decision"],
        "claim": result["claim"],
        "execution_integrity_pass": all(result["integrity_gate_passes"].values()),
        "summary_sha256": v9.sha256_file(summary_path),
        "evaluation_sha256": v9.sha256_file(evaluation_path),
        "prediction_rows_sha256": v9.sha256_file(predictions_path),
        "replay_normalization_sha256": replay_normalization_sha256(),
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
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--evaluator", type=Path, default=DEFAULT_EVALUATOR)
    parser.add_argument("--erratum", type=Path, default=DEFAULT_ERRATUM)
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
    args = build_parser().parse_args()
    command_run(args)


if __name__ == "__main__":
    main()

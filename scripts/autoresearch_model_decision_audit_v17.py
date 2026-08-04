#!/usr/bin/env python3
"""Read-only decision audit between two Council-nominated model factors.

This runner validates existing caches, checkpoints, predictions, provenance, and
measurement resolution. It never constructs candidate predictions or trains a
model.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = (
    ROOT
    / ".omc/autoresearch/elements-baseline-replacement/runs"
    / "20260803-model-decision-audit-v17"
)
SPEC_PATH = RUN_ROOT / "specs/iteration-0001.json"
EVALUATOR_PATH = RUN_ROOT / "evaluator-iteration-0001.json"
DEFAULT_OUTPUT_DIR = RUN_ROOT / "iteration-0001"
DEFAULT_REPLAY_DIR = RUN_ROOT / "iteration-0001-replay"
DEFAULT_EVALUATION_PATH = RUN_ROOT / "evaluations/iteration-0001.json"

V9_RUN = (
    ROOT
    / ".omc/autoresearch/elements-baseline-replacement/runs"
    / "20260803-self-supervised-v9"
)
V10_RUN = (
    ROOT
    / ".omc/autoresearch/elements-baseline-replacement/runs"
    / "20260803-self-supervised-v10"
)
V8_RUN = (
    ROOT
    / ".omc/autoresearch/elements-baseline-replacement/runs"
    / "20260731-council-guided-v8"
)

FROZEN_INPUT_PATHS = {
    "v9_spec_sha256": V9_RUN / "specs/iteration-0001.json",
    "v9_paired_summary_sha256": (
        V9_RUN / "iteration-0001/results/paired_seed_summary.json"
    ),
    "v9_prediction_rows_sha256": (
        V9_RUN / "iteration-0001/results/prediction_rows.jsonl"
    ),
    "v10_coverage_summary_sha256": (
        V10_RUN / "iteration-0004/results/summary.json"
    ),
    "v10_per_class_coverage_sha256": (
        V10_RUN / "iteration-0004/results/per_class_coverage.json"
    ),
    "v10_i7_evaluation_sha256": (
        V10_RUN / "iteration-0007/results/evaluation.json"
    ),
    "v10_i7_prediction_rows_sha256": (
        V10_RUN / "iteration-0007/results/prediction_rows.jsonl"
    ),
    "v8_provenance_manifest_sha256": (
        V8_RUN / "iteration-0002/provenance-manifest.jsonl"
    ),
}

RUNTIME_PATHS = {
    "projection": ROOT / "backend/codex_model/weights/projection.pt",
    "prototypes": ROOT / "backend/codex_model/weights/prototypes.pt",
    "config": ROOT / "backend/codex_model/config.json",
}

EXPECTED_CONTRACT_HASHES = {
    "spec_sha256": "538e409c9ce66381d235c235213606ca76315c088863968c909c9f23ec26f985",
    "evaluator_sha256": "af59beeeda5f434b1d9cfc1c172a7f9f430432264748664a83eedb5483a90e66",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_v9_module() -> Any:
    script_path = ROOT / "scripts/autoresearch_self_supervised_v9.py"
    spec = importlib.util.spec_from_file_location(
        "autoresearch_self_supervised_v9_for_v17", script_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load v9 runner: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validate_contract() -> dict[str, Any]:
    actual_hashes = {
        "spec_sha256": sha256_file(SPEC_PATH),
        "evaluator_sha256": sha256_file(EVALUATOR_PATH),
    }
    if actual_hashes != EXPECTED_CONTRACT_HASHES:
        raise ValueError(
            f"v17 contract hash mismatch: expected={EXPECTED_CONTRACT_HASHES} "
            f"actual={actual_hashes}"
        )
    spec = read_json(SPEC_PATH)
    evaluator = read_json(EVALUATOR_PATH)
    if spec.get("status") != "preregistered_before_audit_output":
        raise ValueError("v17 spec is not prospectively frozen")
    if spec["hard_boundaries"] != {
        "new_feature_extraction": False,
        "augmentation_generation": False,
        "candidate_prediction": False,
        "training": False,
        "hyperparameter_sweep": False,
        "checkpoint_write": False,
        "final_test_read": False,
        "runtime_write": False,
        "promotion": False,
    }:
        raise ValueError("v17 hard boundary changed")
    return {
        "spec": spec,
        "evaluator": evaluator,
        **actual_hashes,
    }


def validate_frozen_inputs(spec: dict[str, Any]) -> dict[str, str]:
    expected = spec["frozen_inputs"]
    actual = {name: sha256_file(path) for name, path in FROZEN_INPUT_PATHS.items()}
    if actual != expected:
        raise ValueError(f"frozen input hash mismatch: expected={expected} actual={actual}")
    return actual


def runtime_hashes() -> dict[str, str]:
    return {name: sha256_file(path) for name, path in RUNTIME_PATHS.items()}


def checkpoint_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def validate_c1_checkpoints(summary: dict[str, Any]) -> list[dict[str, Any]]:
    validations: list[dict[str, Any]] = []
    for diagnostic in summary["diagnostics"]:
        candidate = diagnostic["checkpoints"]["candidate"]
        path = checkpoint_path(str(candidate["path"]))
        actual_sha256 = sha256_file(path)
        expected_sha256 = str(candidate["sha256"])
        if actual_sha256 != expected_sha256:
            raise ValueError(f"C1 checkpoint hash mismatch: {path}")
        validations.append(
            {
                "fold": int(diagnostic["fold"]),
                "seed": int(diagnostic["seed"]),
                "path": str(path.relative_to(ROOT)),
                "sha256": actual_sha256,
            }
        )
    validations.sort(key=lambda row: (row["fold"], row["seed"]))
    return validations


def update_moments(
    values: torch.Tensor,
    moments: dict[str, float | int],
) -> None:
    values64 = values.to(dtype=torch.float64)
    count = values64.numel()
    moments["count"] = int(moments["count"]) + count
    moments["sum"] = float(moments["sum"]) + float(values64.sum())
    moments["sum_sq"] = float(moments["sum_sq"]) + float((values64**2).sum())
    moments["minimum"] = min(float(moments["minimum"]), float(values64.min()))
    moments["maximum"] = max(float(moments["maximum"]), float(values64.max()))


def finalize_moments(moments: dict[str, float | int]) -> dict[str, float | int]:
    count = int(moments["count"])
    mean = float(moments["sum"]) / count
    variance = max(0.0, float(moments["sum_sq"]) / count - mean * mean)
    return {
        "count": count,
        "mean": mean,
        "std": math.sqrt(variance),
        "minimum": float(moments["minimum"]),
        "maximum": float(moments["maximum"]),
    }


def validate_view_caches(
    v9: Any,
    folds: Iterable[int],
    expected_views: int,
    expected_feature_dim: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    validations: list[dict[str, Any]] = []
    total_train_rows = 0
    total_view_slots = 0
    total_view_elements = 0
    finite_view_elements = 0
    cosine_moments: dict[str, float | int] = {
        "count": 0,
        "sum": 0.0,
        "sum_sq": 0.0,
        "minimum": math.inf,
        "maximum": -math.inf,
    }

    for fold in folds:
        path = V9_RUN / f"iteration-0001/caches/fold-{fold:02d}-views08.pt"
        cache = v9.load_fold_cache(
            path,
            expected_fold=fold,
            expected_views=expected_views,
            expected_max_rows_per_class=None,
        )
        train = cache["train"]
        oof = cache["oof"]
        base = train["base_features"]
        views = train["view_features"]
        if base.shape != (len(train["row_id"]), expected_feature_dim):
            raise ValueError(f"unexpected base feature shape for fold {fold}: {base.shape}")
        if views.shape != (
            len(train["row_id"]),
            expected_views,
            expected_feature_dim,
        ):
            raise ValueError(f"unexpected view feature shape for fold {fold}: {views.shape}")
        if "view_features" in oof:
            raise ValueError(f"OOF view features exist for fold {fold}")

        fold_finite = 0
        for offset in range(0, len(base), 512):
            base_chunk = base[offset : offset + 512].float()
            view_chunk = views[offset : offset + 512].float()
            fold_finite += int(torch.isfinite(view_chunk).sum())
            cosine = (
                F.normalize(view_chunk, dim=-1)
                * F.normalize(base_chunk, dim=-1).unsqueeze(1)
            ).sum(dim=-1)
            update_moments(cosine, cosine_moments)

        view_elements = views.numel()
        total_train_rows += len(train["row_id"])
        total_view_slots += len(train["row_id"]) * expected_views
        total_view_elements += view_elements
        finite_view_elements += fold_finite
        validations.append(
            {
                "fold": fold,
                "cache_sha256": sha256_file(path),
                "train_rows": len(train["row_id"]),
                "oof_rows": len(oof["row_id"]),
                "view_slots": len(train["row_id"]) * expected_views,
                "view_elements": view_elements,
                "finite_view_elements": fold_finite,
                "oof_view_feature_count": 0,
            }
        )
        del cache, train, oof, base, views
        gc.collect()

    return validations, {
        "train_rows_across_folds": total_train_rows,
        "observed_view_slots": total_view_slots,
        "expected_view_slots": total_train_rows * expected_views,
        "train_view_coverage_fraction": 1.0
        if total_train_rows
        else 0.0,
        "finite_train_view_fraction": finite_view_elements / total_view_elements,
        "oof_view_feature_count": 0,
        "base_to_view_cosine": finalize_moments(cosine_moments),
    }


def prediction_key(row: dict[str, Any]) -> tuple[str, int, int]:
    return str(row["row_id"]), int(row["outer_fold"]), int(row["seed"])


def validate_control_replay(
    v9_rows: list[dict[str, Any]],
    i7_rows: list[dict[str, Any]],
) -> int:
    v9_by_key = {prediction_key(row): row for row in v9_rows}
    if len(v9_by_key) != len(v9_rows):
        raise ValueError("duplicate v9 prediction key")
    matches = 0
    for row in i7_rows:
        control = v9_by_key.get(prediction_key(row))
        if control is None:
            raise ValueError(f"i7 control row missing from v9: {prediction_key(row)}")
        if list(row["control_topk"]) != list(control["candidate_topk"]):
            raise ValueError(f"i7 control top-k mismatch: {prediction_key(row)}")
        matches += 1
    if matches != len(v9_rows) or len(i7_rows) != len(v9_rows):
        raise ValueError("persisted control replay is incomplete")
    return matches


def support_bin(train_rows: int, low_max: int, medium_max: int) -> str:
    if train_rows <= low_max:
        return f"le_{low_max}"
    if train_rows <= medium_max:
        return f"{low_max + 1}_to_{medium_max}"
    return f"ge_{medium_max + 1}"


def error_burden(
    prediction_rows: list[dict[str, Any]],
    manifest_rows: list[dict[str, Any]],
    *,
    low_support_max: int,
    medium_support_max: int,
) -> dict[str, Any]:
    manifest_by_id = {str(row["row_id"]): row for row in manifest_rows}
    if len(manifest_by_id) != len(manifest_rows):
        raise ValueError("duplicate manifest row_id")

    train_index: dict[tuple[int, int], dict[str, Any]] = {}
    folds = sorted({int(row["outer_fold"]) for row in prediction_rows})
    labels = sorted({int(row["label"]) for row in prediction_rows})
    for fold in folds:
        for label in labels:
            train = [
                row
                for row in manifest_rows
                if int(row["fold"]) != fold and int(row["class_label"]) == label
            ]
            train_index[(fold, label)] = {
                "rows": len(train),
                "source_families": {str(row["source_family"]) for row in train},
                "components": {str(row["component_id"]) for row in train},
            }

    strata: dict[str, dict[str, int]] = defaultdict(
        lambda: {"records": 0, "c1_errors": 0, "b0_errors": 0}
    )
    total_c1_errors = 0
    total_b0_errors = 0
    cross_source_c1_errors = 0
    low_support_c1_errors = 0
    cross_component_records = 0
    missing_manifest = 0
    unique_rows: set[str] = set()
    unique_components: set[str] = set()
    unique_labels: set[int] = set()

    for row in prediction_rows:
        row_id = str(row["row_id"])
        manifest = manifest_by_id.get(row_id)
        if manifest is None:
            missing_manifest += 1
            continue
        fold = int(row["outer_fold"])
        label = int(row["label"])
        train = train_index[(fold, label)]
        if train["rows"] <= 0:
            raise ValueError(f"OOF label has no train rows: fold={fold} label={label}")
        cross_source = str(manifest["source_family"]) not in train["source_families"]
        cross_component = str(manifest["component_id"]) not in train["components"]
        if cross_component:
            cross_component_records += 1
        bin_name = support_bin(train["rows"], low_support_max, medium_support_max)
        c1_error = int(row["candidate_topk"][0]) != label
        b0_error = int(row["baseline_topk"][0]) != label
        stratum_key = f"support_{bin_name}|cross_source_{str(cross_source).lower()}"
        strata[stratum_key]["records"] += 1
        strata[stratum_key]["c1_errors"] += int(c1_error)
        strata[stratum_key]["b0_errors"] += int(b0_error)
        total_c1_errors += int(c1_error)
        total_b0_errors += int(b0_error)
        cross_source_c1_errors += int(c1_error and cross_source)
        low_support_c1_errors += int(c1_error and train["rows"] <= low_support_max)
        unique_rows.add(row_id)
        unique_components.add(str(row["provenance_component"]))
        unique_labels.add(label)

    if total_c1_errors <= 0:
        raise ValueError("no persisted C1 residual errors")
    return {
        "prediction_records": len(prediction_rows),
        "unique_oof_rows": len(unique_rows),
        "unique_oof_components": len(unique_components),
        "evaluable_labels": len(unique_labels),
        "manifest_join_missing_rows": missing_manifest,
        "cross_component_record_fraction": cross_component_records
        / len(prediction_rows),
        "total_c1_top1_errors": total_c1_errors,
        "total_b0_top1_errors": total_b0_errors,
        "cross_source_family_c1_errors": cross_source_c1_errors,
        "cross_source_family_residual_error_share": cross_source_c1_errors
        / total_c1_errors,
        "low_support_c1_errors": low_support_c1_errors,
        "low_support_residual_error_share": low_support_c1_errors
        / total_c1_errors,
        "strata": dict(sorted(strata.items())),
        "train_class_count_key_count": len(train_index),
        "exact_train_class_counts_available": all(
            item["rows"] > 0 for item in train_index.values()
        ),
    }


def select_factor(
    multiview_eligible: bool,
    shrinkage_eligible: bool,
    cross_source_error_share: float,
    low_support_error_share: float,
) -> str:
    if multiview_eligible and not shrinkage_eligible:
        return "select_supervised_multiview_readout"
    if shrinkage_eligible and not multiview_eligible:
        return "select_hierarchical_shrunk_prototypes"
    if not multiview_eligible and not shrinkage_eligible:
        return "select_no_factor"
    if cross_source_error_share >= low_support_error_share:
        return "select_supervised_multiview_readout"
    return "select_hierarchical_shrunk_prototypes"


def run_audit(contract: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    spec = contract["spec"]
    evaluator = contract["evaluator"]
    frozen_hashes = validate_frozen_inputs(spec)
    before_runtime = runtime_hashes()
    if before_runtime != spec["runtime_hashes"]:
        raise ValueError("runtime hashes changed before v17 audit")

    v9 = load_v9_module()
    v9_summary = read_json(FROZEN_INPUT_PATHS["v9_paired_summary_sha256"])
    v9_predictions = read_jsonl(FROZEN_INPUT_PATHS["v9_prediction_rows_sha256"])
    i7_predictions = read_jsonl(
        FROZEN_INPUT_PATHS["v10_i7_prediction_rows_sha256"]
    )
    coverage = read_json(FROZEN_INPUT_PATHS["v10_coverage_summary_sha256"])
    manifest = read_jsonl(FROZEN_INPUT_PATHS["v8_provenance_manifest_sha256"])

    expected_grid = spec["expected_grid"]
    cache_validations, view_summary = validate_view_caches(
        v9,
        expected_grid["folds"],
        expected_grid["cache_views"],
        expected_grid["feature_dim"],
    )
    checkpoint_validations = validate_c1_checkpoints(v9_summary)
    control_matches = validate_control_replay(v9_predictions, i7_predictions)
    strata_spec = spec["descriptive_strata"]
    burden = error_burden(
        v9_predictions,
        manifest,
        low_support_max=int(strata_spec["low_support_train_rows_max"]),
        medium_support_max=int(strata_spec["medium_support_train_rows_max"]),
    )

    top1_interval = v9_summary["bootstrap"]["delta_top1_95"]
    macro_interval = v9_summary["bootstrap"]["delta_macro_top1_95"]
    resolution = {
        "top1_component_bootstrap_interval": top1_interval,
        "top1_component_bootstrap_half_width": (top1_interval[1] - top1_interval[0])
        / 2,
        "macro_component_bootstrap_interval": macro_interval,
        "macro_component_bootstrap_half_width": (
            macro_interval[1] - macro_interval[0]
        )
        / 2,
        "v10_current_evaluable_components": coverage["power_sensitivity"][
            "v9_C1_minus_B0"
        ]["current_evaluable_components"],
        "v10_conditional_components_for_top1_half_width_0_010": coverage[
            "power_sensitivity"
        ]["v9_C1_minus_B0"]["conditional_required_components"][
            "half_width_0.010"
        ],
    }

    multi_thresholds = evaluator["factor_eligibility"][
        "supervised_multiview_readout"
    ]
    shrink_thresholds = evaluator["factor_eligibility"][
        "hierarchical_shrunk_prototypes"
    ]
    multiview_gate_passes = {
        "validated_cache_fold_count": len(cache_validations)
        == expected_grid["folds"][-1],
        "train_view_coverage_fraction_gte": view_summary[
            "train_view_coverage_fraction"
        ]
        >= multi_thresholds["train_view_coverage_fraction_gte"],
        "finite_train_view_fraction_gte": view_summary[
            "finite_train_view_fraction"
        ]
        >= multi_thresholds["finite_train_view_fraction_gte"],
        "oof_view_feature_count_lte": view_summary["oof_view_feature_count"]
        <= multi_thresholds["oof_view_feature_count_lte"],
        "cross_source_family_residual_error_share_gte": burden[
            "cross_source_family_residual_error_share"
        ]
        >= multi_thresholds["cross_source_family_residual_error_share_gte"],
    }
    shrinkage_gate_passes = {
        "validated_c1_checkpoint_count": len(checkpoint_validations)
        == expected_grid["c1_checkpoint_count"],
        "exact_train_class_counts_available": burden[
            "exact_train_class_counts_available"
        ]
        is shrink_thresholds["exact_train_class_counts_available"],
        "exact_control_replay_rows": control_matches
        == expected_grid["paired_prediction_rows"],
        "historical_macro_component_bootstrap_half_width_lte": resolution[
            "macro_component_bootstrap_half_width"
        ]
        <= shrink_thresholds["historical_macro_component_bootstrap_half_width_lte"],
    }
    multiview_eligible = all(multiview_gate_passes.values())
    shrinkage_eligible = all(shrinkage_gate_passes.values())
    decision = select_factor(
        multiview_eligible,
        shrinkage_eligible,
        burden["cross_source_family_residual_error_share"],
        burden["low_support_residual_error_share"],
    )

    after_runtime = runtime_hashes()
    integrity_values = {
        "contract_hashes_match": {
            name: contract[name] for name in EXPECTED_CONTRACT_HASHES
        }
        == EXPECTED_CONTRACT_HASHES,
        "frozen_input_hashes_match": frozen_hashes == spec["frozen_inputs"],
        "validated_cache_fold_count": len(cache_validations),
        "validated_c1_checkpoint_count": len(checkpoint_validations),
        "exact_control_replay_rows": control_matches,
        "unique_oof_rows": burden["unique_oof_rows"],
        "unique_oof_components": burden["unique_oof_components"],
        "evaluable_labels": burden["evaluable_labels"],
        "manifest_join_missing_rows": burden["manifest_join_missing_rows"],
        "candidate_prediction_count": 0,
        "training_operation_count": 0,
        "final_test_read": False,
        "runtime_unchanged": before_runtime == after_runtime,
    }
    integrity_passes = {
        key: integrity_values[key] == expected
        for key, expected in evaluator["integrity_gates"].items()
    }
    execution_integrity_pass = all(integrity_passes.values())

    results = {
        "schema_version": "autoresearch-model-decision-audit-v17.results",
        "iteration": 1,
        "cache_validations": cache_validations,
        "checkpoint_validations": checkpoint_validations,
        "control_replay": {
            "persisted_i7_control_topk_matches_persisted_v9_c1": True,
            "exact_control_replay_rows": control_matches,
        },
        "view_summary": view_summary,
        "residual_error_burden": burden,
        "measurement_resolution": resolution,
        "factor_gate_passes": {
            "supervised_multiview_readout": multiview_gate_passes,
            "hierarchical_shrunk_prototypes": shrinkage_gate_passes,
        },
        "factor_eligibility": {
            "supervised_multiview_readout": multiview_eligible,
            "hierarchical_shrunk_prototypes": shrinkage_eligible,
        },
        "decision": decision,
        "selected_factor": {
            "select_supervised_multiview_readout": "supervised_multiview_readout",
            "select_hierarchical_shrunk_prototypes": "hierarchical_shrunk_prototypes",
            "select_no_factor": None,
        }[decision],
        "selected_factor_authorizes_execution": False,
    }
    audit = {
        "schema_version": "autoresearch-model-decision-audit-v17.audit",
        "iteration": 1,
        "contract_hashes": {
            name: contract[name] for name in EXPECTED_CONTRACT_HASHES
        },
        "frozen_input_hashes": frozen_hashes,
        "runtime_hashes_before": before_runtime,
        "runtime_hashes_after": after_runtime,
        "integrity_values": integrity_values,
        "integrity_gate_passes": integrity_passes,
        "execution_integrity_pass": execution_integrity_pass,
        "candidate_prediction_count": 0,
        "training_operation_count": 0,
        "new_feature_extraction_count": 0,
        "final_test_read": False,
        "runtime_unchanged": before_runtime == after_runtime,
        "decision": decision if execution_integrity_pass else "invalid_audit",
    }
    return results, audit


def execute(output_dir: Path, replay_dir: Path, evaluation_path: Path) -> None:
    contract = validate_contract()
    first_results, first_audit = run_audit(contract)
    replay_results, replay_audit = run_audit(contract)
    deterministic_replay = (
        first_results == replay_results and first_audit == replay_audit
    )
    if not deterministic_replay:
        raise ValueError("v17 audit replay diverged")
    first_audit["deterministic_replay"] = True
    replay_audit["deterministic_replay"] = True

    results_path = output_dir / "decision-audit-results.json"
    audit_path = output_dir / "decision-audit-audit.json"
    replay_results_path = replay_dir / "decision-audit-results.json"
    replay_audit_path = replay_dir / "decision-audit-audit.json"
    write_json(results_path, first_results)
    write_json(audit_path, first_audit)
    write_json(replay_results_path, replay_results)
    write_json(replay_audit_path, replay_audit)

    evaluation = {
        "schema_version": "autoresearch-model-decision-audit-v17.evaluation",
        "iteration": 1,
        "pass": bool(first_audit["execution_integrity_pass"]),
        "execution_integrity_pass": bool(first_audit["execution_integrity_pass"]),
        "hypothesis_supported": first_results["selected_factor"] is not None,
        "decision": first_audit["decision"],
        "selected_factor": first_results["selected_factor"],
        "selected_factor_authorizes_execution": False,
        "deterministic_replay": deterministic_replay,
        "candidate_prediction_count": 0,
        "training_operation_count": 0,
        "final_test_read": False,
        "runtime_unchanged": bool(first_audit["runtime_unchanged"]),
        "promotion_eligible": False,
        "output_hashes": {
            "results_sha256": sha256_file(results_path),
            "audit_sha256": sha256_file(audit_path),
        },
        "replay_hashes": {
            "results_sha256": sha256_file(replay_results_path),
            "audit_sha256": sha256_file(replay_audit_path),
        },
    }
    if evaluation["output_hashes"] != evaluation["replay_hashes"]:
        raise ValueError("v17 serialized replay hashes diverged")
    write_json(evaluation_path, evaluation)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--replay-dir", type=Path, default=DEFAULT_REPLAY_DIR)
    parser.add_argument("--evaluation", type=Path, default=DEFAULT_EVALUATION_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    execute(args.output_dir, args.replay_dir, args.evaluation)
    print(json.dumps(read_json(args.evaluation), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

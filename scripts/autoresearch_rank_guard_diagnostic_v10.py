#!/usr/bin/env python3
"""Diagnose whether the persisted v9 rank guard aligns with fold-level accuracy."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import autoresearch_self_supervised_v9 as v9  # noqa: E402


V9_RUN = ROOT / ".omc/autoresearch/elements-baseline-replacement/runs/20260803-self-supervised-v9"
V10_RUN = ROOT / ".omc/autoresearch/elements-baseline-replacement/runs/20260803-self-supervised-v10"
DEFAULT_V9_SUMMARY = V9_RUN / "iteration-0001/results/paired_seed_summary.json"
DEFAULT_V9_PREDICTIONS = V9_RUN / "iteration-0001/results/prediction_rows.jsonl"
DEFAULT_SPEC = V10_RUN / "specs/iteration-0006.json"
DEFAULT_EVALUATOR = V10_RUN / "evaluator-iteration-0006.json"
DEFAULT_OUTPUT_DIR = V10_RUN / "iteration-0006/results"
DEFAULT_RUNTIME_PROJECTION = ROOT / "backend/codex_model/weights/projection.pt"
DEFAULT_RUNTIME_PROTOTYPES = ROOT / "backend/codex_model/weights/prototypes.pt"

EXPECTED_SPEC_SHA256 = "f23c24984dc446a7813b5baed0e31b7a59819e843e977c232c5d7a7267c1df57"
EXPECTED_EVALUATOR_SHA256 = "7d7cf8e48796376a41f66d4561eb8aa657f4f835e28d6e02dc30299bff7e3164"
EXPECTED_V9_SUMMARY_SHA256 = "d6094bc9887748ba9bc89b3f6d691483b803be806cf0623e450450132387a22d"
EXPECTED_V9_PREDICTIONS_SHA256 = "37a15fb66ef883d15df5db8e1cadb152f9460876b60ace832c434375a3cdf34f"
EXPECTED_RUNTIME_SHA256 = {
    "projection": "0ad6ce348eaf44c8844415a769b048b8f155017befa93f16254b1263b9241210",
    "prototypes": "ffc52f03696f03780a01302708b53e319dc24069187a6fd1a2927ea537b1cc54",
}
EXPECTED_FOLDS = [1, 2, 3, 4, 5]
EXPECTED_SEEDS = [17, 42, 73]
EXPECTED_ROWS = 1959


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate_contract(spec_path: Path, evaluator_path: Path) -> dict[str, str]:
    spec_sha256 = v9.sha256_file(spec_path)
    evaluator_sha256 = v9.sha256_file(evaluator_path)
    if spec_sha256 != EXPECTED_SPEC_SHA256:
        raise ValueError("iteration-6 spec SHA mismatch")
    if evaluator_sha256 != EXPECTED_EVALUATOR_SHA256:
        raise ValueError("iteration-6 evaluator SHA mismatch")
    spec = read_json(spec_path)
    evaluator = read_json(evaluator_path)
    if spec.get("iteration") != 6 or evaluator.get("iteration") != 6:
        raise ValueError("iteration-6 contract mismatch")
    if spec.get("status") != "preregistered_before_iteration_0006_diagnostic_calculation":
        raise ValueError("iteration-6 diagnostic was not preregistered")
    if spec.get("new_model_predictions_allowed") is not False:
        raise ValueError("iteration-6 contract permits new model predictions")
    return {
        "spec_sha256": spec_sha256,
        "evaluator_sha256": evaluator_sha256,
    }


def validate_inputs(
    summary_path: Path,
    predictions_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if v9.sha256_file(summary_path) != EXPECTED_V9_SUMMARY_SHA256:
        raise ValueError("persisted v9 summary SHA mismatch")
    if v9.sha256_file(predictions_path) != EXPECTED_V9_PREDICTIONS_SHA256:
        raise ValueError("persisted v9 prediction rows SHA mismatch")
    summary = read_json(summary_path)
    records = read_jsonl(predictions_path)
    if len(records) != EXPECTED_ROWS:
        raise ValueError("persisted v9 prediction row count mismatch")
    folds = sorted({int(row["outer_fold"]) for row in records})
    seeds = sorted({int(row["seed"]) for row in records})
    if folds != EXPECTED_FOLDS or seeds != EXPECTED_SEEDS:
        raise ValueError("persisted v9 fold/seed grid mismatch")
    diagnostic_keys = {
        (int(item["fold"]), int(item["seed"]))
        for item in summary.get("diagnostics", [])
    }
    expected_keys = {
        (fold, seed) for fold in EXPECTED_FOLDS for seed in EXPECTED_SEEDS
    }
    if diagnostic_keys != expected_keys:
        raise ValueError("persisted v9 rank diagnostics grid mismatch")
    return summary, records


def runtime_hashes(projection: Path, prototypes: Path) -> dict[str, str]:
    hashes = {
        "projection": v9.sha256_file(projection),
        "prototypes": v9.sha256_file(prototypes),
    }
    if hashes != EXPECTED_RUNTIME_SHA256:
        raise ValueError("runtime projection/prototype SHA mismatch")
    return hashes


def average_ranks(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(indexed):
        end = start + 1
        while end < len(indexed) and indexed[end][1] == indexed[start][1]:
            end += 1
        average = ((start + 1) + end) / 2.0
        for position in range(start, end):
            ranks[indexed[position][0]] = average
        start = end
    return ranks


def pearson(values_x: Sequence[float], values_y: Sequence[float]) -> float | None:
    if len(values_x) != len(values_y) or len(values_x) < 2:
        raise ValueError("association requires paired vectors with at least two values")
    if np.std(values_x) == 0.0 or np.std(values_y) == 0.0:
        return None
    value = float(np.corrcoef(values_x, values_y)[0, 1])
    return value if math.isfinite(value) else None


def classify_worst_fold(delta_top1: float) -> str:
    return "aligned" if delta_top1 < 0.0 else "not_aligned"


def diagnose(
    summary: dict[str, Any],
    records: Sequence[dict[str, Any]],
    *,
    runtime_unchanged: bool,
) -> dict[str, Any]:
    diagnostics = {
        (int(item["fold"]), int(item["seed"])): item
        for item in summary["diagnostics"]
    }
    per_fold: list[dict[str, Any]] = []
    for fold in EXPECTED_FOLDS:
        fold_records = [
            row for row in records if int(row["outer_fold"]) == fold
        ]
        pooled = v9.accuracy_metrics(fold_records)
        per_seed = [
            {
                "seed": seed,
                **v9.accuracy_metrics(
                    [
                        row
                        for row in fold_records
                        if int(row["seed"]) == seed
                    ]
                ),
                "effective_rank_ratio": float(
                    diagnostics[(fold, seed)]["effective_rank_ratio"]
                ),
            }
            for seed in EXPECTED_SEEDS
        ]
        rank_ratios = [
            float(diagnostics[(fold, seed)]["effective_rank_ratio"])
            for seed in EXPECTED_SEEDS
        ]
        per_fold.append(
            {
                "fold": fold,
                "pooled_metrics": pooled,
                "per_seed": per_seed,
                "minimum_effective_rank_ratio": min(rank_ratios),
                "mean_effective_rank_ratio": float(np.mean(rank_ratios)),
            }
        )

    worst = min(
        per_fold,
        key=lambda item: (item["minimum_effective_rank_ratio"], item["fold"]),
    )
    top1_deltas = [
        float(item["pooled_metrics"]["delta_top1"]) for item in per_fold
    ]
    minimum_rank_ratios = [
        float(item["minimum_effective_rank_ratio"]) for item in per_fold
    ]
    pearson_value = pearson(minimum_rank_ratios, top1_deltas)
    spearman_value = pearson(
        average_ranks(minimum_rank_ratios),
        average_ranks(top1_deltas),
    )
    integrity_values = {
        "v9_summary_sha256_matches": True,
        "v9_prediction_rows_sha256_matches": True,
        "paired_prediction_rows": len(records),
        "paired_seed_count": len(
            {int(row["seed"]) for row in records}
        ),
        "outer_fold_count": len(
            {int(row["outer_fold"]) for row in records}
        ),
        "diagnostic_reuses_only_persisted_predictions": True,
        "no_new_model_prediction": True,
        "runtime_unchanged": runtime_unchanged,
        "final_test_read": False,
        "automatic_promotion": False,
    }
    integrity_passes = {
        "v9_summary_sha256_matches": integrity_values[
            "v9_summary_sha256_matches"
        ]
        is True,
        "v9_prediction_rows_sha256_matches": integrity_values[
            "v9_prediction_rows_sha256_matches"
        ]
        is True,
        "paired_prediction_rows": integrity_values["paired_prediction_rows"]
        == EXPECTED_ROWS,
        "paired_seed_count": integrity_values["paired_seed_count"]
        == len(EXPECTED_SEEDS),
        "outer_fold_count": integrity_values["outer_fold_count"]
        == len(EXPECTED_FOLDS),
        "diagnostic_reuses_only_persisted_predictions": integrity_values[
            "diagnostic_reuses_only_persisted_predictions"
        ]
        is True,
        "no_new_model_prediction": integrity_values["no_new_model_prediction"]
        is True,
        "runtime_unchanged": integrity_values["runtime_unchanged"] is True,
        "final_test_read": integrity_values["final_test_read"] is False,
        "automatic_promotion": integrity_values["automatic_promotion"] is False,
    }
    worst_delta = float(worst["pooled_metrics"]["delta_top1"])
    return {
        "pass": all(integrity_passes.values()),
        "score": worst_delta,
        "hypothesis_supported": None,
        "decision": classify_worst_fold(worst_delta),
        "promotion_eligible": False,
        "per_fold": per_fold,
        "worst_rank_fold": {
            "fold": int(worst["fold"]),
            "minimum_effective_rank_ratio": float(
                worst["minimum_effective_rank_ratio"]
            ),
            "mean_effective_rank_ratio": float(
                worst["mean_effective_rank_ratio"]
            ),
            "delta_top1": worst_delta,
            "delta_top3": float(worst["pooled_metrics"]["delta_top3"]),
            "delta_macro_top1": float(
                worst["pooled_metrics"]["delta_macro_top1"]
            ),
        },
        "descriptive_association": {
            "fold_count": len(EXPECTED_FOLDS),
            "pearson_minimum_rank_ratio_vs_delta_top1": pearson_value,
            "spearman_minimum_rank_ratio_vs_delta_top1": spearman_value,
            "inferential_claim_allowed": False,
        },
        "integrity_gates": integrity_values,
        "integrity_gate_passes": integrity_passes,
        "new_model_predictions": 0,
        "final_test_read": False,
        "runtime_promotion": False,
    }


def command_run(args: argparse.Namespace) -> dict[str, Any]:
    contract = validate_contract(args.spec, args.evaluator)
    summary, records = validate_inputs(args.v9_summary, args.v9_predictions)
    before_runtime = runtime_hashes(
        args.runtime_projection,
        args.runtime_prototypes,
    )
    result = diagnose(summary, records, runtime_unchanged=True)
    after_runtime = runtime_hashes(
        args.runtime_projection,
        args.runtime_prototypes,
    )
    if before_runtime != after_runtime:
        raise RuntimeError("runtime artifacts changed during diagnostic")
    result.update(
        {
            "schema_version": "autoresearch-self-supervised-v10.rank-guard-diagnostic",
            "iteration": 6,
            "name": "persisted-v9-rank-guard-alignment-diagnostic",
            "spec_sha256": contract["spec_sha256"],
            "evaluator_sha256": contract["evaluator_sha256"],
            "v9_summary_sha256": EXPECTED_V9_SUMMARY_SHA256,
            "v9_prediction_rows_sha256": EXPECTED_V9_PREDICTIONS_SHA256,
            "runtime_checkpoint_sha256": before_runtime,
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    v9.write_json(args.output_dir / "summary.json", result)
    v9.write_json(args.output_dir / "evaluation.json", result)
    print(v9.canonical_json(result), end="")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser(
        "run",
        help="analyze persisted v9 predictions without running a model",
    )
    run.add_argument("--v9-summary", type=Path, default=DEFAULT_V9_SUMMARY)
    run.add_argument("--v9-predictions", type=Path, default=DEFAULT_V9_PREDICTIONS)
    run.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    run.add_argument("--evaluator", type=Path, default=DEFAULT_EVALUATOR)
    run.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
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

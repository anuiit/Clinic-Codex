#!/usr/bin/env python3
"""Evaluate raw DINOv2-S/14 features against persisted v9 B0 predictions.

This is a descriptive, non-promotable diagnostic. It never retrains B0, reads
the final test, or writes production runtime artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import autoresearch_self_supervised_v9 as v9  # noqa: E402


V9_RUN = ROOT / ".omc/autoresearch/elements-baseline-replacement/runs/20260803-self-supervised-v9"
V10_RUN = ROOT / ".omc/autoresearch/elements-baseline-replacement/runs/20260803-self-supervised-v10"
DEFAULT_CACHE_DIR = V9_RUN / "iteration-0001/caches"
DEFAULT_BASELINE_PREDICTIONS = V9_RUN / "iteration-0001/results/prediction_rows.jsonl"
DEFAULT_BASELINE_SUMMARY = V9_RUN / "iteration-0001/results/paired_seed_summary.json"
DEFAULT_SPEC = V10_RUN / "specs/iteration-0001.json"
DEFAULT_EVALUATOR = V10_RUN / "evaluator.json"
DEFAULT_OUTPUT_DIR = V10_RUN / "iteration-0001/results"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            records.append(value)
    if not records:
        raise ValueError(f"no prediction rows found: {path}")
    return records


def classify_interval(interval: Sequence[float]) -> str:
    if len(interval) != 2 or not all(math.isfinite(float(value)) for value in interval):
        raise ValueError("classification requires a finite two-sided interval")
    lower, upper = (float(value) for value in interval)
    if lower > 0.0:
        return "raw_superior"
    if upper < 0.0:
        return "learned_projection_helpful"
    return "neutral_or_inconclusive"


def renamed_metrics(records: Sequence[dict[str, Any]]) -> dict[str, float]:
    metrics = v9.accuracy_metrics(records)
    return {
        "B0_top1": metrics["baseline_top1"],
        "B0_top3": metrics["baseline_top3"],
        "B0_macro_top1": metrics["baseline_macro_top1"],
        "raw_top1": metrics["candidate_top1"],
        "raw_top3": metrics["candidate_top3"],
        "raw_macro_top1": metrics["candidate_macro_top1"],
        "delta_top1_raw_minus_B0": metrics["delta_top1"],
        "delta_top3_raw_minus_B0": metrics["delta_top3"],
        "delta_macro_top1_raw_minus_B0": metrics["delta_macro_top1"],
    }


def raw_seed_variant_report(
    records: Sequence[dict[str, Any]],
    *,
    expected_seeds: Sequence[int],
) -> dict[str, Any]:
    variants: dict[tuple[int, str], set[tuple[int, ...]]] = defaultdict(set)
    observed_seeds: dict[tuple[int, str], set[int]] = defaultdict(set)
    for row in records:
        key = (int(row["outer_fold"]), str(row["row_id"]))
        variants[key].add(tuple(int(value) for value in row["candidate_topk"]))
        observed_seeds[key].add(int(row["seed"]))
    expected = set(int(seed) for seed in expected_seeds)
    missing_seed_groups = sorted(
        f"fold={fold},row_id={row_id}"
        for (fold, row_id), seeds in observed_seeds.items()
        if seeds != expected
    )
    maximum_variant_count = max((len(values) for values in variants.values()), default=0)
    return {
        "oof_row_groups": len(variants),
        "maximum_unique_raw_prediction_variants_per_oof_row": maximum_variant_count,
        "raw_predictions_identical_across_seeds": maximum_variant_count == 1,
        "expected_seed_count_per_oof_row": len(expected),
        "missing_or_extra_seed_groups": missing_seed_groups,
        "variance_note": "raw predictions are deterministic duplicates; all inter-seed delta variance comes from persisted B0",
    }


def build_paired_records(
    baseline_records: Sequence[dict[str, Any]],
    raw_by_fold_row: dict[tuple[int, str], list[int]],
    metadata_by_fold_row: dict[tuple[int, str], dict[str, Any]],
    *,
    cache_hashes: dict[str, str],
    baseline_predictions_sha256: str,
    diagnostic_code_sha256: str,
) -> tuple[list[dict[str, Any]], int]:
    records: list[dict[str, Any]] = []
    cache_hash_mismatch_count = 0
    seen: set[tuple[int, int, str]] = set()
    for source in baseline_records:
        fold = int(source["outer_fold"])
        seed = int(source["seed"])
        row_id = str(source["row_id"])
        unique_key = (fold, seed, row_id)
        if unique_key in seen:
            raise ValueError(f"duplicate persisted B0 row: {unique_key}")
        seen.add(unique_key)
        key = (fold, row_id)
        if key not in raw_by_fold_row or key not in metadata_by_fold_row:
            raise ValueError(f"persisted B0 row is absent from validated cache: {key}")
        metadata = metadata_by_fold_row[key]
        if int(source["label"]) != int(metadata["label"]):
            raise ValueError(f"label mismatch for {key}")
        for field in ("class_name", "provenance_component", "decoded_pixel_sha256"):
            if str(source[field]) != str(metadata[field]):
                raise ValueError(f"{field} mismatch for {key}")
        expected_cache_hash = cache_hashes[str(fold)]
        if str(source.get("data_sha256")) != expected_cache_hash:
            cache_hash_mismatch_count += 1
        records.append(
            {
                "row_id": row_id,
                "label": int(metadata["label"]),
                "class_name": str(metadata["class_name"]),
                "provenance_component": str(metadata["provenance_component"]),
                "decoded_pixel_sha256": str(metadata["decoded_pixel_sha256"]),
                "outer_fold": fold,
                "seed": seed,
                "baseline_topk": [int(value) for value in source["baseline_topk"]],
                "candidate_topk": [int(value) for value in raw_by_fold_row[key]],
                "recipe": "persisted-B0-vs-raw-DINOv2-S14-identity",
                "baseline_retrained": False,
                "raw_prediction_seed_dependent": False,
                "cache_sha256": expected_cache_hash,
                "baseline_predictions_sha256": baseline_predictions_sha256,
                "baseline_code_sha256": str(source["code_sha256"]),
                "diagnostic_code_sha256": diagnostic_code_sha256,
                "folds_sha256": str(source["folds_sha256"]),
            }
        )
    return records, cache_hash_mismatch_count


def run_diagnostic(args: argparse.Namespace) -> dict[str, Any]:
    if args.bootstrap_replicates < 100:
        raise ValueError("at least 100 bootstrap replicates are required")
    spec = read_json(args.spec)
    evaluator = read_json(args.evaluator)
    if spec.get("status") != "preregistered":
        raise ValueError("diagnostic spec must be preregistered")
    if evaluator.get("pass_type") != "strict boolean integrity validity only":
        raise ValueError("unexpected evaluator pass contract")

    before_runtime = v9.runtime_hashes(args.runtime_projection, args.runtime_prototypes)
    baseline_summary = read_json(args.baseline_summary)
    baseline_records = read_jsonl(args.baseline_predictions)
    baseline_predictions_sha256 = v9.sha256_file(args.baseline_predictions)
    diagnostic_code_sha256 = v9.sha256_file(Path(__file__))
    expected_source_code = str(baseline_summary["code_sha256"])
    expected_folds_sha256 = str(baseline_summary["folds_sha256"])

    source_code_hashes = {str(row.get("code_sha256")) for row in baseline_records}
    source_fold_hashes = {str(row.get("folds_sha256")) for row in baseline_records}
    if source_code_hashes != {expected_source_code}:
        raise ValueError("persisted B0 code hashes do not match the v9 summary")
    if source_fold_hashes != {expected_folds_sha256}:
        raise ValueError("persisted B0 fold hashes do not match the v9 summary")

    raw_by_fold_row: dict[tuple[int, str], list[int]] = {}
    metadata_by_fold_row: dict[tuple[int, str], dict[str, Any]] = {}
    cache_hashes: dict[str, str] = {}
    cache_validations: list[dict[str, Any]] = []
    raw_effective_rank_by_fold: dict[str, float] = {}
    identity = torch.nn.Identity()
    device = torch.device("cpu")
    for fold in args.folds:
        path = v9.cache_path(args.cache_dir, fold, args.views, None)
        cache = v9.load_fold_cache(
            path,
            expected_fold=fold,
            expected_views=args.views,
            expected_max_rows_per_class=None,
        )
        validation = dict(cache["_cache_validation"])
        cache_validations.append(validation)
        cache_hashes[str(fold)] = str(validation["cache_sha256"])
        raw_topk, raw_rank = v9.predict_arm(identity, cache["train"], cache["oof"], device=device)
        raw_effective_rank_by_fold[str(fold)] = float(raw_rank)
        for index, row_id_value in enumerate(cache["oof"]["row_id"]):
            row_id = str(row_id_value)
            key = (fold, row_id)
            if key in raw_by_fold_row:
                raise ValueError(f"duplicate OOF cache row: {key}")
            raw_by_fold_row[key] = raw_topk[index]
            metadata_by_fold_row[key] = {
                "label": int(cache["oof"]["class_label"][index]),
                "class_name": str(cache["oof"]["class_name"][index]),
                "provenance_component": str(cache["oof"]["component_id"][index]),
                "decoded_pixel_sha256": str(cache["oof"]["decoded_pixel_sha256"][index]),
            }

    expected_cache_hashes = {str(key): str(value) for key, value in baseline_summary["cache_sha256"].items()}
    sidecar_cache_mismatches = sum(
        cache_hashes.get(str(fold)) != expected_cache_hashes.get(str(fold)) for fold in args.folds
    )
    records, row_cache_mismatches = build_paired_records(
        baseline_records,
        raw_by_fold_row,
        metadata_by_fold_row,
        cache_hashes=cache_hashes,
        baseline_predictions_sha256=baseline_predictions_sha256,
        diagnostic_code_sha256=diagnostic_code_sha256,
    )
    cache_hash_mismatch_count = int(sidecar_cache_mismatches + row_cache_mismatches)

    folds = sorted({int(row["outer_fold"]) for row in records})
    seeds = sorted({int(row["seed"]) for row in records})
    if folds != sorted(args.folds):
        raise ValueError(f"unexpected persisted folds: {folds}")
    if seeds != sorted(args.seeds):
        raise ValueError(f"unexpected persisted seeds: {seeds}")

    variants = raw_seed_variant_report(records, expected_seeds=args.seeds)
    overall_metrics = renamed_metrics(records)
    seed_metrics = [
        {"seed": seed, **renamed_metrics([row for row in records if int(row["seed"]) == seed])}
        for seed in seeds
    ]
    source_bootstrap = v9.paired_component_bootstrap(
        records,
        replicates=args.bootstrap_replicates,
        seed=20260803,
    )
    bootstrap = {
        "method": source_bootstrap["method"],
        "replicates": source_bootstrap["replicates"],
        "delta_top1_raw_minus_B0_95": source_bootstrap["delta_top1_95"],
        "delta_macro_top1_raw_minus_B0_95": source_bootstrap["delta_macro_top1_95"],
    }
    classification = classify_interval(bootstrap["delta_top1_raw_minus_B0_95"])
    after_runtime = v9.runtime_hashes(args.runtime_projection, args.runtime_prototypes)
    runtime_unchanged = before_runtime == after_runtime
    if not runtime_unchanged:
        raise RuntimeError("runtime projection/prototype artifacts changed during diagnostic")

    finite_values = [*overall_metrics.values(), *raw_effective_rank_by_fold.values()]
    nan_or_nonfinite_detected = not all(math.isfinite(float(value)) for value in finite_values)
    gate_values: dict[str, Any] = {
        "outer_fold_count": len(folds),
        "paired_seed_count": len(seeds),
        "paired_prediction_rows": len(records),
        "raw_prediction_unique_seed_variants_per_oof_row": variants[
            "maximum_unique_raw_prediction_variants_per_oof_row"
        ],
        "cache_hash_mismatch_count": cache_hash_mismatch_count,
        "provenance_component_overlap_across_folds": max(
            int(item["provenance_component_overlap_across_folds"]) for item in cache_validations
        ),
        "decoded_pixel_hash_overlap_across_folds": max(
            int(item["decoded_pixel_hash_overlap_across_folds"]) for item in cache_validations
        ),
        "nan_or_nonfinite_detected": nan_or_nonfinite_detected,
        "baseline_retrained": False,
        "final_test_read": False,
        "automatic_promotion": False,
        "runtime_unchanged": runtime_unchanged,
    }
    gate_passes = {
        "outer_fold_count": gate_values["outer_fold_count"] == 5,
        "paired_seed_count": gate_values["paired_seed_count"] == 3,
        "paired_prediction_rows": gate_values["paired_prediction_rows"] == 1959,
        "raw_prediction_unique_seed_variants_per_oof_row": gate_values[
            "raw_prediction_unique_seed_variants_per_oof_row"
        ] == 1,
        "raw_prediction_seed_coverage": not variants["missing_or_extra_seed_groups"],
        "cache_hash_mismatch_count": gate_values["cache_hash_mismatch_count"] == 0,
        "provenance_component_overlap_across_folds": gate_values[
            "provenance_component_overlap_across_folds"
        ] == 0,
        "decoded_pixel_hash_overlap_across_folds": gate_values[
            "decoded_pixel_hash_overlap_across_folds"
        ] == 0,
        "nan_or_nonfinite_detected": gate_values["nan_or_nonfinite_detected"] is False,
        "baseline_retrained": gate_values["baseline_retrained"] is False,
        "final_test_read": gate_values["final_test_read"] is False,
        "automatic_promotion": gate_values["automatic_promotion"] is False,
        "runtime_unchanged": gate_values["runtime_unchanged"] is True,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "prediction_rows.jsonl"
    v9.write_jsonl(predictions_path, records)
    summary: dict[str, Any] = {
        "schema_version": "autoresearch-self-supervised-v10.raw-feature-diagnostic",
        "iteration": 1,
        "name": "raw-dinov2-s14-prototype-diagnostic",
        "pass": all(gate_passes.values()),
        "score": overall_metrics["delta_top1_raw_minus_B0"],
        "diagnostic_classification": classification,
        "promotion_eligible": False,
        "overall_metrics": overall_metrics,
        "seed_metrics": seed_metrics,
        "bootstrap": bootstrap,
        "mcnemar": v9.exact_mcnemar(records),
        "raw_effective_rank_by_fold": raw_effective_rank_by_fold,
        "raw_seed_variants": variants,
        "gates": gate_values,
        "gate_passes": gate_passes,
        "folds": folds,
        "seeds": seeds,
        "folds_sha256": expected_folds_sha256,
        "cache_sha256": cache_hashes,
        "baseline_predictions_path": str(args.baseline_predictions.resolve()),
        "baseline_predictions_sha256": baseline_predictions_sha256,
        "baseline_summary_sha256": v9.sha256_file(args.baseline_summary),
        "baseline_code_sha256": expected_source_code,
        "diagnostic_code_sha256": diagnostic_code_sha256,
        "spec_sha256": v9.sha256_file(args.spec),
        "evaluator_sha256": v9.sha256_file(args.evaluator),
        "runtime_checkpoint_sha256": before_runtime,
        "runtime_unchanged": runtime_unchanged,
        "baseline_retrained": False,
        "final_test_read": False,
        "runtime_promotion": False,
        "paired_predictions_path": predictions_path.name,
        "cache_validations": cache_validations,
    }
    v9.write_json(args.output_dir / "summary.json", summary)
    v9.write_json(
        args.output_dir / "per_arm_metrics.json",
        {"overall": overall_metrics, "seeds": seed_metrics},
    )
    print(v9.canonical_json(summary), end="")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--baseline-predictions", type=Path, default=DEFAULT_BASELINE_PREDICTIONS)
    parser.add_argument("--baseline-summary", type=Path, default=DEFAULT_BASELINE_SUMMARY)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--evaluator", type=Path, default=DEFAULT_EVALUATOR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--folds", type=v9.parse_int_csv, default=[1, 2, 3, 4, 5])
    parser.add_argument("--seeds", type=v9.parse_int_csv, default=[17, 42, 73])
    parser.add_argument("--views", type=int, default=8)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--runtime-projection", type=Path, default=v9.DEFAULT_RUNTIME_PROJECTION)
    parser.add_argument("--runtime-prototypes", type=Path, default=v9.DEFAULT_RUNTIME_PROTOTYPES)
    return parser


def main() -> None:
    run_diagnostic(build_parser().parse_args())


if __name__ == "__main__":
    main()

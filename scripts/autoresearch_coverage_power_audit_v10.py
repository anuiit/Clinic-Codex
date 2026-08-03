#!/usr/bin/env python3
"""Audit provenance coverage, dead folds, and component-level power for Elements v10."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import autoresearch_self_supervised_v9 as v9  # noqa: E402


V8_RUN = ROOT / ".omc/autoresearch/elements-baseline-replacement/runs/20260731-council-guided-v8"
V9_RUN = ROOT / ".omc/autoresearch/elements-baseline-replacement/runs/20260803-self-supervised-v9"
V10_RUN = ROOT / ".omc/autoresearch/elements-baseline-replacement/runs/20260803-self-supervised-v10"
DEFAULT_MANIFEST = V8_RUN / "iteration-0002/provenance-manifest.jsonl"
DEFAULT_AUDIT = V8_RUN / "iteration-0003/collection-provenance-audit.json"
DEFAULT_V9_SUMMARY = V9_RUN / "iteration-0001/results/paired_seed_summary.json"
DEFAULT_V9_PREDICTIONS = V9_RUN / "iteration-0001/results/prediction_rows.jsonl"
DEFAULT_V10_SUMMARY = V10_RUN / "iteration-0003/results/summary.json"
DEFAULT_V10_PREDICTIONS = V10_RUN / "iteration-0003/results/prediction_rows.jsonl"
DEFAULT_SPEC = V10_RUN / "specs/iteration-0004.json"
DEFAULT_EVALUATOR = V10_RUN / "evaluator-iteration-0004.json"
DEFAULT_OUTPUT_DIR = V10_RUN / "iteration-0004/results"
DEFAULT_RUNTIME_PROJECTION = ROOT / "backend/codex_model/weights/projection.pt"
DEFAULT_RUNTIME_PROTOTYPES = ROOT / "backend/codex_model/weights/prototypes.pt"

EXPECTED_HASHES = {
    "manifest": "e918a190195d15ccfa9ea1fd4902607b84eea0c7d6c267d146a757f1b6547385",
    "audit": "e931611a9174fbcb2799c52f6bbb6cc315229d9e198260756ce51f804d032b99",
    "v9_summary": "d6094bc9887748ba9bc89b3f6d691483b803be806cf0623e450450132387a22d",
    "v9_predictions": "37a15fb66ef883d15df5db8e1cadb152f9460876b60ace832c434375a3cdf34f",
    "v10_summary": "6cc3bbfff26036c298c7dab9ede3840bb2041a58a2e46301225a8ec8aa80ad77",
    "v10_predictions": "3a1b489af7c5452454a844a45b18b6143faa1ec20e4bd0a37975afa2815f28aa",
}
EXPECTED_ROWS = 9990
EXPECTED_CLASSES = 286
EXPECTED_COMPONENTS = 300
EXPECTED_PAIRED_ROWS = 1959
EXPECTED_UNIQUE_OOF_ROWS = 653
EXPECTED_SEEDS = [17, 42, 73]
EXPECTED_FOLDS = [1, 2, 3, 4, 5]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    v9.write_json(path, value)


def require_hash(path: Path, expected: str, name: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{name} missing: {path}")
    actual = v9.sha256_file(path)
    if actual != expected:
        raise ValueError(f"{name} SHA-256 mismatch: expected {expected}, found {actual}")


def validate_contract(spec_path: Path, evaluator_path: Path) -> dict[str, Any]:
    spec = read_json(spec_path)
    evaluator = read_json(evaluator_path)
    if spec.get("iteration") != 4 or evaluator.get("iteration") != 4:
        raise ValueError("iteration-4 spec/evaluator mismatch")
    if spec.get("status") != "preregistered_before_iteration_0004_audit_output":
        raise ValueError("iteration-4 audit was not preregistered")
    if spec.get("single_diagnostic_factor", {}).get("model_parameters_changed") is not False:
        raise ValueError("iteration-4 audit must not change model parameters")
    return {
        "spec_sha256": v9.sha256_file(spec_path),
        "evaluator_sha256": v9.sha256_file(evaluator_path),
    }


def validate_inputs(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "manifest": args.manifest,
        "audit": args.audit,
        "v9_summary": args.v9_summary,
        "v9_predictions": args.v9_predictions,
        "v10_summary": args.v10_summary,
        "v10_predictions": args.v10_predictions,
    }
    for name, path in paths.items():
        require_hash(path, EXPECTED_HASHES[name], name)
    return {name: {"path": str(path), "sha256": EXPECTED_HASHES[name]} for name, path in paths.items()}


def coverage_report(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    class_names: dict[int, str] = {}
    class_components: dict[int, set[str]] = defaultdict(set)
    class_folds: dict[int, set[int]] = defaultdict(set)
    component_folds: dict[str, set[int]] = defaultdict(set)
    component_rows: Counter[str] = Counter()
    for row in rows:
        label = int(row["class_label"])
        name = str(row["class_name"])
        previous = class_names.setdefault(label, name)
        if previous != name:
            raise ValueError(f"label/name conflict for label {label}")
        component = str(row["component_id"])
        fold = int(row["fold"])
        class_components[label].add(component)
        class_folds[label].add(fold)
        component_folds[component].add(fold)
        component_rows[component] += 1
    cross_fold_components = [component for component, folds in component_folds.items() if len(folds) > 1]
    if cross_fold_components:
        raise ValueError(f"components cross folds: {cross_fold_components[:5]}")
    distribution = Counter(len(components) for components in class_components.values())
    per_class = [
        {
            "label": label,
            "class_name": class_names[label],
            "rows": sum(1 for row in rows if int(row["class_label"]) == label),
            "provenance_components": len(class_components[label]),
            "component_ids": sorted(class_components[label]),
            "folds": sorted(class_folds[label]),
            "inter_component_transfer_evaluable": len(class_components[label]) >= 2 and len(class_folds[label]) >= 2,
        }
        for label in sorted(class_components)
    ]
    return {
        "rows": len(rows),
        "classes": len(class_components),
        "provenance_components": len(component_folds),
        "component_fold_overlap": 0,
        "classes_by_component_count": {str(key): value for key, value in sorted(distribution.items())},
        "classes_with_one_component": sum(len(values) == 1 for values in class_components.values()),
        "classes_with_two_or_more_components": sum(len(values) >= 2 for values in class_components.values()),
        "classes_with_components_in_two_or_more_folds": sum(len(values) >= 2 for values in class_folds.values()),
        "component_row_count": {
            "minimum": min(component_rows.values()),
            "median": sorted(component_rows.values())[len(component_rows) // 2],
            "maximum": max(component_rows.values()),
        },
        "per_class": per_class,
    }


def validate_prediction_grid(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if len(records) != EXPECTED_PAIRED_ROWS:
        raise ValueError(f"expected {EXPECTED_PAIRED_ROWS} prediction rows, found {len(records)}")
    keys = {(int(row["outer_fold"]), int(row["seed"]), str(row["row_id"])) for row in records}
    if len(keys) != EXPECTED_PAIRED_ROWS:
        raise ValueError("prediction grid contains duplicate fold/seed/row keys")
    seeds = sorted({int(row["seed"]) for row in records})
    folds = sorted({int(row["outer_fold"]) for row in records})
    unique_rows = {(int(row["outer_fold"]), str(row["row_id"])) for row in records}
    if seeds != EXPECTED_SEEDS or folds != EXPECTED_FOLDS or len(unique_rows) != EXPECTED_UNIQUE_OOF_ROWS:
        raise ValueError("prediction grid coverage mismatch")
    return {
        "paired_rows": len(records),
        "unique_oof_rows": len(unique_rows),
        "unique_oof_components": len({str(row["provenance_component"]) for row in records}),
        "evaluable_labels": len({int(row["label"]) for row in records}),
        "evaluable_class_names": sorted({str(row["class_name"]) for row in records}),
        "seeds": seeds,
        "folds": folds,
    }


def _paired_metrics(records: Sequence[dict[str, Any]]) -> dict[str, float]:
    raw = v9.accuracy_metrics(records)
    return {
        "S14_B0_top1": raw["baseline_top1"],
        "S14_B0_top3": raw["baseline_top3"],
        "B14_top1": raw["candidate_top1"],
        "B14_top3": raw["candidate_top3"],
        "delta_top1": raw["delta_top1"],
        "delta_top3": raw["delta_top3"],
    }


def dead_fold_report(
    manifest_rows: Sequence[dict[str, Any]],
    prediction_rows: Sequence[dict[str, Any]],
    folds: Iterable[int] = (3, 5),
) -> dict[str, Any]:
    manifest_by_id = {str(row["row_id"]): row for row in manifest_rows}
    reports: dict[str, Any] = {}
    for fold in folds:
        selected = [row for row in prediction_rows if int(row["outer_fold"]) == fold]
        unique_selected = {
            str(row["row_id"]): row for row in selected if int(row["seed"]) == EXPECTED_SEEDS[0]
        }
        by_label: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in unique_selected.values():
            by_label[int(row["label"])].append(row)
        classes: list[dict[str, Any]] = []
        for label, oof_predictions in sorted(by_label.items()):
            oof_ids = {str(row["row_id"]) for row in oof_predictions}
            if not oof_ids <= set(manifest_by_id):
                raise ValueError(f"fold-{fold} prediction row absent from manifest")
            train_rows = [
                row for row in manifest_rows if int(row["class_label"]) == label and int(row["fold"]) != fold
            ]
            oof_rows = [manifest_by_id[row_id] for row_id in sorted(oof_ids)]
            train_families = {str(row.get("source_family", "")) for row in train_rows}
            oof_families = {str(row.get("source_family", "")) for row in oof_rows}
            family_overlap = sorted(train_families & oof_families)
            class_predictions = [row for row in selected if int(row["label"]) == label]
            metrics = _paired_metrics(class_predictions)
            classes.append(
                {
                    "label": label,
                    "class_name": str(oof_predictions[0]["class_name"]),
                    "train_rows": len(train_rows),
                    "train_components": len({str(row["component_id"]) for row in train_rows}),
                    "oof_rows": len(oof_rows),
                    "oof_components": len({str(row["component_id"]) for row in oof_rows}),
                    "train_source_families": sorted(train_families),
                    "oof_source_families": sorted(oof_families),
                    "source_family_overlap": family_overlap,
                    "metadata_shift_flag": not bool(family_overlap),
                    "metrics": metrics,
                }
            )
        metrics = _paired_metrics(selected)
        reports[str(fold)] = {
            "unique_oof_rows": len(unique_selected),
            "unique_components": len({str(row["provenance_component"]) for row in unique_selected.values()}),
            "classes": classes,
            "metrics": metrics,
            "both_arms_zero_top3": metrics["S14_B0_top3"] == 0.0 and metrics["B14_top3"] == 0.0,
            "metadata_shift_present": any(item["metadata_shift_flag"] for item in classes),
        }
    return reports


def interval_sensitivity(
    interval: Sequence[float],
    *,
    current_components: int,
    targets: Sequence[float] = (0.01, 0.005),
) -> dict[str, Any]:
    if len(interval) != 2 or current_components < 1:
        raise ValueError("invalid interval sensitivity input")
    lower, upper = (float(interval[0]), float(interval[1]))
    half_width = (upper - lower) / 2.0
    required = {
        f"half_width_{target:.3f}": math.ceil(current_components * (half_width / target) ** 2)
        for target in targets
    }
    return {
        "interval": [lower, upper],
        "half_width": half_width,
        "current_evaluable_components": current_components,
        "conditional_required_components": required,
        "assumption": "descriptive 1/sqrt(n) scaling with stable component variance and coverage; not a prospective power guarantee",
    }


def command_run(args: argparse.Namespace) -> dict[str, Any]:
    contract = validate_contract(args.spec, args.evaluator)
    input_provenance = validate_inputs(args)
    before_runtime = v9.runtime_hashes(args.runtime_projection, args.runtime_prototypes)
    manifest_rows, corpus_validation = v9.load_corpus(
        args.manifest,
        args.audit,
        strict_counts=True,
    )
    v9_summary = read_json(args.v9_summary)
    v10_summary = read_json(args.v10_summary)
    v9_predictions = read_jsonl(args.v9_predictions)
    v10_predictions = read_jsonl(args.v10_predictions)
    if v10_summary.get("paired_predictions_sha256") != EXPECTED_HASHES["v10_predictions"]:
        raise ValueError("v10 summary/prediction provenance mismatch")
    if v9_summary.get("paired_predictions_path") != args.v9_predictions.name:
        raise ValueError("v9 summary/prediction provenance mismatch")
    coverage = coverage_report(manifest_rows)
    oof = validate_prediction_grid(v10_predictions)
    v9_oof = validate_prediction_grid(v9_predictions)
    if oof != v9_oof:
        raise ValueError("v9 and v10 OOF grids describe different coverage")
    dead_folds = dead_fold_report(manifest_rows, v10_predictions)
    current_components = oof["unique_oof_components"]
    v9_interval = v9_summary["bootstrap"]["delta_top1_95"]
    v10_interval = v10_summary["bootstrap"]["delta_top1_B14_projection_minus_S14_projection_95"]
    power = {
        "v9_C1_minus_B0": interval_sensitivity(v9_interval, current_components=current_components),
        "v10_B14_minus_S14": interval_sensitivity(v10_interval, current_components=current_components),
    }
    after_runtime = v9.runtime_hashes(args.runtime_projection, args.runtime_prototypes)
    runtime_unchanged = before_runtime == after_runtime
    if not runtime_unchanged:
        raise RuntimeError("runtime artifacts changed during coverage audit")
    gates = {
        "retained_rows": coverage["rows"],
        "classes": coverage["classes"],
        "provenance_components": coverage["provenance_components"],
        "paired_oof_rows": oof["paired_rows"],
        "paired_oof_unique_rows": oof["unique_oof_rows"],
        "paired_seed_count": len(oof["seeds"]),
        "outer_fold_count": len(oof["folds"]),
        "input_hashes_match": True,
        "fold_isolation_preserved": coverage["component_fold_overlap"] == 0,
        "runtime_unchanged": runtime_unchanged,
        "final_test_read": False,
        "automatic_promotion": False,
    }
    gate_passes = {
        "retained_rows": gates["retained_rows"] == EXPECTED_ROWS,
        "classes": gates["classes"] == EXPECTED_CLASSES,
        "provenance_components": gates["provenance_components"] == EXPECTED_COMPONENTS,
        "paired_oof_rows": gates["paired_oof_rows"] == EXPECTED_PAIRED_ROWS,
        "paired_oof_unique_rows": gates["paired_oof_unique_rows"] == EXPECTED_UNIQUE_OOF_ROWS,
        "paired_seed_count": gates["paired_seed_count"] == len(EXPECTED_SEEDS),
        "outer_fold_count": gates["outer_fold_count"] == len(EXPECTED_FOLDS),
        "input_hashes_match": gates["input_hashes_match"] is True,
        "fold_isolation_preserved": gates["fold_isolation_preserved"] is True,
        "runtime_unchanged": gates["runtime_unchanged"] is True,
        "final_test_read": gates["final_test_read"] is False,
        "automatic_promotion": gates["automatic_promotion"] is False,
    }
    measurement_limited = oof["evaluable_labels"] < max(30, math.ceil(EXPECTED_CLASSES * 0.2))
    summary = {
        "schema_version": "autoresearch-self-supervised-v10.coverage-power-audit",
        "iteration": 4,
        "name": "provenance-coverage-and-power-audit",
        "pass": all(gate_passes.values()),
        "integrity_pass": all(gate_passes.values()),
        "outcome_is_descriptive": True,
        "measurement_limited": measurement_limited,
        "measurement_recommendation": (
            "effects near or below the empirical top1 interval half-width remain exploratory on current coverage"
        ),
        "coverage": coverage,
        "oof_coverage": oof,
        "dead_fold_audit": dead_folds,
        "power_sensitivity": power,
        "gates": gates,
        "gate_passes": gate_passes,
        "input_provenance": input_provenance,
        "corpus_validation": corpus_validation,
        "spec_sha256": contract["spec_sha256"],
        "evaluator_sha256": contract["evaluator_sha256"],
        "runner_sha256": v9.sha256_file(Path(__file__)),
        "runtime_checkpoint_sha256": before_runtime,
        "runtime_unchanged": runtime_unchanged,
        "final_test_read": False,
        "runtime_promotion": False,
        "automatic_promotion": False,
        "promotion_eligible": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "summary.json", summary)
    write_json(args.output_dir / "per_class_coverage.json", coverage["per_class"])
    council_summary = {
        **summary,
        "coverage": {
            key: value
            for key, value in coverage.items()
            if key != "per_class"
        },
    }
    write_json(args.output_dir / "council_summary.json", council_summary)
    print(v9.canonical_json(council_summary), end="")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--v9-summary", type=Path, default=DEFAULT_V9_SUMMARY)
    parser.add_argument("--v9-predictions", type=Path, default=DEFAULT_V9_PREDICTIONS)
    parser.add_argument("--v10-summary", type=Path, default=DEFAULT_V10_SUMMARY)
    parser.add_argument("--v10-predictions", type=Path, default=DEFAULT_V10_PREDICTIONS)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--evaluator", type=Path, default=DEFAULT_EVALUATOR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--runtime-projection", type=Path, default=DEFAULT_RUNTIME_PROJECTION)
    parser.add_argument("--runtime-prototypes", type=Path, default=DEFAULT_RUNTIME_PROTOTYPES)
    return parser


def main() -> None:
    command_run(build_parser().parse_args())


if __name__ == "__main__":
    main()

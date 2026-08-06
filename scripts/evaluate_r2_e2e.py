#!/usr/bin/env python3
"""Evaluate paired runtime/R2 predictions against the promotion contract.

The evaluator deliberately consumes a paired inference capture instead of
loading either model.  This keeps the evidence independent from model loading
and lets the same report cover an offline corpus, shadow traffic, or a canary.
Every row must contain the annotated truth and the outputs of both arms.

The generated report is deterministic for fixed inputs and arguments.  It is
bound to the candidate manifest, the immutable R2 build specification, and the
promotion specification.  ``validate_promotion_report`` is the shared gate used
by rollout and promotion code; callers must not trust ``overall_pass`` alone.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_SCHEMA_VERSION = "r2-e2e-evaluation-report-v1"
DEFAULT_SPEC_PATH = REPO_ROOT / "backend" / "model_registry" / "specs" / "r2-e2e-promotion-v1.json"
DEFAULT_BUILD_SPEC_PATH = REPO_ROOT / "backend" / "model_registry" / "specs" / "r2-full-data-production-v1.json"
DEFAULT_BOOTSTRAP_REPLICATES = 2000
DEFAULT_BOOTSTRAP_SEED = 20260805
ECE_BINS = 10


class E2EEvaluationError(RuntimeError):
    """Raised when evidence is malformed or not bound to the candidate."""


def sha256_file(path: Path) -> str:
    if not path.is_file():
        raise E2EEvaluationError(f"missing evidence file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def report_payload_sha256(report: Mapping[str, Any]) -> str:
    """Hash a report while excluding the field that stores the hash itself."""

    payload = dict(report)
    integrity = dict(payload.get("integrity") or {})
    integrity.pop("report_payload_sha256", None)
    payload["integrity"] = integrity
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def seal_report(report: dict[str, Any]) -> dict[str, Any]:
    integrity = report.setdefault("integrity", {})
    integrity["report_payload_sha256"] = report_payload_sha256(report)
    return report


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise E2EEvaluationError(f"missing JSON file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise E2EEvaluationError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise E2EEvaluationError(f"expected a JSON object: {path}")
    return value


def _load_rows(path: Path) -> list[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise E2EEvaluationError(f"missing paired predictions: {path}") from exc
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        rows: list[Any] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise E2EEvaluationError(
                    f"invalid JSONL row {line_number}: {path}"
                ) from exc
    else:
        if isinstance(value, list):
            rows = value
        elif isinstance(value, dict):
            rows = next(
                (
                    value[key]
                    for key in ("rows", "predictions", "samples", "items")
                    if isinstance(value.get(key), list)
                ),
                None,
            )
            if rows is None:
                raise E2EEvaluationError(f"paired prediction object has no rows: {path}")
        else:
            raise E2EEvaluationError(f"unsupported paired prediction format: {path}")
    if not rows:
        raise E2EEvaluationError("paired prediction evidence contains no rows")
    if not all(isinstance(row, dict) for row in rows):
        raise E2EEvaluationError("paired prediction rows must be JSON objects")
    return [dict(row) for row in rows]


def _first(record: Mapping[str, Any], names: Sequence[str], *, required: bool = False) -> Any:
    for name in names:
        if name in record and record[name] is not None:
            return record[name]
    if required:
        raise E2EEvaluationError(f"missing required field; expected one of: {', '.join(names)}")
    return None


def _label_key(value: Any) -> str:
    if value is None or isinstance(value, (dict, list)):
        raise E2EEvaluationError(f"invalid class label: {value!r}")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _prediction(row: Mapping[str, Any], arm: str) -> dict[str, Any]:
    raw = row.get(arm)
    if not isinstance(raw, dict):
        raise E2EEvaluationError(f"row is missing the {arm!r} prediction object")
    error_value = raw.get("error")
    has_error = bool(error_value)
    label = _first(raw, ("class_label", "predicted_label", "label"))
    confidence_raw = _first(raw, ("confidence", "score", "top1_score"))
    latency_raw = _first(raw, ("latency_ms", "elapsed_ms"))
    if not has_error and label is None:
        raise E2EEvaluationError(f"successful {arm} prediction is missing class_label")
    if not has_error and confidence_raw is None:
        raise E2EEvaluationError(f"successful {arm} prediction is missing confidence")
    confidence = 0.0 if confidence_raw is None else float(confidence_raw)
    if not math.isfinite(confidence):
        raise E2EEvaluationError(f"non-finite {arm} confidence")
    latency_ms = None if latency_raw is None else float(latency_raw)
    if latency_ms is not None and (not math.isfinite(latency_ms) or latency_ms < 0.0):
        raise E2EEvaluationError(f"invalid {arm} latency_ms")

    top_raw = _first(raw, ("top_k", "top3", "top_labels"))
    top_labels: list[Any] = []
    if isinstance(top_raw, list):
        for item in top_raw:
            if isinstance(item, dict):
                item = _first(item, ("class_label", "predicted_label", "label"))
            if item is not None:
                top_labels.append(item)
    if label is not None and (not top_labels or _label_key(top_labels[0]) != _label_key(label)):
        top_labels.insert(0, label)
    return {
        "error": has_error,
        "label": label,
        "label_key": None if label is None else _label_key(label),
        "confidence": confidence,
        "latency_ms": latency_ms,
        "top_keys": [_label_key(item) for item in top_labels[:3]],
    }


def _identity(row: Mapping[str, Any], kind: str) -> str | None:
    aliases = {
        "pixel": ("pixel_sha256", "decoded_pixel_sha256", "source_pixel_sha256"),
        "case": ("case_id", "patient_case_id"),
        "specimen": ("specimen_id", "manuscript_id"),
    }
    value = _first(row, aliases[kind])
    if value is None or str(value).strip() == "":
        return None
    return str(value).strip()


def _provenance_sets(path: Path) -> dict[str, set[str]]:
    payload = _load_json_object(path)
    rows_value = next(
        (
            payload[key]
            for key in ("rows", "samples", "items", "entries")
            if isinstance(payload.get(key), list)
        ),
        [],
    )
    rows = [row for row in rows_value if isinstance(row, dict)]
    result: dict[str, set[str]] = {}
    top_level_aliases = {
        "pixel": ("pixel_hashes", "decoded_pixel_sha256s", "source_pixel_sha256s"),
        "case": ("case_ids",),
        "specimen": ("specimen_ids", "manuscript_ids"),
    }
    for kind in ("pixel", "case", "specimen"):
        values = {value for row in rows if (value := _identity(row, kind)) is not None}
        for key in top_level_aliases[kind]:
            raw = payload.get(key)
            if isinstance(raw, list):
                values.update(str(item).strip() for item in raw if str(item).strip())
        result[kind] = values
    return result


def _percentile(sorted_values: Sequence[float], quantile: float) -> float:
    if not sorted_values:
        raise E2EEvaluationError("cannot compute percentile of empty values")
    position = (len(sorted_values) - 1) * quantile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction)


def _paired_bootstrap_interval(
    rows: list[dict[str, Any]],
    *,
    replicates: int,
    seed: int,
) -> tuple[list[float], str, int]:
    if replicates < 100:
        raise E2EEvaluationError("at least 100 bootstrap replicates are required")
    group_field = next(
        (
            field
            for field in ("specimen_id", "case_id")
            if all(row.get(field) is not None for row in rows)
        ),
        None,
    )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if group_field is None:
        for index, row in enumerate(rows):
            grouped[str(index)].append(row)
        unit = "row"
    else:
        for row in rows:
            grouped[str(row[group_field])].append(row)
        unit = group_field.removesuffix("_id")
    groups = [grouped[key] for key in sorted(grouped)]
    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(replicates):
        sampled = [groups[rng.randrange(len(groups))] for _ in groups]
        flattened = [row for group in sampled for row in group]
        delta = sum(
            float(row["candidate_correct"]) - float(row["runtime_correct"])
            for row in flattened
        ) / len(flattened)
        deltas.append(delta)
    deltas.sort()
    return [_percentile(deltas, 0.025), _percentile(deltas, 0.975)], unit, len(groups)


def _mcnemar_exact_p(runtime_only: int, candidate_only: int) -> float:
    discordant = runtime_only + candidate_only
    if discordant == 0:
        return 1.0
    tail = min(runtime_only, candidate_only)
    probability = sum(math.comb(discordant, value) for value in range(tail + 1)) / (2 ** discordant)
    return min(1.0, 2.0 * probability)


def _p95(values: Sequence[float]) -> float:
    return _percentile(sorted(values), 0.95)


def _expected_calibration_error(rows: Sequence[dict[str, Any]]) -> float:
    total = len(rows)
    if not total:
        return 1.0
    weighted_error = 0.0
    for bin_index in range(ECE_BINS):
        lower = bin_index / ECE_BINS
        upper = (bin_index + 1) / ECE_BINS
        bucket = []
        for row in rows:
            confidence = min(1.0, max(0.0, float(row["candidate_confidence"])))
            if lower <= confidence < upper or (bin_index == ECE_BINS - 1 and confidence == 1.0):
                bucket.append((confidence, float(row["candidate_correct"])))
        if bucket:
            mean_confidence = sum(item[0] for item in bucket) / len(bucket)
            mean_accuracy = sum(item[1] for item in bucket) / len(bucket)
            weighted_error += len(bucket) / total * abs(mean_accuracy - mean_confidence)
    return weighted_error


def _macro_accuracy(rows: Sequence[dict[str, Any]], key: str) -> float:
    grouped: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        grouped[row["truth_key"]].append(bool(row[key]))
    return sum(sum(values) / len(values) for values in grouped.values()) / len(grouped)


def _operational_section(
    evidence: Mapping[str, Any] | None,
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {"shadow": None, "canary": None, "pass": False}
    if evidence is None:
        return result
    shadow = evidence.get("shadow")
    if isinstance(shadow, dict):
        requests = int(shadow.get("requests", 0))
        error_rate = float(
            shadow.get(
                "candidate_error_rate",
                int(shadow.get("candidate_errors", 0)) / requests if requests else 1.0,
            )
        )
        shadow_spec = spec["shadow"]
        result["shadow"] = {
            "requests": requests,
            "candidate_error_rate": error_rate,
            "pass": requests >= int(shadow_spec["minimum_requests"])
            and error_rate <= float(shadow_spec["candidate_error_rate_max"]),
        }
    canary = evidence.get("canary")
    if isinstance(canary, dict):
        requests = int(canary.get("requests", 0))
        error_rate = float(
            canary.get(
                "candidate_error_rate",
                int(canary.get("candidate_errors", 0)) / requests if requests else 1.0,
            )
        )
        rollback = bool(canary.get("rollback_rehearsal", canary.get("rollback_rehearsal_passed", False)))
        canary_spec = spec["canary"]
        result["canary"] = {
            "requests": requests,
            "candidate_error_rate": error_rate,
            "rollback_rehearsal": rollback,
            "pass": requests >= int(canary_spec["minimum_requests"])
            and error_rate <= float(canary_spec["candidate_error_rate_max"])
            and (rollback or not bool(canary_spec["rollback_rehearsal_required"])),
        }
    result["pass"] = any(
        isinstance(result[name], dict) and bool(result[name]["pass"])
        for name in ("shadow", "canary")
    )
    return result


def _compute_gates(report: Mapping[str, Any], spec: Mapping[str, Any]) -> dict[str, bool]:
    dataset = report["dataset"]
    comparison = report["comparison"]
    rejection = report["rejection"]
    performance = report["performance"]
    operational = report["operational"]
    offline_spec = spec["offline_dataset"]
    quality_spec = spec["quality"]
    rejection_spec = spec["rejection"]
    performance_spec = spec["performance"]
    identity_available = bool(dataset["case_or_specimen_available"])
    gates = {
        "minimum_rows": int(dataset["rows"]) >= int(offline_spec["minimum_rows"]),
        "minimum_classes": int(dataset["classes"]) >= int(offline_spec["minimum_classes"]),
        "minimum_cases_when_available": (
            not identity_available
            or int(dataset["case_or_specimen_groups"]) >= int(offline_spec["minimum_cases_when_available"])
        ),
        "evaluation_pixel_hashes_complete_and_unique": bool(dataset["pixel_hashes_complete_and_unique"]),
        "pixel_hash_disjoint": (
            (not bool(offline_spec["require_pixel_hash_disjoint"]))
            or (bool(dataset["pixel_disjoint_verified"]) and int(dataset["pixel_hash_overlap_count"]) == 0)
        ),
        "case_or_specimen_disjoint_when_available": (
            not bool(offline_spec["require_case_or_specimen_disjoint_when_available"])
            or not identity_available
            or (
                bool(dataset["case_or_specimen_disjoint_verified"])
                and int(dataset["case_or_specimen_overlap_count"]) == 0
            )
        ),
        "top1_delta": float(comparison["top1_delta"]) >= float(quality_spec["top1_delta_min"]),
        "top1_paired_bootstrap_lower": float(comparison["top1_paired_bootstrap_95"][0])
        > float(quality_spec["top1_paired_bootstrap_95_lower_strictly_greater_than"]),
        "macro_top1_delta": float(comparison["macro_top1_delta"]) >= float(quality_spec["macro_top1_delta_min"]),
        "top3_delta": float(comparison["top3_delta"]) >= float(quality_spec["top3_delta_min"]),
        "mcnemar_exact_p": float(comparison["mcnemar_exact_p"]) <= float(quality_spec["mcnemar_exact_p_max"]),
        "candidate_expected_calibration_error": float(rejection["candidate_expected_calibration_error"])
        <= float(rejection_spec["maximum_expected_calibration_error"]),
        "accepted_accuracy_delta": float(rejection["accepted_accuracy_delta"])
        >= float(rejection_spec["accepted_accuracy_delta_min"]),
        "rejection_rate_absolute_delta": float(rejection["rejection_rate_absolute_delta"])
        <= float(rejection_spec["rejection_rate_absolute_delta_max"]),
        "p95_latency_ratio": float(performance["p95_latency_ratio"])
        <= float(performance_spec["p95_latency_ratio_max"]),
        "candidate_error_rate": float(performance["candidate_error_rate"])
        <= float(performance_spec["candidate_error_rate_max"]),
        "shadow_or_canary": bool(operational["pass"]),
    }
    return gates


def evaluate_r2_e2e(
    *,
    predictions_path: Path,
    training_provenance_path: Path,
    candidate_manifest_path: Path,
    spec_path: Path = DEFAULT_SPEC_PATH,
    build_spec_path: Path = DEFAULT_BUILD_SPEC_PATH,
    operational_evidence_path: Path | None = None,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    spec = _load_json_object(spec_path)
    build_spec = _load_json_object(build_spec_path)
    manifest = _load_json_object(candidate_manifest_path)
    version_id = str(manifest.get("version_id") or "")
    if not version_id or version_id != str(spec.get("model_version_id") or ""):
        raise E2EEvaluationError("candidate manifest version does not match the E2E specification")
    if version_id != str(build_spec.get("version_id") or ""):
        raise E2EEvaluationError("candidate manifest version does not match the build specification")
    promotion = manifest.get("promotion")
    if not isinstance(promotion, dict) or not bool(promotion.get("e2e_report_required")):
        raise E2EEvaluationError("candidate manifest does not require the R2 E2E promotion report")

    expected_spec_sha = str(promotion.get("e2e_spec_sha256") or "")
    expected_build_sha = str(promotion.get("build_spec_sha256") or "")
    actual_spec_sha = sha256_file(spec_path)
    actual_build_sha = sha256_file(build_spec_path)
    if expected_spec_sha != actual_spec_sha:
        raise E2EEvaluationError("candidate manifest E2E spec hash does not match the supplied spec")
    if expected_build_sha != actual_build_sha:
        raise E2EEvaluationError("candidate manifest build spec hash does not match the supplied build spec")

    raw_rows = _load_rows(predictions_path)
    training = _provenance_sets(training_provenance_path)
    normalized: list[dict[str, Any]] = []
    eval_pixels: list[str | None] = []
    eval_cases: list[str | None] = []
    eval_specimens: list[str | None] = []
    runtime_latencies: list[float] = []
    candidate_latencies: list[float] = []
    for row in raw_rows:
        truth = _first(row, ("truth_label", "class_label", "ground_truth_label", "label"), required=True)
        truth_key = _label_key(truth)
        runtime = _prediction(row, "runtime")
        candidate = _prediction(row, "candidate")
        runtime_correct = not runtime["error"] and runtime["label_key"] == truth_key
        candidate_correct = not candidate["error"] and candidate["label_key"] == truth_key
        pixel = _identity(row, "pixel")
        case = _identity(row, "case")
        specimen = _identity(row, "specimen")
        eval_pixels.append(pixel)
        eval_cases.append(case)
        eval_specimens.append(specimen)
        if runtime["latency_ms"] is not None and not runtime["error"]:
            runtime_latencies.append(float(runtime["latency_ms"]))
        if candidate["latency_ms"] is not None and not candidate["error"]:
            candidate_latencies.append(float(candidate["latency_ms"]))
        normalized.append(
            {
                "truth_key": truth_key,
                "runtime_correct": runtime_correct,
                "candidate_correct": candidate_correct,
                "runtime_top3_correct": truth_key in runtime["top_keys"],
                "candidate_top3_correct": truth_key in candidate["top_keys"],
                "runtime_confidence": runtime["confidence"],
                "candidate_confidence": candidate["confidence"],
                "runtime_error": runtime["error"],
                "candidate_error": candidate["error"],
                "case_id": case,
                "specimen_id": specimen,
            }
        )

    row_count = len(normalized)
    pixel_values = [value for value in eval_pixels if value is not None]
    pixels_complete_unique = len(pixel_values) == row_count and len(set(pixel_values)) == row_count
    pixel_verified = bool(training["pixel"]) and len(pixel_values) == row_count
    pixel_overlap = len(set(pixel_values) & training["pixel"]) if pixel_verified else -1

    identity_kind: str | None = None
    identity_values: list[str | None] = []
    for kind, values in (("specimen", eval_specimens), ("case", eval_cases)):
        if values and all(value is not None for value in values) and training[kind]:
            identity_kind = kind
            identity_values = values
            break
    identity_available = bool(
        any(value is not None for value in eval_cases + eval_specimens)
        or training["case"]
        or training["specimen"]
    )
    identity_verified = identity_kind is not None
    identity_overlap = (
        len({str(value) for value in identity_values} & training[identity_kind])
        if identity_verified and identity_kind is not None
        else -1
    )
    identity_groups = len({str(value) for value in identity_values}) if identity_verified else 0

    runtime_top1 = sum(row["runtime_correct"] for row in normalized) / row_count
    candidate_top1 = sum(row["candidate_correct"] for row in normalized) / row_count
    runtime_top3 = sum(row["runtime_top3_correct"] for row in normalized) / row_count
    candidate_top3 = sum(row["candidate_top3_correct"] for row in normalized) / row_count
    runtime_macro = _macro_accuracy(normalized, "runtime_correct")
    candidate_macro = _macro_accuracy(normalized, "candidate_correct")
    runtime_only = sum(row["runtime_correct"] and not row["candidate_correct"] for row in normalized)
    candidate_only = sum(row["candidate_correct"] and not row["runtime_correct"] for row in normalized)
    bootstrap_interval, bootstrap_unit, bootstrap_groups = _paired_bootstrap_interval(
        normalized,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )

    threshold = float(spec["rejection"]["fixed_threshold"])
    runtime_accepted = [row for row in normalized if not row["runtime_error"] and row["runtime_confidence"] >= threshold]
    candidate_accepted = [row for row in normalized if not row["candidate_error"] and row["candidate_confidence"] >= threshold]
    runtime_accepted_accuracy = (
        sum(row["runtime_correct"] for row in runtime_accepted) / len(runtime_accepted)
        if runtime_accepted
        else 0.0
    )
    candidate_accepted_accuracy = (
        sum(row["candidate_correct"] for row in candidate_accepted) / len(candidate_accepted)
        if candidate_accepted
        else 0.0
    )
    runtime_rejection_rate = 1.0 - len(runtime_accepted) / row_count
    candidate_rejection_rate = 1.0 - len(candidate_accepted) / row_count
    runtime_p95 = _p95(runtime_latencies) if runtime_latencies else math.inf
    candidate_p95 = _p95(candidate_latencies) if candidate_latencies else math.inf
    latency_ratio = candidate_p95 / runtime_p95 if math.isfinite(runtime_p95) and runtime_p95 > 0.0 else math.inf
    if not math.isfinite(latency_ratio):
        latency_ratio = sys.float_info.max

    operational_evidence = (
        _load_json_object(operational_evidence_path)
        if operational_evidence_path is not None
        else None
    )
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "candidate_version_id": version_id,
        "bindings": {
            "evaluation_spec": {"sha256": actual_spec_sha},
            "build_spec": {"sha256": actual_build_sha},
            "candidate_manifest": {"sha256": sha256_file(candidate_manifest_path)},
            "paired_predictions": {"sha256": sha256_file(predictions_path)},
            "training_provenance": {"sha256": sha256_file(training_provenance_path)},
            "operational_evidence": {
                "sha256": sha256_file(operational_evidence_path)
            } if operational_evidence_path is not None else None,
        },
        "dataset": {
            "rows": row_count,
            "classes": len({row["truth_key"] for row in normalized}),
            "pixel_hashes_complete_and_unique": pixels_complete_unique,
            "pixel_disjoint_verified": pixel_verified,
            "pixel_hash_overlap_count": pixel_overlap,
            "case_or_specimen_available": identity_available,
            "case_or_specimen_kind": identity_kind,
            "case_or_specimen_groups": identity_groups,
            "case_or_specimen_disjoint_verified": identity_verified,
            "case_or_specimen_overlap_count": identity_overlap,
        },
        "models": {
            "runtime": {
                "top1": runtime_top1,
                "macro_top1": runtime_macro,
                "top3": runtime_top3,
                "error_rate": sum(row["runtime_error"] for row in normalized) / row_count,
            },
            "candidate": {
                "top1": candidate_top1,
                "macro_top1": candidate_macro,
                "top3": candidate_top3,
                "error_rate": sum(row["candidate_error"] for row in normalized) / row_count,
            },
        },
        "comparison": {
            "top1_delta": candidate_top1 - runtime_top1,
            "macro_top1_delta": candidate_macro - runtime_macro,
            "top3_delta": candidate_top3 - runtime_top3,
            "top1_paired_bootstrap_95": bootstrap_interval,
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_seed": bootstrap_seed,
            "bootstrap_unit": bootstrap_unit,
            "bootstrap_groups": bootstrap_groups,
            "mcnemar_runtime_only": runtime_only,
            "mcnemar_candidate_only": candidate_only,
            "mcnemar_exact_p": _mcnemar_exact_p(runtime_only, candidate_only),
        },
        "rejection": {
            "fixed_threshold": threshold,
            "candidate_expected_calibration_error": _expected_calibration_error(normalized),
            "runtime_accepted_accuracy": runtime_accepted_accuracy,
            "candidate_accepted_accuracy": candidate_accepted_accuracy,
            "accepted_accuracy_delta": candidate_accepted_accuracy - runtime_accepted_accuracy,
            "runtime_rejection_rate": runtime_rejection_rate,
            "candidate_rejection_rate": candidate_rejection_rate,
            "rejection_rate_absolute_delta": abs(candidate_rejection_rate - runtime_rejection_rate),
        },
        "performance": {
            "runtime_p95_latency_ms": runtime_p95 if math.isfinite(runtime_p95) else None,
            "candidate_p95_latency_ms": candidate_p95 if math.isfinite(candidate_p95) else None,
            "p95_latency_ratio": latency_ratio,
            "candidate_error_rate": sum(row["candidate_error"] for row in normalized) / row_count,
        },
        "operational": _operational_section(operational_evidence, spec),
        "gates": {},
        "overall_pass": False,
        "integrity": {
            "algorithm": "sha256-canonical-json-v1",
        },
    }
    report["gates"] = _compute_gates(report, spec)
    report["overall_pass"] = all(report["gates"].values())
    return seal_report(report)


def _resolve_repo_file(repo_root: Path, value: Any, field: str) -> Path:
    raw = str(value or "").replace("\\", "/")
    pure = PurePosixPath(raw)
    if not raw or pure.is_absolute() or ".." in pure.parts:
        raise E2EEvaluationError(f"unsafe or missing {field}: {value!r}")
    path = repo_root.joinpath(*pure.parts)
    if not path.is_file():
        raise E2EEvaluationError(f"{field} does not exist: {path}")
    return path


def validate_promotion_report(
    report_path: Path,
    *,
    manifest: Mapping[str, Any],
    manifest_path: Path,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Validate report integrity, candidate bindings, and every spec gate.

    This function performs no writes and is safe to call before enabling a
    canary or entering the promotion lock.
    """

    promotion = manifest.get("promotion")
    if not isinstance(promotion, dict) or not bool(promotion.get("e2e_report_required")):
        raise E2EEvaluationError("candidate manifest does not declare e2e_report_required")
    spec_path = _resolve_repo_file(repo_root, promotion.get("e2e_spec_path"), "promotion.e2e_spec_path")
    build_spec_path = _resolve_repo_file(repo_root, promotion.get("build_spec_path"), "promotion.build_spec_path")
    spec_sha = sha256_file(spec_path)
    build_spec_sha = sha256_file(build_spec_path)
    if str(promotion.get("e2e_spec_sha256") or "") != spec_sha:
        raise E2EEvaluationError("manifest E2E spec hash mismatch")
    if str(promotion.get("build_spec_sha256") or "") != build_spec_sha:
        raise E2EEvaluationError("manifest build spec hash mismatch")

    spec = _load_json_object(spec_path)
    build_spec = _load_json_object(build_spec_path)
    report = _load_json_object(report_path)
    version_id = str(manifest.get("version_id") or "")
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise E2EEvaluationError(f"unsupported E2E report schema: {report.get('schema_version')}")
    if str(report.get("candidate_version_id") or "") != version_id:
        raise E2EEvaluationError("E2E report candidate version mismatch")
    if str(spec.get("model_version_id") or "") != version_id:
        raise E2EEvaluationError("E2E spec candidate version mismatch")
    if str(build_spec.get("version_id") or "") != version_id:
        raise E2EEvaluationError("build spec candidate version mismatch")

    bindings = report.get("bindings")
    if not isinstance(bindings, dict):
        raise E2EEvaluationError("E2E report bindings are missing")
    expected_bindings = {
        "evaluation_spec": spec_sha,
        "build_spec": build_spec_sha,
        "candidate_manifest": sha256_file(manifest_path),
    }
    for name, expected_sha in expected_bindings.items():
        binding = bindings.get(name)
        actual_sha = binding.get("sha256") if isinstance(binding, dict) else None
        if actual_sha != expected_sha:
            raise E2EEvaluationError(f"E2E report {name} hash mismatch")
    for name in ("paired_predictions", "training_provenance"):
        binding = bindings.get(name)
        digest = binding.get("sha256") if isinstance(binding, dict) else None
        if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise E2EEvaluationError(f"E2E report {name} hash is invalid")

    integrity = report.get("integrity")
    recorded_payload_sha = integrity.get("report_payload_sha256") if isinstance(integrity, dict) else None
    if recorded_payload_sha != report_payload_sha256(report):
        raise E2EEvaluationError("E2E report payload hash mismatch; report may be tampered")
    try:
        recomputed_gates = _compute_gates(report, spec)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise E2EEvaluationError("E2E report metrics are incomplete or invalid") from exc
    if report.get("gates") != recomputed_gates:
        raise E2EEvaluationError("E2E report gates do not match its metrics and promotion spec")
    if not recomputed_gates or not all(recomputed_gates.values()) or report.get("overall_pass") is not True:
        failed = sorted(name for name, passed in recomputed_gates.items() if not passed)
        raise E2EEvaluationError("E2E promotion gates did not pass: " + ", ".join(failed))
    return report


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate paired runtime/R2 evidence for promotion.")
    parser.add_argument("--predictions", type=Path, required=True, help="Paired annotated predictions (JSON or JSONL).")
    parser.add_argument("--training-provenance", type=Path, required=True, help="Training pixel/case/specimen identities.")
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC_PATH)
    parser.add_argument("--build-spec", type=Path, default=DEFAULT_BUILD_SPEC_PATH)
    parser.add_argument("--operational-evidence", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = evaluate_r2_e2e(
            predictions_path=args.predictions,
            training_provenance_path=args.training_provenance,
            candidate_manifest_path=args.candidate_manifest,
            spec_path=args.spec,
            build_spec_path=args.build_spec,
            operational_evidence_path=args.operational_evidence,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed,
        )
        _write_report(args.output, report)
    except (E2EEvaluationError, KeyError, TypeError, ValueError, OverflowError) as exc:
        if args.json:
            print(json.dumps({"ok": False, "error": str(exc)}, indent=2, sort_keys=True))
        else:
            print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["overall_pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())

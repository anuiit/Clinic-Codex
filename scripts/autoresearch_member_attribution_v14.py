#!/usr/bin/env python3
"""Resolve frozen v13 collisions at exact ZIP-member granularity.

This is a descriptive, model-free audit. It cannot change the definitive v13
verdict, modify exclusions, provide efficacy evidence, or promote a runtime.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = (
    ROOT
    / ".omc/autoresearch/elements-baseline-replacement/runs/"
    "20260803-member-attribution-v14"
)
SPEC_PATH = RUN_DIR / "specs/iteration-0001.json"
EVALUATOR_PATH = RUN_DIR / "evaluator.json"
DEFAULT_OUTPUT_DIR = RUN_DIR / "iteration-0001"
DEFAULT_REPLAY_DIR = RUN_DIR / "iteration-0001-replay"
DEFAULT_EVALUATION_PATH = RUN_DIR / "evaluations/iteration-0001.json"

EXPECTED_SPEC_SHA256 = "650f15f46e107e858351a96471ee691e0bed6f71091f62dd4614d4c6f5f0da6d"
EXPECTED_EVALUATOR_SHA256 = (
    "e6f0b983ae5be6e385ee9d84e8a741da1d26a4014d3a8fc46c59cb2760f75d01"
)
EXPECTED_RUNTIME_HASHES = {
    "projection_sha256": "0ad6ce348eaf44c8844415a769b048b8f155017befa93f16254b1263b9241210",
    "prototypes_sha256": "ffc52f03696f03780a01302708b53e319dc24069187a6fd1a2927ea537b1cc54",
    "config_sha256": "f2150a97ec7eee50a42ac1a108ae4bc092d46a6b8e4a2d91de59349560fbf18b",
}
SELECTED_STATUSES = {
    "ambiguous_archive",
    "unique_archive_without_unique_codex",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_line(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        for row in rows:
            handle.write(canonical_json_line(row))
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from error
    return rows


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def normalize_cote(value: str) -> str:
    return value.strip().casefold()


def validate_contract(
    spec_path: Path = SPEC_PATH,
    evaluator_path: Path = EVALUATOR_PATH,
) -> dict[str, Any]:
    spec_hash = sha256_file(spec_path)
    evaluator_hash = sha256_file(evaluator_path)
    if spec_hash != EXPECTED_SPEC_SHA256:
        raise ValueError(f"v14 spec hash mismatch: {spec_hash}")
    if evaluator_hash != EXPECTED_EVALUATOR_SHA256:
        raise ValueError(f"v14 evaluator hash mismatch: {evaluator_hash}")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    evaluator = json.loads(evaluator_path.read_text(encoding="utf-8"))
    if spec.get("hypothesis_supported") is not None:
        raise ValueError("v14 hypothesis_supported must remain null")
    if spec["reporting_contract"].get("no_numeric_success_threshold") is not True:
        raise ValueError("v14 must not define a numeric success threshold")
    if evaluator.get("numeric_success_threshold") is not None:
        raise ValueError("v14 evaluator must not define a numeric threshold")
    if spec.get("model_inference_allowed") is not False:
        raise ValueError("v14 must forbid model inference")
    if spec.get("training_allowed") is not False:
        raise ValueError("v14 must forbid training")
    if spec.get("final_test_read_allowed") is not False:
        raise ValueError("v14 must forbid final-test access")
    if spec.get("automatic_promotion_allowed") is not False:
        raise ValueError("v14 must forbid automatic promotion")
    if evaluator.get("exclusion_change_allowed") is not False:
        raise ValueError("v14 must forbid exclusion changes")
    return {
        "spec": spec,
        "evaluator": evaluator,
        "spec_sha256": spec_hash,
        "evaluator_sha256": evaluator_hash,
    }


def validate_frozen_inputs(contract: dict[str, Any]) -> dict[str, str]:
    inputs = contract["spec"]["inputs"]
    fields = (
        "v13_attribution",
        "v13_audit",
        "v13_archive_map",
        "codex_csv",
        "elements_csv",
        "v12_exclusions",
    )
    hashes: dict[str, str] = {}
    for field in fields:
        path = resolve_path(inputs[field])
        actual = sha256_file(path)
        expected = inputs[f"{field}_sha256"]
        if actual != expected:
            raise ValueError(f"frozen input hash mismatch for {field}: {actual}")
        hashes[f"{field}_sha256"] = actual
    synthesis_path = resolve_path(
        contract["spec"]["council_authorization"]["synthesis_path"]
    )
    synthesis_hash = sha256_file(synthesis_path)
    expected_synthesis = contract["spec"]["council_authorization"][
        "synthesis_sha256"
    ]
    if synthesis_hash != expected_synthesis:
        raise ValueError(f"Council synthesis hash mismatch: {synthesis_hash}")
    hashes["council_synthesis_sha256"] = synthesis_hash
    return hashes


def runtime_hashes() -> dict[str, str]:
    paths = {
        "projection_sha256": ROOT / "backend/codex_model/weights/projection.pt",
        "prototypes_sha256": ROOT / "backend/codex_model/weights/prototypes.pt",
        "config_sha256": ROOT / "backend/codex_model/config.json",
    }
    result = {key: sha256_file(path) for key, path in paths.items()}
    if result != EXPECTED_RUNTIME_HASHES:
        raise ValueError(f"runtime hash mismatch: {result}")
    return result


def load_codex_catalog(
    codex_path: Path, elements_path: Path
) -> tuple[dict[int, str], dict[str, set[int]]]:
    with codex_path.open("r", encoding="utf-8-sig", newline="") as handle:
        codex_rows = list(csv.DictReader(handle))
    titles = {int(row["id"]): row["titre"] for row in codex_rows}
    if sorted(titles) != list(range(1, 51)):
        raise ValueError("codex catalogue ids must be exactly 1 through 50")
    cote_to_codex: dict[str, set[int]] = defaultdict(set)
    with elements_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            cote = normalize_cote(row["cote"])
            if cote:
                cote_to_codex[cote].add(int(row["codexid"]))
    return titles, dict(cote_to_codex)


def index_path(index_dir: Path, archive_name: str) -> Path:
    return index_dir / f"{archive_name.removesuffix('.zip')}.jsonl"


def member_descriptor(
    row: dict[str, Any],
    *,
    cote_to_codex: dict[str, set[int]],
    codex_titles: dict[int, str],
) -> dict[str, Any]:
    stem = normalize_cote(PurePosixPath(row["member_name"]).stem)
    codex_ids = sorted(cote_to_codex.get(stem, set()))
    return {
        "archive_name": row["archive_name"],
        "member_index": int(row["member_index"]),
        "member_name": row["member_name"],
        "normalized_stem": stem,
        "exact_codex_ids": codex_ids,
        "exact_codex_titles": [codex_titles[codex_id] for codex_id in codex_ids],
    }


def load_member_lookups(
    *,
    index_dir: Path,
    v13_audit: dict[str, Any],
    cote_to_codex: dict[str, set[int]],
    codex_titles: dict[int, str],
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
    dict[str, Any],
]:
    expected_hashes = v13_audit["archive_inputs"]["archive_index_hashes"]
    if len(expected_hashes) != 49:
        raise ValueError(f"expected 49 v13 archive indexes, got {len(expected_hashes)}")
    byte_lookup: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rgb_lookup: dict[str, list[dict[str, Any]]] = defaultdict(list)
    actual_hashes: dict[str, str] = {}
    index_stats_before: dict[str, tuple[int, int]] = {}
    for archive_name, expected_hash in sorted(expected_hashes.items()):
        path = index_path(index_dir, archive_name)
        index_stats_before[archive_name] = (path.stat().st_size, path.stat().st_mtime_ns)
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise ValueError(f"v13 index hash mismatch for {archive_name}: {actual_hash}")
        actual_hashes[archive_name] = actual_hash
        for row in read_jsonl(path):
            if not row.get("byte_sha256") and not row.get("decoded_rgb_sha256"):
                continue
            descriptor = member_descriptor(
                row,
                cote_to_codex=cote_to_codex,
                codex_titles=codex_titles,
            )
            if row.get("byte_sha256"):
                byte_lookup[row["byte_sha256"]].append(descriptor)
            if row.get("decoded_rgb_sha256"):
                rgb_lookup[row["decoded_rgb_sha256"]].append(descriptor)
    for lookup in (byte_lookup, rgb_lookup):
        for digest in lookup:
            lookup[digest].sort(
                key=lambda row: (row["archive_name"], row["member_index"])
            )
    return dict(byte_lookup), dict(rgb_lookup), {
        "archive_index_count": len(actual_hashes),
        "archive_index_hashes": actual_hashes,
        "index_stats_before": index_stats_before,
    }


def select_population(
    rows: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if len(rows) != 9990:
        raise ValueError(f"expected 9990 v13 attribution rows, got {len(rows)}")
    population = [
        row for row in rows if row["attribution_status"] in SELECTED_STATUSES
    ]
    counts = Counter(row["attribution_status"] for row in population)
    expected = {
        "ambiguous_archive": 37,
        "unique_archive_without_unique_codex": 450,
    }
    if len(population) != 487 or dict(counts) != expected:
        raise ValueError(
            f"frozen v14 population mismatch: rows={len(population)}, counts={dict(counts)}"
        )
    subset_432 = sum(
        row["attribution_status"] == "unique_archive_without_unique_codex"
        and row["unique_archive"] != "t_tepeuc.zip"
        for row in population
    )
    if subset_432 != 432:
        raise ValueError(f"predeclared subset mismatch: {subset_432}")
    return population, {**expected, "population_rows": 487, "subset_432_rows": 432}


def resolve_row(
    row: dict[str, Any],
    *,
    byte_lookup: dict[str, list[dict[str, Any]]],
    rgb_lookup: dict[str, list[dict[str, Any]]],
    codex_titles: dict[int, str],
) -> dict[str, Any]:
    byte_members = byte_lookup.get(row["development_byte_sha256"], [])
    rgb_members = rgb_lookup.get(row["decoded_pixel_sha256"], [])
    members_by_identity: dict[tuple[str, int], dict[str, Any]] = {}
    collision_kinds: dict[tuple[str, int], set[str]] = defaultdict(set)
    for kind, members in (("byte", byte_members), ("rgb", rgb_members)):
        for member in members:
            identity = (member["archive_name"], member["member_index"])
            members_by_identity[identity] = member
            collision_kinds[identity].add(kind)
    if not members_by_identity:
        raise ValueError(f"selected row has no exact colliding member: {row['row_id']}")

    members: list[dict[str, Any]] = []
    reasons: set[str] = set()
    convergent_ids: set[int] = set()
    byte_ids: set[int] = set()
    rgb_ids: set[int] = set()
    for identity in sorted(members_by_identity):
        member = members_by_identity[identity]
        codex_ids = member["exact_codex_ids"]
        kinds = sorted(collision_kinds[identity])
        if not codex_ids:
            reasons.add("unmapped_member")
        elif len(codex_ids) > 1:
            reasons.add("multi_codex_member")
            convergent_ids.update(codex_ids)
        else:
            codex_id = codex_ids[0]
            convergent_ids.add(codex_id)
            if "byte" in kinds:
                byte_ids.add(codex_id)
            if "rgb" in kinds:
                rgb_ids.add(codex_id)
        members.append({**member, "collision_kinds": kinds})
    if len(convergent_ids) > 1:
        reasons.add("divergent_codex_ids")
    if byte_ids and rgb_ids and byte_ids != rgb_ids:
        reasons.add("byte_rgb_codex_conflict")

    all_members_uniquely_mapped = all(
        len(member["exact_codex_ids"]) == 1 for member in members
    )
    resolved = all_members_uniquely_mapped and len(convergent_ids) == 1
    codex_id = next(iter(convergent_ids)) if resolved else None
    if resolved:
        reasons.clear()
    return {
        "row_id": row["row_id"],
        "v13_attribution_status": row["attribution_status"],
        "class_label": row["class_label"],
        "class_name": row["class_name"],
        "component_id": row["component_id"],
        "development_byte_sha256": row["development_byte_sha256"],
        "decoded_pixel_sha256": row["decoded_pixel_sha256"],
        "predeclared_432_subset": (
            row["attribution_status"] == "unique_archive_without_unique_codex"
            and row["unique_archive"] != "t_tepeuc.zip"
        ),
        "exact_colliding_member_count": len(members),
        "exact_colliding_members": members,
        "resolved": resolved,
        "resolution_status": "unique_codex" if resolved else "unresolved",
        "unresolved_reasons": sorted(reasons),
        "unique_codex_id": codex_id,
        "unique_codex_title": codex_titles[codex_id] if codex_id else None,
    }


def build_outputs(
    v13_rows: Sequence[dict[str, Any]],
    *,
    population: Sequence[dict[str, Any]],
    byte_lookup: dict[str, list[dict[str, Any]]],
    rgb_lookup: dict[str, list[dict[str, Any]]],
    codex_titles: dict[int, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    resolutions = [
        resolve_row(
            row,
            byte_lookup=byte_lookup,
            rgb_lookup=rgb_lookup,
            codex_titles=codex_titles,
        )
        for row in population
    ]
    resolutions_by_id = {row["row_id"]: row for row in resolutions}
    if len(resolutions_by_id) != 487:
        raise ValueError("v14 population row ids must be unique")

    merged: list[dict[str, Any]] = []
    outside_population_preserved = True
    for original in v13_rows:
        resolution = resolutions_by_id.get(original["row_id"])
        if resolution is None:
            copied = dict(original)
            outside_population_preserved &= canonical_json_line(copied) == canonical_json_line(
                original
            )
            merged.append(copied)
            continue
        copied = {
            **original,
            "v14_member_refinement": {
                "resolved": resolution["resolved"],
                "resolution_status": resolution["resolution_status"],
                "unresolved_reasons": resolution["unresolved_reasons"],
                "unique_codex_id": resolution["unique_codex_id"],
                "unique_codex_title": resolution["unique_codex_title"],
            },
        }
        if resolution["resolved"]:
            copied["unique_codex_id"] = resolution["unique_codex_id"]
            copied["unique_codex_title"] = resolution["unique_codex_title"]
            copied["attribution_status"] = "unique_codex_member_refined"
        merged.append(copied)

    resolved_rows = [row for row in resolutions if row["resolved"]]
    original_codex_ids = {
        int(row["unique_codex_id"])
        for row in v13_rows
        if row.get("unique_codex_id") is not None
    }
    refined_codex_ids = {int(row["unique_codex_id"]) for row in resolved_rows}
    reason_counts = Counter(
        reason for row in resolutions for reason in row["unresolved_reasons"]
    )
    resolution_status_counts = Counter(
        row["resolution_status"] for row in resolutions
    )
    per_codex_counts = Counter(
        int(row["unique_codex_id"]) for row in resolved_rows
    )
    resolved_subset_432 = sum(
        row["resolved"] and row["predeclared_432_subset"] for row in resolutions
    )
    global_unique_rows = 4680 + len(resolved_rows)
    summary = {
        "population_rows": len(resolutions),
        "resolution_status_counts": dict(sorted(resolution_status_counts.items())),
        "resolved_rows": len(resolved_rows),
        "unresolved_rows": len(resolutions) - len(resolved_rows),
        "population_resolution_rate_descriptive": len(resolved_rows) / 487,
        "predeclared_432_subset_rows": 432,
        "predeclared_432_subset_resolved_rows": resolved_subset_432,
        "predeclared_432_subset_resolution_rate_descriptive": (
            resolved_subset_432 / 432
        ),
        "unresolved_reason_counts": dict(sorted(reason_counts.items())),
        "refined_codex_ids": sorted(refined_codex_ids),
        "new_codex_ids_relative_to_v13": sorted(refined_codex_ids - original_codex_ids),
        "resolved_rows_by_codex_id": {
            str(codex_id): per_codex_counts[codex_id]
            for codex_id in sorted(per_codex_counts)
        },
        "v13_unique_codex_rows": 4680,
        "global_unique_codex_rows_descriptive": global_unique_rows,
        "global_unique_codex_coverage_descriptive": global_unique_rows / 9990,
        "outside_population_rows": 9503,
        "outside_population_preserved": outside_population_preserved,
    }
    return resolutions, merged, summary


def build_audit(
    *,
    contract: dict[str, Any],
    input_hashes: dict[str, str],
    index_validation: dict[str, Any],
    population_counts: dict[str, int],
    summary: dict[str, Any],
    source_indexes_unchanged: bool,
    runtime_before: dict[str, str],
    runtime_after: dict[str, str],
) -> dict[str, Any]:
    return {
        "schema_version": "autoresearch-member-attribution-v14.audit",
        "mission": "elements-baseline-replacement",
        "run_id": "20260803-member-attribution-v14",
        "iteration": 1,
        "mode": "descriptive_build_only",
        "factor": "unanimous exact member-level cote attribution",
        "contract_hashes": {
            "spec_sha256": contract["spec_sha256"],
            "evaluator_sha256": contract["evaluator_sha256"],
        },
        "input_hashes": input_hashes,
        "index_validation": {
            "archive_index_count": index_validation["archive_index_count"],
            "archive_index_hashes": index_validation["archive_index_hashes"],
            "source_indexes_unchanged": source_indexes_unchanged,
        },
        "population_contract": population_counts,
        "attribution_rule": {
            "strict_unanimity": True,
            "all_members_must_map": True,
            "single_codex_per_member": True,
            "all_member_codex_ids_must_match": True,
            "fuzzy_matching": False,
            "majority_voting": False,
            "partial_unanimity": False,
        },
        "descriptive_results": summary,
        "hypothesis_supported": None,
        "numeric_success_threshold": None,
        "decision_band": None,
        "v13_verdict": "partial_recovery",
        "v13_reband_allowed": False,
        "scientific_boundary": {
            "archives_used_as_instrument": False,
            "archives_used_for_training": False,
            "v12_exclusions_modified": 0,
            "model_predictions": 0,
            "holdout_or_final_test_read": False,
            "runtime_before": runtime_before,
            "runtime_after": runtime_after,
            "runtime_unchanged": runtime_before == runtime_after,
            "promotion_eligible": False,
            "efficacy_claim_allowed": False,
            "independence_claim_allowed": False,
        },
        "next_direction_requires_council": True,
        "next_phase": "close local provenance, then institutional metadata inventory and shadow-availability audit",
    }


def write_outputs(
    output_dir: Path,
    *,
    resolutions: Sequence[dict[str, Any]],
    merged: Sequence[dict[str, Any]],
    audit: dict[str, Any],
) -> dict[str, str]:
    member_path = output_dir / "member-attribution.jsonl"
    merged_path = output_dir / "development-provenance-v14.jsonl"
    audit_path = output_dir / "member-attribution-audit.json"
    write_jsonl(member_path, resolutions)
    write_jsonl(merged_path, merged)
    write_json(audit_path, audit)
    return {
        "member_attribution_path": str(member_path.resolve()),
        "member_attribution_sha256": sha256_file(member_path),
        "development_provenance_path": str(merged_path.resolve()),
        "development_provenance_sha256": sha256_file(merged_path),
        "audit_path": str(audit_path.resolve()),
        "audit_sha256": sha256_file(audit_path),
    }


def evaluate(
    *,
    audit: dict[str, Any],
    first_outputs: dict[str, str],
    replay_outputs: dict[str, str],
    output_path: Path,
) -> dict[str, Any]:
    results = audit["descriptive_results"]
    boundary = audit["scientific_boundary"]
    replay_exact = all(
        first_outputs[key] == replay_outputs[key]
        for key in (
            "member_attribution_sha256",
            "development_provenance_sha256",
            "audit_sha256",
        )
    )
    integrity_gates = {
        "contract_hashes_match": audit["contract_hashes"]
        == {
            "spec_sha256": EXPECTED_SPEC_SHA256,
            "evaluator_sha256": EXPECTED_EVALUATOR_SHA256,
        },
        "v13_frozen_input_hashes_match": all(
            value
            for key, value in audit["input_hashes"].items()
            if key.endswith("_sha256")
        ),
        "archive_index_count_equals_49": (
            audit["index_validation"]["archive_index_count"] == 49
        ),
        "archive_index_hashes_match_v13": len(
            audit["index_validation"]["archive_index_hashes"]
        )
        == 49,
        "source_indexes_unchanged": audit["index_validation"][
            "source_indexes_unchanged"
        ],
        "population_rows_equals_487": results["population_rows"] == 487,
        "predeclared_subset_rows_equals_432": (
            results["predeclared_432_subset_rows"] == 432
        ),
        "every_selected_row_has_exact_members": (
            results["resolved_rows"] + results["unresolved_rows"] == 487
        ),
        "strict_unanimous_rule_recorded": all(
            audit["attribution_rule"][key]
            for key in (
                "strict_unanimity",
                "all_members_must_map",
                "single_codex_per_member",
                "all_member_codex_ids_must_match",
            )
        ),
        "outside_population_preserved": results["outside_population_preserved"],
        "deterministic_replay_same_hashes": replay_exact,
        "v13_verdict_remains_partial_recovery": (
            audit["v13_verdict"] == "partial_recovery"
            and audit["v13_reband_allowed"] is False
        ),
        "v12_exclusions_modified_equals_0": boundary["v12_exclusions_modified"]
        == 0,
        "model_predictions_equals_0": boundary["model_predictions"] == 0,
        "holdout_or_final_test_unread": boundary["holdout_or_final_test_read"]
        is False,
        "runtime_unchanged": boundary["runtime_unchanged"],
        "hypothesis_supported_is_null": audit["hypothesis_supported"] is None,
        "numeric_success_threshold_is_null": audit["numeric_success_threshold"]
        is None,
    }
    execution_integrity_pass = all(integrity_gates.values())
    evaluation = {
        "schema_version": "autoresearch-member-attribution-v14.evaluation",
        "mission": "elements-baseline-replacement",
        "run_id": "20260803-member-attribution-v14",
        "iteration": 1,
        "pass": execution_integrity_pass,
        "pass_meaning": "descriptive audit execution integrity only",
        "execution_integrity_pass": execution_integrity_pass,
        "hypothesis_supported": None,
        "numeric_success_threshold": None,
        "promotion_eligible": False,
        "v13_verdict": "partial_recovery",
        "v13_rebanded": False,
        "resolved_rows": results["resolved_rows"],
        "unresolved_rows": results["unresolved_rows"],
        "population_resolution_rate_descriptive": results[
            "population_resolution_rate_descriptive"
        ],
        "predeclared_432_subset_resolved_rows": results[
            "predeclared_432_subset_resolved_rows"
        ],
        "global_unique_codex_rows_descriptive": results[
            "global_unique_codex_rows_descriptive"
        ],
        "global_unique_codex_coverage_descriptive": results[
            "global_unique_codex_coverage_descriptive"
        ],
        "new_codex_ids_relative_to_v13": results[
            "new_codex_ids_relative_to_v13"
        ],
        "unresolved_reason_counts": results["unresolved_reason_counts"],
        "integrity_gates": integrity_gates,
        "artifacts": {"first": first_outputs, "replay": replay_outputs},
        "runner_sha256": sha256_file(Path(__file__)),
        "final_test_read": False,
        "runtime_unchanged": boundary["runtime_unchanged"],
        "model_predictions": 0,
        "exclusions_modified": 0,
        "efficacy_evidence": False,
        "independence_evidence": False,
        "next_direction_requires_council": True,
    }
    write_json(output_path, evaluation)
    return evaluation


def run_all(args: argparse.Namespace) -> dict[str, Any]:
    contract = validate_contract()
    input_hashes = validate_frozen_inputs(contract)
    spec_inputs = contract["spec"]["inputs"]
    runtime_before = runtime_hashes()
    v13_audit = json.loads(
        resolve_path(spec_inputs["v13_audit"]).read_text(encoding="utf-8")
    )
    if v13_audit.get("decision_band") != "partial_recovery":
        raise ValueError("v13 verdict must remain partial_recovery")
    codex_titles, cote_to_codex = load_codex_catalog(
        resolve_path(spec_inputs["codex_csv"]),
        resolve_path(spec_inputs["elements_csv"]),
    )
    index_dir = resolve_path(spec_inputs["v13_index_dir"])
    byte_lookup, rgb_lookup, index_validation = load_member_lookups(
        index_dir=index_dir,
        v13_audit=v13_audit,
        cote_to_codex=cote_to_codex,
        codex_titles=codex_titles,
    )
    v13_rows = read_jsonl(resolve_path(spec_inputs["v13_attribution"]))
    population, population_counts = select_population(v13_rows)
    resolutions, merged, summary = build_outputs(
        v13_rows,
        population=population,
        byte_lookup=byte_lookup,
        rgb_lookup=rgb_lookup,
        codex_titles=codex_titles,
    )
    index_stats_after = {
        archive_name: (
            index_path(index_dir, archive_name).stat().st_size,
            index_path(index_dir, archive_name).stat().st_mtime_ns,
        )
        for archive_name in index_validation["archive_index_hashes"]
    }
    source_indexes_unchanged = (
        index_validation["index_stats_before"] == index_stats_after
    )
    if not source_indexes_unchanged:
        raise ValueError("v13 source index size/mtime changed during v14 read-only audit")
    if validate_frozen_inputs(contract) != input_hashes:
        raise ValueError("a frozen input changed during v14")
    runtime_after = runtime_hashes()
    audit = build_audit(
        contract=contract,
        input_hashes=input_hashes,
        index_validation=index_validation,
        population_counts=population_counts,
        summary=summary,
        source_indexes_unchanged=source_indexes_unchanged,
        runtime_before=runtime_before,
        runtime_after=runtime_after,
    )
    first_outputs = write_outputs(
        args.output_dir,
        resolutions=resolutions,
        merged=merged,
        audit=audit,
    )
    replay_outputs = write_outputs(
        args.replay_dir,
        resolutions=resolutions,
        merged=merged,
        audit=audit,
    )
    return evaluate(
        audit=audit,
        first_outputs=first_outputs,
        replay_outputs=replay_outputs,
        output_path=args.evaluation_path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("all",))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--replay-dir", type=Path, default=DEFAULT_REPLAY_DIR)
    parser.add_argument(
        "--evaluation-path", type=Path, default=DEFAULT_EVALUATION_PATH
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "all":
        evaluation = run_all(args)
        print(json.dumps(evaluation, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if evaluation["execution_integrity_pass"] else 2
    raise AssertionError(args.command)


if __name__ == "__main__":
    sys.exit(main())

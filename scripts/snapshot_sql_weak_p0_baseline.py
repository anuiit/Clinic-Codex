#!/usr/bin/env python3
"""Create the Phase-0 baseline snapshot for SQL-backed weak-class candidates.

This script is intentionally read-only with respect to source corpora. It does
not execute SQL, map cotes to images, import images, or launch training. It
freezes the input identities and a small set of prerequisite invariants before
later SQL adapter phases.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_external_corpus import load_active_classes, sha256_file  # noqa: E402

SCHEMA_VERSION = "p0-baseline-1"
SCRIPT_VERSION = "5"
DEFAULT_OUTPUT = ROOT / ".omx" / "reviews" / "data-discovery" / "sql-weak-import-plan-20260708" / "p0_baseline_snapshot.json"
DEFAULT_SQL_DUMP = Path("/mnt/f/CODEX/AI_clinic_class/clinique1.sql")
DEFAULT_SECONDARY_DUMPS = (
    Path("/mnt/f/CODEX/AI_clinic_class/inst-supinfor-application.sql"),
    Path("/mnt/f/CODEX/AI_clinic/inst-supinfor-application.sql"),
)
DEFAULT_PROTECTED_ROOTS = (Path("/mnt/f/CODEX"),)
DEFAULT_ACTIVE_CONFIG = ROOT / "backend" / "codex_model" / "config.json"
DEFAULT_EXTERNAL_PREVIEW = ROOT / ".omx" / "reviews" / "data-discovery" / "import-plan-20260707" / "import_manifest.preview.json"
DEFAULT_CANDIDATE_CSV = ROOT / ".omx" / "reviews" / "data-discovery" / "sql-weak-mapping-20260708" / "weak_classes_sql_image_candidates.csv"
DEFAULT_V2_JSON = ROOT / ".omx" / "reviews" / "data-discovery" / "sql-weak-mapping-20260708" / "candidate_summary.v2.post_internal_conflict_analysis.json"
DEFAULT_V2_CSV = ROOT / ".omx" / "reviews" / "data-discovery" / "sql-weak-mapping-20260708" / "candidate_summary.v2.post_internal_conflict_analysis.csv"

DEFAULT_EXPECTED_ROW_COUNTS = {
    "ai_codex": 50,
    "ai_element": 44577,
    "ai_glyphe": 23586,
    "ai_plate": 1351,
    "ai_zone": 1998,
}
DEFAULT_UNDER_8_CLASSES = [
    "amacalli",
    "copilli",
    "mapilli",
    "quetzalmiyahuayotl",
    "tenchilnahuayo",
    "yacametztli",
]
REQUIRED_CANDIDATE_COLUMNS = {
    "class_name",
    "active_model_class",
    "source_image_path",
    "sha256_bytes",
    "sha256_pixels",
    "status",
    "existing_plan_duplicate_classes",
    "decode_error",
}
EXPECTED_CANDIDATE_ROWS = 3263
EXPECTED_CONFLICT_GROUPS = 475
EXPECTED_CONFLICT_AFFECTED_CLASSES = 50
EXPECTED_STATUS_VALUES = ["candidate_new_for_weak_class"]
INSERT_RE = re.compile(r"\bINSERT\s+INTO\s+`([^`]+)`", re.IGNORECASE)
GENERATED_AT_RE = re.compile(r"^--\s*(?:Generation Time|Généré le)\s*:\s*(.+)$")
ON_DUPLICATE_RE = re.compile(r"ON\s+DUPLICATE\s+KEY\s+UPDATE", re.IGNORECASE)


def file_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "mtime_authoritative": False,
    }


def statement_complete(statement: str) -> bool:
    in_string = False
    in_identifier = False
    escaped = False
    index = 0
    while index < len(statement):
        char = statement[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == "'":
                if index + 1 < len(statement) and statement[index + 1] == "'":
                    index += 1
                else:
                    in_string = False
        elif in_identifier:
            if char == "`":
                in_identifier = False
        else:
            if char == "`":
                in_identifier = True
            elif char == "'":
                in_string = True
            elif char == ";":
                return True
        index += 1
    return False


def iter_insert_statements(sql_dump: Path) -> Iterable[tuple[str, str]]:
    active_table: str | None = None
    chunks: list[str] = []
    with sql_dump.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        for line in handle:
            if active_table is None:
                match = INSERT_RE.search(line)
                if match is None:
                    continue
                active_table = match.group(1)
                chunks = [line]
            else:
                chunks.append(line)
            if active_table is not None and statement_complete("".join(chunks)):
                yield active_table, "".join(chunks)
                active_table = None
                chunks = []
    if active_table is not None:
        raise ValueError(f"unterminated INSERT statement for {active_table}")


def find_values_start(statement: str) -> int:
    in_string = False
    in_identifier = False
    escaped = False
    index = 0
    while index < len(statement):
        char = statement[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == "'":
                if index + 1 < len(statement) and statement[index + 1] == "'":
                    index += 1
                else:
                    in_string = False
        elif in_identifier:
            if char == "`":
                in_identifier = False
        else:
            if char == "`":
                in_identifier = True
            elif char == "'":
                in_string = True
            elif statement[index : index + 6].lower() == "values":
                before = statement[index - 1] if index > 0 else " "
                after = statement[index + 6] if index + 6 < len(statement) else " "
                if not (before.isalnum() or before == "_") and not (after.isalnum() or after == "_"):
                    return index + 6
        index += 1
    raise ValueError("INSERT statement has no top-level VALUES keyword")


def count_value_tuples(statement: str) -> int:
    index = find_values_start(statement)
    in_string = False
    in_identifier = False
    escaped = False
    depth = 0
    count = 0
    while index < len(statement):
        char = statement[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == "'":
                if index + 1 < len(statement) and statement[index + 1] == "'":
                    index += 1
                else:
                    in_string = False
        elif in_identifier:
            if char == "`":
                in_identifier = False
        else:
            if char == "`":
                in_identifier = True
            elif char == "'":
                in_string = True
            elif depth == 0 and ON_DUPLICATE_RE.match(statement, index):
                break
            elif char == "(":
                depth += 1
            elif char == ")":
                if depth == 1:
                    count += 1
                if depth > 0:
                    depth -= 1
            elif char == ";" and depth == 0:
                break
        index += 1
    if depth != 0:
        raise ValueError("unbalanced tuple parentheses in INSERT statement")
    return count


def extract_generated_at_line(sql_dump: Path) -> str:
    with sql_dump.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            text = line.rstrip("\n\r")
            if GENERATED_AT_RE.match(text):
                return text
    return ""


def count_sql_rows(sql_dump: Path) -> tuple[dict[str, int], dict[str, int]]:
    row_counts: Counter[str] = Counter()
    statement_counts: Counter[str] = Counter()
    for table, statement in iter_insert_statements(sql_dump):
        statement_counts[table] += 1
        row_counts[table] += count_value_tuples(statement)
    return dict(sorted(row_counts.items())), dict(sorted(statement_counts.items()))


def load_expected_counts(path: Path | None) -> dict[str, int]:
    if path is None:
        return dict(DEFAULT_EXPECTED_ROW_COUNTS)
    data = json.loads(path.read_text(encoding="utf-8"))
    counts = data.get("sql_row_counts", data)
    if not isinstance(counts, dict):
        raise ValueError("expected counts must be a JSON object or contain sql_row_counts object")
    return {str(key): int(value) for key, value in counts.items()}


def git_provenance() -> dict[str, Any]:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        ).stdout.strip()
    except Exception:
        sha = None
    try:
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=ROOT,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            ).stdout.strip()
        )
    except Exception:
        dirty = None
    return {"git_sha": sha, "git_dirty": dirty}


def check_path_readable(path: Path, blockers: list[str], name: str) -> bool:
    if not path.exists():
        blockers.append(f"missing_{name}:{path}")
        return False
    if not path.is_file():
        blockers.append(f"not_a_file_{name}:{path}")
        return False
    return True


def summarize_active_config(path: Path, blockers: list[str]) -> dict[str, Any]:
    identity = file_identity(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        classes = load_active_classes(path)
    except Exception as exc:
        blockers.append(f"invalid_active_config:{exc.__class__.__name__}")
        return {**identity, "num_classes": None, "class_names_count": None, "class_names_unique": None, "duplicate_classes": []}
    duplicates = sorted([name for name, count in Counter(classes).items() if count > 1])
    num_classes = data.get("num_classes")
    if num_classes != 286 or len(classes) != 286 or len(set(classes)) != 286:
        blockers.append("active_config_class_count_mismatch")
    if duplicates:
        blockers.append("active_config_duplicate_classes")
    return {
        **identity,
        "num_classes": num_classes,
        "class_names_count": len(classes),
        "class_names_unique": len(set(classes)),
        "duplicate_classes": duplicates,
    }


def summarize_external_manifest(path: Path, active_config_sha256: str | None, blockers: list[str]) -> dict[str, Any]:
    if not path.exists():
        blockers.append("missing_external_preview_manifest")
        return {"path": str(path), "present": False, "hash_method": "raw_bytes_not_canonical_json"}
    identity = file_identity(path)
    config_matches: bool | None = None
    manifest_config_sha: str | None = None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        manifest_config_sha = data.get("active_config_sha256") or data.get("audit_active_config_sha256")
        if active_config_sha256 and manifest_config_sha:
            config_matches = manifest_config_sha == active_config_sha256
            if not config_matches:
                blockers.append("external_preview_manifest_config_mismatch")
    except Exception as exc:
        blockers.append(f"invalid_external_preview_manifest:{exc.__class__.__name__}")
    return {
        **identity,
        "present": True,
        "hash_method": "raw_bytes_not_canonical_json",
        "active_config_sha256": manifest_config_sha,
        "active_config_matches_current": config_matches,
    }


def candidate_csv_summary(path: Path, active_classes: set[str], blockers: list[str]) -> dict[str, Any]:
    identity = file_identity(path)
    missing_columns: list[str] = []
    row_count = 0
    status_values: Counter[str] = Counter()
    empty_sha256_pixels = 0
    empty_source_paths = 0
    non_empty_existing_duplicates = 0
    non_empty_decode_errors = 0
    classes_outside_active: set[str] = set()
    conflict_groups: dict[str, set[str]] = defaultdict(set)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing_columns = sorted(REQUIRED_CANDIDATE_COLUMNS - set(fieldnames))
        if missing_columns:
            blockers.append("candidate_csv_missing_columns")
            return {
                **identity,
                "row_count": 0,
                "missing_columns": missing_columns,
                "required_columns_present": False,
            }
        for row in reader:
            row_count += 1
            status = row.get("status", "")
            status_values[status] += 1
            pixel_hash = row.get("sha256_pixels", "")
            if not pixel_hash:
                empty_sha256_pixels += 1
            if not row.get("source_image_path", ""):
                empty_source_paths += 1
            if row.get("existing_plan_duplicate_classes", ""):
                non_empty_existing_duplicates += 1
            if row.get("decode_error", ""):
                non_empty_decode_errors += 1
            active_class = row.get("active_model_class") or row.get("class_name", "")
            if active_class not in active_classes:
                classes_outside_active.add(active_class)
            if pixel_hash and active_class:
                conflict_groups[pixel_hash].add(active_class)
    cross_class_groups = {key: value for key, value in conflict_groups.items() if len(value) > 1}
    affected_classes = sorted({name for classes in cross_class_groups.values() for name in classes})
    if row_count != EXPECTED_CANDIDATE_ROWS:
        blockers.append("candidate_csv_row_count_mismatch")
    if sorted(status_values) != EXPECTED_STATUS_VALUES:
        blockers.append("candidate_csv_unexpected_status_values")
    if empty_sha256_pixels:
        blockers.append("candidate_csv_empty_sha256_pixels")
    if empty_source_paths:
        blockers.append("candidate_csv_empty_source_image_path")
    if non_empty_existing_duplicates:
        blockers.append("candidate_csv_existing_plan_duplicates_present")
    if non_empty_decode_errors:
        blockers.append("candidate_csv_decode_errors_present")
    if classes_outside_active:
        blockers.append("candidate_csv_classes_outside_active_config")
    if len(cross_class_groups) != EXPECTED_CONFLICT_GROUPS:
        blockers.append("candidate_csv_conflict_group_count_mismatch")
    if len(affected_classes) != EXPECTED_CONFLICT_AFFECTED_CLASSES:
        blockers.append("candidate_csv_conflict_affected_class_count_mismatch")
    return {
        **identity,
        "row_count": row_count,
        "missing_columns": missing_columns,
        "required_columns_present": not missing_columns,
        "status_values": sorted(status_values),
        "status_counts": dict(sorted(status_values.items())),
        "empty_sha256_pixels": empty_sha256_pixels,
        "empty_source_image_path": empty_source_paths,
        "non_empty_existing_plan_duplicate_classes": non_empty_existing_duplicates,
        "non_empty_decode_error": non_empty_decode_errors,
        "classes_outside_active_config": sorted(classes_outside_active),
        "internal_cross_class_pixel_conflict_groups": len(cross_class_groups),
        "classes_affected_by_internal_conflicts": len(affected_classes),
        "conflict_grouping_key": "sha256_pixels over active_model_class",
    }


def v2_summary(path_json: Path, path_csv: Path, blockers: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "json_path": str(path_json),
        "csv_path": str(path_csv),
        "json_present": path_json.exists(),
        "csv_present": path_csv.exists(),
    }
    if not path_json.exists():
        blockers.append("missing_v2_conflict_summary_json")
        return result
    if not path_csv.exists():
        blockers.append("missing_v2_conflict_summary_csv")
        return result
    result["json_sha256"] = sha256_file(path_json)
    result["json_size_bytes"] = path_json.stat().st_size
    result["csv_sha256"] = sha256_file(path_csv)
    result["csv_size_bytes"] = path_csv.stat().st_size
    try:
        data = json.loads(path_json.read_text(encoding="utf-8"))
    except Exception as exc:
        blockers.append(f"invalid_v2_conflict_summary_json:{exc.__class__.__name__}")
        return result
    under_entries = data.get("classes_still_under_8_if_internal_conflicts_excluded", [])
    under_projection: dict[str, int] = {}
    if not isinstance(under_entries, list):
        blockers.append("invalid_v2_conflict_summary_schema")
        under_entries = []
    for item in under_entries:
        if not isinstance(item, dict) or item.get("class_name") is None:
            blockers.append("invalid_v2_conflict_summary_schema")
            continue
        try:
            under_projection[str(item["class_name"])] = int(item["projected_unique_if_conflicts_excluded"])
        except (KeyError, TypeError, ValueError):
            blockers.append("invalid_v2_conflict_summary_schema")
    under_names = sorted(under_projection)
    result.update(
        {
            "candidate_rows": data.get("candidate_rows"),
            "internal_cross_class_pixel_conflict_groups": data.get("internal_cross_class_pixel_conflict_groups"),
            "classes_affected_by_internal_conflicts": data.get("classes_affected_by_internal_conflicts"),
            "classes_still_under_8_if_conflicts_excluded": under_names,
            "projected_unique_if_conflicts_excluded": {key: under_projection[key] for key in under_names},
            "under_8_projection_source": "v2_summary_hash_only_not_recomputed_by_p0",
        }
    )
    if data.get("candidate_rows") != EXPECTED_CANDIDATE_ROWS:
        blockers.append("v2_summary_candidate_rows_mismatch")
    if data.get("internal_cross_class_pixel_conflict_groups") != EXPECTED_CONFLICT_GROUPS:
        blockers.append("v2_summary_conflict_group_count_mismatch")
    if data.get("classes_affected_by_internal_conflicts") != EXPECTED_CONFLICT_AFFECTED_CLASSES:
        blockers.append("v2_summary_affected_class_count_mismatch")
    if under_names != DEFAULT_UNDER_8_CLASSES:
        blockers.append("v2_summary_under_8_classes_mismatch")
    return result


def secondary_dump_identities(paths: Iterable[Path]) -> list[dict[str, Any]]:
    identities = []
    for path in paths:
        if path.exists() and path.is_file():
            identities.append({**file_identity(path), "non_primary": True})
        else:
            identities.append({"path": str(path), "present": False, "non_primary": True})
    return identities


def build_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    blockers: list[str] = []
    expected_counts = load_expected_counts(args.expected_counts)
    for name, path in [
        ("sql_dump", args.sql_dump),
        ("active_config", args.active_config),
        ("candidate_csv", args.candidate_csv),
        ("v2_summary_json", args.v2_summary_json),
        ("v2_summary_csv", args.v2_summary_csv),
    ]:
        check_path_readable(path, blockers, name)
    if blockers:
        return {
            "schema_version": SCHEMA_VERSION,
            "phase": "p0",
            "ready_for_next_phase": False,
            "blockers": sorted(blockers),
            "checks": {"required_inputs_present": "fail"},
        }

    sql_identity = file_identity(args.sql_dump)
    sql_identity["generated_at_line"] = extract_generated_at_line(args.sql_dump)
    row_counts: dict[str, int] = {}
    statement_counts: dict[str, int] = {}
    count_mismatches: dict[str, dict[str, int | None]] = {}
    try:
        row_counts, statement_counts = count_sql_rows(args.sql_dump)
    except Exception as exc:
        blockers.append(f"sql_tuple_count_failed:{exc.__class__.__name__}:{exc}")
    for table, expected in expected_counts.items():
        computed = row_counts.get(table)
        if computed != expected:
            count_mismatches[table] = {"computed": computed, "expected": expected}
    if count_mismatches:
        blockers.append("sql_row_count_mismatch")
    row_table_summary = {
        table: {"computed": row_counts.get(table), "expected": expected_counts.get(table), "match": row_counts.get(table) == expected_counts.get(table)}
        for table in sorted(expected_counts)
    }
    active_config = summarize_active_config(args.active_config, blockers)
    active_classes: set[str] = set()
    if active_config.get("class_names_count"):
        active_classes = set(load_active_classes(args.active_config))
    external_manifest = summarize_external_manifest(args.external_preview_manifest, active_config.get("sha256"), blockers)
    candidate_summary = candidate_csv_summary(args.candidate_csv, active_classes, blockers)
    v2 = v2_summary(args.v2_summary_json, args.v2_summary_csv, blockers)
    checks = {
        "required_inputs_present": "pass",
        "sql_row_counts": "fail" if count_mismatches else "pass",
        "active_config_class_count": "pass" if active_config.get("num_classes") == 286 and active_config.get("class_names_count") == 286 and active_config.get("class_names_unique") == 286 else "fail",
        "external_preview_manifest_present": "pass" if external_manifest.get("present") else "fail",
        "external_preview_manifest_config": "pass" if external_manifest.get("active_config_matches_current") is not False else "fail",
        "candidate_csv_sanity": "pass" if not any(b.startswith("candidate_csv_") for b in blockers) else "fail",
        "v2_conflict_summary": "pass"
        if not any(b.startswith(("v2_summary_", "missing_v2_", "invalid_v2_conflict_summary_")) for b in blockers)
        else "fail",
    }
    ready = not blockers
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": "p0",
        "ready_for_next_phase": ready,
        "blockers": sorted(set(blockers)),
        "checks": checks,
        "inputs_identity": {
            "primary_sql_dump": sql_identity,
            "secondary_dumps": secondary_dump_identities(args.secondary_dump),
            "active_config": active_config,
            "existing_preview_manifest": external_manifest,
            "candidate_csv": candidate_summary,
            "v2_conflict_summary": v2,
        },
        "sql_row_counts": {
            "counting_method": "tuple_level_after_values_with_sql_string_state",
            "counts_source": "computed_by_p0",
            "tables": row_table_summary,
            "all_computed_row_counts": dict(sorted(row_counts.items())),
            "insert_statement_counts": dict(sorted(statement_counts.items())),
            "count_mismatches": count_mismatches,
        },
        "provenance": {
            "script": "scripts/snapshot_sql_weak_p0_baseline.py",
            "script_version": SCRIPT_VERSION,
            "sqlparse_used": False,
            "deps": ["stdlib_only"],
            **git_provenance(),
        },
        "optional_image_path_check": {"enabled": False, "missing_source_image_count": None},
        "drift_history": [],
    }


def stable_digest(value: Any) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def is_relative_to_path(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
        return True
    except ValueError:
        return False


def validate_output_path(path: Path, protected_roots: Iterable[Path]) -> str | None:
    for root in protected_roots:
        if is_relative_to_path(path, root):
            return f"output_path_under_protected_source_root:{path}:{root}"
    return None


def atomic_write_json(path: Path, payload: dict[str, Any], ack_baseline_drift: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not ack_baseline_drift:
            raise FileExistsError(f"{path} already exists; pass --ack-baseline-drift to replace it with drift history")
        previous = json.loads(path.read_text(encoding="utf-8"))
        previous_history = previous.get("drift_history", []) if isinstance(previous, dict) else []
        previous_digest = stable_digest({"inputs_identity": previous.get("inputs_identity"), "sql_row_counts": previous.get("sql_row_counts")})
        new_digest = stable_digest({"inputs_identity": payload.get("inputs_identity"), "sql_row_counts": payload.get("sql_row_counts")})
        payload["drift_history"] = [
            *previous_history,
            {
                "previous_inputs_digest": previous_digest,
                "new_inputs_digest": new_digest,
                "previous_primary_sql_sha256": (previous.get("inputs_identity", {}).get("primary_sql_dump", {}) or {}).get("sha256"),
                "new_primary_sql_sha256": (payload.get("inputs_identity", {}).get("primary_sql_dump", {}) or {}).get("sha256"),
                "acknowledged": True,
            },
        ]
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
        tmp = Path(handle.name)
        handle.write(text)
    os.replace(tmp, path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sql-dump", type=Path, default=DEFAULT_SQL_DUMP)
    parser.add_argument("--secondary-dump", type=Path, action="append", default=None)
    parser.add_argument("--protected-root", type=Path, action="append", default=None)
    parser.add_argument("--active-config", type=Path, default=DEFAULT_ACTIVE_CONFIG)
    parser.add_argument("--external-preview-manifest", type=Path, default=DEFAULT_EXTERNAL_PREVIEW)
    parser.add_argument("--candidate-csv", type=Path, default=DEFAULT_CANDIDATE_CSV)
    parser.add_argument("--v2-summary-json", type=Path, default=DEFAULT_V2_JSON)
    parser.add_argument("--v2-summary-csv", type=Path, default=DEFAULT_V2_CSV)
    parser.add_argument("--expected-counts", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ack-baseline-drift", action="store_true", help="replace an existing P0 baseline and record drift history")
    args = parser.parse_args(argv)
    if args.secondary_dump is None:
        args.secondary_dump = list(DEFAULT_SECONDARY_DUMPS)
    extra_protected_roots = args.protected_root or []
    args.protected_root = [*DEFAULT_PROTECTED_ROOTS, *extra_protected_roots]
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_error = validate_output_path(args.output, args.protected_root)
    if output_error is not None:
        print(output_error, file=sys.stderr)
        return 2
    snapshot = build_snapshot(args)
    try:
        atomic_write_json(args.output, snapshot, args.ack_baseline_drift)
    except FileExistsError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps({"output": str(args.output), "ready_for_next_phase": snapshot.get("ready_for_next_phase"), "blockers": snapshot.get("blockers", [])}, ensure_ascii=False, sort_keys=True))
    return 0 if snapshot.get("ready_for_next_phase") else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

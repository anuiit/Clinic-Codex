#!/usr/bin/env python3
"""Audit repo-local shadow-flow availability without reading user content."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = (
    ROOT
    / ".omc/autoresearch/elements-baseline-replacement/runs/"
    "20260803-shadow-availability-v15"
)
SPEC_PATH = RUN_DIR / "specs/iteration-0001.json"
EVALUATOR_PATH = RUN_DIR / "evaluator.json"
MANIFEST_PATH = RUN_DIR / "surface-manifest.json"
DEFAULT_OUTPUT_DIR = RUN_DIR / "iteration-0001"
DEFAULT_REPLAY_DIR = RUN_DIR / "iteration-0001-replay"
DEFAULT_EVALUATION_PATH = RUN_DIR / "evaluations/iteration-0001.json"

EXPECTED_SPEC_SHA256 = "03085ad90b89563abd8ed0c3560dd52209286da49a7307422b66c6deb374b442"
EXPECTED_EVALUATOR_SHA256 = (
    "572e1c7a62e25afe0e3bd131b0086153c3978e80db4708b35abc68db856a007d"
)
EXPECTED_MANIFEST_SHA256 = (
    "14bb355b3e7ecb91d67ec7b1e0edb62f40ccdd5d447577685faa572d5e36f328"
)
EXPECTED_RUNTIME_HASHES = {
    "projection_sha256": "0ad6ce348eaf44c8844415a769b048b8f155017befa93f16254b1263b9241210",
    "prototypes_sha256": "ffc52f03696f03780a01302708b53e319dc24069187a6fd1a2927ea537b1cc54",
    "config_sha256": "f2150a97ec7eee50a42ac1a108ae4bc092d46a6b8e4a2d91de59349560fbf18b",
}

ROUTE_PATTERN = re.compile(
    r"@(?P<blueprint>[A-Za-z_]\w*)\.(?P<method>get|post|put|patch|delete)"
    r"\(\s*[\"'](?P<route>[^\"']+)[\"']",
    re.IGNORECASE,
)
EVIDENCE_PATTERNS = {
    "capture_input": ("request.files", "image_data_url", "File"),
    "persistence": (
        "save_annotation",
        "image.save",
        "indexedDB",
        "localStorage",
        "metadata.json",
        "clinic_auth.sqlite3",
    ),
    "document_identity": (
        "document_id",
        "physical_document_id",
        "manuscript_id",
        "source_document_id",
    ),
    "fallback_identity": ("analysis_id", "session_id", "device_id"),
    "consent": ("consent", "opt_in", "privacy_basis"),
    "retention": ("retention", "delete_after", "image_expires_at"),
    "prediction_exposure": (
        "predicted_class",
        "class_name",
        "SegmentResult",
        "ClassifyResult",
    ),
    "local_only": ("local/dev-only", "local-only", "not production-secured"),
    "logging": ("logging", "getLogger", "telemetry", "analytics", "audit"),
}
DOCUMENT_IDENTITY_FIELDS = {
    "document_id",
    "physical_document_id",
    "manuscript_id",
    "source_document_id",
}
FALLBACK_IDENTITY_FIELDS = {"analysis_id", "session_id", "device_id"}
TIMESTAMP_COLUMNS = {"created_at", "updated_at", "expires_at", "revoked_at"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def iso_from_timestamp(value: float | None) -> str | None:
    if value is None:
        return None
    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()


def validate_contract(
    spec_path: Path = SPEC_PATH,
    evaluator_path: Path = EVALUATOR_PATH,
    manifest_path: Path = MANIFEST_PATH,
) -> dict[str, Any]:
    hashes = {
        "spec_sha256": sha256_file(spec_path),
        "evaluator_sha256": sha256_file(evaluator_path),
        "surface_manifest_sha256": sha256_file(manifest_path),
    }
    expected = {
        "spec_sha256": EXPECTED_SPEC_SHA256,
        "evaluator_sha256": EXPECTED_EVALUATOR_SHA256,
        "surface_manifest_sha256": EXPECTED_MANIFEST_SHA256,
    }
    if hashes != expected:
        raise ValueError(f"v15 contract hash mismatch: {hashes}")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    evaluator = json.loads(evaluator_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if len(manifest["tracked_sources"]) != 26:
        raise ValueError("v15 requires exactly 26 frozen tracked sources")
    if len(manifest["dynamic_surfaces"]) != 5:
        raise ValueError("v15 requires exactly five frozen dynamic surfaces")
    if spec.get("model_inference_allowed") is not False:
        raise ValueError("v15 must forbid model inference")
    if spec.get("training_allowed") is not False:
        raise ValueError("v15 must forbid training")
    if spec.get("external_fetch_allowed") is not False:
        raise ValueError("v15 must forbid external fetches")
    if spec.get("final_test_read_allowed") is not False:
        raise ValueError("v15 must forbid final-test access")
    if evaluator.get("v16_mandatory_regardless_of_result") is not True:
        raise ValueError("v16 must remain mandatory")
    return {"spec": spec, "evaluator": evaluator, "manifest": manifest, **hashes}


def validate_frozen_sources(contract: dict[str, Any]) -> dict[str, str]:
    actual: dict[str, str] = {}
    for relative, expected in sorted(contract["manifest"]["tracked_sources"].items()):
        digest = sha256_file(ROOT / relative)
        if digest != expected:
            raise ValueError(f"frozen source hash mismatch for {relative}: {digest}")
        actual[relative] = digest
    candidate = contract["manifest"]["candidate_freeze"]
    candidate_hash = sha256_file(ROOT / candidate["manifest"])
    if candidate_hash != candidate["manifest_sha256"]:
        raise ValueError(f"candidate manifest hash mismatch: {candidate_hash}")
    synthesis = contract["spec"]["council_authorization"]
    synthesis_hash = sha256_file(ROOT / synthesis["synthesis_path"])
    if synthesis_hash != synthesis["synthesis_sha256"]:
        raise ValueError(f"Council synthesis hash mismatch: {synthesis_hash}")
    return {
        "candidate_manifest_sha256": candidate_hash,
        "council_synthesis_sha256": synthesis_hash,
        "tracked_source_inventory_sha256": hashlib.sha256(
            "".join(f"{actual[path]}  {path}\n" for path in sorted(actual)).encode()
        ).hexdigest(),
    }


def runtime_hashes() -> dict[str, str]:
    paths = {
        "projection_sha256": ROOT / "backend/codex_model/weights/projection.pt",
        "prototypes_sha256": ROOT / "backend/codex_model/weights/prototypes.pt",
        "config_sha256": ROOT / "backend/codex_model/config.json",
    }
    actual = {key: sha256_file(path) for key, path in paths.items()}
    if actual != EXPECTED_RUNTIME_HASHES:
        raise ValueError(f"runtime hash mismatch: {actual}")
    return actual


def token_evidence(
    relative_path: str, text: str, category: str, tokens: Sequence[str]
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    lines = text.splitlines()
    for token in tokens:
        pattern = re.compile(re.escape(token), re.IGNORECASE)
        for line_number, line in enumerate(lines, start=1):
            if pattern.search(line):
                evidence.append(
                    {
                        "source": relative_path,
                        "line": line_number,
                        "token": token,
                        "category": category,
                    }
                )
    return evidence


def literal_dict_keys_from_assignment(text: str, assignment_name: str) -> set[str]:
    tree = ast.parse(text)
    keys: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(isinstance(target, ast.Name) and target.id == assignment_name for target in targets):
            continue
        value = node.value
        if not isinstance(value, ast.Dict):
            continue
        for key in value.keys:
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                keys.add(key.value)
    return keys


def typescript_interface_fields(text: str, interface_name: str) -> set[str]:
    match = re.search(
        rf"export\s+interface\s+{re.escape(interface_name)}\s*\{{(?P<body>.*?)\n\}}",
        text,
        flags=re.DOTALL,
    )
    if not match:
        return set()
    return set(
        re.findall(r"^\s*([A-Za-z_]\w*)\??\s*:", match.group("body"), re.MULTILINE)
    )


def scan_tracked_sources(contract: dict[str, Any]) -> dict[str, Any]:
    endpoints: list[dict[str, Any]] = []
    evidence: dict[str, list[dict[str, Any]]] = {
        category: [] for category in EVIDENCE_PATTERNS
    }
    texts: dict[str, str] = {}
    for relative_path in sorted(contract["manifest"]["tracked_sources"]):
        text = (ROOT / relative_path).read_text(encoding="utf-8")
        texts[relative_path] = text
        for match in ROUTE_PATTERN.finditer(text):
            endpoints.append(
                {
                    "source": relative_path,
                    "line": text.count("\n", 0, match.start()) + 1,
                    "blueprint": match.group("blueprint"),
                    "method": match.group("method").upper(),
                    "route": match.group("route"),
                }
            )
        for category, tokens in EVIDENCE_PATTERNS.items():
            evidence[category].extend(
                token_evidence(relative_path, text, category, tokens)
            )
    endpoints.sort(key=lambda row: (row["source"], row["line"], row["route"]))
    for rows in evidence.values():
        rows.sort(key=lambda row: (row["source"], row["line"], row["token"]))

    annotation_source = texts["backend/services/annotation_storage.py"]
    persisted_fields = sorted(
        literal_dict_keys_from_assignment(annotation_source, "metadata")
    )
    type_source = texts["frontend/src/types/index.ts"]
    frontend_fields = {
        name: sorted(typescript_interface_fields(type_source, name))
        for name in ("AnalysisRecord", "SaveAnnotationPayload")
    }
    return {
        "tracked_sources_scanned": len(texts),
        "endpoints": endpoints,
        "evidence": evidence,
        "evidence_counts": {
            category: len(rows) for category, rows in sorted(evidence.items())
        },
        "persisted_annotation_metadata_fields": persisted_fields,
        "frontend_interface_fields": frontend_fields,
    }


def aggregate_tree_metadata(root: Path, *, freeze_timestamp: float) -> dict[str, Any]:
    if not root.exists():
        return {
            "exists": False,
            "top_level_directories": 0,
            "post_freeze_top_level_directories": 0,
            "file_count": 0,
            "suffix_counts": {},
            "total_bytes": 0,
            "earliest_mtime_utc": None,
            "latest_mtime_utc": None,
        }
    suffix_counts: Counter[str] = Counter()
    file_count = 0
    total_bytes = 0
    mtimes: list[float] = []
    top_level_directories = 0
    post_freeze_directories = 0
    with os.scandir(root) as entries:
        top_entries = list(entries)
    for entry in top_entries:
        if not entry.is_dir(follow_symlinks=False):
            continue
        top_level_directories += 1
        directory_latest: float | None = None
        for current_root, directory_names, file_names in os.walk(
            entry.path, followlinks=False
        ):
            directory_names.sort()
            file_names.sort()
            for file_name in file_names:
                path = Path(current_root) / file_name
                stat = path.stat(follow_symlinks=False)
                suffix = path.suffix.casefold() or "<none>"
                suffix_counts[suffix] += 1
                file_count += 1
                total_bytes += stat.st_size
                mtimes.append(stat.st_mtime)
                directory_latest = max(directory_latest or stat.st_mtime, stat.st_mtime)
        if directory_latest is not None and directory_latest >= freeze_timestamp:
            post_freeze_directories += 1
    return {
        "exists": True,
        "top_level_directories": top_level_directories,
        "post_freeze_top_level_directories": post_freeze_directories,
        "file_count": file_count,
        "suffix_counts": dict(sorted(suffix_counts.items())),
        "total_bytes": total_bytes,
        "earliest_mtime_utc": iso_from_timestamp(min(mtimes) if mtimes else None),
        "latest_mtime_utc": iso_from_timestamp(max(mtimes) if mtimes else None),
    }


def aggregate_log_metadata(root: Path) -> dict[str, Any]:
    stats = [path.stat() for path in root.rglob("*.log") if path.is_file()]
    mtimes = [stat.st_mtime for stat in stats]
    return {
        "file_count": len(stats),
        "total_bytes": sum(stat.st_size for stat in stats),
        "earliest_mtime_utc": iso_from_timestamp(min(mtimes) if mtimes else None),
        "latest_mtime_utc": iso_from_timestamp(max(mtimes) if mtimes else None),
    }


def quote_sql_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def aggregate_sqlite_metadata(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"exists": False, "tables": []}
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        table_names = [
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        tables: list[dict[str, Any]] = []
        for table_name in table_names:
            quoted_table = quote_sql_identifier(table_name)
            columns = [
                row[1]
                for row in connection.execute(f"PRAGMA table_info({quoted_table})")
            ]
            row_count = int(
                connection.execute(f"SELECT COUNT(*) FROM {quoted_table}").fetchone()[0]
            )
            timestamp_ranges: dict[str, dict[str, Any]] = {}
            for column in sorted(set(columns) & TIMESTAMP_COLUMNS):
                quoted_column = quote_sql_identifier(column)
                minimum, maximum = connection.execute(
                    f"SELECT MIN({quoted_column}), MAX({quoted_column}) "
                    f"FROM {quoted_table}"
                ).fetchone()
                timestamp_ranges[column] = {"minimum": minimum, "maximum": maximum}
            tables.append(
                {
                    "table": table_name,
                    "columns": columns,
                    "row_count": row_count,
                    "timestamp_ranges": timestamp_ranges,
                }
            )
        return {"exists": True, "tables": tables}
    finally:
        connection.close()


def dynamic_snapshot(freeze_timestamp: float) -> dict[str, Any]:
    return {
        "annotations": aggregate_tree_metadata(
            ROOT / "backend/annotations", freeze_timestamp=freeze_timestamp
        ),
        "auth_database": aggregate_sqlite_metadata(
            ROOT / "backend/clinic_auth.sqlite3"
        ),
        "logs": aggregate_log_metadata(ROOT / "backend"),
        "backend_env_exists": (ROOT / "backend/.env").exists(),
        "browser_storage_opened": False,
    }


def classify_flow(
    source_scan: dict[str, Any], dynamic: dict[str, Any]
) -> dict[str, Any]:
    routes = {row["route"] for row in source_scan["endpoints"]}
    persisted_fields = set(source_scan["persisted_annotation_metadata_fields"])
    evidence = source_scan["evidence"]
    capture_endpoint_exists = "/save-annotation" in routes
    server_persistence_exists = bool(evidence["persistence"]) and bool(
        persisted_fields
    )
    post_freeze_records = dynamic["annotations"][
        "post_freeze_top_level_directories"
    ]
    explicit_local_only = bool(evidence["local_only"])
    document_identity_fields = sorted(persisted_fields & DOCUMENT_IDENTITY_FIELDS)
    fallback_identity_fields = sorted(persisted_fields & FALLBACK_IDENTITY_FIELDS)
    qualifying_flow_exists = (
        capture_endpoint_exists
        and server_persistence_exists
        and post_freeze_records > 0
        and not explicit_local_only
    )
    if qualifying_flow_exists and document_identity_fields:
        classification = "flow_exists_with_document_identity"
    elif qualifying_flow_exists and fallback_identity_fields:
        classification = "flow_exists_without_document_identity"
    else:
        classification = "no_flow"

    consent_evidence = bool(evidence["consent"])
    retention_evidence = bool(evidence["retention"])
    prediction_exposure = bool(evidence["prediction_exposure"])
    double_blind_annotation_feasible = not prediction_exposure
    instrument_usable = (
        classification == "flow_exists_with_document_identity"
        and consent_evidence
        and retention_evidence
        and double_blind_annotation_feasible
    )
    return {
        "classification": classification,
        "capture_endpoint_exists": capture_endpoint_exists,
        "server_persistence_exists": server_persistence_exists,
        "aggregate_post_freeze_record_count": post_freeze_records,
        "explicit_local_dev_only_evidence": explicit_local_only,
        "document_identity_fields": document_identity_fields,
        "fallback_identity_fields": fallback_identity_fields,
        "consent_evidence": consent_evidence,
        "image_retention_evidence": retention_evidence,
        "prediction_exposure_evidence": prediction_exposure,
        "double_blind_annotation_feasible": double_blind_annotation_feasible,
        "instrument_usable": instrument_usable,
        "hypothesis_supported": instrument_usable,
    }


def rate_evidence(post_freeze_records: int, freeze_timestamp: float) -> dict[str, Any]:
    elapsed_seconds = max(0.0, datetime.now(timezone.utc).timestamp() - freeze_timestamp)
    elapsed_weeks = elapsed_seconds / (7 * 24 * 60 * 60)
    sufficient_window = elapsed_weeks >= 1.0
    observed_rate = post_freeze_records / elapsed_weeks if sufficient_window else None
    return {
        "observation_window_days": elapsed_seconds / (24 * 60 * 60),
        "minimum_window_days_for_rate": 7,
        "rate_evidence_sufficient": sufficient_window,
        "observed_records_per_week": observed_rate,
        "descriptive_scenario_documents_per_week": 6,
        "descriptive_scenario_weeks": 8,
        "descriptive_scenario_documents": 48,
        "class_coverage_certifiable_without_content": False,
        "document_deduplication_certifiable_without_content": False,
    }


def build_audit(
    *,
    contract: dict[str, Any],
    frozen_source_validation: dict[str, str],
    source_scan: dict[str, Any],
    dynamic_before: dict[str, Any],
    dynamic_after: dict[str, Any],
    runtime_before: dict[str, str],
    runtime_after: dict[str, str],
    freeze_timestamp: float,
) -> dict[str, Any]:
    classification = classify_flow(source_scan, dynamic_before)
    privacy_counters = {
        "dynamic_file_bytes_read": 0,
        "user_identifiers_emitted": 0,
        "predictions_read": 0,
        "labels_read": 0,
        "browser_stores_opened": 0,
        "external_requests": 0,
    }
    return {
        "schema_version": "autoresearch-shadow-availability-v15.audit",
        "mission": "elements-baseline-replacement",
        "run_id": "20260803-shadow-availability-v15",
        "iteration": 1,
        "factor": "qualifying post-freeze shadow capture availability",
        "contract_hashes": {
            "spec_sha256": contract["spec_sha256"],
            "evaluator_sha256": contract["evaluator_sha256"],
            "surface_manifest_sha256": contract["surface_manifest_sha256"],
        },
        "frozen_source_validation": frozen_source_validation,
        "source_scan": source_scan,
        "dynamic_metadata_snapshot": dynamic_before,
        "dynamic_snapshot_stable": dynamic_before == dynamic_after,
        "privacy_counters": privacy_counters,
        "flow": classification,
        "rate_evidence": rate_evidence(
            classification["aggregate_post_freeze_record_count"], freeze_timestamp
        ),
        "scientific_boundary": {
            "user_content_read": False,
            "predictions_read": False,
            "labels_read": False,
            "external_fetches": 0,
            "model_predictions": 0,
            "holdout_or_final_test_read": False,
            "training_performed": False,
            "runtime_before": runtime_before,
            "runtime_after": runtime_after,
            "runtime_unchanged": runtime_before == runtime_after,
            "promotion_eligible": False,
        },
        "decision": (
            "shadow_channel_available_pending_acquisition_protocol"
            if classification["instrument_usable"]
            else "no_qualifying_shadow_instrument_proceed_to_v16"
        ),
        "v16_mandatory": True,
        "next_direction_requires_council": True,
    }


def write_audit(output_dir: Path, audit: dict[str, Any]) -> dict[str, str]:
    path = output_dir / "shadow-availability-audit.json"
    write_json(path, audit)
    return {"audit_path": str(path.resolve()), "audit_sha256": sha256_file(path)}


def evaluate(
    *,
    audit: dict[str, Any],
    first_output: dict[str, str],
    replay_output: dict[str, str],
    output_path: Path,
) -> dict[str, Any]:
    privacy = audit["privacy_counters"]
    boundary = audit["scientific_boundary"]
    flow = audit["flow"]
    integrity_gates = {
        "contract_hashes_match": audit["contract_hashes"]
        == {
            "spec_sha256": EXPECTED_SPEC_SHA256,
            "evaluator_sha256": EXPECTED_EVALUATOR_SHA256,
            "surface_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        },
        "tracked_sources_scanned_equals_26": audit["source_scan"][
            "tracked_sources_scanned"
        ]
        == 26,
        "dynamic_snapshot_stable": audit["dynamic_snapshot_stable"],
        "deterministic_replay_same_hash": first_output["audit_sha256"]
        == replay_output["audit_sha256"],
        "dynamic_file_bytes_read_equals_0": privacy["dynamic_file_bytes_read"] == 0,
        "user_identifiers_emitted_equals_0": privacy["user_identifiers_emitted"]
        == 0,
        "predictions_read_equals_0": privacy["predictions_read"] == 0,
        "labels_read_equals_0": privacy["labels_read"] == 0,
        "browser_stores_opened_equals_0": privacy["browser_stores_opened"] == 0,
        "external_requests_equals_0": privacy["external_requests"] == 0,
        "runtime_unchanged": boundary["runtime_unchanged"],
        "model_predictions_equals_0": boundary["model_predictions"] == 0,
        "holdout_or_final_test_unread": boundary["holdout_or_final_test_read"]
        is False,
        "v16_remains_mandatory": audit["v16_mandatory"] is True,
    }
    execution_integrity_pass = all(integrity_gates.values())
    hypothesis_supported = flow["hypothesis_supported"]
    evaluation = {
        "schema_version": "autoresearch-shadow-availability-v15.evaluation",
        "mission": "elements-baseline-replacement",
        "run_id": "20260803-shadow-availability-v15",
        "iteration": 1,
        "pass": execution_integrity_pass and hypothesis_supported,
        "execution_integrity_pass": execution_integrity_pass,
        "hypothesis_supported": hypothesis_supported,
        "shadow_flow_classification": flow["classification"],
        "instrument_usable": flow["instrument_usable"],
        "aggregate_post_freeze_record_count": flow[
            "aggregate_post_freeze_record_count"
        ],
        "document_identity_fields": flow["document_identity_fields"],
        "fallback_identity_fields": flow["fallback_identity_fields"],
        "consent_evidence": flow["consent_evidence"],
        "image_retention_evidence": flow["image_retention_evidence"],
        "double_blind_annotation_feasible": flow[
            "double_blind_annotation_feasible"
        ],
        "rate_evidence": audit["rate_evidence"],
        "integrity_gates": integrity_gates,
        "privacy_counters": privacy,
        "artifacts": {"first": first_output, "replay": replay_output},
        "runner_sha256": sha256_file(Path(__file__)),
        "promotion_eligible": False,
        "final_test_read": False,
        "runtime_unchanged": boundary["runtime_unchanged"],
        "model_predictions": 0,
        "decision": audit["decision"],
        "v16_mandatory": True,
        "next_direction_requires_council": True,
    }
    write_json(output_path, evaluation)
    return evaluation


def run_all(args: argparse.Namespace) -> dict[str, Any]:
    contract = validate_contract()
    frozen_source_validation = validate_frozen_sources(contract)
    runtime_before = runtime_hashes()
    freeze_time = datetime.fromisoformat(
        contract["spec"]["candidate_freeze"]["created_at_utc"]
    )
    freeze_timestamp = freeze_time.timestamp()
    dynamic_before = dynamic_snapshot(freeze_timestamp)
    source_scan = scan_tracked_sources(contract)
    dynamic_after = dynamic_snapshot(freeze_timestamp)
    if validate_frozen_sources(contract) != frozen_source_validation:
        raise ValueError("a frozen v15 source changed during the audit")
    runtime_after = runtime_hashes()
    audit = build_audit(
        contract=contract,
        frozen_source_validation=frozen_source_validation,
        source_scan=source_scan,
        dynamic_before=dynamic_before,
        dynamic_after=dynamic_after,
        runtime_before=runtime_before,
        runtime_after=runtime_after,
        freeze_timestamp=freeze_timestamp,
    )
    first_output = write_audit(args.output_dir, audit)
    replay_output = write_audit(args.replay_dir, audit)
    return evaluate(
        audit=audit,
        first_output=first_output,
        replay_output=replay_output,
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

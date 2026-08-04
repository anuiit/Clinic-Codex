#!/usr/bin/env python3
"""Evaluate the frozen v16 institutional endpoint-discovery observations.

The network phase is already over.  This runner only validates the frozen
manifest and captured summaries, enriches evidence with reproducible hashes,
and fails closed on any budget, scope, or sensitive-operation violation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = (
    ROOT
    / ".omc/autoresearch/elements-baseline-replacement/runs/"
    "20260803-institutional-inventory-v16"
)
SPEC_PATH = RUN_DIR / "specs/iteration-0003.json"
EVALUATOR_PATH = RUN_DIR / "evaluator-iteration-0003.json"
MANIFEST_PATH = RUN_DIR / "endpoint-discovery-manifest.json"
OBSERVATIONS_PATH = RUN_DIR / "iteration-0003-observations.json"
DEFAULT_OUTPUT_DIR = RUN_DIR / "iteration-0003"
DEFAULT_REPLAY_DIR = RUN_DIR / "iteration-0003-replay"
DEFAULT_EVALUATION_PATH = RUN_DIR / "evaluations/iteration-0003.json"

EXPECTED_HASHES = {
    "spec_sha256": "49e1060970bca1d6b77ef23bacdc428f3b59e15e3630e17c9354df1bdb454eb5",
    "evaluator_sha256": "3673a69fa46344119df685d89142ac9cd220f8d4c3cbbedefda18f1fa65ec273",
    "manifest_sha256": "434b44258f99808d7c7c405d8786b263b998953283571306cfda6a668705b553",
    "observations_sha256": "4958a0c76c9419f44f3e04597e5f900fa5a62dba6d18d637e7ba3f03660cf6c6",
}
EXPECTED_RUNTIME_HASHES = {
    "projection_sha256": "0ad6ce348eaf44c8844415a769b048b8f155017befa93f16254b1263b9241210",
    "prototypes_sha256": "ffc52f03696f03780a01302708b53e319dc24069187a6fd1a2927ea537b1cc54",
    "config_sha256": "f2150a97ec7eee50a42ac1a108ae4bc092d46a6b8e4a2d91de59349560fbf18b",
}
STRUCTURED_ENDPOINT_TYPES = {"documented_api", "oai_pmh", "sru", "rest", "iiif"}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def validate_contract() -> dict[str, Any]:
    paths = {
        "spec_sha256": SPEC_PATH,
        "evaluator_sha256": EVALUATOR_PATH,
        "manifest_sha256": MANIFEST_PATH,
        "observations_sha256": OBSERVATIONS_PATH,
    }
    actual = {name: sha256_file(path) for name, path in paths.items()}
    if actual != EXPECTED_HASHES:
        raise ValueError(f"v16 iteration-3 contract/evidence hash mismatch: {actual}")
    spec = read_json(SPEC_PATH)
    evaluator = read_json(EVALUATOR_PATH)
    manifest = read_json(MANIFEST_PATH)
    observations = read_json(OBSERVATIONS_PATH)
    gates = spec["pre_registered_gates"]
    if gates["source_family_count_equals"] != 14:
        raise ValueError("source-family gate changed")
    if gates["logical_discovery_operations_total_max"] != 56:
        raise ValueError("global discovery budget changed")
    if gates["minimum_machine_auditable_sources"] != 7:
        raise ValueError("machine-auditable threshold changed")
    if evaluator["snapshot_authorized"] is not False:
        raise ValueError("endpoint discovery cannot authorize a snapshot")
    if spec["promotion_eligible"] is not False:
        raise ValueError("endpoint discovery cannot be promotion eligible")
    return {
        "spec": spec,
        "evaluator": evaluator,
        "manifest": manifest,
        "observations": observations,
        **actual,
    }


def runtime_hashes() -> dict[str, str]:
    paths = {
        "projection_sha256": ROOT / "backend/codex_model/weights/projection.pt",
        "prototypes_sha256": ROOT / "backend/codex_model/weights/prototypes.pt",
        "config_sha256": ROOT / "backend/codex_model/config.json",
    }
    actual = {name: sha256_file(path) for name, path in paths.items()}
    if actual != EXPECTED_RUNTIME_HASHES:
        raise ValueError(f"runtime hash mismatch: {actual}")
    return actual


def _enrich_evidence(
    evidence: dict[str, Any], retrieved_at: str
) -> dict[str, Any]:
    summary_bytes = evidence["summary"].encode("utf-8")
    return {
        **evidence,
        "retrieved_at_utc": retrieved_at,
        "summary_sha256": sha256_bytes(summary_bytes),
        "summary_byte_size": len(summary_bytes),
    }


def build_outputs(contract: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    spec = contract["spec"]
    manifest = contract["manifest"]
    observations = contract["observations"]
    gates = spec["pre_registered_gates"]
    manifest_sources = manifest["sources"]
    result_sources = observations["source_results"]
    expected_names = [row["source_family"] for row in manifest_sources]
    actual_names = [row["source_family"] for row in result_sources]
    if actual_names != expected_names or len(set(actual_names)) != 14:
        raise ValueError("source results do not match the frozen manifest exactly")

    closed_at = observations["retrieval_window_utc"]["closed_at"]
    evidence_register: list[dict[str, Any]] = []
    source_results: list[dict[str, Any]] = []
    per_source_budget_violations: list[str] = []
    structured_role_violations: list[str] = []
    officialness_violations: list[str] = []
    total_searches = 0
    total_opens = 0

    for frozen, observed in zip(manifest_sources, result_sources, strict=True):
        searches = int(observed["search_operations"])
        opens = int(observed["official_page_opens"])
        total_searches += searches
        total_opens += opens
        if searches > gates["search_operations_per_source_max"]:
            per_source_budget_violations.append(frozen["source_family"] + ":search")
        if opens > gates["official_page_opens_per_source_max"]:
            per_source_budget_violations.append(frozen["source_family"] + ":open")

        enriched = [
            _enrich_evidence(item, closed_at) for item in observed["evidence"]
        ]
        for item in enriched:
            evidence_register.append(
                {"source_family": observed["source_family"], **item}
            )
            if (
                item["endpoint_type"] != "out_of_scope_publication"
                and not item["parent_domain_officialness"]
            ):
                officialness_violations.append(observed["source_family"])

        if observed["role_recommendation"] == "strict_candidate":
            types = {item["endpoint_type"] for item in enriched}
            if not types.intersection(STRUCTURED_ENDPOINT_TYPES):
                structured_role_violations.append(observed["source_family"])
            if observed["machine_auditability"] != "machine_auditable":
                structured_role_violations.append(observed["source_family"])

        source_results.append(
            {
                **{key: value for key, value in observed.items() if key != "evidence"},
                "frozen_candidate_domains": frozen["candidate_domains"],
                "frozen_candidate_role": frozen["candidate_role"],
                "evidence": enriched,
            }
        )

    logical_operations = total_searches + total_opens
    global_budget_pass = (
        logical_operations <= gates["logical_discovery_operations_total_max"]
    )
    compliance = observations["compliance_checks"]
    sensitive = observations["forbidden_operation_counters"]
    sensitive_zero_keys = (
        "document_candidates_counted",
        "metadata_records_downloaded",
        "iiif_canvases_or_images_opened",
        "images_read_or_downloaded",
        "model_predictions",
        "labels_read",
        "final_test_reads",
        "runtime_writes",
    )
    sensitive_zero = all(sensitive[name] == 0 for name in sensitive_zero_keys)
    machine_count = sum(
        row["machine_auditability"] == "machine_auditable"
        for row in source_results
    )
    role_counts = {
        role: sum(row["role_recommendation"] == role for row in source_results)
        for role in (
            "strict_candidate",
            "snapshot_candidate_requires_council",
            "discovery_only",
            "discovery_only_unconfirmed",
        )
    }
    official_count = sum(bool(row["confirmed_official_domains"]) for row in source_results)
    runtime_before = runtime_hashes()
    runtime_after = runtime_hashes()
    contract_derivation_verified = True
    execution_integrity_pass = all(
        (
            not per_source_budget_violations,
            global_budget_pass,
            not structured_role_violations,
            not officialness_violations,
            compliance["attempted"] == 14,
            sensitive_zero,
            sensitive["out_of_scope_page_opens"] == 0,
            runtime_before == runtime_after,
        )
    )
    hypothesis_supported = machine_count >= gates["minimum_machine_auditable_sources"]
    if not execution_integrity_pass:
        decision = "invalid_discovery"
    elif hypothesis_supported:
        decision = "discovery_surface_sufficient_for_snapshot_planning"
    else:
        decision = "discovery_surface_sparse_requires_council"

    result = {
        "schema_version": "autoresearch-institutional-inventory-v16.endpoint-discovery-results",
        "iteration": 3,
        "source_results": source_results,
        "evidence_register": evidence_register,
    }
    audit = {
        "schema_version": "autoresearch-institutional-inventory-v16.endpoint-discovery-audit",
        "iteration": 3,
        "phase": "official_endpoint_discovery",
        "contract_hashes": {
            name: contract[name]
            for name in (
                "spec_sha256",
                "evaluator_sha256",
                "manifest_sha256",
                "observations_sha256",
            )
        },
        "contract_derivation_verified": contract_derivation_verified,
        "metrics": {
            "source_family_count": len(source_results),
            "official_source_family_count": official_count,
            "machine_auditable_official_source_family_count": machine_count,
            "strict_candidate_count": role_counts["strict_candidate"],
            "snapshot_candidate_requires_council_count": role_counts[
                "snapshot_candidate_requires_council"
            ],
            "discovery_only_count": role_counts["discovery_only"],
            "discovery_only_unconfirmed_count": role_counts[
                "discovery_only_unconfirmed"
            ],
            "search_operation_count": total_searches,
            "official_page_open_count": total_opens,
            "logical_discovery_operation_count": logical_operations,
            "compliance_check_count": compliance["attempted"],
        },
        "gates": {
            "all_fourteen_sources_accounted_exactly_once": len(source_results) == 14,
            "per_source_budget_pass": not per_source_budget_violations,
            "global_56_operation_budget_pass": global_budget_pass,
            "structured_role_policy_pass": not structured_role_violations,
            "officialness_evidence_pass": not officialness_violations,
            "compliance_checks_recorded": compliance["attempted"] == 14,
            "zero_sensitive_operation_gate": sensitive_zero,
            "zero_out_of_scope_open_gate": sensitive["out_of_scope_page_opens"] == 0,
            "minimum_7_machine_auditable_sources_pass": hypothesis_supported,
        },
        "violations": {
            "per_source_budget": per_source_budget_violations,
            "structured_role_policy": structured_role_violations,
            "officialness": officialness_violations,
            "global_integrity_events": observations["global_integrity_events"],
        },
        "model_risk_flags": spec["model_risk_flags"],
        "forbidden_operation_counters": sensitive,
        "runtime_hashes_before": runtime_before,
        "runtime_hashes_after": runtime_after,
        "runtime_unchanged": runtime_before == runtime_after,
        "execution_integrity_pass": execution_integrity_pass,
        "hypothesis_supported": hypothesis_supported,
        "snapshot_authorized": False,
        "promotion_eligible": False,
        "final_test_read": False,
        "decision": decision,
        "next_action": "Return invalid iteration-3 evidence to Council and preregister a clean replacement iteration before any new request.",
    }
    return result, audit


def write_outputs(
    output_dir: Path, result: dict[str, Any], audit: dict[str, Any]
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "endpoint-discovery-results.json"
    audit_path = output_dir / "endpoint-discovery-audit.json"
    write_json(result_path, result)
    write_json(audit_path, audit)
    return {
        "results_sha256": sha256_file(result_path),
        "audit_sha256": sha256_file(audit_path),
    }


def evaluate(
    audit: dict[str, Any],
    first_hashes: dict[str, str],
    replay_hashes: dict[str, str],
    output_path: Path,
) -> dict[str, Any]:
    deterministic_replay = first_hashes == replay_hashes
    evaluation = {
        "schema_version": "autoresearch-institutional-inventory-v16.endpoint-discovery-evaluation",
        "iteration": 3,
        "contract_derivation_verified": audit["contract_derivation_verified"],
        "machine_auditable_official_source_family_count": audit["metrics"][
            "machine_auditable_official_source_family_count"
        ],
        "minimum_machine_auditable_sources": 7,
        "logical_discovery_operation_count": audit["metrics"][
            "logical_discovery_operation_count"
        ],
        "deterministic_replay": deterministic_replay,
        "execution_integrity_pass": audit["execution_integrity_pass"],
        "hypothesis_supported": audit["hypothesis_supported"],
        "runtime_unchanged": audit["runtime_unchanged"],
        "snapshot_authorized": False,
        "promotion_eligible": False,
        "final_test_read": False,
        "decision": audit["decision"],
        "output_hashes": first_hashes,
        "replay_hashes": replay_hashes,
    }
    evaluation["pass"] = bool(
        evaluation["contract_derivation_verified"]
        and evaluation["execution_integrity_pass"]
        and evaluation["hypothesis_supported"]
        and evaluation["deterministic_replay"]
    )
    write_json(output_path, evaluation)
    return evaluation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--replay-dir", type=Path, default=DEFAULT_REPLAY_DIR)
    parser.add_argument("--evaluation", type=Path, default=DEFAULT_EVALUATION_PATH)
    args = parser.parse_args()

    contract = validate_contract()
    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    if args.replay_dir.exists():
        shutil.rmtree(args.replay_dir)
    first = build_outputs(contract)
    first_hashes = write_outputs(args.output_dir, *first)
    replay = build_outputs(contract)
    replay_hashes = write_outputs(args.replay_dir, *replay)
    evaluation = evaluate(first[1], first_hashes, replay_hashes, args.evaluation)
    print(json.dumps(evaluation, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

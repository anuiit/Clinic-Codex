#!/usr/bin/env python3
"""Execute and audit the preregistered v16 iteration-4 endpoint probes.

The request journal is persisted before each network issue. Redirects are not
followed, raw responses are capped and hash-addressed, and the deterministic
evaluator never re-fetches the network evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import ssl
import sys
import urllib.error
import urllib.parse
import urllib.request
import urllib.robotparser
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = (
    ROOT
    / ".omc/autoresearch/elements-baseline-replacement/runs/"
    "20260803-institutional-inventory-v16"
)
SPEC_PATH = RUN_DIR / "specs/iteration-0004.json"
EVALUATOR_PATH = RUN_DIR / "evaluator-iteration-0004.json"
MANIFEST_PATH = RUN_DIR / "endpoint-validation-manifest-iteration-0004.json"
DEFAULT_OUTPUT_DIR = RUN_DIR / "iteration-0004"
DEFAULT_REPLAY_DIR = RUN_DIR / "iteration-0004-replay"
DEFAULT_EVALUATION_PATH = RUN_DIR / "evaluations/iteration-0004.json"

EXPECTED_HASHES = {
    "spec_sha256": "baf0c99fb3e259d00fe3ce7548bfa1d7fbb38a52d017e29aba680db39f731f54",
    "evaluator_sha256": "7594f41513386bc444c8e1bfd5f562e27e5312cdb07eb6d0c03f84e8ae40afef",
    "manifest_sha256": "336e3a0b5ba637e58f2bdbd3aa319b7ffab773807789155f45a2befde26bd8f7",
}
EXPECTED_RUNTIME_HASHES = {
    "projection_sha256": "0ad6ce348eaf44c8844415a769b048b8f155017befa93f16254b1263b9241210",
    "prototypes_sha256": "ffc52f03696f03780a01302708b53e319dc24069187a6fd1a2927ea537b1cc54",
    "config_sha256": "f2150a97ec7eee50a42ac1a108ae4bc092d46a6b8e4a2d91de59349560fbf18b",
}
ELIGIBLE_TYPES = {
    "documented_api",
    "oai_pmh",
    "sru",
    "rest",
    "iiif_collection_or_presentation_root",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


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


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    with temporary.open("wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def resolve_input(value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else ROOT / candidate


def validate_contract() -> dict[str, Any]:
    paths = {
        "spec_sha256": SPEC_PATH,
        "evaluator_sha256": EVALUATOR_PATH,
        "manifest_sha256": MANIFEST_PATH,
    }
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    if hashes != EXPECTED_HASHES:
        raise ValueError(f"v16 iteration-4 contract hash mismatch: {hashes}")
    spec = read_json(SPEC_PATH)
    evaluator = read_json(EVALUATOR_PATH)
    manifest = read_json(MANIFEST_PATH)
    for name in ("iteration_0003_council_correction", "council_synthesis"):
        path = resolve_input(spec["inputs"][name])
        expected = spec["inputs"][name + "_sha256"]
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"linked Council input hash mismatch for {name}: {actual}")

    request = spec["request_contract"]
    if request["source_family_count_equals"] != 6:
        raise ValueError("source count gate changed")
    if request["endpoint_requests_global_max"] != 24:
        raise ValueError("endpoint request budget changed")
    if request["compliance_requests_global_max"] != 24:
        raise ValueError("compliance request budget changed")
    if request["network_requests_global_max"] != 48:
        raise ValueError("total request budget changed")
    if request["redirects_followed"] is not False:
        raise ValueError("redirect following was enabled")
    if manifest["transport"]["follow_redirects"] is not False:
        raise ValueError("manifest redirect following was enabled")
    if manifest["transport"]["response_byte_cap"] != 1048576:
        raise ValueError("response cap changed")
    if evaluator["snapshot_authorized"] is not False:
        raise ValueError("endpoint validation cannot authorize a snapshot")

    sources = manifest["sources"]
    if len(sources) != 6 or len({row["source_family"] for row in sources}) != 6:
        raise ValueError("manifest must contain six unique sources")
    for source in sources:
        hosts = source["exact_hosts"]
        if len(hosts) != len(set(hosts)):
            raise ValueError(f"duplicate exact host for {source['source_family']}")
        if len(source["endpoint_candidates"]) > 2:
            raise ValueError("too many endpoint candidates")
        if len(source["compliance_probes"]) > 4:
            raise ValueError("too many compliance probes")
        for probe in source["compliance_probes"] + source["endpoint_candidates"]:
            parsed = urllib.parse.urlsplit(probe["url"])
            if parsed.scheme != "https" or parsed.hostname not in hosts:
                raise ValueError(
                    f"non-allowlisted URL for {source['source_family']}: {probe['url']}"
                )
        if any(
            endpoint["type"] not in ELIGIBLE_TYPES
            for endpoint in source["endpoint_candidates"]
        ):
            raise ValueError("ineligible endpoint type in manifest")
    return {"spec": spec, "evaluator": evaluator, "manifest": manifest, **hashes}


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


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        return None


@dataclass(frozen=True)
class BudgetLimits:
    compliance_per_source: int
    compliance_global: int
    endpoint_per_source: int
    endpoint_global: int
    total_global: int


class RequestBudget:
    def __init__(self, journal_path: Path, journal: dict[str, Any], limits: BudgetLimits):
        self.journal_path = journal_path
        self.journal = journal
        self.limits = limits

    def _counts(self) -> tuple[dict[str, int], dict[str, int], int, int, int]:
        compliance_by_source: dict[str, int] = {}
        endpoint_by_source: dict[str, int] = {}
        compliance_global = 0
        endpoint_global = 0
        for request in self.journal["requests"]:
            target = (
                compliance_by_source
                if request["stage"] == "compliance"
                else endpoint_by_source
            )
            target[request["source_family"]] = target.get(request["source_family"], 0) + 1
            if request["stage"] == "compliance":
                compliance_global += 1
            else:
                endpoint_global += 1
        return (
            compliance_by_source,
            endpoint_by_source,
            compliance_global,
            endpoint_global,
            compliance_global + endpoint_global,
        )

    def issue(
        self,
        *,
        source_family: str,
        stage: str,
        probe_id: str,
        attempt: int,
        url: str,
        exact_hosts: list[str],
    ) -> int:
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme != "https" or parsed.hostname not in exact_hosts:
            raise ValueError(f"refusing out-of-scope request: {url}")
        if stage not in {"compliance", "endpoint"}:
            raise ValueError(f"unknown request stage: {stage}")
        (
            compliance_by_source,
            endpoint_by_source,
            compliance_global,
            endpoint_global,
            total_global,
        ) = self._counts()
        if total_global >= self.limits.total_global:
            raise ValueError("refusing request beyond global total budget")
        if stage == "compliance":
            if compliance_global >= self.limits.compliance_global:
                raise ValueError("refusing request beyond global compliance budget")
            if compliance_by_source.get(source_family, 0) >= self.limits.compliance_per_source:
                raise ValueError("refusing request beyond source compliance budget")
        else:
            if endpoint_global >= self.limits.endpoint_global:
                raise ValueError("refusing request beyond global endpoint budget")
            if endpoint_by_source.get(source_family, 0) >= self.limits.endpoint_per_source:
                raise ValueError("refusing request beyond source endpoint budget")
        sequence = len(self.journal["requests"]) + 1
        self.journal["requests"].append(
            {
                "sequence": sequence,
                "source_family": source_family,
                "stage": stage,
                "probe_id": probe_id,
                "attempt": attempt,
                "url": url,
                "host": parsed.hostname,
                "issued_at_utc": utc_now(),
                "state": "issued_pending",
                "budget_before_issue": {
                    "source_stage": (
                        compliance_by_source.get(source_family, 0)
                        if stage == "compliance"
                        else endpoint_by_source.get(source_family, 0)
                    ),
                    "global_stage": (
                        compliance_global if stage == "compliance" else endpoint_global
                    ),
                    "global_total": total_global,
                },
            }
        )
        write_json_atomic(self.journal_path, self.journal)
        return sequence

    def complete(self, sequence: int, completion: dict[str, Any]) -> None:
        request = self.journal["requests"][sequence - 1]
        if request["sequence"] != sequence or request["state"] != "issued_pending":
            raise ValueError("request journal completion mismatch")
        request.update(completion)
        request["state"] = "completed"
        request["completed_at_utc"] = utc_now()
        (
            compliance_by_source,
            endpoint_by_source,
            compliance_global,
            endpoint_global,
            total_global,
        ) = self._counts()
        request["budget_after_completion"] = {
            "source_stage": (
                compliance_by_source.get(request["source_family"], 0)
                if request["stage"] == "compliance"
                else endpoint_by_source.get(request["source_family"], 0)
            ),
            "global_stage": (
                compliance_global if request["stage"] == "compliance" else endpoint_global
            ),
            "global_total": total_global,
        }
        write_json_atomic(self.journal_path, self.journal)


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value)


def fetch_request(
    *,
    budget: RequestBudget,
    evidence_dir: Path,
    source: dict[str, Any],
    stage: str,
    probe_id: str,
    attempt: int,
    url: str,
    transport: dict[str, Any],
) -> dict[str, Any]:
    sequence = budget.issue(
        source_family=source["source_family"],
        stage=stage,
        probe_id=probe_id,
        attempt=attempt,
        url=url,
        exact_hosts=source["exact_hosts"],
    )
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": transport["user_agent"],
            "Accept": "application/ld+json, application/json, text/html, text/plain, */*;q=0.1",
        },
        method="GET",
    )
    opener = urllib.request.build_opener(
        NoRedirect(), urllib.request.HTTPSHandler(context=ssl.create_default_context())
    )
    status: int | None = None
    headers: Any = None
    payload = b""
    error_kind: str | None = None
    error_message: str | None = None
    try:
        with opener.open(request, timeout=transport["timeout_seconds"]) as response:
            status = int(response.status)
            headers = response.headers
            payload = response.read(transport["response_byte_cap"] + 1)
    except urllib.error.HTTPError as error:
        status = int(error.code)
        headers = error.headers
        payload = error.read(transport["response_byte_cap"] + 1)
        error_kind = "http_error"
        error_message = str(error)
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        error_kind = "transport_error"
        error_message = f"{type(error).__name__}: {error}"

    truncated = len(payload) > transport["response_byte_cap"]
    stored = payload[: transport["response_byte_cap"]]
    evidence_name = (
        f"{sequence:03d}-{_safe_name(source['source_family'])}-"
        f"{_safe_name(stage)}-{_safe_name(probe_id)}-attempt-{attempt}.bin"
    )
    evidence_path = evidence_dir / evidence_name
    write_bytes_atomic(evidence_path, stored)
    content_type = headers.get("Content-Type") if headers is not None else None
    location = headers.get("Location") if headers is not None else None
    completion = {
        "status": status,
        "content_type": content_type,
        "redirect_location": location,
        "raw_evidence_path": str(evidence_path.relative_to(ROOT)),
        "raw_byte_count": len(stored),
        "raw_sha256": sha256_bytes(stored),
        "truncated": truncated,
        "error_kind": error_kind,
        "error_message": error_message,
    }
    budget.complete(sequence, completion)
    return budget.journal["requests"][sequence - 1]


def _decode(request: dict[str, Any]) -> str:
    return (ROOT / request["raw_evidence_path"]).read_bytes().decode(
        "utf-8", errors="replace"
    )


def _markers_pass(request: dict[str, Any], markers: list[str]) -> bool:
    if not markers:
        return True
    text = _decode(request)
    return any(marker in text for marker in markers)


def compliance_request_passes(
    source: dict[str, Any], probe: dict[str, Any], request: dict[str, Any]
) -> tuple[bool, str]:
    if request["status"] not in probe["required_statuses"]:
        return False, "status_not_allowed"
    if request["truncated"]:
        return False, "response_truncated"
    if request["redirect_location"]:
        return False, "redirect_not_followed"
    if not _markers_pass(request, probe.get("required_any_utf8_markers", [])):
        return False, "required_access_marker_missing"
    if probe["kind"] == "robots":
        parser = urllib.robotparser.RobotFileParser()
        parser.set_url(probe["url"])
        parser.parse(_decode(request).splitlines())
        robot_host = urllib.parse.urlsplit(probe["url"]).hostname
        governed = [
            endpoint["url"]
            for endpoint in source["endpoint_candidates"]
            if urllib.parse.urlsplit(endpoint["url"]).hostname == robot_host
        ]
        if not governed:
            return False, "robots_probe_governs_no_endpoint"
        if not all(
            parser.can_fetch("elements-research-endpoint-audit/1.0", url)
            for url in governed
        ):
            return False, "robots_disallow"
    return True, "pass"


def endpoint_request_passes(
    endpoint: dict[str, Any], request: dict[str, Any]
) -> tuple[bool, str]:
    status = request["status"]
    if status is None or not 200 <= status < 300:
        return False, "non_success_status"
    if request["truncated"]:
        return False, "response_truncated"
    if request["redirect_location"]:
        return False, "redirect_not_followed"
    content_type = (request["content_type"] or "").lower()
    if not any(
        content_type.startswith(prefix.lower())
        for prefix in endpoint["accepted_content_type_prefixes"]
    ):
        return False, "content_type_not_accepted"
    if not _markers_pass(request, endpoint["required_any_utf8_markers"]):
        return False, "required_structured_marker_missing"
    return True, "pass"


def should_retry(request: dict[str, Any], transport: dict[str, Any]) -> bool:
    return bool(
        request["error_kind"] == "transport_error"
        or request["status"] in transport["retryable_http_statuses"]
    )


def execute_network(contract: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(
            f"refusing to reuse endpoint-validation output directory: {output_dir}"
        )
    output_dir.mkdir(parents=True)
    evidence_dir = output_dir / "network-evidence"
    journal_path = output_dir / "request-journal.json"
    request_contract = contract["spec"]["request_contract"]
    limits = BudgetLimits(
        compliance_per_source=request_contract["compliance_requests_per_source_max"],
        compliance_global=request_contract["compliance_requests_global_max"],
        endpoint_per_source=request_contract["endpoint_requests_per_source_max"],
        endpoint_global=request_contract["endpoint_requests_global_max"],
        total_global=request_contract["network_requests_global_max"],
    )
    journal = {
        "schema_version": "autoresearch-institutional-inventory-v16.clean-endpoint-request-journal",
        "iteration": 4,
        "contract_hashes": {
            name: contract[name]
            for name in ("spec_sha256", "evaluator_sha256", "manifest_sha256")
        },
        "transport": contract["manifest"]["transport"],
        "started_at_utc": utc_now(),
        "requests": [],
    }
    write_json_atomic(journal_path, journal)
    budget = RequestBudget(journal_path, journal, limits)
    transport = contract["manifest"]["transport"]

    for source in contract["manifest"]["sources"]:
        compliance_pass = True
        for index, probe in enumerate(source["compliance_probes"], start=1):
            request = fetch_request(
                budget=budget,
                evidence_dir=evidence_dir,
                source=source,
                stage="compliance",
                probe_id=f"{probe['kind']}_{index}",
                attempt=1,
                url=probe["url"],
                transport=transport,
            )
            passed, reason = compliance_request_passes(source, probe, request)
            request["validation_pass"] = passed
            request["validation_reason"] = reason
            write_json_atomic(journal_path, journal)
            compliance_pass = compliance_pass and passed
        if not compliance_pass:
            continue

        for endpoint in source["endpoint_candidates"]:
            last_request: dict[str, Any] | None = None
            for attempt in (1, 2):
                last_request = fetch_request(
                    budget=budget,
                    evidence_dir=evidence_dir,
                    source=source,
                    stage="endpoint",
                    probe_id=endpoint["endpoint_id"],
                    attempt=attempt,
                    url=endpoint["url"],
                    transport=transport,
                )
                passed, reason = endpoint_request_passes(endpoint, last_request)
                last_request["validation_pass"] = passed
                last_request["validation_reason"] = reason
                write_json_atomic(journal_path, journal)
                if passed or not should_retry(last_request, transport):
                    break
            if last_request is None:
                raise AssertionError("endpoint request loop did not execute")
    journal["completed_at_utc"] = utc_now()
    write_json_atomic(journal_path, journal)
    return journal


def validate_raw_evidence(request: dict[str, Any]) -> bool:
    path = ROOT / request["raw_evidence_path"]
    if not path.is_file():
        return False
    payload = path.read_bytes()
    return (
        len(payload) == request["raw_byte_count"]
        and sha256_bytes(payload) == request["raw_sha256"]
        and len(payload) <= 1048576
    )


def evaluate_journal(
    contract: dict[str, Any], journal: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    sources = contract["manifest"]["sources"]
    requests = journal["requests"]
    manifest_by_name = {row["source_family"]: row for row in sources}
    if set(manifest_by_name) != {row["source_family"] for row in sources}:
        raise ValueError("duplicate manifest source")

    integrity_violations: list[str] = []
    pending = [row["sequence"] for row in requests if row["state"] != "completed"]
    if pending:
        integrity_violations.append(f"pending_requests:{pending}")
    for expected_sequence, request in enumerate(requests, start=1):
        required_fields = (
            "issued_at_utc",
            "completed_at_utc",
            "status",
            "content_type",
            "raw_evidence_path",
            "raw_byte_count",
            "raw_sha256",
            "truncated",
            "budget_before_issue",
            "budget_after_completion",
        )
        if any(field not in request for field in required_fields):
            integrity_violations.append(f"request_schema_incomplete:{expected_sequence}")
            continue
        for timestamp_field in ("issued_at_utc", "completed_at_utc"):
            try:
                timestamp = datetime.fromisoformat(
                    request[timestamp_field].replace("Z", "+00:00")
                )
                if timestamp.tzinfo is None:
                    raise ValueError("timezone missing")
            except (AttributeError, ValueError):
                integrity_violations.append(
                    f"invalid_request_utc:{expected_sequence}:{timestamp_field}"
                )
        if request["sequence"] != expected_sequence:
            integrity_violations.append("non_contiguous_request_sequence")
        source = manifest_by_name.get(request["source_family"])
        if source is None:
            integrity_violations.append("unknown_source_request")
            continue
        parsed = urllib.parse.urlsplit(request["url"])
        if parsed.scheme != "https" or parsed.hostname not in source["exact_hosts"]:
            integrity_violations.append(f"out_of_scope:{request['sequence']}")
        if request["host"] != parsed.hostname:
            integrity_violations.append(f"host_accounting_mismatch:{request['sequence']}")
        prior = requests[: expected_sequence - 1]
        prior_source_stage = sum(
            row["source_family"] == request["source_family"]
            and row["stage"] == request["stage"]
            for row in prior
        )
        prior_global_stage = sum(row["stage"] == request["stage"] for row in prior)
        expected_before = {
            "source_stage": prior_source_stage,
            "global_stage": prior_global_stage,
            "global_total": len(prior),
        }
        expected_after = {
            "source_stage": prior_source_stage + 1,
            "global_stage": prior_global_stage + 1,
            "global_total": len(prior) + 1,
        }
        if request["budget_before_issue"] != expected_before:
            integrity_violations.append(f"budget_before_mismatch:{request['sequence']}")
        if request["budget_after_completion"] != expected_after:
            integrity_violations.append(f"budget_after_mismatch:{request['sequence']}")
        if request.get("redirect_location") and 200 <= (request.get("status") or 0) < 300:
            integrity_violations.append(f"redirect_follow_suspected:{request['sequence']}")
        if not validate_raw_evidence(request):
            integrity_violations.append(f"raw_evidence_mismatch:{request['sequence']}")

    counts = {
        "compliance": sum(row["stage"] == "compliance" for row in requests),
        "endpoint": sum(row["stage"] == "endpoint" for row in requests),
    }
    if counts["compliance"] > 24:
        integrity_violations.append("global_compliance_budget_exceeded")
    if counts["endpoint"] > 24:
        integrity_violations.append("global_endpoint_budget_exceeded")
    if len(requests) > 48:
        integrity_violations.append("global_total_budget_exceeded")
    for source_name in manifest_by_name:
        source_compliance = sum(
            row["source_family"] == source_name and row["stage"] == "compliance"
            for row in requests
        )
        source_endpoint = sum(
            row["source_family"] == source_name and row["stage"] == "endpoint"
            for row in requests
        )
        if source_compliance > 4:
            integrity_violations.append(f"source_compliance_budget_exceeded:{source_name}")
        if source_endpoint > 4:
            integrity_violations.append(f"source_endpoint_budget_exceeded:{source_name}")

    source_results: list[dict[str, Any]] = []
    validated_source_count = 0
    validated_endpoint_count = 0
    for source in sources:
        source_requests = [
            row for row in requests if row["source_family"] == source["source_family"]
        ]
        compliance_requests = [
            row for row in source_requests if row["stage"] == "compliance"
        ]
        endpoint_requests = [row for row in source_requests if row["stage"] == "endpoint"]
        compliance_expected = len(source["compliance_probes"])
        recomputed_compliance: list[bool] = []
        for index, probe in enumerate(source["compliance_probes"], start=1):
            probe_id = f"{probe['kind']}_{index}"
            matching = [row for row in compliance_requests if row["probe_id"] == probe_id]
            if len(matching) != 1:
                recomputed_compliance.append(False)
                continue
            passed, reason = compliance_request_passes(source, probe, matching[0])
            recomputed_compliance.append(passed)
            if (
                matching[0].get("validation_pass") != passed
                or matching[0].get("validation_reason") != reason
            ):
                integrity_violations.append(
                    f"compliance_validation_mismatch:{source['source_family']}:{probe_id}"
                )
        compliance_pass = bool(
            len(compliance_requests) == compliance_expected
            and len(recomputed_compliance) == compliance_expected
            and all(recomputed_compliance)
        )
        if endpoint_requests and not compliance_pass:
            integrity_violations.append(
                f"endpoint_issued_without_compliance_interlock:{source['source_family']}"
            )
        endpoint_final: list[dict[str, Any]] = []
        for endpoint in source["endpoint_candidates"]:
            attempts = [
                row
                for row in endpoint_requests
                if row["probe_id"] == endpoint["endpoint_id"]
            ]
            recomputed_attempts: list[bool] = []
            for attempt in attempts:
                attempt_pass, reason = endpoint_request_passes(endpoint, attempt)
                recomputed_attempts.append(attempt_pass)
                if (
                    attempt.get("validation_pass") != attempt_pass
                    or attempt.get("validation_reason") != reason
                ):
                    integrity_violations.append(
                        f"endpoint_validation_mismatch:{source['source_family']}:"
                        f"{endpoint['endpoint_id']}:{attempt['attempt']}"
                    )
            passed = any(recomputed_attempts)
            if passed:
                validated_endpoint_count += 1
            endpoint_final.append(
                {
                    "endpoint_id": endpoint["endpoint_id"],
                    "type": endpoint["type"],
                    "url": endpoint["url"],
                    "attempt_count": len(attempts),
                    "validated": passed,
                    "terminal_reason": (
                        attempts[-1].get("validation_reason") if attempts else "not_issued"
                    ),
                }
            )
        source_validated = compliance_pass and any(
            row["validated"] for row in endpoint_final
        )
        if source_validated:
            validated_source_count += 1
        source_results.append(
            {
                "source_family": source["source_family"],
                "exact_hosts": source["exact_hosts"],
                "compliance_request_count": len(compliance_requests),
                "compliance_pass": compliance_pass,
                "endpoint_request_count": len(endpoint_requests),
                "endpoint_results": endpoint_final,
                "endpoint_validated": source_validated,
            }
        )

    runtime_before = journal["runtime_hashes_before"]
    runtime_after = journal["runtime_hashes_after"]
    runtime_unchanged = runtime_before == runtime_after == EXPECTED_RUNTIME_HASHES
    if not runtime_unchanged:
        integrity_violations.append("runtime_changed")
    execution_integrity_pass = not integrity_violations
    hypothesis_supported = validated_source_count >= 2
    if not execution_integrity_pass:
        decision = "invalid_endpoint_validation"
    elif hypothesis_supported:
        decision = "structured_surface_sufficient_to_submit_snapshot_feasibility_plan_to_council"
    else:
        decision = "institutional_lane_viability_review"

    results = {
        "schema_version": "autoresearch-institutional-inventory-v16.clean-endpoint-validation-results",
        "iteration": 4,
        "source_results": source_results,
    }
    audit = {
        "schema_version": "autoresearch-institutional-inventory-v16.clean-endpoint-validation-audit",
        "iteration": 4,
        "contract_hashes": journal["contract_hashes"],
        "contract_derivation_verified": journal["contract_hashes"] == {
            name: contract[name]
            for name in ("spec_sha256", "evaluator_sha256", "manifest_sha256")
        },
        "metrics": {
            "source_family_count": len(source_results),
            "validated_compliant_structured_source_family_count": validated_source_count,
            "validated_endpoint_count": validated_endpoint_count,
            "compliance_request_count": counts["compliance"],
            "endpoint_request_count": counts["endpoint"],
            "network_request_count": len(requests),
            "redirect_count": sum(bool(row.get("redirect_location")) for row in requests),
            "retry_count": sum(row["attempt"] == 2 for row in requests),
            "raw_evidence_byte_count": sum(row["raw_byte_count"] for row in requests),
        },
        "integrity_violations": integrity_violations,
        "forbidden_operation_counters": {
            "generic_web_searches": 0,
            "record_or_item_content_fetches": 0,
            "document_candidates_counted": 0,
            "iiif_canvases_or_images_opened": 0,
            "images_read_or_downloaded": 0,
            "model_predictions": 0,
            "labels_read": 0,
            "final_test_reads": 0,
            "runtime_writes": 0,
        },
        "runtime_hashes_before": runtime_before,
        "runtime_hashes_after": runtime_after,
        "runtime_unchanged": runtime_unchanged,
        "execution_integrity_pass": execution_integrity_pass,
        "hypothesis_supported": hypothesis_supported,
        "snapshot_authorized": False,
        "promotion_eligible": False,
        "final_test_read": False,
        "decision": decision,
        "next_action": "Return the clean endpoint-validation result to Council before any metadata snapshot manifest is frozen.",
    }
    return results, audit


def write_evaluation_outputs(
    output_dir: Path, results: dict[str, Any], audit: dict[str, Any]
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "endpoint-validation-results.json"
    audit_path = output_dir / "endpoint-validation-audit.json"
    write_json_atomic(results_path, results)
    write_json_atomic(audit_path, audit)
    return {
        "results_sha256": sha256_file(results_path),
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
        "schema_version": "autoresearch-institutional-inventory-v16.clean-endpoint-validation-evaluation",
        "iteration": 4,
        "contract_derivation_verified": audit["contract_derivation_verified"],
        "validated_compliant_structured_source_family_count": audit["metrics"][
            "validated_compliant_structured_source_family_count"
        ],
        "minimum_validated_structured_sources": 2,
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
    write_json_atomic(output_path, evaluation)
    return evaluation


def run(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    replay_dir: Path = DEFAULT_REPLAY_DIR,
    evaluation_path: Path = DEFAULT_EVALUATION_PATH,
) -> dict[str, Any]:
    contract = validate_contract()
    before = runtime_hashes()
    journal = execute_network(contract, output_dir)
    after = runtime_hashes()
    journal["runtime_hashes_before"] = before
    journal["runtime_hashes_after"] = after
    write_json_atomic(output_dir / "request-journal.json", journal)
    first = evaluate_journal(contract, journal)
    first_hashes = write_evaluation_outputs(output_dir, *first)
    replay = evaluate_journal(contract, read_json(output_dir / "request-journal.json"))
    replay_hashes = write_evaluation_outputs(replay_dir, *replay)
    return evaluate(first[1], first_hashes, replay_hashes, evaluation_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-network", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--replay-dir", type=Path, default=DEFAULT_REPLAY_DIR)
    parser.add_argument("--evaluation", type=Path, default=DEFAULT_EVALUATION_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.execute_network:
        print("--execute-network is required for the preregistered iteration", file=sys.stderr)
        return 2
    result = run(args.output_dir, args.replay_dir, args.evaluation)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

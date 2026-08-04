#!/usr/bin/env python3
"""Run the preregistered v16 iteration-5 metadata-feasibility pass.

Only frozen LOC JSON and Gallica SRU search responses are retrieved.  The
first query for each source is a schema gate; later queries are mechanically
blocked unless that gate passes.  Raw responses are journaled before issue,
stored verbatim under a byte cap, and replayed without network access.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import ssl
import sys
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import urllib.robotparser
import xml.etree.ElementTree as ET
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
SPEC_PATH = RUN_DIR / "specs/iteration-0005.json"
EVALUATOR_PATH = RUN_DIR / "evaluator-iteration-0005.json"
MANIFEST_PATH = RUN_DIR / "metadata-feasibility-manifest-iteration-0005.json"
DEFAULT_OUTPUT_DIR = RUN_DIR / "iteration-0005"
DEFAULT_REPLAY_DIR = RUN_DIR / "iteration-0005-replay"
DEFAULT_EVALUATION_PATH = RUN_DIR / "evaluations/iteration-0005.json"

EXPECTED_HASHES = {
    "spec_sha256": "fb6d948883b379792fc9e812132b4bc69b9f1fcde022f9d51b50ac1ba0222a01",
    "evaluator_sha256": "4fbcc5484d55274577a2054c4c37e01e7ecd8c79c139db053d3a320f2ec964e5",
    "manifest_sha256": "675f349e5f4cc7dbd57c2f991dca2fb64c6b1fd4847452b38ae46a47c6796d72",
}
EXPECTED_RUNTIME_HASHES = {
    "projection_sha256": "0ad6ce348eaf44c8844415a769b048b8f155017befa93f16254b1263b9241210",
    "prototypes_sha256": "ffc52f03696f03780a01302708b53e319dc24069187a6fd1a2927ea537b1cc54",
    "config_sha256": "f2150a97ec7eee50a42ac1a108ae4bc092d46a6b8e4a2d91de59349560fbf18b",
}
LINKED_INPUTS = (
    "council_synthesis",
    "independent_components",
    "manuscript_exclusion_rules",
    "local_identity_crosswalk_rules",
)
GENERIC_IDENTITY_TOKENS = {
    "a",
    "aka",
    "and",
    "codex",
    "de",
    "del",
    "exp",
    "map",
    "mapa",
    "of",
    "rg",
    "the",
    "y",
}

SRU_NAMESPACE = "http://www.loc.gov/zing/srw/"

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


def normalize_text(value: Any) -> str:
    decomposed = unicodedata.normalize("NFKD", str(value))
    without_marks = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    return " ".join(re.findall(r"[a-z0-9]+", without_marks.casefold()))


def flatten_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(flatten_text(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return " ".join(flatten_text(item) for item in value)
    return str(value)


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def validate_contract() -> dict[str, Any]:
    paths = {
        "spec_sha256": SPEC_PATH,
        "evaluator_sha256": EVALUATOR_PATH,
        "manifest_sha256": MANIFEST_PATH,
    }
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    if hashes != EXPECTED_HASHES:
        raise ValueError(f"v16 iteration-5 contract hash mismatch: {hashes}")
    spec = read_json(SPEC_PATH)
    evaluator = read_json(EVALUATOR_PATH)
    manifest = read_json(MANIFEST_PATH)
    for name in LINKED_INPUTS:
        path = resolve_input(spec["inputs"][name])
        expected = spec["inputs"][name + "_sha256"]
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"linked input hash mismatch for {name}: {actual}")

    request = spec["request_contract"]
    if request["source_family_count_equals"] != 2:
        raise ValueError("source count gate changed")
    if request["frozen_queries_per_source_equals"] != 4:
        raise ValueError("query count gate changed")
    if request["query_requests_global_max"] != 16:
        raise ValueError("query request budget changed")
    if request["compliance_requests_global_max"] != 4:
        raise ValueError("compliance request budget changed")
    if request["network_requests_global_max"] != 20:
        raise ValueError("total request budget changed")
    if request["records_per_response_max"] != 25:
        raise ValueError("record response cap changed")
    if request["raw_records_per_source_max"] != 100:
        raise ValueError("source record cap changed")
    if manifest["frozen_screening"]["unknown_t_tepeuc_provenance_component_reserve"] != 1:
        raise ValueError("t_tepeuc uncertainty reserve changed")
    if request["redirects_followed"] is not False:
        raise ValueError("redirect following was enabled")
    if manifest["transport"]["follow_redirects"] is not False:
        raise ValueError("manifest redirect following was enabled")
    if manifest["transport"]["response_byte_cap"] != 1048576:
        raise ValueError("response cap changed")
    if evaluator["image_or_snapshot_extension_authorized"] is not False:
        raise ValueError("metadata feasibility cannot authorize images")

    sources = manifest["sources"]
    if len(sources) != 2 or len({row["source_family"] for row in sources}) != 2:
        raise ValueError("manifest must contain two unique sources")
    for source in sources:
        if len(source["queries"]) != 4:
            raise ValueError("each source must have four frozen queries")
        if [row["stage"] for row in source["queries"]] != [1, 2, 2, 2]:
            raise ValueError("query stage order changed")
        exact_hosts = source["exact_hosts"]
        allowed_urls = [source["compliance_probe"]["url"]] + [
            row["url"] for row in source["queries"]
        ]
        for url in allowed_urls:
            parsed = urllib.parse.urlsplit(url)
            if parsed.scheme != "https" or parsed.hostname not in exact_hosts:
                raise ValueError(f"non-allowlisted URL: {url}")
        for query in source["queries"]:
            parsed = urllib.parse.urlsplit(query["url"])
            if parsed.path != source["exact_query_path"]:
                raise ValueError(f"query path changed: {query['url']}")
        if len({row["query_id"] for row in source["queries"]}) != 4:
            raise ValueError("duplicate query id")
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
    query_per_source: int
    query_global: int
    total_global: int


class RequestBudget:
    def __init__(self, journal_path: Path, journal: dict[str, Any], limits: BudgetLimits):
        self.journal_path = journal_path
        self.journal = journal
        self.limits = limits

    def _counts(self) -> tuple[dict[str, int], dict[str, int], int, int, int]:
        compliance_by_source: dict[str, int] = {}
        query_by_source: dict[str, int] = {}
        compliance_global = 0
        query_global = 0
        for request in self.journal["requests"]:
            if request["stage"] == "compliance":
                compliance_by_source[request["source_family"]] = (
                    compliance_by_source.get(request["source_family"], 0) + 1
                )
                compliance_global += 1
            else:
                query_by_source[request["source_family"]] = (
                    query_by_source.get(request["source_family"], 0) + 1
                )
                query_global += 1
        return (
            compliance_by_source,
            query_by_source,
            compliance_global,
            query_global,
            compliance_global + query_global,
        )

    def issue(
        self,
        *,
        source: dict[str, Any],
        stage: str,
        probe_id: str,
        query_stage: int | None,
        attempt: int,
        url: str,
    ) -> int:
        if stage not in {"compliance", "query"}:
            raise ValueError(f"unknown request stage: {stage}")
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme != "https" or parsed.hostname not in source["exact_hosts"]:
            raise ValueError(f"refusing out-of-scope request: {url}")
        if stage == "query" and parsed.path != source["exact_query_path"]:
            raise ValueError(f"refusing out-of-scope query path: {url}")
        (
            compliance_by_source,
            query_by_source,
            compliance_global,
            query_global,
            total_global,
        ) = self._counts()
        source_name = source["source_family"]
        if total_global >= self.limits.total_global:
            raise ValueError("refusing request beyond global total budget")
        if stage == "compliance":
            if compliance_global >= self.limits.compliance_global:
                raise ValueError("refusing request beyond global compliance budget")
            if compliance_by_source.get(source_name, 0) >= self.limits.compliance_per_source:
                raise ValueError("refusing request beyond source compliance budget")
        else:
            if query_global >= self.limits.query_global:
                raise ValueError("refusing request beyond global query budget")
            if query_by_source.get(source_name, 0) >= self.limits.query_per_source:
                raise ValueError("refusing request beyond source query budget")
        sequence = len(self.journal["requests"]) + 1
        self.journal["requests"].append(
            {
                "sequence": sequence,
                "source_family": source_name,
                "stage": stage,
                "query_stage": query_stage,
                "probe_id": probe_id,
                "attempt": attempt,
                "url": url,
                "host": parsed.hostname,
                "issued_at_utc": utc_now(),
                "state": "issued_pending",
                "budget_before_issue": {
                    "source_stage": (
                        compliance_by_source.get(source_name, 0)
                        if stage == "compliance"
                        else query_by_source.get(source_name, 0)
                    ),
                    "global_stage": (
                        compliance_global if stage == "compliance" else query_global
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
        counts = self._counts()
        request["budget_after_completion"] = {
            "source_stage": (
                counts[0].get(request["source_family"], 0)
                if request["stage"] == "compliance"
                else counts[1].get(request["source_family"], 0)
            ),
            "global_stage": counts[2] if request["stage"] == "compliance" else counts[3],
            "global_total": counts[4],
        }
        write_json_atomic(self.journal_path, self.journal)


def safe_name(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value)


def fetch_request(
    *,
    budget: RequestBudget,
    evidence_dir: Path,
    source: dict[str, Any],
    stage: str,
    probe_id: str,
    query_stage: int | None,
    attempt: int,
    url: str,
    transport: dict[str, Any],
) -> dict[str, Any]:
    sequence = budget.issue(
        source=source,
        stage=stage,
        probe_id=probe_id,
        query_stage=query_stage,
        attempt=attempt,
        url=url,
    )
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": transport["user_agent"],
            "Accept": "application/json, application/xml, text/xml, text/plain, */*;q=0.1",
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
        f"{sequence:03d}-{safe_name(source['source_family'])}-"
        f"{safe_name(stage)}-{safe_name(probe_id)}-attempt-{attempt}.bin"
    )
    evidence_path = evidence_dir / evidence_name
    write_bytes_atomic(evidence_path, stored)
    completion = {
        "status": status,
        "content_type": headers.get("Content-Type") if headers is not None else None,
        "redirect_location": headers.get("Location") if headers is not None else None,
        "raw_evidence_path": str(evidence_path.relative_to(ROOT)),
        "raw_byte_count": len(stored),
        "raw_sha256": sha256_bytes(stored),
        "truncated": truncated,
        "error_kind": error_kind,
        "error_message": error_message,
    }
    budget.complete(sequence, completion)
    return budget.journal["requests"][sequence - 1]


def raw_bytes(request: dict[str, Any]) -> bytes:
    return (ROOT / request["raw_evidence_path"]).read_bytes()


def should_retry(request: dict[str, Any], transport: dict[str, Any]) -> bool:
    return bool(
        request["error_kind"] == "transport_error"
        or request["status"] in transport["retryable_http_statuses"]
    )


def compliance_passes(
    source: dict[str, Any], request: dict[str, Any]
) -> tuple[bool, str]:
    probe = source["compliance_probe"]
    if request["truncated"]:
        return False, "response_truncated"
    if request["redirect_location"]:
        return False, "redirect_not_followed"
    if request["status"] in probe["accepted_unavailable_statuses"]:
        return True, "robots_unavailable_allow_all_preregistered"
    if request["status"] in probe["denied_statuses"]:
        return False, "robots_access_denied"
    if request["status"] != 200:
        return False, "robots_status_unresolved"
    parser = urllib.robotparser.RobotFileParser()
    parser.set_url(probe["url"])
    parser.parse(raw_bytes(request).decode("utf-8", errors="replace").splitlines())
    if not all(
        parser.can_fetch("elements-research-endpoint-audit/1.0", query["url"])
        for query in source["queries"]
    ):
        return False, "robots_disallow_exact_query"
    return True, "pass"


def parse_loc(payload: bytes) -> tuple[bool, str, list[dict[str, Any]], int]:
    try:
        parsed = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False, "json_parse_failed", [], 0
    if not isinstance(parsed, dict):
        return False, "json_root_not_object", [], 0
    results = parsed.get("results")
    pagination = parsed.get("pagination")
    if not isinstance(results, list) or not isinstance(pagination, dict):
        return False, "json_schema_missing_results_or_pagination", [], 0
    if len(results) > 25:
        return False, "record_response_cap_exceeded", [], len(results)
    if results and not any(
        isinstance(row, dict) and row.get("id") and row.get("title") for row in results
    ):
        return False, "json_result_identity_schema_missing", [], len(results)
    records: list[dict[str, Any]] = []
    selected_fields = (
        "id",
        "title",
        "description",
        "subject",
        "location",
        "original_format",
        "partof",
        "rights",
        "rights_advisory",
        "rights_information",
        "access_restricted",
        "digitized",
        "date",
        "type",
        "format",
    )
    for row in results:
        if not isinstance(row, dict):
            records.append({"stable_id": None, "canonical_url": None, "title": ""})
            continue
        record = {field: row.get(field) for field in selected_fields}
        item = row.get("item") if isinstance(row.get("item"), dict) else {}
        for key, value in item.items():
            if "right" in key.casefold() or key in {"access_restricted", "digitized"}:
                record[f"item_{key}"] = value
        raw_id = row.get("id") if isinstance(row.get("id"), str) else None
        canonical = raw_id.replace("http://www.loc.gov/", "https://www.loc.gov/") if raw_id else None
        record["stable_id"] = canonical
        record["canonical_url"] = canonical
        records.append(record)
    total = pagination.get("of") if isinstance(pagination.get("of"), int) else len(results)
    return True, "pass", records, total


def parse_bnf(payload: bytes) -> tuple[bool, str, list[dict[str, Any]], int]:
    try:
        root = ET.fromstring(payload)
    except ET.ParseError:
        return False, "xml_parse_failed", [], 0
    if root.tag != f"{{{SRU_NAMESPACE}}}searchRetrieveResponse":
        return False, "xml_root_not_searchRetrieveResponse", [], 0
    number_nodes = [node for node in root.iter() if local_name(node.tag) == "numberOfRecords"]
    if len(number_nodes) != 1:
        return False, "xml_numberOfRecords_missing", [], 0
    try:
        total = int((number_nodes[0].text or "0").strip())
    except ValueError:
        return False, "xml_numberOfRecords_invalid", [], 0
    record_nodes = [node for node in root.iter() if local_name(node.tag) == "record"]
    if len(record_nodes) > 25:
        return False, "record_response_cap_exceeded", [], len(record_nodes)
    records: list[dict[str, Any]] = []
    fields = {
        "identifier",
        "title",
        "description",
        "subject",
        "type",
        "format",
        "rights",
        "coverage",
        "source",
        "relation",
    }
    for record_node in record_nodes:
        values: dict[str, list[str]] = {name: [] for name in fields}
        for node in record_node.iter():
            name = local_name(node.tag)
            text = (node.text or "").strip()
            if name in values and text:
                values[name].append(text)
        identifiers = values["identifier"]
        canonical = next(
            (
                value.replace("http://gallica.bnf.fr/", "https://gallica.bnf.fr/")
                for value in identifiers
                if "gallica.bnf.fr/" in value
            ),
            None,
        )
        stable = canonical or (identifiers[0] if identifiers else None)
        records.append(
            {
                **values,
                "stable_id": stable,
                "canonical_url": canonical,
                "title": values["title"][0] if values["title"] else "",
            }
        )
    if records and not any(row["stable_id"] and row["title"] for row in records):
        return False, "xml_record_identity_schema_missing", [], total
    return True, "pass", records, total


def query_passes(
    source: dict[str, Any], request: dict[str, Any]
) -> tuple[bool, str, list[dict[str, Any]], int]:
    if request["status"] is None or not 200 <= request["status"] < 300:
        return False, "non_success_status", [], 0
    if request["truncated"]:
        return False, "response_truncated", [], 0
    if request["redirect_location"]:
        return False, "redirect_not_followed", [], 0
    content_type = (request["content_type"] or "").casefold()
    prefixes = source["schema_requirements"]["content_type_prefixes"]
    if not any(content_type.startswith(prefix.casefold()) for prefix in prefixes):
        return False, "content_type_not_accepted", [], 0
    if source["response_format"] == "json":
        return parse_loc(raw_bytes(request))
    return parse_bnf(raw_bytes(request))


def execute_network(contract: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse metadata output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    evidence_dir = output_dir / "network-evidence"
    journal_path = output_dir / "request-journal.json"
    request_contract = contract["spec"]["request_contract"]
    limits = BudgetLimits(
        compliance_per_source=request_contract[
            "compliance_requests_including_retries_per_source_max"
        ],
        compliance_global=request_contract["compliance_requests_global_max"],
        query_per_source=request_contract[
            "query_requests_including_retries_per_source_max"
        ],
        query_global=request_contract["query_requests_global_max"],
        total_global=request_contract["network_requests_global_max"],
    )
    journal = {
        "schema_version": "autoresearch-institutional-inventory-v16.metadata-feasibility-request-journal",
        "iteration": 5,
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
        probe = source["compliance_probe"]
        compliance_request: dict[str, Any] | None = None
        for attempt in (1, 2):
            compliance_request = fetch_request(
                budget=budget,
                evidence_dir=evidence_dir,
                source=source,
                stage="compliance",
                probe_id="robots",
                query_stage=None,
                attempt=attempt,
                url=probe["url"],
                transport=transport,
            )
            passed, reason = compliance_passes(source, compliance_request)
            compliance_request["validation_pass"] = passed
            compliance_request["validation_reason"] = reason
            write_json_atomic(journal_path, journal)
            if passed or not should_retry(compliance_request, transport):
                break
        if compliance_request is None or not compliance_request["validation_pass"]:
            continue

        for query in source["queries"]:
            last_request: dict[str, Any] | None = None
            for attempt in (1, 2):
                last_request = fetch_request(
                    budget=budget,
                    evidence_dir=evidence_dir,
                    source=source,
                    stage="query",
                    probe_id=query["query_id"],
                    query_stage=query["stage"],
                    attempt=attempt,
                    url=query["url"],
                    transport=transport,
                )
                passed, reason, records, total = query_passes(source, last_request)
                last_request["validation_pass"] = passed
                last_request["validation_reason"] = reason
                last_request["parsed_record_count"] = len(records)
                last_request["reported_total_record_count"] = total
                write_json_atomic(journal_path, journal)
                if passed or not should_retry(last_request, transport):
                    break
            if last_request is None or not last_request["validation_pass"]:
                break
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


def frozen_alias_signatures(contract: dict[str, Any]) -> list[dict[str, Any]]:
    aliases: list[tuple[str, str]] = []
    for component in read_json(resolve_input(contract["spec"]["inputs"]["independent_components"]))[
        "components"
    ]:
        for name in component["display_names"]:
            aliases.append((component["component_id"], name.split("(", 1)[0]))
    exclusions = read_json(
        resolve_input(contract["spec"]["inputs"]["manuscript_exclusion_rules"])
    )
    for row in exclusions["excluded_visual_lexicon_terms"]:
        aliases.append((f"v12_term_{row['term_id']}", row["visual_manuscript"].split("(", 1)[0]))
        for title in row["local_titles"]:
            aliases.append((f"v12_term_{row['term_id']}", title))
    crosswalk = read_json(
        resolve_input(contract["spec"]["inputs"]["local_identity_crosswalk_rules"])
    )
    for row in crosswalk["visual_exclusion_mappings"]:
        for alias in row["frozen_aliases"]:
            aliases.append((row["physical_identity"], alias))
    for alias in contract["manifest"]["frozen_screening"]["telleriano_cognate_terms"]:
        aliases.append(("telleriano_cognate_family", alias))

    signatures: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}
    for identity, alias in aliases:
        tokens = tuple(
            token
            for token in normalize_text(alias).split()
            if len(token) >= 3 and token not in GENERIC_IDENTITY_TOKENS
        )
        if not tokens:
            continue
        signatures[(identity, tokens)] = {
            "identity": identity,
            "alias": alias,
            "tokens": list(tokens),
        }
    return [signatures[key] for key in sorted(signatures)]


def record_disposition(
    source_name: str,
    query: dict[str, Any],
    record: dict[str, Any],
    screening: dict[str, Any],
    alias_signatures: list[dict[str, Any]],
) -> dict[str, Any]:
    stable_id = record.get("stable_id")
    canonical_url = record.get("canonical_url")
    title = flatten_text(record.get("title"))
    text = normalize_text(flatten_text(record))
    text_tokens = set(text.split())
    reasons: list[str] = []
    matched_local_identity: str | None = None
    matched_local_alias: str | None = None
    if not stable_id:
        reasons.append("missing_stable_institutional_identifier")
    if not canonical_url:
        reasons.append("missing_canonical_source_url")
    domain_terms = [normalize_text(term) for term in screening["domain_positive_terms"]]
    if not any(term in text for term in domain_terms):
        reasons.append("target_domain_evidence_missing")
    mixtec_terms = [normalize_text(term) for term in screening["mixtec_only_terms"]]
    if any(term in text for term in mixtec_terms) and not any(
        term in text for term in domain_terms
    ):
        reasons.append("mixtec_only")
    physical_terms = [normalize_text(term) for term in screening["physical_positive_terms"]]
    if not any(term in text for term in physical_terms):
        reasons.append("physical_document_evidence_missing")
    secondary_terms = [
        normalize_text(term) for term in screening["secondary_or_copy_terms"]
    ]
    if any(term in text for term in secondary_terms):
        reasons.append("secondary_copy_or_reproduction_evidence")

    rights_text = normalize_text(
        " ".join(
            flatten_text(value)
            for key, value in record.items()
            if "right" in key.casefold() or "access" in key.casefold()
        )
    )
    rights_terms = [normalize_text(term) for term in screening["positive_rights_terms"]]
    bnf_filter = query.get("rights_filter") == "access all fayes"
    loc_open = record.get("access_restricted") is False and bool(record.get("digitized"))
    if not bnf_filter and not loc_open and not any(term in rights_text for term in rights_terms):
        reasons.append("positive_rights_or_open_access_evidence_missing")
    if record.get("access_restricted") is True:
        reasons.append("access_restricted")

    for signature in alias_signatures:
        if all(token in text_tokens for token in signature["tokens"]):
            matched_local_identity = signature["identity"]
            matched_local_alias = signature["alias"]
            reasons.append("local_or_exclusion_envelope_match")
            break
    identity_title = normalize_text(title)
    return {
        "source_family": source_name,
        "query_id": query["query_id"],
        "stable_id": stable_id,
        "canonical_url": canonical_url,
        "title": title,
        "identity_title": identity_title,
        "matched_local_identity": matched_local_identity,
        "matched_local_alias": matched_local_alias,
        "gate_reasons": sorted(set(reasons)),
        "pre_dedup_strict": not reasons,
    }


def evaluate_journal(
    contract: dict[str, Any], journal: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    sources = contract["manifest"]["sources"]
    requests = journal["requests"]
    source_map = {source["source_family"]: source for source in sources}
    integrity: list[str] = []
    if any(row["state"] != "completed" for row in requests):
        integrity.append("pending_request")
    allowed_urls = {
        source["source_family"]: {
            source["compliance_probe"]["url"],
            *(query["url"] for query in source["queries"]),
        }
        for source in sources
    }
    for expected_sequence, request in enumerate(requests, start=1):
        required = (
            "issued_at_utc",
            "completed_at_utc",
            "raw_evidence_path",
            "raw_byte_count",
            "raw_sha256",
            "budget_before_issue",
            "budget_after_completion",
        )
        if any(field not in request for field in required):
            integrity.append(f"request_schema_incomplete:{expected_sequence}")
            continue
        if request["sequence"] != expected_sequence:
            integrity.append("non_contiguous_request_sequence")
        for field in ("issued_at_utc", "completed_at_utc"):
            try:
                timestamp = datetime.fromisoformat(request[field].replace("Z", "+00:00"))
                if timestamp.tzinfo is None:
                    raise ValueError("timezone missing")
            except (AttributeError, ValueError):
                integrity.append(f"invalid_request_utc:{expected_sequence}:{field}")
        source = source_map.get(request["source_family"])
        if source is None:
            integrity.append(f"unknown_source:{expected_sequence}")
            continue
        if request["url"] not in allowed_urls[source["source_family"]]:
            integrity.append(f"out_of_scope_url:{expected_sequence}")
        parsed = urllib.parse.urlsplit(request["url"])
        if request["host"] != parsed.hostname or parsed.hostname not in source["exact_hosts"]:
            integrity.append(f"host_mismatch:{expected_sequence}")
        if request["stage"] == "query" and parsed.path != source["exact_query_path"]:
            integrity.append(f"query_path_mismatch:{expected_sequence}")
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
            integrity.append(f"budget_before_mismatch:{expected_sequence}")
        if request["budget_after_completion"] != expected_after:
            integrity.append(f"budget_after_mismatch:{expected_sequence}")
        if not validate_raw_evidence(request):
            integrity.append(f"raw_evidence_mismatch:{expected_sequence}")

    compliance_count = sum(row["stage"] == "compliance" for row in requests)
    query_count = sum(row["stage"] == "query" for row in requests)
    if compliance_count > 4:
        integrity.append("global_compliance_budget_exceeded")
    if query_count > 16:
        integrity.append("global_query_budget_exceeded")
    if len(requests) > 20:
        integrity.append("global_total_budget_exceeded")

    aliases = frozen_alias_signatures(contract)
    screening = contract["manifest"]["frozen_screening"]
    source_results: list[dict[str, Any]] = []
    all_dispositions: list[dict[str, Any]] = []
    stage_1_passing_count = 0
    atomic_pass_count = 0
    raw_record_count = 0
    for source in sources:
        source_requests = [
            row for row in requests if row["source_family"] == source["source_family"]
        ]
        compliance_attempts = [row for row in source_requests if row["stage"] == "compliance"]
        compliance_pass = False
        if compliance_attempts:
            compliance_pass, compliance_reason = compliance_passes(source, compliance_attempts[-1])
            for attempt in compliance_attempts:
                attempt_pass, attempt_reason = compliance_passes(source, attempt)
                if (
                    attempt.get("validation_pass") != attempt_pass
                    or attempt.get("validation_reason") != attempt_reason
                ):
                    integrity.append(
                        f"compliance_validation_mismatch:{source['source_family']}:{attempt['attempt']}"
                    )
        else:
            compliance_reason = "not_issued"
        query_requests = [row for row in source_requests if row["stage"] == "query"]
        if query_requests and not compliance_pass:
            integrity.append(f"query_without_compliance:{source['source_family']}")
        query_results: list[dict[str, Any]] = []
        source_records: list[tuple[dict[str, Any], dict[str, Any]]] = []
        stage_1_pass = False
        source_atomic_pass = compliance_pass
        for query_index, query in enumerate(source["queries"]):
            attempts = [row for row in query_requests if row["probe_id"] == query["query_id"]]
            terminal_pass = False
            terminal_reason = "not_issued"
            parsed_records: list[dict[str, Any]] = []
            reported_total = 0
            for attempt in attempts:
                passed, reason, records, total = query_passes(source, attempt)
                if (
                    attempt.get("validation_pass") != passed
                    or attempt.get("validation_reason") != reason
                    or attempt.get("parsed_record_count") != len(records)
                    or attempt.get("reported_total_record_count") != total
                ):
                    integrity.append(
                        f"query_validation_mismatch:{source['source_family']}:{query['query_id']}:{attempt['attempt']}"
                    )
                terminal_pass = passed
                terminal_reason = reason
                parsed_records = records
                reported_total = total
                if passed:
                    break
            if query_index == 0:
                stage_1_pass = terminal_pass
                if stage_1_pass:
                    stage_1_passing_count += 1
            elif attempts and not stage_1_pass:
                integrity.append(f"stage_2_without_stage_1:{source['source_family']}")
            if not attempts or not terminal_pass:
                source_atomic_pass = False
            if terminal_pass:
                raw_record_count += len(parsed_records)
                source_records.extend((query, record) for record in parsed_records)
            query_results.append(
                {
                    "query_id": query["query_id"],
                    "stage": query["stage"],
                    "attempt_count": len(attempts),
                    "validated": terminal_pass,
                    "terminal_reason": terminal_reason,
                    "record_count": len(parsed_records),
                    "reported_total_record_count": reported_total,
                }
            )
        if len(source_records) > 100:
            integrity.append(f"source_record_cap_exceeded:{source['source_family']}")
            source_atomic_pass = False
        if source_atomic_pass:
            atomic_pass_count += 1
        seen_source_ids: set[str] = set()
        for query, record in source_records:
            disposition = record_disposition(
                source["source_family"], query, record, screening, aliases
            )
            stable_key = normalize_text(disposition["stable_id"] or "")
            if stable_key and stable_key in seen_source_ids:
                disposition["gate_reasons"].append("duplicate_stable_identifier")
                disposition["pre_dedup_strict"] = False
            elif stable_key:
                seen_source_ids.add(stable_key)
            if not source_atomic_pass:
                disposition["gate_reasons"].append("source_atomic_query_failure")
                disposition["pre_dedup_strict"] = False
            disposition["gate_reasons"] = sorted(set(disposition["gate_reasons"]))
            all_dispositions.append(disposition)
        source_results.append(
            {
                "source_family": source["source_family"],
                "compliance_attempt_count": len(compliance_attempts),
                "compliance_pass": compliance_pass,
                "compliance_reason": compliance_reason,
                "stage_1_pass": stage_1_pass,
                "source_atomic_pass": source_atomic_pass,
                "query_results": query_results,
                "raw_record_count": len(source_records),
            }
        )

    strict_candidates: list[dict[str, Any]] = []
    near_misses: list[dict[str, Any]] = []
    seen_titles: dict[str, str] = {}
    seen_ids: set[str] = set()
    for disposition in sorted(
        all_dispositions,
        key=lambda row: (
            row["source_family"],
            normalize_text(row["stable_id"] or ""),
            row["identity_title"],
            row["query_id"],
        ),
    ):
        reasons = list(disposition["gate_reasons"])
        stable_key = normalize_text(disposition["stable_id"] or "")
        title_key = disposition["identity_title"]
        if disposition["pre_dedup_strict"]:
            if stable_key in seen_ids:
                reasons.append("within_or_cross_source_duplicate_id")
            elif title_key and title_key in seen_titles:
                reasons.append("within_or_cross_source_duplicate_title")
            else:
                seen_ids.add(stable_key)
                if title_key:
                    seen_titles[title_key] = disposition["source_family"]
        disposition["gate_reasons"] = sorted(set(reasons))
        disposition["strict_before_t_tepeuc_reserve"] = not disposition["gate_reasons"]
        if disposition["strict_before_t_tepeuc_reserve"]:
            strict_candidates.append(disposition)
        else:
            near_misses.append(disposition)

    reserve_max = screening["unknown_t_tepeuc_provenance_component_reserve"]
    t_tepeuc_reserve = min(reserve_max, len(strict_candidates))
    conservative_count = max(0, len(strict_candidates) - t_tepeuc_reserve)
    runtime_before = journal["runtime_hashes_before"]
    runtime_after = journal["runtime_hashes_after"]
    runtime_unchanged = runtime_before == runtime_after == EXPECTED_RUNTIME_HASHES
    if not runtime_unchanged:
        integrity.append("runtime_changed")
    execution_integrity_pass = not integrity
    hypothesis_supported = conservative_count >= 26
    if not execution_integrity_pass:
        decision = "invalid_metadata_feasibility"
    elif hypothesis_supported:
        decision = "feasibility_ceiling_reached"
    else:
        decision = "not_reached"

    results = {
        "schema_version": "autoresearch-institutional-inventory-v16.metadata-feasibility-results",
        "iteration": 5,
        "source_results": source_results,
        "strict_candidates_before_t_tepeuc_reserve": strict_candidates,
        "diagnostic_near_misses": near_misses,
        "t_tepeuc_unknown_provenance_component_reserve": t_tepeuc_reserve,
        "conservatively_net_new_independent_physical_component_count": conservative_count,
    }
    unique_metadata_ids = {
        normalize_text(row["stable_id"])
        for row in all_dispositions
        if row["stable_id"]
    }
    audit = {
        "schema_version": "autoresearch-institutional-inventory-v16.metadata-feasibility-audit",
        "iteration": 5,
        "contract_hashes": journal["contract_hashes"],
        "contract_derivation_verified": journal["contract_hashes"]
        == {
            name: contract[name]
            for name in ("spec_sha256", "evaluator_sha256", "manifest_sha256")
        },
        "metrics": {
            "stage_1_passing_source_count": stage_1_passing_count,
            "source_atomic_pass_count": atomic_pass_count,
            "raw_record_count": raw_record_count,
            "unique_metadata_record_count": len(unique_metadata_ids),
            "strict_candidates_before_t_tepeuc_reserve_count": len(strict_candidates),
            "t_tepeuc_unknown_provenance_component_reserve": t_tepeuc_reserve,
            "conservatively_net_new_independent_physical_component_count": conservative_count,
            "near_miss_count": len(near_misses),
            "compliance_request_count": compliance_count,
            "query_request_count": query_count,
            "network_request_count": len(requests),
            "retry_count": sum(row["attempt"] == 2 for row in requests),
            "redirect_count": sum(bool(row.get("redirect_location")) for row in requests),
            "raw_evidence_byte_count": sum(row["raw_byte_count"] for row in requests),
        },
        "integrity_violations": sorted(set(integrity)),
        "forbidden_operation_counters": {
            "generic_web_searches_during_execution": 0,
            "item_detail_fetches": 0,
            "iiif_manifests_opened": 0,
            "iiif_canvases_or_images_opened": 0,
            "thumbnails_or_images_read": 0,
            "model_predictions_or_training_runs": 0,
            "labels_read": 0,
            "final_test_reads": 0,
            "checkpoint_or_runtime_writes": 0,
        },
        "runtime_hashes_before": runtime_before,
        "runtime_hashes_after": runtime_after,
        "runtime_unchanged": runtime_unchanged,
        "execution_integrity_pass": execution_integrity_pass,
        "hypothesis_supported": hypothesis_supported,
        "minimum_conservatively_net_new_components": 26,
        "image_or_snapshot_extension_authorized": False,
        "promotion_eligible": False,
        "final_test_read": False,
        "decision": decision,
        "next_action": "Return the bounded metadata-feasibility result to Council before any image or model operation.",
    }
    return results, audit


def write_evaluation_outputs(
    output_dir: Path, results: dict[str, Any], audit: dict[str, Any]
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "metadata-feasibility-results.json"
    audit_path = output_dir / "metadata-feasibility-audit.json"
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
        "schema_version": "autoresearch-institutional-inventory-v16.metadata-feasibility-evaluation",
        "iteration": 5,
        "contract_derivation_verified": audit["contract_derivation_verified"],
        "conservatively_net_new_independent_physical_component_count": audit["metrics"][
            "conservatively_net_new_independent_physical_component_count"
        ],
        "minimum_conservatively_net_new_components": 26,
        "deterministic_replay": deterministic_replay,
        "execution_integrity_pass": audit["execution_integrity_pass"],
        "hypothesis_supported": audit["hypothesis_supported"],
        "runtime_unchanged": audit["runtime_unchanged"],
        "image_or_snapshot_extension_authorized": False,
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

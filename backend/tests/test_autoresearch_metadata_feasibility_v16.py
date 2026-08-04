from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/autoresearch_metadata_feasibility_v16.py"
spec = importlib.util.spec_from_file_location(
    "autoresearch_metadata_feasibility_v16", SCRIPT
)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_contract_is_hash_pinned_with_two_sources_and_four_queries_each() -> None:
    contract = module.validate_contract()
    assert {
        name: contract[name] for name in module.EXPECTED_HASHES
    } == module.EXPECTED_HASHES
    sources = contract["manifest"]["sources"]
    assert len(sources) == 2
    assert all(len(source["queries"]) == 4 for source in sources)
    assert all(
        [query["stage"] for query in source["queries"]] == [1, 2, 2, 2]
        for source in sources
    )
    assert contract["manifest"]["transport"]["follow_redirects"] is False
    assert contract["spec"]["decision_rule"][
        "minimum_conservatively_net_new_components"
    ] == 26


def test_manifest_queries_use_only_exact_search_paths_and_never_image_resources() -> None:
    contract = module.validate_contract()
    expected = {
        "library_of_congress": ("www.loc.gov", "/manuscripts/"),
        "bnf_gallica_sru": ("gallica.bnf.fr", "/SRU"),
    }
    for source in contract["manifest"]["sources"]:
        host, path = expected[source["source_family"]]
        for query in source["queries"]:
            parsed = module.urllib.parse.urlsplit(query["url"])
            assert (parsed.hostname, parsed.path) == (host, path)
            lowered = query["url"].lower()
            assert "manifest.json" not in lowered
            assert "/iiif/" not in lowered
            assert "/item/" not in lowered
            assert "/resource/" not in lowered


def test_budget_persists_before_issue_and_refuses_out_of_scope_path(
    tmp_path: Path,
) -> None:
    source = module.validate_contract()["manifest"]["sources"][0]
    journal_path = tmp_path / "journal.json"
    journal = {"requests": []}
    budget = module.RequestBudget(
        journal_path, journal, module.BudgetLimits(2, 4, 8, 16, 20)
    )
    sequence = budget.issue(
        source=source,
        stage="query",
        probe_id="loc_aztec",
        query_stage=1,
        attempt=1,
        url=source["queries"][0]["url"],
    )
    persisted = module.read_json(journal_path)
    assert persisted["requests"][0]["state"] == "issued_pending"
    assert persisted["requests"][0]["budget_before_issue"]["global_total"] == 0
    budget.complete(
        sequence,
        {
            "status": 200,
            "content_type": "application/json",
            "redirect_location": None,
            "raw_evidence_path": str(tmp_path / "response.bin"),
            "raw_byte_count": 0,
            "raw_sha256": hashlib.sha256(b"").hexdigest(),
            "truncated": False,
            "error_kind": None,
            "error_message": None,
        },
    )
    with pytest.raises(ValueError, match="query path"):
        budget.issue(
            source=source,
            stage="query",
            probe_id="bad",
            query_stage=2,
            attempt=1,
            url="https://www.loc.gov/item/123/?fo=json",
        )
    assert len(journal["requests"]) == 1


def test_loc_and_bnf_schema_parsers_are_bounded() -> None:
    loc = {
        "pagination": {"of": 1},
        "results": [
            {
                "id": "http://www.loc.gov/item/alpha/",
                "title": "Nahua manuscript alpha",
                "access_restricted": False,
                "digitized": True,
            }
        ],
    }
    passed, reason, records, total = module.parse_loc(json.dumps(loc).encode())
    assert (passed, reason, total) == (True, "pass", 1)
    assert records[0]["canonical_url"] == "https://www.loc.gov/item/alpha/"

    xml = b'''<?xml version="1.0"?>
    <srw:searchRetrieveResponse xmlns:srw="http://www.loc.gov/zing/srw/"
      xmlns:oai_dc="http://www.openarchives.org/OAI/2.0/oai_dc/"
      xmlns:dc="http://purl.org/dc/elements/1.1/">
      <srw:numberOfRecords>1</srw:numberOfRecords>
      <srw:records><srw:record><srw:recordData><oai_dc:dc>
        <dc:identifier>https://gallica.bnf.fr/ark:/12148/alpha</dc:identifier>
        <dc:title>Manuscrit nahuatl alpha</dc:title>
        <dc:rights>Libre de droits</dc:rights>
      </oai_dc:dc></srw:recordData></srw:record></srw:records>
    </srw:searchRetrieveResponse>'''
    passed, reason, records, total = module.parse_bnf(xml)
    assert (passed, reason, total) == (True, "pass", 1)
    assert records[0]["stable_id"] == "https://gallica.bnf.fr/ark:/12148/alpha"


def _append_request(
    requests: list[dict],
    tmp_path: Path,
    *,
    source_family: str,
    stage: str,
    probe_id: str,
    query_stage: int | None,
    url: str,
    content_type: str,
    payload: bytes,
    parsed_record_count: int | None = None,
    reported_total: int | None = None,
) -> None:
    sequence = len(requests) + 1
    path = tmp_path / f"evidence-{sequence}.bin"
    path.write_bytes(payload)
    prior_source_stage = sum(
        row["source_family"] == source_family and row["stage"] == stage
        for row in requests
    )
    prior_global_stage = sum(row["stage"] == stage for row in requests)
    request = {
        "sequence": sequence,
        "source_family": source_family,
        "stage": stage,
        "query_stage": query_stage,
        "probe_id": probe_id,
        "attempt": 1,
        "url": url,
        "host": module.urllib.parse.urlsplit(url).hostname,
        "issued_at_utc": "2026-08-04T00:30:00Z",
        "completed_at_utc": "2026-08-04T00:30:01Z",
        "state": "completed",
        "status": 200,
        "content_type": content_type,
        "redirect_location": None,
        "raw_evidence_path": str(path),
        "raw_byte_count": len(payload),
        "raw_sha256": hashlib.sha256(payload).hexdigest(),
        "truncated": False,
        "error_kind": None,
        "error_message": None,
        "validation_pass": True,
        "validation_reason": "pass",
        "budget_before_issue": {
            "source_stage": prior_source_stage,
            "global_stage": prior_global_stage,
            "global_total": sequence - 1,
        },
        "budget_after_completion": {
            "source_stage": prior_source_stage + 1,
            "global_stage": prior_global_stage + 1,
            "global_total": sequence,
        },
    }
    if parsed_record_count is not None:
        request["parsed_record_count"] = parsed_record_count
        request["reported_total_record_count"] = reported_total
    requests.append(request)


def _bnf_payload(index: int, term: str) -> bytes:
    return f'''<?xml version="1.0"?>
    <srw:searchRetrieveResponse xmlns:srw="http://www.loc.gov/zing/srw/"
      xmlns:oai_dc="http://www.openarchives.org/OAI/2.0/oai_dc/"
      xmlns:dc="http://purl.org/dc/elements/1.1/">
      <srw:numberOfRecords>1</srw:numberOfRecords>
      <srw:records><srw:record><srw:recordData><oai_dc:dc>
        <dc:identifier>https://gallica.bnf.fr/ark:/12148/synthetic{index}</dc:identifier>
        <dc:title>Manuscrit {term} synthétique {index}</dc:title>
        <dc:rights>Libre de droits</dc:rights>
      </oai_dc:dc></srw:recordData></srw:record></srw:records>
    </srw:searchRetrieveResponse>'''.encode()


def _complete_journal(contract: dict, tmp_path: Path) -> dict:
    requests: list[dict] = []
    for source in contract["manifest"]["sources"]:
        _append_request(
            requests,
            tmp_path,
            source_family=source["source_family"],
            stage="compliance",
            probe_id="robots",
            query_stage=None,
            url=source["compliance_probe"]["url"],
            content_type="text/plain",
            payload=b"User-agent: *\nAllow: /\n",
        )
        for index, query in enumerate(source["queries"]):
            if source["response_format"] == "json":
                payload = json.dumps(
                    {
                        "pagination": {"of": 1},
                        "results": [
                            {
                                "id": f"http://www.loc.gov/item/synthetic{index}/",
                                "title": f"Nahua manuscript synthetic {index}",
                                "access_restricted": False,
                                "digitized": True,
                            }
                        ],
                    }
                ).encode()
                content_type = "application/json; charset=utf-8"
            else:
                payload = _bnf_payload(index, query["term"])
                content_type = "text/xml; charset=utf-8"
            _append_request(
                requests,
                tmp_path,
                source_family=source["source_family"],
                stage="query",
                probe_id=query["query_id"],
                query_stage=query["stage"],
                url=query["url"],
                content_type=content_type,
                payload=payload,
                parsed_record_count=1,
                reported_total=1,
            )
    return {
        "contract_hashes": {
            name: contract[name]
            for name in ("spec_sha256", "evaluator_sha256", "manifest_sha256")
        },
        "requests": requests,
        "runtime_hashes_before": module.EXPECTED_RUNTIME_HASHES,
        "runtime_hashes_after": module.EXPECTED_RUNTIME_HASHES,
    }


def test_replay_enforces_atomic_sources_and_t_tepeuc_reserve(tmp_path: Path) -> None:
    contract = module.validate_contract()
    results, audit = module.evaluate_journal(contract, _complete_journal(contract, tmp_path))
    assert audit["execution_integrity_pass"] is True
    assert audit["metrics"]["stage_1_passing_source_count"] == 2
    assert audit["metrics"]["source_atomic_pass_count"] == 2
    assert len(results["strict_candidates_before_t_tepeuc_reserve"]) == 8
    assert results["t_tepeuc_unknown_provenance_component_reserve"] == 1
    assert results[
        "conservatively_net_new_independent_physical_component_count"
    ] == 7
    assert audit["decision"] == "not_reached"


def test_replay_detects_stage_two_without_stage_one(tmp_path: Path) -> None:
    contract = module.validate_contract()
    journal = _complete_journal(contract, tmp_path)
    loc_stage_one_id = contract["manifest"]["sources"][0]["queries"][0]["query_id"]
    journal["requests"] = [
        row
        for row in journal["requests"]
        if not (
            row["source_family"] == "library_of_congress"
            and row["probe_id"] == loc_stage_one_id
        )
    ]
    for sequence, row in enumerate(journal["requests"], start=1):
        row["sequence"] = sequence
    _, audit = module.evaluate_journal(contract, journal)
    assert audit["execution_integrity_pass"] is False
    assert any(
        violation.startswith("stage_2_without_stage_1")
        for violation in audit["integrity_violations"]
    )


def test_replay_detects_tampered_raw_bytes_and_validation_claim(tmp_path: Path) -> None:
    contract = module.validate_contract()
    journal = _complete_journal(contract, tmp_path)
    changed = copy.deepcopy(journal)
    changed["requests"][0]["validation_pass"] = False
    Path(changed["requests"][1]["raw_evidence_path"]).write_bytes(b"tampered")
    _, audit = module.evaluate_journal(contract, changed)
    assert audit["execution_integrity_pass"] is False
    assert any(
        violation.startswith("compliance_validation_mismatch")
        for violation in audit["integrity_violations"]
    )
    assert any(
        violation.startswith("raw_evidence_mismatch")
        for violation in audit["integrity_violations"]
    )


def test_local_alias_match_excludes_known_huexotzinco_component() -> None:
    contract = module.validate_contract()
    aliases = module.frozen_alias_signatures(contract)
    source = contract["manifest"]["sources"][0]
    query = source["queries"][0]
    disposition = module.record_disposition(
        source["source_family"],
        query,
        {
            "stable_id": "https://www.loc.gov/item/known/",
            "canonical_url": "https://www.loc.gov/item/known/",
            "title": "Codex Huexotzinco, Nahua manuscript",
            "access_restricted": False,
            "digitized": True,
        },
        contract["manifest"]["frozen_screening"],
        aliases,
    )
    assert "local_or_exclusion_envelope_match" in disposition["gate_reasons"]
    assert disposition["pre_dedup_strict"] is False


def test_recorded_iteration_0005_replays_to_published_not_reached_result() -> None:
    contract = module.validate_contract()
    output_dir = module.DEFAULT_OUTPUT_DIR
    journal = module.read_json(output_dir / "request-journal.json")
    recorded_results = module.read_json(output_dir / "metadata-feasibility-results.json")
    recorded_audit = module.read_json(output_dir / "metadata-feasibility-audit.json")

    replayed_results, replayed_audit = module.evaluate_journal(contract, journal)

    assert replayed_results == recorded_results
    assert replayed_audit == recorded_audit
    assert replayed_audit["execution_integrity_pass"] is True
    assert replayed_audit["decision"] == "not_reached"
    assert replayed_audit["metrics"]["network_request_count"] == 10
    assert replayed_audit["metrics"]["source_atomic_pass_count"] == 2
    assert replayed_results[
        "conservatively_net_new_independent_physical_component_count"
    ] == 3


def test_redirect_handler_never_follows() -> None:
    handler = module.NoRedirect()
    request = module.urllib.request.Request("https://www.loc.gov/")
    assert (
        handler.redirect_request(
            request,
            None,
            302,
            "Found",
            {"Location": "https://example.com"},
            "https://example.com",
        )
        is None
    )

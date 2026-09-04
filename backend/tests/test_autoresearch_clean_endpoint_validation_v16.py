from __future__ import annotations

import copy
import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/autoresearch_clean_endpoint_validation_v16.py"
spec = importlib.util.spec_from_file_location(
    "autoresearch_clean_endpoint_validation_v16", SCRIPT
)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[2] / ".omc/autoresearch/elements-baseline-replacement/runs/20260803-institutional-inventory-v16/endpoint-validation-manifest-iteration-0004.json").exists(),
    reason="requires unshipped research artifact: 20260803-institutional-inventory-v16/endpoint-validation-manifest-iteration-0004.json",
)
def test_contract_is_hash_pinned_before_network_and_has_six_exact_sources() -> None:
    contract = module.validate_contract()
    assert {
        name: contract[name] for name in module.EXPECTED_HASHES
    } == module.EXPECTED_HASHES
    sources = contract["manifest"]["sources"]
    assert len(sources) == 6
    assert len({source["source_family"] for source in sources}) == 6
    assert contract["spec"]["validation_gates"][
        "exact_host_allowlist_no_implicit_subdomains"
    ] is True
    assert contract["manifest"]["transport"]["follow_redirects"] is False


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[2] / ".omc/autoresearch/elements-baseline-replacement/runs/20260803-institutional-inventory-v16/endpoint-validation-manifest-iteration-0004.json").exists(),
    reason="requires unshipped research artifact: 20260803-institutional-inventory-v16/endpoint-validation-manifest-iteration-0004.json",
)
def test_manifest_contains_only_structured_endpoint_types_and_no_item_urls() -> None:
    contract = module.validate_contract()
    for source in contract["manifest"]["sources"]:
        for endpoint in source["endpoint_candidates"]:
            assert endpoint["type"] in module.ELIGIBLE_TYPES
            lowered = endpoint["url"].lower()
            assert "/canvas" not in lowered
            assert "/image" not in lowered
            assert "/item/" not in lowered


def test_budget_is_persisted_before_issue_and_refuses_fifth_source_request(
    tmp_path: Path,
) -> None:
    journal_path = tmp_path / "journal.json"
    journal = {"requests": []}
    limits = module.BudgetLimits(4, 24, 4, 24, 48)
    budget = module.RequestBudget(journal_path, journal, limits)
    for index in range(4):
        sequence = budget.issue(
            source_family="library_of_congress",
            stage="compliance",
            probe_id=f"probe_{index}",
            attempt=1,
            url="https://www.loc.gov/robots.txt",
            exact_hosts=["www.loc.gov"],
        )
        assert module.read_json(journal_path)["requests"][-1]["state"] == "issued_pending"
        budget.complete(
            sequence,
            {
                "status": 200,
                "content_type": "text/plain",
                "redirect_location": None,
                "raw_evidence_path": str(tmp_path / f"{index}.bin"),
                "raw_byte_count": 0,
                "raw_sha256": hashlib.sha256(b"").hexdigest(),
                "truncated": False,
                "error_kind": None,
                "error_message": None,
            },
        )
    with pytest.raises(ValueError, match="source compliance budget"):
        budget.issue(
            source_family="library_of_congress",
            stage="compliance",
            probe_id="probe_5",
            attempt=1,
            url="https://www.loc.gov/robots.txt",
            exact_hosts=["www.loc.gov"],
        )


def test_out_of_scope_host_is_refused_before_journal_issue(tmp_path: Path) -> None:
    journal_path = tmp_path / "journal.json"
    journal = {"requests": []}
    budget = module.RequestBudget(
        journal_path, journal, module.BudgetLimits(4, 24, 4, 24, 48)
    )
    with pytest.raises(ValueError, match="out-of-scope"):
        budget.issue(
            source_family="bne_linked_data",
            stage="endpoint",
            probe_id="bad",
            attempt=1,
            url="https://sub.datos.bne.es/api",
            exact_hosts=["datos.bne.es"],
        )
    assert journal["requests"] == []
    assert not journal_path.exists()


def test_redirect_handler_never_follows() -> None:
    handler = module.NoRedirect()
    request = module.urllib.request.Request("https://www.loc.gov/")
    assert (
        handler.redirect_request(
            request, None, 302, "Found", {"Location": "https://example.com"}, "https://example.com"
        )
        is None
    )


def _append_request(
    requests: list[dict],
    tmp_path: Path,
    *,
    source_family: str,
    stage: str,
    probe_id: str,
    url: str,
    content_type: str,
    payload: bytes,
) -> None:
    sequence = len(requests) + 1
    path = tmp_path / f"evidence-{sequence}.bin"
    path.write_bytes(payload)
    prior_source_stage = sum(
        row["source_family"] == source_family and row["stage"] == stage
        for row in requests
    )
    prior_global_stage = sum(row["stage"] == stage for row in requests)
    requests.append(
        {
            "sequence": sequence,
            "source_family": source_family,
            "stage": stage,
            "probe_id": probe_id,
            "attempt": 1,
            "url": url,
            "host": module.urllib.parse.urlsplit(url).hostname,
            "issued_at_utc": "2026-08-03T23:50:00Z",
            "completed_at_utc": "2026-08-03T23:50:01Z",
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
    )


def _two_source_journal(contract: dict, tmp_path: Path) -> dict:
    requests: list[dict] = []
    _append_request(
        requests,
        tmp_path,
        source_family="library_of_congress",
        stage="compliance",
        probe_id="robots_1",
        url="https://www.loc.gov/robots.txt",
        content_type="text/plain",
        payload=b"User-agent: *\nAllow: /\n",
    )
    _append_request(
        requests,
        tmp_path,
        source_family="library_of_congress",
        stage="compliance",
        probe_id="documented_access_policy_2",
        url="https://www.loc.gov/apis/json-and-yaml/",
        content_type="text/html",
        payload=b"Official JSON YAML API documentation",
    )
    _append_request(
        requests,
        tmp_path,
        source_family="library_of_congress",
        stage="endpoint",
        probe_id="loc_json_yaml_docs",
        url="https://www.loc.gov/apis/json-and-yaml/",
        content_type="text/html; charset=utf-8",
        payload=b"JSON and YAML",
    )
    _append_request(
        requests,
        tmp_path,
        source_family="bne_linked_data",
        stage="compliance",
        probe_id="robots_1",
        url="https://datos.bne.es/robots.txt",
        content_type="text/plain",
        payload=b"User-agent: *\nAllow: /\n",
    )
    _append_request(
        requests,
        tmp_path,
        source_family="bne_linked_data",
        stage="compliance",
        probe_id="documented_access_policy_2",
        url="https://datos.bne.es/inicio.html",
        content_type="text/html",
        payload=b"datos BNE RDF JSON-LD",
    )
    _append_request(
        requests,
        tmp_path,
        source_family="bne_linked_data",
        stage="endpoint",
        probe_id="bne_linked_data_docs",
        url="https://datos.bne.es/inicio.html",
        content_type="text/html; charset=utf-8",
        payload=b"RDF JSON-LD datos",
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


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[2] / ".omc/autoresearch/elements-baseline-replacement/runs/20260803-institutional-inventory-v16/endpoint-validation-manifest-iteration-0004.json").exists(),
    reason="requires unshipped research artifact: 20260803-institutional-inventory-v16/endpoint-validation-manifest-iteration-0004.json",
)
def test_local_replay_recomputes_two_validated_sources_from_raw_bytes(
    tmp_path: Path,
) -> None:
    contract = module.validate_contract()
    journal = _two_source_journal(contract, tmp_path)
    results, audit = module.evaluate_journal(contract, journal)
    assert audit["execution_integrity_pass"] is True
    assert audit["hypothesis_supported"] is True
    assert audit["metrics"][
        "validated_compliant_structured_source_family_count"
    ] == 2
    validated = {
        row["source_family"]
        for row in results["source_results"]
        if row["endpoint_validated"]
    }
    assert validated == {"library_of_congress", "bne_linked_data"}


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[2] / ".omc/autoresearch/elements-baseline-replacement/runs/20260803-institutional-inventory-v16/endpoint-validation-manifest-iteration-0004.json").exists(),
    reason="requires unshipped research artifact: 20260803-institutional-inventory-v16/endpoint-validation-manifest-iteration-0004.json",
)
def test_replay_detects_tampered_validation_claim(tmp_path: Path) -> None:
    contract = module.validate_contract()
    journal = _two_source_journal(contract, tmp_path)
    changed = copy.deepcopy(journal)
    changed["requests"][0]["validation_pass"] = False
    _, audit = module.evaluate_journal(contract, changed)
    assert audit["execution_integrity_pass"] is False
    assert any(
        violation.startswith("compliance_validation_mismatch")
        for violation in audit["integrity_violations"]
    )


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[2] / ".omc/autoresearch/elements-baseline-replacement/runs/20260803-institutional-inventory-v16/endpoint-validation-manifest-iteration-0004.json").exists(),
    reason="requires unshipped research artifact: 20260803-institutional-inventory-v16/endpoint-validation-manifest-iteration-0004.json",
)
def test_replay_detects_raw_evidence_tampering(tmp_path: Path) -> None:
    contract = module.validate_contract()
    journal = _two_source_journal(contract, tmp_path)
    Path(journal["requests"][0]["raw_evidence_path"]).write_bytes(b"tampered")
    _, audit = module.evaluate_journal(contract, journal)
    assert audit["execution_integrity_pass"] is False
    assert "raw_evidence_mismatch:1" in audit["integrity_violations"]


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[2] / ".omc/autoresearch/elements-baseline-replacement/runs/20260803-institutional-inventory-v16/endpoint-validation-manifest-iteration-0004.json").exists(),
    reason="requires unshipped research artifact: 20260803-institutional-inventory-v16/endpoint-validation-manifest-iteration-0004.json",
)
def test_evaluation_output_replay_is_byte_identical(tmp_path: Path) -> None:
    contract = module.validate_contract()
    journal = _two_source_journal(contract, tmp_path)
    first = module.evaluate_journal(contract, journal)
    first_hashes = module.write_evaluation_outputs(tmp_path / "first", *first)
    replay = module.evaluate_journal(contract, copy.deepcopy(journal))
    replay_hashes = module.write_evaluation_outputs(tmp_path / "replay", *replay)
    evaluation = module.evaluate(
        first[1], first_hashes, replay_hashes, tmp_path / "evaluation.json"
    )
    assert evaluation["deterministic_replay"] is True
    assert evaluation["execution_integrity_pass"] is True
    assert evaluation["pass"] is True
    assert evaluation["snapshot_authorized"] is False
    assert evaluation["final_test_read"] is False


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[2] / ".omc/autoresearch/elements-baseline-replacement/runs/20260803-institutional-inventory-v16/endpoint-validation-manifest-iteration-0004.json").exists(),
    reason="requires unshipped research artifact: 20260803-institutional-inventory-v16/endpoint-validation-manifest-iteration-0004.json",
)
def test_actual_frozen_iteration_recomputes_from_the_request_journal() -> None:
    contract = module.validate_contract()
    output_dir = module.DEFAULT_OUTPUT_DIR
    journal = module.read_json(output_dir / "request-journal.json")
    results, audit = module.evaluate_journal(contract, journal)
    assert results == module.read_json(output_dir / "endpoint-validation-results.json")
    assert audit == module.read_json(output_dir / "endpoint-validation-audit.json")
    assert audit["execution_integrity_pass"] is True
    assert audit["metrics"]["network_request_count"] == 20
    assert audit["metrics"][
        "validated_compliant_structured_source_family_count"
    ] == 2
    validated = {
        row["source_family"]
        for row in results["source_results"]
        if row["endpoint_validated"]
    }
    assert validated == {"library_of_congress", "bnf_gallica_api"}
    assert audit["forbidden_operation_counters"] == {
        "generic_web_searches": 0,
        "record_or_item_content_fetches": 0,
        "document_candidates_counted": 0,
        "iiif_canvases_or_images_opened": 0,
        "images_read_or_downloaded": 0,
        "model_predictions": 0,
        "labels_read": 0,
        "final_test_reads": 0,
        "runtime_writes": 0,
    }

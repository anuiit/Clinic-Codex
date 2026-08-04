from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/autoresearch_endpoint_discovery_v16.py"
spec = importlib.util.spec_from_file_location(
    "autoresearch_endpoint_discovery_v16", SCRIPT
)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def _built() -> tuple[dict, dict]:
    return module.build_outputs(module.validate_contract())


def test_contract_manifest_and_observations_are_hash_pinned() -> None:
    contract = module.validate_contract()
    assert {
        name: contract[name] for name in module.EXPECTED_HASHES
    } == module.EXPECTED_HASHES
    assert contract["spec"]["pre_registered_gates"][
        "logical_discovery_operations_total_max"
    ] == 56
    assert contract["evaluator"]["snapshot_authorized"] is False


def test_all_fourteen_frozen_sources_are_accounted_exactly_once() -> None:
    results, audit = _built()
    names = [row["source_family"] for row in results["source_results"]]
    assert len(names) == 14
    assert len(set(names)) == 14
    assert audit["gates"]["all_fourteen_sources_accounted_exactly_once"] is True


def test_only_loc_and_bne_are_machine_auditable_strict_candidates() -> None:
    results, audit = _built()
    strict = {
        row["source_family"]
        for row in results["source_results"]
        if row["role_recommendation"] == "strict_candidate"
    }
    assert strict == {"library_of_congress", "bne_bdh"}
    assert audit["metrics"][
        "machine_auditable_official_source_family_count"
    ] == 2
    assert audit["gates"]["minimum_7_machine_auditable_sources_pass"] is False


def test_evidence_summaries_have_reproducible_integrity_fields() -> None:
    results, _ = _built()
    assert results["evidence_register"]
    for item in results["evidence_register"]:
        assert len(item["summary_sha256"]) == 64
        assert item["summary_byte_size"] == len(item["summary"].encode("utf-8"))
        assert item["retrieved_at_utc"].endswith("Z")
        assert item["content_type"]
        assert item["url"].startswith("https://")


def test_budget_and_domain_violations_fail_iteration_closed() -> None:
    _, audit = _built()
    assert audit["metrics"]["logical_discovery_operation_count"] == 57
    assert audit["gates"]["global_56_operation_budget_pass"] is False
    assert audit["gates"]["per_source_budget_pass"] is False
    assert set(audit["violations"]["per_source_budget"]) == {
        "british_museum:open",
        "agn_mexico:open",
    }
    assert audit["forbidden_operation_counters"]["out_of_scope_page_opens"] == 1
    assert audit["decision"] == "invalid_discovery"


def test_sensitive_model_and_dataset_counters_remain_zero() -> None:
    _, audit = _built()
    counters = audit["forbidden_operation_counters"]
    assert counters["document_candidates_counted"] == 0
    assert counters["metadata_records_downloaded"] == 0
    assert counters["iiif_canvases_or_images_opened"] == 0
    assert counters["images_read_or_downloaded"] == 0
    assert counters["model_predictions"] == 0
    assert counters["labels_read"] == 0
    assert counters["final_test_reads"] == 0
    assert counters["runtime_writes"] == 0


def test_contract_derivation_is_valid_but_execution_integrity_is_not() -> None:
    _, audit = _built()
    assert audit["contract_derivation_verified"] is True
    assert audit["execution_integrity_pass"] is False
    assert audit["hypothesis_supported"] is False
    assert audit["snapshot_authorized"] is False
    assert audit["promotion_eligible"] is False


def test_replay_is_exact_and_runtime_is_untouched(tmp_path: Path) -> None:
    contract = module.validate_contract()
    first = module.build_outputs(contract)
    first_hashes = module.write_outputs(tmp_path / "first", *first)
    replay = module.build_outputs(contract)
    replay_hashes = module.write_outputs(tmp_path / "replay", *replay)
    evaluation = module.evaluate(
        first[1], first_hashes, replay_hashes, tmp_path / "evaluation.json"
    )
    assert evaluation["deterministic_replay"] is True
    assert evaluation["runtime_unchanged"] is True
    assert evaluation["execution_integrity_pass"] is False
    assert evaluation["pass"] is False
    assert evaluation["decision"] == "invalid_discovery"
    assert evaluation["final_test_read"] is False

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/autoresearch_institutional_inventory_v16.py"
spec = importlib.util.spec_from_file_location(
    "autoresearch_institutional_inventory_v16", SCRIPT
)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def _prepared() -> tuple[dict, dict, list[dict], list[dict]]:
    contract = module.validate_contract()
    frozen = module.load_frozen_inputs(contract)
    gates = contract["spec"]["pre_registered_gates"]
    documents = module.build_physical_documents(
        frozen["codex_rows"], contract["rules"], gates, frozen["v13_crosswalk"]
    )
    crosswalk = module.build_visual_crosswalk(
        frozen["visual_inventory"],
        frozen["v12_rules"],
        contract["rules"],
        documents,
        gates,
    )
    return contract, frozen, documents, crosswalk


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[2] / ".omc/autoresearch/elements-baseline-replacement/runs/20260803-institutional-inventory-v16/local-identity-crosswalk-rules.json").exists(),
    reason="requires unshipped research artifact: 20260803-institutional-inventory-v16/local-identity-crosswalk-rules.json",
)
def test_contract_is_hash_pinned_and_forbids_every_sensitive_operation() -> None:
    contract = module.validate_contract()
    assert contract["spec_sha256"] == module.EXPECTED_SPEC_SHA256
    assert contract["evaluator_sha256"] == module.EXPECTED_EVALUATOR_SHA256
    assert contract["rules_sha256"] == module.EXPECTED_RULES_SHA256
    integrity = contract["spec"]["integrity_constraints"]
    assert integrity["network_fetch_allowed"] is False
    assert integrity["image_read_or_download_allowed"] is False
    assert integrity["model_inference_allowed"] is False
    assert integrity["labels_or_final_test_read_allowed"] is False
    assert integrity["runtime_write_allowed"] is False


def test_normalization_is_frozen_and_removes_only_generic_identity_tokens() -> None:
    assert module.normalize_identity("Códice de ÁSCUNCIÓN!") == "de ascuncion"
    assert module.normalize_identity("Xolotl, Codex (Xolo)") == "xolotl xolo"


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[2] / ".omc/autoresearch/elements-baseline-replacement/runs/20260803-institutional-inventory-v16/local-identity-crosswalk-rules.json").exists(),
    reason="requires unshipped research artifact: 20260803-institutional-inventory-v16/local-identity-crosswalk-rules.json",
)
def test_local_catalogue_collapses_to_28_physical_documents() -> None:
    _, _, documents, _ = _prepared()
    assert len(documents) == 28
    all_ids = [value for row in documents for value in row["local_codex_ids"]]
    assert sorted(all_ids) == list(range(1, 51))
    assert len(all_ids) == len(set(all_ids))


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[2] / ".omc/autoresearch/elements-baseline-replacement/runs/20260803-institutional-inventory-v16/local-identity-crosswalk-rules.json").exists(),
    reason="requires unshipped research artifact: 20260803-institutional-inventory-v16/local-identity-crosswalk-rules.json",
)
def test_all_23_mh_sections_are_one_physical_document_with_archive_support() -> None:
    _, _, documents, _ = _prepared()
    mh = next(
        row
        for row in documents
        if row["physical_identity"] == "matricula_de_huexotzinco"
    )
    assert mh["local_codex_ids"] == list(range(19, 42))
    assert mh["supporting_unique_archives"] == [
        f"t_387_{index:02d}.zip" for index in range(1, 24)
    ]


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[2] / ".omc/autoresearch/elements-baseline-replacement/runs/20260803-institutional-inventory-v16/local-identity-crosswalk-rules.json").exists(),
    reason="requires unshipped research artifact: 20260803-institutional-inventory-v16/local-identity-crosswalk-rules.json",
)
def test_crosswalk_is_exhaustive_and_preserves_all_v12_exclusions() -> None:
    _, _, _, crosswalk = _prepared()
    assert len(crosswalk) == 50
    assert len({row["term_id"] for row in crosswalk}) == 50
    excluded = [
        row for row in crosswalk if row["disposition"] == "excluded_preserved_from_v12"
    ]
    assert len(excluded) == 9
    assert {row["term_id"] for row in excluded} == {
        112,
        118,
        121,
        124,
        224,
        293,
        318,
        376,
        759,
    }


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[2] / ".omc/autoresearch/elements-baseline-replacement/runs/20260803-institutional-inventory-v16/local-identity-crosswalk-rules.json").exists(),
    reason="requires unshipped research artifact: 20260803-institutional-inventory-v16/local-identity-crosswalk-rules.json",
)
def test_removing_a_v12_exclusion_is_rejected() -> None:
    contract = module.validate_contract()
    frozen = module.load_frozen_inputs(contract)
    gates = contract["spec"]["pre_registered_gates"]
    changed_rules = copy.deepcopy(contract["rules"])
    changed_rules["visual_exclusion_mappings"].pop()
    documents = module.build_physical_documents(
        frozen["codex_rows"], changed_rules, gates, frozen["v13_crosswalk"]
    )
    with pytest.raises(ValueError, match="preserve exactly every v12 exclusion"):
        module.build_visual_crosswalk(
            frozen["visual_inventory"],
            frozen["v12_rules"],
            changed_rules,
            documents,
            gates,
        )


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[2] / ".omc/autoresearch/elements-baseline-replacement/runs/20260803-institutional-inventory-v16/local-identity-crosswalk-rules.json").exists(),
    reason="requires unshipped research artifact: 20260803-institutional-inventory-v16/local-identity-crosswalk-rules.json",
)
def test_duplicate_physical_assignment_is_rejected() -> None:
    contract = module.validate_contract()
    frozen = module.load_frozen_inputs(contract)
    gates = contract["spec"]["pre_registered_gates"]
    changed_rules = copy.deepcopy(contract["rules"])
    changed_rules["physical_group_overrides"].append(
        {
            "physical_identity": "duplicate",
            "local_codex_ids": [19],
            "evidence": "test",
        }
    )
    with pytest.raises(ValueError, match="assigned twice"):
        module.build_physical_documents(
            frozen["codex_rows"], changed_rules, gates, frozen["v13_crosswalk"]
        )


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[2] / ".omc/autoresearch/elements-baseline-replacement/runs/20260803-institutional-inventory-v16/local-identity-crosswalk-rules.json").exists(),
    reason="requires unshipped research artifact: 20260803-institutional-inventory-v16/local-identity-crosswalk-rules.json",
)
def test_gate_zero_recertifies_41_and_short_circuits_acquisition() -> None:
    contract = module.validate_contract()
    _, _, audit = module.build_audit(contract)
    assert audit["metrics"][
        "strict_admissible_visual_lexicon_physical_document_ceiling"
    ] == 41
    assert audit["gates"]["minimum_42_gate_pass"] is False
    assert audit["instrument_acquired"] is False
    assert audit["decision"] == (
        "visual_lexicon_ceiling_recertified_but_insufficient"
    )
    assert set(audit["forbidden_operation_counters"].values()) == {0}
    assert audit["runtime_unchanged"] is True


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[2] / ".omc/autoresearch/elements-baseline-replacement/runs/20260803-institutional-inventory-v16/local-identity-crosswalk-rules.json").exists(),
    reason="requires unshipped research artifact: 20260803-institutional-inventory-v16/local-identity-crosswalk-rules.json",
)
def test_replay_is_byte_exact_but_overall_pass_remains_false(tmp_path: Path) -> None:
    contract = module.validate_contract()
    documents, crosswalk, audit = module.build_audit(contract)
    first = module.write_outputs(tmp_path / "first", documents, crosswalk, audit)
    replay = module.write_outputs(
        tmp_path / "replay",
        copy.deepcopy(documents),
        copy.deepcopy(crosswalk),
        copy.deepcopy(audit),
    )
    evaluation = module.evaluate(
        audit, first, replay, tmp_path / "evaluation.json"
    )
    assert evaluation["deterministic_replay"] is True
    assert evaluation["execution_integrity_pass"] is True
    assert evaluation["hypothesis_supported"] is True
    assert evaluation["pass"] is False
    assert evaluation["strict_component_ceiling"] == 41
    assert evaluation["final_test_read"] is False

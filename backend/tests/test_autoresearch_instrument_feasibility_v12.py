from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/autoresearch_instrument_feasibility_v12.py"
spec = importlib.util.spec_from_file_location(
    "autoresearch_instrument_feasibility_v12", SCRIPT
)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def _catalogue_html(contract: dict, *, wrong_term: int | None = None) -> bytes:
    exclusions = contract["rules"]["excluded_visual_lexicon_terms"]
    rows = {
        int(row["term_id"]): row["visual_manuscript"] for row in exclusions
    }
    dummy_id = 1000
    while len(rows) < 50:
        if dummy_id not in rows:
            rows[dummy_id] = f"Independent Manuscript {dummy_id} (I{dummy_id})"
        dummy_id += 1
    if wrong_term is not None:
        rows[wrong_term] = "Changed official display name"
    options = ['<option value="All">- Any -</option>']
    options.extend(
        f'<option value="{term_id}">{display_name}</option>'
        for term_id, display_name in sorted(rows.items())
    )
    return (
        '<html><select id="edit-field-manuscript-tid">'
        + "".join(options)
        + "</select></html>"
    ).encode()


def _local_evidence() -> dict:
    return {
        "local_codex_rows": 50,
        "local_archive_count": 49,
        "runtime_hashes": {},
        "source_hashes": {},
        "linked_hashes": {},
    }


def test_v12_contract_is_hash_pinned_and_forbids_inference() -> None:
    contract = module.validate_contract()
    assert contract["spec_sha256"] == module.EXPECTED_SPEC_SHA256
    assert contract["evaluator_sha256"] == module.EXPECTED_EVALUATOR_SHA256
    assert contract["exclusion_rules_sha256"] == (
        module.EXPECTED_EXCLUSION_RULES_SHA256
    )
    assert contract["spec"]["model_inference_allowed"] is False
    assert contract["spec"]["final_test_read_allowed"] is False


def test_parser_extracts_integer_terms_and_decodes_entities() -> None:
    html = b"""
    <select id="edit-field-manuscript-tid">
      <option value="All">- Any -</option>
      <option value="12">A &amp; B</option>
      <option value="4">C</option>
    </select>
    """
    assert module.parse_manuscript_inventory(
        html, select_id="edit-field-manuscript-tid"
    ) == [
        {"term_id": 4, "display_name": "C"},
        {"term_id": 12, "display_name": "A & B"},
    ]


def test_gate_zero_short_circuits_at_theoretical_ceiling_41() -> None:
    contract = module.validate_contract()
    _, audit = module.build_audit(
        _catalogue_html(contract),
        contract=contract,
        local_evidence=_local_evidence(),
    )
    assert audit["gate_zero"]["catalogue_manuscript_count"] == 50
    assert audit["gate_zero"]["predeclared_excluded_manuscript_count"] == 9
    assert (
        audit["gate_zero"]["strict_admissible_manuscript_component_ceiling"]
        == 41
    )
    assert audit["gate_zero"]["component_ceiling_gate_pass"] is False
    assert audit["short_circuit"]["triggered"] is True
    assert audit["short_circuit"]["record_detail_pages_fetched"] == 0
    assert audit["short_circuit"]["images_downloaded"] == 0
    assert audit["short_circuit"]["model_predictions"] == 0
    assert audit["decision"] == "instrument_not_acquired"


def test_changed_official_exclusion_name_is_rejected() -> None:
    contract = module.validate_contract()
    changed_term = contract["rules"]["excluded_visual_lexicon_terms"][0]["term_id"]
    with pytest.raises(ValueError, match="display mismatch"):
        module.build_audit(
            _catalogue_html(contract, wrong_term=changed_term),
            contract=contract,
            local_evidence=_local_evidence(),
        )


def test_replay_is_exact_but_feasibility_pass_remains_false(tmp_path: Path) -> None:
    contract = module.validate_contract()
    inventory, audit = module.build_audit(
        _catalogue_html(contract),
        contract=contract,
        local_evidence=_local_evidence(),
    )
    first = module.write_audit_outputs(tmp_path / "first", inventory, audit)
    replay = module.write_audit_outputs(
        tmp_path / "replay", copy.deepcopy(inventory), copy.deepcopy(audit)
    )
    evaluation = module.evaluate(
        audit=audit,
        first_outputs=first,
        replay_outputs=replay,
        output_path=tmp_path / "evaluation.json",
    )
    assert evaluation["execution_integrity_pass"] is True
    assert evaluation["pass"] is False
    assert evaluation["instrument_acquired"] is False
    assert evaluation["strict_component_ceiling"] == 41
    assert evaluation["promotion_eligible"] is False

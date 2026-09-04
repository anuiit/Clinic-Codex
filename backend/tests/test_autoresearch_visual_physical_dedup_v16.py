from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/autoresearch_visual_physical_dedup_v16.py"
spec = importlib.util.spec_from_file_location(
    "autoresearch_visual_physical_dedup_v16", SCRIPT
)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def _built() -> tuple[list[dict], list[dict], dict, dict]:
    contract = module.validate_contract()
    return module.build_gate_zero_bis(contract, module.load_inputs(contract))


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[2] / ".omc/autoresearch/elements-baseline-replacement/runs/20260803-institutional-inventory-v16/visual-physical-dedup-rules.json").exists(),
    reason="requires unshipped research artifact: 20260803-institutional-inventory-v16/visual-physical-dedup-rules.json",
)
def test_contract_is_hash_pinned_and_forbids_sensitive_operations() -> None:
    contract = module.validate_contract()
    assert contract["spec_sha256"] == module.EXPECTED_SPEC_SHA256
    assert contract["evaluator_sha256"] == module.EXPECTED_EVALUATOR_SHA256
    assert contract["rules_sha256"] == module.EXPECTED_RULES_SHA256
    integrity = contract["spec"]["integrity_constraints"]
    assert integrity["network_fetch_allowed"] is False
    assert integrity["image_read_or_download_allowed"] is False
    assert integrity["model_inference_allowed"] is False
    assert integrity["labels_or_final_test_read_allowed"] is False


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[2] / ".omc/autoresearch/elements-baseline-replacement/runs/20260803-institutional-inventory-v16/visual-physical-dedup-rules.json").exists(),
    reason="requires unshipped research artifact: 20260803-institutional-inventory-v16/visual-physical-dedup-rules.json",
)
def test_all_50_terms_remain_traceable_and_v12_exclusions_are_preserved() -> None:
    dispositions, _, _, _ = _built()
    assert len(dispositions) == 50
    assert len({row["term_id"] for row in dispositions}) == 50
    assert sum(
        row["disposition"] == "excluded_preserved_from_v12"
        for row in dispositions
    ) == 9


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[2] / ".omc/autoresearch/elements-baseline-replacement/runs/20260803-institutional-inventory-v16/visual-physical-dedup-rules.json").exists(),
    reason="requires unshipped research artifact: 20260803-institutional-inventory-v16/visual-physical-dedup-rules.json",
)
def test_same_archival_signature_folios_form_one_component() -> None:
    dispositions, _, _, _ = _built()
    by_id = {row["term_id"]: row for row in dispositions}
    assert by_id[291]["component_id"] == "agn_tierras_1735_exp_2"
    assert by_id[292]["component_id"] == "agn_tierras_1735_exp_2"


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[2] / ".omc/autoresearch/elements-baseline-replacement/runs/20260803-institutional-inventory-v16/visual-physical-dedup-rules.json").exists(),
    reason="requires unshipped research artifact: 20260803-institutional-inventory-v16/visual-physical-dedup-rules.json",
)
def test_lienzo_witnesses_form_one_provenance_component() -> None:
    dispositions, components, _, _ = _built()
    by_id = {row["term_id"]: row for row in dispositions}
    assert by_id[416]["component_id"] == "lienzo_de_tlaxcala_provenance"
    assert by_id[728]["component_id"] == "lienzo_de_tlaxcala_provenance"
    component = next(
        row
        for row in components
        if row["component_id"] == "lienzo_de_tlaxcala_provenance"
    )
    assert component["component_kind"] == "shared_original_provenance"


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[2] / ".omc/autoresearch/elements-baseline-replacement/runs/20260803-institutional-inventory-v16/visual-physical-dedup-rules.json").exists(),
    reason="requires unshipped research artifact: 20260803-institutional-inventory-v16/visual-physical-dedup-rules.json",
)
def test_two_new_ambiguities_are_excluded_but_specific_agn_records_stay_distinct() -> None:
    dispositions, _, _, _ = _built()
    by_id = {row["term_id"]: row for row in dispositions}
    assert by_id[122]["component_id"] is None
    assert by_id[342]["component_id"] is None
    assert by_id[300]["component_id"] != by_id[367]["component_id"]
    assert by_id[300]["component_id"] is not None
    assert by_id[367]["component_id"] is not None


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[2] / ".omc/autoresearch/elements-baseline-replacement/runs/20260803-institutional-inventory-v16/visual-physical-dedup-rules.json").exists(),
    reason="requires unshipped research artifact: 20260803-institutional-inventory-v16/visual-physical-dedup-rules.json",
)
def test_corrected_strict_ceiling_is_37_and_all_admitted_terms_are_pending_collision() -> None:
    dispositions, components, _, audit = _built()
    admitted = [row for row in dispositions if row["component_id"] is not None]
    assert len(admitted) == 39
    assert len(components) == 37
    assert all(
        row["disposition"]
        == "admissible_no_known_overlap_pending_collision_audit"
        for row in admitted
    )
    assert audit["metrics"][
        "strict_independent_visual_lexicon_component_ceiling"
    ] == 37
    assert audit["decision"] == (
        "visual_lexicon_physical_ceiling_corrected_but_insufficient"
    )


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[2] / ".omc/autoresearch/elements-baseline-replacement/runs/20260803-institutional-inventory-v16/visual-physical-dedup-rules.json").exists(),
    reason="requires unshipped research artifact: 20260803-institutional-inventory-v16/visual-physical-dedup-rules.json",
)
def test_codex_49_aliases_and_future_telleriano_candidates_are_frozen() -> None:
    _, _, envelope, _ = _built()
    aliases = next(
        row["aliases"]
        for row in envelope["local_alias_additions"]
        if row["physical_identity"] == "codex_49"
    )
    assert set(aliases) == {"TEPEUCILA", "tepeucila", "t_tepeuc", "t_tepeuc.zip"}
    future = envelope["future_external_exclusion_candidates"][0]
    assert future["against_physical_identity"] == "codex_18"
    assert "Codex Ríos" in future["aliases"]


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[2] / ".omc/autoresearch/elements-baseline-replacement/runs/20260803-institutional-inventory-v16/visual-physical-dedup-rules.json").exists(),
    reason="requires unshipped research artifact: 20260803-institutional-inventory-v16/visual-physical-dedup-rules.json",
)
def test_group_and_exclusion_overlap_is_rejected() -> None:
    contract = module.validate_contract()
    changed = copy.deepcopy(contract)
    changed["rules"]["additional_ambiguity_exclusions"][0]["term_id"] = 291
    with pytest.raises(ValueError, match="grouped and additionally excluded"):
        module.build_gate_zero_bis(changed, module.load_inputs(contract))


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[2] / ".omc/autoresearch/elements-baseline-replacement/runs/20260803-institutional-inventory-v16/visual-physical-dedup-rules.json").exists(),
    reason="requires unshipped research artifact: 20260803-institutional-inventory-v16/visual-physical-dedup-rules.json",
)
def test_replay_is_exact_and_runtime_remains_untouched(tmp_path: Path) -> None:
    contract = module.validate_contract()
    first_outputs = module.build_audit(contract)
    first = module.write_outputs(tmp_path / "first", *first_outputs)
    replay_outputs = module.build_audit(contract)
    replay = module.write_outputs(tmp_path / "replay", *replay_outputs)
    evaluation = module.evaluate(
        first_outputs[3], first, replay, tmp_path / "evaluation.json"
    )
    assert evaluation["deterministic_replay"] is True
    assert evaluation["execution_integrity_pass"] is True
    assert evaluation["hypothesis_supported"] is True
    assert evaluation["corrected_strict_component_ceiling"] == 37
    assert evaluation["runtime_unchanged"] is True
    assert evaluation["pass"] is False
    assert evaluation["final_test_read"] is False

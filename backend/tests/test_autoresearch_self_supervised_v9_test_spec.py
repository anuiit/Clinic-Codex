from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / ".omc" / "autoresearch" / "elements-baseline-replacement" / "runs" / "20260803-self-supervised-v9" / "test-spec.json"


def test_self_supervised_v9_test_spec_is_present_and_machine_readable() -> None:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))

    assert spec["schema_version"] == "autoresearch-self-supervised-b0-c1-test-spec.v1"
    assert spec["mission"] == "elements-baseline-replacement"
    assert spec["run_id"] == "20260803-self-supervised-v9"
    assert spec["comparison_scope"]["baseline_id"] == "B0"
    assert spec["comparison_scope"]["candidate_id"] == "C1"
    assert "fold-local VICReg pretraining" in spec["comparison_scope"]["change_factor"]
    assert spec["alignment_source"] == (
        ".omc/autoresearch/elements-baseline-replacement/runs/"
        "20260803-self-supervised-v9/specs/iteration-0001.json"
    )


def test_self_supervised_v9_test_spec_locks_the_comparison_contract() -> None:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))

    invariants = spec["non_negotiable_invariants"]
    guardrails = spec["data_guardrails"]
    budget = spec["training_budget"]
    evaluation = spec["evaluation_contract"]

    assert guardrails == {
        "fold_local_isolation": True,
        "component_disjointness": True,
        "oof_contamination_forbidden": True,
        "held_out_rows_never_seen": True,
        "paired_predictions_required": True,
    }
    assert budget["supervised"]["same_for_both_arms"] is True
    assert budget["supervised"]["same_epochs"] is True
    assert budget["supervised"]["same_batch_size"] is True
    assert budget["supervised"]["same_seed_schedule"] is True
    assert budget["supervised"]["seed_schedule"] == [17, 42, 73]
    assert budget["ssl_pretraining"]["method"] == "VICReg"
    assert budget["ssl_pretraining"]["views_per_image"] == 8
    assert budget["ssl_pretraining"]["epochs"] == 30
    assert budget["ssl_pretraining"]["fold_local_only"] is True
    assert budget["ssl_pretraining"]["applies_to"] == "candidate_only"
    assert evaluation["selection_rule"] == (
        "Candidate may advance only if it improves paired OOF top-1 and does not regress paired OOF macro top-1."
    )
    assert "NaN/Inf" in "\n".join(spec["failure_conditions"])
    assert "embedding collapse" in "\n".join(spec["failure_conditions"])
    assert len(invariants) >= 6

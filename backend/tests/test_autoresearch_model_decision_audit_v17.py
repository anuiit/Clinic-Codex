from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/autoresearch_model_decision_audit_v17.py"
spec = importlib.util.spec_from_file_location(
    "autoresearch_model_decision_audit_v17", SCRIPT
)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_contract_is_hash_pinned_and_forbids_candidate_work() -> None:
    contract = module.validate_contract()
    assert {
        name: contract[name] for name in module.EXPECTED_CONTRACT_HASHES
    } == module.EXPECTED_CONTRACT_HASHES
    boundaries = contract["spec"]["hard_boundaries"]
    assert boundaries["candidate_prediction"] is False
    assert boundaries["training"] is False
    assert boundaries["final_test_read"] is False
    assert boundaries["runtime_write"] is False


def test_factor_selection_chooses_only_eligible_factor() -> None:
    assert (
        module.select_factor(True, False, 0.6, 0.1)
        == "select_supervised_multiview_readout"
    )
    assert (
        module.select_factor(False, True, 0.1, 0.6)
        == "select_hierarchical_shrunk_prototypes"
    )
    assert module.select_factor(False, False, 0.9, 0.9) == "select_no_factor"


def test_factor_selection_uses_error_burden_and_view_tiebreak() -> None:
    assert (
        module.select_factor(True, True, 0.7, 0.2)
        == "select_supervised_multiview_readout"
    )
    assert (
        module.select_factor(True, True, 0.2, 0.7)
        == "select_hierarchical_shrunk_prototypes"
    )
    assert (
        module.select_factor(True, True, 0.5, 0.5)
        == "select_supervised_multiview_readout"
    )


def test_support_bins_are_frozen_at_eight_and_thirty_one() -> None:
    assert module.support_bin(8, 8, 31) == "le_8"
    assert module.support_bin(9, 8, 31) == "9_to_31"
    assert module.support_bin(31, 8, 31) == "9_to_31"
    assert module.support_bin(32, 8, 31) == "ge_32"


def test_recorded_output_respects_no_candidate_boundary_when_present() -> None:
    evaluation_path = module.DEFAULT_EVALUATION_PATH
    if not evaluation_path.is_file():
        return
    evaluation = module.read_json(evaluation_path)
    assert evaluation["execution_integrity_pass"] is True
    assert evaluation["candidate_prediction_count"] == 0
    assert evaluation["training_operation_count"] == 0
    assert evaluation["final_test_read"] is False
    assert evaluation["runtime_unchanged"] is True
    assert evaluation["selected_factor_authorizes_execution"] is False

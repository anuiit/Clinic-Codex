from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/autoresearch_rank_guard_diagnostic_v10.py"
SPEC = ROOT / ".omc/autoresearch/elements-baseline-replacement/runs/20260803-self-supervised-v10/specs/iteration-0006.json"
EVALUATOR = ROOT / ".omc/autoresearch/elements-baseline-replacement/runs/20260803-self-supervised-v10/evaluator-iteration-0006.json"
spec = importlib.util.spec_from_file_location("autoresearch_rank_guard_diagnostic_v10", SCRIPT)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_iteration_6_contract_is_hash_pinned() -> None:
    contract = module.validate_contract(SPEC, EVALUATOR)
    assert contract["spec_sha256"] == module.EXPECTED_SPEC_SHA256
    assert contract["evaluator_sha256"] == module.EXPECTED_EVALUATOR_SHA256


def test_average_ranks_handles_ties() -> None:
    assert module.average_ranks([30.0, 10.0, 20.0, 20.0]) == [4.0, 1.0, 2.5, 2.5]


def test_worst_fold_classification_is_directional_only() -> None:
    assert module.classify_worst_fold(-0.01) == "aligned"
    assert module.classify_worst_fold(0.0) == "not_aligned"
    assert module.classify_worst_fold(0.01) == "not_aligned"


def test_runner_exposes_run_command() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "run" in completed.stdout

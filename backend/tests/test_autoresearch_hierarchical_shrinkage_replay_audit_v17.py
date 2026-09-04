from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/autoresearch_hierarchical_shrinkage_replay_audit_v17.py"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "autoresearch_hierarchical_shrinkage_replay_audit_v17_tested", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[2] / ".omc/autoresearch/elements-baseline-replacement/runs/20260803-model-decision-audit-v17/iteration-0003/summary.json").exists(),
    reason="requires unshipped research artifact: iteration-0003/summary.json",
)
def test_canonical_and_replay_are_byte_identical() -> None:
    module = load_module()

    result = module.audit_replay(
        module.DEFAULT_CANONICAL_DIR,
        module.DEFAULT_REPLAY_DIR,
    )

    assert result["pass"] is True
    assert all(
        item["byte_identical"] for item in result["comparisons"].values()
    )
    assert result["normalization_contract"]["summary_removed_keys"] == []
    assert result["normalization_contract"]["pair_removed_keys"] == []
    assert result["same_machine_only"] is True
    assert result["cross_machine_determinism_claimed"] is False

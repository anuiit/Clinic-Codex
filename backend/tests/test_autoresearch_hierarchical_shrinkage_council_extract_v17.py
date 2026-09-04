from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/autoresearch_hierarchical_shrinkage_council_extract_v17.py"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "autoresearch_hierarchical_shrinkage_council_extract_v17_tested", SCRIPT
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
def test_packet_preserves_decision_gates_support_and_replay() -> None:
    module = load_module()

    packet = module.build_packet()
    serialized = json.dumps(packet, indent=2, sort_keys=True, ensure_ascii=False)

    assert packet["decision"] == "neutral_preservation"
    assert packet["claim"] == "not_evidence_of_a_better_model"
    assert all(packet["integrity_gate_passes"].values())
    assert all(packet["engagement_gate_passes"].values())
    assert all(packet["b0_mission_anchor_gate_passes"].values())
    assert all(packet["c1_noninferiority_gate_passes"].values())
    assert packet["deterministic_replay"]["pass"] is True
    assert "three_bin_candidate_vs_c1" in packet["support_diagnostics"]
    assert len(packet["diagnostics"]) == 15
    assert len(serialized) <= 60_000

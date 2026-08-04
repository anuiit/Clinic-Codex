from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/autoresearch_lora_vicreg_replay_audit_v18.py"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "autoresearch_lora_vicreg_replay_audit_v18_tested", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def populate_pair(module, canonical: Path, replay: Path) -> None:
    for directory in (canonical, replay):
        (directory / "candidate_states").mkdir(parents=True)
        for name in module.REQUIRED_CORE_FILES:
            (directory / name).write_bytes(f"core:{name}".encode())
        for index in range(module.EXPECTED_STATE_COUNT):
            name = f"fold-{index // 3 + 1:02d}-seed-{index % 3:02d}.bin"
            (directory / "candidate_states" / name).write_bytes(
                f"candidate-state:{index}".encode()
            )


def test_compare_required_artifacts_includes_all_states(tmp_path: Path) -> None:
    module = load_module()
    canonical = tmp_path / "canonical"
    replay = tmp_path / "replay"
    populate_pair(module, canonical, replay)

    comparisons = module.compare_required_artifacts(canonical, replay)

    assert len(comparisons) == len(module.REQUIRED_CORE_FILES) + 15
    assert all(item["byte_identical"] for item in comparisons.values())
    assert sum(name.startswith("candidate_states/") for name in comparisons) == 15


def test_compare_reports_byte_difference(tmp_path: Path) -> None:
    module = load_module()
    canonical = tmp_path / "canonical"
    replay = tmp_path / "replay"
    populate_pair(module, canonical, replay)
    changed = replay / "candidate_states/fold-01-seed-00.bin"
    changed.write_bytes(b"changed")

    comparisons = module.compare_required_artifacts(canonical, replay)

    assert comparisons[
        "candidate_states/fold-01-seed-00.bin"
    ]["byte_identical"] is False


def test_state_set_mismatch_is_rejected(tmp_path: Path) -> None:
    module = load_module()
    canonical = tmp_path / "canonical"
    replay = tmp_path / "replay"
    populate_pair(module, canonical, replay)
    (replay / "candidate_states/fold-01-seed-00.bin").unlink()

    with pytest.raises(ValueError, match="artifact sets differ"):
        module.compare_required_artifacts(canonical, replay)


def test_replay_normalization_contract_is_identity_and_pinned() -> None:
    module = load_module()

    assert module.v18_3.REPLAY_NORMALIZATION_CONTRACT[
        "summary_removed_keys"
    ] == []
    assert module.v18_3.REPLAY_NORMALIZATION_CONTRACT[
        "pair_removed_keys"
    ] == []
    assert module.v18_3.replay_normalization_sha256() == (
        module.v18_3.EXPECTED_REPLAY_NORMALIZATION_SHA256
    )

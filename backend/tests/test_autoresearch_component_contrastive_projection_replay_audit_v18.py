from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    ROOT
    / "scripts/autoresearch_component_contrastive_projection_replay_audit_v18.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "autoresearch_component_contrastive_projection_replay_audit_v18_tested",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def populate_pair(module, canonical: Path, replay: Path) -> None:
    for directory in (canonical, replay):
        (directory / "adapted_layers").mkdir(parents=True)
        for name in module.REQUIRED_CORE_FILES:
            (directory / name).write_bytes(f"core:{name}".encode())
        for index in range(module.EXPECTED_LAYER_COUNT):
            name = f"fold-{index // 3 + 1:02d}-seed-{index % 3:02d}.bin"
            (directory / "adapted_layers" / name).write_bytes(
                f"adapted-layer:{index}".encode()
            )


def test_compare_required_artifacts_includes_all_layers(tmp_path: Path) -> None:
    module = load_module()
    canonical = tmp_path / "canonical"
    replay = tmp_path / "replay"
    populate_pair(module, canonical, replay)

    comparisons = module.compare_required_artifacts(canonical, replay)

    assert len(comparisons) == len(module.REQUIRED_CORE_FILES) + 15
    assert all(item["byte_identical"] for item in comparisons.values())
    assert sum(name.startswith("adapted_layers/") for name in comparisons) == 15


def test_compare_reports_byte_difference(tmp_path: Path) -> None:
    module = load_module()
    canonical = tmp_path / "canonical"
    replay = tmp_path / "replay"
    populate_pair(module, canonical, replay)
    (replay / "adapted_layers/fold-01-seed-00.bin").write_bytes(b"changed")

    comparisons = module.compare_required_artifacts(canonical, replay)

    assert comparisons[
        "adapted_layers/fold-01-seed-00.bin"
    ]["byte_identical"] is False


def test_layer_set_mismatch_is_rejected(tmp_path: Path) -> None:
    module = load_module()
    canonical = tmp_path / "canonical"
    replay = tmp_path / "replay"
    populate_pair(module, canonical, replay)
    (replay / "adapted_layers/fold-01-seed-00.bin").unlink()

    with pytest.raises(ValueError, match="artifact sets differ"):
        module.compare_required_artifacts(canonical, replay)


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[2] / ".omc/autoresearch/elements-baseline-replacement/runs/20260804-discriminative-readout-v18/iteration-0002/adapted_layers").exists(),
    reason="requires unshipped research artifact: iteration-0002/adapted_layers",
)
def test_real_canonical_replay_audit_passes() -> None:
    module = load_module()

    audit = module.audit_replay(
        module.DEFAULT_CANONICAL_DIR,
        module.DEFAULT_REPLAY_DIR,
    )

    assert audit["pass"] is True
    assert audit["decision"] == "not_supported"
    assert audit["gates"]["all_required_artifacts_byte_identical"] is True
    assert audit["gates"]["compared_adapted_layer_artifact_count"] is True

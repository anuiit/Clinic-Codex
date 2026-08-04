from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    ROOT / "scripts/autoresearch_discriminative_cosine_head_replay_audit_v18.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "autoresearch_discriminative_cosine_head_replay_audit_v18_tested",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def populate_pair(module, canonical: Path, replay: Path) -> None:
    for directory in (canonical, replay):
        (directory / "heads").mkdir(parents=True)
        for name in module.REQUIRED_CORE_FILES:
            (directory / name).write_bytes(f"core:{name}".encode())
        for index in range(module.EXPECTED_HEAD_COUNT):
            name = f"fold-{index // 3 + 1:02d}-seed-{index % 3:02d}.bin"
            (directory / "heads" / name).write_bytes(f"head:{index}".encode())


def test_compare_required_artifacts_includes_all_heads(tmp_path: Path) -> None:
    module = load_module()
    canonical = tmp_path / "canonical"
    replay = tmp_path / "replay"
    populate_pair(module, canonical, replay)

    comparisons = module.compare_required_artifacts(canonical, replay)

    assert len(comparisons) == len(module.REQUIRED_CORE_FILES) + 15
    assert all(item["byte_identical"] for item in comparisons.values())
    assert sum(name.startswith("heads/") for name in comparisons) == 15


def test_compare_reports_byte_difference(tmp_path: Path) -> None:
    module = load_module()
    canonical = tmp_path / "canonical"
    replay = tmp_path / "replay"
    populate_pair(module, canonical, replay)
    (replay / "heads/fold-01-seed-00.bin").write_bytes(b"changed")

    comparisons = module.compare_required_artifacts(canonical, replay)

    assert comparisons["heads/fold-01-seed-00.bin"]["byte_identical"] is False


def test_head_set_mismatch_is_rejected(tmp_path: Path) -> None:
    module = load_module()
    canonical = tmp_path / "canonical"
    replay = tmp_path / "replay"
    populate_pair(module, canonical, replay)
    (replay / "heads/fold-01-seed-00.bin").unlink()

    with pytest.raises(ValueError, match="artifact sets differ"):
        module.compare_required_artifacts(canonical, replay)


def test_real_canonical_replay_audit_passes() -> None:
    module = load_module()

    audit = module.audit_replay(
        module.DEFAULT_CANONICAL_DIR,
        module.DEFAULT_REPLAY_DIR,
    )

    assert audit["pass"] is True
    assert audit["decision"] == "not_supported"
    assert audit["gates"]["all_required_artifacts_byte_identical"] is True
    assert audit["gates"]["compared_head_artifact_count"] is True

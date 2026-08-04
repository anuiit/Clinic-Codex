from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import autoresearch_storage_probe_v18 as probe


def test_storage_probe_writes_verifies_and_removes_file(tmp_path: Path) -> None:
    probe_file = tmp_path / "probe.bin"
    audit_output = tmp_path / "audit.json"

    result = probe.run_probe(
        probe_file=probe_file,
        audit_output=audit_output,
        total_bytes=4 * 1024**2,
        chunk_bytes=1024**2,
        random_block_bytes=256 * 1024,
        minimum_write_bytes_per_second=0,
        minimum_random_read_bytes_per_second=0,
        seed=123,
    )

    assert result["pass"] is True
    assert result["gates"] == {
        "apparent_size_exact": True,
        "physical_allocation_at_least_99_percent": True,
        "sequential_mmap_sha256_match": True,
        "write_throughput_pass": True,
        "random_mmap_throughput_pass": True,
    }
    assert result["probe_file_removed"] is True
    assert not probe_file.exists()
    assert json.loads(audit_output.read_text(encoding="utf-8")) == result


def test_storage_probe_refuses_existing_probe_file(tmp_path: Path) -> None:
    probe_file = tmp_path / "probe.bin"
    probe_file.write_bytes(b"preserve")
    audit_output = tmp_path / "audit.json"

    with pytest.raises(FileExistsError, match="refusing to overwrite probe file"):
        probe.run_probe(
            probe_file=probe_file,
            audit_output=audit_output,
            total_bytes=1024,
            chunk_bytes=1024,
            random_block_bytes=512,
            minimum_write_bytes_per_second=0,
            minimum_random_read_bytes_per_second=0,
            seed=123,
        )

    assert probe_file.read_bytes() == b"preserve"
    assert not audit_output.exists()

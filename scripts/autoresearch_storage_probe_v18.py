#!/usr/bin/env python3
"""Deterministic non-scientific storage probe for v18 recovery output."""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import os
import random
import time
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "autoresearch-discriminative-readout-v18.storage-probe-v1"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_probe(
    *,
    probe_file: Path,
    audit_output: Path,
    total_bytes: int,
    chunk_bytes: int,
    random_block_bytes: int,
    minimum_write_bytes_per_second: float,
    minimum_random_read_bytes_per_second: float,
    seed: int,
) -> dict[str, Any]:
    if probe_file.exists():
        raise FileExistsError(f"refusing to overwrite probe file: {probe_file}")
    if audit_output.exists():
        raise FileExistsError(f"refusing to overwrite probe audit: {audit_output}")
    if total_bytes <= 0 or chunk_bytes <= 0 or random_block_bytes <= 0:
        raise ValueError("probe sizes must be positive")
    if total_bytes % chunk_bytes:
        raise ValueError("total_bytes must be divisible by chunk_bytes")
    if chunk_bytes % random_block_bytes:
        raise ValueError("chunk_bytes must be divisible by random_block_bytes")

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "pass": False,
        "probe_file": str(probe_file),
        "audit_output": str(audit_output),
        "total_bytes": total_bytes,
        "chunk_bytes": chunk_bytes,
        "random_block_bytes": random_block_bytes,
        "minimum_write_bytes_per_second": minimum_write_bytes_per_second,
        "minimum_random_read_bytes_per_second": minimum_random_read_bytes_per_second,
        "seed": seed,
        "dataset_access_count": 0,
        "model_load_count": 0,
        "candidate_optimizer_steps": 0,
        "candidate_predictions": 0,
        "final_test_read": False,
        "runtime_write": False,
    }
    probe_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        chunk = random.Random(seed).randbytes(chunk_bytes)
        expected = hashlib.sha256()
        write_started = time.perf_counter()
        with probe_file.open("xb", buffering=0) as handle:
            for _ in range(total_bytes // chunk_bytes):
                handle.write(chunk)
                expected.update(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        write_seconds = time.perf_counter() - write_started
        stat = probe_file.stat()
        allocated_bytes = int(getattr(stat, "st_blocks", 0)) * 512
        result.update(
            {
                "apparent_bytes": stat.st_size,
                "allocated_bytes": allocated_bytes,
                "allocated_ratio": allocated_bytes / total_bytes,
                "write_seconds": write_seconds,
                "write_bytes_per_second": total_bytes / write_seconds,
                "expected_sha256": expected.hexdigest(),
            }
        )

        sequential = hashlib.sha256()
        with probe_file.open("rb", buffering=0) as handle:
            with mmap.mmap(handle.fileno(), length=0, access=mmap.ACCESS_READ) as mapped:
                for offset in range(0, total_bytes, chunk_bytes):
                    sequential.update(mapped[offset : offset + chunk_bytes])
                order = list(range(total_bytes // random_block_bytes))
                random.Random(seed ^ 0xA5A5A5A5).shuffle(order)
                random_digest = hashlib.sha256()
                random_started = time.perf_counter()
                for block_index in order:
                    offset = block_index * random_block_bytes
                    random_digest.update(
                        mapped[offset : offset + random_block_bytes]
                    )
                random_seconds = time.perf_counter() - random_started

        sequential_sha256 = sequential.hexdigest()
        random_bytes_per_second = total_bytes / random_seconds
        gates = {
            "apparent_size_exact": stat.st_size == total_bytes,
            "physical_allocation_at_least_99_percent": allocated_bytes >= int(total_bytes * 0.99),
            "sequential_mmap_sha256_match": sequential_sha256 == expected.hexdigest(),
            "write_throughput_pass": (
                total_bytes / write_seconds >= minimum_write_bytes_per_second
            ),
            "random_mmap_throughput_pass": (
                random_bytes_per_second >= minimum_random_read_bytes_per_second
            ),
        }
        result.update(
            {
                "sequential_mmap_sha256": sequential_sha256,
                "random_mmap_sha256": random_digest.hexdigest(),
                "random_read_seconds": random_seconds,
                "random_read_bytes_per_second": random_bytes_per_second,
                "gates": gates,
                "pass": all(gates.values()),
            }
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if probe_file.exists():
            probe_file.unlink()
        result["probe_file_removed"] = not probe_file.exists()
        result["pass"] = bool(result["pass"] and result["probe_file_removed"])
        write_json(audit_output, result)

    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-file", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--bytes", type=int, default=2 * 1024**3)
    parser.add_argument("--chunk-bytes", type=int, default=8 * 1024**2)
    parser.add_argument("--random-block-bytes", type=int, default=512 * 1024)
    parser.add_argument("--minimum-write-bytes-per-second", type=float, default=50 * 1024**2)
    parser.add_argument("--minimum-random-read-bytes-per-second", type=float, default=50 * 1024**2)
    parser.add_argument("--seed", type=int, default=20260804)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = run_probe(
            probe_file=args.probe_file,
            audit_output=args.audit_output,
            total_bytes=args.bytes,
            chunk_bytes=args.chunk_bytes,
            random_block_bytes=args.random_block_bytes,
            minimum_write_bytes_per_second=args.minimum_write_bytes_per_second,
            minimum_random_read_bytes_per_second=args.minimum_random_read_bytes_per_second,
            seed=args.seed,
        )
    except Exception as exc:
        result = {
            "schema_version": SCHEMA_VERSION,
            "pass": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    print(canonical_json(result))
    return 0 if result.get("pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())

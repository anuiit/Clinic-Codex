#!/usr/bin/env python3
"""Audit independent replay equivalence for the v17.2 multi-view experiment."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import autoresearch_self_supervised_v9 as v9  # noqa: E402


RUN = ROOT / ".omc/autoresearch/elements-baseline-replacement/runs/20260803-model-decision-audit-v17"
CANONICAL = RUN / "iteration-0002"
REPLAY = RUN / "iteration-0002-replay"
OUTPUT = RUN / "iteration-0002-replay-audit.json"
EXPECTED_RUNTIME = {
    "projection": "0ad6ce348eaf44c8844415a769b048b8f155017befa93f16254b1263b9241210",
    "prototypes": "ffc52f03696f03780a01302708b53e319dc24069187a6fd1a2927ea537b1cc54",
    "config": "f2150a97ec7eee50a42ac1a108ae4bc092d46a6b8e4a2d91de59349560fbf18b",
}
PATH_KEYS = {"candidate_checkpoint_path", "view_plan_audit_path"}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_output_paths(value: Any, *, key: str | None = None) -> Any:
    if isinstance(value, dict):
        return {
            item_key: normalize_output_paths(item_value, key=item_key)
            for item_key, item_value in value.items()
        }
    if isinstance(value, list):
        return [normalize_output_paths(item, key=key) for item in value]
    if key in PATH_KEYS and isinstance(value, str):
        return Path(value).name
    return value


def keyed_files(directory: Path, pattern: str) -> dict[str, Path]:
    return {path.name: path for path in sorted(directory.glob(pattern))}


def compare_file_sets(
    canonical_dir: Path,
    replay_dir: Path,
    pattern: str,
) -> dict[str, Any]:
    canonical = keyed_files(canonical_dir, pattern)
    replay = keyed_files(replay_dir, pattern)
    names_match = set(canonical) == set(replay)
    hashes = {
        name: {
            "canonical": v9.sha256_file(canonical[name]),
            "replay": v9.sha256_file(replay[name]),
        }
        for name in sorted(set(canonical) & set(replay))
    }
    return {
        "file_count": len(hashes),
        "names_match": names_match,
        "byte_identical_count": sum(
            item["canonical"] == item["replay"] for item in hashes.values()
        ),
        "all_byte_identical": names_match
        and all(item["canonical"] == item["replay"] for item in hashes.values()),
        "hashes": hashes,
    }


def compare_json_sets_normalized(
    canonical_dir: Path,
    replay_dir: Path,
    pattern: str,
) -> dict[str, Any]:
    result = compare_file_sets(canonical_dir, replay_dir, pattern)
    canonical = keyed_files(canonical_dir, pattern)
    replay = keyed_files(replay_dir, pattern)
    normalized_hashes = {
        name: {
            "canonical": v9.sha256_json(
                normalize_output_paths(read_json(canonical[name]))
            ),
            "replay": v9.sha256_json(
                normalize_output_paths(read_json(replay[name]))
            ),
        }
        for name in sorted(set(canonical) & set(replay))
    }
    result.update(
        {
            "normalized_identical_count": sum(
                item["canonical"] == item["replay"]
                for item in normalized_hashes.values()
            ),
            "all_normalized_identical": result["names_match"]
            and all(
                item["canonical"] == item["replay"]
                for item in normalized_hashes.values()
            ),
            "normalized_hashes": normalized_hashes,
        }
    )
    return result


def command_audit() -> dict[str, Any]:
    canonical_summary_path = CANONICAL / "summary.json"
    replay_summary_path = REPLAY / "summary.json"
    canonical_predictions_path = CANONICAL / "prediction_rows.jsonl"
    replay_predictions_path = REPLAY / "prediction_rows.jsonl"
    canonical_summary = read_json(canonical_summary_path)
    replay_summary = read_json(replay_summary_path)
    canonical_normalized = normalize_output_paths(copy.deepcopy(canonical_summary))
    replay_normalized = normalize_output_paths(copy.deepcopy(replay_summary))
    normalized_summary_hashes = {
        "canonical": v9.sha256_json(canonical_normalized),
        "replay": v9.sha256_json(replay_normalized),
    }
    prediction_hashes = {
        "canonical": v9.sha256_file(canonical_predictions_path),
        "replay": v9.sha256_file(replay_predictions_path),
    }
    checkpoints = compare_file_sets(
        CANONICAL / "checkpoints",
        REPLAY / "checkpoints",
        "*.pt",
    )
    view_plans = compare_file_sets(
        CANONICAL / "view-plans",
        REPLAY / "view-plans",
        "*.json",
    )
    pairs = compare_json_sets_normalized(
        CANONICAL / "pairs",
        REPLAY / "pairs",
        "*.json",
    )
    runtime_match = (
        canonical_summary.get("runtime_checkpoint_sha256") == EXPECTED_RUNTIME
        and replay_summary.get("runtime_checkpoint_sha256") == EXPECTED_RUNTIME
        and canonical_summary.get("runtime_unchanged") is True
        and replay_summary.get("runtime_unchanged") is True
    )
    gates = {
        "prediction_rows_byte_identical": prediction_hashes["canonical"]
        == prediction_hashes["replay"],
        "normalized_scientific_summary_identical": normalized_summary_hashes[
            "canonical"
        ]
        == normalized_summary_hashes["replay"],
        "checkpoint_count_and_bytes_identical": checkpoints["file_count"] == 15
        and checkpoints["all_byte_identical"],
        "view_plan_count_and_bytes_identical": view_plans["file_count"] == 15
        and view_plans["all_byte_identical"],
        "pair_count_and_normalized_content_identical": pairs["file_count"] == 15
        and pairs["all_normalized_identical"],
        "decision_identical": canonical_summary.get("decision")
        == replay_summary.get("decision"),
        "claim_identical": canonical_summary.get("claim")
        == replay_summary.get("claim"),
        "runtime_unchanged_and_expected": runtime_match,
        "final_test_unread": canonical_summary.get("final_test_read") is False
        and replay_summary.get("final_test_read") is False,
        "promotion_forbidden": canonical_summary.get("promotion_eligible") is False
        and replay_summary.get("promotion_eligible") is False,
    }
    result = {
        "schema_version": "autoresearch-model-decision-audit-v17.multiview-replay-audit",
        "iteration": 2,
        "pass": all(gates.values()),
        "gates": gates,
        "canonical_summary_sha256": v9.sha256_file(canonical_summary_path),
        "replay_summary_sha256": v9.sha256_file(replay_summary_path),
        "normalized_scientific_summary_sha256": normalized_summary_hashes,
        "prediction_rows_sha256": prediction_hashes,
        "checkpoints": checkpoints,
        "view_plans": view_plans,
        "pairs": pairs,
        "decision": canonical_summary.get("decision"),
        "claim": canonical_summary.get("claim"),
        "runtime_unchanged": runtime_match,
        "final_test_read": False,
        "promotion": False,
    }
    if not result["pass"]:
        failures = [key for key, value in gates.items() if not value]
        raise ValueError(f"v17.2 independent replay mismatch: {failures}")
    v9.write_json(OUTPUT, result)
    print(v9.canonical_json(result), end="")
    return result


if __name__ == "__main__":
    command_audit()

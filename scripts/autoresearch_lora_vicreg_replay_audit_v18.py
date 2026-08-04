#!/usr/bin/env python3
"""Audit byte-identical canonical/replay evidence for v18.3 LoRA-VICReg."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import autoresearch_lora_vicreg_v18 as v18_3  # noqa: E402
import autoresearch_self_supervised_v9 as v9  # noqa: E402


DEFAULT_CANONICAL_DIR = v18_3.V18_RUN / "iteration-0003"
DEFAULT_REPLAY_DIR = v18_3.V18_RUN / "iteration-0003-replay"
DEFAULT_OUTPUT = v18_3.V18_RUN / "iteration-0003-replay-audit.json"
REQUIRED_CORE_FILES = (
    "summary.json",
    "evaluation.json",
    "prediction_rows.jsonl",
    "audit.json",
)
EXPECTED_STATE_COUNT = 15


def required_relative_paths(
    canonical_dir: Path, replay_dir: Path
) -> list[Path]:
    core = [Path(name) for name in REQUIRED_CORE_FILES]
    canonical_states = sorted(
        path.relative_to(canonical_dir)
        for path in (canonical_dir / "candidate_states").glob("*.bin")
    )
    replay_states = sorted(
        path.relative_to(replay_dir)
        for path in (replay_dir / "candidate_states").glob("*.bin")
    )
    if len(canonical_states) != EXPECTED_STATE_COUNT:
        raise ValueError(
            f"canonical candidate-state count mismatch: {len(canonical_states)}"
        )
    if canonical_states != replay_states:
        raise ValueError("canonical/replay candidate-state artifact sets differ")
    return core + canonical_states


def compare_required_artifacts(
    canonical_dir: Path,
    replay_dir: Path,
) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    for relative in required_relative_paths(canonical_dir, replay_dir):
        canonical = canonical_dir / relative
        replay = replay_dir / relative
        if not canonical.is_file() or not replay.is_file():
            raise ValueError(f"missing canonical/replay artifact: {relative}")
        canonical_sha = v9.sha256_file(canonical)
        replay_sha = v9.sha256_file(replay)
        comparisons[relative.as_posix()] = {
            "canonical_sha256": canonical_sha,
            "replay_sha256": replay_sha,
            "byte_identical": canonical.read_bytes() == replay.read_bytes(),
        }
    return comparisons


def audit_replay(canonical_dir: Path, replay_dir: Path) -> dict[str, Any]:
    comparisons = compare_required_artifacts(canonical_dir, replay_dir)
    canonical_summary = json.loads(
        (canonical_dir / "summary.json").read_text(encoding="utf-8")
    )
    replay_summary = json.loads(
        (replay_dir / "summary.json").read_text(encoding="utf-8")
    )
    actual_runner_sha = v9.sha256_file(Path(v18_3.__file__).resolve())
    runtime = v18_3.runtime_hashes(
        v18_3.DEFAULT_RUNTIME_PROJECTION,
        v18_3.DEFAULT_RUNTIME_PROTOTYPES,
        v18_3.DEFAULT_RUNTIME_CONFIG,
    )
    fixed_hash_fields = {
        "spec_sha256": v18_3.EXPECTED_SPEC_SHA256,
        "evaluator_sha256": v18_3.EXPECTED_EVALUATOR_SHA256,
        "cache_schema_sha256": v18_3.EXPECTED_CACHE_SCHEMA_SHA256,
        "feasibility_audit_sha256": v18_3.EXPECTED_FEASIBILITY_SHA256,
        "predecessor_audit_sha256": v18_3.EXPECTED_PREDECESSOR_SHA256,
        "source_inventory_sha256": v18_3.EXPECTED_SOURCE_INVENTORY_SHA256,
    }
    gates = {
        "all_required_artifacts_byte_identical": all(
            bool(item["byte_identical"]) for item in comparisons.values()
        ),
        "compared_candidate_state_artifact_count": sum(
            name.startswith("candidate_states/") for name in comparisons
        )
        == EXPECTED_STATE_COUNT,
        "canonical_runner_sha_matches_current": (
            canonical_summary.get("runner_sha256") == actual_runner_sha
        ),
        "replay_runner_sha_matches_current": (
            replay_summary.get("runner_sha256") == actual_runner_sha
        ),
        "canonical_fixed_contract_hashes_match": all(
            canonical_summary.get(key) == value
            for key, value in fixed_hash_fields.items()
        ),
        "replay_fixed_contract_hashes_match": all(
            replay_summary.get(key) == value
            for key, value in fixed_hash_fields.items()
        ),
        "train_token_manifest_hash_matches": (
            canonical_summary.get("train_token_manifest_sha256")
            == replay_summary.get("train_token_manifest_sha256")
            == v9.sha256_file(
                v18_3.DEFAULT_TOKEN_CACHE_DIR
                / "train-token-cache-manifest.json"
            )
        ),
        "final_token_manifest_hash_matches": (
            canonical_summary.get("final_token_manifest_sha256")
            == replay_summary.get("final_token_manifest_sha256")
            == v9.sha256_file(
                v18_3.DEFAULT_TOKEN_CACHE_DIR / "token-cache-manifest.json"
            )
        ),
        "normalization_contract_is_identity": (
            v18_3.REPLAY_NORMALIZATION_CONTRACT["summary_removed_keys"] == []
            and v18_3.REPLAY_NORMALIZATION_CONTRACT["pair_removed_keys"] == []
        ),
        "normalization_contract_sha_matches": (
            v18_3.replay_normalization_sha256()
            == v18_3.EXPECTED_REPLAY_NORMALIZATION_SHA256
        ),
        "canonical_integrity_pass": all(
            bool(value)
            for value in canonical_summary.get(
                "integrity_gate_passes", {}
            ).values()
        ),
        "replay_integrity_pass": all(
            bool(value)
            for value in replay_summary.get(
                "integrity_gate_passes", {}
            ).values()
        ),
        "canonical_engagement_pass": all(
            bool(value)
            for value in canonical_summary.get(
                "engagement_gate_passes", {}
            ).values()
        ),
        "replay_engagement_pass": all(
            bool(value)
            for value in replay_summary.get(
                "engagement_gate_passes", {}
            ).values()
        ),
        "runtime_hashes_match": runtime
        == v18_3.read_json(v18_3.DEFAULT_SPEC)["runtime_hashes"],
        "runtime_unchanged": (
            canonical_summary.get("runtime_unchanged") is True
            and replay_summary.get("runtime_unchanged") is True
        ),
        "final_test_unread": (
            canonical_summary.get("final_test_read") is False
            and replay_summary.get("final_test_read") is False
        ),
        "runtime_not_promoted": (
            canonical_summary.get("runtime_promotion") is False
            and replay_summary.get("runtime_promotion") is False
        ),
    }
    return {
        "schema_version": (
            "autoresearch-discriminative-readout-v18."
            "lora-vicreg-replay-audit-v1"
        ),
        "iteration": 3,
        "pass": all(gates.values()),
        "decision": canonical_summary.get("decision"),
        "claim": canonical_summary.get("claim"),
        "score": canonical_summary.get("score"),
        "normalization_contract": v18_3.REPLAY_NORMALIZATION_CONTRACT,
        "normalization_contract_sha256": (
            v18_3.replay_normalization_sha256()
        ),
        "comparisons": comparisons,
        "gates": gates,
        "runner_sha256": actual_runner_sha,
        "runtime_sha256": runtime,
        "same_machine_only": True,
        "cross_machine_determinism_claimed": False,
        "replay_reextracted_token_cache": False,
        "final_test_read": False,
        "runtime_promotion": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--canonical-dir", type=Path, default=DEFAULT_CANONICAL_DIR
    )
    parser.add_argument("--replay-dir", type=Path, default=DEFAULT_REPLAY_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = audit_replay(args.canonical_dir, args.replay_dir)
    if not result["pass"]:
        raise ValueError("v18.3 replay audit failed")
    v9.write_json(args.output, result)
    print(v9.canonical_json(result), end="")


if __name__ == "__main__":
    main()

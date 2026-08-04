#!/usr/bin/env python3
"""Build the under-60k Council packet for v17.3 shrinkage."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import autoresearch_hierarchical_shrinkage_v17 as v17  # noqa: E402
import autoresearch_self_supervised_v9 as v9  # noqa: E402


RUN = v17.V17_RUN
RESULTS = RUN / "iteration-0003"
REPLAY_RESULTS = RUN / "iteration-0003-replay"
REPLAY_AUDIT = RUN / "iteration-0003-replay-audit.json"
OUTPUT = RUN / "iteration-0003-council-result.json"
RUNNER = ROOT / "scripts/autoresearch_hierarchical_shrinkage_v17.py"
REPLAY_RUNNER = ROOT / "scripts/autoresearch_hierarchical_shrinkage_replay_audit_v17.py"
TESTS = ROOT / "backend/tests/test_autoresearch_hierarchical_shrinkage_v17.py"
REPLAY_TESTS = (
    ROOT / "backend/tests/test_autoresearch_hierarchical_shrinkage_replay_audit_v17.py"
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def compact_diagnostic(item: dict[str, Any]) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in item["prototype_rows"]:
        grouped[str(row["support_bin_three"])].append(
            float(row["control_candidate_cosine"])
        )
    support = {
        key: {
            "classes": len(values),
            "mean_cosine": sum(values) / len(values),
            "minimum_cosine": min(values),
        }
        for key, values in sorted(grouped.items())
    }
    return {
        "fold": item["fold"],
        "seed": item["seed"],
        "device": item["device"],
        "source_checkpoint_sha256": item["source_checkpoint_sha256"],
        "source_checkpoint_hash_matches": item["source_checkpoint_hash_matches"],
        "source_state_dict_sha256": item["source_state_dict_sha256"],
        "control_topk_match_count": item["control_topk_match_count"],
        "training_operation_count": item["training_operation_count"],
        "ssl_operation_count": item["ssl_operation_count"],
        "lambda": item["lambda"],
        "prototype_class_count": item["prototype_class_count"],
        "candidate_prototype_changed_class_count": item[
            "candidate_prototype_changed_class_count"
        ],
        "prototype_support_summary": support,
        "persisted_v9_B0_effective_rank": item["persisted_v9_B0_effective_rank"],
        "control_C1_effective_rank": item["control_C1_effective_rank"],
        "persisted_C1_effective_rank": item["persisted_C1_effective_rank"],
        "control_rank_absolute_delta_from_persisted_C1": item[
            "control_rank_absolute_delta_from_persisted_C1"
        ],
        "candidate_over_persisted_B0_effective_rank_ratio": item[
            "candidate_over_persisted_B0_effective_rank_ratio"
        ],
        "control_prototype_effective_rank": item[
            "control_prototype_effective_rank"
        ],
        "candidate_prototype_effective_rank": item[
            "candidate_prototype_effective_rank"
        ],
        "candidate_over_control_prototype_effective_rank_ratio": item[
            "candidate_over_control_prototype_effective_rank_ratio"
        ],
        "nan_or_nonfinite_detected": item["nan_or_nonfinite_detected"],
    }


def build_packet() -> dict[str, Any]:
    summary_path = RESULTS / "summary.json"
    evaluation_path = RESULTS / "evaluation.json"
    predictions_path = RESULTS / "prediction_rows.jsonl"
    audit_path = RESULTS / "audit.json"
    replay_summary_path = REPLAY_RESULTS / "summary.json"
    summary = read_json(summary_path)
    replay_audit = read_json(REPLAY_AUDIT)
    return {
        "schema_version": "autoresearch-model-decision-audit-v17.hierarchical-shrinkage-council-result",
        "iteration": 3,
        "decision": summary["decision"],
        "claim": summary["claim"],
        "failure_interpretation": summary["failure_interpretation"],
        "pass": summary["pass"],
        "hypothesis_supported": summary["hypothesis_supported"],
        "promotion_eligible": summary["promotion_eligible"],
        "comparison_vs_persisted_B0": summary["comparison_vs_persisted_B0"],
        "comparison_vs_exact_C1": summary["comparison_vs_exact_C1"],
        "integrity_gates": summary["integrity_gates"],
        "integrity_gate_passes": summary["integrity_gate_passes"],
        "engagement_gates": summary["engagement_gates"],
        "engagement_gate_passes": summary["engagement_gate_passes"],
        "b0_mission_anchor_gates": summary["b0_mission_anchor_gates"],
        "b0_mission_anchor_gate_passes": summary[
            "b0_mission_anchor_gate_passes"
        ],
        "c1_noninferiority_gates": summary["c1_noninferiority_gates"],
        "c1_noninferiority_gate_passes": summary[
            "c1_noninferiority_gate_passes"
        ],
        "support_diagnostics": summary["support_diagnostics"],
        "rank_diagnostic": summary["rank_diagnostic"],
        "folds": summary["folds"],
        "seeds": summary["seeds"],
        "folds_sha256": summary["folds_sha256"],
        "cache_sha256": summary["cache_sha256"],
        "runtime_checkpoint_sha256": summary["runtime_checkpoint_sha256"],
        "runtime_unchanged": summary["runtime_unchanged"],
        "final_test_read": summary["final_test_read"],
        "runtime_promotion": summary["runtime_promotion"],
        "diagnostics": [compact_diagnostic(item) for item in summary["diagnostics"]],
        "deterministic_replay": replay_audit,
        "artifact_sha256": {
            "spec": v9.sha256_file(v17.DEFAULT_SPEC),
            "evaluator": v9.sha256_file(v17.DEFAULT_EVALUATOR),
            "erratum": v9.sha256_file(v17.DEFAULT_ERRATUM),
            "runner": v9.sha256_file(RUNNER),
            "replay_runner": v9.sha256_file(REPLAY_RUNNER),
            "tests": v9.sha256_file(TESTS),
            "replay_tests": v9.sha256_file(REPLAY_TESTS),
            "summary": v9.sha256_file(summary_path),
            "evaluation": v9.sha256_file(evaluation_path),
            "prediction_rows": v9.sha256_file(predictions_path),
            "audit": v9.sha256_file(audit_path),
            "replay_summary": v9.sha256_file(replay_summary_path),
            "replay_audit": v9.sha256_file(REPLAY_AUDIT),
        },
    }


def main() -> None:
    packet = build_packet()
    v9.write_json(OUTPUT, packet)
    characters = len(OUTPUT.read_text(encoding="utf-8"))
    if characters > 60_000:
        raise ValueError(f"Council packet exceeds per-file limit: {characters}")
    print(
        v9.canonical_json(
            {
                "output": str(OUTPUT.relative_to(ROOT)),
                "sha256": v9.sha256_file(OUTPUT),
                "characters": characters,
            }
        ),
        end="",
    )


if __name__ == "__main__":
    main()

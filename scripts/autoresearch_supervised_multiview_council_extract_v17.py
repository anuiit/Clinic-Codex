#!/usr/bin/env python3
"""Build a compact, lossless-enough Council packet from the full v17.2 outputs."""

from __future__ import annotations

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
RESULTS = RUN / "iteration-0002"
REPLAY_RESULTS = RUN / "iteration-0002-replay"
REPLAY_AUDIT = RUN / "iteration-0002-replay-audit.json"
OUTPUT = RUN / "iteration-0002-council-result.json"
SPEC = RUN / "specs/iteration-0002.json"
EVALUATOR = RUN / "evaluator-iteration-0002.json"
RUNNER = ROOT / "scripts/autoresearch_supervised_multiview_v17.py"
TESTS = ROOT / "backend/tests/test_autoresearch_supervised_multiview_v17.py"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def compact_diagnostic(item: dict[str, Any]) -> dict[str, Any]:
    view_plan = dict(item["view_plan"])
    per_row = view_plan.pop("per_row_view_usage_counts")
    view_plan["per_row_view_usage_row_count"] = len(per_row)
    view_plan["per_row_view_usage_total"] = sum(
        sum(int(value) for value in counts) for counts in per_row.values()
    )
    return {
        "fold": item["fold"],
        "seed": item["seed"],
        "devices": item["devices"],
        "initial_state_sha256": item["initial_state_sha256"],
        "initial_state_matches_persisted_v9_C1": item[
            "initial_state_matches_persisted_v9_C1"
        ],
        "shared_pretrained_state_sha256": item["shared_pretrained_state_sha256"],
        "candidate_and_control_shared_pretrained_state": item[
            "candidate_and_control_shared_pretrained_state"
        ],
        "episode_plan_sha256": item["episode_plan_sha256"],
        "episode_plan_matches_persisted_v9_C1": item[
            "episode_plan_matches_persisted_v9_C1"
        ],
        "control_state_dict_sha256": item["control_state_dict_sha256"],
        "persisted_v9_C1_state_dict_sha256": item[
            "persisted_v9_C1_state_dict_sha256"
        ],
        "control_state_dict_matches_persisted_v9_C1": item[
            "control_state_dict_matches_persisted_v9_C1"
        ],
        "control_topk_match_count": item["control_topk_match_count"],
        "candidate_and_control_equal_supervised_budget": item[
            "candidate_and_control_equal_supervised_budget"
        ],
        "persisted_v9_B0_effective_rank": item[
            "persisted_v9_B0_effective_rank"
        ],
        "control_C1_effective_rank": item["control_C1_effective_rank"],
        "candidate_effective_rank": item["candidate_effective_rank"],
        "candidate_over_control_C1_effective_rank_ratio": item[
            "candidate_over_control_C1_effective_rank_ratio"
        ],
        "candidate_over_persisted_B0_effective_rank_ratio": item[
            "candidate_over_persisted_B0_effective_rank_ratio"
        ],
        "candidate_checkpoint_sha256": item["candidate_checkpoint_sha256"],
        "view_plan_audit_file_sha256": item["view_plan_audit_file_sha256"],
        "view_plan": view_plan,
        "ssl_plan_sha256": item["ssl"]["plan_sha256"],
        "ssl_plan_matches_persisted_v9_C1": item[
            "ssl_plan_matches_persisted_v9_C1"
        ],
        "candidate_training": item["candidate_training"],
        "candidate_episode_oof_row_exposure": item[
            "candidate_episode_oof_row_exposure"
        ],
        "candidate_episode_index_out_of_range": item[
            "candidate_episode_index_out_of_range"
        ],
        "nan_or_nonfinite_detected": item["nan_or_nonfinite_detected"],
    }


def main() -> None:
    summary_path = RESULTS / "summary.json"
    evaluation_path = RESULTS / "evaluation.json"
    predictions_path = RESULTS / "prediction_rows.jsonl"
    audit_path = RESULTS / "audit.json"
    replay_summary_path = REPLAY_RESULTS / "summary.json"
    summary = read_json(summary_path)
    replay_audit = read_json(REPLAY_AUDIT)
    compact = {
        "schema_version": "autoresearch-model-decision-audit-v17.multiview-council-result",
        "iteration": 2,
        "decision": summary["decision"],
        "claim": summary["claim"],
        "pass": summary["pass"],
        "hypothesis_supported": summary["hypothesis_supported"],
        "promotion_eligible": summary["promotion_eligible"],
        "comparison_vs_persisted_B0": summary["comparison_vs_persisted_B0"],
        "comparison_vs_exact_C1": summary["comparison_vs_exact_C1"],
        "integrity_gates": summary["integrity_gates"],
        "integrity_gate_passes": summary["integrity_gate_passes"],
        "engagement_and_stability_gates": summary[
            "engagement_and_stability_gates"
        ],
        "engagement_and_stability_gate_passes": summary[
            "engagement_and_stability_gate_passes"
        ],
        "b0_mission_anchor_gates": summary["b0_mission_anchor_gates"],
        "b0_mission_anchor_gate_passes": summary[
            "b0_mission_anchor_gate_passes"
        ],
        "c1_noninferiority_gates": summary["c1_noninferiority_gates"],
        "c1_noninferiority_gate_passes": summary[
            "c1_noninferiority_gate_passes"
        ],
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
        "deterministic_replay": {
            "pass": replay_audit["pass"],
            "gates": replay_audit["gates"],
            "canonical_summary_sha256": replay_audit[
                "canonical_summary_sha256"
            ],
            "replay_summary_sha256": replay_audit["replay_summary_sha256"],
            "normalized_scientific_summary_sha256": replay_audit[
                "normalized_scientific_summary_sha256"
            ],
            "prediction_rows_sha256": replay_audit["prediction_rows_sha256"],
            "checkpoint_file_count": replay_audit["checkpoints"]["file_count"],
            "checkpoint_byte_identical_count": replay_audit["checkpoints"][
                "byte_identical_count"
            ],
            "view_plan_file_count": replay_audit["view_plans"]["file_count"],
            "view_plan_byte_identical_count": replay_audit["view_plans"][
                "byte_identical_count"
            ],
            "pair_file_count": replay_audit["pairs"]["file_count"],
            "pair_normalized_identical_count": replay_audit["pairs"][
                "normalized_identical_count"
            ],
        },
        "artifact_sha256": {
            "spec": v9.sha256_file(SPEC),
            "evaluator": v9.sha256_file(EVALUATOR),
            "runner": v9.sha256_file(RUNNER),
            "tests": v9.sha256_file(TESTS),
            "summary": v9.sha256_file(summary_path),
            "evaluation": v9.sha256_file(evaluation_path),
            "prediction_rows": v9.sha256_file(predictions_path),
            "audit": v9.sha256_file(audit_path),
            "replay_summary": v9.sha256_file(replay_summary_path),
            "replay_audit": v9.sha256_file(REPLAY_AUDIT),
        },
    }
    v9.write_json(OUTPUT, compact)
    print(
        v9.canonical_json(
            {
                "output": str(OUTPUT.relative_to(ROOT)),
                "sha256": v9.sha256_file(OUTPUT),
                "characters": len(OUTPUT.read_text(encoding="utf-8")),
            }
        ),
        end="",
    )


if __name__ == "__main__":
    main()

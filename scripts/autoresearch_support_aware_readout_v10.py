#!/usr/bin/env python3
"""Compare exact v9 C1 control against a support-aware episodic readout."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import autoresearch_self_supervised_v9 as v9  # noqa: E402


V9_RUN = ROOT / ".omc/autoresearch/elements-baseline-replacement/runs/20260803-self-supervised-v9"
V10_RUN = ROOT / ".omc/autoresearch/elements-baseline-replacement/runs/20260803-self-supervised-v10"
DEFAULT_V9_SUMMARY = V9_RUN / "iteration-0001/results/paired_seed_summary.json"
DEFAULT_V9_PREDICTIONS = V9_RUN / "iteration-0001/results/prediction_rows.jsonl"
DEFAULT_SPEC = V10_RUN / "specs/iteration-0005.json"
DEFAULT_EVALUATOR = V10_RUN / "evaluator-iteration-0005.json"
DEFAULT_OUTPUT_DIR = V10_RUN / "iteration-0005/results"
DEFAULT_CACHE_DIR = V9_RUN / "iteration-0001/caches"
DEFAULT_RUNTIME_PROJECTION = ROOT / "backend/codex_model/weights/projection.pt"
DEFAULT_RUNTIME_PROTOTYPES = ROOT / "backend/codex_model/weights/prototypes.pt"

EXPECTED_V9_SUMMARY_SHA256 = "d6094bc9887748ba9bc89b3f6d691483b803be806cf0623e450450132387a22d"
EXPECTED_V9_PREDICTIONS_SHA256 = "37a15fb66ef883d15df5db8e1cadb152f9460876b60ace832c434375a3cdf34f"
EXPECTED_V9_RUNNER_SHA256 = "e03e1e29340a1a165ab2f82001236f5aaae4450afcee862dff8cb27ef26c48a9"
EXPECTED_V9_DATA_SHA256 = "45745373a1feac6731acb50ec87de699f2cf613314c6285279a1f551a893b4ac"
EXPECTED_SPEC_SHA256 = "b5d02e47fa25689554c19a192cf9c01bdb80dbc3768b7d47da163b5855e5ab0a"
EXPECTED_EVALUATOR_SHA256 = "3592d79fa7c4ed89eebabd1d506ae0ff4d4c4451cfa4d677719143f332e0dc90"
EXPECTED_CACHE_SHA256 = {
    1: "d7b2db5c0fe08d95e1b44cf8623eeb83abc48a3f00e9982af59d62a2d3744b39",
    2: "c9d08f6d6b22e16c9d66b33f646d83b301e74524b3b511d007a5b9743b0767c2",
    3: "79581b7e7cef86268315f962a7c958fa7444c61355e08e489b1f8d408d4b950e",
    4: "ae16f78ed4a7fdfb7c7de30867326a3429204f7c261adfa9b3d632de4a4409e1",
    5: "b3b8368f95251dae162aab73c399b0cf71226f0ac6bc879f40ea3d20a58894e8",
}
EXPECTED_RUNTIME_SHA256 = {
    "projection": "0ad6ce348eaf44c8844415a769b048b8f155017befa93f16254b1263b9241210",
    "prototypes": "ffc52f03696f03780a01302708b53e319dc24069187a6fd1a2927ea537b1cc54",
}
EXPECTED_FOLDS = [1, 2, 3, 4, 5]
EXPECTED_SEEDS = [17, 42, 73]
EXPECTED_PAIRED_ROWS = 1959
EXPECTED_UNIQUE_OOF_ROWS = 653
EXPECTED_ROWS = 9990
EXPECTED_CLASSES = 286
EXPECTED_COMPONENTS = 300


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    v9.write_json(path, value)


def validate_contract(spec_path: Path, evaluator_path: Path) -> dict[str, Any]:
    spec_sha256 = v9.sha256_file(spec_path)
    evaluator_sha256 = v9.sha256_file(evaluator_path)
    if spec_sha256 != EXPECTED_SPEC_SHA256:
        raise ValueError("iteration-5 spec SHA mismatch")
    if evaluator_sha256 != EXPECTED_EVALUATOR_SHA256:
        raise ValueError("iteration-5 evaluator SHA mismatch")
    spec = read_json(spec_path)
    evaluator = read_json(evaluator_path)
    if spec.get("iteration") != 5 or evaluator.get("iteration") != 5:
        raise ValueError("iteration-5 spec/evaluator mismatch")
    if spec.get("status") != "preregistered_before_any_iteration_0005_candidate_prediction":
        raise ValueError("iteration-5 spec was not preregistered")
    if spec.get("single_factor", {}).get("name") != "episodic_class_eligibility_and_scarce_class_sampling_rule":
        raise ValueError("iteration-5 single factor changed")
    coverage = spec.get("pretraining_coverage_diagnostic", {})
    if coverage.get("support_aware_eligible_classes_by_fold") != [230, 230, 230, 233, 231]:
        raise ValueError("iteration-5 support-aware coverage diagnostic changed")
    if coverage.get("newly_eligible_classes_by_fold") != [28, 28, 27, 30, 29]:
        raise ValueError("iteration-5 newly eligible coverage changed")
    if coverage.get("affected_oof_rows_by_fold") != [27, 22, 32, 225, 109]:
        raise ValueError("iteration-5 affected row diagnostic changed")
    expected_spec_gates = {
        "delta_top1_gte": 0.01,
        "positive_seed_count_gte": 2,
        "delta_top1_component_bootstrap_lower_95_gt": 0.0,
        "delta_macro_top1_gte": -0.005,
        "affected_oof_delta_top1_gt": 0.0,
        "candidate_over_control_effective_rank_ratio_gte": 0.9,
        "candidate_over_persisted_B0_effective_rank_ratio_gte": 0.9,
    }
    if spec.get("evaluation", {}).get("efficacy_gates") != expected_spec_gates:
        raise ValueError("iteration-5 spec efficacy gates changed")
    expected_evaluator_gates = {
        "delta_top1_gte": 0.01,
        "positive_seed_count_gte": 2,
        "delta_top1_component_bootstrap_lower_95_gt": 0.0,
        "delta_macro_top1_gte": -0.005,
        "affected_oof_delta_top1_gt": 0.0,
        "minimum_candidate_over_control_effective_rank_ratio_gte": 0.9,
        "minimum_candidate_over_persisted_B0_effective_rank_ratio_gte": 0.9,
    }
    if evaluator.get("efficacy_gates") != expected_evaluator_gates:
        raise ValueError("iteration-5 evaluator efficacy gates changed")
    pinned = spec.get("pinned_v9_inputs", {})
    if pinned.get("runner_sha256") != EXPECTED_V9_RUNNER_SHA256:
        raise ValueError("iteration-5 spec lost the v9 runner pin")
    if pinned.get("summary_sha256") != EXPECTED_V9_SUMMARY_SHA256:
        raise ValueError("iteration-5 spec lost the v9 summary pin")
    pinned_caches = {int(key): value for key, value in pinned.get("cache_sha256_by_fold", {}).items()}
    if pinned_caches != EXPECTED_CACHE_SHA256:
        raise ValueError("iteration-5 spec cache pins changed")
    return {"spec_sha256": spec_sha256, "evaluator_sha256": evaluator_sha256}


def validate_v9_runner() -> str:
    actual = v9.sha256_file(Path(v9.__file__).resolve())
    if actual != EXPECTED_V9_RUNNER_SHA256:
        raise ValueError("imported v9 runner SHA mismatch")
    return actual


def validate_inputs(
    v9_summary_path: Path,
    v9_predictions_path: Path,
) -> dict[str, Any]:
    if v9.sha256_file(v9_summary_path) != EXPECTED_V9_SUMMARY_SHA256:
        raise ValueError("persisted v9 summary SHA mismatch")
    if v9.sha256_file(v9_predictions_path) != EXPECTED_V9_PREDICTIONS_SHA256:
        raise ValueError("persisted v9 predictions SHA mismatch")
    summary = read_json(v9_summary_path)
    if summary.get("code_sha256") != EXPECTED_V9_RUNNER_SHA256:
        raise ValueError("v9 summary runner provenance mismatch")
    if summary.get("data_sha256") != EXPECTED_V9_DATA_SHA256:
        raise ValueError("v9 summary data provenance mismatch")
    if summary.get("paired_predictions_path") != v9_predictions_path.name:
        raise ValueError("v9 prediction filename mismatch")
    diagnostics = {(int(item["fold"]), int(item["seed"])): item for item in summary.get("diagnostics", [])}
    expected = {(fold, seed) for fold in EXPECTED_FOLDS for seed in EXPECTED_SEEDS}
    if set(diagnostics) != expected:
        raise ValueError("v9 diagnostics do not cover the exact fold/seed grid")
    records = read_jsonl(v9_predictions_path)
    if len(records) != EXPECTED_PAIRED_ROWS:
        raise ValueError("v9 persisted predictions row count mismatch")
    return {
        "summary": summary,
        "diagnostics": diagnostics,
        "records": records,
        "summary_sha256": EXPECTED_V9_SUMMARY_SHA256,
        "predictions_sha256": EXPECTED_V9_PREDICTIONS_SHA256,
    }


def build_historical_episode_plan(labels: torch.Tensor, **kwargs: Any) -> tuple[list[list[v9.Episode]], str]:
    return v9.build_episode_plan(labels, **kwargs)



def build_support_aware_episode_plan(
    labels: torch.Tensor,
    *,
    n_way: int,
    k_shot: int,
    q_queries: int,
    epochs: int,
    episodes_per_epoch: int,
    seed: int,
) -> tuple[list[list[v9.Episode]], str, dict[str, Any]]:
    class_indices: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels.tolist()):
        class_indices[int(label)].append(index)
    required = k_shot + q_queries
    eligible = sorted(label for label, indices in class_indices.items() if len(indices) >= 2)
    if len(eligible) < n_way:
        raise ValueError(f"only {len(eligible)} classes support support-aware {n_way}-way sampling")

    # This exact fast path proves the candidate is a no-op on a fully supported corpus.
    if all(len(indices) >= required for indices in class_indices.values()):
        plan, digest = v9.build_episode_plan(
            labels,
            n_way=n_way,
            k_shot=k_shot,
            q_queries=q_queries,
            epochs=epochs,
            episodes_per_epoch=episodes_per_epoch,
            seed=seed,
        )
        return plan, digest, {
            "eligible_classes": len(eligible),
            "historically_eligible_classes": len(eligible),
            "scarce_eligible_classes": 0,
            "scarce_class_slots": 0,
            "support_query_source_row_overlap": 0,
            "fixed_support_slots_per_class": k_shot,
            "fixed_query_slots_per_class": q_queries,
        }

    rng = random.Random(seed)
    plan: list[list[v9.Episode]] = []
    digest = hashlib.sha256()
    scarce_class_slots = 0
    support_query_source_row_overlap = 0
    for _ in range(epochs):
        epoch_plan: list[v9.Episode] = []
        for _ in range(episodes_per_epoch):
            classes = rng.sample(eligible, n_way)
            support_indices: list[int] = []
            query_indices: list[int] = []
            support_labels: list[int] = []
            query_labels: list[int] = []
            for local_label, label in enumerate(classes):
                indices = class_indices[label]
                if len(indices) >= required:
                    selected = rng.sample(indices, required)
                    support = selected[:k_shot]
                    query = selected[k_shot:]
                else:
                    scarce_class_slots += 1
                    shuffled = indices[:]
                    rng.shuffle(shuffled)
                    support_pool_size = min(k_shot, len(shuffled) - 1)
                    support_pool = shuffled[:support_pool_size]
                    query_pool = shuffled[support_pool_size:]
                    if not support_pool or not query_pool:
                        raise ValueError(f"scarce class {label} cannot form disjoint pools")
                    support = support_pool + [
                        rng.choice(support_pool) for _ in range(k_shot - len(support_pool))
                    ]
                    query = query_pool + [
                        rng.choice(query_pool) for _ in range(q_queries - len(query_pool))
                    ]
                support_query_source_row_overlap += len(set(support) & set(query))
                support_indices.extend(support)
                query_indices.extend(query)
                support_labels.extend([local_label] * k_shot)
                query_labels.extend([local_label] * q_queries)
            episode = (support_indices, support_labels, query_indices, query_labels)
            epoch_plan.append(episode)
            for values in episode:
                digest.update(np.asarray(values, dtype=np.int32).tobytes())
        plan.append(epoch_plan)
    return plan, digest.hexdigest(), {
        "eligible_classes": len(eligible),
        "historically_eligible_classes": sum(
            len(indices) >= required for indices in class_indices.values()
        ),
        "scarce_eligible_classes": sum(
            2 <= len(indices) < required for indices in class_indices.values()
        ),
        "scarce_class_slots": scarce_class_slots,
        "support_query_source_row_overlap": support_query_source_row_overlap,
        "fixed_support_slots_per_class": k_shot,
        "fixed_query_slots_per_class": q_queries,
    }


def audit_plan_train_only(
    plan: Sequence[Sequence[v9.Episode]],
    train_row_ids: Sequence[str],
    oof_row_ids: Sequence[str],
) -> dict[str, int]:
    oof_ids = {str(value) for value in oof_row_ids}
    exposure = 0
    invalid = 0
    for epoch in plan:
        for support, _support_labels, query, _query_labels in epoch:
            for index in (*support, *query):
                if index < 0 or index >= len(train_row_ids):
                    invalid += 1
                elif str(train_row_ids[index]) in oof_ids:
                    exposure += 1
    return {
        "candidate_episode_oof_row_exposure": exposure,
        "candidate_episode_index_out_of_range": invalid,
    }


def efficacy_gate_passes(values: dict[str, Any]) -> dict[str, bool]:
    return {
        "delta_top1_gte": float(values["delta_top1"]) >= 0.01,
        "positive_seed_count_gte": int(values["positive_seed_count"]) >= 2,
        "delta_top1_component_bootstrap_lower_95_gt": float(
            values["delta_top1_component_bootstrap_lower_95"]
        ) > 0.0,
        "delta_macro_top1_gte": float(values["delta_macro_top1"]) >= -0.005,
        "affected_oof_delta_top1_gt": float(values["affected_oof_delta_top1"]) > 0.0,
        "minimum_candidate_over_control_effective_rank_ratio_gte": float(
            values["minimum_candidate_over_control_effective_rank_ratio"]
        ) >= 0.9,
        "minimum_candidate_over_persisted_B0_effective_rank_ratio_gte": float(
            values["minimum_candidate_over_persisted_B0_effective_rank_ratio"]
        ) >= 0.9,
    }


def coverage_for_fold(cache: dict[str, Any]) -> dict[str, Any]:
    fold = int(cache["fold"])
    counts = Counter(int(label) for label in cache["train"]["class_label"])
    standard = {label for label, count in counts.items() if count >= 8}
    adaptive = {label for label, count in counts.items() if count >= 2}
    newly = adaptive - standard
    affected_rows = sum(
        int(label) in newly for label in cache["oof"]["class_label"]
    )
    return {
        "fold": fold,
        "standard_eligible_classes": len(standard),
        "support_aware_eligible_classes": len(adaptive),
        "newly_eligible_classes": len(newly),
        "affected_oof_rows": affected_rows,
        "train_support_counts": {str(label): count for label, count in sorted(counts.items())},
    }


def combine_coverage(reports: Sequence[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(reports, key=lambda item: int(item["fold"]))
    return {
        "standard_eligible_classes_by_fold": [
            int(item["standard_eligible_classes"]) for item in ordered
        ],
        "support_aware_eligible_classes_by_fold": [
            int(item["support_aware_eligible_classes"]) for item in ordered
        ],
        "newly_eligible_classes_by_fold": [
            int(item["newly_eligible_classes"]) for item in ordered
        ],
        "affected_oof_rows_by_fold": [
            int(item["affected_oof_rows"]) for item in ordered
        ],
        "affected_oof_rows_total": sum(int(item["affected_oof_rows"]) for item in ordered),
        "diagnostic_used_no_model_predictions": True,
    }


def _paired_metrics(records: Sequence[dict[str, Any]]) -> dict[str, float]:
    raw = v9.accuracy_metrics(records)
    return {
        "control_top1": raw["baseline_top1"],
        "control_top3": raw["baseline_top3"],
        "candidate_top1": raw["candidate_top1"],
        "candidate_top3": raw["candidate_top3"],
        "control_macro_top1": raw["baseline_macro_top1"],
        "candidate_macro_top1": raw["candidate_macro_top1"],
        "delta_top1": raw["delta_top1"],
        "delta_top3": raw["delta_top3"],
        "delta_macro_top1": raw["delta_macro_top1"],
    }


def _optional_metrics(records: Sequence[dict[str, Any]]) -> dict[str, float] | None:
    return _paired_metrics(records) if records else None


def stratified_metrics(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    strata = ("supported", "affected", "ineligible")
    return {
        "overall": _paired_metrics(records),
        "by_stratum": {
            stratum: _optional_metrics(
                [row for row in records if row["coverage_stratum"] == stratum]
            )
            for stratum in strata
        },
        "rows_by_stratum": {
            stratum: len([row for row in records if row["coverage_stratum"] == stratum])
            for stratum in strata
        },
        "unique_rows_by_stratum": {
            stratum: len(
                {
                    str(row["row_id"])
                    for row in records
                    if row["coverage_stratum"] == stratum
                }
            )
            for stratum in strata
        },
        "per_fold": {
            str(fold): {
                stratum: _optional_metrics(
                    [
                        row
                        for row in records
                        if int(row["outer_fold"]) == fold
                        and row["coverage_stratum"] == stratum
                    ]
                )
                for stratum in strata
            }
            for fold in EXPECTED_FOLDS
        },
    }


def per_class_metrics(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[int(row["label"])].append(row)
    return [
        {
            "label": label,
            "class_name": str(rows[0]["class_name"]),
            "train_support_counts": sorted(
                {int(row["train_support_count_for_label"]) for row in rows}
            ),
            "coverage_strata": sorted({str(row["coverage_stratum"]) for row in rows}),
            "unique_oof_rows": len({str(row["row_id"]) for row in rows}),
            **_paired_metrics(rows),
        }
        for label, rows in sorted(groups.items())
    ]


def _load_projection_state(path: Path) -> tuple[dict[str, torch.Tensor], str]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = v9.ProjectionHead(input_dim=384, embedding_dim=128)
    model.load_state_dict(payload["model_state_dict"])
    return payload["model_state_dict"], v9.state_dict_sha256(model)


def _save_candidate_checkpoint(
    path: Path,
    model: torch.nn.Module,
    *,
    fold: int,
    seed: int,
    runner_sha256: str,
    spec_sha256: str,
    evaluator_sha256: str,
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "autoresearch-self-supervised-v10.support-aware-checkpoint",
        "fold": fold,
        "seed": seed,
        "arm": "support-aware",
        "runner_sha256": runner_sha256,
        "spec_sha256": spec_sha256,
        "evaluator_sha256": evaluator_sha256,
        "model_state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return v9.sha256_file(path)



def load_validated_cache(path: Path, *, fold: int, views: int) -> tuple[dict[str, Any], dict[str, Any]]:
    actual_sha256 = v9.sha256_file(path)
    if actual_sha256 != EXPECTED_CACHE_SHA256[fold]:
        raise ValueError(f"v9 cache SHA mismatch for fold={fold}")
    cache = v9.load_fold_cache(
        path,
        expected_fold=fold,
        expected_views=views,
        expected_max_rows_per_class=None,
    )
    corpus = cache.get("provenance", {}).get("corpus_validation", {})
    if corpus.get("retained_rows") != EXPECTED_ROWS:
        raise ValueError(f"strict quarantined corpus row count mismatch for fold={fold}")
    if corpus.get("components_after_quarantine") != EXPECTED_COMPONENTS:
        raise ValueError(f"strict quarantined corpus component count mismatch for fold={fold}")
    validation = dict(cache["_cache_validation"])
    validation.update(
        {
            "fold": fold,
            "cache_sha256": actual_sha256,
            "expected_cache_sha256": EXPECTED_CACHE_SHA256[fold],
            "cache_sha256_matches": True,
            "strict_quarantined_corpus_retained_rows": int(corpus["retained_rows"]),
            "strict_quarantined_corpus_components": int(
                corpus["components_after_quarantine"]
            ),
        }
    )
    return cache, validation


def _resolve_source_checkpoint(source_diagnostic: dict[str, Any]) -> Path:
    raw = Path(str(source_diagnostic["checkpoints"]["candidate"]["path"]))
    return raw if raw.is_absolute() else ROOT / raw


def _load_resumable_pair(
    pair_path: Path,
    candidate_checkpoint_path: Path,
    *,
    fold: int,
    seed: int,
    expected_rows: int,
    runner_sha256: str,
    cache_sha256: str,
    contract: dict[str, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
    if not pair_path.is_file() or not candidate_checkpoint_path.is_file():
        return None
    try:
        payload = read_json(pair_path)
        expected_metadata = {
            "fold": fold,
            "seed": seed,
            "runner_sha256": runner_sha256,
            "cache_sha256": cache_sha256,
            "spec_sha256": contract["spec_sha256"],
            "evaluator_sha256": contract["evaluator_sha256"],
            "v9_summary_sha256": EXPECTED_V9_SUMMARY_SHA256,
            "v9_predictions_sha256": EXPECTED_V9_PREDICTIONS_SHA256,
        }
        if any(payload.get(key) != value for key, value in expected_metadata.items()):
            return None
        records = payload["records"]
        diagnostics = payload["diagnostics"]
        if len(records) != expected_rows:
            return None
        if diagnostics.get("candidate_checkpoint_sha256") != v9.sha256_file(
            candidate_checkpoint_path
        ):
            return None
        if not diagnostics.get("control_state_dict_matches_persisted_v9_C1"):
            return None
        if int(diagnostics.get("control_topk_match_count", -1)) != expected_rows:
            return None
        diagnostics = dict(diagnostics)
        diagnostics["resumed_from_pair_artifact"] = True
        return records, diagnostics
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _persist_pair(
    pair_path: Path,
    records: Sequence[dict[str, Any]],
    diagnostics: dict[str, Any],
    *,
    fold: int,
    seed: int,
    runner_sha256: str,
    cache_sha256: str,
    contract: dict[str, str],
) -> None:
    payload = {
        "schema_version": "autoresearch-self-supervised-v10.support-aware-pair",
        "fold": fold,
        "seed": seed,
        "runner_sha256": runner_sha256,
        "cache_sha256": cache_sha256,
        "spec_sha256": contract["spec_sha256"],
        "evaluator_sha256": contract["evaluator_sha256"],
        "v9_summary_sha256": EXPECTED_V9_SUMMARY_SHA256,
        "v9_predictions_sha256": EXPECTED_V9_PREDICTIONS_SHA256,
        "records": list(records),
        "diagnostics": diagnostics,
    }
    temporary = pair_path.with_suffix(pair_path.suffix + ".tmp")
    write_json(temporary, payload)
    temporary.replace(pair_path)


def run_fold_seed(
    cache: dict[str, Any],
    *,
    fold: int,
    seed: int,
    persisted: dict[str, Any],
    checkpoint_dir: Path,
    pair_dir: Path,
    device: torch.device,
    supervised_device: torch.device,
    ssl_epochs: int,
    ssl_batch_size: int,
    supervised_epochs: int,
    episodes_per_epoch: int,
    n_way: int,
    k_shot: int,
    q_queries: int,
    runner_sha256: str,
    cache_sha256: str,
    contract: dict[str, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_diagnostic = persisted["diagnostics"][(fold, seed)]
    source_rows = {
        str(row["row_id"]): row
        for row in persisted["records"]
        if int(row["outer_fold"]) == fold and int(row["seed"]) == seed
    }
    if len(source_rows) != len(cache["oof"]["row_id"]):
        raise ValueError(f"persisted v9 OOF row count mismatch for fold={fold}, seed={seed}")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    pair_dir.mkdir(parents=True, exist_ok=True)
    candidate_checkpoint_path = (
        checkpoint_dir / f"fold-{fold:02d}-seed-{seed}-support-aware.pt"
    )
    pair_path = pair_dir / f"fold-{fold:02d}-seed-{seed}.json"
    resumed = _load_resumable_pair(
        pair_path,
        candidate_checkpoint_path,
        fold=fold,
        seed=seed,
        expected_rows=len(source_rows),
        runner_sha256=runner_sha256,
        cache_sha256=cache_sha256,
        contract=contract,
    )
    if resumed is not None:
        return resumed

    # Mirror v9's model construction and device order exactly. The unused B0 copy
    # must remain alive through VICReg pretraining to preserve the v9 CUDA path.
    v9.configure_determinism(seed)
    initial_model = v9.ProjectionHead(input_dim=384, embedding_dim=128)
    unused_b0 = copy.deepcopy(initial_model).to(device)
    pretrained = copy.deepcopy(initial_model).to(device)
    initial_hash = v9.state_dict_sha256(initial_model)
    if initial_hash != str(source_diagnostic["initial_state_sha256"]):
        raise ValueError(f"initial state mismatch for fold={fold}, seed={seed}")
    if v9.state_dict_sha256(unused_b0) != initial_hash:
        raise AssertionError("B0 parity copy changed the initial state")
    ssl_diagnostics = v9.pretrain_vicreg(
        pretrained,
        cache["train"],
        epochs=ssl_epochs,
        batch_size=ssl_batch_size,
        learning_rate=3e-4,
        weight_decay=1e-4,
        seed=seed,
        device=device,
    )
    if ssl_diagnostics["plan_sha256"] != source_diagnostic["ssl"]["plan_sha256"]:
        raise ValueError(f"VICReg plan mismatch for fold={fold}, seed={seed}")

    control = copy.deepcopy(pretrained).to(supervised_device)
    candidate = copy.deepcopy(pretrained).to(supervised_device)
    shared_pretrained_hash = v9.state_dict_sha256(pretrained)
    if len(
        {
            shared_pretrained_hash,
            v9.state_dict_sha256(control),
            v9.state_dict_sha256(candidate),
        }
    ) != 1:
        raise AssertionError("control/candidate pretrained initialization mismatch")

    train_labels = torch.tensor(cache["train"]["class_label"], dtype=torch.long)
    episode_seed = v9.stable_seed("supervised-episodes", fold, seed)
    control_plan, control_plan_hash = v9.build_episode_plan(
        train_labels,
        n_way=n_way,
        k_shot=k_shot,
        q_queries=q_queries,
        epochs=supervised_epochs,
        episodes_per_epoch=episodes_per_epoch,
        seed=episode_seed,
    )
    candidate_plan, candidate_plan_hash, plan_diagnostics = (
        build_support_aware_episode_plan(
            train_labels,
            n_way=n_way,
            k_shot=k_shot,
            q_queries=q_queries,
            epochs=supervised_epochs,
            episodes_per_epoch=episodes_per_epoch,
            seed=episode_seed,
        )
    )
    if control_plan_hash != str(source_diagnostic["episode_plan_sha256"]):
        raise ValueError(f"control episode plan mismatch for fold={fold}, seed={seed}")
    plan_audit = audit_plan_train_only(
        candidate_plan,
        cache["train"]["row_id"],
        cache["oof"]["row_id"],
    )
    if any(plan_audit.values()):
        raise ValueError(f"candidate episode plan leaks or has invalid indices: {plan_audit}")
    if plan_diagnostics["support_query_source_row_overlap"] != 0:
        raise ValueError("candidate episode support/query source overlap")

    supervised_rng_seed = v9.stable_seed("supervised-rng", fold, seed)
    control_training = v9.train_supervised(
        control,
        cache["train"]["base_features"],
        control_plan,
        learning_rate=1e-3,
        weight_decay=1e-4,
        temperature=0.1,
        warmup_epochs=min(5, supervised_epochs),
        rng_seed=supervised_rng_seed,
        device=supervised_device,
    )
    candidate_training = v9.train_supervised(
        candidate,
        cache["train"]["base_features"],
        candidate_plan,
        learning_rate=1e-3,
        weight_decay=1e-4,
        temperature=0.1,
        warmup_epochs=min(5, supervised_epochs),
        rng_seed=supervised_rng_seed,
        device=supervised_device,
    )

    control = control.to(device)
    candidate = candidate.to(device)
    control_topk, control_rank = v9.predict_arm(
        control, cache["train"], cache["oof"], device=device
    )
    candidate_topk, candidate_rank = v9.predict_arm(
        candidate, cache["train"], cache["oof"], device=device
    )

    source_checkpoint_path = _resolve_source_checkpoint(source_diagnostic)
    expected_source_file_hash = str(
        source_diagnostic["checkpoints"]["candidate"]["sha256"]
    )
    if v9.sha256_file(source_checkpoint_path) != expected_source_file_hash:
        raise ValueError(f"persisted C1 checkpoint file SHA mismatch for fold={fold}, seed={seed}")
    _source_state, source_state_hash = _load_projection_state(source_checkpoint_path)
    control_state_hash = v9.state_dict_sha256(control)
    control_state_matches = control_state_hash == source_state_hash
    if not control_state_matches:
        raise ValueError(f"control state_dict mismatch for fold={fold}, seed={seed}")

    support_counts = Counter(int(label) for label in cache["train"]["class_label"])
    records: list[dict[str, Any]] = []
    control_topk_match_count = 0
    for index, row_id_value in enumerate(cache["oof"]["row_id"]):
        row_id = str(row_id_value)
        scaffold = source_rows.get(row_id)
        if scaffold is None:
            raise ValueError(f"OOF row absent from persisted v9 predictions: {row_id}")
        expected_metadata = (
            int(scaffold["label"]) == int(cache["oof"]["class_label"][index])
            and str(scaffold["provenance_component"])
            == str(cache["oof"]["component_id"][index])
            and str(scaffold["decoded_pixel_sha256"])
            == str(cache["oof"]["decoded_pixel_sha256"][index])
        )
        if not expected_metadata:
            raise ValueError(f"OOF metadata mismatch for {row_id}")
        expected_control_topk = [int(value) for value in scaffold["candidate_topk"]]
        actual_control_topk = [int(value) for value in control_topk[index]]
        if actual_control_topk != expected_control_topk:
            raise ValueError(f"control topk diverged from persisted v9 C1 for {row_id}")
        control_topk_match_count += 1
        label = int(cache["oof"]["class_label"][index])
        support_count = int(support_counts.get(label, 0))
        stratum = (
            "supported"
            if support_count >= 8
            else "affected"
            if support_count >= 2
            else "ineligible"
        )
        records.append(
            {
                "row_id": row_id,
                "provenance_component": str(cache["oof"]["component_id"][index]),
                "decoded_pixel_sha256": str(
                    cache["oof"]["decoded_pixel_sha256"][index]
                ),
                "label": label,
                "class_name": str(cache["oof"]["class_name"][index]),
                "outer_fold": fold,
                "seed": seed,
                "recipe": "exact-v9-C1-vs-support-aware-episodic-readout",
                "baseline_topk": actual_control_topk,
                "control_topk": actual_control_topk,
                "candidate_topk": [int(value) for value in candidate_topk[index]],
                "control_episode_plan_sha256": control_plan_hash,
                "candidate_episode_plan_sha256": candidate_plan_hash,
                "train_support_count_for_label": support_count,
                "coverage_stratum": stratum,
                "v9_control_checkpoint_sha256": expected_source_file_hash,
                "data_sha256": cache_sha256,
            }
        )

    candidate_checkpoint_sha256 = _save_candidate_checkpoint(
        candidate_checkpoint_path,
        candidate,
        fold=fold,
        seed=seed,
        runner_sha256=runner_sha256,
        spec_sha256=contract["spec_sha256"],
        evaluator_sha256=contract["evaluator_sha256"],
    )
    numeric_values = [
        float(control_rank),
        float(candidate_rank),
        float(control_training["last_loss"]),
        float(candidate_training["last_loss"]),
        float(ssl_diagnostics["last_loss"]),
    ]
    diagnostics = {
        "fold": fold,
        "seed": seed,
        "devices": {
            "ssl_and_prediction": str(device),
            "supervised": str(supervised_device),
        },
        "initial_state_sha256": initial_hash,
        "shared_pretrained_state_sha256": shared_pretrained_hash,
        "baseline_candidate_initial_state_equal": True,
        "control_episode_plan_sha256": control_plan_hash,
        "candidate_episode_plan_sha256": candidate_plan_hash,
        "control_episode_plan_matches_persisted_v9_C1": True,
        "control_state_dict_sha256": control_state_hash,
        "persisted_v9_C1_state_dict_sha256": source_state_hash,
        "control_state_dict_matches_persisted_v9_C1": control_state_matches,
        "control_topk_match_count": control_topk_match_count,
        "control_topk_matches_persisted_v9_C1": (
            control_topk_match_count == len(records)
        ),
        "baseline_candidate_supervised_budget_equal": (
            control_training["epochs"] == candidate_training["epochs"]
            and control_training["episodes_per_epoch"]
            == candidate_training["episodes_per_epoch"]
        ),
        "control_effective_rank": float(control_rank),
        "candidate_effective_rank": float(candidate_rank),
        "persisted_v9_B0_effective_rank": float(
            source_diagnostic["baseline_effective_rank"]
        ),
        "candidate_over_control_effective_rank_ratio": (
            float(candidate_rank / control_rank) if control_rank else 0.0
        ),
        "candidate_over_persisted_B0_effective_rank_ratio": (
            float(candidate_rank / float(source_diagnostic["baseline_effective_rank"]))
            if source_diagnostic["baseline_effective_rank"]
            else 0.0
        ),
        "candidate_checkpoint_path": str(candidate_checkpoint_path),
        "candidate_checkpoint_sha256": candidate_checkpoint_sha256,
        "candidate_checkpoint_exists": candidate_checkpoint_path.is_file(),
        "ssl": ssl_diagnostics,
        "control_training": control_training,
        "candidate_training": candidate_training,
        "episode_plan_diagnostics": plan_diagnostics,
        **plan_audit,
        "support_query_source_row_overlap": int(
            plan_diagnostics["support_query_source_row_overlap"]
        ),
        "fixed_support_slots_per_class": k_shot,
        "fixed_query_slots_per_class": q_queries,
        "nan_or_nonfinite_detected": not all(math.isfinite(value) for value in numeric_values),
        "resumed_from_pair_artifact": False,
    }
    _persist_pair(
        pair_path,
        records,
        diagnostics,
        fold=fold,
        seed=seed,
        runner_sha256=runner_sha256,
        cache_sha256=cache_sha256,
        contract=contract,
    )
    del unused_b0
    return records, diagnostics


def _fixed_value(values: Sequence[int], expected: int) -> int:
    return expected if values and all(value == expected for value in values) else -1


def _runtime_hashes(projection: Path, prototypes: Path) -> dict[str, str]:
    hashes = {
        "projection": v9.sha256_file(projection),
        "prototypes": v9.sha256_file(prototypes),
    }
    if hashes != EXPECTED_RUNTIME_SHA256:
        raise ValueError("runtime projection/prototype SHA mismatch before research run")
    return hashes


def aggregate_results(
    records: Sequence[dict[str, Any]],
    diagnostics: Sequence[dict[str, Any]],
    cache_validations: Sequence[dict[str, Any]],
    coverage_reports: Sequence[dict[str, Any]],
    *,
    runtime_unchanged: bool,
    bootstrap_replicates: int = 2000,
) -> dict[str, Any]:
    if not records or not diagnostics or not cache_validations or not coverage_reports:
        raise ValueError("evaluation requires predictions, diagnostics, validated caches, and coverage")
    seeds = sorted({int(row["seed"]) for row in records})
    folds = sorted({int(row["outer_fold"]) for row in records})
    seed_metrics = [
        {
            "seed": seed,
            **_paired_metrics([row for row in records if int(row["seed"]) == seed]),
        }
        for seed in seeds
    ]
    fold_metrics = [
        {
            "fold": fold,
            **_paired_metrics([row for row in records if int(row["outer_fold"]) == fold]),
        }
        for fold in folds
    ]
    overall = _paired_metrics(records)
    bootstrap = v9.paired_component_bootstrap(
        records,
        replicates=bootstrap_replicates,
        seed=20260803,
    )
    strata = stratified_metrics(records)
    affected = strata["by_stratum"]["affected"]
    if affected is None:
        raise ValueError("iteration-5 evaluation has no affected OOF rows")
    coverage = combine_coverage(coverage_reports)
    positive_seed_count = sum(item["delta_top1"] > 0.0 for item in seed_metrics)
    minimum_control_rank_ratio = min(
        float(item["candidate_over_control_effective_rank_ratio"])
        for item in diagnostics
    )
    minimum_b0_rank_ratio = min(
        float(item["candidate_over_persisted_B0_effective_rank_ratio"])
        for item in diagnostics
    )

    integrity_values: dict[str, Any] = {
        "paired_prediction_rows": len(records),
        "paired_seed_count": len(seeds),
        "outer_fold_count": len(folds),
        "candidate_checkpoint_count": sum(
            bool(item["candidate_checkpoint_exists"]) for item in diagnostics
        ),
        "control_state_dict_matches_persisted_v9_C1_count": sum(
            bool(item["control_state_dict_matches_persisted_v9_C1"])
            for item in diagnostics
        ),
        "control_topk_matches_persisted_v9_C1_rows": sum(
            int(item["control_topk_match_count"]) for item in diagnostics
        ),
        "strict_episode_plan_matches_persisted_v9_C1_count": sum(
            bool(item["control_episode_plan_matches_persisted_v9_C1"])
            for item in diagnostics
        ),
        "standard_eligible_classes_by_fold": coverage[
            "standard_eligible_classes_by_fold"
        ],
        "support_aware_eligible_classes_by_fold": coverage[
            "support_aware_eligible_classes_by_fold"
        ],
        "newly_eligible_classes_by_fold": coverage[
            "newly_eligible_classes_by_fold"
        ],
        "affected_oof_rows_by_fold": coverage["affected_oof_rows_by_fold"],
        "v9_runner_sha256": validate_v9_runner(),
        "v9_summary_sha256": EXPECTED_V9_SUMMARY_SHA256,
        "v9_cache_hashes_match_count": sum(
            bool(item["cache_sha256_matches"]) for item in cache_validations
        ),
        "strict_quarantined_corpus_retained_rows": _fixed_value(
            [
                int(item["strict_quarantined_corpus_retained_rows"])
                for item in cache_validations
            ],
            EXPECTED_ROWS,
        ),
        "strict_quarantined_corpus_components": _fixed_value(
            [
                int(item["strict_quarantined_corpus_components"])
                for item in cache_validations
            ],
            EXPECTED_COMPONENTS,
        ),
        "candidate_episode_oof_row_exposure": sum(
            int(item["candidate_episode_oof_row_exposure"])
            for item in diagnostics
        ),
        "candidate_episode_index_out_of_range": sum(
            int(item["candidate_episode_index_out_of_range"])
            for item in diagnostics
        ),
        "support_query_source_row_overlap": sum(
            int(item["support_query_source_row_overlap"]) for item in diagnostics
        ),
        "fixed_support_slots_per_class": _fixed_value(
            [int(item["fixed_support_slots_per_class"]) for item in diagnostics],
            3,
        ),
        "fixed_query_slots_per_class": _fixed_value(
            [int(item["fixed_query_slots_per_class"]) for item in diagnostics],
            5,
        ),
        "baseline_candidate_initial_state_equal": all(
            bool(item["baseline_candidate_initial_state_equal"])
            for item in diagnostics
        ),
        "baseline_candidate_supervised_budget_equal": all(
            bool(item["baseline_candidate_supervised_budget_equal"])
            for item in diagnostics
        ),
        "provenance_component_overlap_across_folds": max(
            int(item["provenance_component_overlap_across_folds"])
            for item in cache_validations
        ),
        "decoded_pixel_hash_overlap_across_folds": max(
            int(item["decoded_pixel_hash_overlap_across_folds"])
            for item in cache_validations
        ),
        "ssl_outer_fold_row_exposure": max(
            int(item["ssl_outer_fold_row_exposure"]) for item in cache_validations
        ),
        "learned_statistics_outer_fold_exposure": max(
            int(item["learned_statistics_outer_fold_exposure"])
            for item in cache_validations
        ),
        "nan_or_nonfinite_detected": any(
            bool(item["nan_or_nonfinite_detected"]) for item in diagnostics
        ),
        "runtime_unchanged": runtime_unchanged,
        "final_test_read": False,
        "automatic_promotion": False,
    }
    integrity_passes = {
        "paired_prediction_rows": integrity_values["paired_prediction_rows"]
        == EXPECTED_PAIRED_ROWS,
        "paired_seed_count": integrity_values["paired_seed_count"]
        == len(EXPECTED_SEEDS),
        "outer_fold_count": integrity_values["outer_fold_count"]
        == len(EXPECTED_FOLDS),
        "candidate_checkpoint_count": integrity_values["candidate_checkpoint_count"]
        == len(EXPECTED_FOLDS) * len(EXPECTED_SEEDS),
        "control_state_dict_matches_persisted_v9_C1_count": integrity_values[
            "control_state_dict_matches_persisted_v9_C1_count"
        ]
        == len(EXPECTED_FOLDS) * len(EXPECTED_SEEDS),
        "control_topk_matches_persisted_v9_C1_rows": integrity_values[
            "control_topk_matches_persisted_v9_C1_rows"
        ]
        == EXPECTED_PAIRED_ROWS,
        "strict_episode_plan_matches_persisted_v9_C1_count": integrity_values[
            "strict_episode_plan_matches_persisted_v9_C1_count"
        ]
        == len(EXPECTED_FOLDS) * len(EXPECTED_SEEDS),
        "standard_eligible_classes_by_fold": integrity_values[
            "standard_eligible_classes_by_fold"
        ]
        == [202, 202, 203, 203, 202],
        "support_aware_eligible_classes_by_fold": integrity_values[
            "support_aware_eligible_classes_by_fold"
        ]
        == [230, 230, 230, 233, 231],
        "newly_eligible_classes_by_fold": integrity_values[
            "newly_eligible_classes_by_fold"
        ]
        == [28, 28, 27, 30, 29],
        "affected_oof_rows_by_fold": integrity_values["affected_oof_rows_by_fold"]
        == [27, 22, 32, 225, 109],
        "v9_runner_sha256": integrity_values["v9_runner_sha256"]
        == EXPECTED_V9_RUNNER_SHA256,
        "v9_summary_sha256": integrity_values["v9_summary_sha256"]
        == EXPECTED_V9_SUMMARY_SHA256,
        "v9_cache_hashes_match_count": integrity_values[
            "v9_cache_hashes_match_count"
        ]
        == len(EXPECTED_FOLDS),
        "strict_quarantined_corpus_retained_rows": integrity_values[
            "strict_quarantined_corpus_retained_rows"
        ]
        == EXPECTED_ROWS,
        "strict_quarantined_corpus_components": integrity_values[
            "strict_quarantined_corpus_components"
        ]
        == EXPECTED_COMPONENTS,
        "candidate_episode_oof_row_exposure": integrity_values[
            "candidate_episode_oof_row_exposure"
        ]
        == 0,
        "candidate_episode_index_out_of_range": integrity_values[
            "candidate_episode_index_out_of_range"
        ]
        == 0,
        "support_query_source_row_overlap": integrity_values[
            "support_query_source_row_overlap"
        ]
        == 0,
        "fixed_support_slots_per_class": integrity_values[
            "fixed_support_slots_per_class"
        ]
        == 3,
        "fixed_query_slots_per_class": integrity_values[
            "fixed_query_slots_per_class"
        ]
        == 5,
        "baseline_candidate_initial_state_equal": integrity_values[
            "baseline_candidate_initial_state_equal"
        ]
        is True,
        "baseline_candidate_supervised_budget_equal": integrity_values[
            "baseline_candidate_supervised_budget_equal"
        ]
        is True,
        "provenance_component_overlap_across_folds": integrity_values[
            "provenance_component_overlap_across_folds"
        ]
        == 0,
        "decoded_pixel_hash_overlap_across_folds": integrity_values[
            "decoded_pixel_hash_overlap_across_folds"
        ]
        == 0,
        "ssl_outer_fold_row_exposure": integrity_values[
            "ssl_outer_fold_row_exposure"
        ]
        == 0,
        "learned_statistics_outer_fold_exposure": integrity_values[
            "learned_statistics_outer_fold_exposure"
        ]
        == 0,
        "nan_or_nonfinite_detected": integrity_values["nan_or_nonfinite_detected"]
        is False,
        "runtime_unchanged": integrity_values["runtime_unchanged"] is True,
        "final_test_read": integrity_values["final_test_read"] is False,
        "automatic_promotion": integrity_values["automatic_promotion"] is False,
    }

    efficacy_values = {
        "delta_top1": float(overall["delta_top1"]),
        "positive_seed_count": positive_seed_count,
        "delta_top1_component_bootstrap_lower_95": float(
            bootstrap["delta_top1_95"][0]
        ),
        "delta_macro_top1": float(overall["delta_macro_top1"]),
        "affected_oof_delta_top1": float(affected["delta_top1"]),
        "minimum_candidate_over_control_effective_rank_ratio": minimum_control_rank_ratio,
        "minimum_candidate_over_persisted_B0_effective_rank_ratio": minimum_b0_rank_ratio,
    }
    efficacy_passes = efficacy_gate_passes(efficacy_values)
    integrity_ok = all(integrity_passes.values())
    efficacy_ok = all(efficacy_passes.values())
    return {
        "pass": integrity_ok and efficacy_ok,
        "score": efficacy_values["delta_top1"],
        "hypothesis_supported": efficacy_ok,
        "decision": (
            "supported"
            if integrity_ok and efficacy_ok
            else "neutral_or_not_supported"
            if integrity_ok
            else "invalid"
        ),
        "promotion_eligible": False,
        "overall_metrics": overall,
        "seed_metrics": seed_metrics,
        "fold_metrics": fold_metrics,
        "bootstrap": bootstrap,
        "mcnemar": v9.exact_mcnemar(records),
        "stratified_metrics": strata,
        "coverage": coverage,
        "positive_seed_count": positive_seed_count,
        "minimum_candidate_over_control_effective_rank_ratio": minimum_control_rank_ratio,
        "minimum_candidate_over_persisted_B0_effective_rank_ratio": minimum_b0_rank_ratio,
        "integrity_gates": integrity_values,
        "integrity_gate_passes": integrity_passes,
        "efficacy_gates": efficacy_values,
        "efficacy_gate_passes": efficacy_passes,
        "final_test_read": False,
        "runtime_promotion": False,
    }


def command_run(args: argparse.Namespace) -> dict[str, Any]:
    contract = validate_contract(args.spec, args.evaluator)
    v9_runner_sha256 = validate_v9_runner()
    persisted = validate_inputs(args.v9_summary, args.v9_predictions)
    before_runtime = _runtime_hashes(
        args.runtime_projection,
        args.runtime_prototypes,
    )
    device = v9.resolve_device(args.device)
    supervised_device = v9.resolve_device(args.supervised_device)
    if supervised_device.type == "cpu":
        if args.supervised_cpu_threads < 1:
            raise ValueError("supervised CPU threads must be positive")
        torch.set_num_threads(args.supervised_cpu_threads)

    caches: dict[int, dict[str, Any]] = {}
    cache_validations: list[dict[str, Any]] = []
    coverage_reports: list[dict[str, Any]] = []
    for fold in EXPECTED_FOLDS:
        cache_path = v9.cache_path(args.cache_dir, fold, views=8, max_rows_per_class=None)
        cache, validation = load_validated_cache(cache_path, fold=fold, views=8)
        caches[fold] = cache
        cache_validations.append(validation)
        coverage_reports.append(coverage_for_fold(cache))
    coverage = combine_coverage(coverage_reports)
    expected_coverage = {
        "standard_eligible_classes_by_fold": [202, 202, 203, 203, 202],
        "support_aware_eligible_classes_by_fold": [230, 230, 230, 233, 231],
        "newly_eligible_classes_by_fold": [28, 28, 27, 30, 29],
        "affected_oof_rows_by_fold": [27, 22, 32, 225, 109],
        "affected_oof_rows_total": 415,
        "diagnostic_used_no_model_predictions": True,
    }
    if coverage != expected_coverage:
        raise ValueError("fold-local support coverage differs from preregistration")

    runner_sha256 = v9.sha256_file(Path(__file__).resolve())
    all_records: list[dict[str, Any]] = []
    all_diagnostics: list[dict[str, Any]] = []
    for fold in EXPECTED_FOLDS:
        cache = caches[fold]
        cache_sha256 = str(cache["_cache_validation"]["cache_sha256"])
        for seed in EXPECTED_SEEDS:
            records, diagnostics = run_fold_seed(
                cache,
                fold=fold,
                seed=seed,
                persisted=persisted,
                checkpoint_dir=args.output_dir / "checkpoints",
                pair_dir=args.output_dir / "pairs",
                device=device,
                supervised_device=supervised_device,
                ssl_epochs=30,
                ssl_batch_size=256,
                supervised_epochs=30,
                episodes_per_epoch=100,
                n_way=20,
                k_shot=3,
                q_queries=5,
                runner_sha256=runner_sha256,
                cache_sha256=cache_sha256,
                contract=contract,
            )
            record_provenance = {
                "runner_sha256": runner_sha256,
                "v9_runner_sha256": v9_runner_sha256,
                "v9_summary_sha256": persisted["summary_sha256"],
                "v9_predictions_sha256": persisted["predictions_sha256"],
                "spec_sha256": contract["spec_sha256"],
                "evaluator_sha256": contract["evaluator_sha256"],
                "runtime_projection_sha256": before_runtime["projection"],
                "runtime_prototypes_sha256": before_runtime["prototypes"],
                "candidate_checkpoint_sha256": diagnostics[
                    "candidate_checkpoint_sha256"
                ],
                "ssl_device": str(device),
                "supervised_device": str(supervised_device),
            }
            for record in records:
                record.update(record_provenance)
            all_records.extend(records)
            all_diagnostics.append(diagnostics)

    after_runtime = _runtime_hashes(
        args.runtime_projection,
        args.runtime_prototypes,
    )
    runtime_unchanged = before_runtime == after_runtime
    if not runtime_unchanged:
        raise RuntimeError("runtime projection/prototype artifacts changed during research run")

    fold_assignments = sorted(
        {
            (
                str(record["row_id"]),
                int(record["outer_fold"]),
                str(record["provenance_component"]),
                str(record["decoded_pixel_sha256"]),
            )
            for record in all_records
        }
    )
    folds_sha256 = v9.sha256_json(fold_assignments)
    for record in all_records:
        record["folds_sha256"] = folds_sha256

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "prediction_rows.jsonl"
    v9.write_jsonl(predictions_path, all_records)
    result = aggregate_results(
        all_records,
        all_diagnostics,
        cache_validations,
        coverage_reports,
        runtime_unchanged=runtime_unchanged,
        bootstrap_replicates=2000,
    )
    result.update(
        {
            "schema_version": "autoresearch-self-supervised-v10.evaluation",
            "iteration": 5,
            "name": "support-aware-episodic-readout-on-vicreg-s14",
            "paired_predictions_path": predictions_path.name,
            "folds": EXPECTED_FOLDS,
            "seeds": EXPECTED_SEEDS,
            "folds_sha256": folds_sha256,
            "cache_sha256": {
                str(item["fold"]): item["cache_sha256"]
                for item in cache_validations
            },
            "runner_sha256": runner_sha256,
            "v9_runner_sha256": v9_runner_sha256,
            "v9_summary_sha256": persisted["summary_sha256"],
            "v9_predictions_sha256": persisted["predictions_sha256"],
            "spec_sha256": contract["spec_sha256"],
            "evaluator_sha256": contract["evaluator_sha256"],
            "runtime_checkpoint_sha256": before_runtime,
            "runtime_unchanged": runtime_unchanged,
            "cache_validations": cache_validations,
            "diagnostics": all_diagnostics,
        }
    )
    write_json(args.output_dir / "summary.json", result)
    write_json(args.output_dir / "evaluation.json", result)
    write_json(
        args.output_dir / "per_arm_metrics.json",
        {
            "overall": result["overall_metrics"],
            "seeds": result["seed_metrics"],
            "folds": result["fold_metrics"],
            "strata": result["stratified_metrics"],
        },
    )
    write_json(
        args.output_dir / "per_class_metrics.json",
        per_class_metrics(all_records),
    )
    print(v9.canonical_json(result), end="")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser(
        "run",
        help="execute the preregistered exact-C1 versus support-aware comparison",
    )
    run.add_argument("--v9-summary", type=Path, default=DEFAULT_V9_SUMMARY)
    run.add_argument("--v9-predictions", type=Path, default=DEFAULT_V9_PREDICTIONS)
    run.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    run.add_argument("--evaluator", type=Path, default=DEFAULT_EVALUATOR)
    run.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    run.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    run.add_argument("--device", default="auto")
    run.add_argument("--supervised-device", default="cpu")
    run.add_argument("--supervised-cpu-threads", type=int, default=1)
    run.add_argument(
        "--runtime-projection",
        type=Path,
        default=DEFAULT_RUNTIME_PROJECTION,
    )
    run.add_argument(
        "--runtime-prototypes",
        type=Path,
        default=DEFAULT_RUNTIME_PROTOTYPES,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "run":
        command_run(args)
    else:  # pragma: no cover
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()

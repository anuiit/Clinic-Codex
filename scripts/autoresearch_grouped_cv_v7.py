#!/usr/bin/env python3
"""Grouped-CV autoresearch for Elements without any final test read.

This runner reuses the historical white runtime as baseline, filters external
rows that overlap legacy source groups, deduplicates by source pixel hash, and
evaluates ten deterministic prototype-only recipes under source-group-disjoint
cross-validation. No sealed test or historical test artifact is read.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(BACKEND_ROOT))

import autoresearch_elements_model as elements  # noqa: E402
import autoresearch_external_model_v5 as v5  # noqa: E402
from codex_pipeline.determinism import configure_determinism  # noqa: E402

CLASS_COUNT = v5.CLASS_COUNT
EMBEDDING_DIM = v5.EMBEDDING_DIM
DEFAULT_LEGACY_CACHE = Path("/tmp/baseline-replacement-e2e/perclass-split/features_train_augmented_safe.pt")
DEFAULT_LEGACY_MANIFEST = (
    BACKEND_ROOT / "training_corpus/frozen/legacy-elements-v1/legacy_elements_manifest.json"
)
DEFAULT_EXTERNAL_CACHE = (
    BACKEND_ROOT
    / "model_registry/versions/20260711T220000Z-external286-weak-v3"
    / "training_data/precomputed/features.pt"
)
DEFAULT_IMPORT_SNAPSHOT = (
    BACKEND_ROOT
    / "training_corpus/external/20260711-approved-external-corpus-v1"
    / "import_snapshot.json"
)
DEFAULT_RUNTIME_WEIGHTS = BACKEND_ROOT / "codex_model" / "weights"
DEFAULT_RUNTIME_CONFIG = BACKEND_ROOT / "codex_model" / "config.json"
DEFAULT_RUN_DIR = (
    REPO_ROOT
    / ".omc/autoresearch/elements-baseline-replacement/runs"
    / "20260729-grouped-cv-v7"
)
DEFAULT_MAX_RUNTIME_SECONDS = 6 * 60 * 60
SCHEMA_VERSION = "autoresearch-grouped-cv-v7"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def dump_json(path: Path, value: Any) -> None:
    v5.dump_json(path, value)


def resolved(path: str | Path) -> str:
    source = PurePosixPath(str(path).replace("\\", "/"))
    if not source.is_absolute() and not PureWindowsPath(str(path)).is_absolute():
        source = PurePosixPath(REPO_ROOT.as_posix()) / source
    return source.as_posix()


def canonical_source_group(path: str | Path) -> str:
    source = PurePosixPath(str(path).replace("\\", "/"))
    return re.sub(r"-\d+$", "", source.stem).casefold()


def elements_relative_path(path: str | Path) -> str:
    canonical = str(path).replace("\\", "/")
    marker = "/Elements/"
    if marker not in canonical:
        raise ValueError(f"path is not inside an Elements tree: {path}")
    return "Elements/" + canonical.split(marker, 1)[1].lstrip("/")


def load_legacy_digest_index(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    images = payload.get("images") if isinstance(payload, dict) else None
    if not isinstance(images, list):
        raise ValueError("legacy manifest must expose an images list")
    index: dict[str, str] = {}
    for row in images:
        if not isinstance(row, dict):
            raise ValueError("legacy manifest image row must be an object")
        output_path = row.get("output_path")
        output_sha256 = row.get("output_sha256")
        if not isinstance(output_path, str) or not isinstance(output_sha256, str):
            raise ValueError("legacy manifest image row lacks output_path/output_sha256")
        relative_path = elements_relative_path(output_path)
        previous = index.setdefault(relative_path, output_sha256)
        if previous != output_sha256:
            raise ValueError(f"conflicting digest for {relative_path}")
    if not index:
        raise ValueError("legacy manifest digest index is empty")
    return index


def load_cache(path: Path, *, cache_name: str) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {"features", "labels", "image_paths", "class_names"}
    if not isinstance(payload, dict) or not required <= payload.keys():
        raise ValueError(f"{cache_name} must expose {sorted(required)}")
    if len(payload["features"]) != len(payload["labels"]):
        raise ValueError(f"{cache_name} has mismatched features and labels")
    if len(payload["image_paths"]) != len(payload["labels"]):
        raise ValueError(f"{cache_name} has mismatched image_paths")
    digests = payload.get("source_pixel_sha256")
    if digests is not None and len(digests) != len(payload["labels"]):
        raise ValueError(f"{cache_name} has mismatched source_pixel_sha256")
    return payload


def load_snapshot_rows_by_output(snapshot: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = snapshot.get("rows")
    if not isinstance(rows, list):
        raise ValueError("import snapshot rows must be a list")
    by_output: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("import snapshot row must be an object")
        output_path = row.get("output_path")
        source_path = row.get("source_path")
        if not isinstance(output_path, str) or not isinstance(source_path, str):
            raise ValueError("import snapshot row lacks output_path/source_path")
        key = resolved(output_path)
        if key in by_output:
            raise ValueError(f"duplicate import snapshot output_path: {output_path}")
        by_output[key] = row
    return by_output


def cache_rows(
    cache: dict[str, Any],
    *,
    cache_name: str,
    snapshot_rows_by_output: dict[str, dict[str, Any]] | None = None,
    source_digest_by_relative_path: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    names = v5.class_names(cache)
    rows: list[dict[str, Any]] = []
    cache_digests = cache.get("source_pixel_sha256")
    file_digest_cache: dict[str, str] = {}
    for index, path in enumerate(cache["image_paths"]):
        output_path = resolved(path)
        class_label = int(cache["labels"][index])
        class_name = names[class_label]
        source_path = output_path
        digest = str(cache_digests[index]) if cache_digests is not None else None
        if snapshot_rows_by_output is not None:
            snapshot_row = snapshot_rows_by_output.get(output_path)
            if snapshot_row is None:
                raise ValueError(f"{cache_name} row missing snapshot provenance: {output_path}")
            source_path = resolved(snapshot_row["source_path"])
            snapshot_digest = snapshot_row.get("source_pixel_sha256")
            if not isinstance(snapshot_digest, str) or not snapshot_digest:
                raise ValueError(f"{cache_name} snapshot lacks source_pixel_sha256: {output_path}")
            if digest is not None and snapshot_digest != digest:
                raise ValueError(f"{cache_name} source_pixel_sha256 mismatch: {output_path}")
            digest = snapshot_digest
        elif digest is None and source_digest_by_relative_path is not None:
            relative_path = elements_relative_path(source_path)
            digest = source_digest_by_relative_path.get(relative_path)
            if digest is None:
                raise ValueError(f"{cache_name} row missing legacy manifest digest: {relative_path}")
        elif digest is None:
            source_file = Path(source_path)
            if not source_file.is_file():
                raise FileNotFoundError(f"{cache_name} source image is unavailable: {source_path}")
            if source_path not in file_digest_cache:
                file_digest_cache[source_path] = v5.sha256_file(source_file)
            digest = file_digest_cache[source_path]
        rows.append(
            {
                "cache_name": cache_name,
                "cache_index": index,
                "output_path": output_path,
                "source_path": source_path,
                "source_group": canonical_source_group(source_path),
                "source_pixel_sha256": str(digest),
                "class_label": class_label,
                "class_name": class_name,
            }
        )
    return rows

def dedupe_by_source_pixel_sha256(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    kept: list[dict[str, Any]] = []
    seen: set[str] = set()
    removed = 0
    for row in rows:
        digest = str(row["source_pixel_sha256"])
        if digest in seen:
            removed += 1
            continue
        seen.add(digest)
        kept.append(row)
    return kept, removed


def build_admissible_rows(
    legacy_cache: dict[str, Any],
    external_cache: dict[str, Any],
    snapshot: dict[str, Any],
    legacy_digest_index: dict[str, str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    snapshot_rows_by_output = load_snapshot_rows_by_output(snapshot)
    legacy_rows = cache_rows(
        legacy_cache,
        cache_name="legacy cache",
        source_digest_by_relative_path=legacy_digest_index,
    )
    external_rows = cache_rows(
        external_cache,
        cache_name="external cache",
        snapshot_rows_by_output=snapshot_rows_by_output,
    )
    legacy_groups = {row["source_group"] for row in legacy_rows}
    external_kept: list[dict[str, Any]] = []
    overlap_removed = 0
    for row in external_rows:
        if row["source_group"] in legacy_groups:
            overlap_removed += 1
            continue
        external_kept.append(row)
    ordered_rows = legacy_rows + external_kept
    deduped_rows, removed_by_digest = dedupe_by_source_pixel_sha256(ordered_rows)
    if not deduped_rows:
        raise ValueError("admissible grouped-CV rows are empty after filtering and deduplication")
    audit = {
        "legacy_row_count": len(legacy_rows),
        "external_row_count": len(external_rows),
        "external_kept_count": len(external_kept),
        "external_overlap_removed": overlap_removed,
        "legacy_group_count": len(legacy_groups),
        "deduplicated_by_source_pixel_sha256": removed_by_digest,
        "admissible_row_count": len(deduped_rows),
    }
    return deduped_rows, audit


def load_runtime(
    weights_dir: Path,
    runtime_config_path: Path,
    device: torch.device,
) -> v5.Runtime:
    return v5.load_runtime(weights_dir, runtime_config_path, device)


def row_feature(row: dict[str, Any], legacy_cache: dict[str, Any], external_cache: dict[str, Any]) -> torch.Tensor:
    cache = legacy_cache if row["cache_name"] == "legacy cache" else external_cache
    return cache["features"][row["cache_index"]]


def build_group_folds(rows: list[dict[str, Any]], folds: int) -> list[dict[str, Any]]:
    if folds < 2:
        raise ValueError("folds must be at least 2")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["source_group"]].append(row)
    if len(groups) < folds:
        raise ValueError("number of unique source groups must be >= folds")
    ordered_groups = sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))
    global_counts = Counter(row["class_label"] for row in rows)
    targets = {label: count / folds for label, count in global_counts.items()}
    buckets = [{"rows": [], "groups": [], "counts": Counter()} for _ in range(folds)]
    for group_name, group_rows in ordered_groups:
        group_counts = Counter(row["class_label"] for row in group_rows)
        group_size = len(group_rows)
        best_index = min(
            range(folds),
            key=lambda idx: (
                sum(abs((buckets[idx]["counts"][label] + count) - targets[label]) for label, count in group_counts.items()),
                len(buckets[idx]["rows"]) + group_size,
                idx,
            ),
        )
        buckets[best_index]["groups"].append(group_name)
        buckets[best_index]["rows"].extend(group_rows)
        buckets[best_index]["counts"].update(group_counts)
    if any(not bucket["rows"] for bucket in buckets):
        raise ValueError("group-fold assignment produced an empty fold")
    return [
        {
            "fold": index + 1,
            "source_groups": bucket["groups"],
            "row_indices": [row["admissible_index"] for row in bucket["rows"]],
            "class_counts": dict(sorted(bucket["counts"].items())),
        }
        for index, bucket in enumerate(buckets)
    ]


def enforce_fold_train_support(
    rows: list[dict[str, Any]],
    folds_payload: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    row_lookup = {row["admissible_index"]: row for row in rows}
    total_class_counts = Counter(row["class_label"] for row in rows)
    supported_folds: list[dict[str, Any]] = []
    excluded_indices: set[int] = set()
    for fold in folds_payload:
        holdout_indices = list(fold["row_indices"])
        holdout_class_counts = Counter(row_lookup[index]["class_label"] for index in holdout_indices)
        supported_indices = [
            index
            for index in holdout_indices
            if total_class_counts[row_lookup[index]["class_label"]]
            > holdout_class_counts[row_lookup[index]["class_label"]]
        ]
        excluded_indices.update(set(holdout_indices) - set(supported_indices))
        if not supported_indices:
            raise ValueError(f"fold {fold['fold']} has no validation rows with independent train support")
        supported_class_counts = Counter(row_lookup[index]["class_label"] for index in supported_indices)
        supported_folds.append(
            {
                **fold,
                "holdout_row_indices": holdout_indices,
                "row_indices": supported_indices,
                "class_counts": dict(sorted(supported_class_counts.items())),
            }
        )
    excluded_classes = sorted({row_lookup[index]["class_label"] for index in excluded_indices})
    return supported_folds, {
        "oof_row_count": sum(len(fold["row_indices"]) for fold in supported_folds),
        "excluded_oof_row_count": len(excluded_indices),
        "excluded_oof_class_count": len(excluded_classes),
        "excluded_oof_class_labels": excluded_classes,
    }

def build_specs(quick: bool = False) -> list[dict[str, Any]]:
    base = {
        "objective": "grouped_cv_prototype_ce",
        "projection": "historical_white_frozen",
        "batch_size": 1024,
        "epochs": 96,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "temperature": 0.08,
        "label_smoothing": 0.0,
        "noise_std": 0.0,
        "anchor_weight": 0.0,
    }
    grid = [
        {"iteration": 1, "name": "runtime-refit-train-means-baseline", "track": "runtime_refit", "initialization": "train_means_with_runtime_fallback", "seed": 7101, "modification": "train_means"},
        {"iteration": 2, "name": "runtime-refit-lr-075", "track": "runtime_refit", "initialization": "train_means_with_runtime_fallback", "seed": 7102, "modification": "lr", "lr": 7.5e-4},
        {"iteration": 3, "name": "runtime-refit-lr-125", "track": "runtime_refit", "initialization": "train_means_with_runtime_fallback", "seed": 7103, "modification": "lr", "lr": 1.25e-3},
        {"iteration": 4, "name": "runtime-refit-label-smoothing", "track": "runtime_refit", "initialization": "train_means_with_runtime_fallback", "seed": 7104, "modification": "label_smoothing", "label_smoothing": 0.02},
        {"iteration": 5, "name": "runtime-refit-temperature", "track": "runtime_refit", "initialization": "train_means_with_runtime_fallback", "seed": 7105, "modification": "temperature", "temperature": 0.06},
        {"iteration": 6, "name": "warm-start-anchor-025", "track": "warm_start", "initialization": "historical_runtime", "seed": 7106, "modification": "anchor_weight", "anchor_weight": 0.25},
        {"iteration": 7, "name": "warm-start-anchor-050", "track": "warm_start", "initialization": "historical_runtime", "seed": 7107, "modification": "anchor_weight", "anchor_weight": 0.50},
        {"iteration": 8, "name": "warm-start-anchor-100", "track": "warm_start", "initialization": "historical_runtime", "seed": 7108, "modification": "anchor_weight", "anchor_weight": 1.00},
        {"iteration": 9, "name": "warm-start-noise-001", "track": "warm_start", "initialization": "historical_runtime", "seed": 7109, "modification": "noise_std", "noise_std": 0.01},
        {"iteration": 10, "name": "warm-start-epochs-128", "track": "warm_start", "initialization": "historical_runtime", "seed": 7110, "modification": "epochs", "epochs": 128},
    ]
    if quick:
        for spec in grid:
            spec["epochs"] = max(16, int(spec.get("epochs", base["epochs"]) // 2))
            spec["batch_size"] = 256
    return [{**base, **item} for item in grid]


def initialize_prototypes(
    train_embeddings: torch.Tensor,
    train_labels: torch.Tensor,
    baseline_prototypes: torch.Tensor,
    spec: dict[str, Any],
) -> torch.Tensor:
    baseline = F.normalize(baseline_prototypes.float(), dim=1)
    init = str(spec["initialization"])
    if init == "historical_runtime":
        return baseline.cpu()
    if init != "train_means_with_runtime_fallback":
        raise ValueError(f"unsupported initialization: {init!r}")
    counts = torch.bincount(train_labels.long(), minlength=CLASS_COUNT).float()
    sums = torch.zeros(CLASS_COUNT, EMBEDDING_DIM, dtype=torch.float32)
    sums.index_add_(0, train_labels.long(), train_embeddings.float())
    prototypes = baseline.clone()
    present = counts > 0
    prototypes[present] = F.normalize(sums[present] / counts[present, None], dim=1)
    return prototypes.cpu()


def score_prototypes(
    prototypes: torch.Tensor,
    embeddings: torch.Tensor,
    truth: torch.Tensor,
) -> tuple[dict[str, float], torch.Tensor, torch.Tensor]:
    scores = embeddings.float() @ F.normalize(prototypes.float(), dim=1).T
    top3 = scores.topk(3, dim=1).indices
    predictions = top3[:, 0]
    per_class: list[float] = []
    for label in truth.unique(sorted=True):
        mask = truth == label
        per_class.append(float((predictions[mask] == truth[mask]).float().mean()))
    metrics = {
        "top1": float((predictions == truth).float().mean()),
        "macro_top1": sum(per_class) / len(per_class),
        "top3": float((top3 == truth[:, None]).any(dim=1).float().mean()),
    }
    return metrics, predictions, top3


def aggregate_predictions(
    predictions: list[int],
    truth: list[int],
    top3_hits: list[bool],
) -> dict[str, Any]:
    pred_tensor = torch.tensor(predictions, dtype=torch.long)
    truth_tensor = torch.tensor(truth, dtype=torch.long)
    hit_tensor = torch.tensor(top3_hits, dtype=torch.bool)
    per_class: dict[str, dict[str, float | int]] = {}
    class_top1: list[float] = []
    for label in truth_tensor.unique(sorted=True):
        mask = truth_tensor == label
        top1 = float((pred_tensor[mask] == truth_tensor[mask]).float().mean())
        top3 = float(hit_tensor[mask].float().mean())
        class_top1.append(top1)
        per_class[str(int(label))] = {
            "count": int(mask.sum()),
            "top1": top1,
            "top3": top3,
        }
    return {
        "top1": float((pred_tensor == truth_tensor).float().mean()),
        "macro_top1": sum(class_top1) / len(class_top1),
        "top3": float(hit_tensor.float().mean()),
        "per_class": per_class,
    }


def train_grouped_prototypes(
    train_embeddings: torch.Tensor,
    train_labels: torch.Tensor,
    baseline_prototypes: torch.Tensor,
    spec: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    configure_determinism(int(spec["seed"]))
    prototypes = nn.Parameter(initialize_prototypes(train_embeddings, train_labels, baseline_prototypes, spec).to(device))
    optimizer = torch.optim.AdamW([prototypes], lr=float(spec["lr"]), weight_decay=float(spec["weight_decay"]))
    epochs = int(spec["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    batch_size = int(spec["batch_size"])
    temperature = float(spec["temperature"])
    label_smoothing = float(spec["label_smoothing"])
    noise_std = float(spec["noise_std"])
    anchor_weight = float(spec["anchor_weight"])
    counts = torch.bincount(train_labels.long(), minlength=CLASS_COUNT).float()
    weights = torch.zeros(CLASS_COUNT, dtype=torch.float32)
    present = counts > 0
    weights[present] = counts[present].sum() / (present.sum().float() * counts[present])
    weights = weights.to(device)
    baseline = F.normalize(baseline_prototypes.float(), dim=1).to(device)
    generator = torch.Generator(device="cpu").manual_seed(int(spec["seed"]))

    for _ in range(epochs):
        permutation = torch.randperm(len(train_labels), generator=generator)
        for indices in permutation.split(batch_size):
            batch_embeddings = train_embeddings[indices].float().to(device)
            batch_labels = train_labels[indices].long().to(device)
            if noise_std > 0:
                batch_embeddings = batch_embeddings + torch.randn_like(batch_embeddings) * noise_std
            normalized = F.normalize(prototypes, dim=1)
            logits = batch_embeddings @ normalized.T / temperature
            loss = F.cross_entropy(logits, batch_labels, weight=weights, label_smoothing=label_smoothing)
            if anchor_weight > 0:
                anchor_loss = 1 - F.cosine_similarity(normalized[present], baseline[present], dim=1).mean()
                loss = loss + anchor_weight * anchor_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([prototypes], max_norm=5.0)
            optimizer.step()
        scheduler.step()
    return F.normalize(prototypes.detach(), dim=1).cpu()

def evaluate_spec(
    spec: dict[str, Any],
    runtime: v5.Runtime,
    admissible_rows: list[dict[str, Any]],
    folds_payload: list[dict[str, Any]],
    legacy_cache: dict[str, Any],
    external_cache: dict[str, Any],
    device: torch.device,
    embeddings: torch.Tensor,
) -> dict[str, Any]:
    row_lookup = {row["admissible_index"]: row for row in admissible_rows}
    fold_results: list[dict[str, Any]] = []
    candidate_preds_all: list[int] = []
    candidate_truth_all: list[int] = []
    candidate_top3_hits_all: list[bool] = []
    baseline_preds_all: list[int] = []
    baseline_truth_all: list[int] = []
    baseline_top3_hits_all: list[bool] = []
    for fold in folds_payload:
        val_indices = torch.tensor(fold["row_indices"], dtype=torch.long)
        val_set = set(fold.get("holdout_row_indices", fold["row_indices"]))
        train_indices = torch.tensor(
            [row["admissible_index"] for row in admissible_rows if row["admissible_index"] not in val_set],
            dtype=torch.long,
        )
        train_embeddings = embeddings[train_indices]
        train_labels = torch.tensor([row_lookup[i]["class_label"] for i in train_indices.tolist()], dtype=torch.long)
        val_embeddings = embeddings[val_indices]
        val_truth = torch.tensor([row_lookup[i]["class_label"] for i in val_indices.tolist()], dtype=torch.long)
        candidate = train_grouped_prototypes(train_embeddings, train_labels, runtime.prototypes.cpu(), spec, device)
        candidate_metrics, candidate_preds, candidate_top3 = score_prototypes(candidate, val_embeddings, val_truth)
        baseline_metrics, baseline_preds, baseline_top3 = score_prototypes(runtime.prototypes.cpu(), val_embeddings, val_truth)
        fold_results.append(
            {
                "fold": fold["fold"],
                "source_groups": fold["source_groups"],
                "train_count": int(len(train_indices)),
                "val_count": int(len(val_indices)),
                "class_counts": fold["class_counts"],
                "baseline": baseline_metrics,
                "candidate": candidate_metrics,
            }
        )
        candidate_preds_all.extend(candidate_preds.tolist())
        candidate_truth_all.extend(val_truth.tolist())
        candidate_top3_hits_all.extend((candidate_top3 == val_truth[:, None]).any(dim=1).tolist())
        baseline_preds_all.extend(baseline_preds.tolist())
        baseline_truth_all.extend(val_truth.tolist())
        baseline_top3_hits_all.extend((baseline_top3 == val_truth[:, None]).any(dim=1).tolist())

    candidate_oof = aggregate_predictions(
        candidate_preds_all, candidate_truth_all, candidate_top3_hits_all
    )
    baseline_oof = aggregate_predictions(
        baseline_preds_all, baseline_truth_all, baseline_top3_hits_all
    )
    pass_gate = bool(candidate_oof["top1"] > baseline_oof["top1"] and candidate_oof["macro_top1"] >= baseline_oof["macro_top1"])
    return {
        "schema_version": SCHEMA_VERSION,
        "iteration": int(spec["iteration"]),
        "name": spec["name"],
        "track": spec["track"],
        "pass": pass_gate,
        "score": candidate_oof["top1"],
        "objective": spec["objective"],
        "baseline_runtime": "white",
        "projection": spec["projection"],
        "initialization": spec["initialization"],
        "modification": spec["modification"],
        "folds": fold_results,
        "oof": {
            "baseline": baseline_oof,
            "candidate": candidate_oof,
            "delta_top1": candidate_oof["top1"] - baseline_oof["top1"],
            "delta_macro_top1": candidate_oof["macro_top1"] - baseline_oof["macro_top1"],
        },
        "runtime_compatible": True,
        "promotion_eligible": False,
        "final_test_unavailable": True,
        "oof_count": len(candidate_truth_all),
        "spec": spec,
    }

def refit_selected(
    selected_spec: dict[str, Any],
    runtime: v5.Runtime,
    admissible_rows: list[dict[str, Any]],
    embeddings: torch.Tensor,
    device: torch.device,
    run_dir: Path,
    *,
    oof_summary: dict[str, Any],
) -> dict[str, Any]:
    all_indices = torch.tensor([row["admissible_index"] for row in admissible_rows], dtype=torch.long)
    all_labels = torch.tensor([row["class_label"] for row in admissible_rows], dtype=torch.long)
    selected = train_grouped_prototypes(embeddings[all_indices], all_labels, runtime.prototypes.cpu(), selected_spec, device)
    refit_dir = run_dir / "refit" / "checkpoints"
    refit_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_payload = {
        "prototypes": selected,
        "class_labels": runtime.class_labels.clone(),
        "class_names": dict(runtime.class_names),
        "spec": selected_spec,
        "validation_mode": "grouped_cv_oof_no_final_test",
        "promotion_eligible": False,
        "final_test_unavailable": True,
        "baseline_runtime": "white",
        "oof_summary": oof_summary,
    }
    latest_path = refit_dir / "latest.pt"
    best_path = refit_dir / "best.pt"
    torch.save(checkpoint_payload, latest_path)
    torch.save(checkpoint_payload, best_path)
    provenance = {
        "schema_version": f"{SCHEMA_VERSION}.refit",
        "selected_spec": selected_spec,
        "checkpoint_latest_path": str(latest_path.resolve()),
        "checkpoint_best_path": str(best_path.resolve()),
        "checkpoint_latest_sha256": v5.sha256_file(latest_path),
        "checkpoint_best_sha256": v5.sha256_file(best_path),
        "promotion_eligible": False,
        "final_test_unavailable": True,
        "baseline_runtime": "white",
        "oof_summary": oof_summary,
    }
    provenance_path = run_dir / "refit" / "refit_provenance.json"
    dump_json(provenance_path, provenance)
    return {**provenance, "provenance_path": str(provenance_path.resolve())}


def run_grouped_cv(
    *,
    legacy_cache_path: Path,
    legacy_manifest_path: Path,
    external_cache_path: Path,
    import_snapshot_path: Path,
    runtime_weights_dir: Path,
    runtime_config_path: Path,
    run_dir: Path,
    device_name: str,
    folds: int,
    max_runtime_seconds: float,
    quick: bool,
) -> dict[str, Any]:
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the requested device")
    device = torch.device(device_name)
    started_at = utc_now()
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    state_path = run_dir / "state.json"
    mission_path = run_dir / "mission.md"
    dump_json(
        state_path,
        {
            "schema_version": f"{SCHEMA_VERSION}.state",
            "status": "running",
            "run_id": run_dir.name,
            "mission": "elements-baseline-replacement-v7-grouped-cv",
            "started_at": started_at,
            "updated_at": started_at,
            "iteration": 0,
            "folds": folds,
            "spec_count": 10,
            "promotion_eligible": False,
            "final_test_unavailable": True,
            "max_runtime_seconds": max_runtime_seconds,
        },
    )

    runtime = load_runtime(runtime_weights_dir, runtime_config_path, device)
    legacy_probe = elements.load_cache(legacy_cache_path)
    legacy_cache = load_cache(legacy_cache_path, cache_name="legacy cache")
    external_cache = load_cache(external_cache_path, cache_name="external cache")
    if len(legacy_probe["labels"]) != len(legacy_cache["labels"]):
        raise ValueError("legacy cache shape changed while probing legacy loader")
    snapshot = v5.load_snapshot(import_snapshot_path, runtime.ordered_names)
    legacy_digest_index = load_legacy_digest_index(legacy_manifest_path)
    v5.validate_cache_taxonomy(external_cache, runtime.ordered_names, cache_name="external cache")
    v5.validate_cache_taxonomy(legacy_cache, runtime.ordered_names, cache_name="legacy cache")

    admissible_rows, audit = build_admissible_rows(
        legacy_cache, external_cache, snapshot, legacy_digest_index
    )
    audit["legacy_manifest_path"] = str(legacy_manifest_path.resolve())
    audit["legacy_manifest_sha256"] = v5.sha256_file(legacy_manifest_path)
    audit["legacy_manifest_digest_count"] = len(legacy_digest_index)
    for index, row in enumerate(admissible_rows):
        row["admissible_index"] = index
    raw_fold_assignments = build_group_folds(admissible_rows, folds)
    fold_assignments, support_audit = enforce_fold_train_support(admissible_rows, raw_fold_assignments)
    audit.update(support_audit)
    dump_json(
        run_dir / "folds.json",
        {"schema_version": f"{SCHEMA_VERSION}.folds", "folds": fold_assignments, "support_audit": support_audit},
    )
    ordered_features = torch.stack([row_feature(row, legacy_cache, external_cache) for row in admissible_rows])
    embeddings = v5.project(runtime.model, ordered_features, device)

    evaluator = {
        "schema_version": f"{SCHEMA_VERSION}.evaluator",
        "mission": "elements-baseline-replacement-v7-grouped-cv",
        "baseline_runtime": "white",
        "selection_metric": "oof_top1",
        "secondary_gate": "candidate oof macro_top1 >= baseline oof macro_top1",
        "required_iteration_output": {"pass": "boolean", "score": "number"},
        "promotion_eligible": False,
        "final_test_unavailable": True,
        "spec_count": 10,
        "folds": folds,
        "filters": {
            "external_group_overlap_removed": audit["external_overlap_removed"],
            "deduplicated_by_source_pixel_sha256": audit["deduplicated_by_source_pixel_sha256"],
            "oof_row_count": audit["oof_row_count"],
            "excluded_oof_row_count": audit["excluded_oof_row_count"],
            "excluded_oof_class_count": audit["excluded_oof_class_count"],
        },
    }
    dump_json(run_dir / "evaluator.json", evaluator)
    mission_path.write_text(
        "\n".join(
            [
                "# Grouped-CV autoresearch v7",
                "",
                f"- Started: {started_at}",
                "- Baseline runtime: white",
                "- Promotion eligible: false",
                "- Final test unavailable: true",
                f"- Legacy rows: {audit['legacy_row_count']}",
                f"- External rows: {audit['external_row_count']}",
                f"- External kept after legacy overlap filter: {audit['external_kept_count']}",
                f"- Deduplicated by source_pixel_sha256: {audit['deduplicated_by_source_pixel_sha256']}",
                f"- Admissible rows: {audit['admissible_row_count']}",
                f"- OOF rows with independent train support: {audit['oof_row_count']}",
                f"- OOF rows excluded for missing train support: {audit['excluded_oof_row_count']}",
                f"- Folds: {folds}",
                "",
                "The runner never reads a final test artifact. Selection is based only on OOF folds.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    specs_payload = build_specs(quick=quick)
    decision_lines = [
        "# Grouped-CV autoresearch v7",
        "",
        f"Started: {started_at}",
        "Baseline runtime: white",
        "Final test unavailable: true",
        "",
    ]
    best: dict[str, Any] | None = None
    deadline = time.time() + max_runtime_seconds
    for spec in specs_payload:
        if time.time() >= deadline:
            raise TimeoutError("grouped-CV autoresearch exceeded its runtime budget")
        iteration_started = time.perf_counter()
        result = evaluate_spec(
            spec,
            runtime,
            admissible_rows,
            fold_assignments,
            legacy_cache,
            external_cache,
            device,
            embeddings,
        )
        result["elapsed_seconds"] = time.perf_counter() - iteration_started
        dump_json(run_dir / "specs" / f"iteration-{spec['iteration']:04d}.json", spec)
        dump_json(run_dir / "evaluations" / f"iteration-{spec['iteration']:04d}.json", result)
        decision_lines.extend(
            [
                f"## Iteration {spec['iteration']:04d} — {spec['name']}",
                "",
                f"- Track: {spec['track']}",
                f"- Modification: {spec['modification']}",
                f"- Pass: {result['pass']}",
                f"- Score: {result['score']:.6f}",
                f"- OOF candidate top-1: {result['oof']['candidate']['top1']:.6f}",
                f"- OOF baseline top-1: {result['oof']['baseline']['top1']:.6f}",
                "",
            ]
        )
        if best is None or (
            result["score"],
            result["oof"]["candidate"]["macro_top1"],
            -spec["iteration"],
        ) > (
            best["score"],
            best["oof"]["candidate"]["macro_top1"],
            -best["iteration"],
        ):
            best = result
        dump_json(
            state_path,
            {
                "schema_version": f"{SCHEMA_VERSION}.state",
                "status": "running",
                "run_id": run_dir.name,
                "mission": "elements-baseline-replacement-v7-grouped-cv",
                "started_at": started_at,
                "updated_at": utc_now(),
                "iteration": spec["iteration"],
                "leader": best["name"] if best else None,
                "folds": folds,
                "spec_count": len(specs_payload),
                "promotion_eligible": False,
                "final_test_unavailable": True,
                "max_runtime_seconds": max_runtime_seconds,
            },
        )

    assert best is not None
    refit = refit_selected(
        best["spec"],
        runtime,
        admissible_rows,
        embeddings,
        device,
        run_dir,
        oof_summary=best["oof"],
    )
    final = {
        "schema_version": f"{SCHEMA_VERSION}.final",
        "pass": bool(best["pass"]),
        "score": float(best["score"]),
        "selected": {
            "iteration": best["iteration"],
            "name": best["name"],
            "track": best["track"],
            "modification": best["modification"],
        },
        "oof": best["oof"],
        "promotion_eligible": False,
        "final_test_unavailable": True,
        "baseline_runtime": "white",
        "folds": folds,
        "spec_count": len(specs_payload),
        "refit": refit,
        "filters": audit,
        "runtime_exported": False,
    }
    dump_json(run_dir / "final.json", final)
    dump_json(run_dir / "final-evaluation.json", final)
    decision_lines.extend(
        [
            "## Final",
            "",
            f"- Selected: {best['name']}",
            f"- OOF candidate top-1: {best['oof']['candidate']['top1']:.6f}",
            f"- OOF baseline top-1: {best['oof']['baseline']['top1']:.6f}",
            "- Promotion eligible: false",
            "- Final test unavailable: true",
            "",
        ]
    )
    (run_dir / "decision-log.md").write_text("\n".join(decision_lines), encoding="utf-8")
    dump_json(
        state_path,
        {
            "schema_version": f"{SCHEMA_VERSION}.state",
            "status": "completed",
            "run_id": run_dir.name,
            "mission": "elements-baseline-replacement-v7-grouped-cv",
            "started_at": started_at,
            "updated_at": utc_now(),
            "completed_at": utc_now(),
            "iteration": len(specs_payload),
            "leader": best["name"],
            "pass": bool(best["pass"]),
            "folds": folds,
            "spec_count": len(specs_payload),
            "promotion_eligible": False,
            "final_test_unavailable": True,
            "max_runtime_seconds": max_runtime_seconds,
            "refit_checkpoint_path": refit["checkpoint_best_path"],
        },
    )
    return final


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-cache", type=Path, default=DEFAULT_LEGACY_CACHE)
    parser.add_argument("--legacy-manifest", type=Path, default=DEFAULT_LEGACY_MANIFEST)
    parser.add_argument("--external-cache", type=Path, default=DEFAULT_EXTERNAL_CACHE)
    parser.add_argument("--import-snapshot", type=Path, default=DEFAULT_IMPORT_SNAPSHOT)
    parser.add_argument("--runtime-weights", type=Path, default=DEFAULT_RUNTIME_WEIGHTS)
    parser.add_argument("--runtime-config", type=Path, default=DEFAULT_RUNTIME_CONFIG)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-runtime", type=float, default=DEFAULT_MAX_RUNTIME_SECONDS)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)

    result = run_grouped_cv(
        legacy_cache_path=args.legacy_cache,
        legacy_manifest_path=args.legacy_manifest,
        external_cache_path=args.external_cache,
        import_snapshot_path=args.import_snapshot,
        runtime_weights_dir=args.runtime_weights,
        runtime_config_path=args.runtime_config,
        run_dir=args.run_dir,
        device_name=args.device,
        folds=args.folds,
        max_runtime_seconds=args.max_runtime,
        quick=args.quick,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

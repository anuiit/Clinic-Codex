#!/usr/bin/env python3
"""Thin, gated v19-A runner reusing the canonical v9 mechanism unchanged."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
BACKEND = ROOT / "backend"
for import_path in (SCRIPTS_DIR, BACKEND):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import autoresearch_backbone_screen_v10 as backbone_screen  # noqa: E402
import autoresearch_hierarchical_shrinkage_v17 as v17  # noqa: E402
import autoresearch_self_supervised_v9 as v9  # noqa: E402
from codex_pipeline.determinism import configure_determinism  # noqa: E402
from codex_pipeline.models.projection_head import ProjectionHead  # noqa: E402


RUN = ROOT / ".omc/autoresearch/elements-baseline-replacement/runs/20260804-vicreg-backbone-renomination-v19"
V9_RUN = ROOT / ".omc/autoresearch/elements-baseline-replacement/runs/20260803-self-supervised-v9"
V10_RUN = ROOT / ".omc/autoresearch/elements-baseline-replacement/runs/20260803-self-supervised-v10"
V18_RUN = ROOT / ".omc/autoresearch/elements-baseline-replacement/runs/20260804-discriminative-readout-v18"

DEFAULT_SPEC = RUN / "specs/iteration-0002.json"
DEFAULT_EVALUATOR = RUN / "evaluator-iteration-0002.json"
DEFAULT_FREEZE_AUDIT = RUN / "iteration-0002-contract-freeze-audit.json"
DEFAULT_B14_MANIFEST = V10_RUN / "inputs/dinov2-vitb14-local-manifest.json"
DEFAULT_SOURCE_INVENTORY = V18_RUN / "specs/iteration-0003-source-inventory.jsonl"
DEFAULT_CORPUS_MANIFEST = v9.DEFAULT_MANIFEST
DEFAULT_COLLECTION_AUDIT = v9.DEFAULT_AUDIT
DEFAULT_C1_SUMMARY = V9_RUN / "iteration-0001/results/paired_seed_summary.json"
DEFAULT_C1_PREDICTIONS = V9_RUN / "iteration-0001/results/prediction_rows.jsonl"
DEFAULT_C1_CACHE_DIR = V9_RUN / "iteration-0001/caches"
DEFAULT_B14_CACHE_DIR = RUN / "iteration-0002/caches"
DEFAULT_CACHE_MANIFEST = RUN / "iteration-0002/cache-manifest.json"
DEFAULT_CONTROL_AUDIT = RUN / "iteration-0002/c1-control-replay-audit.json"
DEFAULT_OUTPUT_DIR = RUN / "iteration-0003"
DEFAULT_REPLAY_DIR = RUN / "iteration-0003-replay"
DEFAULT_RUNTIME_PROJECTION = BACKEND / "codex_model/weights/projection.pt"
DEFAULT_RUNTIME_PROTOTYPES = BACKEND / "codex_model/weights/prototypes.pt"
DEFAULT_RUNTIME_CONFIG = BACKEND / "codex_model/config.json"

EXPECTED_SPEC_SHA256 = "d4da848144eb5b0353919556fcd8ce42cff0a78a5d91e9a914b2e937ff270087"
EXPECTED_EVALUATOR_SHA256 = "48494ce2fda34f05f8f0202e164f1d1582008eab02bb5bb0444d3c18f7033b92"
EXPECTED_V9_RUNNER_SHA256 = "e03e1e29340a1a165ab2f82001236f5aaae4450afcee862dff8cb27ef26c48a9"
EXPECTED_V9_RECOVERY_AUDIT_SHA256 = "8f04000fd14e61e899e05c034fb36ab71dce3fecd7cb423fbba11ba476583cf5"
EXPECTED_V9_TEST_SPEC_SHA256 = "cb9f4f65ab1f82b7db9e2a7a0201c8461ea59eca8a6801d4f6474d9432f32803"
EXPECTED_C1_SUMMARY_SHA256 = "d6094bc9887748ba9bc89b3f6d691483b803be806cf0623e450450132387a22d"
EXPECTED_C1_PREDICTIONS_SHA256 = "37a15fb66ef883d15df5db8e1cadb152f9460876b60ace832c434375a3cdf34f"
EXPECTED_V17_RUNNER_SHA256 = "a1f0fb9e0fae18b0f5fb39dbbfa93361f041980447403eeb6a12f31630bafb08"
EXPECTED_B14_MANIFEST_SHA256 = "ea73c8b9d2cc8a81a48183a43516fc76ad6b2bd5694442229d9b03ce3774e4d5"
EXPECTED_SOURCE_INVENTORY_SHA256 = "d21ccbc123d454db773eb31fbc2fe3cf225c21c79534a142cf0b3d34340a87d2"
EXPECTED_RUNTIME_SHA256 = {
    "projection": "0ad6ce348eaf44c8844415a769b048b8f155017befa93f16254b1263b9241210",
    "prototypes": "ffc52f03696f03780a01302708b53e319dc24069187a6fd1a2927ea537b1cc54",
    "config": "f2150a97ec7eee50a42ac1a108ae4bc092d46a6b8e4a2d91de59349560fbf18b",
}
EXPECTED_FOLDS = (1, 2, 3, 4, 5)
EXPECTED_SEEDS = (17, 42, 73)
EXPECTED_PAIRED_ROWS = 1959
EXPECTED_UNIQUE_OOF_ROWS = 653
EXPECTED_TRAIN_ROWS_ACROSS_FOLDS = 39960
EXPECTED_CACHE_TENSOR_BYTES = 552_407_040
MAX_CACHE_BYTES = 2 * 1024**3
BACKBONE = "dinov2_vitb14"
EMBED_DIM = 768
VIEWS = 8
VIEW_SEED = 20260803
IMAGE_SIZE = 224
IMAGE_BATCH_SIZE = 32

REPLAY_NORMALIZATION_CONTRACT = {
    "schema_version": "autoresearch-v19.replay-normalization-v1",
    "summary_removed_keys": ["artifact_paths"],
    "diagnostic_removed_keys": ["checkpoint_path"],
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"expected JSON objects in {path}")
    return rows


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def runtime_hashes(args: argparse.Namespace) -> dict[str, str]:
    return {
        "projection": v9.sha256_file(args.runtime_projection),
        "prototypes": v9.sha256_file(args.runtime_prototypes),
        "config": v9.sha256_file(args.runtime_config),
    }


def resolve_artifact(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def canonical_diagnostic_map(summary: dict[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    diagnostics = {
        (int(item["fold"]), int(item["seed"])): item
        for item in summary["diagnostics"]
    }
    expected = {(fold, seed) for fold in EXPECTED_FOLDS for seed in EXPECTED_SEEDS}
    if set(diagnostics) != expected:
        raise ValueError("canonical C1 diagnostics do not cover 5 folds x 3 seeds")
    return diagnostics


def validate_static_contract(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "spec": (args.spec, EXPECTED_SPEC_SHA256),
        "evaluator": (args.evaluator, EXPECTED_EVALUATOR_SHA256),
        "v9_runner": (Path(v9.__file__).resolve(), EXPECTED_V9_RUNNER_SHA256),
        "v9_recovery_audit": (
            RUN / "v9-canonical-runner-recovery-audit.json",
            EXPECTED_V9_RECOVERY_AUDIT_SHA256,
        ),
        "v9_test_spec": (V9_RUN / "test-spec.json", EXPECTED_V9_TEST_SPEC_SHA256),
        "c1_summary": (args.c1_summary, EXPECTED_C1_SUMMARY_SHA256),
        "c1_predictions": (args.c1_predictions, EXPECTED_C1_PREDICTIONS_SHA256),
        "v17_runner": (Path(v17.__file__).resolve(), EXPECTED_V17_RUNNER_SHA256),
        "b14_manifest": (args.b14_manifest, EXPECTED_B14_MANIFEST_SHA256),
        "source_inventory": (args.source_inventory, EXPECTED_SOURCE_INVENTORY_SHA256),
    }
    verified: dict[str, dict[str, Any]] = {}
    for name, (path, expected_sha256) in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing frozen input {name}: {path}")
        actual_sha256 = v9.sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(f"frozen input hash mismatch for {name}: {actual_sha256}")
        verified[name] = {
            "path": relative_path(path),
            "sha256": actual_sha256,
            "bytes": path.stat().st_size,
        }

    runtime = runtime_hashes(args)
    if runtime != EXPECTED_RUNTIME_SHA256:
        raise ValueError(f"runtime hash mismatch: {runtime}")
    pin = backbone_screen.validate_backbone_manifest(args.b14_manifest)
    if pin["manifest_sha256"] != EXPECTED_B14_MANIFEST_SHA256:
        raise ValueError("validated B/14 manifest hash mismatch")

    recovery = read_json(RUN / "v9-canonical-runner-recovery-audit.json")
    if recovery.get("pass") is not True:
        raise ValueError("v9 runner recovery audit did not pass")

    summary = read_json(args.c1_summary)
    if summary.get("code_sha256") != EXPECTED_V9_RUNNER_SHA256:
        raise ValueError("canonical C1 summary code hash mismatch")
    diagnostics = canonical_diagnostic_map(summary)
    predictions = read_jsonl(args.c1_predictions)
    if len(predictions) != EXPECTED_PAIRED_ROWS:
        raise ValueError("canonical C1 prediction-row count mismatch")
    if {str(row.get("code_sha256")) for row in predictions} != {
        EXPECTED_V9_RUNNER_SHA256
    }:
        raise ValueError("canonical C1 rows do not share the recovered v9 code hash")
    if len({(str(row["row_id"]), int(row["outer_fold"])) for row in predictions}) != (
        EXPECTED_UNIQUE_OOF_ROWS
    ):
        raise ValueError("canonical C1 unique OOF-row count mismatch")

    cache_pins: dict[str, dict[str, Any]] = {}
    for fold in EXPECTED_FOLDS:
        cache_path = v9.cache_path(args.c1_cache_dir, fold, VIEWS, None)
        sidecar_path = cache_path.with_suffix(cache_path.suffix + ".prov.json")
        expected_cache_sha256 = str(summary["cache_sha256"][str(fold)])
        if v9.sha256_file(cache_path) != expected_cache_sha256:
            raise ValueError(f"canonical S/14 cache hash mismatch for fold {fold}")
        sidecar = read_json(sidecar_path)
        if sidecar.get("cache_sha256") != expected_cache_sha256:
            raise ValueError(f"canonical S/14 cache sidecar mismatch for fold {fold}")
        cache_pins[str(fold)] = {
            "cache_path": relative_path(cache_path),
            "cache_sha256": expected_cache_sha256,
            "sidecar_path": relative_path(sidecar_path),
            "sidecar_sha256": v9.sha256_file(sidecar_path),
            "train_row_ids_sha256": sidecar["train_row_ids_sha256"],
            "oof_row_ids_sha256": sidecar["oof_row_ids_sha256"],
        }

    checkpoint_count = 0
    for diagnostic in diagnostics.values():
        for arm in ("baseline", "candidate"):
            checkpoint = resolve_artifact(diagnostic["checkpoints"][arm]["path"])
            expected = str(diagnostic["checkpoints"][arm]["sha256"])
            if v9.sha256_file(checkpoint) != expected:
                raise ValueError(f"canonical {arm} checkpoint hash mismatch: {checkpoint}")
            checkpoint_count += 1
    if checkpoint_count != 30:
        raise ValueError("canonical checkpoint count mismatch")

    return {
        "schema_version": "autoresearch-v19.static-contract-validation-v1",
        "pass": True,
        "verified_inputs": verified,
        "runtime_sha256": runtime,
        "backbone_pin": pin,
        "canonical_cache_pins": cache_pins,
        "canonical_checkpoint_hash_match_count": checkpoint_count,
        "canonical_diagnostic_count": len(diagnostics),
        "canonical_prediction_row_count": len(predictions),
        "operation_counts": {
            "source_image_reads": 0,
            "oof_accesses": 0,
            "real_feature_extractions": 0,
            "candidate_optimizer_steps": 0,
            "candidate_predictions": 0,
            "candidate_scores": 0,
            "runtime_writes": 0,
        },
        "final_test_read": False,
        "runtime_unchanged": True,
    }


def validate_phase_authorization(
    args: argparse.Namespace,
    *,
    expected_phase: str,
) -> dict[str, Any]:
    if args.authorization is None:
        raise ValueError(f"{expected_phase} requires --authorization")
    freeze_audit = read_json(args.freeze_audit)
    authorization = read_json(args.authorization)
    if authorization.get("schema_version") != "autoresearch-v19.phase-authorization-v1":
        raise ValueError("unexpected phase authorization schema")
    if authorization.get("authorized") is not True:
        raise ValueError("phase authorization is not affirmative")
    if authorization.get("phase") != expected_phase:
        raise ValueError("phase authorization targets a different phase")
    if authorization.get("council_session_id") != "adv_20260803T072824_f903d4d6":
        raise ValueError("phase authorization Council session mismatch")
    freeze_sha256 = v9.sha256_file(args.freeze_audit)
    if authorization.get("contract_freeze_audit_sha256") != freeze_sha256:
        raise ValueError("phase authorization freeze-audit hash mismatch")
    current_runner_sha256 = v9.sha256_file(Path(__file__))
    expected_artifacts = freeze_audit.get("frozen_artifacts", {})
    if expected_artifacts.get("runner_sha256") != current_runner_sha256:
        raise ValueError("current runner does not match frozen runner")
    if expected_artifacts.get("spec_sha256") != EXPECTED_SPEC_SHA256:
        raise ValueError("freeze audit spec hash mismatch")
    if expected_artifacts.get("evaluator_sha256") != EXPECTED_EVALUATOR_SHA256:
        raise ValueError("freeze audit evaluator hash mismatch")
    if authorization.get("published_synthesis_sha256") in (None, ""):
        raise ValueError("phase authorization lacks published synthesis hash")
    return {
        "path": relative_path(args.authorization),
        "sha256": v9.sha256_file(args.authorization),
        "phase": expected_phase,
        "published_synthesis_sha256": authorization["published_synthesis_sha256"],
        "contract_freeze_audit_sha256": freeze_sha256,
    }

METADATA_FIELDS = (
    "row_id",
    "class_label",
    "class_name",
    "component_id",
    "decoded_pixel_sha256",
    "fold",
)


def sha256_sequence(values: Sequence[Any]) -> str:
    return v9.sha256_json(list(values))


def view_plan_sha256(
    row_ids: Sequence[Any],
    *,
    views: int = VIEWS,
    seed: int = VIEW_SEED,
) -> str:
    digest = hashlib.sha256()
    for row_id in row_ids:
        for view_index in range(views):
            payload = {
                "row_id": str(row_id),
                "view_index": view_index,
                "rng_seed": v9.stable_seed("ssl-view", seed, str(row_id), view_index),
            }
            digest.update(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))
            digest.update(b"\n")
    return digest.hexdigest()


def assert_ext4_workspace_path(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    root = ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"cache path is outside the workspace: {resolved}")
    if str(resolved).startswith("/mnt/"):
        raise ValueError("drvfs cache paths are forbidden")
    existing = resolved
    while not existing.exists():
        if existing.parent == existing:
            raise FileNotFoundError(f"cannot resolve parent filesystem for {resolved}")
        existing = existing.parent
    if os.stat(existing).st_dev != os.stat(root).st_dev:
        raise ValueError("cache path is not on the workspace filesystem")
    mounts: list[tuple[int, str, str]] = []
    for line in Path("/proc/mounts").read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 3:
            continue
        mountpoint = fields[1].replace("\\040", " ")
        try:
            mount = Path(mountpoint).resolve()
        except OSError:
            continue
        if resolved == mount or mount in resolved.parents:
            mounts.append((len(str(mount)), fields[2], str(mount)))
    if not mounts:
        raise ValueError(f"cannot identify filesystem for {resolved}")
    _, filesystem, mountpoint = max(mounts)
    if filesystem != "ext4":
        raise ValueError(f"cache filesystem must be ext4, found {filesystem}")
    return {
        "resolved_path": str(resolved),
        "filesystem": filesystem,
        "mountpoint": mountpoint,
        "workspace_device": int(os.stat(root).st_dev),
    }


def load_canonical_cache(
    args: argparse.Namespace,
    fold: int,
) -> dict[str, Any]:
    return v9.load_fold_cache(
        v9.cache_path(args.c1_cache_dir, fold, VIEWS, None),
        expected_fold=fold,
        expected_views=VIEWS,
        expected_view_seed=VIEW_SEED,
        expected_image_size=IMAGE_SIZE,
        expected_max_rows_per_class=None,
    )


def validate_b14_cache(
    path: Path,
    *,
    fold: int,
    canonical: dict[str, Any],
    backbone_provenance: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = v9.load_fold_cache(
        path,
        expected_fold=fold,
        expected_views=VIEWS,
        expected_view_seed=VIEW_SEED,
        expected_image_size=IMAGE_SIZE,
        expected_max_rows_per_class=None,
        expected_backbone_provenance=backbone_provenance,
    )
    train = payload["train"]
    oof = payload["oof"]
    canonical_train = canonical["train"]
    canonical_oof = canonical["oof"]
    for split_name, split, reference in (
        ("train", train, canonical_train),
        ("oof", oof, canonical_oof),
    ):
        for field in METADATA_FIELDS:
            if list(split[field]) != list(reference[field]):
                raise ValueError(f"B/14 {split_name}.{field} order differs from canonical S/14")
    if tuple(train["base_features"].shape) != (len(train["row_id"]), EMBED_DIM):
        raise ValueError("B/14 train base-feature shape mismatch")
    if tuple(train["view_features"].shape) != (len(train["row_id"]), VIEWS, EMBED_DIM):
        raise ValueError("B/14 train view-feature shape mismatch")
    if tuple(oof["base_features"].shape) != (len(oof["row_id"]), EMBED_DIM):
        raise ValueError("B/14 OOF base-feature shape mismatch")
    for tensor in (train["base_features"], train["view_features"], oof["base_features"]):
        if tensor.dtype != torch.float16:
            raise ValueError("B/14 cache tensors must be float16")
        if not torch.isfinite(tensor).all():
            raise FloatingPointError("non-finite B/14 cache tensor")
    if "view_features" in oof:
        raise ValueError("B/14 OOF augmented views are forbidden")
    sidecar_path = path.with_suffix(path.suffix + ".prov.json")
    sidecar = read_json(sidecar_path)
    train_row_hash = sha256_sequence(train["row_id"])
    oof_row_hash = sha256_sequence(oof["row_id"])
    if train_row_hash != sidecar["train_row_ids_sha256"]:
        raise ValueError("B/14 train order hash differs from sidecar")
    if oof_row_hash != sidecar["oof_row_ids_sha256"]:
        raise ValueError("B/14 OOF order hash differs from sidecar")
    validation = {
        "fold": fold,
        "cache_path": relative_path(path),
        "cache_sha256": v9.sha256_file(path),
        "byte_identical_readback": sidecar["cache_sha256"] == v9.sha256_file(path),
        "cache_bytes": path.stat().st_size,
        "sidecar_path": relative_path(sidecar_path),
        "sidecar_sha256": v9.sha256_file(sidecar_path),
        "train_rows": len(train["row_id"]),
        "oof_rows": len(oof["row_id"]),
        "train_row_ids_sha256": train_row_hash,
        "oof_row_ids_sha256": oof_row_hash,
        "canonical_train_row_ids_sha256": sha256_sequence(canonical_train["row_id"]),
        "canonical_oof_row_ids_sha256": sha256_sequence(canonical_oof["row_id"]),
        "view_plan_sha256": view_plan_sha256(train["row_id"]),
        "train_shape": list(train["base_features"].shape),
        "view_shape": list(train["view_features"].shape),
        "oof_shape": list(oof["base_features"].shape),
        "dtype": str(train["base_features"].dtype),
        "oof_view_features_persisted": False,
        "final_test_read": False,
    }
    return payload, validation


def validate_cache_manifest(args: argparse.Namespace) -> dict[str, Any]:
    manifest = read_json(args.cache_manifest)
    if manifest.get("schema_version") != "autoresearch-v19.b14-fold-cache-manifest-v1":
        raise ValueError("unexpected B/14 cache manifest schema")
    if manifest.get("pass") is not True:
        raise ValueError("B/14 cache manifest did not pass")
    if manifest.get("spec_sha256") != EXPECTED_SPEC_SHA256:
        raise ValueError("B/14 cache manifest spec mismatch")
    if manifest.get("evaluator_sha256") != EXPECTED_EVALUATOR_SHA256:
        raise ValueError("B/14 cache manifest evaluator mismatch")
    if manifest.get("expected_tensor_bytes") != EXPECTED_CACHE_TENSOR_BYTES:
        raise ValueError("B/14 cache manifest byte projection mismatch")
    entries = {int(item["fold"]): item for item in manifest.get("folds", [])}
    if set(entries) != set(EXPECTED_FOLDS):
        raise ValueError("B/14 cache manifest does not cover all folds")
    total = 0
    for fold, item in entries.items():
        path = resolve_artifact(item["cache_path"])
        sidecar = resolve_artifact(item["sidecar_path"])
        if v9.sha256_file(path) != item["cache_sha256"]:
            raise ValueError(f"B/14 cache hash mismatch for fold {fold}")
        if v9.sha256_file(sidecar) != item["sidecar_sha256"]:
            raise ValueError(f"B/14 sidecar hash mismatch for fold {fold}")
        total += path.stat().st_size
    if total != int(manifest["total_cache_bytes"]):
        raise ValueError("B/14 cache manifest total-byte mismatch")
    if manifest.get("runtime_sha256_after") != EXPECTED_RUNTIME_SHA256:
        raise ValueError("runtime changed during B/14 extraction")
    if manifest.get("final_test_read") is not False:
        raise ValueError("cache manifest reports final-test access")
    return manifest


def command_precompute(args: argparse.Namespace) -> dict[str, Any]:
    authorization = validate_phase_authorization(args, expected_phase="phase_2_extraction_and_c1_control")
    static = validate_static_contract(args)
    filesystem = assert_ext4_workspace_path(args.b14_cache_dir)
    if args.b14_cache_dir.exists() and any(args.b14_cache_dir.iterdir()):
        raise FileExistsError("B/14 cache directory must be absent or empty")
    args.b14_cache_dir.mkdir(parents=True, exist_ok=True)
    rows, corpus_report = v9.load_corpus(args.manifest, args.audit, strict_counts=True)
    if len(rows) != 9990:
        raise ValueError("retained corpus row count mismatch")
    device = v9.resolve_device(args.device)
    pin = backbone_screen.validate_backbone_manifest(args.b14_manifest)
    backbone, provenance = v9.load_backbone(BACKBONE, device, args.b14_manifest)
    backbone.requires_grad_(False).eval()
    entries: list[dict[str, Any]] = []
    source_image_reads = 0
    started = time.time()
    try:
        for fold in EXPECTED_FOLDS:
            canonical = load_canonical_cache(args, fold)
            cache_path = v9.precompute_fold(
                rows,
                corpus_report,
                fold=fold,
                views=VIEWS,
                view_seed=VIEW_SEED,
                image_size=IMAGE_SIZE,
                batch_size=IMAGE_BATCH_SIZE,
                num_workers=args.num_workers,
                device=device,
                backbone=backbone,
                backbone_provenance=provenance,
                cache_dir=args.b14_cache_dir,
                max_rows_per_class=None,
                force=False,
            )
            _, validation = validate_b14_cache(
                cache_path,
                fold=fold,
                canonical=canonical,
                backbone_provenance=provenance,
            )
            source_image_reads += validation["train_rows"] + validation["oof_rows"]
            entries.append(validation)
    finally:
        del backbone
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if source_image_reads != EXPECTED_TRAIN_ROWS_ACROSS_FOLDS + EXPECTED_UNIQUE_OOF_ROWS:
        raise ValueError("unexpected B/14 extraction row count")
    total_bytes = sum(item["cache_bytes"] for item in entries)
    result = {
        "schema_version": "autoresearch-v19.b14-fold-cache-manifest-v1",
        "pass": True,
        "phase_authorization": authorization,
        "static_contract_sha256": v9.sha256_json(static),
        "spec_sha256": EXPECTED_SPEC_SHA256,
        "evaluator_sha256": EXPECTED_EVALUATOR_SHA256,
        "backbone_pin": pin,
        "backbone_provenance": provenance,
        "filesystem": filesystem,
        "folds": entries,
        "expected_tensor_bytes": EXPECTED_CACHE_TENSOR_BYTES,
        "total_cache_bytes": total_bytes,
        "under_two_gib": total_bytes < MAX_CACHE_BYTES,
        "duration_seconds_descriptive_only": time.time() - started,
        "operation_counts": {
            "source_image_reads": source_image_reads,
            "real_feature_extractions": source_image_reads,
            "candidate_optimizer_steps": 0,
            "candidate_predictions": 0,
            "candidate_scores": 0,
            "runtime_writes": 0,
        },
        "runtime_sha256_after": runtime_hashes(args),
        "runtime_unchanged": runtime_hashes(args) == EXPECTED_RUNTIME_SHA256,
        "final_test_read": False,
    }
    if not result["under_two_gib"] or not result["runtime_unchanged"]:
        raise ValueError("B/14 cache extraction violated a hard gate")
    write_json(args.cache_manifest, result)
    return result


def load_projection_checkpoint(path: Path, expected_sha256: str) -> dict[str, torch.Tensor]:
    if v9.sha256_file(path) != expected_sha256:
        raise ValueError(f"checkpoint SHA-256 mismatch: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError(f"checkpoint lacks model_state_dict: {path}")
    return state


def command_replay_c1(args: argparse.Namespace) -> dict[str, Any]:
    authorization = validate_phase_authorization(args, expected_phase="phase_2_extraction_and_c1_control")
    static = validate_static_contract(args)
    cache_manifest = validate_cache_manifest(args)
    summary = read_json(args.c1_summary)
    diagnostics = canonical_diagnostic_map(summary)
    canonical_rows = read_jsonl(args.c1_predictions)
    rows_by_key = {
        (int(row["outer_fold"]), int(row["seed"]), str(row["row_id"])): row
        for row in canonical_rows
    }
    if len(rows_by_key) != EXPECTED_PAIRED_ROWS:
        raise ValueError("canonical C1 replay lookup is not one-to-one")
    device = v9.resolve_device(args.device)
    mismatches: list[dict[str, Any]] = []
    replayed_rows = 0
    replay_entries: list[dict[str, Any]] = []
    for fold in EXPECTED_FOLDS:
        cache = load_canonical_cache(args, fold)
        for seed in EXPECTED_SEEDS:
            diagnostic = diagnostics[(fold, seed)]
            checkpoint_info = diagnostic["checkpoints"]["candidate"]
            checkpoint = resolve_artifact(checkpoint_info["path"])
            model = ProjectionHead(input_dim=384, embedding_dim=128)
            model.load_state_dict(load_projection_checkpoint(checkpoint, checkpoint_info["sha256"]))
            model = model.to(device)
            topk, effective_rank = v9.predict_arm(
                model,
                cache["train"],
                cache["oof"],
                device=device,
            )
            for index, row_id in enumerate(cache["oof"]["row_id"]):
                key = (fold, seed, str(row_id))
                expected = rows_by_key[key]["candidate_topk"]
                if topk[index] != expected:
                    mismatches.append(
                        {"fold": fold, "seed": seed, "row_id": str(row_id), "expected": expected, "actual": topk[index]}
                    )
                replayed_rows += 1
            replay_entries.append(
                {
                    "fold": fold,
                    "seed": seed,
                    "checkpoint_path": relative_path(checkpoint),
                    "checkpoint_sha256": checkpoint_info["sha256"],
                    "prediction_rows": len(topk),
                    "effective_rank": effective_rank,
                    "episode_plan_sha256": diagnostic["episode_plan_sha256"],
                    "ssl_plan_sha256": diagnostic["ssl"]["plan_sha256"],
                }
            )
            del model
    result = {
        "schema_version": "autoresearch-v19.c1-control-replay-audit-v1",
        "pass": not mismatches and replayed_rows == EXPECTED_PAIRED_ROWS,
        "phase_authorization": authorization,
        "static_contract_sha256": v9.sha256_json(static),
        "cache_manifest_path": relative_path(args.cache_manifest),
        "cache_manifest_sha256": v9.sha256_file(args.cache_manifest),
        "cache_manifest_verified": cache_manifest["pass"],
        "replayed_checkpoints": len(replay_entries),
        "replayed_prediction_rows": replayed_rows,
        "exact_topk_match_count": replayed_rows - len(mismatches),
        "mismatches": mismatches[:20],
        "entries": replay_entries,
        "operation_counts": {
            "source_image_reads": 0,
            "real_feature_extractions": 0,
            "candidate_optimizer_steps": 0,
            "candidate_predictions": 0,
            "control_predictions": replayed_rows,
            "candidate_scores": 0,
            "runtime_writes": 0,
        },
        "runtime_sha256_after": runtime_hashes(args),
        "runtime_unchanged": runtime_hashes(args) == EXPECTED_RUNTIME_SHA256,
        "final_test_read": False,
    }
    if not result["pass"] or not result["runtime_unchanged"]:
        raise ValueError("canonical C1 replay control failed")
    write_json(args.control_audit, result)
    return result


def same_process_c1_interlock(
    args: argparse.Namespace,
    *,
    device: torch.device,
) -> dict[str, Any]:
    summary = read_json(args.c1_summary)
    diagnostics = canonical_diagnostic_map(summary)
    canonical_rows = read_jsonl(args.c1_predictions)
    rows_by_key = {
        (int(row["outer_fold"]), int(row["seed"]), str(row["row_id"])): row
        for row in canonical_rows
    }
    if len(rows_by_key) != EXPECTED_PAIRED_ROWS:
        raise ValueError("same-process C1 lookup is not one-to-one")
    mismatches: list[dict[str, Any]] = []
    replayed_rows = 0
    checkpoint_count = 0
    entries: list[dict[str, Any]] = []
    for fold in EXPECTED_FOLDS:
        cache = load_canonical_cache(args, fold)
        for seed in EXPECTED_SEEDS:
            diagnostic = diagnostics[(fold, seed)]
            checkpoint_info = diagnostic["checkpoints"]["candidate"]
            checkpoint = resolve_artifact(checkpoint_info["path"])
            model = ProjectionHead(input_dim=384, embedding_dim=128)
            model.load_state_dict(
                load_projection_checkpoint(checkpoint, checkpoint_info["sha256"])
            )
            model = model.to(device)
            topk, effective_rank = v9.predict_arm(
                model,
                cache["train"],
                cache["oof"],
                device=device,
            )
            for index, row_id_value in enumerate(cache["oof"]["row_id"]):
                row_id = str(row_id_value)
                expected = rows_by_key[(fold, seed, row_id)]["candidate_topk"]
                if topk[index] != expected:
                    mismatches.append(
                        {
                            "fold": fold,
                            "seed": seed,
                            "row_id": row_id,
                            "expected": expected,
                            "actual": topk[index],
                        }
                    )
                replayed_rows += 1
            checkpoint_count += 1
            entries.append(
                {
                    "fold": fold,
                    "seed": seed,
                    "checkpoint_sha256": checkpoint_info["sha256"],
                    "prediction_rows": len(topk),
                    "effective_rank": effective_rank,
                }
            )
            del model
    result = {
        "schema_version": "autoresearch-v19.same-process-c1-interlock-v1",
        "pass": (
            checkpoint_count == 15
            and replayed_rows == EXPECTED_PAIRED_ROWS
            and not mismatches
        ),
        "replayed_checkpoints": checkpoint_count,
        "replayed_prediction_rows": replayed_rows,
        "exact_topk_match_count": replayed_rows - len(mismatches),
        "mismatches": mismatches[:20],
        "entries": entries,
        "candidate_optimizer_steps_before_interlock": 0,
        "candidate_gradient_before_interlock": False,
        "runtime_sha256_after": runtime_hashes(args),
        "runtime_unchanged": runtime_hashes(args) == EXPECTED_RUNTIME_SHA256,
        "final_test_read": False,
    }
    if not result["pass"] or not result["runtime_unchanged"]:
        raise ValueError("same-process C1 interlock failed before candidate gradient")
    return result


def torch_rng_sha256() -> str:
    digest = hashlib.sha256(torch.get_rng_state().numpy().tobytes())
    if torch.cuda.is_available():
        for state in torch.cuda.get_rng_state_all():
            digest.update(state.cpu().numpy().tobytes())
    return digest.hexdigest()


def aligned_candidate_initialization(
    seed: int,
) -> tuple[ProjectionHead, dict[str, str]]:
    configure_determinism(seed)
    candidate = ProjectionHead(input_dim=EMBED_DIM, embedding_dim=128)
    candidate_initial_sha256 = v9.state_dict_sha256(candidate)
    rng_after_candidate_head = torch_rng_sha256()
    configure_determinism(seed)
    canonical_head = ProjectionHead(input_dim=384, embedding_dim=128)
    canonical_initial_sha256 = v9.state_dict_sha256(canonical_head)
    rng_after_canonical_head = torch_rng_sha256()
    del canonical_head
    return candidate, {
        "candidate_initial_state_sha256": candidate_initial_sha256,
        "canonical_c1_initial_state_sha256": canonical_initial_sha256,
        "rng_after_candidate_head_sha256": rng_after_candidate_head,
        "rng_restored_to_canonical_post_head_sha256": rng_after_canonical_head,
    }


@contextmanager
def observe_vicreg_losses() -> Iterator[list[float]]:
    original = v9.vicreg_loss
    losses: list[float] = []

    def wrapped(*values: Any, **kwargs: Any) -> Any:
        terms = original(*values, **kwargs)
        losses.append(float(terms.total.detach().cpu()))
        return terms

    v9.vicreg_loss = wrapped
    try:
        yield losses
    finally:
        v9.vicreg_loss = original


def loss_trajectory(
    batch_losses: Sequence[float],
    *,
    train_rows: int,
    epochs: int,
    batch_size: int,
) -> list[float]:
    batches_per_epoch = sum(
        1
        for offset in range(0, train_rows, batch_size)
        if min(batch_size, train_rows - offset) >= 2
    )
    expected = batches_per_epoch * epochs
    if len(batch_losses) != expected:
        raise ValueError(
            f"VICReg observer count mismatch: expected {expected}, found {len(batch_losses)}"
        )
    return [
        float(np.mean(batch_losses[index * batches_per_epoch : (index + 1) * batches_per_epoch]))
        for index in range(epochs)
    ]


def embedding_snapshot(
    model: ProjectionHead,
    features: torch.Tensor,
    *,
    device: torch.device,
    limit: int = 4096,
) -> torch.Tensor:
    with torch.inference_mode():
        model = model.to(device)
        return v9.embed_in_batches(model, features[:limit], device)


def support_bin_three(count: int) -> str:
    if count <= 8:
        return "n_y_lte_8"
    if count <= 31:
        return "n_y_9_to_31"
    return "n_y_gte_32"


def support_bin_binary(count: int) -> str:
    return "n_y_lte_8" if count <= 8 else "n_y_gt_8"


def atomic_save_checkpoint(
    path: Path,
    *,
    fold: int,
    seed: int,
    model: ProjectionHead,
    initial_state_sha256: str,
    final_state_sha256: str,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "schema_version": "autoresearch-v19.candidate-checkpoint-v1",
        "arm": "v19-A",
        "fold": fold,
        "seed": seed,
        "input_dim": EMBED_DIM,
        "embedding_dim": 128,
        "initial_state_sha256": initial_state_sha256,
        "final_state_sha256": final_state_sha256,
        "model_state_dict": {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in model.state_dict().items()
        },
    }
    torch.save(payload, temporary, _use_new_zipfile_serialization=False)
    temporary.replace(path)
    return {
        "path": relative_path(path),
        "sha256": v9.sha256_file(path),
        "bytes": path.stat().st_size,
        "state_sha256": final_state_sha256,
    }


def run_candidate_fold_seed(
    cache: dict[str, Any],
    *,
    fold: int,
    seed: int,
    canonical_diagnostic: dict[str, Any],
    canonical_rows: Sequence[dict[str, Any]],
    device: torch.device,
    supervised_device: torch.device,
    checkpoint_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model, initialization = aligned_candidate_initialization(seed)
    initial_state_sha256 = initialization["candidate_initial_state_sha256"]
    initial_embedding = embedding_snapshot(
        model,
        cache["train"]["base_features"],
        device=device,
    )
    model = model.to(device)
    with observe_vicreg_losses() as batch_losses:
        ssl = v9.pretrain_vicreg(
            model,
            cache["train"],
            epochs=30,
            batch_size=256,
            learning_rate=3e-4,
            weight_decay=1e-4,
            seed=seed,
            device=device,
        )
    trajectory = loss_trajectory(
        batch_losses,
        train_rows=len(cache["train"]["row_id"]),
        epochs=30,
        batch_size=256,
    )
    if ssl["plan_sha256"] != canonical_diagnostic["ssl"]["plan_sha256"]:
        raise ValueError(f"SSL pair-plan hash differs from C1 for fold={fold}, seed={seed}")
    labels = torch.tensor(cache["train"]["class_label"], dtype=torch.long)
    episode_plan, episode_plan_sha256 = v9.build_episode_plan(
        labels,
        n_way=20,
        k_shot=3,
        q_queries=5,
        epochs=30,
        episodes_per_epoch=100,
        seed=v9.stable_seed("supervised-episodes", fold, seed),
    )
    if episode_plan_sha256 != canonical_diagnostic["episode_plan_sha256"]:
        raise ValueError(f"episode-plan hash differs from C1 for fold={fold}, seed={seed}")
    model = model.to(supervised_device)
    supervised = v9.train_supervised(
        model,
        cache["train"]["base_features"],
        episode_plan,
        learning_rate=1e-3,
        weight_decay=1e-4,
        temperature=0.1,
        warmup_epochs=5,
        rng_seed=v9.stable_seed("supervised-rng", fold, seed),
        device=supervised_device,
    )
    gradients_finite = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )
    weights_finite = all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters())
    model = model.to(device)
    candidate_topk, candidate_rank = v9.predict_arm(
        model,
        cache["train"],
        cache["oof"],
        device=device,
    )
    final_embedding = embedding_snapshot(
        model,
        cache["train"]["base_features"],
        device=device,
    )
    embedding_movement = float(torch.mean(torch.abs(final_embedding - initial_embedding)))
    final_state_sha256 = v9.state_dict_sha256(model)
    checkpoint = atomic_save_checkpoint(
        checkpoint_dir / f"fold-{fold:02d}-seed-{seed}-v19A.pt",
        fold=fold,
        seed=seed,
        model=model,
        initial_state_sha256=initial_state_sha256,
        final_state_sha256=final_state_sha256,
    )
    canonical_by_row = {str(row["row_id"]): row for row in canonical_rows}
    support = Counter(int(label) for label in cache["train"]["class_label"])
    records: list[dict[str, Any]] = []
    for index, row_id_value in enumerate(cache["oof"]["row_id"]):
        row_id = str(row_id_value)
        source = canonical_by_row[row_id]
        if int(source["outer_fold"]) != fold or int(source["seed"]) != seed:
            raise ValueError("canonical C1 row key mismatch")
        if (
            int(source["label"]) != int(cache["oof"]["class_label"][index])
            or str(source["provenance_component"]) != str(cache["oof"]["component_id"][index])
            or str(source["decoded_pixel_sha256"]) != str(cache["oof"]["decoded_pixel_sha256"][index])
        ):
            raise ValueError("canonical C1 row metadata differs from B/14 cache")
        train_support = support[int(source["label"])]
        records.append(
            {
                "row_id": row_id,
                "provenance_component": str(source["provenance_component"]),
                "decoded_pixel_sha256": str(source["decoded_pixel_sha256"]),
                "label": int(source["label"]),
                "class_name": str(source["class_name"]),
                "outer_fold": fold,
                "seed": seed,
                "recipe": "v19A-B14-v9-VICReg-episodic",
                "b0_topk": list(source["baseline_topk"]),
                "c1_topk": list(source["candidate_topk"]),
                "candidate_topk": candidate_topk[index],
                "episode_plan_sha256": episode_plan_sha256,
                "ssl_plan_sha256": ssl["plan_sha256"],
                "train_support": train_support,
                "support_bin_three": support_bin_three(train_support),
                "support_bin_binary": support_bin_binary(train_support),
            }
        )
    diagnostics = {
        "fold": fold,
        "seed": seed,
        "devices": {"ssl_and_prediction": str(device), "supervised": str(supervised_device)},
        "initialization": initialization,
        "initial_state_sha256": initial_state_sha256,
        "final_state_sha256": final_state_sha256,
        "candidate_final_state_differs_from_initial": final_state_sha256 != initial_state_sha256,
        "embedding_movement_mean_absolute": embedding_movement,
        "episode_plan_sha256": episode_plan_sha256,
        "episode_plan_matches_c1": True,
        "ssl": {
            **ssl,
            "plan_matches_c1": True,
            "epoch_loss_mean": trajectory,
            "start_epoch_loss": trajectory[0],
            "end_epoch_loss": trajectory[-1],
            "end_epoch_loss_below_start": trajectory[-1] < trajectory[0],
        },
        "supervised": supervised,
        "candidate_effective_rank": candidate_rank,
        "c1_effective_rank": float(canonical_diagnostic["candidate_effective_rank"]),
        "candidate_to_c1_effective_rank_ratio": (
            candidate_rank / float(canonical_diagnostic["candidate_effective_rank"])
            if canonical_diagnostic["candidate_effective_rank"]
            else 0.0
        ),
        "finite_gradients": gradients_finite,
        "finite_weights": weights_finite,
        "checkpoint": checkpoint,
    }
    return records, diagnostics


def comparison_records(
    records: Sequence[dict[str, Any]],
    *,
    baseline_field: str,
) -> list[dict[str, Any]]:
    return [
        {
            **row,
            "baseline_topk": list(row[baseline_field]),
            "candidate_topk": list(row["candidate_topk"]),
        }
        for row in records
    ]


def comparison_summary(
    records: Sequence[dict[str, Any]],
    *,
    baseline_field: str,
    bootstrap_replicates: int = 2000,
) -> dict[str, Any]:
    paired = comparison_records(records, baseline_field=baseline_field)
    seeds = sorted({int(row["seed"]) for row in paired})
    folds = sorted({int(row["outer_fold"]) for row in paired})
    overall = v9.accuracy_metrics(paired)
    seed_metrics = [
        {
            "seed": seed,
            **v9.accuracy_metrics([row for row in paired if int(row["seed"]) == seed]),
        }
        for seed in seeds
    ]
    fold_metrics = [
        {
            "fold": fold,
            **v9.accuracy_metrics([row for row in paired if int(row["outer_fold"]) == fold]),
        }
        for fold in folds
    ]
    bootstrap = v9.paired_component_bootstrap(
        paired,
        replicates=bootstrap_replicates,
        seed=v9.stable_seed("v19-bootstrap", baseline_field),
    )
    return {
        "baseline_field": baseline_field,
        "overall_metrics": overall,
        "seed_metrics": seed_metrics,
        "fold_metrics": fold_metrics,
        "bootstrap": bootstrap,
        "mcnemar": v9.exact_mcnemar(paired),
        "positive_seed_count": sum(item["delta_top1"] > 0.0 for item in seed_metrics),
        "minimum_fold_delta_top1": min(item["delta_top1"] for item in fold_metrics),
        "discordant_topk_rows": sum(
            row["baseline_topk"] != row["candidate_topk"] for row in paired
        ),
        "discordant_top1_rows": sum(
            row["baseline_topk"][0] != row["candidate_topk"][0] for row in paired
        ),
    }


def support_strata_diagnostics(
    records: Sequence[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for field in ("support_bin_three", "support_bin_binary"):
        values: dict[str, Any] = {}
        for value in sorted({str(row[field]) for row in records}):
            subset = [row for row in records if str(row[field]) == value]
            values[value] = {
                "prediction_rows": len(subset),
                "candidate_minus_c1": v9.accuracy_metrics(
                    comparison_records(subset, baseline_field="c1_topk")
                ),
                "candidate_minus_b0": v9.accuracy_metrics(
                    comparison_records(subset, baseline_field="b0_topk")
                ),
            }
        result[field] = values
    return result


def evaluate_candidate(
    records: Sequence[dict[str, Any]],
    diagnostics: Sequence[dict[str, Any]],
    cache_validations: Sequence[dict[str, Any]],
    *,
    control_audit: dict[str, Any],
) -> dict[str, Any]:
    if len(records) != EXPECTED_PAIRED_ROWS:
        raise ValueError("candidate prediction-row count mismatch")
    if len(diagnostics) != 15 or len(cache_validations) != 5:
        raise ValueError("candidate diagnostic/cache count mismatch")
    c1 = comparison_summary(records, baseline_field="c1_topk")
    b0 = comparison_summary(records, baseline_field="b0_topk")
    c1_overall = c1["overall_metrics"]
    b0_overall = b0["overall_metrics"]
    integrity_gates = {
        "paired_prediction_rows": len(records) == EXPECTED_PAIRED_ROWS,
        "paired_seed_count": len({int(row["seed"]) for row in records}) == 3,
        "outer_fold_count": len({int(row["outer_fold"]) for row in records}) == 5,
        "candidate_checkpoint_count": sum(
            bool(item["checkpoint"]["sha256"]) for item in diagnostics
        )
        == 15,
        "c1_same_process_topk_match_count": (
            int(control_audit.get("exact_topk_match_count", -1)) == EXPECTED_PAIRED_ROWS
        ),
        "b14_cache_count": len(cache_validations) == 5,
        "b14_cache_row_order_hash_match_count": sum(
            item["train_row_ids_sha256"] == item["canonical_train_row_ids_sha256"]
            and item["oof_row_ids_sha256"] == item["canonical_oof_row_ids_sha256"]
            for item in cache_validations
        )
        == 5,
        "b14_view_plan_hash_count": sum(bool(item["view_plan_sha256"]) for item in cache_validations)
        == 5,
        "b14_cache_byte_identical_readback_count": sum(
            item["byte_identical_readback"] is True for item in cache_validations
        )
        == 5,
        "oof_view_feature_count_is_zero": True,
        "episode_plan_sha256_matches_c1_count": sum(
            item["episode_plan_matches_c1"] is True for item in diagnostics
        )
        == 15,
        "ssl_pair_plan_sha256_matches_c1_count": sum(
            item["ssl"]["plan_matches_c1"] is True for item in diagnostics
        )
        == 15,
        "finite_losses_gradients_weights_and_embeddings": all(
            item["finite_gradients"]
            and item["finite_weights"]
            and math.isfinite(item["embedding_movement_mean_absolute"])
            and all(math.isfinite(value) for value in item["ssl"]["epoch_loss_mean"])
            for item in diagnostics
        ),
        "candidate_count_is_one": True,
        "final_test_unread": True,
        "runtime_unchanged": True,
        "automatic_promotion_disabled": True,
    }
    engagement_gates = {
        "vicreg_end_epoch_loss_below_start_epoch_count": sum(
            item["ssl"]["end_epoch_loss_below_start"] is True for item in diagnostics
        ),
        "candidate_final_state_differs_from_initial_count": sum(
            item["candidate_final_state_differs_from_initial"] is True
            for item in diagnostics
        ),
        "nonzero_embedding_movement_count": sum(
            item["embedding_movement_mean_absolute"] > 0.0 for item in diagnostics
        ),
        "candidate_vs_c1_discordant_prediction_rows": c1["discordant_topk_rows"],
    }
    engagement_pass = (
        engagement_gates["vicreg_end_epoch_loss_below_start_epoch_count"] == 15
        and engagement_gates["candidate_final_state_differs_from_initial_count"] == 15
        and engagement_gates["nonzero_embedding_movement_count"] == 15
        and engagement_gates["candidate_vs_c1_discordant_prediction_rows"] > 0
    )
    b0_gates = {
        "delta_top1_gte_0_01": b0_overall["delta_top1"] >= 0.01,
        "bootstrap_lower_gt_0": b0["bootstrap"]["delta_top1_95"][0] > 0.0,
        "positive_seed_count_gte_2": b0["positive_seed_count"] >= 2,
        "delta_macro_top1_gte_minus_0_005": b0_overall["delta_macro_top1"] >= -0.005,
        "delta_top3_gte_0": b0_overall["delta_top3"] >= 0.0,
        "minimum_fold_delta_top1_gte_minus_0_05": b0["minimum_fold_delta_top1"] >= -0.05,
    }
    c1_gates = {
        "delta_top1_gte_minus_0_005": c1_overall["delta_top1"] >= -0.005,
        "bootstrap_lower_gt_minus_0_01": c1["bootstrap"]["delta_top1_95"][0] > -0.01,
        "delta_top3_gte_minus_0_01": c1_overall["delta_top3"] >= -0.01,
        "delta_macro_top1_gte_minus_0_01": c1_overall["delta_macro_top1"] >= -0.01,
    }
    integrity_pass = all(integrity_gates.values())
    b0_pass = all(b0_gates.values())
    c1_pass = all(c1_gates.values())
    verdict, claim = v17.classify_verdict(
        integrity_pass=integrity_pass,
        engagement_pass=engagement_pass,
        b0_anchor_pass=b0_pass,
        c1_noninferiority_pass=c1_pass,
        c1_delta_top1=c1_overall["delta_top1"],
        c1_bootstrap_lower=c1["bootstrap"]["delta_top1_95"][0],
        c1_positive_seed_count=c1["positive_seed_count"],
    )
    return {
        "schema_version": "autoresearch-v19.candidate-evaluation-v1",
        "integrity_gates": integrity_gates,
        "integrity_pass": integrity_pass,
        "engagement_gates": engagement_gates,
        "engagement_pass": engagement_pass,
        "b0_mission_anchor_gates": b0_gates,
        "b0_mission_anchor_pass": b0_pass,
        "c1_noninferiority_gates": c1_gates,
        "c1_noninferiority_pass": c1_pass,
        "candidate_minus_b0": b0,
        "candidate_minus_c1": c1,
        "support_strata": support_strata_diagnostics(records),
        "minimum_candidate_to_c1_rank_ratio": min(
            item["candidate_to_c1_effective_rank_ratio"] for item in diagnostics
        ),
        "rank_ratio_below_0_8_diagnostic_only": any(
            item["candidate_to_c1_effective_rank_ratio"] < 0.8 for item in diagnostics
        ),
        "provisional_verdict_before_replay": verdict,
        "provisional_claim_before_replay": claim,
        "replay_pending": True,
        "score": c1_overall["delta_top1"],
        "promotion_eligible": False,
        "research_reference_only": True,
        "final_test_read": False,
    }


def validate_control_audit(args: argparse.Namespace) -> dict[str, Any]:
    audit = read_json(args.control_audit)
    if audit.get("schema_version") != "autoresearch-v19.c1-control-replay-audit-v1":
        raise ValueError("unexpected C1 control-audit schema")
    if audit.get("pass") is not True:
        raise ValueError("C1 control replay did not pass")
    if int(audit.get("replayed_checkpoints", -1)) != 15:
        raise ValueError("C1 control checkpoint count mismatch")
    if int(audit.get("exact_topk_match_count", -1)) != EXPECTED_PAIRED_ROWS:
        raise ValueError("C1 control exact-top-k count mismatch")
    if audit.get("runtime_unchanged") is not True or audit.get("final_test_read") is not False:
        raise ValueError("C1 control violated runtime/final-test constraints")
    if audit.get("cache_manifest_sha256") != v9.sha256_file(args.cache_manifest):
        raise ValueError("C1 control was not run against this cache manifest")
    return audit


def command_train_evaluate(args: argparse.Namespace) -> dict[str, Any]:
    authorization = validate_phase_authorization(args, expected_phase="phase_3_train_evaluate_replay")
    static = validate_static_contract(args)
    cache_manifest = validate_cache_manifest(args)
    control_audit = validate_control_audit(args)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("candidate output directory must be absent or empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = read_json(args.c1_summary)
    canonical_diagnostics = canonical_diagnostic_map(summary)
    canonical_rows = read_jsonl(args.c1_predictions)
    rows_by_fold_seed: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in canonical_rows:
        rows_by_fold_seed[(int(row["outer_fold"]), int(row["seed"]))].append(row)
    device = v9.resolve_device(args.device)
    same_process_control = same_process_c1_interlock(
        args,
        device=device,
    )
    supervised_device = v9.resolve_device(args.supervised_device)
    if supervised_device.type != "cpu":
        raise ValueError("v19 supervised training must remain on CPU")
    torch.set_num_threads(1)
    all_records: list[dict[str, Any]] = []
    all_diagnostics: list[dict[str, Any]] = []
    cache_validations: list[dict[str, Any]] = []
    optimizer_steps = 0
    for fold in EXPECTED_FOLDS:
        canonical_cache = load_canonical_cache(args, fold)
        cache_path = v9.cache_path(args.b14_cache_dir, fold, VIEWS, None)
        cache, validation = validate_b14_cache(
            cache_path,
            fold=fold,
            canonical=canonical_cache,
        )
        cache_validations.append(validation)
        ssl_batches_per_epoch = sum(
            1
            for offset in range(0, len(cache["train"]["row_id"]), 256)
            if min(256, len(cache["train"]["row_id"]) - offset) >= 2
        )
        for seed in EXPECTED_SEEDS:
            records, diagnostic = run_candidate_fold_seed(
                cache,
                fold=fold,
                seed=seed,
                canonical_diagnostic=canonical_diagnostics[(fold, seed)],
                canonical_rows=rows_by_fold_seed[(fold, seed)],
                device=device,
                supervised_device=supervised_device,
                checkpoint_dir=args.output_dir / "checkpoints",
            )
            all_records.extend(records)
            all_diagnostics.append(diagnostic)
            optimizer_steps += 30 * ssl_batches_per_epoch + 30 * 100
    all_records.sort(key=lambda row: (int(row["outer_fold"]), int(row["seed"]), str(row["row_id"])))
    all_diagnostics.sort(key=lambda item: (int(item["fold"]), int(item["seed"])))
    evaluation = evaluate_candidate(
        all_records,
        all_diagnostics,
        cache_validations,
        control_audit=same_process_control,
    )
    predictions_path = args.output_dir / "prediction_rows.jsonl"
    diagnostics_path = args.output_dir / "diagnostics.json"
    write_jsonl(predictions_path, all_records)
    write_json(diagnostics_path, {"diagnostics": all_diagnostics})
    result = {
        "schema_version": "autoresearch-v19.candidate-run-summary-v1",
        "pass": False,
        "decision_status": "pending_deterministic_replay",
        "phase_authorization": authorization,
        "static_contract_sha256": v9.sha256_json(static),
        "spec_sha256": EXPECTED_SPEC_SHA256,
        "evaluator_sha256": EXPECTED_EVALUATOR_SHA256,
        "cache_manifest_sha256": v9.sha256_file(args.cache_manifest),
        "control_audit_sha256": v9.sha256_file(args.control_audit),
        "same_process_c1_interlock": same_process_control,
        "cache_manifest_verified": cache_manifest["pass"],
        "control_audit_verified": control_audit["pass"],
        "prediction_rows_sha256": v9.sha256_file(predictions_path),
        "diagnostics_sha256": v9.sha256_file(diagnostics_path),
        "candidate_checkpoint_sha256": {
            f"{item['fold']}:{item['seed']}": item["checkpoint"]["sha256"]
            for item in all_diagnostics
        },
        "candidate_state_sha256": {
            f"{item['fold']}:{item['seed']}": item["final_state_sha256"]
            for item in all_diagnostics
        },
        "cache_validations": cache_validations,
        "diagnostics": all_diagnostics,
        "evaluation": evaluation,
        "operation_counts": {
            "source_image_reads": 0,
            "cache_reextractions": 0,
            "candidate_optimizer_steps": optimizer_steps,
            "candidate_predictions": len(all_records),
            "candidate_scores": len(all_records),
            "control_predictions_before_candidate_gradient": EXPECTED_PAIRED_ROWS,
            "runtime_writes": 0,
        },
        "artifact_paths": {
            "prediction_rows": relative_path(predictions_path),
            "diagnostics": relative_path(diagnostics_path),
        },
        "runtime_sha256_after": runtime_hashes(args),
        "runtime_unchanged": runtime_hashes(args) == EXPECTED_RUNTIME_SHA256,
        "final_test_read": False,
        "promotion_eligible": False,
    }
    if not result["runtime_unchanged"]:
        raise ValueError("runtime changed during candidate training")
    write_json(args.output_dir / "summary.json", result)
    return result


def normalized_replay_value(value: Any) -> Any:
    if isinstance(value, list):
        return [normalized_replay_value(item) for item in value]
    if not isinstance(value, dict):
        return value
    normalized: dict[str, Any] = {}
    for key, item in value.items():
        if key in {"artifact_paths", "diagnostics_sha256"}:
            continue
        if key == "path" and "sha256" in value:
            continue
        normalized[key] = normalized_replay_value(item)
    return normalized


def normalized_replay_sha256(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            normalized_replay_value(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def command_replay(args: argparse.Namespace) -> dict[str, Any]:
    validate_phase_authorization(args, expected_phase="phase_3_train_evaluate_replay")
    original_summary_path = args.output_dir / "summary.json"
    original_predictions_path = args.output_dir / "prediction_rows.jsonl"
    if not original_summary_path.is_file() or not original_predictions_path.is_file():
        raise FileNotFoundError("original candidate run is incomplete")
    if args.replay_dir.exists() and any(args.replay_dir.iterdir()):
        raise FileExistsError("replay directory must be absent or empty")
    original = read_json(original_summary_path)
    replay_args = copy.copy(args)
    replay_args.output_dir = args.replay_dir
    replay = command_train_evaluate(replay_args)
    replay_summary_path = args.replay_dir / "summary.json"
    replay_predictions_path = args.replay_dir / "prediction_rows.jsonl"
    original_prediction_sha256 = v9.sha256_file(original_predictions_path)
    replay_prediction_sha256 = v9.sha256_file(replay_predictions_path)
    checkpoint_match = (
        original["candidate_checkpoint_sha256"]
        == replay["candidate_checkpoint_sha256"]
    )
    state_match = original["candidate_state_sha256"] == replay["candidate_state_sha256"]
    original_normalized_sha256 = normalized_replay_sha256(original)
    replay_normalized_sha256 = normalized_replay_sha256(replay)
    gates = {
        "cache_reextraction_count": 0,
        "replay_candidate_state_count": len(replay["candidate_state_sha256"]),
        "normalized_prediction_rows_byte_identical": (
            original_prediction_sha256 == replay_prediction_sha256
        ),
        "candidate_checkpoints_byte_identical": checkpoint_match,
        "candidate_states_identical": state_match,
        "normalized_audit_byte_identical": (
            original_normalized_sha256 == replay_normalized_sha256
        ),
    }
    replay_pass = (
        gates["cache_reextraction_count"] == 0
        and gates["replay_candidate_state_count"] == 15
        and gates["normalized_prediction_rows_byte_identical"]
        and gates["candidate_checkpoints_byte_identical"]
        and gates["candidate_states_identical"]
        and gates["normalized_audit_byte_identical"]
    )
    audit = {
        "schema_version": "autoresearch-v19.deterministic-replay-audit-v1",
        "pass": replay_pass,
        "normalization_contract": REPLAY_NORMALIZATION_CONTRACT,
        "gates": gates,
        "original": {
            "summary_path": relative_path(original_summary_path),
            "summary_sha256": v9.sha256_file(original_summary_path),
            "normalized_summary_sha256": original_normalized_sha256,
            "prediction_rows_sha256": original_prediction_sha256,
        },
        "replay": {
            "summary_path": relative_path(replay_summary_path),
            "summary_sha256": v9.sha256_file(replay_summary_path),
            "normalized_summary_sha256": replay_normalized_sha256,
            "prediction_rows_sha256": replay_prediction_sha256,
        },
        "runtime_sha256_after": runtime_hashes(args),
        "runtime_unchanged": runtime_hashes(args) == EXPECTED_RUNTIME_SHA256,
        "final_test_read": False,
        "promotion_eligible": False,
    }
    write_json(args.output_dir / "replay-audit.json", audit)
    if not replay_pass or not audit["runtime_unchanged"]:
        raise ValueError("v19 deterministic replay failed")
    final_evaluation = copy.deepcopy(original["evaluation"])
    verdict = final_evaluation.pop("provisional_verdict_before_replay")
    claim = final_evaluation.pop("provisional_claim_before_replay")
    final_evaluation["replay_pending"] = False
    final_evaluation["verdict"] = verdict
    final_evaluation["claim"] = claim
    final_evaluation["replay_pass"] = True
    final_evaluation["pass"] = verdict in {"supported_strong", "supported_reference"}
    final = {
        **original,
        "schema_version": "autoresearch-v19.final-summary-v1",
        "pass": final_evaluation["pass"],
        "decision_status": "complete_research_reference_only",
        "evaluation": final_evaluation,
        "replay_audit_path": relative_path(args.output_dir / "replay-audit.json"),
        "replay_audit_sha256": v9.sha256_file(args.output_dir / "replay-audit.json"),
        "runtime_sha256_after": runtime_hashes(args),
        "runtime_unchanged": True,
        "final_test_read": False,
        "promotion_eligible": False,
    }
    write_json(args.output_dir / "final-summary.json", final)
    return final


def command_validate_contract(args: argparse.Namespace) -> dict[str, Any]:
    result = validate_static_contract(args)
    if args.output is not None:
        write_json(args.output, result)
    return result


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--evaluator", type=Path, default=DEFAULT_EVALUATOR)
    parser.add_argument("--freeze-audit", type=Path, default=DEFAULT_FREEZE_AUDIT)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--b14-manifest", type=Path, default=DEFAULT_B14_MANIFEST)
    parser.add_argument("--source-inventory", type=Path, default=DEFAULT_SOURCE_INVENTORY)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_CORPUS_MANIFEST)
    parser.add_argument("--audit", type=Path, default=DEFAULT_COLLECTION_AUDIT)
    parser.add_argument("--c1-summary", type=Path, default=DEFAULT_C1_SUMMARY)
    parser.add_argument("--c1-predictions", type=Path, default=DEFAULT_C1_PREDICTIONS)
    parser.add_argument("--c1-cache-dir", type=Path, default=DEFAULT_C1_CACHE_DIR)
    parser.add_argument("--b14-cache-dir", type=Path, default=DEFAULT_B14_CACHE_DIR)
    parser.add_argument("--cache-manifest", type=Path, default=DEFAULT_CACHE_MANIFEST)
    parser.add_argument("--control-audit", type=Path, default=DEFAULT_CONTROL_AUDIT)
    parser.add_argument("--runtime-projection", type=Path, default=DEFAULT_RUNTIME_PROJECTION)
    parser.add_argument("--runtime-prototypes", type=Path, default=DEFAULT_RUNTIME_PROTOTYPES)
    parser.add_argument("--runtime-config", type=Path, default=DEFAULT_RUNTIME_CONFIG)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--supervised-device", default="cpu")
    parser.add_argument("--num-workers", type=int, default=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-contract")
    add_common_arguments(validate)
    validate.add_argument("--output", type=Path)
    validate.set_defaults(handler=command_validate_contract)

    precompute = subparsers.add_parser("precompute")
    add_common_arguments(precompute)
    precompute.set_defaults(handler=command_precompute)

    control = subparsers.add_parser("replay-c1")
    add_common_arguments(control)
    control.set_defaults(handler=command_replay_c1)

    train = subparsers.add_parser("train-evaluate")
    add_common_arguments(train)
    train.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    train.add_argument("--replay-dir", type=Path, default=DEFAULT_REPLAY_DIR)
    train.set_defaults(handler=command_train_evaluate)

    replay = subparsers.add_parser("replay")
    add_common_arguments(replay)
    replay.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    replay.add_argument("--replay-dir", type=Path, default=DEFAULT_REPLAY_DIR)
    replay.set_defaults(handler=command_replay)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = args.handler(args)
    print(canonical_json(result), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


#!/usr/bin/env python3
"""Zero-candidate feasibility gate for the bounded Elements v19-A experiment."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
BACKEND = ROOT / "backend"
for import_path in (SCRIPTS_DIR, BACKEND):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import autoresearch_backbone_screen_v10 as backbone_screen  # noqa: E402
import autoresearch_self_supervised_v9 as v9  # noqa: E402
from codex_pipeline.models.projection_head import ProjectionHead  # noqa: E402


RUN = ROOT / ".omc/autoresearch/elements-baseline-replacement/runs/20260804-vicreg-backbone-renomination-v19"
V10_RUN = ROOT / ".omc/autoresearch/elements-baseline-replacement/runs/20260803-self-supervised-v10"
V18_RUN = ROOT / ".omc/autoresearch/elements-baseline-replacement/runs/20260804-discriminative-readout-v18"
COUNCIL_TURN = ROOT / ".omx/argos-council/elements-model-improvement/turns/043"

DEFAULT_BACKBONE_MANIFEST = V10_RUN / "inputs/dinov2-vitb14-local-manifest.json"
DEFAULT_SOURCE_INVENTORY = V18_RUN / "specs/iteration-0003-source-inventory.jsonl"
DEFAULT_OUTPUT = RUN / "iteration-0001-feasibility-audit.json"
DEFAULT_MINI_CACHE = RUN / "iteration-0001/feasibility-mini-cache.npy"
DEFAULT_RUNTIME_PROJECTION = BACKEND / "codex_model/weights/projection.pt"
DEFAULT_RUNTIME_PROTOTYPES = BACKEND / "codex_model/weights/prototypes.pt"
DEFAULT_RUNTIME_CONFIG = BACKEND / "codex_model/config.json"

EXPECTED_BACKBONE = "dinov2_vitb14"
EXPECTED_EMBED_DIM = 768
EXPECTED_ROWS = 9990
EXPECTED_FOLDS = (1, 2, 3, 4, 5)
EXPECTED_VIEWS_PER_TRAIN_ROW = 9
EXPECTED_CACHE_BYTES = 552_407_040
MAX_CACHE_BYTES = 2 * 1024**3
MIN_MEMORY_MARGIN_FRACTION = 0.15
BATCH_GRID = (8, 16, 32)

EXPECTED_HASHES = {
    "council_synthesis": (
        COUNCIL_TURN / "synthesis.md",
        "0379abee5893d619069eae42753dfae80d59682cf66517c7cc1d23452b5f56c0",
    ),
    "history_correction": (
        COUNCIL_TURN / "branch-history-correction.json",
        "aa9c8d75c6ad5c454e39409a679c56ea8bb6049d0b15617d49bbbcd5b63ceeca",
    ),
    "v10_decision_log": (
        V10_RUN / "decision-log.md",
        "a5e7fb2560b2e636aed4f711d4aedfbb2d7728eda118253f4e5fadf3cfe4fc78",
    ),
    "v10_raw_summary": (
        V10_RUN / "iteration-0002/results/summary.json",
        "13c3d1063e81175e39744c58c467665fe6a0d1061e127229b6026e4c79a8b1bf",
    ),
    "v10_learned_summary": (
        V10_RUN / "iteration-0003/results/summary.json",
        "6cc3bbfff26036c298c7dab9ede3840bb2041a58a2e46301225a8ec8aa80ad77",
    ),
    "v10_learned_spec": (
        V10_RUN / "specs/iteration-0003.json",
        "70b270bc6cbb16d574e69fc5e25950b9919819ae396c2b2025f4c955be5b119b",
    ),
    "b14_manifest": (
        DEFAULT_BACKBONE_MANIFEST,
        "ea73c8b9d2cc8a81a48183a43516fc76ad6b2bd5694442229d9b03ce3774e4d5",
    ),
    "source_inventory": (
        DEFAULT_SOURCE_INVENTORY,
        "d21ccbc123d454db773eb31fbc2fe3cf225c21c79534a142cf0b3d34340a87d2",
    ),
    "runtime_projection": (
        DEFAULT_RUNTIME_PROJECTION,
        "0ad6ce348eaf44c8844415a769b048b8f155017befa93f16254b1263b9241210",
    ),
    "runtime_prototypes": (
        DEFAULT_RUNTIME_PROTOTYPES,
        "ffc52f03696f03780a01302708b53e319dc24069187a6fd1a2927ea537b1cc54",
    ),
    "runtime_config": (
        DEFAULT_RUNTIME_CONFIG,
        "f2150a97ec7eee50a42ac1a108ae4bc092d46a6b8e4a2d91de59349560fbf18b",
    ),
}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def verify_file_hashes() -> dict[str, dict[str, Any]]:
    verified: dict[str, dict[str, Any]] = {}
    for name, (path, expected) in EXPECTED_HASHES.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing frozen input {name}: {path}")
        actual = v9.sha256_file(path)
        if actual != expected:
            raise ValueError(f"frozen input hash mismatch for {name}: {actual}")
        verified[name] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": actual,
            "bytes": path.stat().st_size,
        }
    return verified


def projection_architecture() -> dict[str, Any]:
    head = ProjectionHead(input_dim=EXPECTED_EMBED_DIM, embedding_dim=128)
    layers = list(head.net)
    architecture = [
        layers[0].in_features,
        layers[0].out_features,
        layers[-1].out_features,
    ]
    return {
        "architecture": "->".join(str(value) for value in architecture),
        "parameter_count": sum(parameter.numel() for parameter in head.parameters()),
        "pass": architecture == [768, 768, 128],
    }


def inventory_cache_estimate(path: Path) -> dict[str, Any]:
    fold_counts: Counter[int] = Counter()
    row_ids: set[str] = set()
    row_count = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            row_id = str(row["row_id"])
            if row_id in row_ids:
                raise ValueError(f"duplicate source inventory row_id: {row_id}")
            row_ids.add(row_id)
            fold_counts[int(row["fold"])] += 1
            row_count += 1
    if row_count != EXPECTED_ROWS or tuple(sorted(fold_counts)) != EXPECTED_FOLDS:
        raise ValueError(f"unexpected source inventory shape: {row_count}, {fold_counts}")
    per_fold: dict[str, dict[str, int]] = {}
    total_train_rows = 0
    for fold in EXPECTED_FOLDS:
        train_rows = row_count - fold_counts[fold]
        total_train_rows += train_rows
        per_fold[str(fold)] = {
            "train_rows": train_rows,
            "held_out_rows_not_accessed": fold_counts[fold],
            "float16_cls_bytes": (
                train_rows
                * EXPECTED_VIEWS_PER_TRAIN_ROW
                * EXPECTED_EMBED_DIM
                * np.dtype(np.float16).itemsize
            ),
        }
    estimated = (
        total_train_rows
        * EXPECTED_VIEWS_PER_TRAIN_ROW
        * EXPECTED_EMBED_DIM
        * np.dtype(np.float16).itemsize
    )
    return {
        "inventory_rows": row_count,
        "fold_counts": {str(key): fold_counts[key] for key in EXPECTED_FOLDS},
        "total_fold_local_train_rows": total_train_rows,
        "views_per_train_row_including_base": EXPECTED_VIEWS_PER_TRAIN_ROW,
        "embedding_dim": EXPECTED_EMBED_DIM,
        "dtype": "float16",
        "per_fold": per_fold,
        "estimated_cache_bytes": estimated,
        "maximum_cache_bytes": MAX_CACHE_BYTES,
        "filesystem_required": "WSL ext4",
        "drvfs_forbidden": True,
        "large_memmap_forbidden": True,
        "pass": estimated == EXPECTED_CACHE_BYTES and estimated < MAX_CACHE_BYTES,
    }


def choose_batch(probes: Sequence[dict[str, Any]]) -> int | None:
    eligible = [
        int(probe["batch_size"])
        for probe in probes
        if probe.get("status") == "pass"
        and float(probe.get("free_memory_fraction_during_forward", 0.0))
        >= MIN_MEMORY_MARGIN_FRACTION
    ]
    return max(eligible, default=None)


@torch.inference_mode()
def probe_cuda_batches(
    backbone: torch.nn.Module,
    device: torch.device,
    batch_grid: Sequence[int] = BATCH_GRID,
) -> tuple[list[dict[str, Any]], int | None, np.ndarray | None]:
    if device.type != "cuda":
        raise RuntimeError("v19 feasibility requires CUDA")
    total_memory = int(torch.cuda.get_device_properties(device).total_memory)
    probes: list[dict[str, Any]] = []
    mini_features: np.ndarray | None = None
    for batch_size in batch_grid:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        generator = torch.Generator(device="cpu").manual_seed(20260804 + batch_size)
        host = torch.randn(batch_size, 3, 224, 224, generator=generator)
        try:
            inputs = host.to(device)
            torch.cuda.synchronize(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                features = backbone(inputs)
            torch.cuda.synchronize(device)
            if tuple(features.shape) != (batch_size, EXPECTED_EMBED_DIM):
                raise ValueError(f"unexpected B/14 feature shape: {tuple(features.shape)}")
            if not bool(torch.isfinite(features).all()):
                raise FloatingPointError("non-finite B/14 feasibility features")
            free_during, reported_total = torch.cuda.mem_get_info(device)
            if int(reported_total) != total_memory:
                raise RuntimeError("CUDA total-memory reports disagree")
            peak_allocated = int(torch.cuda.max_memory_allocated(device))
            probe = {
                "batch_size": batch_size,
                "status": "pass",
                "peak_allocated_bytes": peak_allocated,
                "free_memory_bytes_during_forward": int(free_during),
                "total_memory_bytes": total_memory,
                "free_memory_fraction_during_forward": float(free_during / total_memory),
                "feature_shape": list(features.shape),
                "finite": True,
            }
            probes.append(probe)
            if mini_features is None:
                mini_features = features.detach().cpu().to(torch.float16).numpy()
            del features, inputs, host
        except torch.cuda.OutOfMemoryError as exc:
            probes.append(
                {
                    "batch_size": batch_size,
                    "status": "oom",
                    "exception": f"{type(exc).__name__}: {exc}",
                    "total_memory_bytes": total_memory,
                }
            )
            torch.cuda.empty_cache()
            break
    return probes, choose_batch(probes), mini_features


def seal_mini_cache(path: Path, features: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    if partial.exists() or path.exists():
        raise FileExistsError(f"refusing to overwrite feasibility cache: {path}")
    with partial.open("wb") as handle:
        np.save(handle, np.asarray(features, dtype=np.float16), allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)
    first_hash = v9.sha256_file(path)
    loaded = np.load(path, allow_pickle=False)
    second_hash = v9.sha256_file(path)
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": first_hash,
        "bytes": path.stat().st_size,
        "shape": list(loaded.shape),
        "dtype": str(loaded.dtype),
        "readback_sha256": second_hash,
        "byte_identical_readback": first_hash == second_hash,
        "value_identical_readback": bool(np.array_equal(loaded, features)),
        "synthetic_only": True,
        "oof_rows_accessed": 0,
    }


def runtime_hashes(args: argparse.Namespace) -> dict[str, str]:
    return {
        "projection": v9.sha256_file(args.runtime_projection),
        "prototypes": v9.sha256_file(args.runtime_prototypes),
        "config": v9.sha256_file(args.runtime_config),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    frozen_inputs = verify_file_hashes()
    pin = backbone_screen.validate_backbone_manifest(args.backbone_manifest)
    if pin["manifest_sha256"] != EXPECTED_HASHES["b14_manifest"][1]:
        raise ValueError("B/14 validated manifest hash mismatch")
    cache_estimate = inventory_cache_estimate(args.source_inventory)
    architecture = projection_architecture()
    runtime_before = runtime_hashes(args)
    device = v9.resolve_device(args.device)
    if device.type != "cuda":
        raise RuntimeError("v19 feasibility requires an explicit CUDA device")
    torch.cuda.empty_cache()
    backbone, provenance = v9.load_backbone(EXPECTED_BACKBONE, device, args.backbone_manifest)
    backbone.requires_grad_(False).eval()
    if any(parameter.requires_grad for parameter in backbone.parameters()):
        raise RuntimeError("B/14 backbone is not fully frozen")
    probes, selected_batch, mini_features = probe_cuda_batches(backbone, device)
    if mini_features is None:
        raise RuntimeError("no successful CUDA forward available for mini-cache")
    mini_cache = seal_mini_cache(args.mini_cache, mini_features)
    runtime_after = runtime_hashes(args)
    operation_counts = {
        "synthetic_cuda_forward_batches": len(
            [probe for probe in probes if probe["status"] == "pass"]
        ),
        "source_image_reads": 0,
        "candidate_feature_extractions": 0,
        "candidate_optimizer_steps": 0,
        "candidate_predictions": 0,
        "candidate_scores": 0,
        "oof_rows_accessed": 0,
        "final_test_reads": 0,
        "runtime_writes": 0,
    }
    gates = {
        "frozen_inputs_verified": True,
        "official_b14_pin_verified": pin["weights_sha256_verified"]
        == backbone_screen.EXPECTED_WEIGHTS_SHA256,
        "backbone_frozen": True,
        "projection_architecture_dimension_forced": architecture["pass"],
        "cache_estimate_bounded_ext4": cache_estimate["pass"],
        "cuda_batch_with_15_percent_margin": selected_batch is not None,
        "mini_cache_byte_identical": mini_cache["byte_identical_readback"],
        "zero_candidate_information": all(
            operation_counts[key] == 0
            for key in (
                "source_image_reads",
                "candidate_feature_extractions",
                "candidate_optimizer_steps",
                "candidate_predictions",
                "candidate_scores",
                "oof_rows_accessed",
                "final_test_reads",
                "runtime_writes",
            )
        ),
        "runtime_unchanged": runtime_before == runtime_after,
    }
    audit = {
        "schema_version": "autoresearch-vicreg-backbone-renomination-v19.feasibility-v1",
        "iteration": 1,
        "factor": (
            "explicit one-time renomination of frozen DINOv2-B/14 under the exact "
            "v9 VICReg plus episodic-readout mechanism"
        ),
        "pass": all(gates.values()),
        "gates": gates,
        "frozen_inputs": frozen_inputs,
        "backbone_pin": pin,
        "backbone_loader_provenance": provenance,
        "projection_head": architecture,
        "cache_estimate": cache_estimate,
        "cuda": {
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device),
            "batch_grid": list(BATCH_GRID),
            "minimum_memory_margin_fraction": MIN_MEMORY_MARGIN_FRACTION,
            "probes": probes,
            "selected_extraction_batch": selected_batch,
            "training_batch_note": (
                "The frozen-backbone extraction batch is distinct from the v9 "
                "VICReg projection training batch of 256 cached CLS rows."
            ),
        },
        "mini_cache": mini_cache,
        "operation_counts": operation_counts,
        "runtime_before": runtime_before,
        "runtime_after": runtime_after,
        "final_test_read": False,
        "runtime_promotion": False,
        "next_action": (
            "Return to Argos Council before freezing the v19 runner/spec/evaluator "
            "or performing any candidate extraction, gradient, prediction, or score."
        ),
    }
    write_json(args.output, audit)
    return audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backbone-manifest", type=Path, default=DEFAULT_BACKBONE_MANIFEST)
    parser.add_argument("--source-inventory", type=Path, default=DEFAULT_SOURCE_INVENTORY)
    parser.add_argument("--mini-cache", type=Path, default=DEFAULT_MINI_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--runtime-projection", type=Path, default=DEFAULT_RUNTIME_PROJECTION)
    parser.add_argument("--runtime-prototypes", type=Path, default=DEFAULT_RUNTIME_PROTOTYPES)
    parser.add_argument("--runtime-config", type=Path, default=DEFAULT_RUNTIME_CONFIG)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    audit = run(args)
    print(json.dumps(audit, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if audit["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

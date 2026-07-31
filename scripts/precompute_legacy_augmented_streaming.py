#!/usr/bin/env python3
"""Precompute the legacy augmented Elements cache without retaining images.

The original legacy script first materializes every augmented 224x224 tensor
in RAM.  At the historic 29,551-row multiplier this requires more memory than
the training workstation provides.  This implementation preserves row order
and the adaptive-multiplier formula while forwarding bounded batches through
DINOv2 as soon as they are ready.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
LEGACY_ROOT = REPO_ROOT / "_legacy" / "frontend_integration"
sys.path.insert(0, str(LEGACY_ROOT))

from codex_pipeline.data.augmentation import get_train_transform_numpy
from codex_pipeline.data.metadata import filter_classes, load_metadata
from codex_pipeline.scripts.precompute_augmented import (
    compute_adaptive_multiplier,
    load_and_augment,
    load_clean,
)


def configure_seed(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    workspace_config = os.environ["CUBLAS_WORKSPACE_CONFIG"]
    if workspace_config not in {":4096:8", ":16:8"}:
        raise RuntimeError(
            f"CUBLAS_WORKSPACE_CONFIG must be deterministic, got {workspace_config!r}"
        )
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def git_revision(path: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def flush_batch(
    backbone: torch.nn.Module | None,
    device: torch.device,
    tensors: list[torch.Tensor],
    output: list[torch.Tensor],
    backbone_loader,
) -> torch.nn.Module | None:
    if not tensors:
        return backbone
    if backbone is None:
        backbone = backbone_loader()
    batch = torch.stack(tensors).to(device)
    with torch.inference_mode():
        output.append(backbone(batch).cpu())
    tensors.clear()
    return backbone


def _normalized_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def load_allowed_paths_cache(cache_path: Path) -> tuple[list[str], int]:
    try:
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(cache_path, map_location="cpu")
    if not isinstance(payload, dict) or "image_paths" not in payload:
        raise RuntimeError(f"{cache_path} is missing image_paths")
    image_paths = payload["image_paths"]
    if not isinstance(image_paths, list):
        raise RuntimeError(f"{cache_path} image_paths must be a list")
    allowed_paths = [str(path) for path in image_paths]
    missing = [path for path in allowed_paths if not _normalized_path(path).exists()]
    if missing:
        preview = ", ".join(missing[:5])
        suffix = "" if len(missing) <= 5 else f" (+{len(missing) - 5} more)"
        raise RuntimeError(f"allowed paths cache references missing files: {preview}{suffix}")
    return allowed_paths, len(allowed_paths)


def build_cache(args: argparse.Namespace) -> dict[str, Any]:
    with args.config.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    configure_seed(args.seed)
    device = torch.device(args.device)
    image_size = int(cfg["data"]["image_size"])
    metadata = filter_classes(
        load_metadata(cfg["paths"]["metadata_csv"]),
        min_images=int(cfg["data"]["min_images_per_class"]),
    )

    allowed_source_count: int | None = None
    if args.allowed_paths_cache is not None:
        allowed_paths, allowed_source_count = load_allowed_paths_cache(args.allowed_paths_cache)
        allowed_set = {_normalized_path(path) for path in allowed_paths}
        metadata = metadata[
            metadata["image_path"].map(_normalized_path).isin(allowed_set)
        ].reset_index(drop=True)

    class_names: dict[int, str] = {}
    class_counts: defaultdict[int, int] = defaultdict(int)
    for _, row in metadata.iterrows():
        label = int(row["class_label"])
        class_names[label] = str(row["element_name"])
        class_counts[label] += 1

    multipliers = (
        compute_adaptive_multiplier(class_counts, args.multiplier)
        if args.adaptive
        else {label: args.multiplier for label in class_counts}
    )
    expected = len(metadata) + sum(
        multipliers[int(row["class_label"])] for _, row in metadata.iterrows()
    )
    print(
        f"Elements={len(metadata)} classes={len(class_counts)} "
        f"expected_features={expected}",
        flush=True,
    )

    transform = get_train_transform_numpy(cfg, image_size)
    if not hasattr(transform, "set_random_seed"):
        raise RuntimeError("training transform must expose set_random_seed for deterministic streaming cache precompute")
    transform.set_random_seed(args.seed)

    def load_backbone() -> torch.nn.Module:
        backbone = torch.hub.load(
            "facebookresearch/dinov2",
            cfg["model"]["backbone"],
            pretrained=True,
        ).to(device)
        backbone.eval()
        return backbone

    backbone: torch.nn.Module | None = None

    features: list[torch.Tensor] = []
    labels: list[int] = []
    image_paths: list[str] = []
    is_augmented: list[bool] = []
    pending: list[torch.Tensor] = []
    started = time.perf_counter()

    def append(tensor: torch.Tensor, label: int, path: str, augmented: bool) -> None:
        pending.append(tensor)
        labels.append(label)
        image_paths.append(path)
        is_augmented.append(augmented)

    for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Elements"):
        path = str(row["image_path"])
        label = int(row["class_label"])
        try:
            append(load_clean(path, image_size), label, path, False)
        except Exception as exc:
            raise RuntimeError(f"load_clean failed for {path} (label={label})") from exc
        for _ in range(multipliers[label]):
            try:
                append(load_and_augment(path, transform, image_size), label, path, True)
            except Exception as exc:
                raise RuntimeError(
                    f"augmentation failed for {path} (label={label})"
                ) from exc
        if len(pending) >= args.batch_size:
            backbone = flush_batch(backbone, device, pending, features, load_backbone)

    backbone = flush_batch(backbone, device, pending, features, load_backbone)
    feature_tensor = torch.cat(features, dim=0)
    if len(feature_tensor) != expected:
        raise RuntimeError(
            f"expected {expected} features, generated {len(feature_tensor)}"
        )

    return {
        "features": feature_tensor,
        "labels": torch.tensor(labels, dtype=torch.long),
        "image_paths": image_paths,
        "is_augmented": is_augmented,
        "class_names": class_names,
        "backbone": cfg["model"]["backbone"],
        "hidden_dim": int(backbone.embed_dim),
        "image_size": image_size,
        "multiplier": args.multiplier,
        "adaptive": args.adaptive,
        "allowed_source_count": allowed_source_count,
        "seed": args.seed,
        "streaming": True,
        "elapsed_seconds": time.perf_counter() - started,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--multiplier", type=int, default=5)
    parser.add_argument(
        "--adaptive",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allowed-paths-cache", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = build_cache(args)
    output = args.output_dir / "features_aug.pt"
    torch.save(payload, output)
    provenance = {
        "schema_version": "legacy-augmented-cache.v1",
        "output": str(output),
        "feature_count": len(payload["labels"]),
        "class_count": len(payload["class_names"]),
        "multiplier": payload["multiplier"],
        "adaptive": payload["adaptive"],
        "allowed_source_count": payload["allowed_source_count"],
        "seed": payload["seed"],
        "legacy_revision": git_revision(LEGACY_ROOT),
    }
    (args.output_dir / "features_aug.pt.prov.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(provenance, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

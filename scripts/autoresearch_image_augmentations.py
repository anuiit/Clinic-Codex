#!/usr/bin/env python3
"""Leakage-safe image-augmentation sweep for the Elements candidate.

The four augmentation profiles are applied exactly once to each source-disjoint
external training image. DINOv2 features are cached per profile, then appended
to the safe legacy features and the clean external features. Only the 286
historical prototypes are optimized, using the frozen v5 iteration-1 recipe.

Development selects one profile by top-1 then macro top-1. The preserved v5
sealed test is materialized once, after selection, and evaluates historical,
v5-iteration-1, and the selected augmentation candidate on identical rows.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance

SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_ROOT.parent
sys.path.insert(0, str(SCRIPT_ROOT))

import autoresearch_external_model_v5 as v5

IMAGE_SIZE = 224
BASE_SEED = 7201
PROFILE_NAMES = (
    "control_current",
    "geom_mild",
    "bbox_pad_crop",
    "combo_sobre",
)
DEFAULT_RUN_DIR = (
    REPO_ROOT
    / ".omc/autoresearch/elements-baseline-replacement/runs"
    / "image-augmentations-v1"
)
DEFAULT_V5_CHECKPOINT = (
    REPO_ROOT
    / ".omc/autoresearch/elements-baseline-replacement/runs"
    / "20260729-external-proxy-v5/checkpoints/iteration-0001.pt"
)
DINO_CHECKPOINT_NAME = "dinov2_vits14_pretrain.pth"


def _white_square(image: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    height, width = image.shape[:2]
    scale = size / max(width, height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = cv2.resize(
        image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR
    )
    canvas = np.full((size, size, 3), 255, dtype=np.uint8)
    offset_x = (size - resized_width) // 2
    offset_y = (size - resized_height) // 2
    canvas[
        offset_y : offset_y + resized_height,
        offset_x : offset_x + resized_width,
    ] = resized
    return canvas


def _affine(
    image: np.ndarray,
    rng: np.random.Generator,
    *,
    scale_range: tuple[float, float],
    rotation_limit: float,
    translation_limit: float,
) -> np.ndarray:
    height, width = image.shape[:2]
    scale = float(rng.uniform(*scale_range))
    angle = float(rng.uniform(-rotation_limit, rotation_limit))
    matrix = cv2.getRotationMatrix2D(
        ((width - 1) / 2, (height - 1) / 2), angle, scale
    )
    matrix[0, 2] += float(rng.uniform(-translation_limit, translation_limit)) * width
    matrix[1, 2] += float(rng.uniform(-translation_limit, translation_limit)) * height
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def _perspective_white(
    image: np.ndarray,
    rng: np.random.Generator,
    scale_range: tuple[float, float],
) -> np.ndarray:
    """Apply perspective while explicitly keeping an RGB-white background."""

    height, width = image.shape[:2]
    amount = float(rng.uniform(*scale_range)) * min(width, height)
    source = np.float32(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
    )
    destination = source + rng.uniform(-amount, amount, source.shape).astype(
        np.float32
    )
    matrix = cv2.getPerspectiveTransform(source, destination)
    return cv2.warpPerspective(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def _elastic(
    image: np.ndarray,
    rng: np.random.Generator,
    *,
    alpha: float = 3.0,
    sigma: float = 5.0,
) -> np.ndarray:
    height, width = image.shape[:2]
    noise_x = rng.normal(size=(height, width)).astype(np.float32)
    noise_y = rng.normal(size=(height, width)).astype(np.float32)
    kernel = max(3, int(round(sigma * 4)) | 1)
    displacement_x = cv2.GaussianBlur(noise_x, (kernel, kernel), sigma) * alpha
    displacement_y = cv2.GaussianBlur(noise_y, (kernel, kernel), sigma) * alpha
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    return cv2.remap(
        image,
        grid_x + displacement_x,
        grid_y + displacement_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def foreground_bbox(image: np.ndarray, threshold: int = 18) -> tuple[int, int, int, int]:
    """Return an inclusive-exclusive foreground bbox against a white canvas."""

    distance = np.max(np.abs(image.astype(np.int16) - 255), axis=2)
    ys, xs = np.where(distance > threshold)
    if not len(xs):
        return 0, 0, image.shape[1], image.shape[0]
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def _bbox_pad_crop(
    image: np.ndarray,
    rng: np.random.Generator,
    *,
    pad_range: tuple[float, float],
) -> np.ndarray:
    left, top, right, bottom = foreground_bbox(image)
    width = right - left
    height = bottom - top
    pad = float(rng.uniform(*pad_range)) * max(width, height)
    left = max(0, int(math.floor(left - pad)))
    top = max(0, int(math.floor(top - pad)))
    right = min(image.shape[1], int(math.ceil(right + pad)))
    bottom = min(image.shape[0], int(math.ceil(bottom + pad)))
    return _white_square(image[top:bottom, left:right])


def _photometric(
    image: np.ndarray,
    rng: np.random.Generator,
    *,
    strength: float,
    noise_probability: float,
) -> np.ndarray:
    pil = Image.fromarray(image)
    pil = ImageEnhance.Brightness(pil).enhance(
        float(rng.uniform(1 - strength, 1 + strength))
    )
    pil = ImageEnhance.Contrast(pil).enhance(
        float(rng.uniform(1 - strength, 1 + strength))
    )
    result = np.asarray(pil, dtype=np.uint8)
    if rng.random() < noise_probability:
        noise = rng.normal(0, 255 * strength / 4, result.shape)
        result = np.clip(result.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return result


def augment_image(
    image: np.ndarray,
    profile: str,
    seed: int,
    *,
    size: int = IMAGE_SIZE,
) -> np.ndarray:
    """Return one deterministic RGB uint8 view for a named profile."""

    if profile not in PROFILE_NAMES:
        raise ValueError(f"unknown profile {profile!r}; expected one of {PROFILE_NAMES}")
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("augmentation input must be RGB uint8")
    rng = np.random.default_rng(seed)

    if profile == "bbox_pad_crop":
        result = _bbox_pad_crop(image, rng, pad_range=(0.08, 0.18))
    elif profile == "combo_sobre":
        result = _bbox_pad_crop(image, rng, pad_range=(0.08, 0.16))
        result = _affine(
            result,
            rng,
            scale_range=(0.95, 1.05),
            rotation_limit=5,
            translation_limit=0.02,
        )
        if rng.random() < 0.15:
            result = _perspective_white(result, rng, (0.01, 0.025))
        if rng.random() < 0.35:
            result = _photometric(
                result, rng, strength=0.08, noise_probability=0.15
            )
    elif profile == "geom_mild":
        result = _white_square(image, size)
        result = _affine(
            result,
            rng,
            scale_range=(0.92, 1.08),
            rotation_limit=8,
            translation_limit=0.03,
        )
        if rng.random() < 0.2:
            result = _perspective_white(result, rng, (0.01, 0.03))
    else:
        # Mirrors the current tier-1 geometry, but fixes Perspective's border
        # to white. It intentionally retains flip/elastic as the control arm.
        result = _white_square(image, size)
        if rng.random() < 0.7:
            result = _affine(
                result,
                rng,
                scale_range=(0.85, 1.15),
                rotation_limit=15,
                translation_limit=0.05,
            )
        if rng.random() < 0.3:
            result = _elastic(result, rng)
        if rng.random() < 0.3:
            result = _perspective_white(result, rng, (0.02, 0.06))
        if rng.random() < 0.3:
            result = np.ascontiguousarray(result[:, ::-1])
        if rng.random() < 0.5:
            result = _photometric(
                result, rng, strength=0.15, noise_probability=0.3
            )
        if rng.random() < 0.2:
            result = cv2.GaussianBlur(result, (3, 3), 0)

    if result.shape != (size, size, 3):
        result = _white_square(result, size)
    return np.ascontiguousarray(result, dtype=np.uint8)


def dino_tensor(image: np.ndarray) -> torch.Tensor:
    """Apply the exact resize/pad + ImageNet normalization used by DINO caches."""

    if image.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
        image = _white_square(image)
    value = image.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    value = (value - mean) / std
    return torch.from_numpy(value).permute(2, 0, 1)


def load_rgb(path: str | Path) -> np.ndarray:
    with Image.open(path) as source:
        rgba = source.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        background.alpha_composite(rgba)
        return np.asarray(background.convert("RGB"), dtype=np.uint8)


def profile_seed(profile: str, cache_index: int, base_seed: int = BASE_SEED) -> int:
    return (base_seed + v5.stable_hash(f"{profile}:{cache_index}")) % (2**32)


def parse_profiles(values: Iterable[str]) -> list[str]:
    parsed: list[str] = []
    for value in values:
        parsed.extend(part.strip() for part in value.split(",") if part.strip())
    if not parsed:
        raise ValueError("at least one augmentation profile is required")
    unknown = sorted(set(parsed) - set(PROFILE_NAMES))
    if unknown:
        raise ValueError(f"unknown augmentation profiles: {unknown}")
    return list(dict.fromkeys(parsed))


def smoke_indices(
    indices: list[int],
    labels: torch.Tensor,
    limit_per_class: int,
) -> list[int]:
    if limit_per_class <= 0:
        return list(indices)
    counts: dict[int, int] = {}
    selected: list[int] = []
    for index in indices:
        label = int(labels[index])
        if counts.get(label, 0) < limit_per_class:
            selected.append(index)
            counts[label] = counts.get(label, 0) + 1
    return selected


def validate_split_contract(split: dict[str, Any]) -> None:
    train = set(split["external_train"])
    dev = set(split["dev"])
    test = set(split["test"])
    if train & dev or train & test or dev & test:
        raise ValueError("train/development/sealed-test indices overlap")
    audit = split.get("audit", {})
    if audit.get("overlap_total") != 0:
        raise ValueError("source-group overlap audit is non-zero")
    if audit.get("v4_test_preserved_exactly") is not True:
        raise ValueError("sealed v4 test is not preserved")


def _load_backbone(device: torch.device) -> tuple[torch.nn.Module, dict[str, str]]:
    hub_dir = Path(torch.hub.get_dir())
    repository = hub_dir / "facebookresearch_dinov2_main"
    checkpoint = hub_dir / "checkpoints" / DINO_CHECKPOINT_NAME
    if not repository.is_dir() or not (repository / "hubconf.py").is_file():
        raise FileNotFoundError(f"local DINOv2 torch-hub source missing: {repository}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"cached DINOv2 weights missing: {checkpoint}")
    backbone = torch.hub.load(
        str(repository),
        "dinov2_vits14",
        source="local",
        pretrained=True,
    ).to(device)
    backbone.eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    return backbone, {
        "source": str(repository.resolve()),
        "source_mode": "local_torch_hub_cache",
        "weights": str(checkpoint.resolve()),
        "weights_sha256": v5.sha256_file(checkpoint),
    }


@torch.inference_mode()
def _profile_features(
    profile: str,
    indices: list[int],
    external: dict[str, Any],
    run_dir: Path,
    device: torch.device,
    backbone_holder: dict[str, Any],
    *,
    batch_size: int = 32,
) -> dict[str, Any]:
    cache_path = run_dir / "profile-caches" / profile / "features.pt"
    expected_indices = torch.tensor(indices, dtype=torch.long)
    if cache_path.is_file():
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        if (
            cached.get("profile") == profile
            and torch.equal(cached.get("source_indices"), expected_indices)
            and cached.get("features", torch.empty(0)).shape
            == (len(indices), v5.FEATURE_DIM)
        ):
            return cached
        raise ValueError(f"incompatible cached profile features: {cache_path}")

    if "model" not in backbone_holder:
        backbone_holder["model"], backbone_holder["audit"] = _load_backbone(device)
    backbone = backbone_holder["model"]
    example_dir = run_dir / "augmented-examples" / profile
    example_dir.mkdir(parents=True, exist_ok=True)
    features: list[torch.Tensor] = []
    tensors: list[torch.Tensor] = []
    paths: list[str] = []

    def flush() -> None:
        if not tensors:
            return
        output = backbone(torch.stack(tensors).to(device))
        if not isinstance(output, torch.Tensor) or output.ndim != 2:
            raise ValueError("DINOv2 backbone returned an unexpected output")
        features.append(output.cpu())
        tensors.clear()

    for position, cache_index in enumerate(indices):
        path = str(external["image_paths"][cache_index])
        augmented = augment_image(
            load_rgb(path), profile, profile_seed(profile, cache_index)
        )
        if position < 8:
            Image.fromarray(augmented).save(
                example_dir / f"{position:02d}-index-{cache_index}.png"
            )
        tensors.append(dino_tensor(augmented))
        paths.append(path)
        if len(tensors) == batch_size:
            flush()
    flush()
    payload = {
        "schema_version": "autoresearch-image-augmentation-features.v1",
        "profile": profile,
        "seed": BASE_SEED,
        "features": torch.cat(features) if features else torch.empty((0, 384)),
        "labels": external["labels"][expected_indices].long().clone(),
        "image_paths": paths,
        "source_indices": expected_indices,
        "class_names": dict(external["class_names"]),
        "one_augmented_view_per_source": True,
        "dino_preprocessing": "resize-pad-white-224 + ImageNet mean/std",
        "backbone": dict(backbone_holder["audit"]),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    return payload


def evaluate(
    prototypes: torch.Tensor,
    embeddings: torch.Tensor,
    truth: torch.Tensor,
) -> tuple[dict[str, float | int], torch.Tensor, torch.Tensor, torch.Tensor]:
    scores = embeddings.float() @ F.normalize(prototypes.float(), dim=1).T
    top3_scores, top3 = scores.topk(3, dim=1)
    predictions = top3[:, 0]
    per_class = [
        float((predictions[truth == label] == label).float().mean())
        for label in truth.unique(sorted=True)
    ]
    result: dict[str, float | int] = {
        "top1": float((predictions == truth).float().mean()),
        "macro_top1": sum(per_class) / len(per_class),
        "top3": float((top3 == truth[:, None]).any(1).float().mean()),
        "mean_confidence": float(top3_scores[:, 0].mean()),
        "count": int(len(truth)),
        "class_count": int(truth.unique().numel()),
    }
    return result, predictions, top3, top3_scores


def development_rank(metrics_value: dict[str, float | int]) -> tuple[float, float]:
    return float(metrics_value["top1"]), float(metrics_value["macro_top1"])


def passes_historical_gate(
    candidate: dict[str, float | int],
    historical: dict[str, float | int],
) -> bool:
    return bool(
        float(candidate["top1"]) > float(historical["top1"])
        and float(candidate["macro_top1"]) >= float(historical["macro_top1"])
    )


def prediction_rows(
    *,
    split_name: str,
    indices: list[int],
    external: dict[str, Any],
    truth: torch.Tensor,
    results: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> list[dict[str, Any]]:
    names = v5.class_names(external)
    rows: list[dict[str, Any]] = []
    for position, cache_index in enumerate(indices):
        models: dict[str, Any] = {}
        for model_name, (prediction, top3, top3_scores) in results.items():
            predicted_label = int(prediction[position])
            models[model_name] = {
                "prediction_label": predicted_label,
                "prediction_name": names[predicted_label],
                "correct": predicted_label == int(truth[position]),
                "top3": [
                    {
                        "label": int(label),
                        "name": names[int(label)],
                        "score": float(score),
                    }
                    for label, score in zip(
                        top3[position].tolist(), top3_scores[position].tolist()
                    )
                ],
            }
        truth_label = int(truth[position])
        rows.append(
            {
                "split": split_name,
                "cache_index": cache_index,
                "path": str(external["image_paths"][cache_index]),
                "truth_label": truth_label,
                "truth_name": names[truth_label],
                "models": models,
            }
        )
    return rows


@dataclass
class SealedFeatures:
    external: dict[str, Any]
    indices: list[int]
    projection: torch.nn.Module
    device: torch.device
    read_count: int = 0

    def read(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.read_count:
            raise RuntimeError("sealed test features may be read exactly once")
        self.read_count += 1
        index = torch.tensor(self.indices, dtype=torch.long)
        return (
            v5.project(self.projection, self.external["features"][index], self.device),
            self.external["labels"][index].long(),
        )


def _checkpoint_prototypes(path: Path, expected_projection_sha: str) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    prototypes = payload.get("prototypes") if isinstance(payload, dict) else None
    if not isinstance(prototypes, torch.Tensor) or prototypes.shape != (
        v5.CLASS_COUNT,
        v5.EMBEDDING_DIM,
    ):
        raise ValueError(f"invalid v5 iteration-1 checkpoint: {path}")
    if payload.get("projection_sha256") != expected_projection_sha:
        raise ValueError("v5 checkpoint projection differs from historical")
    return F.normalize(prototypes.float(), dim=1)


def _write_state(
    path: Path,
    *,
    started_at: str,
    status: str,
    iteration: int,
    leader: str | None,
    sealed_reads: int,
    max_runtime: int,
    passed: bool | None = None,
) -> None:
    value: dict[str, Any] = {
        "status": status,
        "mission": "elements-image-augmentations-v1",
        "started_at": started_at,
        "updated_at": v5.utc_now(),
        "iteration": iteration,
        "leader": leader,
        "sealed_test_read_count": sealed_reads,
        "max_runtime_seconds": max_runtime,
    }
    if passed is not None:
        value["pass"] = passed
        value["completed_at"] = v5.utc_now()
    v5.dump_json(path, value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--profiles", nargs="+", default=list(PROFILE_NAMES))
    parser.add_argument("--max-runtime", type=int, default=6 * 60 * 60)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--smoke-limit-per-class", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--external-cache", type=Path, default=v5.DEFAULT_EXTERNAL_CACHE)
    parser.add_argument("--legacy-cache", type=Path, default=v5.DEFAULT_LEGACY_CACHE)
    parser.add_argument("--import-snapshot", type=Path, default=v5.DEFAULT_IMPORT_SNAPSHOT)
    parser.add_argument(
        "--v4-strict-results", type=Path, default=v5.DEFAULT_V4_STRICT_RESULTS
    )
    parser.add_argument(
        "--historical-weights", type=Path, default=v5.DEFAULT_HISTORICAL_WEIGHTS
    )
    parser.add_argument("--runtime-config", type=Path, default=v5.DEFAULT_RUNTIME_CONFIG)
    parser.add_argument("--v5-checkpoint", type=Path, default=DEFAULT_V5_CHECKPOINT)
    args = parser.parse_args(argv)

    profiles = parse_profiles(args.profiles)
    if args.max_runtime <= 0:
        raise ValueError("--max-runtime must be positive")
    if args.smoke_limit_per_class < 0:
        raise ValueError("--smoke-limit-per-class must be non-negative")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the requested device")
    device = torch.device(args.device)
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    started_at = v5.utc_now()
    deadline = time.time() + args.max_runtime
    state_path = run_dir / "state.json"
    _write_state(
        state_path,
        started_at=started_at,
        status="running",
        iteration=0,
        leader=None,
        sealed_reads=0,
        max_runtime=args.max_runtime,
    )
    v5.dump_json(
        run_dir / "mission.json",
        {
            "schema_version": "autoresearch-image-augmentation-mission.v1",
            "mission": "test whether one restrained image augmentation view improves the v5 prototype-only candidate",
            "profiles": profiles,
            "one_view_per_external_train_image": True,
            "selection_split": "source-group-disjoint development",
            "sealed_test_policy": "read once after development selection",
            "projection": "historical frozen",
            "prototype_recipe": "v5 iteration-1 frozen",
            "started_at": started_at,
        },
    )

    historical = v5.load_runtime(
        args.historical_weights, args.runtime_config, device
    )
    external = v5.load_cache(args.external_cache)
    legacy = v5.load_cache(args.legacy_cache)
    v5.validate_cache_taxonomy(
        external, historical.ordered_names, cache_name="external cache"
    )
    v5.validate_cache_taxonomy(
        legacy, historical.ordered_names, cache_name="safe legacy cache"
    )
    snapshot = v5.load_snapshot(args.import_snapshot, historical.ordered_names)
    provenance = v5.provenance_for_cache(external, snapshot)
    split = v5.leakage_safe_split(
        external, provenance, args.v4_strict_results
    )
    validate_split_contract(split)
    train_indices = smoke_indices(
        split["external_train"],
        external["labels"].long(),
        args.smoke_limit_per_class,
    )
    v5.dump_json(
        run_dir / "split-audit.json",
        {
            **split["audit"],
            "effective_external_train_count": len(train_indices),
            "full_external_train_count": len(split["external_train"]),
            "smoke_limit_per_class": args.smoke_limit_per_class,
            "external_cache_sha256": v5.sha256_file(args.external_cache),
            "legacy_cache_sha256": v5.sha256_file(args.legacy_cache),
            "historical_projection_sha256": v5.sha256_file(
                args.historical_weights / "projection.pt"
            ),
        },
    )
    v5.dump_json(
        run_dir / "split-indices.json",
        {
            "schema_version": "autoresearch-image-augmentation-split.v1",
            "external_train": train_indices,
            "development": split["dev"],
            "sealed_test": split["test"],
        },
    )

    projection_sha = v5.sha256_file(args.historical_weights / "projection.pt")
    v5_prototypes = _checkpoint_prototypes(args.v5_checkpoint, projection_sha)
    dev_index = torch.tensor(split["dev"], dtype=torch.long)
    dev_embeddings = v5.project(
        historical.model, external["features"][dev_index], device
    )
    dev_truth = external["labels"][dev_index].long()
    historical_dev, historical_dev_pred, historical_dev_top3, historical_dev_scores = (
        evaluate(historical.prototypes.cpu(), dev_embeddings, dev_truth)
    )
    v5_dev, v5_dev_pred, v5_dev_top3, v5_dev_scores = evaluate(
        v5_prototypes, dev_embeddings, dev_truth
    )
    spec = dict(v5.experiment_specs()[0])
    evaluator = {
        "schema_version": "autoresearch-image-augmentation-evaluator.v1",
        "primary_metric": "development top1",
        "tie_breaker": "development macro_top1",
        "required_iteration_output": {"pass": "boolean", "score": "number"},
        "historical_development": historical_dev,
        "v5_iteration_1_development": v5_dev,
        "selection": "development only",
        "final_gate": {
            "development_top1": "strictly greater than historical",
            "development_macro_top1": "greater than or equal to historical",
            "sealed_test_top1": "strictly greater than historical",
            "sealed_test_macro_top1": "greater than or equal to historical",
        },
        "frozen_spec": spec,
    }
    v5.dump_json(run_dir / "evaluator.json", evaluator)

    clean_index = torch.tensor(train_indices, dtype=torch.long)
    clean_features = external["features"][clean_index]
    clean_labels = external["labels"][clean_index].long()
    backbone_holder: dict[str, Any] = {}
    leader: dict[str, Any] | None = None
    incumbent_rank: tuple[float, float] | None = None
    decision_lines = [
        "# Image augmentation autoresearch",
        "",
        f"Started: {started_at}",
        f"Historical DEV: {development_rank(historical_dev)}",
        f"v5 iteration-1 DEV: {development_rank(v5_dev)}",
        "Sealed TEST read: no",
        "",
    ]

    for iteration, profile in enumerate(profiles, start=1):
        if time.time() >= deadline:
            raise TimeoutError("image augmentation autoresearch exceeded max runtime")
        iteration_started = time.perf_counter()
        augmented = _profile_features(
            profile,
            train_indices,
            external,
            run_dir,
            device,
            backbone_holder,
            batch_size=args.batch_size,
        )
        train_features = torch.cat(
            [legacy["features"], clean_features, augmented["features"]]
        )
        train_labels = torch.cat(
            [legacy["labels"].long(), clean_labels, augmented["labels"].long()]
        )
        train_embeddings = v5.project(
            historical.model, train_features, device
        )
        prototypes = v5.train_prototypes(
            train_embeddings,
            train_labels,
            historical.prototypes,
            spec,
            device,
        )
        dev_metrics, dev_pred, dev_top3, dev_scores = evaluate(
            prototypes, dev_embeddings, dev_truth
        )
        rank = development_rank(dev_metrics)
        improves_incumbent = incumbent_rank is None or rank > incumbent_rank
        gate = passes_historical_gate(dev_metrics, historical_dev)
        passed = bool(improves_incumbent and gate)
        checkpoint = run_dir / "checkpoints" / f"{iteration:04d}-{profile}.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "prototypes": prototypes,
                "class_labels": historical.class_labels,
                "class_names": historical.class_names,
                "projection_sha256": projection_sha,
                "profile": profile,
                "spec": spec,
            },
            checkpoint,
        )
        evaluation_value = {
            "schema_version": "autoresearch-image-augmentation-evaluation.v1",
            "iteration": iteration,
            "profile": profile,
            "pass": passed,
            "score": float(dev_metrics["top1"]),
            "development": dev_metrics,
            "historical_development": historical_dev,
            "v5_iteration_1_development": v5_dev,
            "passes_historical_gate": gate,
            "improves_incumbent": improves_incumbent,
            "sealed_test_evaluated": False,
            "external_clean_count": len(train_indices),
            "external_augmented_count": len(train_indices),
            "safe_legacy_count": len(legacy["labels"]),
            "source_group_overlap_total": split["audit"]["overlap_total"],
            "checkpoint": str(checkpoint.resolve()),
            "profile_cache": str(
                (run_dir / "profile-caches" / profile / "features.pt").resolve()
            ),
            "elapsed_seconds": time.perf_counter() - iteration_started,
            "spec": spec,
        }
        v5.dump_json(
            run_dir / "evaluations" / f"{iteration:04d}-{profile}.json",
            evaluation_value,
        )
        if improves_incumbent:
            incumbent_rank = rank
            leader = {
                "profile": profile,
                "iteration": iteration,
                "prototypes": prototypes,
                "development": dev_metrics,
                "dev_predictions": (dev_pred, dev_top3, dev_scores),
                "checkpoint": checkpoint,
            }
        decision_lines.extend(
            [
                f"## {iteration:04d} — {profile}",
                "",
                f"- DEV top-1: {float(dev_metrics['top1']):.6f}",
                f"- DEV macro top-1: {float(dev_metrics['macro_top1']):.6f}",
                f"- Historical gate: {gate}",
                f"- Decision: {'KEEP' if improves_incumbent else 'REJECT'}",
                "- Sealed TEST read: no",
                "",
            ]
        )
        _write_state(
            state_path,
            started_at=started_at,
            status="running",
            iteration=iteration,
            leader=leader["profile"] if leader else None,
            sealed_reads=0,
            max_runtime=args.max_runtime,
        )

    if leader is None:
        raise RuntimeError("no augmentation profile was evaluated")

    sealed = SealedFeatures(
        external=external,
        indices=split["test"],
        projection=historical.model,
        device=device,
    )
    test_embeddings, test_truth = sealed.read()
    historical_test, historical_test_pred, historical_test_top3, historical_test_scores = (
        evaluate(historical.prototypes.cpu(), test_embeddings, test_truth)
    )
    v5_test, v5_test_pred, v5_test_top3, v5_test_scores = evaluate(
        v5_prototypes, test_embeddings, test_truth
    )
    leader_test, leader_test_pred, leader_test_top3, leader_test_scores = evaluate(
        leader["prototypes"], test_embeddings, test_truth
    )
    dev_gate = passes_historical_gate(leader["development"], historical_dev)
    test_gate = passes_historical_gate(leader_test, historical_test)
    final_pass = bool(dev_gate and test_gate)
    paired = v5.exact_mcnemar(
        leader_test_pred == test_truth,
        historical_test_pred == test_truth,
    )
    prediction_payload = {
        "schema_version": "autoresearch-image-augmentation-predictions.v1",
        "selected_profile": leader["profile"],
        "rows": prediction_rows(
            split_name="development",
            indices=split["dev"],
            external=external,
            truth=dev_truth,
            results={
                "historical": (
                    historical_dev_pred,
                    historical_dev_top3,
                    historical_dev_scores,
                ),
                "v5_iteration_1": (v5_dev_pred, v5_dev_top3, v5_dev_scores),
                "augmentation_leader": leader["dev_predictions"],
            },
        )
        + prediction_rows(
            split_name="sealed_test",
            indices=split["test"],
            external=external,
            truth=test_truth,
            results={
                "historical": (
                    historical_test_pred,
                    historical_test_top3,
                    historical_test_scores,
                ),
                "v5_iteration_1": (v5_test_pred, v5_test_top3, v5_test_scores),
                "augmentation_leader": (
                    leader_test_pred,
                    leader_test_top3,
                    leader_test_scores,
                ),
            },
        ),
    }
    v5.dump_json(run_dir / "prediction-rows.json", prediction_payload)
    final_path = run_dir / "final-evaluation.json"
    final = {
        "schema_version": "autoresearch-image-augmentation-final.v1",
        "pass": final_pass,
        "score": float(leader_test["top1"]) - float(historical_test["top1"]),
        "selected_profile": leader["profile"],
        "selected_iteration": leader["iteration"],
        "development": {
            "historical": historical_dev,
            "v5_iteration_1": v5_dev,
            "selected": leader["development"],
        },
        "sealed_test": {
            "historical": historical_test,
            "v5_iteration_1": v5_test,
            "selected": leader_test,
            "paired_mcnemar_vs_historical": paired,
        },
        "development_gate": dev_gate,
        "sealed_test_gate": test_gate,
        "sealed_test_read_count": sealed.read_count,
        "runtime_exported": final_pass,
        "prediction_rows": str((run_dir / "prediction-rows.json").resolve()),
    }
    v5.dump_json(final_path, final)
    if final_pass:
        v5.export_runtime(
            run_dir,
            historical,
            args.runtime_config,
            leader["prototypes"],
            {
                "iteration": leader["iteration"],
                "name": f"image-augmentation-{leader['profile']}",
            },
            final_path,
        )
    decision_lines.extend(
        [
            "## Final sealed TEST",
            "",
            f"- Selected: {leader['profile']}",
            f"- Historical: {development_rank(historical_test)}",
            f"- v5 iteration-1: {development_rank(v5_test)}",
            f"- Selected: {development_rank(leader_test)}",
            f"- Exact McNemar p: {float(paired['exact_two_sided_p']):.6f}",
            f"- Final pass: {final_pass}",
            f"- Runtime exported: {final_pass}",
            f"- Sealed TEST read count: {sealed.read_count}",
            "",
        ]
    )
    (run_dir / "decision-log.md").write_text(
        "\n".join(decision_lines), encoding="utf-8"
    )
    _write_state(
        state_path,
        started_at=started_at,
        status="completed",
        iteration=len(profiles),
        leader=leader["profile"],
        sealed_reads=sealed.read_count,
        max_runtime=args.max_runtime,
        passed=final_pass,
    )
    print(json.dumps(final, indent=2))
    return 0 if final_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())

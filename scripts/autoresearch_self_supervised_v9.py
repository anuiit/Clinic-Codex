#!/usr/bin/env python3
"""Fold-local B0/C1 self-supervised experiment for Elements v9.

The script is intentionally isolated from runtime export.  It reconstructs the
conservative v8 components after conflict quarantine, creates train-only DINO
view caches per outer fold, and compares identical supervised training runs
with and without VICReg projection-head pretraining.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

from codex_pipeline.determinism import configure_determinism  # noqa: E402
from codex_pipeline.models.projection_head import ProjectionHead  # noqa: E402
from codex_pipeline.models.prototypical import PrototypicalLoss  # noqa: E402
from codex_pipeline.scripts.precompute_embeddings import load_backbone  # noqa: E402


V8_RUN = ROOT / ".omc/autoresearch/elements-baseline-replacement/runs/20260731-council-guided-v8"
V9_RUN = ROOT / ".omc/autoresearch/elements-baseline-replacement/runs/20260803-self-supervised-v9"
DEFAULT_MANIFEST = V8_RUN / "iteration-0002/provenance-manifest.jsonl"
DEFAULT_AUDIT = V8_RUN / "iteration-0003/collection-provenance-audit.json"
DEFAULT_BACKBONE_MANIFEST = V9_RUN / "inputs/dinov2-vits14-local-manifest.json"
DEFAULT_CACHE_DIR = V9_RUN / "iteration-0001/caches"
DEFAULT_OUTPUT_DIR = V9_RUN / "iteration-0001/results"
DEFAULT_RUNTIME_PROJECTION = BACKEND / "codex_model/weights/projection.pt"
DEFAULT_RUNTIME_PROTOTYPES = BACKEND / "codex_model/weights/prototypes.pt"

SAFE_VIEW_OPERATIONS = (
    "resize_longest_side",
    "white_pad",
    "affine_rotation_translation_scale",
    "brightness",
    "contrast",
    "mild_blur_or_noise",
)
FORBIDDEN_VIEW_OPERATIONS = frozenset(
    {
        "horizontal_flip",
        "vertical_flip",
        "elastic_transform",
        "perspective",
        "crop",
        "mixup",
        "cutmix",
        "hue_shift",
    }
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def stable_seed(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & 0x7FFFFFFF


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def rebuild_conservative_components(rows: list[dict[str, Any]]) -> dict[str, str]:
    """Reproduce v8.3: source-family and exact RGB edges after quarantine."""
    union_find = UnionFind(len(rows))
    family_owner: dict[str, int] = {}
    pixel_owner: dict[str, int] = {}
    for index, row in enumerate(rows):
        family = str(row["source_family"])
        pixel_hash = str(row["decoded_pixel_sha256"])
        if family in family_owner:
            union_find.union(index, family_owner[family])
        else:
            family_owner[family] = index
        if pixel_hash in pixel_owner:
            union_find.union(index, pixel_owner[pixel_hash])
        else:
            pixel_owner[pixel_hash] = index

    indices_by_root: dict[int, list[int]] = defaultdict(list)
    for index in range(len(rows)):
        indices_by_root[union_find.find(index)].append(index)
    mapping: dict[str, str] = {}
    for indices in indices_by_root.values():
        row_ids = sorted(str(rows[index]["row_id"]) for index in indices)
        digest = hashlib.sha256("\n".join(row_ids).encode("utf-8")).hexdigest()
        component_id = f"component:{digest[:20]}"
        for index in indices:
            mapping[str(rows[index]["row_id"])] = component_id
    return mapping


def assert_fold_isolation(train_rows: Sequence[dict[str, Any]], oof_rows: Sequence[dict[str, Any]]) -> None:
    fields = ("row_id", "component_id", "decoded_pixel_sha256")
    for field in fields:
        train_values = {str(row[field]) for row in train_rows}
        oof_values = {str(row[field]) for row in oof_rows}
        overlap = sorted(train_values & oof_values)
        if overlap:
            raise ValueError(f"{field} leakage across train/OOF: {overlap[:5]}")


def load_corpus(
    manifest_path: Path,
    audit_path: Path,
    *,
    strict_counts: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    required = {
        "row_id",
        "image_path",
        "class_label",
        "class_name",
        "source_family",
        "decoded_pixel_sha256",
        "fold",
    }
    if not rows:
        raise ValueError("empty provenance manifest")
    missing_fields = required - set(rows[0])
    if missing_fields:
        raise ValueError(f"manifest missing fields: {sorted(missing_fields)}")
    row_ids = [str(row["row_id"]) for row in rows]
    if len(set(row_ids)) != len(row_ids):
        raise ValueError("duplicate row_id in provenance manifest")

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    quarantined_ids = {str(value) for value in audit["quarantine"]["row_ids"]}
    quarantined_rows = [row for row in rows if str(row["row_id"]) in quarantined_ids]
    conflicting_hashes = {str(row["decoded_pixel_sha256"]) for row in quarantined_rows}
    retained = [row for row in rows if str(row["decoded_pixel_sha256"]) not in conflicting_hashes]
    if {str(row["row_id"]) for row in rows if str(row["decoded_pixel_sha256"]) in conflicting_hashes} != quarantined_ids:
        raise ValueError("audit quarantine does not equal all rows on conflicting RGB hashes")

    mapping = rebuild_conservative_components(retained)
    retained = [dict(row, component_id=mapping[str(row["row_id"])]) for row in retained]
    components = {str(row["component_id"]) for row in retained}
    folds = sorted({int(row["fold"]) for row in retained})
    for fold in folds:
        assert_fold_isolation(
            [row for row in retained if int(row["fold"]) != fold],
            [row for row in retained if int(row["fold"]) == fold],
        )

    per_class_components: dict[int, set[str]] = defaultdict(set)
    per_class_folds: dict[int, set[int]] = defaultdict(set)
    for row in retained:
        label = int(row["class_label"])
        per_class_components[label].add(str(row["component_id"]))
        per_class_folds[label].add(int(row["fold"]))
    evaluable_rows = sum(
        1 for row in retained if len(per_class_folds[int(row["class_label"])]) >= 2
    )
    report = {
        "schema_version": "autoresearch-self-supervised-v9.corpus-validation",
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "audit_path": str(audit_path.resolve()),
        "audit_sha256": sha256_file(audit_path),
        "source_rows": len(rows),
        "retained_rows": len(retained),
        "quarantined_rows": len(quarantined_ids),
        "conflicting_rgb_hashes": len(conflicting_hashes),
        "components_after_quarantine": len(components),
        "folds": folds,
        "classes": len(per_class_components),
        "classes_at_least_two_components": sum(len(value) >= 2 for value in per_class_components.values()),
        "classes_at_least_three_components": sum(len(value) >= 3 for value in per_class_components.values()),
        "independently_evaluable_oof_rows": evaluable_rows,
        "component_overlap_across_folds": 0,
        "decoded_pixel_overlap_across_folds": 0,
        "final_test_read": False,
    }
    if strict_counts:
        expected = {
            "source_rows": 10039,
            "retained_rows": 9990,
            "quarantined_rows": 49,
            "conflicting_rgb_hashes": 22,
            "components_after_quarantine": 300,
            "classes": 286,
            "classes_at_least_two_components": 12,
            "classes_at_least_three_components": 1,
            "independently_evaluable_oof_rows": 653,
        }
        mismatches = {key: (report[key], value) for key, value in expected.items() if report[key] != value}
        if mismatches:
            raise ValueError(f"v8 corpus contract mismatch: {mismatches}")
    return sorted(retained, key=lambda row: str(row["row_id"])), report


def resize_and_pad(image: Image.Image, size: int = 224) -> Image.Image:
    image = image.convert("RGB")
    width, height = image.size
    scale = size / max(width, height)
    resized = image.resize(
        (max(1, int(width * scale)), max(1, int(height * scale))),
        Image.Resampling.BILINEAR,
    )
    padded = Image.new("RGB", (size, size), (255, 255, 255))
    padded.paste(resized, ((size - resized.width) // 2, (size - resized.height) // 2))
    return padded


def safe_ssl_view(base: Image.Image, row_id: str, view_index: int, seed: int) -> Image.Image:
    rng_seed = stable_seed("ssl-view", seed, row_id, view_index)
    rng = random.Random(rng_seed)
    angle = rng.uniform(-7.0, 7.0)
    translate = [int(rng.uniform(-0.03, 0.03) * base.width), int(rng.uniform(-0.03, 0.03) * base.height)]
    scale = rng.uniform(0.9, 1.1)
    image = TF.affine(
        base,
        angle=angle,
        translate=translate,
        scale=scale,
        shear=[0.0, 0.0],
        interpolation=InterpolationMode.BILINEAR,
        fill=[255, 255, 255],
    )
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.9, 1.1))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.9, 1.1))
    if rng.random() < 0.5:
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.0, 0.6)))
    else:
        array = np.asarray(image, dtype=np.float32)
        noise = np.random.default_rng(rng_seed ^ 0x5A17).normal(0.0, rng.uniform(0.0, 2.55), array.shape)
        image = Image.fromarray(np.clip(array + noise, 0, 255).astype(np.uint8), mode="RGB")
    return image


def image_tensor(image: Image.Image) -> torch.Tensor:
    tensor = TF.pil_to_tensor(image).float().div_(255.0)
    return TF.normalize(tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


class FoldImageDataset(Dataset):
    def __init__(self, rows: Sequence[dict[str, Any]], views: int, seed: int, image_size: int) -> None:
        self.rows = list(rows)
        self.views = views
        self.seed = seed
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> torch.Tensor:
        row = self.rows[index]
        with Image.open(row["image_path"]) as source:
            base = resize_and_pad(source, self.image_size)
        tensors = [image_tensor(base)]
        tensors.extend(
            image_tensor(safe_ssl_view(base, str(row["row_id"]), view_index, self.seed))
            for view_index in range(self.views)
        )
        return torch.stack(tensors)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


@torch.inference_mode()
def extract_features(
    backbone: nn.Module,
    rows: Sequence[dict[str, Any]],
    *,
    views: int,
    seed: int,
    image_size: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    dataset = FoldImageDataset(rows, views, seed, image_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    bases: list[torch.Tensor] = []
    view_features: list[torch.Tensor] = []
    for batch in loader:
        batch_size_actual, view_count, channels, height, width = batch.shape
        flattened = batch.reshape(batch_size_actual * view_count, channels, height, width).to(
            device, non_blocking=True
        )
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                features = backbone(flattened)
        else:
            features = backbone(flattened)
        features = features.float().reshape(batch_size_actual, view_count, -1).cpu()
        if not torch.isfinite(features).all():
            raise FloatingPointError("non-finite DINO features")
        bases.append(features[:, 0])
        if views:
            view_features.append(features[:, 1:].to(torch.float16))
    base_tensor = torch.cat(bases).to(torch.float16)
    views_tensor = torch.cat(view_features) if view_features else None
    return base_tensor, views_tensor


def select_smoke_rows(rows: Sequence[dict[str, Any]], max_rows_per_class: int | None) -> list[dict[str, Any]]:
    if max_rows_per_class is None:
        return list(rows)
    buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[int(row["class_label"])].append(row)
    return [row for label in sorted(buckets) for row in buckets[label][:max_rows_per_class]]


def cache_path(cache_dir: Path, fold: int, views: int, max_rows_per_class: int | None) -> Path:
    suffix = f"-smoke{max_rows_per_class}" if max_rows_per_class else ""
    return cache_dir / f"fold-{fold:02d}-views{views:02d}{suffix}.pt"


def row_metadata(rows: Sequence[dict[str, Any]]) -> dict[str, list[Any]]:
    fields = ("row_id", "class_label", "class_name", "component_id", "decoded_pixel_sha256", "fold")
    return {field: [row[field] for row in rows] for field in fields}


def precompute_fold(
    rows: Sequence[dict[str, Any]],
    report: dict[str, Any],
    *,
    fold: int,
    views: int,
    view_seed: int,
    image_size: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    backbone: nn.Module,
    backbone_provenance: dict[str, Any],
    cache_dir: Path,
    max_rows_per_class: int | None,
    force: bool,
) -> Path:
    if views < 2:
        raise ValueError("VICReg precomputation requires at least two views")
    path = cache_path(cache_dir, fold, views, max_rows_per_class)
    if path.exists() and not force:
        load_fold_cache(
            path,
            expected_fold=fold,
            expected_views=views,
            expected_view_seed=view_seed,
            expected_image_size=image_size,
            expected_max_rows_per_class=max_rows_per_class,
            expected_backbone_provenance=backbone_provenance,
        )
        return path
    train_rows = select_smoke_rows([row for row in rows if int(row["fold"]) != fold], max_rows_per_class)
    train_labels = {int(row["class_label"]) for row in train_rows}
    oof_rows = [
        row for row in rows if int(row["fold"]) == fold and int(row["class_label"]) in train_labels
    ]
    if max_rows_per_class is not None:
        oof_rows = select_smoke_rows(oof_rows, max_rows_per_class)
    assert_fold_isolation(train_rows, oof_rows)
    train_base, train_views = extract_features(
        backbone,
        train_rows,
        views=views,
        seed=view_seed,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )
    oof_base, _ = extract_features(
        backbone,
        oof_rows,
        views=0,
        seed=view_seed,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )
    if train_views is None:
        raise ValueError("SSL cache requires at least one view")
    payload = {
        "schema_version": "autoresearch-self-supervised-v9.fold-cache",
        "fold": fold,
        "views": views,
        "view_seed": view_seed,
        "image_size": image_size,
        "max_rows_per_class": max_rows_per_class,
        "train": {"base_features": train_base, "view_features": train_views, **row_metadata(train_rows)},
        "oof": {"base_features": oof_base, **row_metadata(oof_rows)},
        "provenance": {
            "corpus_validation": report,
            "backbone": backbone_provenance,
            "safe_view_operations": list(SAFE_VIEW_OPERATIONS),
            "forbidden_view_operations": sorted(FORBIDDEN_VIEW_OPERATIONS),
            "ssl_outer_fold_row_exposure": 0,
            "learned_statistics_outer_fold_exposure": 0,
            "labels_used_by_ssl": False,
            "oof_view_features_persisted": False,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)
    write_json(
        path.with_suffix(path.suffix + ".prov.json"),
        {
            "schema_version": payload["schema_version"] + ".provenance",
            "cache_path": str(path.resolve()),
            "cache_sha256": sha256_file(path),
            "fold": fold,
            "train_rows": len(train_rows),
            "oof_rows": len(oof_rows),
            "views": views,
            "train_row_ids_sha256": sha256_json(payload["train"]["row_id"]),
            "oof_row_ids_sha256": sha256_json(payload["oof"]["row_id"]),
            "ssl_outer_fold_row_exposure": 0,
            "learned_statistics_outer_fold_exposure": 0,
            "final_test_read": False,
        },
    )
    return path


class VICRegExpander(nn.Module):
    def __init__(self, input_dim: int = 128, hidden_dim: int = 512) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


@dataclass(frozen=True)
class VICRegTerms:
    total: torch.Tensor
    invariance: torch.Tensor
    variance: torch.Tensor
    covariance: torch.Tensor
    minimum_std: torch.Tensor


def off_diagonal(matrix: torch.Tensor) -> torch.Tensor:
    size, other = matrix.shape
    if size != other:
        raise ValueError("covariance matrix must be square")
    return matrix.flatten()[:-1].view(size - 1, size + 1)[:, 1:].flatten()


def vicreg_loss(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    invariance_weight: float = 25.0,
    variance_weight: float = 25.0,
    covariance_weight: float = 1.0,
) -> VICRegTerms:
    if first.shape != second.shape or first.ndim != 2 or first.shape[0] < 2:
        raise ValueError("VICReg expects two equal 2D batches with at least two rows")
    invariance = F.mse_loss(first, second)
    centered_first = first - first.mean(dim=0)
    centered_second = second - second.mean(dim=0)
    std_first = torch.sqrt(first.var(dim=0, unbiased=False) + 1e-4)
    std_second = torch.sqrt(second.var(dim=0, unbiased=False) + 1e-4)
    variance = 0.5 * (F.relu(1.0 - std_first).mean() + F.relu(1.0 - std_second).mean())
    divisor = max(1, first.shape[0] - 1)
    cov_first = centered_first.T @ centered_first / divisor
    cov_second = centered_second.T @ centered_second / divisor
    covariance = (
        off_diagonal(cov_first).pow(2).sum() / first.shape[1]
        + off_diagonal(cov_second).pow(2).sum() / second.shape[1]
    )
    total = invariance_weight * invariance + variance_weight * variance + covariance_weight * covariance
    if not torch.isfinite(total):
        raise FloatingPointError("non-finite VICReg loss")
    return VICRegTerms(
        total=total,
        invariance=invariance,
        variance=variance,
        covariance=covariance,
        minimum_std=torch.minimum(std_first.min(), std_second.min()).detach(),
    )


def component_hash_buckets(metadata: dict[str, list[Any]]) -> dict[str, dict[str, list[int]]]:
    buckets: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index, (component, pixel_hash) in enumerate(
        zip(metadata["component_id"], metadata["decoded_pixel_sha256"])
    ):
        buckets[str(component)][str(pixel_hash)].append(index)
    return buckets


def ssl_pair_plan(
    metadata: dict[str, list[Any]],
    *,
    views: int,
    samples: int,
    seed: int,
) -> list[tuple[int, int, int]]:
    if views < 2:
        raise ValueError("VICReg needs at least two cached views")
    buckets = component_hash_buckets(metadata)
    components = sorted(buckets)
    rng = random.Random(seed)
    pairs: list[tuple[int, int, int]] = []
    for _ in range(samples):
        component = rng.choice(components)
        pixel_hash = rng.choice(sorted(buckets[component]))
        row_index = rng.choice(buckets[component][pixel_hash])
        first_view, second_view = rng.sample(range(views), 2)
        pairs.append((row_index, first_view, second_view))
    return pairs


def state_dict_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def effective_rank(values: torch.Tensor) -> float:
    if values.ndim != 2 or values.shape[0] < 2:
        return 0.0
    centered = values.float() - values.float().mean(dim=0)
    singular = torch.linalg.svdvals(centered)
    total = singular.sum()
    if not torch.isfinite(total) or float(total) <= 0.0:
        return 0.0
    probabilities = singular / total
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
    return float(entropy.exp().cpu())


def pretrain_vicreg(
    model: ProjectionHead,
    train: dict[str, Any],
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    model.train()
    expander = VICRegExpander().to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(expander.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    views = train["view_features"].to(device=device, dtype=torch.float32)
    samples_per_epoch = len(train["row_id"])
    plan_hash = hashlib.sha256()
    last_terms: VICRegTerms | None = None
    minimum_std: torch.Tensor | None = None
    for epoch in range(epochs):
        plan = ssl_pair_plan(
            train,
            views=int(views.shape[1]),
            samples=samples_per_epoch,
            seed=stable_seed("ssl-plan", seed, epoch),
        )
        plan_array = np.asarray(plan, dtype=np.int32)
        plan_hash.update(plan_array.tobytes())
        plan_tensor = torch.as_tensor(plan_array, dtype=torch.long, device=device)
        for offset in range(0, len(plan), batch_size):
            batch = plan_tensor[offset : offset + batch_size]
            if len(batch) < 2:
                continue
            first = views[batch[:, 0], batch[:, 1]]
            second = views[batch[:, 0], batch[:, 2]]
            first_projection = expander(model.net(first))
            second_projection = expander(model.net(second))
            terms = vicreg_loss(first_projection, second_projection)
            optimizer.zero_grad(set_to_none=True)
            terms.total.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(expander.parameters()), 5.0)
            optimizer.step()
            minimum_std = (
                terms.minimum_std
                if minimum_std is None
                else torch.minimum(minimum_std, terms.minimum_std)
            )
            last_terms = terms
    if last_terms is None or minimum_std is None:
        raise RuntimeError("VICReg produced no training batch")
    model.eval()
    with torch.inference_mode():
        base = train["base_features"][: min(4096, len(train["row_id"]))].to(
            device=device,
            dtype=torch.float32,
        )
        projection = model.net(base).cpu()
    rank = effective_rank(projection)
    minimum_std_value = float(minimum_std.cpu())
    if rank <= 1.0 or minimum_std_value <= 0.0:
        raise FloatingPointError("collapsed VICReg representation")
    return {
        "epochs": epochs,
        "samples_per_epoch": samples_per_epoch,
        "plan_sha256": plan_hash.hexdigest(),
        "last_loss": float(last_terms.total.detach().cpu()),
        "last_invariance": float(last_terms.invariance.detach().cpu()),
        "last_variance": float(last_terms.variance.detach().cpu()),
        "last_covariance": float(last_terms.covariance.detach().cpu()),
        "minimum_batch_std": minimum_std_value,
        "effective_rank": rank,
    }


Episode = tuple[list[int], list[int], list[int], list[int]]


def build_episode_plan(
    labels: torch.Tensor,
    *,
    n_way: int,
    k_shot: int,
    q_queries: int,
    epochs: int,
    episodes_per_epoch: int,
    seed: int,
) -> tuple[list[list[Episode]], str]:
    class_indices: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels.tolist()):
        class_indices[int(label)].append(index)
    required = k_shot + q_queries
    valid_classes = sorted(label for label, indices in class_indices.items() if len(indices) >= required)
    if len(valid_classes) < n_way:
        raise ValueError(f"only {len(valid_classes)} classes support {n_way}-way {k_shot}+{q_queries}")
    rng = random.Random(seed)
    plan: list[list[Episode]] = []
    digest = hashlib.sha256()
    for _ in range(epochs):
        epoch_plan: list[Episode] = []
        for _ in range(episodes_per_epoch):
            classes = rng.sample(valid_classes, n_way)
            support_indices: list[int] = []
            query_indices: list[int] = []
            support_labels: list[int] = []
            query_labels: list[int] = []
            for local_label, label in enumerate(classes):
                selected = rng.sample(class_indices[label], required)
                support_indices.extend(selected[:k_shot])
                query_indices.extend(selected[k_shot:])
                support_labels.extend([local_label] * k_shot)
                query_labels.extend([local_label] * q_queries)
            episode = (support_indices, support_labels, query_indices, query_labels)
            epoch_plan.append(episode)
            for values in episode:
                digest.update(np.asarray(values, dtype=np.int32).tobytes())
        plan.append(epoch_plan)
    return plan, digest.hexdigest()


def reset_training_rng(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_supervised(
    model: ProjectionHead,
    features: torch.Tensor,
    episode_plan: Sequence[Sequence[Episode]],
    *,
    learning_rate: float,
    weight_decay: float,
    temperature: float,
    warmup_epochs: int,
    rng_seed: int,
    device: torch.device,
) -> dict[str, Any]:
    reset_training_rng(rng_seed)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = PrototypicalLoss(temperature=temperature)
    features_on_device = features.to(device=device, dtype=torch.float32)
    device_episode_plan = [
        tuple(
            torch.tensor(
                [episode[field_index] for episode in episodes],
                dtype=torch.long,
                device=device,
            )
            for field_index in range(4)
        )
        for episodes in episode_plan
    ]
    total_epochs = len(device_episode_plan)
    last_loss = 0.0
    last_accuracy = 0.0
    for epoch, episode_tensors in enumerate(device_episode_plan):
        if epoch < warmup_epochs:
            factor = 0.01 + 0.99 * (epoch + 1) / max(1, warmup_epochs)
        else:
            progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs - 1)
            factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = learning_rate * factor
        epoch_loss = torch.zeros((), device=device)
        epoch_accuracy = torch.zeros((), device=device)
        episode_count = episode_tensors[0].shape[0]
        for episode_index in range(episode_count):
            support_indices, support_labels, query_indices, query_labels = (
                values[episode_index] for values in episode_tensors
            )
            result = criterion(
                model(features_on_device[support_indices]),
                support_labels,
                model(features_on_device[query_indices]),
                query_labels,
            )
            optimizer.zero_grad(set_to_none=True)
            result["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_loss = epoch_loss + result["loss"].detach()
            epoch_accuracy = epoch_accuracy + result["accuracy"].detach()
        last_loss = float((epoch_loss / episode_count).cpu())
        last_accuracy = float((epoch_accuracy / episode_count).cpu())
    return {
        "epochs": total_epochs,
        "episodes_per_epoch": device_episode_plan[0][0].shape[0],
        "last_loss": last_loss,
        "last_accuracy": last_accuracy,
    }


@torch.inference_mode()
def embed_in_batches(model: ProjectionHead, features: torch.Tensor, device: torch.device, batch_size: int = 1024) -> torch.Tensor:
    model.eval()
    values = [model(features[offset : offset + batch_size].float().to(device)).cpu() for offset in range(0, len(features), batch_size)]
    result = torch.cat(values)
    if not torch.isfinite(result).all():
        raise FloatingPointError("non-finite projected embedding")
    return result


def predict_arm(
    model: ProjectionHead,
    train: dict[str, Any],
    oof: dict[str, Any],
    *,
    device: torch.device,
) -> tuple[list[list[int]], float]:
    train_embeddings = embed_in_batches(model, train["base_features"], device)
    oof_embeddings = embed_in_batches(model, oof["base_features"], device)
    train_labels = torch.tensor(train["class_label"], dtype=torch.long)
    prototype_labels = train_labels.unique(sorted=True)
    prototypes = []
    for label in prototype_labels.tolist():
        prototype = train_embeddings[train_labels == label].mean(dim=0)
        prototypes.append(F.normalize(prototype, dim=0))
    prototype_tensor = torch.stack(prototypes)
    logits = oof_embeddings @ prototype_tensor.T
    top_indices = logits.topk(k=min(3, len(prototype_labels)), dim=1).indices
    top_labels = prototype_labels[top_indices]
    return [[int(value) for value in row] for row in top_labels.tolist()], effective_rank(oof_embeddings)


def load_fold_cache(
    path: Path,
    *,
    expected_fold: int,
    expected_views: int,
    expected_max_rows_per_class: int | None,
    expected_view_seed: int | None = None,
    expected_image_size: int | None = None,
    expected_backbone_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sidecar_path = path.with_suffix(path.suffix + ".prov.json")
    if not sidecar_path.is_file():
        raise ValueError(f"missing cache provenance sidecar: {sidecar_path}")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    actual_cache_sha256 = sha256_file(path)
    if sidecar.get("cache_sha256") != actual_cache_sha256:
        raise ValueError(f"cache hash does not match provenance sidecar: {path}")

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != "autoresearch-self-supervised-v9.fold-cache":
        raise ValueError(f"unexpected cache schema: {path}")
    fold = int(payload.get("fold", -1))
    views = int(payload.get("views", -1))
    if fold != expected_fold:
        raise ValueError(f"cache fold mismatch: expected {expected_fold}, found {fold}")
    if views != expected_views:
        raise ValueError(f"cache view-count mismatch: expected {expected_views}, found {views}")
    if payload.get("max_rows_per_class") != expected_max_rows_per_class:
        raise ValueError("cache smoke-row limit mismatch")
    if expected_view_seed is not None and int(payload.get("view_seed", -1)) != expected_view_seed:
        raise ValueError("cache view seed mismatch")
    if expected_image_size is not None and int(payload.get("image_size", -1)) != expected_image_size:
        raise ValueError("cache image size mismatch")
    provenance = payload.get("provenance", {})
    if (
        expected_backbone_provenance is not None
        and sha256_json(provenance.get("backbone")) != sha256_json(expected_backbone_provenance)
    ):
        raise ValueError("cache backbone provenance mismatch")

    metadata_fields = (
        "row_id",
        "class_label",
        "class_name",
        "component_id",
        "decoded_pixel_sha256",
        "fold",
    )
    for split in ("train", "oof"):
        split_length = len(payload[split]["row_id"])
        for field in metadata_fields:
            if len(payload[split][field]) != split_length:
                raise ValueError(f"{split}.{field} length mismatch")
        if len(payload[split]["base_features"]) != split_length:
            raise ValueError(f"{split}.base_features length mismatch")
    view_features = payload["train"].get("view_features")
    if (
        not isinstance(view_features, torch.Tensor)
        or view_features.ndim != 3
        or view_features.shape[0] != len(payload["train"]["row_id"])
        or view_features.shape[1] != views
    ):
        raise ValueError("train view feature tensor does not match cache contract")
    if "view_features" in payload["oof"]:
        raise ValueError("OOF view features must never be persisted")

    train_rows = [
        {
            field: payload["train"][field][index]
            for field in ("row_id", "component_id", "decoded_pixel_sha256")
        }
        for index in range(len(payload["train"]["row_id"]))
    ]
    oof_rows = [
        {
            field: payload["oof"][field][index]
            for field in ("row_id", "component_id", "decoded_pixel_sha256")
        }
        for index in range(len(payload["oof"]["row_id"]))
    ]
    if any(int(value) == fold for value in payload["train"]["fold"]):
        raise ValueError("outer-fold row found in SSL train cache")
    if any(int(value) != fold for value in payload["oof"]["fold"]):
        raise ValueError("non-outer-fold row found in OOF cache")
    assert_fold_isolation(train_rows, oof_rows)
    if tuple(provenance.get("safe_view_operations", ())) != SAFE_VIEW_OPERATIONS:
        raise ValueError("safe-view operation provenance mismatch")
    if set(provenance.get("forbidden_view_operations", ())) != FORBIDDEN_VIEW_OPERATIONS:
        raise ValueError("forbidden-view operation provenance mismatch")
    if provenance.get("ssl_outer_fold_row_exposure") != 0:
        raise ValueError("cache reports SSL outer-fold exposure")
    if provenance.get("learned_statistics_outer_fold_exposure") != 0:
        raise ValueError("cache reports learned-statistic outer-fold exposure")
    if provenance.get("labels_used_by_ssl") is not False:
        raise ValueError("cache does not prove label-free SSL")
    if provenance.get("oof_view_features_persisted") is not False:
        raise ValueError("cache does not prove absence of OOF augmented views")
    if sidecar.get("fold") != fold or sidecar.get("views") != views:
        raise ValueError("cache sidecar fold/view contract mismatch")
    if sidecar.get("ssl_outer_fold_row_exposure") != 0:
        raise ValueError("cache sidecar reports SSL outer-fold exposure")
    if sidecar.get("learned_statistics_outer_fold_exposure") != 0:
        raise ValueError("cache sidecar reports learned-statistic outer-fold exposure")
    if sidecar.get("train_row_ids_sha256") != sha256_json(payload["train"]["row_id"]):
        raise ValueError("train row IDs do not match provenance sidecar")
    if sidecar.get("oof_row_ids_sha256") != sha256_json(payload["oof"]["row_id"]):
        raise ValueError("OOF row IDs do not match provenance sidecar")
    corpus_validation = provenance.get("corpus_validation", {})
    quarantine_proven = all(
        (
            corpus_validation.get("source_rows") == 10039,
            corpus_validation.get("retained_rows") == 9990,
            corpus_validation.get("quarantined_rows") == 49,
            corpus_validation.get("conflicting_rgb_hashes") == 22,
            corpus_validation.get("components_after_quarantine") == 300,
        )
    )
    if not quarantine_proven:
        raise ValueError("cache does not prove the preregistered v8 conflict quarantine")
    payload["_cache_validation"] = {
        "cache_sha256": actual_cache_sha256,
        "v8_conflicting_rgb_hashes_quarantined": quarantine_proven,
        "provenance_component_overlap_across_folds": 0,
        "decoded_pixel_hash_overlap_across_folds": 0,
        "ssl_outer_fold_row_exposure": 0,
        "learned_statistics_outer_fold_exposure": 0,
    }
    return payload


def run_fold_seed(
    cache: dict[str, Any],
    *,
    seed: int,
    ssl_epochs: int,
    ssl_batch_size: int,
    supervised_epochs: int,
    episodes_per_epoch: int,
    n_way: int,
    k_shot: int,
    q_queries: int,
    device: torch.device,
    checkpoint_dir: Path,
    supervised_device: torch.device | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fold = int(cache["fold"])
    supervised_device = supervised_device or device
    configure_determinism(seed)
    initial_model = ProjectionHead(input_dim=384, embedding_dim=128)
    baseline = copy.deepcopy(initial_model).to(device)
    candidate = copy.deepcopy(initial_model).to(device)
    initial_hash = state_dict_sha256(initial_model)
    baseline_initial_hash = state_dict_sha256(baseline)
    candidate_initial_hash = state_dict_sha256(candidate)
    if len({initial_hash, baseline_initial_hash, candidate_initial_hash}) != 1:
        raise AssertionError("B0/C1 initial state mismatch")

    ssl_diagnostics = pretrain_vicreg(
        candidate,
        cache["train"],
        epochs=ssl_epochs,
        batch_size=ssl_batch_size,
        learning_rate=3e-4,
        weight_decay=1e-4,
        seed=seed,
        device=device,
    )
    train_labels = torch.tensor(cache["train"]["class_label"], dtype=torch.long)
    episode_plan, episode_hash = build_episode_plan(
        train_labels,
        n_way=n_way,
        k_shot=k_shot,
        q_queries=q_queries,
        epochs=supervised_epochs,
        episodes_per_epoch=episodes_per_epoch,
        seed=stable_seed("supervised-episodes", fold, seed),
    )
    supervised_rng_seed = stable_seed("supervised-rng", fold, seed)
    baseline = baseline.to(supervised_device)
    candidate = candidate.to(supervised_device)
    baseline_training = train_supervised(
        baseline,
        cache["train"]["base_features"],
        episode_plan,
        learning_rate=1e-3,
        weight_decay=1e-4,
        temperature=0.1,
        warmup_epochs=min(5, supervised_epochs),
        rng_seed=supervised_rng_seed,
        device=supervised_device,
    )
    candidate_training = train_supervised(
        candidate,
        cache["train"]["base_features"],
        episode_plan,
        learning_rate=1e-3,
        weight_decay=1e-4,
        temperature=0.1,
        warmup_epochs=min(5, supervised_epochs),
        rng_seed=supervised_rng_seed,
        device=supervised_device,
    )
    baseline = baseline.to(device)
    candidate = candidate.to(device)
    baseline_topk, baseline_rank = predict_arm(baseline, cache["train"], cache["oof"], device=device)
    candidate_topk, candidate_rank = predict_arm(candidate, cache["train"], cache["oof"], device=device)
    records = []
    for index, row_id in enumerate(cache["oof"]["row_id"]):
        records.append(
            {
                "row_id": str(row_id),
                "provenance_component": str(cache["oof"]["component_id"][index]),
                "decoded_pixel_sha256": str(cache["oof"]["decoded_pixel_sha256"][index]),
                "label": int(cache["oof"]["class_label"][index]),
                "class_name": str(cache["oof"]["class_name"][index]),
                "outer_fold": fold,
                "seed": seed,
                "recipe": "B0-vs-C1-vicreg-projection",
                "baseline_topk": baseline_topk[index],
                "candidate_topk": candidate_topk[index],
                "episode_plan_sha256": episode_hash,
            }
        )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = checkpoint_dir / f"fold-{fold:02d}-seed-{seed}-B0.pt"
    candidate_path = checkpoint_dir / f"fold-{fold:02d}-seed-{seed}-C1.pt"
    torch.save({"model_state_dict": baseline.state_dict(), "fold": fold, "seed": seed, "arm": "B0"}, baseline_path)
    torch.save({"model_state_dict": candidate.state_dict(), "fold": fold, "seed": seed, "arm": "C1"}, candidate_path)
    diagnostics = {
        "fold": fold,
        "seed": seed,
        "devices": {
            "ssl_and_prediction": str(device),
            "supervised": str(supervised_device),
        },
        "initial_state_sha256": initial_hash,
        "baseline_candidate_initial_state_equal": True,
        "episode_plan_sha256": episode_hash,
        "baseline_candidate_supervised_episode_trace_equal": True,
        "baseline_candidate_supervised_budget_equal": baseline_training["epochs"] == candidate_training["epochs"],
        "ssl": ssl_diagnostics,
        "baseline_training": baseline_training,
        "candidate_training": candidate_training,
        "baseline_effective_rank": baseline_rank,
        "candidate_effective_rank": candidate_rank,
        "effective_rank_ratio": candidate_rank / baseline_rank if baseline_rank else 0.0,
        "checkpoints": {
            "baseline": {"path": str(baseline_path), "sha256": sha256_file(baseline_path)},
            "candidate": {"path": str(candidate_path), "sha256": sha256_file(candidate_path)},
        },
    }
    return records, diagnostics


def accuracy_metrics(records: Sequence[dict[str, Any]]) -> dict[str, float]:
    if not records:
        raise ValueError("cannot score empty predictions")
    result: dict[str, float] = {}
    for arm in ("baseline", "candidate"):
        correct = [int(row[f"{arm}_topk"][0] == row["label"]) for row in records]
        top3 = [int(row["label"] in row[f"{arm}_topk"][:3]) for row in records]
        by_class: dict[int, list[int]] = defaultdict(list)
        for row, value in zip(records, correct):
            by_class[int(row["label"])].append(value)
        result[f"{arm}_top1"] = sum(correct) / len(correct)
        result[f"{arm}_top3"] = sum(top3) / len(top3)
        result[f"{arm}_macro_top1"] = sum(sum(values) / len(values) for values in by_class.values()) / len(by_class)
    result["delta_top1"] = result["candidate_top1"] - result["baseline_top1"]
    result["delta_top3"] = result["candidate_top3"] - result["baseline_top3"]
    result["delta_macro_top1"] = result["candidate_macro_top1"] - result["baseline_macro_top1"]
    return result


def paired_component_bootstrap(
    records: Sequence[dict[str, Any]],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    """Cluster bootstrap components, preserving B0/C1 and averaging fixed seeds."""
    if replicates < 100:
        raise ValueError("at least 100 bootstrap replicates are required")
    components = sorted({str(row["provenance_component"]) for row in records})
    seeds = sorted({int(row["seed"]) for row in records})
    lookup: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        lookup[(str(row["provenance_component"]), int(row["seed"]))].append(row)
    missing = [
        (component, selected_seed)
        for component in components
        for selected_seed in seeds
        if not lookup[(component, selected_seed)]
    ]
    if missing:
        raise ValueError(f"unpaired component/seed blocks: {missing[:5]}")

    rng = random.Random(seed)
    top1_values: list[float] = []
    macro_values: list[float] = []
    for _ in range(replicates):
        sampled_components = [rng.choice(components) for _ in components]
        seed_metrics = []
        for selected_seed in seeds:
            sample = [
                row
                for component in sampled_components
                for row in lookup[(component, selected_seed)]
            ]
            seed_metrics.append(accuracy_metrics(sample))
        top1_values.append(float(np.mean([item["delta_top1"] for item in seed_metrics])))
        macro_values.append(float(np.mean([item["delta_macro_top1"] for item in seed_metrics])))
    return {
        "method": "paired provenance-component cluster bootstrap; fixed preregistered seeds averaged",
        "replicates": replicates,
        "delta_top1_95": [
            float(np.quantile(top1_values, 0.025)),
            float(np.quantile(top1_values, 0.975)),
        ],
        "delta_macro_top1_95": [
            float(np.quantile(macro_values, 0.025)),
            float(np.quantile(macro_values, 0.975)),
        ],
    }


def exact_mcnemar(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    baseline_only = sum(
        row["baseline_topk"][0] == row["label"] and row["candidate_topk"][0] != row["label"]
        for row in records
    )
    candidate_only = sum(
        row["candidate_topk"][0] == row["label"] and row["baseline_topk"][0] != row["label"]
        for row in records
    )
    discordant = baseline_only + candidate_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, index) for index in range(min(baseline_only, candidate_only) + 1)) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    return {"baseline_only_correct": baseline_only, "candidate_only_correct": candidate_only, "exact_p_value": p_value}


def aggregate_results(
    records: Sequence[dict[str, Any]],
    diagnostics: Sequence[dict[str, Any]],
    cache_validations: Sequence[dict[str, Any]],
    *,
    bootstrap_replicates: int,
    expected_seed_count: int,
    expected_fold_count: int,
    runtime_unchanged: bool,
) -> dict[str, Any]:
    if not records or not diagnostics or not cache_validations:
        raise ValueError("evaluation requires predictions, diagnostics, and validated caches")
    seeds = sorted({int(row["seed"]) for row in records})
    folds = sorted({int(row["outer_fold"]) for row in records})
    seed_metrics = [
        {"seed": seed, **accuracy_metrics([row for row in records if int(row["seed"]) == seed])}
        for seed in seeds
    ]
    overall = accuracy_metrics(records)
    intervals = paired_component_bootstrap(
        records,
        replicates=bootstrap_replicates,
        seed=20260803,
    )
    positive_seed_count = sum(item["delta_top1"] > 0.0 for item in seed_metrics)
    minimum_rank_ratio = min(item["effective_rank_ratio"] for item in diagnostics)
    gate_values: dict[str, Any] = {
        "v8_conflicting_rgb_hashes_quarantined": all(
            item["v8_conflicting_rgb_hashes_quarantined"] for item in cache_validations
        ),
        "provenance_component_overlap_across_folds": max(
            item["provenance_component_overlap_across_folds"] for item in cache_validations
        ),
        "decoded_pixel_hash_overlap_across_folds": max(
            item["decoded_pixel_hash_overlap_across_folds"] for item in cache_validations
        ),
        "ssl_outer_fold_row_exposure": max(
            item["ssl_outer_fold_row_exposure"] for item in cache_validations
        ),
        "learned_statistics_outer_fold_exposure": max(
            item["learned_statistics_outer_fold_exposure"] for item in cache_validations
        ),
        "baseline_candidate_initial_state_equal": all(
            item["baseline_candidate_initial_state_equal"] for item in diagnostics
        ),
        "baseline_candidate_supervised_episode_trace_equal": all(
            item["baseline_candidate_supervised_episode_trace_equal"] for item in diagnostics
        ),
        "baseline_candidate_supervised_budget_equal": all(
            item["baseline_candidate_supervised_budget_equal"] for item in diagnostics
        ),
        "paired_predictions_persisted": bool(records),
        "paired_seed_count": len(seeds),
        "positive_seed_count_gte": positive_seed_count,
        "delta_top1_component_bootstrap_lower_95_gt": intervals["delta_top1_95"][0],
        "delta_macro_top1_component_bootstrap_lower_95_gte": intervals["delta_macro_top1_95"][0],
        "candidate_effective_rank_ratio_gte": minimum_rank_ratio,
        "nan_or_collapse_detected": any(
            item.get("nan_or_collapse_detected", False) for item in diagnostics
        ),
        "final_test_read": False,
        "runtime_unchanged": runtime_unchanged,
        "automatic_promotion": False,
        "outer_fold_count": len(folds),
    }
    gate_passes = {
        "v8_conflicting_rgb_hashes_quarantined": gate_values["v8_conflicting_rgb_hashes_quarantined"] is True,
        "provenance_component_overlap_across_folds": gate_values["provenance_component_overlap_across_folds"] == 0,
        "decoded_pixel_hash_overlap_across_folds": gate_values["decoded_pixel_hash_overlap_across_folds"] == 0,
        "ssl_outer_fold_row_exposure": gate_values["ssl_outer_fold_row_exposure"] == 0,
        "learned_statistics_outer_fold_exposure": gate_values["learned_statistics_outer_fold_exposure"] == 0,
        "baseline_candidate_initial_state_equal": gate_values["baseline_candidate_initial_state_equal"] is True,
        "baseline_candidate_supervised_episode_trace_equal": gate_values["baseline_candidate_supervised_episode_trace_equal"] is True,
        "baseline_candidate_supervised_budget_equal": gate_values["baseline_candidate_supervised_budget_equal"] is True,
        "paired_predictions_persisted": gate_values["paired_predictions_persisted"] is True,
        "paired_seed_count": gate_values["paired_seed_count"] == expected_seed_count,
        "positive_seed_count_gte": gate_values["positive_seed_count_gte"] >= 2,
        "delta_top1_component_bootstrap_lower_95_gt": gate_values["delta_top1_component_bootstrap_lower_95_gt"] > 0.0,
        "delta_macro_top1_component_bootstrap_lower_95_gte": gate_values["delta_macro_top1_component_bootstrap_lower_95_gte"] >= 0.0,
        "candidate_effective_rank_ratio_gte": gate_values["candidate_effective_rank_ratio_gte"] >= 0.9,
        "nan_or_collapse_detected": gate_values["nan_or_collapse_detected"] is False,
        "final_test_read": gate_values["final_test_read"] is False,
        "runtime_unchanged": gate_values["runtime_unchanged"] is True,
        "automatic_promotion": gate_values["automatic_promotion"] is False,
        "outer_fold_count": gate_values["outer_fold_count"] == expected_fold_count,
    }
    score = float(np.mean([item["delta_top1"] for item in seed_metrics]))
    return {
        "pass": all(gate_passes.values()),
        "score": score,
        "hypothesis_supported": all(
            (
                positive_seed_count >= 2,
                float(np.mean([item["delta_macro_top1"] for item in seed_metrics])) >= -0.005,
                minimum_rank_ratio >= 0.9,
            )
        ),
        "promotion_eligible": False,
        "overall_metrics": overall,
        "seed_metrics": seed_metrics,
        "bootstrap": intervals,
        "mcnemar": exact_mcnemar(records),
        "positive_seed_count": positive_seed_count,
        "minimum_effective_rank_ratio": minimum_rank_ratio,
        "gates": gate_values,
        "gate_passes": gate_passes,
        "final_test_read": False,
        "runtime_promotion": False,
    }


def parse_int_csv(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return result


def runtime_hashes(projection: Path, prototypes: Path) -> dict[str, str]:
    return {"projection": sha256_file(projection), "prototypes": sha256_file(prototypes)}


def command_validate(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows, report = load_corpus(args.manifest, args.audit, strict_counts=args.strict_counts)
    if args.validation_output:
        write_json(args.validation_output, report)
    print(canonical_json(report), end="")
    return rows, report


def command_precompute(args: argparse.Namespace) -> list[Path]:
    rows, report = load_corpus(args.manifest, args.audit, strict_counts=args.strict_counts)
    device = resolve_device(args.device)
    backbone, provenance = load_backbone("dinov2_vits14", device, args.backbone_manifest)
    backbone.requires_grad_(False).eval()
    paths = []
    for fold in args.folds:
        paths.append(
            precompute_fold(
                rows,
                report,
                fold=fold,
                views=args.views,
                view_seed=args.view_seed,
                image_size=args.image_size,
                batch_size=args.image_batch_size,
                num_workers=args.num_workers,
                device=device,
                backbone=backbone,
                backbone_provenance=provenance,
                cache_dir=args.cache_dir,
                max_rows_per_class=args.max_rows_per_class,
                force=args.force,
            )
        )
    print(canonical_json({"cache_paths": [str(path.resolve()) for path in paths]}), end="")
    return paths


def command_run(args: argparse.Namespace) -> dict[str, Any]:
    before_runtime = runtime_hashes(args.runtime_projection, args.runtime_prototypes)
    device = resolve_device(args.device)
    supervised_device = resolve_device(args.supervised_device)
    if supervised_device.type == "cpu":
        if args.supervised_cpu_threads < 1:
            raise ValueError("supervised CPU threads must be positive")
        torch.set_num_threads(args.supervised_cpu_threads)
    all_records: list[dict[str, Any]] = []
    all_diagnostics: list[dict[str, Any]] = []
    cache_validations: list[dict[str, Any]] = []
    cache_hashes: dict[str, str] = {}
    code_sha256 = sha256_file(Path(__file__))
    for fold in args.folds:
        path = cache_path(args.cache_dir, fold, args.views, args.max_rows_per_class)
        cache = load_fold_cache(
            path,
            expected_fold=fold,
            expected_views=args.views,
            expected_max_rows_per_class=args.max_rows_per_class,
        )
        cache_validation = dict(cache["_cache_validation"])
        cache_validations.append(cache_validation)
        cache_hashes[str(fold)] = cache_validation["cache_sha256"]
        for seed in args.seeds:
            records, diagnostics = run_fold_seed(
                cache,
                seed=seed,
                ssl_epochs=args.ssl_epochs,
                ssl_batch_size=args.ssl_batch_size,
                supervised_epochs=args.supervised_epochs,
                episodes_per_epoch=args.episodes_per_epoch,
                n_way=args.n_way,
                k_shot=args.k_shot,
                q_queries=args.q_queries,
                device=device,
                checkpoint_dir=args.output_dir / "checkpoints",
                supervised_device=supervised_device,
            )
            record_provenance = {
                "code_sha256": code_sha256,
                "data_sha256": cache_hashes[str(fold)],
                "runtime_projection_sha256": before_runtime["projection"],
                "runtime_prototypes_sha256": before_runtime["prototypes"],
                "baseline_checkpoint_sha256": diagnostics["checkpoints"]["baseline"]["sha256"],
                "candidate_checkpoint_sha256": diagnostics["checkpoints"]["candidate"]["sha256"],
                "ssl_device": str(device),
                "supervised_device": str(supervised_device),
            }
            for record in records:
                record.update(record_provenance)
            all_records.extend(records)
            all_diagnostics.append(diagnostics)

    after_runtime = runtime_hashes(args.runtime_projection, args.runtime_prototypes)
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
    folds_sha256 = sha256_json(fold_assignments)
    for record in all_records:
        record["folds_sha256"] = folds_sha256

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "prediction_rows.jsonl"
    write_jsonl(predictions_path, all_records)
    result = aggregate_results(
        all_records,
        all_diagnostics,
        cache_validations,
        bootstrap_replicates=args.bootstrap_replicates,
        expected_seed_count=3,
        expected_fold_count=5,
        runtime_unchanged=runtime_unchanged,
    )
    result.update(
        {
            "schema_version": "autoresearch-self-supervised-v9.evaluation",
            "iteration": 1,
            "name": "fold-local-vicreg-projection-initialization",
            "paired_predictions_path": predictions_path.name,
            "folds": args.folds,
            "seeds": args.seeds,
            "folds_sha256": folds_sha256,
            "cache_sha256": cache_hashes,
            "code_sha256": code_sha256,
            "data_sha256": sha256_json(cache_hashes),
            "runtime_checkpoint_sha256": before_runtime,
            "runtime_unchanged": runtime_unchanged,
            "cache_validations": cache_validations,
            "diagnostics": all_diagnostics,
        }
    )
    write_json(args.output_dir / "paired_seed_summary.json", result)
    write_json(
        args.output_dir / "per_arm_metrics.json",
        {"overall": result["overall_metrics"], "seeds": result["seed_metrics"]},
    )
    print(canonical_json(result), end="")
    return result


def add_corpus_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--strict-counts", action=argparse.BooleanOptionalAction, default=True)


def add_precompute_args(parser: argparse.ArgumentParser) -> None:
    add_corpus_args(parser)
    parser.add_argument("--backbone-manifest", type=Path, default=DEFAULT_BACKBONE_MANIFEST)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--folds", type=parse_int_csv, default=[1, 2, 3, 4, 5])
    parser.add_argument("--views", type=int, default=8)
    parser.add_argument("--view-seed", type=int, default=20260803)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--image-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-rows-per-class", type=int, default=None)
    parser.add_argument("--force", action="store_true")


def add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--folds", type=parse_int_csv, default=[1, 2, 3, 4, 5])
    parser.add_argument("--seeds", type=parse_int_csv, default=[17, 42, 73])
    parser.add_argument("--views", type=int, default=8)
    parser.add_argument("--ssl-epochs", type=int, default=30)
    parser.add_argument("--ssl-batch-size", type=int, default=256)
    parser.add_argument("--supervised-epochs", type=int, default=30)
    parser.add_argument("--episodes-per-epoch", type=int, default=100)
    parser.add_argument("--n-way", type=int, default=20)
    parser.add_argument("--k-shot", type=int, default=3)
    parser.add_argument("--q-queries", type=int, default=5)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--supervised-device", default="cpu")
    parser.add_argument("--supervised-cpu-threads", type=int, default=1)
    parser.add_argument("--max-rows-per-class", type=int, default=None)
    parser.add_argument("--runtime-projection", type=Path, default=DEFAULT_RUNTIME_PROJECTION)
    parser.add_argument("--runtime-prototypes", type=Path, default=DEFAULT_RUNTIME_PROTOTYPES)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate", help="validate v8 quarantine and fold isolation")
    add_corpus_args(validate)
    validate.add_argument("--validation-output", type=Path, default=V9_RUN / "iteration-0001/corpus-validation.json")
    precompute = subparsers.add_parser("precompute", help="create train-only DINO view caches")
    add_precompute_args(precompute)
    run = subparsers.add_parser("run", help="run paired B0/C1 training and evaluation")
    add_run_args(run)
    all_parser = subparsers.add_parser("all", help="precompute then run")
    add_precompute_args(all_parser)
    all_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    all_parser.add_argument("--seeds", type=parse_int_csv, default=[17, 42, 73])
    all_parser.add_argument("--ssl-epochs", type=int, default=30)
    all_parser.add_argument("--ssl-batch-size", type=int, default=256)
    all_parser.add_argument("--supervised-epochs", type=int, default=30)
    all_parser.add_argument("--episodes-per-epoch", type=int, default=100)
    all_parser.add_argument("--n-way", type=int, default=20)
    all_parser.add_argument("--k-shot", type=int, default=3)
    all_parser.add_argument("--q-queries", type=int, default=5)
    all_parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    all_parser.add_argument("--supervised-device", default="cpu")
    all_parser.add_argument("--supervised-cpu-threads", type=int, default=1)
    all_parser.add_argument("--runtime-projection", type=Path, default=DEFAULT_RUNTIME_PROJECTION)
    all_parser.add_argument("--runtime-prototypes", type=Path, default=DEFAULT_RUNTIME_PROTOTYPES)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "validate":
        command_validate(args)
    elif args.command == "precompute":
        command_precompute(args)
    elif args.command == "run":
        command_run(args)
    elif args.command == "all":
        command_precompute(args)
        command_run(args)
    else:  # pragma: no cover
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()


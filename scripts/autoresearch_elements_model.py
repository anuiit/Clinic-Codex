#!/usr/bin/env python3
"""Run one evaluator-driven Elements projection-head experiment.

This is intentionally a single-iteration runner. Autoresearch owns the loop,
the experiment specs, and the keep/reject decision log.
"""

from __future__ import annotations

import argparse
import os
import copy
import hashlib
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"

import sys

sys.path.insert(0, str(BACKEND_ROOT))

from codex_pipeline.determinism import configure_determinism
from codex_pipeline.models.projection_head import ProjectionHead
from codex_pipeline.models.prototypical import compute_prototypes


DEFAULT_DATA_ROOT = Path("/tmp/baseline-replacement-e2e/perclass-split")
RUNTIME_PROJECTION = BACKEND_ROOT / "codex_model" / "weights" / "projection.pt"
EXPECTED_STATE_SHAPES = {
    "net.0.weight": (384, 384),
    "net.0.bias": (384,),
    "net.3.weight": (128, 384),
    "net.3.bias": (128,),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_cache(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {"features", "labels"}
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"{path} is missing cache keys: {sorted(missing)}")
    if len(payload["features"]) != len(payload["labels"]):
        raise ValueError(f"{path} has mismatched features and labels")
    return payload


def cache_source_paths(cache: dict[str, Any]) -> set[str] | None:
    image_paths = cache.get("image_paths")
    if image_paths is None:
        return None
    return {
        os.path.normcase(str(Path(path).expanduser().resolve(strict=False)))
        for path in image_paths
    }


def resolve_train_cache(data_root: Path, train_cache: str) -> Path:
    choices = {
        "raw": data_root / "features_train.pt",
        "augmented_safe": data_root / "features_train_augmented_safe.pt",
    }
    try:
        path = choices[train_cache]
    except KeyError as exc:
        raise ValueError(f"unsupported train_cache: {train_cache!r}") from exc
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def validate_labels(train_labels: torch.Tensor, val_labels: torch.Tensor) -> int:
    train_classes = train_labels.unique(sorted=True)
    if train_classes.numel() == 0:
        raise ValueError("train labels must be non-empty")
    missing = sorted(set(val_labels.tolist()) - set(train_classes.tolist()))
    if missing:
        raise ValueError(f"validation labels missing from train: {missing}")
    return len(train_classes)


def validate_train_labels(train_labels: torch.Tensor) -> int:
    train_classes = train_labels.unique(sorted=True)
    if not torch.equal(train_classes, torch.arange(len(train_classes))):
        raise ValueError("train labels must be contiguous from zero")
    return len(train_classes)


def model_hidden_and_embedding(
    model: ProjectionHead,
    features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden = F.gelu(model.net[0](features))
    projected = model.net[3](model.net[2](hidden))
    return hidden, F.normalize(projected, p=2, dim=-1)


@torch.no_grad()
def embed_all(
    model: ProjectionHead,
    features: torch.Tensor,
    device: torch.device,
    batch_size: int = 2048,
) -> torch.Tensor:
    model.eval()
    return torch.cat(
        [model(batch.float().to(device)).cpu() for batch in features.split(batch_size)]
    )


def runtime_compatible(state: dict[str, torch.Tensor]) -> bool:
    if set(state) != set(EXPECTED_STATE_SHAPES):
        return False
    return all(tuple(state[key].shape) == shape for key, shape in EXPECTED_STATE_SHAPES.items())


def load_checkpoint_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict"):
            state_dict = checkpoint.get(key)
            if isinstance(state_dict, dict):
                return state_dict
        if all(key in checkpoint for key in EXPECTED_STATE_SHAPES):
            return checkpoint
    raise ValueError("checkpoint does not contain a model state dict")


def classification_metrics(
    model: ProjectionHead,
    train: dict[str, Any],
    evaluation: dict[str, Any],
    device: torch.device,
) -> dict[str, float]:
    train_embeddings = embed_all(model, train["features"], device)
    eval_embeddings = embed_all(model, evaluation["features"], device)
    prototype_labels = train["labels"].unique(sorted=True)
    prototypes = compute_prototypes(train_embeddings, train["labels"])
    similarities = eval_embeddings @ prototypes.T
    top3 = similarities.topk(k=3, dim=1).indices
    predictions = prototype_labels[top3[:, 0]]
    top3_labels = prototype_labels[top3]
    labels = evaluation["labels"]

    per_class: list[float] = []
    for label in labels.unique(sorted=True):
        mask = labels == label
        per_class.append(float((predictions[mask] == labels[mask]).float().mean()))

    return {
        "top1": float((predictions == labels).float().mean()),
        "macro_top1": sum(per_class) / len(per_class),
        "top3": float((top3_labels == labels[:, None]).any(dim=1).float().mean()),
    }


def build_loader(
    features: torch.Tensor,
    labels: torch.Tensor,
    spec: dict[str, Any],
    seed: int,
) -> DataLoader:
    batch_size = int(spec.get("batch_size", 512))
    generator = torch.Generator().manual_seed(seed)
    if spec.get("class_balanced", False):
        counts = Counter(labels.tolist())
        sample_weights = torch.tensor([1.0 / counts[int(label)] for label in labels])
        sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(labels),
            replacement=True,
            generator=generator,
        )
        return DataLoader(
            TensorDataset(features, labels),
            batch_size=batch_size,
            sampler=sampler,
            num_workers=0,
        )
    return DataLoader(
        TensorDataset(features, labels),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )


def load_teacher(device: torch.device) -> ProjectionHead:
    teacher = ProjectionHead(384, 128).to(device)
    teacher.load_state_dict(
        torch.load(RUNTIME_PROJECTION, map_location=device, weights_only=False)
    )
    teacher.eval()
    return teacher


def train_proxy_or_distillation(
    model: ProjectionHead,
    train: dict[str, Any],
    spec: dict[str, Any],
    device: torch.device,
    class_count: int,
) -> None:
    seed = int(spec.get("seed", 42))
    objective = spec["objective"]
    teacher_weight = float(spec.get("teacher_weight", 0.0))
    hidden_teacher_weight = float(spec.get("hidden_teacher_weight", 0.0))
    proxy_weight = float(spec.get("proxy_weight", 1.0))
    temperature = float(spec.get("temperature", 0.1))
    label_smoothing = float(spec.get("label_smoothing", 0.0))
    noise_std = float(spec.get("noise_std", 0.0))
    epochs = int(spec.get("epochs", 30))

    proxies = nn.Parameter(torch.randn(class_count, 128, device=device))
    nn.init.normal_(proxies, std=0.02)
    parameters: list[nn.Parameter] = list(model.parameters())
    if proxy_weight > 0:
        parameters.append(proxies)

    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(spec.get("learning_rate", 1e-3)),
        weight_decay=float(spec.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    loader = build_loader(train["features"], train["labels"], spec, seed)
    teacher = load_teacher(device) if teacher_weight > 0 or hidden_teacher_weight > 0 else None

    model.net[2].p = float(spec.get("training_dropout", 0.1))
    for _ in range(epochs):
        model.train()
        for features, labels in loader:
            features = features.float().to(device)
            labels = labels.long().to(device)
            if noise_std > 0:
                features = features + torch.randn_like(features) * noise_std
            student_hidden, student_embeddings = model_hidden_and_embedding(model, features)
            loss = torch.zeros((), device=device)

            if proxy_weight > 0:
                normalized_proxies = F.normalize(proxies, p=2, dim=1)
                logits = student_embeddings @ normalized_proxies.T / temperature
                loss = loss + proxy_weight * F.cross_entropy(
                    logits,
                    labels,
                    label_smoothing=label_smoothing,
                )

            if teacher is not None:
                with torch.no_grad():
                    teacher_hidden, teacher_embeddings = model_hidden_and_embedding(
                        teacher, features
                    )
                if teacher_weight > 0:
                    loss = loss + teacher_weight * (
                        1.0 - F.cosine_similarity(student_embeddings, teacher_embeddings).mean()
                    )
                if hidden_teacher_weight > 0:
                    loss = loss + hidden_teacher_weight * F.mse_loss(
                        F.normalize(student_hidden, dim=1),
                        F.normalize(teacher_hidden, dim=1),
                    )

            if not loss.requires_grad:
                raise ValueError(f"objective {objective!r} produced no differentiable loss")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=5.0)
            optimizer.step()
        scheduler.step()


def refit_experiment(
    spec_path: Path,
    train_cache_path: Path,
    output_dir: Path,
    device_name: str,
) -> dict[str, Any]:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    seed = int(spec.get("seed", 42))
    configure_determinism(seed)
    random.seed(seed)

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("refit requires CUDA for the selected recipe")

    device = torch.device(device_name)
    train = load_cache(train_cache_path)
    class_count = validate_train_labels(train["labels"])
    if class_count != 286:
        raise ValueError(f"refit requires the full 286-class Elements cache, got {class_count}")

    model = ProjectionHead(384, 128).to(device)
    initialization = spec.get("initialization", "random")
    if initialization == "teacher":
        model.load_state_dict(
            torch.load(RUNTIME_PROJECTION, map_location=device, weights_only=False)
        )
    elif initialization != "random":
        raise ValueError(f"unsupported initialization: {initialization!r}")

    teacher_weight = float(spec.get("teacher_weight", 0.0))
    hidden_teacher_weight = float(spec.get("hidden_teacher_weight", 0.0))
    teacher_temperature = spec.get("teacher_temperature")
    if teacher_temperature is not None:
        teacher_temperature = float(teacher_temperature)
    teacher_assisted = teacher_weight > 0 or hidden_teacher_weight > 0

    started = time.perf_counter()
    objective = spec["objective"]
    if objective in {"proxy_ce", "distillation", "proxy_distillation"}:
        train_proxy_or_distillation(model, train, spec, device, class_count)
    elif objective == "leave_one_out":
        train_leave_one_out(model, train, spec, device, class_count)
    elif objective == "teacher_control":
        if initialization != "teacher":
            raise ValueError("teacher_control requires initialization=teacher")
    else:
        raise ValueError(f"unsupported objective: {objective!r}")
    elapsed = time.perf_counter() - started

    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    compatible = runtime_compatible(state) and class_count == 286
    if not compatible:
        raise ValueError("refit did not produce a runtime-compatible checkpoint")

    train_metrics = classification_metrics(model, train, train, device)

    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_payload = {
        "model_state_dict": state,
        "spec": spec,
        "train_cache_path": str(train_cache_path.resolve()),
        "train_cache_sha256": sha256_file(train_cache_path),
        "train_top1": train_metrics["top1"],
        "train_macro_top1": train_metrics["macro_top1"],
        "train_top3": train_metrics["top3"],
        "elapsed_seconds": elapsed,
        "runtime_compatible": compatible,
        "class_count": class_count,
        "validation_mode": "refit_full_cache_no_holdout",
    }
    latest_path = checkpoint_dir / "latest.pt"
    best_path = checkpoint_dir / "best.pt"
    torch.save(checkpoint_payload, latest_path)
    torch.save(checkpoint_payload, best_path)

    provenance = {
        "schema_version": "autoresearch-refit.v1",
        "spec_path": str(spec_path.resolve()),
        "spec_sha256": sha256_file(spec_path),
        "train_cache_path": str(train_cache_path.resolve()),
        "train_cache_sha256": sha256_file(train_cache_path),
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_latest_sha256": sha256_file(latest_path),
        "checkpoint_best_sha256": sha256_file(best_path),
        "class_count": class_count,
        "runtime_compatible": compatible,
        "objective": objective,
        "initialization": initialization,
        "seed": seed,
        "teacher_assisted": teacher_assisted,
        "teacher_weight": teacher_weight,
        "hidden_teacher_weight": hidden_teacher_weight,
        "teacher_temperature": teacher_temperature,
        "runtime_projection_path": str(RUNTIME_PROJECTION.resolve()),
        "runtime_projection_sha256": sha256_file(RUNTIME_PROJECTION),
        "train_top1": train_metrics["top1"],
        "train_macro_top1": train_metrics["macro_top1"],
        "train_top3": train_metrics["top3"],
        "elapsed_seconds": elapsed,
        "glyphs_allowed": False,
    }
    provenance_path = output_dir / "refit_provenance.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        **provenance,
        "checkpoint_latest_path": str(latest_path),
        "checkpoint_best_path": str(best_path),
        "provenance_path": str(provenance_path),
    }


def train_leave_one_out(
    model: ProjectionHead,
    train: dict[str, Any],
    spec: dict[str, Any],
    device: torch.device,
    class_count: int,
) -> None:
    features = train["features"].float().to(device)
    labels = train["labels"].long().to(device)
    counts = torch.bincount(labels, minlength=class_count).float()
    eligible = counts[labels] > 1
    if not bool(eligible.any()):
        raise ValueError("leave-one-out objective needs a non-singleton train class")

    epochs = int(spec.get("epochs", 100))
    temperature = float(spec.get("temperature", 0.1))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(spec.get("learning_rate", 5e-4)),
        weight_decay=float(spec.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    model.net[2].p = float(spec.get("training_dropout", 0.1))

    for _ in range(epochs):
        model.train()
        embeddings = model(features)
        sums = torch.zeros(class_count, 128, device=device).index_add_(0, labels, embeddings)
        prototypes = F.normalize(sums / counts[:, None], p=2, dim=1)
        logits = embeddings @ prototypes.T / temperature
        eligible_labels = labels[eligible]
        eligible_embeddings = embeddings[eligible]
        eligible_logits = logits[eligible]
        leave_one_out = F.normalize(
            (sums[eligible_labels] - eligible_embeddings)
            / (counts[eligible_labels, None] - 1.0),
            p=2,
            dim=1,
        )
        row_indices = torch.arange(len(eligible_labels), device=device)
        eligible_logits[row_indices, eligible_labels] = (
            eligible_embeddings * leave_one_out
        ).sum(dim=1) / temperature
        loss = F.cross_entropy(
            eligible_logits,
            eligible_labels,
            label_smoothing=float(spec.get("label_smoothing", 0.0)),
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()


def run_experiment(spec_path: Path, output_path: Path, data_root: Path) -> dict[str, Any]:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    seed = int(spec.get("seed", 42))
    configure_determinism(seed)
    random.seed(seed)

    if not torch.cuda.is_available():
        raise RuntimeError("autoresearch Elements experiments require CUDA")
    device = torch.device("cuda")

    train_path = resolve_train_cache(data_root, spec.get("train_cache", "raw"))
    validation_path = data_root / "features_val.pt"
    train = load_cache(train_path)
    validation = load_cache(validation_path)
    class_count = validate_labels(train["labels"], validation["labels"])

    model = ProjectionHead(384, 128).to(device)
    initialization = spec.get("initialization", "random")
    if initialization == "teacher":
        model.load_state_dict(
            torch.load(RUNTIME_PROJECTION, map_location=device, weights_only=False)
        )
    elif initialization != "random":
        raise ValueError(f"unsupported initialization: {initialization!r}")

    teacher_weight = float(spec.get("teacher_weight", 0.0))
    hidden_teacher_weight = float(spec.get("hidden_teacher_weight", 0.0))
    teacher_temperature = spec.get("teacher_temperature")
    if teacher_temperature is not None:
        teacher_temperature = float(teacher_temperature)
    teacher_assisted = teacher_weight > 0 or hidden_teacher_weight > 0

    started = time.perf_counter()
    objective = spec["objective"]
    if objective in {"proxy_ce", "distillation", "proxy_distillation"}:
        train_proxy_or_distillation(model, train, spec, device, class_count)
    elif objective == "leave_one_out":
        train_leave_one_out(model, train, spec, device, class_count)
    elif objective == "teacher_control":
        if initialization != "teacher":
            raise ValueError("teacher_control requires initialization=teacher")
    else:
        raise ValueError(f"unsupported objective: {objective!r}")
    elapsed = time.perf_counter() - started

    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    train_metrics = classification_metrics(model, train, train, device)
    validation_metrics = classification_metrics(model, train, validation, device)
    incumbent = spec.get("incumbent_score")
    score = validation_metrics["top1"]
    passed = incumbent is None or score > float(incumbent)
    compatible = runtime_compatible(state)
    passed = bool(passed and compatible and class_count == 286)

    checkpoint_path = output_path.with_suffix(".pt")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": state,
            "spec": spec,
            "train_cache_sha256": sha256_file(train_path),
            "validation_cache_sha256": sha256_file(validation_path),
            "validation_top1": score,
        },
        checkpoint_path,
    )

    result = {
        "schema_version": "autoresearch-evaluation.v1",
        "iteration": int(spec["iteration"]),
        "name": spec["name"],
        "track": spec["track"],
        "pass": passed,
        "score": score,
        "validation_top1": score,
        "validation_macro_top1": validation_metrics["macro_top1"],
        "validation_top3": validation_metrics["top3"],
        "train_top1": train_metrics["top1"],
        "runtime_compatible": compatible,
        "prototype_class_count": class_count,
        "objective": objective,
        "initialization": initialization,
        "train_cache": spec.get("train_cache", "raw"),
        "train_image_or_feature_count": len(train["labels"]),
        "validation_count": len(validation["labels"]),
        "elapsed_seconds": elapsed,
        "spec_path": str(spec_path),
        "checkpoint_path": str(checkpoint_path),
        "spec_sha256": sha256_file(spec_path),
        "train_cache_sha256": sha256_file(train_path),
        "validation_cache_sha256": sha256_file(validation_path),
        "test_evaluated": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def evaluate_cache(
    checkpoint_path: Path,
    prototype_cache_path: Path,
    evaluation_cache_path: Path,
    device_name: str,
    min_score: float,
    allow_source_overlap: bool = False,
) -> dict[str, Any]:
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cache evaluation requires CUDA for the selected device")

    prototype_cache = load_cache(prototype_cache_path)
    evaluation_cache = load_cache(evaluation_cache_path)
    prototype_sources = cache_source_paths(prototype_cache)
    evaluation_sources = cache_source_paths(evaluation_cache)
    prototype_evaluation_source_overlap_count: int | None = None
    if prototype_sources is None or evaluation_sources is None:
        if not allow_source_overlap:
            raise ValueError("prototype and evaluation caches must expose image_paths")
        source_disjoint_enforced = False
        source_disjoint_check = "unavailable"
    else:
        prototype_evaluation_source_overlap_count = len(prototype_sources & evaluation_sources)
        source_disjoint_enforced = not allow_source_overlap
        source_disjoint_check = "allowed" if allow_source_overlap else "enforced"
        if prototype_evaluation_source_overlap_count and not allow_source_overlap:
            raise ValueError(
                "prototype and evaluation caches share "
                f"{prototype_evaluation_source_overlap_count} source images"
            )

    device = torch.device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = load_checkpoint_state_dict(checkpoint)
    model = ProjectionHead(384, 128).to(device)
    model.load_state_dict(state_dict)

    prototype_class_count = validate_labels(prototype_cache["labels"], evaluation_cache["labels"])

    metrics = classification_metrics(model, prototype_cache, evaluation_cache, device)
    compatible = runtime_compatible({key: value.detach().cpu() for key, value in model.state_dict().items()})
    score = metrics["top1"]
    passed = bool(compatible and score >= min_score)

    result = {
        "schema_version": "autoresearch-cache-eval.v1",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "prototype_cache_path": str(prototype_cache_path),
        "prototype_cache_sha256": sha256_file(prototype_cache_path),
        "evaluation_cache_path": str(evaluation_cache_path),
        "evaluation_cache_sha256": sha256_file(evaluation_cache_path),
        "runtime_compatible": compatible,
        "pass": passed,
        "score": score,
        "train_prototype_top1": metrics["top1"],
        "train_prototype_macro_top1": metrics["macro_top1"],
        "train_prototype_top3": metrics["top3"],
        "prototype_count": int(prototype_cache["labels"].numel()),
        "evaluation_count": int(evaluation_cache["labels"].numel()),
        "prototype_class_count": prototype_class_count,
        "evaluation_class_count": int(evaluation_cache["labels"].unique().numel()),
        "prototype_evaluation_source_overlap_count": prototype_evaluation_source_overlap_count,
        "source_disjoint_enforced": source_disjoint_enforced,
        "source_disjoint_check": source_disjoint_check,
        "min_score": float(min_score),
    }
    return result

def evaluate_test(checkpoint_path: Path, data_root: Path) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("test evaluation requires CUDA")
    device = torch.device("cuda")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    spec = checkpoint["spec"]
    train = load_cache(resolve_train_cache(data_root, spec.get("train_cache", "raw")))
    test = load_cache(data_root / "features_test.pt")
    model = ProjectionHead(384, 128).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    result = classification_metrics(model, train, test, device)
    return {
        "checkpoint": str(checkpoint_path),
        "track": spec["track"],
        "name": spec["name"],
        "test_top1": result["top1"],
        "test_macro_top1": result["macro_top1"],
        "test_top3": result["top3"],
        "test_count": len(test["labels"]),
        "test_class_count": len(test["labels"].unique()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--spec", type=Path, required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)

    test_parser = subparsers.add_parser("test")
    test_parser.add_argument("--checkpoint", type=Path, required=True)
    test_parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)

    refit_parser = subparsers.add_parser("refit")
    refit_parser.add_argument("--spec", type=Path, required=True)
    refit_parser.add_argument("--train-cache", type=Path, required=True)
    refit_parser.add_argument("--output-dir", type=Path, required=True)
    refit_parser.add_argument("--device", default="cuda")

    evaluate_cache_parser = subparsers.add_parser("evaluate-cache")
    evaluate_cache_parser.add_argument("--checkpoint", type=Path, required=True)
    evaluate_cache_parser.add_argument("--prototype-cache", type=Path, required=True)
    evaluate_cache_parser.add_argument("--evaluation-cache", type=Path, required=True)
    evaluate_cache_parser.add_argument("--device", default="cpu")
    evaluate_cache_parser.add_argument("--min-score", type=float, default=0.0)
    evaluate_cache_parser.add_argument(
        "--allow-source-overlap",
        action="store_true",
        help="Allow explicit posthoc overlap between prototype and evaluation source images.",
    )
    args = parser.parse_args(argv)
    if args.command == "run":
        print(json.dumps(run_experiment(args.spec, args.output, args.data_root), indent=2))
    elif args.command == "refit":
        print(
            json.dumps(
                refit_experiment(args.spec, args.train_cache, args.output_dir, args.device),
                indent=2,
            )
        )
    elif args.command == "evaluate-cache":
        print(
            json.dumps(
                evaluate_cache(
                    args.checkpoint,
                    args.prototype_cache,
                    args.evaluation_cache,
                    args.device,
                    args.min_score,
                    args.allow_source_overlap,
                ),                indent=2,
            )
        )
    else:
        print(json.dumps(evaluate_test(args.checkpoint, args.data_root), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

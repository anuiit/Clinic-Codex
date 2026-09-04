#!/usr/bin/env python3
"""
Train projection head with prototypical network episodic training
on pre-computed DINOv2 features.

Step 1: python -m codex_pipeline.scripts.precompute_embeddings
Step 2: python -m codex_pipeline.scripts.train

This only trains the small projection head (~50K params) on cached
384-dim vectors. No images are loaded, no backbone forward pass needed.
Epochs take seconds, not hours.
"""

import argparse
import hashlib
import json
import shutil
import sys
import time
from collections.abc import Mapping
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from codex_pipeline.data.cached_dataset import (
    CachedFeatureDataset,
    CachedEpisodicSampler,
    collate_cached_episodes,
)
from codex_pipeline.determinism import configure_determinism
from codex_pipeline.models.projection_head import ProjectionHead, get_device
from codex_pipeline.models.prototypical import PrototypicalLoss, compute_prototypes




def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_initial_projection(
    model: ProjectionHead,
    path: str | Path,
) -> dict[str, str]:
    """Load and validate an exact ProjectionHead state before optimization."""
    source_path = Path(path)
    if not source_path.is_file():
        raise SystemExit(f"Initial projection does not exist: {source_path}")

    source_sha256 = sha256_file(source_path)
    try:
        payload = torch.load(source_path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise SystemExit(f"Could not load initial projection {source_path}: {exc}") from exc

    source_format = "raw_state_dict"
    state_dict = payload
    if isinstance(payload, Mapping) and "model_state_dict" in payload:
        source_format = "checkpoint_model_state_dict"
        state_dict = payload["model_state_dict"]

    if not isinstance(state_dict, Mapping):
        raise SystemExit(
            "Initial projection must be a raw state_dict or a checkpoint "
            "containing model_state_dict"
        )

    expected = model.state_dict()
    expected_keys = set(expected)
    actual_keys = set(state_dict)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        unexpected = sorted(actual_keys - expected_keys)
        raise SystemExit(
            "Initial projection keys do not exactly match ProjectionHead; "
            f"missing={missing}, unexpected={unexpected}"
        )

    for key, expected_tensor in expected.items():
        value = state_dict[key]
        if not isinstance(value, torch.Tensor):
            raise SystemExit(f"Initial projection value for {key!r} is not a tensor")
        if value.shape != expected_tensor.shape:
            raise SystemExit(
                f"Initial projection shape mismatch for {key!r}: "
                f"expected {tuple(expected_tensor.shape)}, got {tuple(value.shape)}"
            )
        if not torch.isfinite(value).all().item():
            raise SystemExit(f"Initial projection contains non-finite values in {key!r}")

    # strict=True is deliberate even after the diagnostic validation above.
    model.load_state_dict(state_dict, strict=True)
    return {
        "path": str(source_path.resolve()),
        "sha256": source_sha256,
        "format": source_format,
    }


def format_class_counts(dataset, class_names) -> str:
    """Return a compact class=count summary for diagnostics."""
    if not dataset.classes:
        return "none"

    parts = []
    for class_label in dataset.classes:
        name = class_names.get(class_label, str(class_label))
        parts.append(f"{name}={len(dataset.class_to_indices[class_label])}")
    return ", ".join(parts)


def require_episode_support(dataset, split_name: str, n_way: int, k_shot: int, q_queries: int, class_names) -> None:
    """Fail early if a split cannot form leakage-free n-way episodes.

    Sparse classes remain in the feature set and receive prototypes at export
    time. They are deliberately excluded from episodic head training when
    they do not have distinct support and query examples.
    """
    episode_classes = [
        class_label
        for class_label in dataset.classes
        if len(dataset.class_to_indices[class_label]) >= k_shot + q_queries
    ]
    if len(episode_classes) >= n_way:
        return

    counts = format_class_counts(dataset, class_names)
    raise SystemExit(
        "\nERROR: Not enough approved training data for leakage-free episodic training.\n"
        f"Split '{split_name}' has {len(episode_classes)} classes with at least "
        f"k_shot + q_queries = {k_shot + q_queries} examples; n_way={n_way} is required.\n"
        f"Class counts in this split: {counts}\n\n"
        "Sparse classes remain prototype-only, so lower n_way/k_shot/q_queries "
        "for a smoke run or approve more examples for full episodic training."
    )


def train_one_epoch(model, dataloader, criterion, optimizer, device, n_way, k_shot, q_queries):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    num_episodes = 0

    for batch in dataloader:
        s_feats, s_labs, q_feats, q_labs = collate_cached_episodes(
            batch, n_way, k_shot, q_queries,
        )
        s_feats = s_feats.to(device)
        q_feats = q_feats.to(device)
        s_labs = s_labs.to(device)
        q_labs = q_labs.to(device)

        s_emb = model(s_feats)
        q_emb = model(q_feats)

        result = criterion(s_emb, s_labs, q_emb, q_labs)
        loss = result["loss"]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        total_acc += result["accuracy"].item()
        num_episodes += 1

    return total_loss / num_episodes, total_acc / num_episodes


@torch.no_grad()
def collect_embeddings(model, dataloader, device):
    model.eval()
    embeddings = []
    labels = []

    with torch.no_grad():
        for feats, batch_labels in dataloader:
            feats = feats.to(device)
            batch_embeddings = model(feats).cpu()
            embeddings.append(batch_embeddings)
            labels.append(batch_labels.cpu())

    return torch.cat(embeddings, dim=0), torch.cat(labels, dim=0)


def evaluate_global_prototype_validation(
    train_embeddings,
    train_labels,
    val_embeddings,
    val_labels,
    temperature: float,
):
    train_prototype_labels = train_labels.unique(sorted=True)
    missing_labels = sorted(set(val_labels.tolist()) - set(train_prototype_labels.tolist()))
    if missing_labels:
        raise SystemExit(
            "Validation labels missing from train prototypes: "
            + ", ".join(str(label) for label in missing_labels)
        )

    prototypes = compute_prototypes(train_embeddings, train_labels)
    label_to_proto_idx = {label.item(): idx for idx, label in enumerate(train_prototype_labels)}
    target = torch.tensor([label_to_proto_idx[label.item()] for label in val_labels], dtype=torch.long)
    logits = torch.mm(val_embeddings, prototypes.t()) / temperature
    loss = F.cross_entropy(logits, target)
    predictions = logits.argmax(dim=-1)
    predicted_labels = train_prototype_labels[predictions]
    accuracy = (predicted_labels == val_labels).float().mean()
    return loss.item(), accuracy.item()


def evaluate_global_validation(model, train_loader, val_loader, device, temperature: float):
    train_embeddings, train_labels = collect_embeddings(model, train_loader, device)
    val_embeddings, val_labels = collect_embeddings(model, val_loader, device)
    return evaluate_global_prototype_validation(
        train_embeddings,
        train_labels,
        val_embeddings,
        val_labels,
        temperature,
    )


def main():
    parser = argparse.ArgumentParser(description="Train projection head on cached features")
    parser.add_argument("--config", default="codex_pipeline/config/default.yaml")
    parser.add_argument("--features", default="./precomputed/features.pt")
    parser.add_argument(
        "--validation-features",
        default=None,
        help="Optional cache reserved for global validation; disables the generated train/val split.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Explicit checkpoint output directory. Overrides paths.checkpoint_dir from config.",
    )
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument("--resume", default=None)
    initialization.add_argument(
        "--init-projection",
        default=None,
        help="Raw ProjectionHead state_dict or checkpoint containing model_state_dict.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Evaluate --init-projection on global validation without optimizer updates.",
    )
    parser.add_argument("--noise-std", type=float, default=0.0,
                        help="Gaussian noise std for feature augmentation (0=off)")
    parser.add_argument("--mixup-prob", type=float, default=0.0,
                        help="Feature-level mixup probability (0=off)")
    args = parser.parse_args()
    if args.eval_only and not args.init_projection:
        parser.error("--eval-only requires --init-projection")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train_cfg = cfg["training"]
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    paths_cfg = cfg["paths"]

    device = get_device(train_cfg["device"])
    print(f"Device: {device}")
    configure_determinism(int(train_cfg["seed"]))

    # --- Load cached features ---
    print(f"Loading cached features from {args.features}...")
    if args.validation_features:
        train_dataset, data_info = CachedFeatureDataset.from_file(
            args.features,
            split=None,
            noise_std=args.noise_std,
            feature_mixup_prob=args.mixup_prob,
        )
        prototype_train_dataset, _ = CachedFeatureDataset.from_file(args.features, split=None)
        val_dataset, _ = CachedFeatureDataset.from_file(args.validation_features, split=None)
    else:
        split_strategy = data_cfg.get("split_strategy", "per_class")
        train_dataset, data_info = CachedFeatureDataset.from_file(
            args.features,
            split="train",
            split_strategy=split_strategy,
            val_fraction=data_cfg["val_fraction"],
            seed=int(train_cfg["seed"]),
            noise_std=args.noise_std,
            feature_mixup_prob=args.mixup_prob,
        )
        prototype_train_dataset, _ = CachedFeatureDataset.from_file(
            args.features,
            split="train",
            split_strategy=split_strategy,
            val_fraction=data_cfg["val_fraction"],
            seed=int(train_cfg["seed"]),
        )
        val_dataset, _ = CachedFeatureDataset.from_file(
            args.features,
            split="val",
            split_strategy=split_strategy,
            val_fraction=data_cfg["val_fraction"],
            seed=int(train_cfg["seed"]),
        )

    hidden_dim = data_info["hidden_dim"]
    class_names = data_info["class_names"]

    print(f"  Train: {len(train_dataset)} vectors | Val: {len(val_dataset)} vectors")
    print(f"  Classes: {train_dataset.num_classes} | Feature dim: {hidden_dim}")

    # --- Validation and episodic loaders ---
    n_way = train_cfg["n_way"]
    k_shot = train_cfg["k_shot"]
    q_queries = train_cfg["q_queries"]
    prototype_train_loader = DataLoader(prototype_train_dataset, batch_size=256, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=0)

    # --- Model (projection head only) ---
    model = ProjectionHead(
        input_dim=hidden_dim,
        embedding_dim=model_cfg["embedding_dim"],
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Projection head params: {num_params:,}")

    init_projection = None
    if args.init_projection:
        init_projection = load_initial_projection(model, args.init_projection)
        print(
            "Loaded initial projection: "
            f"{init_projection['path']} ({init_projection['format']}, "
            f"sha256={init_projection['sha256']})"
        )

    # --- Logging ---
    ckpt_dir = Path(args.checkpoint_dir or paths_cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    num_epochs = train_cfg["num_epochs"]
    warmup_epochs = train_cfg["warmup_epochs"]

    if args.eval_only:
        val_loss, val_acc = evaluate_global_validation(
            model,
            prototype_train_loader,
            val_loader,
            device,
            train_cfg["temperature"],
        )
        ckpt_data = {
            "epoch": -1,
            "model_state_dict": model.state_dict(),
            # NaN is explicitly descriptive here and keeps legacy evaluators,
            # which format these fields numerically, compatible with epoch -1.
            "train_loss": float("nan"),
            "train_acc": float("nan"),
            "val_loss": val_loss,
            "val_acc": val_acc,
            "best_val_acc": val_acc,
            "config": cfg,
            "hidden_dim": hidden_dim,
            "validation_mode": "global_286_way_train_prototypes",
            "eval_only": True,
            "init_projection": init_projection,
        }
        # Copying one serialization makes the fixed-selection aliases byte-identical.
        torch.save(ckpt_data, ckpt_dir / "latest.pt")
        shutil.copyfile(ckpt_dir / "latest.pt", ckpt_dir / "best.pt")
        manifest = {
            "schema_version": "baseline-training.v1",
            "config_path": str(Path(args.config).resolve()),
            "config_sha256": sha256_file(args.config),
            "features_path": str(Path(args.features).resolve()),
            "features_sha256": sha256_file(args.features),
            "validation_features_path": (
                str(Path(args.validation_features).resolve()) if args.validation_features else None
            ),
            "validation_features_sha256": (
                sha256_file(args.validation_features) if args.validation_features else None
            ),
            "seed": int(train_cfg["seed"]),
            "noise_std": args.noise_std,
            "mixup_prob": args.mixup_prob,
            "epoch": -1,
            "eval_only": True,
            "init_projection": init_projection,
            "best_val_acc": val_acc,
            "checkpoints": {
                "latest": sha256_file(ckpt_dir / "latest.pt"),
                "best": sha256_file(ckpt_dir / "best.pt"),
            },
        }
        (ckpt_dir / "training_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"\nEvaluation only. Global val accuracy: {val_acc:.3f}")
        print(f"Checkpoints: {ckpt_dir}")
        return

    if num_epochs <= 0:
        raise SystemExit("training.num_epochs must be positive unless --eval-only is used")
    require_episode_support(train_dataset, "train", n_way, k_shot, q_queries, class_names)
    train_sampler = CachedEpisodicSampler(
        train_dataset, n_way, k_shot, q_queries, train_cfg["episodes_per_epoch"],
    )
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=0)

    # --- Loss, optimizer, scheduler ---
    criterion = PrototypicalLoss(temperature=train_cfg["temperature"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )

    if train_cfg["lr_scheduler"] == "cosine":
        cosine_epochs = num_epochs - warmup_epochs
        if cosine_epochs <= 0:
            raise SystemExit(
                "training.num_epochs must be greater than training.warmup_epochs "
                "for the cosine scheduler"
            )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cosine_epochs,
        )
    else:
        scheduler = None

    warmup_scheduler = None
    if warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=warmup_epochs,
        )

    # --- Resume ---
    start_epoch = 0
    best_val_acc = float("-inf")

    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if (
            warmup_scheduler is not None
            and ckpt.get("warmup_scheduler_state_dict") is not None
        ):
            warmup_scheduler.load_state_dict(ckpt["warmup_scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_acc = ckpt.get("best_val_acc", 0.0)

    # --- Training loop ---
    print(f"\nTraining for {num_epochs} epochs...")
    print(f"  {n_way}-way {k_shot}-shot, {q_queries} queries")
    print(f"  {train_cfg['episodes_per_epoch']} train episodes, "
          f"{cfg['evaluation']['num_eval_episodes']} val episodes\n")

    for epoch in range(start_epoch, num_epochs):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            n_way, k_shot, q_queries,
        )

        val_loss, val_acc = evaluate_global_validation(
            model,
            prototype_train_loader,
            val_loader,
            device,
            train_cfg["temperature"],
        )

        if warmup_scheduler and epoch < warmup_epochs:
            warmup_scheduler.step()
        elif scheduler:
            scheduler.step()

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch+1:>3}/{num_epochs} | "
            f"Train L:{train_loss:.4f} A:{train_acc:.3f} | "
            f"Val L:{val_loss:.4f} A:{val_acc:.3f} | "
            f"LR:{lr:.6f} | {elapsed:.1f}s"
        )

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc

        ckpt_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "warmup_scheduler_state_dict": (
                warmup_scheduler.state_dict() if warmup_scheduler is not None else None
            ),
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "best_val_acc": best_val_acc,
            "config": cfg,
            "hidden_dim": hidden_dim,
            "validation_mode": "global_286_way_train_prototypes",
            "eval_only": False,
            "init_projection": init_projection,
        }

        torch.save(ckpt_data, ckpt_dir / "latest.pt")
        if is_best:
            torch.save(ckpt_data, ckpt_dir / "best.pt")
            print(f"  -> New best global val accuracy: {val_acc:.3f}")

        if (epoch + 1) % 10 == 0:
            torch.save(ckpt_data, ckpt_dir / f"epoch_{epoch+1:03d}.pt")

    manifest = {
        "schema_version": "baseline-training.v1",
        "config_path": str(Path(args.config).resolve()),
        "config_sha256": sha256_file(args.config),
        "features_path": str(Path(args.features).resolve()),
        "features_sha256": sha256_file(args.features),
        "seed": int(train_cfg["seed"]),
        "noise_std": args.noise_std,
        "mixup_prob": args.mixup_prob,
        "epoch": num_epochs - 1,
        "eval_only": False,
        "init_projection": init_projection,
        "best_val_acc": best_val_acc,
        "checkpoints": {
            "latest": sha256_file(ckpt_dir / "latest.pt"),
            "best": sha256_file(ckpt_dir / "best.pt"),
        },
    }
    (ckpt_dir / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"\nDone. Best val accuracy: {best_val_acc:.3f}")
    print(f"Checkpoints: {ckpt_dir}")


if __name__ == "__main__":
    main()

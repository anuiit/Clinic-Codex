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
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
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
from codex_pipeline.models.prototypical import PrototypicalLoss


class ProjectionHead(nn.Module):
    """Small MLP that maps frozen DINOv2 features to an embedding space."""

    def __init__(self, input_dim: int = 384, embedding_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim, embedding_dim),
        )

    def forward(self, x):
        out = self.net(x)
        return F.normalize(out, p=2, dim=-1)


def get_device(device_cfg: str) -> torch.device:
    if device_cfg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_cfg)


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
def evaluate(model, dataloader, criterion, device, n_way, k_shot, q_queries):
    model.eval()
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
        total_loss += result["loss"].item()
        total_acc += result["accuracy"].item()
        num_episodes += 1

    return total_loss / num_episodes, total_acc / num_episodes


def main():
    parser = argparse.ArgumentParser(description="Train projection head on cached features")
    parser.add_argument("--config", default="codex_pipeline/config/default.yaml")
    parser.add_argument("--features", default="./precomputed/features_aug.pt")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--noise-std", type=float, default=0.02,
                        help="Gaussian noise std for feature augmentation (0=off)")
    parser.add_argument("--mixup-prob", type=float, default=0.3,
                        help="Feature-level mixup probability (0=off)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train_cfg = cfg["training"]
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    paths_cfg = cfg["paths"]

    device = get_device(train_cfg["device"])
    print(f"Device: {device}")
    torch.manual_seed(train_cfg["seed"])

    # --- Load cached features ---
    print(f"Loading cached features from {args.features}...")
    train_dataset, data_info = CachedFeatureDataset.from_file(
        args.features, split="train", val_fraction=data_cfg["val_fraction"],
        noise_std=args.noise_std,
        feature_mixup_prob=args.mixup_prob,
    )
    val_dataset, _ = CachedFeatureDataset.from_file(
        args.features, split="val", val_fraction=data_cfg["val_fraction"],
        # No augmentation on val
    )

    hidden_dim = data_info["hidden_dim"]
    class_names = data_info["class_names"]

    print(f"  Train: {len(train_dataset)} vectors | Val: {len(val_dataset)} vectors")
    print(f"  Classes: {train_dataset.num_classes} | Feature dim: {hidden_dim}")

    # --- Episodic samplers ---
    n_way = train_cfg["n_way"]
    k_shot = train_cfg["k_shot"]
    q_queries = train_cfg["q_queries"]

    train_sampler = CachedEpisodicSampler(
        train_dataset, n_way, k_shot, q_queries, train_cfg["episodes_per_epoch"],
    )
    val_sampler = CachedEpisodicSampler(
        val_dataset, n_way, k_shot, q_queries, cfg["evaluation"]["num_eval_episodes"],
    )

    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, num_workers=0)

    # --- Model (projection head only) ---
    model = ProjectionHead(
        input_dim=hidden_dim,
        embedding_dim=model_cfg["embedding_dim"],
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Projection head params: {num_params:,}")

    # --- Loss, optimizer, scheduler ---
    criterion = PrototypicalLoss(temperature=train_cfg["temperature"])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )

    num_epochs = train_cfg["num_epochs"]
    warmup_epochs = train_cfg["warmup_epochs"]

    if train_cfg["lr_scheduler"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs - warmup_epochs,
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
    best_val_acc = 0.0

    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_acc = ckpt.get("best_val_acc", 0.0)

    # --- Logging ---
    ckpt_dir = Path(paths_cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

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

        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device, n_way, k_shot, q_queries,
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
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "best_val_acc": best_val_acc,
            "config": cfg,
            "hidden_dim": hidden_dim,
        }

        torch.save(ckpt_data, ckpt_dir / "latest.pt")
        if is_best:
            torch.save(ckpt_data, ckpt_dir / "best.pt")
            print(f"  -> New best val accuracy: {val_acc:.3f}")

        if (epoch + 1) % 10 == 0:
            torch.save(ckpt_data, ckpt_dir / f"epoch_{epoch+1:03d}.pt")

    print(f"\nDone. Best val accuracy: {best_val_acc:.3f}")
    print(f"Checkpoints: {ckpt_dir}")


if __name__ == "__main__":
    main()

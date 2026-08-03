#!/usr/bin/env python3
"""
Evaluate trained projection head on cached DINOv2 features.

Computes:
- Few-shot accuracy at k=1, 3, 5
- Per-class prototype quality
- Confusion analysis for closest class pairs
- Exports prototype embeddings for inference

Usage:
    python -m codex_pipeline.scripts.evaluate --checkpoint checkpoints/best.pt
    python -m codex_pipeline.scripts.evaluate --checkpoint checkpoints/best.pt --export-prototypes
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from codex_pipeline.data.cached_dataset import (
    CachedFeatureDataset,
    CachedEpisodicSampler,
    collate_cached_episodes,
)
from codex_pipeline.models.prototypical import PrototypicalLoss, compute_prototypes


class ProjectionHead(nn.Module):
    """Must match the one in train.py."""

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


@torch.no_grad()
def embed_all(model, dataset, device, batch_size=512):
    """Project all cached features through the projection head."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_embeddings = []
    all_labels = []

    for features, labels in loader:
        features = features.to(device)
        emb = model(features)
        all_embeddings.append(emb.cpu())
        all_labels.append(labels)

    return torch.cat(all_embeddings), torch.cat(all_labels)


@torch.no_grad()
def evaluate_few_shot(model, dataset, criterion, device, n_way, k_shot, q_queries, num_episodes):
    """Run episodic evaluation at a given k-shot."""
    model.eval()
    sampler = CachedEpisodicSampler(dataset, n_way, k_shot, q_queries, num_episodes)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)

    total_acc = 0.0
    num = 0

    for batch in loader:
        s_feats, s_labs, q_feats, q_labs = collate_cached_episodes(
            batch, n_way, k_shot, q_queries,
        )
        s_emb = model(s_feats.to(device))
        q_emb = model(q_feats.to(device))
        result = criterion(s_emb, s_labs.to(device), q_emb, q_labs.to(device))
        total_acc += result["accuracy"].item()
        num += 1

    return total_acc / num


def analyze_prototypes(embeddings, labels, class_names):
    """Analyze prototype quality: inter-class distances, intra-class variance."""
    prototypes = compute_prototypes(embeddings, labels)
    unique_labels = labels.unique(sorted=True)

    # Inter-class similarity matrix
    sim_matrix = torch.mm(prototypes, prototypes.t())

    # Find most confusable pairs
    n = sim_matrix.size(0)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((sim_matrix[i, j].item(), i, j))
    pairs.sort(reverse=True)

    # Intra-class variance (1 - cosine_sim to centroid)
    class_variances = {}
    for idx, label in enumerate(unique_labels):
        mask = labels == label
        class_emb = embeddings[mask]
        if class_emb.size(0) > 1:
            centroid = F.normalize(class_emb.mean(dim=0, keepdim=True), p=2, dim=-1)
            sims = torch.mm(class_emb, centroid.t()).squeeze()
            class_variances[label.item()] = (1.0 - sims).mean().item()

    return {
        "prototypes": prototypes,
        "similarity_matrix": sim_matrix,
        "confusable_pairs": pairs[:20],
        "class_variances": class_variances,
        "class_labels": unique_labels,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate projection head")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/best.pt")
    parser.add_argument("--features", type=str, default="./precomputed/features_aug.pt")
    parser.add_argument("--export-prototypes", action="store_true")
    parser.add_argument("--num-episodes", type=int, default=500)
    args = parser.parse_args()

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    train_cfg = cfg["training"]
    model_cfg = cfg["model"]
    paths_cfg = cfg["paths"]
    hidden_dim = ckpt["hidden_dim"]

    device = get_device(train_cfg["device"])
    print(f"Device: {device}")
    print(f"Checkpoint: epoch {ckpt['epoch']+1}")
    print(f"  Train acc: {ckpt['train_acc']:.3f} | Val acc: {ckpt['val_acc']:.3f}")

    # --- Load cached features (no augmentation for eval) ---
    print(f"\nLoading features from {args.features}...")
    dataset, data_info = CachedFeatureDataset.from_file(args.features, split=None)
    class_names = data_info["class_names"]
    print(f"  {len(dataset)} features across {dataset.num_classes} classes")

    # --- Model ---
    model = ProjectionHead(
        input_dim=hidden_dim,
        embedding_dim=model_cfg["embedding_dim"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # --- Few-shot evaluation ---
    criterion = PrototypicalLoss(temperature=train_cfg["temperature"])
    n_way = train_cfg["n_way"]
    q_queries = train_cfg["q_queries"]

    print(f"\nFew-shot evaluation ({n_way}-way, {args.num_episodes} episodes):")
    for k in cfg["evaluation"]["k_shot_values"]:
        acc = evaluate_few_shot(
            model, dataset, criterion, device,
            n_way, k, q_queries, args.num_episodes,
        )
        print(f"  {k}-shot accuracy: {acc:.3f}")

    # --- Compute all embeddings for prototype analysis ---
    print("\nComputing embeddings for all features...")
    embeddings, labels = embed_all(model, dataset, device)
    print(f"  Embeddings shape: {embeddings.shape}")

    analysis = analyze_prototypes(embeddings, labels, class_names)

    # --- Confusable pairs ---
    print(f"\nTop 10 most confusable class pairs (cosine similarity):")
    for sim, i, j in analysis["confusable_pairs"][:10]:
        lab_i = analysis["class_labels"][i].item()
        lab_j = analysis["class_labels"][j].item()
        name_i = class_names.get(lab_i, str(lab_i))
        name_j = class_names.get(lab_j, str(lab_j))
        print(f"  {name_i:<25} <-> {name_j:<25} sim={sim:.3f}")

    # --- Least consistent classes ---
    sorted_var = sorted(analysis["class_variances"].items(), key=lambda x: -x[1])
    print(f"\nTop 10 least consistent classes (highest intra-class variance):")
    for label, var in sorted_var[:10]:
        name = class_names.get(label, str(label))
        count = (labels == label).sum().item()
        print(f"  {name:<30} variance={var:.4f}  (n={count})")

    # --- Most consistent classes ---
    sorted_var_best = sorted(analysis["class_variances"].items(), key=lambda x: x[1])
    print(f"\nTop 10 most consistent classes (lowest intra-class variance):")
    for label, var in sorted_var_best[:10]:
        name = class_names.get(label, str(label))
        count = (labels == label).sum().item()
        print(f"  {name:<30} variance={var:.4f}  (n={count})")

    # --- Overall stats ---
    all_vars = list(analysis["class_variances"].values())
    print(f"\nOverall prototype quality:")
    print(f"  Mean intra-class variance:   {sum(all_vars)/len(all_vars):.4f}")
    print(f"  Median intra-class variance: {sorted(all_vars)[len(all_vars)//2]:.4f}")

    # Average inter-class similarity (excluding diagonal)
    sim_mat = analysis["similarity_matrix"]
    n = sim_mat.size(0)
    mask = ~torch.eye(n, dtype=torch.bool)
    avg_inter = sim_mat[mask].mean().item()
    print(f"  Mean inter-class similarity: {avg_inter:.4f}")
    print(f"  Separation ratio:            {avg_inter / (sum(all_vars)/len(all_vars)):.2f}x")

    # --- Export prototypes ---
    if args.export_prototypes:
        proto_dir = Path(paths_cfg["prototype_dir"])
        proto_dir.mkdir(parents=True, exist_ok=True)

        # Also compute per-class count and variance for metadata
        class_meta = {}
        for label in analysis["class_labels"].tolist():
            name = class_names.get(label, str(label))
            count = (labels == label).sum().item()
            var = analysis["class_variances"].get(label, 0.0)
            class_meta[label] = {"name": name, "count": count, "variance": var}

        proto_data = {
            "prototypes": analysis["prototypes"],
            "class_labels": analysis["class_labels"],
            "class_names": class_names,
            "class_meta": class_meta,
            "embedding_dim": model_cfg["embedding_dim"],
            "hidden_dim": hidden_dim,
            "model_state_dict": ckpt["model_state_dict"],
        }
        proto_path = proto_dir / "prototypes.pt"
        torch.save(proto_data, proto_path)
        print(f"\nPrototypes exported to {proto_path}")
        print(f"  {len(analysis['class_labels'])} class prototypes ({model_cfg['embedding_dim']}-dim)")
        print(f"  Includes projection head weights for inference")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Visualize embedding quality to demonstrate model accuracy.

Generates:
1. t-SNE of all embeddings (top N classes, color-coded)
2. Zoomed t-SNE of the most confusable pairs
3. Inter-class similarity heatmap (top classes)
4. Per-class variance distribution
5. Example nearest-neighbor retrievals with actual images

Usage:
    python -m codex_pipeline.scripts.visualize
"""

import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.manifold import TSNE
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from codex_pipeline.data.cached_dataset import CachedFeatureDataset
from codex_pipeline.models.prototypical import compute_prototypes


class ProjectionHead(nn.Module):
    def __init__(self, input_dim=384, embedding_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim, embedding_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), p=2, dim=-1)


def load_thumbnail(path, size=40):
    """Load and resize an image for embedding in plots."""
    try:
        img = Image.open(path).convert("RGB")
        img.thumbnail((size, size))
        return np.array(img)
    except Exception:
        return np.zeros((size, size, 3), dtype=np.uint8)


def main():
    out_dir = Path("./visualizations")
    out_dir.mkdir(exist_ok=True)

    # Load model and features
    ckpt = torch.load("./checkpoints/best.pt", map_location="cpu", weights_only=False)
    hidden_dim = ckpt["hidden_dim"]
    embedding_dim = ckpt["config"]["model"]["embedding_dim"]

    model = ProjectionHead(input_dim=hidden_dim, embedding_dim=embedding_dim)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    data = torch.load("./precomputed/features_aug.pt", map_location="cpu", weights_only=False)
    features = data["features"]
    labels = data["labels"]
    image_paths = data["image_paths"]
    class_names = data["class_names"]
    is_augmented = data.get("is_augmented", [False] * len(features))

    # Compute embeddings
    with torch.no_grad():
        embeddings = model(features)
    embeddings = embeddings.numpy()
    labels_np = labels.numpy()

    # --- Only use ORIGINAL images for cleaner visualization ---
    orig_mask = np.array([not aug for aug in is_augmented])
    orig_embeddings = embeddings[orig_mask]
    orig_labels = labels_np[orig_mask]
    orig_paths = [p for p, aug in zip(image_paths, is_augmented) if not aug]

    print(f"Original images: {len(orig_embeddings)}")
    print(f"Total embeddings: {len(embeddings)}")

    # Pick top 20 classes by count for visualization
    unique, counts = np.unique(orig_labels, return_counts=True)
    top20_idx = np.argsort(-counts)[:20]
    top20_classes = unique[top20_idx]
    top20_names = [class_names.get(int(c), str(c)) for c in top20_classes]

    mask_top20 = np.isin(orig_labels, top20_classes)
    emb_top20 = orig_embeddings[mask_top20]
    lab_top20 = orig_labels[mask_top20]

    # ===== PLOT 1: t-SNE of top 20 classes =====
    print("Computing t-SNE (top 20 classes)...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    coords = tsne.fit_transform(emb_top20)

    fig, ax = plt.subplots(figsize=(16, 12))
    cmap = plt.cm.get_cmap("tab20", 20)

    for i, cls in enumerate(top20_classes):
        mask = lab_top20 == cls
        name = class_names.get(int(cls), str(cls))
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[cmap(i)], label=name, s=20, alpha=0.7,
        )

    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, markerscale=2)
    ax.set_title("t-SNE of Element Embeddings (Top 20 Classes)", fontsize=14)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig(out_dir / "01_tsne_top20.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_dir / '01_tsne_top20.png'}")

    # ===== PLOT 2: Confusable pairs zoomed =====
    confusable_pairs = [
        ("itzcuintli", "coyotl"),
        ("chiquihuitl", "petlatl"),
        ("toztli", "ihuitl"),
        ("ohuatl", "acatl"),
        ("popoca", "tlatoa"),
    ]

    # Reverse lookup: name -> label
    name_to_label = {v: k for k, v in class_names.items()}

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    for ax, (name_a, name_b) in zip(axes, confusable_pairs):
        lab_a = name_to_label.get(name_a)
        lab_b = name_to_label.get(name_b)
        if lab_a is None or lab_b is None:
            ax.set_visible(False)
            continue

        mask_ab = (orig_labels == lab_a) | (orig_labels == lab_b)
        emb_ab = orig_embeddings[mask_ab]
        lab_ab = orig_labels[mask_ab]

        if len(emb_ab) < 5:
            ax.set_visible(False)
            continue

        tsne2 = TSNE(n_components=2, perplexity=min(15, len(emb_ab) - 1),
                      random_state=42, max_iter=1000)
        coords2 = tsne2.fit_transform(emb_ab)

        mask_a = lab_ab == lab_a
        mask_b = lab_ab == lab_b
        ax.scatter(coords2[mask_a, 0], coords2[mask_a, 1],
                   c="tab:blue", label=name_a, s=40, alpha=0.7)
        ax.scatter(coords2[mask_b, 0], coords2[mask_b, 1],
                   c="tab:red", label=name_b, s=40, alpha=0.7)
        ax.legend(fontsize=8)
        ax.set_title(f"{name_a} vs {name_b}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle("Most Confusable Pairs (t-SNE zoom)", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_dir / "02_confusable_pairs.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_dir / '02_confusable_pairs.png'}")

    # ===== PLOT 3: Inter-class similarity heatmap (top 30) =====
    top30_idx = np.argsort(-counts)[:30]
    top30_classes = unique[top30_idx]

    # Compute prototypes for top 30
    mask_t30 = np.isin(orig_labels, top30_classes)
    emb_t30 = torch.from_numpy(orig_embeddings[mask_t30])
    lab_t30 = torch.from_numpy(orig_labels[mask_t30])
    protos_t30 = compute_prototypes(emb_t30, lab_t30)
    sim_mat = torch.mm(protos_t30, protos_t30.t()).numpy()

    t30_labels = lab_t30.unique(sorted=True).tolist()
    t30_names = [class_names.get(int(c), str(c)) for c in t30_labels]

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(sim_mat, cmap="RdBu_r", vmin=-0.5, vmax=1.0)
    ax.set_xticks(range(len(t30_names)))
    ax.set_yticks(range(len(t30_names)))
    ax.set_xticklabels(t30_names, rotation=90, fontsize=7)
    ax.set_yticklabels(t30_names, fontsize=7)
    ax.set_title("Inter-Class Cosine Similarity (Top 30 Classes)", fontsize=13)
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(out_dir / "03_similarity_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_dir / '03_similarity_heatmap.png'}")

    # ===== PLOT 4: Per-class variance distribution =====
    all_protos = compute_prototypes(
        torch.from_numpy(orig_embeddings),
        torch.from_numpy(orig_labels),
    )
    all_unique = torch.from_numpy(orig_labels).unique(sorted=True)

    variances = []
    var_names = []
    for idx, label in enumerate(all_unique):
        mask = orig_labels == label.item()
        class_emb = torch.from_numpy(orig_embeddings[mask])
        if class_emb.size(0) > 1:
            centroid = F.normalize(class_emb.mean(dim=0, keepdim=True), p=2, dim=-1)
            sims = torch.mm(class_emb, centroid.t()).squeeze()
            var = (1.0 - sims).mean().item()
            variances.append(var)
            var_names.append(class_names.get(label.item(), str(label.item())))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.hist(variances, bins=40, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(np.median(variances), color="red", linestyle="--",
               label=f"Median: {np.median(variances):.3f}")
    ax.axvline(np.mean(variances), color="orange", linestyle="--",
               label=f"Mean: {np.mean(variances):.3f}")
    ax.set_xlabel("Intra-class variance (1 - cos_sim to centroid)")
    ax.set_ylabel("Number of classes")
    ax.set_title("Distribution of Per-Class Embedding Consistency")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "04_variance_distribution.png", dpi=150)
    plt.close()
    print(f"  Saved {out_dir / '04_variance_distribution.png'}")

    # ===== PLOT 5: Nearest-neighbor retrieval with actual images =====
    # Pick 8 random query images, show top-5 nearest neighbors
    rng = np.random.RandomState(42)
    query_indices = rng.choice(len(orig_embeddings), size=8, replace=False)

    fig, axes = plt.subplots(8, 6, figsize=(18, 24))
    fig.suptitle("Nearest-Neighbor Retrieval: Query → Top 5 Matches", fontsize=14, y=1.01)

    for row, q_idx in enumerate(query_indices):
        q_emb = orig_embeddings[q_idx]
        q_label = orig_labels[q_idx]
        q_name = class_names.get(int(q_label), str(q_label))
        q_path = orig_paths[q_idx]

        # Compute similarities to all other originals
        sims = orig_embeddings @ q_emb
        sims[q_idx] = -2  # exclude self
        top5_idx = np.argsort(-sims)[:5]

        # Query image
        ax = axes[row, 0]
        thumb = load_thumbnail(q_path, size=80)
        ax.imshow(thumb)
        ax.set_title(f"QUERY\n{q_name}", fontsize=8, fontweight="bold", color="blue")
        ax.axis("off")

        # Top 5 neighbors
        for col, nn_idx in enumerate(top5_idx):
            ax = axes[row, col + 1]
            nn_label = orig_labels[nn_idx]
            nn_name = class_names.get(int(nn_label), str(nn_label))
            nn_path = orig_paths[nn_idx]
            nn_sim = sims[nn_idx]

            thumb = load_thumbnail(nn_path, size=80)
            ax.imshow(thumb)

            is_correct = nn_label == q_label
            color = "green" if is_correct else "red"
            marker = "✓" if is_correct else "✗"
            ax.set_title(f"{marker} {nn_name}\nsim={nn_sim:.3f}",
                         fontsize=7, color=color)
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_dir / "05_nearest_neighbors.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_dir / '05_nearest_neighbors.png'}")

    # ===== PLOT 6: Accuracy vs number of shots =====
    # Simulate by computing prototype from k examples and classifying rest
    print("\nComputing accuracy vs k-shot curve...")
    k_values = [1, 2, 3, 5, 10, 15, 20]
    accuracies = []

    for k in k_values:
        correct = 0
        total = 0

        for label in all_unique.tolist():
            mask = orig_labels == label
            indices = np.where(mask)[0]
            if len(indices) < k + 1:
                continue

            rng.shuffle(indices)
            support_idx = indices[:k]
            query_idx = indices[k:]

            # Prototype from k support
            proto = orig_embeddings[support_idx].mean(axis=0)
            proto = proto / np.linalg.norm(proto)

            # Classify queries against ALL prototypes
            # (simplified: just check if query is closest to own prototype)
            for qi in query_idx:
                q_emb = orig_embeddings[qi]
                # Sim to own prototype
                sim_own = q_emb @ proto

                # Sim to other class prototypes
                is_closest = True
                for other_label in all_unique.tolist():
                    if other_label == label:
                        continue
                    other_mask = orig_labels == other_label
                    other_indices = np.where(other_mask)[0]
                    if len(other_indices) < k:
                        continue
                    rng2 = np.random.RandomState(other_label)
                    other_support = other_indices[rng2.choice(len(other_indices), size=min(k, len(other_indices)), replace=False)]
                    other_proto = orig_embeddings[other_support].mean(axis=0)
                    other_proto = other_proto / np.linalg.norm(other_proto)
                    if q_emb @ other_proto > sim_own:
                        is_closest = False
                        break

                if is_closest:
                    correct += 1
                total += 1

        acc = correct / total if total > 0 else 0
        accuracies.append(acc)
        print(f"  {k}-shot: {acc:.3f} ({correct}/{total})")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_values, accuracies, "o-", color="steelblue", linewidth=2, markersize=8)
    for k, acc in zip(k_values, accuracies):
        ax.annotate(f"{acc:.1%}", (k, acc), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9)
    ax.set_xlabel("Number of support examples (k-shot)")
    ax.set_ylabel("Classification accuracy")
    ax.set_title("Few-Shot Classification Accuracy vs Number of Support Examples")
    ax.set_ylim(0.8, 1.01)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "06_accuracy_vs_kshot.png", dpi=150)
    plt.close()
    print(f"  Saved {out_dir / '06_accuracy_vs_kshot.png'}")

    print(f"\nAll visualizations saved to {out_dir}/")


if __name__ == "__main__":
    main()

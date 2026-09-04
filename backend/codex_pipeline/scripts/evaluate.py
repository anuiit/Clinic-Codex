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
import hashlib
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from codex_pipeline.data.cached_dataset import (
    CachedFeatureDataset,
    CachedEpisodicSampler,
    collate_cached_episodes,
)
from codex_pipeline.determinism import configure_determinism
from codex_pipeline.models.projection_head import ProjectionHead, get_device
from codex_pipeline.models.prototypical import PrototypicalLoss, compute_prototypes
from codex_pipeline.data.snapshot import live_annotation_usage
from codex_pipeline.scripts.export_model import _prototype_contract
from codex_model.classifier import PREPROCESSING_VERSION




def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def update_annotated_prototypes(analysis, checkpoint, cache, manifest, base_path):
    """Replace only annotated-class centroids; preserve the active embedding space."""
    usage = live_annotation_usage(manifest)
    if cache.get("preprocessing") != PREPROCESSING_VERSION:
        raise ValueError("feature cache preprocessing is not runtime-aligned; rebuild the cache")
    base_config = json.loads((base_path.parent.parent / "config.json").read_text(encoding="utf-8"))
    for key in ("backbone", "image_size", "hidden_dim"):
        if cache.get(key) is None or cache[key] != base_config.get(key):
            raise ValueError(f"feature cache {key} differs from base runtime")
    counts = usage["split_counts"]
    if not counts["train"] or any(counts[key] for key in ("dev", "locked_test", "excluded")):
        raise ValueError("all approved annotations must be represented in train")
    rows = {row["row_id"]: row for row in manifest["rows"]}
    row_ids = cache.get("row_ids") or []
    splits = cache.get("dataset_splits") or []
    if len(row_ids) != len(rows) or set(row_ids) != set(rows) or len(splits) != len(row_ids) or len(cache["labels"]) != len(row_ids):
        raise ValueError("feature cache does not match the snapshot rows")
    for index, row_id in enumerate(row_ids):
        row = rows[row_id]
        label = int(cache["labels"][index])
        if splits[index] != row["dataset_split"] or cache["class_names"].get(label) != row["class_name"]:
            raise ValueError("feature cache labels/splits disagree with snapshot")
    base = torch.load(base_path, map_location="cpu", weights_only=True)
    base_labels, base_names = _prototype_contract(base, label="base runtime")
    base_projection = torch.load(base_path.with_name("projection.pt"), map_location="cpu", weights_only=True)
    state = checkpoint["model_state_dict"]
    if set(state) != set(base_projection) or any(not torch.equal(state[key].cpu(), base_projection[key]) for key in state):
        raise ValueError("selective prototype update requires the unchanged base projection")
    base_by_name = {base_names[label]: index for index, label in enumerate(base_labels)}
    names = cache["class_names"]
    if set(names.values()) != set(base_by_name) or base["prototypes"].shape != analysis["prototypes"].shape:
        raise ValueError("base and snapshot prototype taxonomies/shapes disagree")
    updated_classes = set(usage["training_class_names"])
    for index, label in enumerate(analysis["class_labels"].tolist()):
        name = names[label]
        if name not in updated_classes:
            analysis["prototypes"][index] = base["prototypes"][base_by_name[name]]
    return {
        "mode": "annotated_prototypes",
        "base_prototypes_sha256": sha256_file(base_path),
        "base_projection_sha256": sha256_file(base_path.with_name("projection.pt")),
        "updated_class_names": sorted(updated_classes),
        "preserved_class_count": len(base_labels) - len(updated_classes),
        "live_annotation_usage": usage,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate projection head")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/best.pt")
    parser.add_argument("--features", type=str, default="./precomputed/features.pt")
    parser.add_argument("--export-prototypes", action="store_true")
    parser.add_argument("--base-prototypes", type=Path, help="Keep base prototypes for classes without approved annotations (requires unchanged projection)")
    parser.add_argument("--snapshot-manifest", type=Path)
    parser.add_argument(
        "--prototype-dir",
        type=str,
        default=None,
        help="Explicit prototype export directory. Overrides paths.prototype_dir from checkpoint config.",
    )
    parser.add_argument("--num-episodes", type=int, default=500)
    parser.add_argument(
        "--split-strategy",
        choices=("per_class", "source_group", "persisted"),
        default=None,
        help="Split strategy used when --prototype-split is train or val.",
    )
    parser.add_argument(
        "--prototype-split",
        choices=("all", "train", "val"),
        default="all",
        help="Rows used to compute exported prototypes. Snapshot training must use train.",
    )
    parser.add_argument(
        "--skip-few-shot",
        action="store_true",
        help="Skip episodic evaluation (useful for sparse persisted dev splits).",
    )
    args = parser.parse_args()
    if args.base_prototypes and (not args.snapshot_manifest or args.prototype_split != "train" or args.split_strategy != "persisted"):
        parser.error("--base-prototypes requires --snapshot-manifest and persisted train-only prototypes")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    cfg = ckpt["config"]
    train_cfg = cfg["training"]
    model_cfg = cfg["model"]
    paths_cfg = cfg["paths"]
    hidden_dim = ckpt["hidden_dim"]

    configure_determinism(int(train_cfg["seed"]))
    device = get_device(train_cfg["device"])
    print(f"Device: {device}")
    print(f"Checkpoint: epoch {ckpt['epoch']+1}")
    print(f"  Train acc: {ckpt['train_acc']:.3f} | Val acc: {ckpt['val_acc']:.3f}")

    # --- Load cached features (no augmentation for eval) ---
    print(f"\nLoading features from {args.features}...")
    requested_split = None if args.prototype_split == "all" else args.prototype_split
    if requested_split is not None and args.split_strategy is None:
        parser.error("--prototype-split train/val requires --split-strategy")
    dataset, data_info = CachedFeatureDataset.from_file(
        args.features,
        split=requested_split,
        split_strategy=args.split_strategy or "per_class",
        val_fraction=cfg["data"]["val_fraction"],
        seed=int(train_cfg["seed"]),
    )
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

    if args.skip_few_shot:
        print("\nFew-shot evaluation skipped by request.")
    else:
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
    mean_intra = sum(all_vars) / len(all_vars) if all_vars else None
    median_intra = sorted(all_vars)[len(all_vars) // 2] if all_vars else None
    print(
        f"  Mean intra-class variance:   {mean_intra:.4f}"
        if mean_intra is not None
        else "  Mean intra-class variance:   n/a (all classes have one prototype row)"
    )
    print(
        f"  Median intra-class variance: {median_intra:.4f}"
        if median_intra is not None
        else "  Median intra-class variance: n/a (all classes have one prototype row)"
    )

    # Average inter-class similarity (excluding diagonal)
    sim_mat = analysis["similarity_matrix"]
    n = sim_mat.size(0)
    mask = ~torch.eye(n, dtype=torch.bool)
    avg_inter = sim_mat[mask].mean().item()
    print(f"  Mean inter-class similarity: {avg_inter:.4f}")
    if mean_intra not in (None, 0.0):
        print(f"  Separation ratio:            {avg_inter / mean_intra:.2f}x")
    else:
        print("  Separation ratio:            n/a")

    # --- Export prototypes ---
    if args.export_prototypes:
        update_provenance = None
        if args.base_prototypes:
            manifest = json.loads(args.snapshot_manifest.read_text(encoding="utf-8"))
            update_provenance = update_annotated_prototypes(analysis, ckpt, data_info, manifest, args.base_prototypes)
            print(f"Selective update: {update_provenance['updated_class_names']}; {update_provenance['preserved_class_count']} prototypes preserved")
        proto_dir = Path(args.prototype_dir or paths_cfg["prototype_dir"])
        proto_dir.mkdir(parents=True, exist_ok=True)

        # Also compute per-class count and variance for metadata
        class_meta = {}
        for label in analysis["class_labels"].tolist():
            name = class_names.get(label, str(label))
            count = (labels == label).sum().item()
            var = analysis["class_variances"].get(label)
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
        provenance = {
            "schema_version": "baseline-prototypes.v1",
            "checkpoint_path": str(Path(args.checkpoint).resolve()),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "features_path": str(Path(args.features).resolve()),
            "features_sha256": sha256_file(args.features),
            "prototypes_path": str(proto_path.resolve()),
            "prototypes_sha256": sha256_file(proto_path),
            "seed": int(train_cfg["seed"]),
            "class_count": len(analysis["class_labels"]),
            "prototype_split": args.prototype_split,
            "split_strategy": args.split_strategy,
            "prototype_update": update_provenance,
        }
        (proto_dir / "provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"\nPrototypes exported to {proto_path}")
        print(f"  {len(analysis['class_labels'])} class prototypes ({model_cfg['embedding_dim']}-dim)")
        print(f"  Includes projection head weights for inference")


if __name__ == "__main__":
    main()

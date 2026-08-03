#!/usr/bin/env python3
"""
Pre-compute DINOv2 features with data augmentation.

For each original image, generates N augmented versions and extracts
DINOv2 features for all of them. This creates a richer feature set
that reduces overfitting during projection head training.

Augmentation multiplier is adaptive: thin classes (few images) get
more augmented copies than large classes, balancing the dataset.

Usage:
    python -m codex_pipeline.scripts.precompute_augmented
    python -m codex_pipeline.scripts.precompute_augmented --multiplier 5 --device cpu

Output:
    ./precomputed/features_aug.pt
"""

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from codex_pipeline.data.augmentation import get_train_transform_numpy
from codex_pipeline.data.metadata import filter_classes, load_metadata


def get_device(device_str):
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def load_and_augment(image_path, transform, image_size):
    """Load image, apply augmentation, return normalized tensor."""
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)

    # Apply augmentation (returns numpy)
    augmented = transform(image=img_np)["image"]

    # Normalize for DINOv2
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_float = augmented.astype(np.float32) / 255.0
    img_float = (img_float - mean) / std

    return torch.from_numpy(img_float).permute(2, 0, 1)  # (3, H, W)


def load_clean(image_path, image_size):
    """Load image without augmentation (for the original copy)."""
    img = Image.open(image_path).convert("RGB")

    # Resize + pad to square
    w, h = img.size
    scale = image_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.BILINEAR)
    padded = Image.new("RGB", (image_size, image_size), (255, 255, 255))
    padded.paste(img, ((image_size - new_w) // 2, (image_size - new_h) // 2))

    img_np = np.array(padded, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_np = (img_np - mean) / std

    return torch.from_numpy(img_np).permute(2, 0, 1)


def compute_adaptive_multiplier(class_counts, base_multiplier, max_multiplier=15):
    """
    Compute per-class augmentation multiplier.
    Thin classes get more augmented copies to balance the dataset.
    """
    median_count = np.median(list(class_counts.values()))
    multipliers = {}
    for cls, count in class_counts.items():
        # Scale inversely with class size, capped at max_multiplier
        ratio = median_count / max(count, 1)
        m = int(base_multiplier * max(1.0, ratio))
        multipliers[cls] = min(m, max_multiplier)
    return multipliers


def main():
    parser = argparse.ArgumentParser(description="Pre-compute augmented DINOv2 features")
    parser.add_argument("--config", default="codex_pipeline/config/default.yaml")
    parser.add_argument("--multiplier", type=int, default=5,
                        help="Base augmentation multiplier per image")
    parser.add_argument("--adaptive", action="store_true", default=True,
                        help="Adapt multiplier per class (thin classes get more)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="./precomputed")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = get_device(args.device)
    image_size = cfg["data"]["image_size"]
    print(f"Device: {device}")

    # Load metadata
    metadata = load_metadata(cfg["paths"]["metadata_csv"])
    metadata = filter_classes(metadata, min_images=cfg["data"]["min_images_per_class"])
    print(f"Original: {len(metadata)} images across {metadata['class_label'].nunique()} classes")

    # Class name mapping and counts
    class_names = {}
    class_counts = defaultdict(int)
    for _, row in metadata.iterrows():
        class_names[row["class_label"]] = row["element_name"]
        class_counts[row["class_label"]] += 1

    # Compute per-class multiplier
    if args.adaptive:
        multipliers = compute_adaptive_multiplier(class_counts, args.multiplier)
        total_aug = sum(
            multipliers[row["class_label"]]
            for _, row in metadata.iterrows()
        )
        print(f"Adaptive multipliers: min={min(multipliers.values())}, "
              f"max={max(multipliers.values())}, median={int(np.median(list(multipliers.values())))}")
    else:
        multipliers = {cls: args.multiplier for cls in class_counts}
        total_aug = len(metadata) * args.multiplier

    total_images = len(metadata) + total_aug  # originals + augmented
    print(f"Will produce: {len(metadata)} originals + {total_aug} augmented = {total_images} total")

    # Load augmentation transform
    transform = get_train_transform_numpy(cfg, image_size)

    # Load DINOv2 backbone
    print("Loading DINOv2-S/14 backbone...")
    backbone = torch.hub.load(
        "facebookresearch/dinov2",
        cfg["model"]["backbone"],
        pretrained=True,
    )
    backbone = backbone.to(device)
    backbone.eval()
    hidden_dim = backbone.embed_dim

    # Process all images
    all_features = []
    all_labels = []
    all_paths = []
    all_is_augmented = []  # track which are augmented vs original

    print(f"\nExtracting features...")
    t0 = time.time()

    # Collect all images (original + augmented) with their labels
    image_batch = []
    label_batch = []
    path_batch = []
    aug_flag_batch = []

    for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Preparing"):
        img_path = row["image_path"]
        label = row["class_label"]
        n_aug = multipliers[label]

        # Original (clean) image
        try:
            img_tensor = load_clean(img_path, image_size)
            image_batch.append(img_tensor)
            label_batch.append(label)
            path_batch.append(img_path)
            aug_flag_batch.append(False)
        except Exception as e:
            print(f"  SKIP {img_path}: {e}")
            continue

        # Augmented copies
        for _ in range(n_aug):
            try:
                aug_tensor = load_and_augment(img_path, transform, image_size)
                image_batch.append(aug_tensor)
                label_batch.append(label)
                path_batch.append(img_path)
                aug_flag_batch.append(True)
            except Exception as e:
                continue

    print(f"Prepared {len(image_batch)} images ({time.time()-t0:.1f}s)")

    # Extract features in batches
    t1 = time.time()
    n_total = len(image_batch)
    for i in tqdm(range(0, n_total, args.batch_size), desc="Extracting"):
        batch = torch.stack(image_batch[i:i + args.batch_size]).to(device)
        with torch.no_grad():
            features = backbone(batch)
        all_features.append(features.cpu())

    features = torch.cat(all_features, dim=0)
    labels = torch.tensor(label_batch, dtype=torch.long)

    elapsed = time.time() - t1
    print(f"Extraction done in {elapsed:.1f}s ({n_total/elapsed:.1f} images/sec)")
    print(f"Features shape: {features.shape}")

    # Stats
    n_orig = sum(1 for f in aug_flag_batch if not f)
    n_aug = sum(1 for f in aug_flag_batch if f)
    print(f"  Originals: {n_orig} | Augmented: {n_aug}")

    # Per-class count
    label_counts = defaultdict(int)
    for l in label_batch:
        label_counts[l] += 1
    counts = list(label_counts.values())
    print(f"  Per-class features: min={min(counts)}, max={max(counts)}, "
          f"median={int(np.median(counts))}, mean={np.mean(counts):.0f}")

    # Save
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "features_aug.pt"

    torch.save({
        "features": features,
        "labels": labels,
        "image_paths": path_batch,
        "is_augmented": aug_flag_batch,
        "class_names": class_names,
        "backbone": cfg["model"]["backbone"],
        "hidden_dim": hidden_dim,
        "image_size": image_size,
        "multiplier": args.multiplier,
        "adaptive": args.adaptive,
    }, out_path)

    print(f"\nSaved to {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

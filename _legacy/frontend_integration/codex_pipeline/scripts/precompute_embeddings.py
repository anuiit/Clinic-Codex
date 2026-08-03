#!/usr/bin/env python3
"""
Pre-compute DINOv2 backbone features for all element images.

Runs the frozen DINOv2-S/14 backbone once on every image and saves
the resulting 384-dim feature vectors to disk. Training then operates
only on these cached vectors (projection head + prototypical loss),
which is orders of magnitude faster.

Usage:
    python -m codex_pipeline.scripts.precompute_embeddings
    python -m codex_pipeline.scripts.precompute_embeddings --batch-size 32 --device mps

Output:
    ./precomputed/features.pt  — dict with keys:
        "features":  (N, 384) float32 tensor
        "labels":    (N,) int64 tensor
        "image_paths": list of N strings
        "class_names": dict {class_label: element_name}
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from codex_pipeline.data.metadata import filter_classes, load_metadata


class SimpleImageDataset(Dataset):
    """Minimal dataset: load image, resize, normalize. No augmentation."""

    def __init__(self, metadata, image_size=224):
        self.metadata = metadata.reset_index(drop=True)
        self.image_size = image_size
        # ImageNet normalization (DINOv2 pretrained stats)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")

        # Resize maintaining aspect ratio, then center-crop/pad to square
        img = self._resize_and_pad(img, self.image_size)

        # To float tensor and normalize
        img = np.array(img, dtype=np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = torch.from_numpy(img).permute(2, 0, 1)  # (3, H, W)

        return img, row["class_label"]

    def _resize_and_pad(self, img, size):
        """Resize longest side to `size`, pad shorter side with white."""
        w, h = img.size
        scale = size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.BILINEAR)

        # Pad to square
        padded = Image.new("RGB", (size, size), (255, 255, 255))
        offset_x = (size - new_w) // 2
        offset_y = (size - new_h) // 2
        padded.paste(img, (offset_x, offset_y))
        return padded


def get_device(device_str):
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def main():
    parser = argparse.ArgumentParser(description="Pre-compute DINOv2 features")
    parser.add_argument("--config", default="codex_pipeline/config/default.yaml")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="./precomputed")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = get_device(args.device)
    print(f"Device: {device}")

    # Load metadata
    metadata = load_metadata(cfg["paths"]["metadata_csv"])
    metadata = filter_classes(metadata, min_images=cfg["data"]["min_images_per_class"])
    print(f"Images: {len(metadata)} across {metadata['class_label'].nunique()} classes")

    # Class name mapping
    class_names = {}
    for _, row in metadata.drop_duplicates("class_label").iterrows():
        class_names[row["class_label"]] = row["element_name"]

    # Dataset + loader
    dataset = SimpleImageDataset(metadata, image_size=cfg["data"]["image_size"])
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  # avoid multiprocessing issues on macOS
        pin_memory=False,
    )

    # Load frozen DINOv2 backbone
    print("Loading DINOv2-S/14 backbone...")
    backbone = torch.hub.load(
        "facebookresearch/dinov2",
        cfg["model"]["backbone"],
        pretrained=True,
    )
    backbone = backbone.to(device)
    backbone.eval()

    hidden_dim = backbone.embed_dim
    print(f"Backbone embed_dim: {hidden_dim}")

    # Extract features
    all_features = []
    all_labels = []
    all_paths = list(metadata["image_path"])

    print(f"Extracting features ({len(loader)} batches)...")
    t0 = time.time()

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Extracting"):
            images = images.to(device)
            features = backbone(images)  # (B, hidden_dim)
            all_features.append(features.cpu())
            all_labels.append(labels)

    features = torch.cat(all_features, dim=0)  # (N, hidden_dim)
    labels = torch.cat(all_labels, dim=0)       # (N,)

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s ({len(metadata)/elapsed:.1f} images/sec)")
    print(f"Features shape: {features.shape}")

    # Save
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "features.pt"

    torch.save({
        "features": features,
        "labels": labels,
        "image_paths": all_paths,
        "class_names": class_names,
        "backbone": cfg["model"]["backbone"],
        "hidden_dim": hidden_dim,
        "image_size": cfg["data"]["image_size"],
    }, out_path)

    print(f"Saved to {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

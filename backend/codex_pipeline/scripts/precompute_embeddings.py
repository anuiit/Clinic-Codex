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
        "source_groups": list of N strings
        "class_names": dict {class_label: element_name}
        "backbone": str
        "hidden_dim": int
        "image_size": int
    ./precomputed/features.pt.prov.json  — provenance for the cached features
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from codex_pipeline.data.class_order import class_order_sha256, load_runtime_class_order, validate_metadata_class_order
from codex_pipeline.data.metadata import filter_classes, load_metadata


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_optional_text(value) -> str | None:
    if value is None:
        return None
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text


def build_source_groups(metadata) -> list[str]:
    source_groups: list[str] = []
    for row in metadata.itertuples(index=False):
        codex = _normalise_optional_text(getattr(row, "codex", None))
        folio = _normalise_optional_text(getattr(row, "folio", None))
        page = _normalise_optional_text(getattr(row, "page", None))
        if codex is not None and folio is not None and page is not None:
            source_groups.append(f"page:{codex}:{folio}:{page}")
        else:
            image_path = _normalise_optional_text(getattr(row, "image_path", None))
            if image_path is None:
                raise ValueError("metadata row is missing both page coordinates and image_path")
            source_groups.append(f"image:{image_path}")
    return source_groups


def load_backbone(backbone_name: str, device: torch.device, backbone_manifest: str | Path | None = None):
    if backbone_manifest is None:
        print("Loading DINOv2-S/14 backbone from torch.hub...")
        backbone = torch.hub.load(
            "facebookresearch/dinov2",
            backbone_name,
            pretrained=True,
        )
        return backbone.to(device).eval(), {
            "mode": "torch_hub_pretrained",
            "backbone": backbone_name,
        }

    manifest_path = Path(backbone_manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"backbone manifest must be a JSON object: {manifest_path}")
    if manifest.get("backbone") != backbone_name:
        raise ValueError(
            f"backbone manifest targets {manifest.get('backbone')!r}, expected {backbone_name!r}"
        )

    repository_path = Path(manifest["repository_path"])
    weights_path = Path(manifest["weights_path"])
    if not repository_path.exists():
        raise FileNotFoundError(f"backbone repository path missing: {repository_path}")
    if not weights_path.is_file():
        raise FileNotFoundError(f"backbone weights file missing: {weights_path}")

    print("Loading DINOv2-S/14 backbone from local pin...")
    backbone = torch.hub.load(str(repository_path), backbone_name, source="local", pretrained=False)
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    backbone.load_state_dict(state_dict, strict=True)
    backbone = backbone.to(device).eval()
    return backbone, {
        "mode": "local_pin",
        "backbone": backbone_name,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "repository_path": str(repository_path),
        "repository_sha256": manifest.get("source_tree_sha256"),
        "weights_path": str(weights_path),
        "weights_sha256": sha256_file(weights_path),
    }


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
    parser.add_argument(
        "--runtime-config",
        default=None,
        help="Runtime class-order contract. Validates the metadata label order before feature export.",
    )
    parser.add_argument(
        "--metadata-csv",
        default=None,
        help="Explicit metadata CSV path. Overrides paths.metadata_csv from config.",
    )
    parser.add_argument(
        "--backbone-manifest",
        default=None,
        help="Optional local pin manifest for the DINOv2 backbone.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="./precomputed")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = get_device(args.device)
    print(f"Device: {device}")

    # Load metadata
    metadata_csv = args.metadata_csv or cfg["paths"]["metadata_csv"]
    metadata = load_metadata(metadata_csv)
    # Retain sparse classes for prototype export; the episodic sampler skips thin classes.
    runtime_config = args.runtime_config or cfg.get("paths", {}).get("runtime_config")
    runtime_class_order = None
    if runtime_config:
        runtime_class_order = load_runtime_class_order(runtime_config)
        validate_metadata_class_order(metadata, runtime_class_order)
        print(f"Runtime class-order hash: {class_order_sha256(runtime_class_order)}")
    print(f"Images: {len(metadata)} across {metadata['class_label'].nunique()} classes")

    # Class name mapping
    class_names = {}
    for _, row in metadata.drop_duplicates("class_label").iterrows():
        class_names[row["class_label"]] = row["element_name"]
    source_groups = build_source_groups(metadata)

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
    backbone, backbone_info = load_backbone(
        cfg["model"]["backbone"],
        device,
        backbone_manifest=args.backbone_manifest,
    )

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
        "source_groups": source_groups,
        "class_names": class_names,
        "backbone": cfg["model"]["backbone"],
        "hidden_dim": hidden_dim,
        "image_size": cfg["data"]["image_size"],
    }, out_path)

    provenance = {
        "schema_version": "features-cache.v1",
        "config_path": str(Path(args.config).resolve()),
        "config_sha256": sha256_file(args.config),
        "metadata_csv_path": str(Path(metadata_csv).resolve()),
        "metadata_csv_sha256": sha256_file(metadata_csv),
        "runtime_config_path": str(Path(runtime_config).resolve()) if runtime_config else None,
        "runtime_class_order_sha256": class_order_sha256(runtime_class_order) if runtime_class_order else None,
        "backbone": backbone_info,
        "class_count": len(class_names),
        "image_count": len(metadata),
        "hidden_dim": hidden_dim,
        "image_size": cfg["data"]["image_size"],
        "torch_version": str(torch.__version__),
    }
    prov_path = Path(str(out_path) + ".prov.json")
    prov_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Saved to {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    print(f"Saved provenance to {prov_path}")


if __name__ == "__main__":
    main()

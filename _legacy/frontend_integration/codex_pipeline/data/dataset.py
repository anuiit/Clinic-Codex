"""
PyTorch dataset and episodic sampler for prototypical network training.
"""

import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

from .augmentation import get_train_transform, get_val_transform, mixup, cutmix


class ElementDataset(Dataset):
    """
    Dataset for codex element images.

    Loads images from metadata CSV, applies augmentation, and supports
    Tier 3 (mixup/cutmix) for within-class augmentation.
    """

    def __init__(
        self,
        metadata: pd.DataFrame,
        transform=None,
        tier3_cfg: dict = None,
    ):
        self.metadata = metadata.reset_index(drop=True)
        self.transform = transform
        self.tier3_cfg = tier3_cfg or {}

        # Build class-to-indices mapping
        self.class_to_indices = defaultdict(list)
        for idx, row in self.metadata.iterrows():
            self.class_to_indices[row["class_label"]].append(idx)

        self.classes = sorted(self.class_to_indices.keys())
        self.num_classes = len(self.classes)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        img = self._load_image(row["image_path"])

        # Tier 3: mixup or cutmix with another image from the same class
        if self.tier3_cfg.get("enabled", False) and random.random() < 0.5:
            class_indices = self.class_to_indices[row["class_label"]]
            if len(class_indices) > 1:
                other_idx = random.choice(
                    [i for i in class_indices if i != idx]
                )
                other_row = self.metadata.iloc[other_idx]
                other_img = self._load_image(other_row["image_path"])

                # Resize both to same size before mixing
                target_h = max(img.shape[0], other_img.shape[0])
                target_w = max(img.shape[1], other_img.shape[1])
                img = self._pad_to_size(img, target_h, target_w)
                other_img = self._pad_to_size(other_img, target_h, target_w)

                if random.random() < self.tier3_cfg.get("cutmix_prob", 0.5):
                    img = cutmix(img, other_img)
                else:
                    img = mixup(
                        img, other_img,
                        alpha=self.tier3_cfg.get("mixup_alpha", 0.3),
                    )

        if self.transform:
            transformed = self.transform(image=img)
            img = transformed["image"]

        label = row["class_label"]
        return img, label

    def _load_image(self, path: str) -> np.ndarray:
        """Load image as RGB numpy array."""
        img = Image.open(path).convert("RGB")
        return np.array(img)

    def _pad_to_size(
        self, img: np.ndarray, target_h: int, target_w: int
    ) -> np.ndarray:
        """Pad image to target size with white background."""
        h, w = img.shape[:2]
        if h >= target_h and w >= target_w:
            return img
        padded = np.full((target_h, target_w, 3), 255, dtype=img.dtype)
        padded[:h, :w] = img
        return padded

    def get_class_images(self, class_label: int, k: int = None) -> list:
        """Get k random image indices for a given class."""
        indices = self.class_to_indices[class_label]
        if k is not None and k < len(indices):
            return random.sample(indices, k)
        return indices


class EpisodicSampler(Sampler):
    """
    Sampler for prototypical network episodic training.

    Each episode samples:
    - n_way classes
    - k_shot support + q_queries query images per class
    """

    def __init__(
        self,
        dataset: ElementDataset,
        n_way: int,
        k_shot: int,
        q_queries: int,
        episodes_per_epoch: int,
    ):
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_queries = q_queries
        self.episodes_per_epoch = episodes_per_epoch

        # Only use classes with enough images for support + query
        min_required = k_shot + q_queries
        self.valid_classes = [
            c for c in dataset.classes
            if len(dataset.class_to_indices[c]) >= min_required
        ]

        # Classes with fewer images can still participate with replacement
        self.thin_classes = [
            c for c in dataset.classes
            if len(dataset.class_to_indices[c]) >= k_shot
            and c not in self.valid_classes
        ]

    def __iter__(self):
        for _ in range(self.episodes_per_epoch):
            # Sample n_way classes, preferring valid but including thin classes
            if len(self.valid_classes) >= self.n_way:
                episode_classes = random.sample(self.valid_classes, self.n_way)
            else:
                # Use all valid + some thin classes
                episode_classes = list(self.valid_classes)
                remaining = self.n_way - len(episode_classes)
                if remaining > 0 and self.thin_classes:
                    extra = random.sample(
                        self.thin_classes,
                        min(remaining, len(self.thin_classes)),
                    )
                    episode_classes.extend(extra)

            indices = []
            for c in episode_classes:
                class_indices = self.dataset.class_to_indices[c]
                n_available = len(class_indices)
                n_needed = self.k_shot + self.q_queries

                if n_available >= n_needed:
                    selected = random.sample(class_indices, n_needed)
                else:
                    # Sample with replacement for thin classes
                    selected = random.choices(class_indices, k=n_needed)

                indices.extend(selected)

            yield indices

    def __len__(self):
        return self.episodes_per_epoch


def collate_episodes(batch, n_way, k_shot, q_queries):
    """
    Collate function that splits a flat batch into support and query sets.

    Args:
        batch: list of (image_tensor, label) tuples from one episode
        n_way: number of classes in the episode
        k_shot: support images per class
        q_queries: query images per class

    Returns:
        support_images: (n_way * k_shot, C, H, W)
        support_labels: (n_way * k_shot,)
        query_images: (n_way * q_queries, C, H, W)
        query_labels: (n_way * q_queries,)
    """
    images = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch])

    # Remap global labels to episode-local labels (0..n_way-1)
    unique_labels = labels.unique()
    label_map = {lab.item(): i for i, lab in enumerate(unique_labels)}
    local_labels = torch.tensor([label_map[l.item()] for l in labels])

    # Split into support and query per class
    support_imgs, support_labs = [], []
    query_imgs, query_labs = [], []

    for local_lab in range(len(unique_labels)):
        mask = local_labels == local_lab
        class_imgs = images[mask]
        class_labs = local_labels[mask]

        support_imgs.append(class_imgs[:k_shot])
        support_labs.append(class_labs[:k_shot])
        query_imgs.append(class_imgs[k_shot:k_shot + q_queries])
        query_labs.append(class_labs[k_shot:k_shot + q_queries])

    support_images = torch.cat(support_imgs)
    support_labels = torch.cat(support_labs)
    query_images = torch.cat(query_imgs)
    query_labels = torch.cat(query_labs)

    return support_images, support_labels, query_images, query_labels

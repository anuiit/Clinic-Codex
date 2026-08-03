"""
Dataset and sampler that operate on pre-computed DINOv2 feature vectors.

No image loading, no backbone forward pass — just vector lookups.
Training with this is orders of magnitude faster than image-based training.
"""

import random
from collections import defaultdict

import torch
from torch.utils.data import Dataset, Sampler


class CachedFeatureDataset(Dataset):
    """
    Dataset backed by pre-computed feature vectors.

    Loads the output of precompute_embeddings.py and serves
    (feature_vector, class_label) pairs.
    """

    def __init__(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        noise_std: float = 0.0,
        feature_mixup_prob: float = 0.0,
        feature_mixup_alpha: float = 0.3,
    ):
        """
        Args:
            features: (N, hidden_dim) float tensor
            labels: (N,) int tensor
            noise_std: Gaussian noise std to add during training (0 = off)
            feature_mixup_prob: probability of mixing with another same-class feature
            feature_mixup_alpha: Beta distribution alpha for mixup ratio
        """
        self.features = features
        self.labels = labels
        self.noise_std = noise_std
        self.feature_mixup_prob = feature_mixup_prob
        self.feature_mixup_alpha = feature_mixup_alpha

        # Build class-to-indices mapping
        self.class_to_indices = defaultdict(list)
        for idx in range(len(labels)):
            self.class_to_indices[labels[idx].item()].append(idx)

        self.classes = sorted(self.class_to_indices.keys())
        self.num_classes = len(self.classes)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feat = self.features[idx].clone()
        label = self.labels[idx].item()

        # Feature-level Gaussian noise
        if self.noise_std > 0:
            feat = feat + torch.randn_like(feat) * self.noise_std

        # Feature-level mixup with same-class sample
        if self.feature_mixup_prob > 0 and random.random() < self.feature_mixup_prob:
            class_indices = self.class_to_indices[label]
            if len(class_indices) > 1:
                other_idx = random.choice([i for i in class_indices if i != idx])
                other_feat = self.features[other_idx]
                lam = random.betavariate(self.feature_mixup_alpha, self.feature_mixup_alpha)
                feat = lam * feat + (1 - lam) * other_feat

        return feat, label

    @classmethod
    def from_file(
        cls,
        path: str,
        split: str = None,
        val_fraction: float = 0.15,
        seed: int = 42,
        noise_std: float = 0.0,
        feature_mixup_prob: float = 0.0,
    ):
        """
        Load from a .pt file saved by precompute_embeddings.

        Args:
            path: path to features.pt or features_aug.pt
            split: None (all data), "train", or "val"
            val_fraction: fraction of each class held out for val
            seed: random seed for reproducible splits
            noise_std: Gaussian noise for training (only applied to train split)
            feature_mixup_prob: mixup probability (only applied to train split)
        """
        data = torch.load(path, map_location="cpu", weights_only=False)
        features = data["features"]
        labels = data["labels"]

        if split is None:
            return cls(features, labels), data

        # Deterministic per-class split
        rng = random.Random(seed)
        train_indices = []
        val_indices = []

        unique_labels = labels.unique().tolist()
        for lab in unique_labels:
            indices = (labels == lab).nonzero(as_tuple=True)[0].tolist()
            rng.shuffle(indices)
            n_val = max(1, int(len(indices) * val_fraction))
            n_val = min(n_val, len(indices) - 1)  # keep at least 1 for train
            val_indices.extend(indices[:n_val])
            train_indices.extend(indices[n_val:])

        if split == "train":
            idx = torch.tensor(train_indices, dtype=torch.long)
            return cls(features[idx], labels[idx],
                       noise_std=noise_std,
                       feature_mixup_prob=feature_mixup_prob), data
        else:
            idx = torch.tensor(val_indices, dtype=torch.long)
            return cls(features[idx], labels[idx]), data


class CachedEpisodicSampler(Sampler):
    """
    Episodic sampler for cached features.

    Same logic as EpisodicSampler but operates on CachedFeatureDataset.
    Much faster since there's no image loading overhead.
    """

    def __init__(
        self,
        dataset: CachedFeatureDataset,
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

        min_required = k_shot + q_queries
        self.valid_classes = [
            c for c in dataset.classes
            if len(dataset.class_to_indices[c]) >= min_required
        ]
        self.thin_classes = [
            c for c in dataset.classes
            if len(dataset.class_to_indices[c]) >= k_shot
            and c not in self.valid_classes
        ]

    def __iter__(self):
        for _ in range(self.episodes_per_epoch):
            if len(self.valid_classes) >= self.n_way:
                episode_classes = random.sample(self.valid_classes, self.n_way)
            else:
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
                n_needed = self.k_shot + self.q_queries

                if len(class_indices) >= n_needed:
                    selected = random.sample(class_indices, n_needed)
                else:
                    selected = random.choices(class_indices, k=n_needed)

                indices.extend(selected)

            yield indices

    def __len__(self):
        return self.episodes_per_epoch


def collate_cached_episodes(batch, n_way, k_shot, q_queries):
    """
    Split a batch (already collated by DataLoader) into support/query sets.

    Args:
        batch: [features_tensor (N, D), labels_tensor (N,)] from DataLoader
        n_way: classes per episode
        k_shot: support per class
        q_queries: queries per class

    Returns:
        support_features: (n_way * k_shot, hidden_dim)
        support_labels:   (n_way * k_shot,)
        query_features:   (n_way * q_queries, hidden_dim)
        query_labels:     (n_way * q_queries,)
    """
    features, labels = batch[0], batch[1]

    # Remap to episode-local labels (0..n_way-1)
    unique_labels = labels.unique()
    label_map = {lab.item(): i for i, lab in enumerate(unique_labels)}
    local_labels = torch.tensor([label_map[l.item()] for l in labels])

    support_feats, support_labs = [], []
    query_feats, query_labs = [], []

    for local_lab in range(len(unique_labels)):
        mask = local_labels == local_lab
        class_feats = features[mask]
        class_labs = local_labels[mask]

        support_feats.append(class_feats[:k_shot])
        support_labs.append(class_labs[:k_shot])
        query_feats.append(class_feats[k_shot:k_shot + q_queries])
        query_labs.append(class_labs[k_shot:k_shot + q_queries])

    return (
        torch.cat(support_feats),
        torch.cat(support_labs),
        torch.cat(query_feats),
        torch.cat(query_labs),
    )

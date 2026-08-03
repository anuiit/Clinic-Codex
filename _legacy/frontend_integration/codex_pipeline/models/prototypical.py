"""
Prototypical Network training logic.

Given support and query sets within an episode, computes prototypes
(class centroids in embedding space) and classifies queries by
nearest prototype using cosine similarity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypicalLoss(nn.Module):
    """
    Prototypical network loss for episodic training.

    For each episode:
    1. Compute prototype = mean embedding of support images per class
    2. Compute cosine similarity between each query and all prototypes
    3. Cross-entropy loss on the similarity scores
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        support_embeddings: torch.Tensor,
        support_labels: torch.Tensor,
        query_embeddings: torch.Tensor,
        query_labels: torch.Tensor,
    ) -> dict:
        """
        Args:
            support_embeddings: (n_way * k_shot, embed_dim)
            support_labels: (n_way * k_shot,) with values in [0, n_way)
            query_embeddings: (n_way * q_queries, embed_dim)
            query_labels: (n_way * q_queries,) with values in [0, n_way)

        Returns:
            dict with 'loss', 'accuracy', 'prototypes'
        """
        n_way = support_labels.unique().size(0)

        # Compute prototypes: mean embedding per class
        prototypes = []
        for c in range(n_way):
            mask = support_labels == c
            class_embeddings = support_embeddings[mask]
            prototype = class_embeddings.mean(dim=0)
            prototype = F.normalize(prototype, p=2, dim=-1)
            prototypes.append(prototype)

        prototypes = torch.stack(prototypes)  # (n_way, embed_dim)

        # Cosine similarity between queries and prototypes
        # Both are L2-normalized, so dot product = cosine similarity
        logits = torch.mm(query_embeddings, prototypes.t())  # (n_queries, n_way)
        logits = logits / self.temperature

        # Cross-entropy loss
        loss = F.cross_entropy(logits, query_labels)

        # Accuracy
        preds = logits.argmax(dim=-1)
        accuracy = (preds == query_labels).float().mean()

        return {
            "loss": loss,
            "accuracy": accuracy,
            "prototypes": prototypes.detach(),
            "logits": logits.detach(),
            "predictions": preds.detach(),
        }


def compute_prototypes(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    Compute class prototypes from a set of embeddings.

    Args:
        embeddings: (N, embed_dim) L2-normalized embeddings
        labels: (N,) integer class labels

    Returns:
        prototypes: (num_classes, embed_dim) L2-normalized prototype per class
    """
    unique_labels = labels.unique(sorted=True)
    prototypes = []

    for label in unique_labels:
        mask = labels == label
        class_mean = embeddings[mask].mean(dim=0)
        class_mean = F.normalize(class_mean, p=2, dim=-1)
        prototypes.append(class_mean)

    return torch.stack(prototypes)


def classify_by_prototype(
    query_embedding: torch.Tensor,
    prototypes: torch.Tensor,
    class_names: list = None,
    top_k: int = 3,
    rejection_threshold: float = 0.35,
) -> list:
    """
    Classify a query embedding against stored prototypes.

    Args:
        query_embedding: (embed_dim,) or (N, embed_dim)
        prototypes: (num_classes, embed_dim)
        class_names: optional list of class names
        top_k: number of top matches to return
        rejection_threshold: minimum similarity to accept a match

    Returns:
        list of dicts with 'class_label', 'class_name', 'similarity', 'rejected'
    """
    if query_embedding.dim() == 1:
        query_embedding = query_embedding.unsqueeze(0)

    # Cosine similarity (both L2-normalized)
    similarities = torch.mm(query_embedding, prototypes.t())  # (N, num_classes)

    results = []
    for i in range(query_embedding.size(0)):
        sims = similarities[i]
        top_values, top_indices = sims.topk(min(top_k, len(sims)))

        matches = []
        for sim_val, class_idx in zip(top_values, top_indices):
            name = class_names[class_idx] if class_names else str(class_idx.item())
            matches.append({
                "class_label": class_idx.item(),
                "class_name": name,
                "similarity": sim_val.item(),
                "rejected": sim_val.item() < rejection_threshold,
            })

        results.append(matches)

    return results if len(results) > 1 else results[0]

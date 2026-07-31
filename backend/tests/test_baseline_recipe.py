from __future__ import annotations

import pandas as pd
import pytest
import torch

from backend.codex_pipeline.data.cached_dataset import CachedEpisodicSampler, CachedFeatureDataset
from backend.codex_pipeline.data.class_order import validate_metadata_class_order
from backend.codex_pipeline.models.projection_head import ProjectionHead


def test_runtime_class_order_accepts_integral_float_labels() -> None:
    metadata = pd.DataFrame(
        {"class_label": [0.0, 0.0, 1.0], "element_name": ["first", "first", "second"]}
    )

    validate_metadata_class_order(metadata, ["first", "second"])


def test_runtime_class_order_rejects_conflicting_names_for_one_label() -> None:
    metadata = pd.DataFrame(
        {"class_label": [0, 0, 1], "element_name": ["first", "other", "second"]}
    )

    with pytest.raises(ValueError, match="conflicting names"):
        validate_metadata_class_order(metadata, ["first", "second"])


def test_cached_sampler_never_reuses_support_as_query() -> None:
    # Class 3 is intentionally sparse and therefore prototype-only.
    features = torch.arange(7 * 4, dtype=torch.float32).reshape(7, 4)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3])
    dataset = CachedFeatureDataset(features, labels)
    sampler = CachedEpisodicSampler(dataset, n_way=3, k_shot=1, q_queries=1, episodes_per_epoch=1)

    indices = next(iter(sampler))

    assert len(indices) == 6
    assert len(set(indices)) == 6
    assert 6 not in indices


def test_projection_head_has_runtime_embedding_shape() -> None:
    model = ProjectionHead(input_dim=384, embedding_dim=128)

    embeddings = model(torch.zeros(2, 384))

    assert embeddings.shape == (2, 128)
    assert torch.allclose(embeddings.norm(dim=1), torch.ones(2))

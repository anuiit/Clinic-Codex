import torch
import pytest

from backend.codex_pipeline.data.cached_dataset import CachedFeatureDataset


def test_source_group_split_keeps_groups_together_and_preserves_train_examples_per_class(tmp_path) -> None:
    features = torch.tensor(
        [
            [0.0, 0.1],
            [1.0, 0.1],
            [2.0, 0.1],
            [3.0, 0.1],
            [4.0, 0.1],
            [5.0, 0.1],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
    source_groups = ["g1", "g1", "g2", "g2", "g3", "g4"]

    payload = {
        "features": features,
        "labels": labels,
        "source_groups": source_groups,
        "hidden_dim": 2,
        "class_names": {0: "zero", 1: "one"},
    }
    path = tmp_path / "features.pt"
    torch.save(payload, path)

    train_dataset, _ = CachedFeatureDataset.from_file(
        str(path),
        split="train",
        split_strategy="source_group",
        val_fraction=0.34,
        seed=7,
    )
    val_dataset, _ = CachedFeatureDataset.from_file(
        str(path),
        split="val",
        split_strategy="source_group",
        val_fraction=0.34,
        seed=7,
    )

    row_to_group = {tuple(features[idx].tolist()): source_groups[idx] for idx in range(len(features))}
    train_groups = {row_to_group[tuple(row.tolist())] for row in train_dataset.features}
    val_groups = {row_to_group[tuple(row.tolist())] for row in val_dataset.features}

    assert train_groups.isdisjoint(val_groups)
    assert len(train_dataset) + len(val_dataset) == len(features)
    assert all(len(train_dataset.class_to_indices[int(label)]) >= 1 for label in torch.unique(labels))


def test_source_group_split_requires_group_metadata(tmp_path) -> None:
    features = torch.tensor([[0.0, 0.1], [1.0, 0.1]], dtype=torch.float32)
    labels = torch.tensor([0, 1], dtype=torch.long)
    path = tmp_path / "features.pt"
    torch.save({"features": features, "labels": labels, "hidden_dim": 2}, path)

    with pytest.raises(ValueError, match="source_groups"):
        CachedFeatureDataset.from_file(
            str(path),
            split="train",
            split_strategy="source_group",
        )


def test_source_group_split_refuses_per_class_fallback_when_no_group_can_be_held_out(tmp_path) -> None:
    features = torch.tensor(
        [
            [0.0, 0.1],
            [1.0, 0.1],
            [2.0, 0.1],
            [3.0, 0.1],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    source_groups = ["g1", "g1", "g1", "g1"]
    path = tmp_path / "features.pt"
    torch.save({"features": features, "labels": labels, "source_groups": source_groups, "hidden_dim": 2}, path)

    with pytest.raises(ValueError, match="refusing to fall back"):
        CachedFeatureDataset.from_file(
            str(path),
            split="train",
            split_strategy="source_group",
            val_fraction=0.5,
            seed=7,
        )


def test_persisted_split_uses_train_and_dev_and_excludes_locked_test(tmp_path) -> None:
    features = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    labels = torch.tensor([0, 0, 1, 1, 0, 1], dtype=torch.long)
    path = tmp_path / "features.pt"
    torch.save(
        {
            "features": features,
            "labels": labels,
            "dataset_splits": ["train", "dev", "train", "dev", "locked_test", "locked_test"],
            "hidden_dim": 2,
        },
        path,
    )

    train_dataset, _ = CachedFeatureDataset.from_file(
        str(path), split="train", split_strategy="persisted"
    )
    val_dataset, _ = CachedFeatureDataset.from_file(
        str(path), split="val", split_strategy="persisted"
    )

    assert torch.equal(train_dataset.features, features[[0, 2]])
    assert torch.equal(val_dataset.features, features[[1, 3]])


@pytest.mark.parametrize(
    ("dataset_splits", "message"),
    [
        (None, "requires dataset_splits"),
        (["train", "locked_test"], "no dev rows"),
        (["dev", "locked_test"], "no train rows"),
        (["train", "validation"], "invalid values"),
    ],
)
def test_persisted_split_rejects_incomplete_or_invalid_metadata(
    tmp_path, dataset_splits, message
) -> None:
    path = tmp_path / "features.pt"
    payload = {
        "features": torch.tensor([[0.0, 0.1], [1.0, 0.1]]),
        "labels": torch.tensor([0, 1]),
        "hidden_dim": 2,
    }
    if dataset_splits is not None:
        payload["dataset_splits"] = dataset_splits
    torch.save(payload, path)

    with pytest.raises(ValueError, match=message):
        CachedFeatureDataset.from_file(
            str(path), split="train", split_strategy="persisted"
        )

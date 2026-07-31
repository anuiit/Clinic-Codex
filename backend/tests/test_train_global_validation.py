import torch
import pytest

from backend.codex_pipeline.scripts.train import evaluate_global_prototype_validation


def test_global_prototype_validation_maps_train_labels_correctly() -> None:
    train_embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    train_labels = torch.tensor([10, 10, 20, 20], dtype=torch.long)
    val_embeddings = torch.tensor(
        [
            [0.0, 1.0],
            [1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    val_labels = torch.tensor([20, 10], dtype=torch.long)

    loss, acc = evaluate_global_prototype_validation(
        train_embeddings,
        train_labels,
        val_embeddings,
        val_labels,
        temperature=1.0,
    )

    assert loss >= 0.0
    assert acc == 1.0


def test_global_prototype_validation_rejects_unseen_val_class() -> None:
    train_embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    train_labels = torch.tensor([10, 20], dtype=torch.long)
    val_embeddings = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    val_labels = torch.tensor([30], dtype=torch.long)

    with pytest.raises(SystemExit, match='Validation labels missing from train prototypes'):
        evaluate_global_prototype_validation(
            train_embeddings,
            train_labels,
            val_embeddings,
            val_labels,
            temperature=1.0,
        )

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/autoresearch_discriminative_cosine_head_v18.py"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "autoresearch_discriminative_cosine_head_v18_tested", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[2] / ".omc/autoresearch/elements-baseline-replacement/runs/20260803-model-decision-audit-v17/iteration-0003-amendment-log.json").exists(),
    reason="requires unshipped research artifact: 20260803-model-decision-audit-v17/iteration-0003-amendment-log.json",
)
def test_contract_dependencies_and_canonical_cache_pins_are_frozen() -> None:
    module = load_module()
    contract = module.validate_contract(
        module.DEFAULT_SPEC,
        module.DEFAULT_EVALUATOR,
        module.DEFAULT_AMENDMENT,
    )

    declared = {
        int(key): value
        for key, value in contract["spec"]["frozen_inputs"][
            "cache_sha256_by_fold"
        ].items()
    }
    assert declared == module.i5.EXPECTED_CACHE_SHA256
    assert contract["spec_sha256"] == module.EXPECTED_SPEC_SHA256
    assert contract["evaluator_sha256"] == module.EXPECTED_EVALUATOR_SHA256
    assert contract["amendment_sha256"] == module.EXPECTED_AMENDMENT_SHA256
    assert module.validate_dependencies()["v17_runner_sha256"] == (
        module.EXPECTED_V17_RUNNER_SHA256
    )


def test_replay_normalization_contract_is_identity_and_hash_pinned() -> None:
    module = load_module()

    assert module.REPLAY_NORMALIZATION_CONTRACT["summary_removed_keys"] == []
    assert module.REPLAY_NORMALIZATION_CONTRACT["pair_removed_keys"] == []
    assert (
        module.replay_normalization_sha256()
        == module.EXPECTED_REPLAY_NORMALIZATION_SHA256
    )


def test_control_prototypes_are_sorted_normalized_class_means() -> None:
    module = load_module()
    embeddings = F.normalize(
        torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.8, 0.6, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.8, 0.6],
            ]
        ),
        dim=1,
    )
    labels = torch.tensor([7, 7, 2, 2])

    prototype_labels, prototypes, rows = module.build_control_prototypes(
        embeddings, labels
    )

    assert prototype_labels.tolist() == [2, 7]
    assert torch.equal(prototypes[0], F.normalize(embeddings[2:].mean(dim=0), dim=0))
    assert torch.equal(prototypes[1], F.normalize(embeddings[:2].mean(dim=0), dim=0))
    assert [row["class_label"] for row in rows] == [2, 7]
    assert all(row["train_support"] == 2 for row in rows)


def test_class_balanced_weights_are_mean_one_with_equal_class_mass() -> None:
    module = load_module()
    labels = torch.tensor([1, 1, 2, 2, 2, 2])
    prototype_labels = torch.tensor([1, 2])

    weights = module.class_balanced_row_weights(labels, prototype_labels)

    assert weights.tolist() == pytest.approx([1.5, 1.5, 0.75, 0.75, 0.75, 0.75])
    assert float(weights.mean()) == pytest.approx(1.0)
    assert float(weights[labels == 1].sum()) == pytest.approx(3.0)
    assert float(weights[labels == 2].sum()) == pytest.approx(3.0)


def test_target_indices_follow_sorted_prototype_labels() -> None:
    module = load_module()

    assert module.target_indices(
        torch.tensor([7, 2, 9, 7]), torch.tensor([2, 7, 9])
    ).tolist() == [1, 0, 2, 1]


def test_tensor_hash_covers_shape_dtype_and_bytes() -> None:
    module = load_module()
    value = torch.arange(6, dtype=torch.float32).reshape(2, 3)

    assert module.tensor_sha256(value) == module.tensor_sha256(value.clone())
    assert module.tensor_sha256(value) != module.tensor_sha256(value.reshape(3, 2))
    assert module.tensor_sha256(value) != module.tensor_sha256(value.double())


def test_training_is_deterministic_engaged_and_loss_decreasing() -> None:
    module = load_module()
    embeddings = F.normalize(
        torch.tensor(
            [
                [1.0, 0.0, 0.2],
                [0.7, 0.6, 0.2],
                [0.2, 1.0, 0.0],
                [-0.5, 0.8, 0.3],
                [-1.0, 0.0, 0.2],
                [-0.7, -0.6, 0.2],
            ],
            dtype=torch.float32,
        ),
        dim=1,
    )
    labels = torch.tensor([2, 2, 7, 7, 9, 9])
    prototype_labels, initial, _rows = module.build_control_prototypes(
        embeddings, labels
    )

    first, first_trajectory, first_audit = module.train_cosine_head(
        embeddings,
        labels,
        prototype_labels,
        initial,
        device=torch.device("cpu"),
    )
    second, second_trajectory, second_audit = module.train_cosine_head(
        embeddings,
        labels,
        prototype_labels,
        initial,
        device=torch.device("cpu"),
    )

    assert torch.equal(first, second)
    assert first_trajectory == second_trajectory
    assert first_audit == second_audit
    assert len(first_trajectory) == 101
    assert first_trajectory[-1] < first_trajectory[0]
    assert first_audit["optimizer_step_count"] == 100
    assert first_audit["final_head_hash_differs_from_initial"] is True
    assert first_audit["finite_losses_logits_gradients_and_weights"] is True


def test_head_artifact_bytes_are_deterministic(tmp_path: Path) -> None:
    module = load_module()
    labels = torch.tensor([2, 7])
    weights = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    initial_sha = module.tensor_sha256(weights)

    first_sha = module.write_head_artifact(
        tmp_path / "first.bin",
        fold=1,
        seed=17,
        prototype_labels=labels,
        initial_sha256=initial_sha,
        final_weights=weights,
    )
    second_sha = module.write_head_artifact(
        tmp_path / "second.bin",
        fold=1,
        seed=17,
        prototype_labels=labels,
        initial_sha256=initial_sha,
        final_weights=weights.clone(),
    )

    assert first_sha == second_sha
    assert (tmp_path / "first.bin").read_bytes() == (tmp_path / "second.bin").read_bytes()


def test_full_batch_row_order_hash_is_order_sensitive() -> None:
    module = load_module()
    cache = {
        "train": {
            "row_id": ["a", "b"],
            "class_label": [2, 7],
            "component_id": ["ca", "cb"],
            "decoded_pixel_sha256": ["ha", "hb"],
        }
    }
    reversed_cache = {
        "train": {key: list(reversed(value)) for key, value in cache["train"].items()}
    }

    assert module.full_batch_row_order_sha256(cache) != (
        module.full_batch_row_order_sha256(reversed_cache)
    )


@pytest.mark.parametrize(
    ("count", "three", "binary"),
    [
        (8, "n_y_lte_8", "n_y_lte_8"),
        (9, "n_y_9_to_31", "n_y_gt_8"),
        (31, "n_y_9_to_31", "n_y_gt_8"),
        (32, "n_y_gte_32", "n_y_gt_8"),
    ],
)
def test_support_bins_remain_frozen(count: int, three: str, binary: str) -> None:
    module = load_module()

    assert module.v17.support_bin_three(count) == three
    assert module.v17.support_bin_binary(count) == binary

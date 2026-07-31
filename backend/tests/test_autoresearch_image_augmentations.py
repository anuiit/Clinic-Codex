from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import autoresearch_image_augmentations as subject


def asymmetric_symbol() -> np.ndarray:
    image = np.full((80, 140, 3), 255, dtype=np.uint8)
    image[18:66, 22:35] = 0
    image[52:66, 22:105] = 0
    return image


@pytest.mark.parametrize("profile", subject.PROFILE_NAMES)
def test_profiles_are_deterministic_rgb_squares(profile: str) -> None:
    first = subject.augment_image(asymmetric_symbol(), profile, seed=1234)
    second = subject.augment_image(asymmetric_symbol(), profile, seed=1234)

    assert first.shape == (224, 224, 3)
    assert first.dtype == np.uint8
    assert np.array_equal(first, second)


def test_perspective_uses_white_background() -> None:
    image = np.full((64, 64, 3), 255, dtype=np.uint8)
    image[20:44, 20:44] = 0

    transformed = subject._perspective_white(
        image, np.random.default_rng(7), (0.12, 0.12)
    )

    corners = transformed[[0, 0, -1, -1], [0, -1, 0, -1]]
    assert np.all(corners >= 250)
    assert np.any(transformed < 20)


def test_bbox_pad_crop_zooms_foreground_without_clipping() -> None:
    image = np.full((200, 300, 3), 255, dtype=np.uint8)
    image[90:110, 140:160] = 0

    transformed = subject.augment_image(image, "bbox_pad_crop", seed=9)
    foreground = np.max(np.abs(transformed.astype(np.int16) - 255), axis=2) > 18

    assert foreground.sum() > 2_000
    assert not foreground[0].any()
    assert not foreground[-1].any()
    assert not foreground[:, 0].any()
    assert not foreground[:, -1].any()


def test_dino_tensor_matches_imagenet_normalization() -> None:
    white = np.full((224, 224, 3), 255, dtype=np.uint8)

    tensor = subject.dino_tensor(white)

    expected = torch.tensor(
        [
            (1 - 0.485) / 0.229,
            (1 - 0.456) / 0.224,
            (1 - 0.406) / 0.225,
        ]
    )
    assert tensor.shape == (3, 224, 224)
    assert torch.allclose(tensor[:, 0, 0], expected)


def test_parse_profiles_accepts_commas_and_rejects_unknown() -> None:
    assert subject.parse_profiles(["geom_mild,bbox_pad_crop", "geom_mild"]) == [
        "geom_mild",
        "bbox_pad_crop",
    ]
    with pytest.raises(ValueError, match="unknown augmentation profiles"):
        subject.parse_profiles(["made-up"])


def test_smoke_indices_limits_each_class_deterministically() -> None:
    labels = torch.tensor([0, 0, 0, 1, 1, 2])
    indices = [0, 1, 2, 3, 4, 5]

    assert subject.smoke_indices(indices, labels, 2) == [0, 1, 3, 4, 5]
    assert subject.smoke_indices(indices, labels, 0) == indices


def test_split_contract_rejects_index_or_source_overlap() -> None:
    valid = {
        "external_train": [0, 1],
        "dev": [2],
        "test": [3],
        "audit": {"overlap_total": 0, "v4_test_preserved_exactly": True},
    }
    subject.validate_split_contract(valid)

    with pytest.raises(ValueError, match="indices overlap"):
        subject.validate_split_contract({**valid, "dev": [1, 2]})
    with pytest.raises(ValueError, match="source-group overlap"):
        subject.validate_split_contract(
            {**valid, "audit": {**valid["audit"], "overlap_total": 1}}
        )


def test_development_rank_and_historical_gate_use_top1_then_macro() -> None:
    historical = {"top1": 0.40, "macro_top1": 0.30}
    better = {"top1": 0.41, "macro_top1": 0.30}
    macro_regression = {"top1": 0.42, "macro_top1": 0.29}

    assert subject.development_rank(better) == (0.41, 0.30)
    assert subject.passes_historical_gate(better, historical)
    assert not subject.passes_historical_gate(macro_regression, historical)


def test_evaluate_reports_top1_macro_and_top3() -> None:
    prototypes = torch.eye(3)
    embeddings = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.8, 0.2, 0.0]]
    )
    truth = torch.tensor([0, 1, 2])

    metrics, predictions, top3, scores = subject.evaluate(
        prototypes, embeddings, truth
    )

    assert metrics["top1"] == pytest.approx(2 / 3)
    assert metrics["macro_top1"] == pytest.approx(2 / 3)
    assert metrics["top3"] == 1.0
    assert predictions.tolist() == [0, 1, 0]
    assert top3.shape == scores.shape == (3, 3)


class IdentityProjection(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value


def test_sealed_features_can_only_be_read_once() -> None:
    sealed = subject.SealedFeatures(
        external={
            "features": torch.eye(3),
            "labels": torch.tensor([0, 1, 2]),
        },
        indices=[1, 2],
        projection=IdentityProjection(),
        device=torch.device("cpu"),
    )

    features, truth = sealed.read()

    assert features.tolist() == [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    assert truth.tolist() == [1, 2]
    assert sealed.read_count == 1
    with pytest.raises(RuntimeError, match="exactly once"):
        sealed.read()


def test_prediction_rows_keeps_paths_truth_and_three_predictions() -> None:
    external = {
        "image_paths": ["/data/example.png"],
        "class_names": {0: "alpha", 1: "beta", 2: "gamma"},
    }
    predictions = torch.tensor([1])
    top3 = torch.tensor([[1, 0, 2]])
    scores = torch.tensor([[0.9, 0.2, -0.1]])

    rows = subject.prediction_rows(
        split_name="development",
        indices=[0],
        external=external,
        truth=torch.tensor([0]),
        results={"historical": (predictions, top3, scores)},
    )

    assert rows[0]["path"] == "/data/example.png"
    assert rows[0]["truth_name"] == "alpha"
    assert rows[0]["models"]["historical"]["prediction_name"] == "beta"
    assert len(rows[0]["models"]["historical"]["top3"]) == 3

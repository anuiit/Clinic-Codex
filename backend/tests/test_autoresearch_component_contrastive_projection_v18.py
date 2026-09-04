from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/autoresearch_component_contrastive_projection_v18.py"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "autoresearch_component_contrastive_projection_v18_tested", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_contract_dependencies_manifest_and_cache_pins_are_frozen() -> None:
    module = load_module()
    contract = module.validate_contract(
        module.DEFAULT_SPEC,
        module.DEFAULT_EVALUATOR,
        module.DEFAULT_MANIFEST,
        module.DEFAULT_V18_1_REPLAY_AUDIT,
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
    assert contract["manifest_sha256"] == module.EXPECTED_MANIFEST_SHA256
    assert contract["v18_1_replay_audit_sha256"] == (
        module.EXPECTED_V18_1_REPLAY_AUDIT_SHA256
    )
    assert module.validate_dependencies()["v18_1_runner_sha256"] == (
        module.EXPECTED_V18_1_RUNNER_SHA256
    )


def test_replay_normalization_contract_is_identity_and_hash_pinned() -> None:
    module = load_module()

    assert module.REPLAY_NORMALIZATION_CONTRACT["summary_removed_keys"] == []
    assert module.REPLAY_NORMALIZATION_CONTRACT["pair_removed_keys"] == []
    assert (
        module.replay_normalization_sha256()
        == module.EXPECTED_REPLAY_NORMALIZATION_SHA256
    )


def test_eligibility_requires_cross_component_and_distinct_pixels() -> None:
    module = load_module()
    train = {
        "row_id": ["a", "b", "c", "d", "e"],
        "class_label": [1, 1, 1, 2, 2],
        "component_id": ["ca", "cb", "cb", "ca", "ca"],
        "decoded_pixel_sha256": ["pa", "pb", "pa", "pd", "pe"],
    }

    eligible, positive_mask, audit = module.build_eligibility(train)

    assert eligible == [0, 1]
    assert positive_mask.tolist() == [[False, True], [True, False]]
    assert audit["eligible_anchor_count"] == 2
    assert audit["eligible_class_count"] == 1
    assert audit["eligible_class_labels"] == [1]
    assert audit["eligible_row_ids"] == ["a", "b"]
    assert audit["positive_ordered_pair_count"] == 2


def test_eligibility_has_no_within_component_fallback() -> None:
    module = load_module()
    train = {
        "row_id": ["a", "b"],
        "class_label": [1, 1],
        "component_id": ["same", "same"],
        "decoded_pixel_sha256": ["pa", "pb"],
    }

    eligible, positive_mask, audit = module.build_eligibility(train)

    assert eligible == []
    assert positive_mask.shape == (0, 0)
    assert audit["eligible_anchor_count"] == 0
    assert audit["positive_ordered_pair_count"] == 0


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[2] / ".omc/autoresearch/elements-baseline-replacement/runs/20260803-self-supervised-v9/iteration-0001/caches/fold-01-views08.pt").exists(),
    reason="requires unshipped research artifact: caches/fold-01-views08.pt",
)
def test_real_cache_eligibility_matches_every_frozen_fold_manifest() -> None:
    module = load_module()
    manifest = module.read_json(module.DEFAULT_MANIFEST)

    for fold in module.EXPECTED_FOLDS:
        cache_path = module.v9.cache_path(
            module.DEFAULT_CACHE_DIR, fold, module.EXPECTED_CACHE_VIEWS, None
        )
        cache, _validation = module.i5.load_validated_cache(
            cache_path, fold=fold, views=module.EXPECTED_CACHE_VIEWS
        )
        _eligible, _mask, audit = module.validate_eligibility(
            cache, fold=fold, manifest=manifest
        )
        assert audit["manifest_matches"] is True


def test_supervised_contrastive_loss_matches_uniform_reference() -> None:
    module = load_module()
    embeddings = torch.tensor(
        [[1.0, 0.0], [0.8, 0.2], [-1.0, 0.0], [-0.8, -0.2]],
        dtype=torch.float64,
    )
    positive_mask = torch.tensor(
        [
            [False, True, False, False],
            [True, False, False, False],
            [False, False, False, True],
            [False, False, True, False],
        ]
    )

    actual = module.supervised_contrastive_loss(embeddings, positive_mask)
    normalized = F.normalize(embeddings, dim=1)
    logits = normalized @ normalized.T / module.TEMPERATURE
    per_anchor = []
    for anchor in range(len(embeddings)):
        denominator = torch.logsumexp(
            torch.cat((logits[anchor, :anchor], logits[anchor, anchor + 1 :])),
            dim=0,
        )
        positives = torch.where(positive_mask[anchor])[0]
        per_anchor.append(-(logits[anchor, positives] - denominator).mean())
    expected = torch.stack(per_anchor).mean()

    assert torch.allclose(actual, expected, atol=1e-12, rtol=0.0)


def test_supervised_contrastive_loss_rewards_clustered_positives() -> None:
    module = load_module()
    positive_mask = torch.tensor(
        [
            [False, True, False, False],
            [True, False, False, False],
            [False, False, False, True],
            [False, False, True, False],
        ]
    )
    clustered = torch.tensor(
        [[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0], [-0.9, -0.1]]
    )
    opposed = torch.tensor(
        [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]
    )

    assert module.supervised_contrastive_loss(
        clustered, positive_mask
    ) < module.supervised_contrastive_loss(opposed, positive_mask)


def test_trainable_boundary_is_exactly_final_linear_layer() -> None:
    module = load_module()
    model = module.v9.ProjectionHead(input_dim=4, embedding_dim=3)
    model.train()

    layer = module.configure_trainable_final_layer(model)

    assert layer is model.net[3]
    assert model.training is False
    assert [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ] == ["net.3.weight", "net.3.bias"]


def test_training_is_deterministic_engaged_and_uses_frozen_optimizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_module()
    torch.manual_seed(20260804)
    first_model = module.v9.ProjectionHead(input_dim=4, embedding_dim=3)
    second_model = module.v9.ProjectionHead(input_dim=4, embedding_dim=3)
    second_model.load_state_dict(first_model.state_dict())
    first_layer = module.configure_trainable_final_layer(first_model)
    second_layer = module.configure_trainable_final_layer(second_model)
    hidden = torch.randn(6, 4, generator=torch.Generator().manual_seed(91))
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    positive_mask = labels[:, None].eq(labels[None, :])
    positive_mask.fill_diagonal_(False)
    calls: list[dict[str, object]] = []
    original_adamw = torch.optim.AdamW

    def recording_adamw(params, **kwargs):
        calls.append(kwargs)
        return original_adamw(params, **kwargs)

    monkeypatch.setattr(module.torch.optim, "AdamW", recording_adamw)
    first_trajectory, first_audit = module.train_final_layer(
        first_layer, hidden, positive_mask
    )
    second_trajectory, second_audit = module.train_final_layer(
        second_layer, hidden, positive_mask
    )

    assert calls == [
        {
            "lr": 3e-4,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "weight_decay": 0.0,
            "amsgrad": False,
            "maximize": False,
            "foreach": False,
            "fused": False,
        }
    ] * 2
    assert first_trajectory == second_trajectory
    assert first_audit == second_audit
    assert len(first_trajectory) == 101
    assert first_trajectory[-1] < first_trajectory[0]
    assert first_audit["optimizer_step_count"] == 100
    assert first_audit["final_layer_hash_differs_from_initial"] is True
    assert first_audit[
        "finite_losses_logits_gradients_weights_and_embeddings"
    ] is True
    assert torch.equal(first_layer.weight, second_layer.weight)
    assert torch.equal(first_layer.bias, second_layer.bias)


def test_layer_artifact_bytes_are_deterministic(tmp_path: Path) -> None:
    module = load_module()
    torch.manual_seed(17)
    first = torch.nn.Linear(4, 3)
    second = torch.nn.Linear(4, 3)
    second.load_state_dict(first.state_dict())
    initial_sha = module.layer_state_sha256(first)

    first_sha = module.write_layer_artifact(
        tmp_path / "first.bin",
        fold=1,
        seed=17,
        initial_sha256=initial_sha,
        layer=first,
    )
    second_sha = module.write_layer_artifact(
        tmp_path / "second.bin",
        fold=1,
        seed=17,
        initial_sha256=initial_sha,
        layer=second,
    )

    assert first_sha == second_sha
    assert (tmp_path / "first.bin").read_bytes() == (tmp_path / "second.bin").read_bytes()

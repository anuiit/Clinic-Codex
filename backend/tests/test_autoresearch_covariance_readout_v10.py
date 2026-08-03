from __future__ import annotations

import copy
import math
import importlib.util
import subprocess
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/autoresearch_covariance_readout_v10.py"
SPEC = ROOT / ".omc/autoresearch/elements-baseline-replacement/runs/20260803-self-supervised-v10/specs/iteration-0007.json"
EVALUATOR = ROOT / ".omc/autoresearch/elements-baseline-replacement/runs/20260803-self-supervised-v10/evaluator-iteration-0007.json"
spec = importlib.util.spec_from_file_location("autoresearch_covariance_readout_v10", SCRIPT)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_iteration_7_contract_is_hash_pinned() -> None:
    contract = module.validate_contract(SPEC, EVALUATOR)
    assert contract["spec_sha256"] == module.EXPECTED_SPEC_SHA256
    assert contract["evaluator_sha256"] == module.EXPECTED_EVALUATOR_SHA256


def test_off_diagonal_covariance_penalty_detects_correlation() -> None:
    diagonal = torch.tensor(
        [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]
    )
    correlated = torch.tensor(
        [[1.0, 1.0], [-1.0, -1.0], [2.0, 2.0], [-2.0, -2.0]]
    )
    assert module.off_diagonal_covariance_penalty(diagonal).item() == 0.0
    assert module.off_diagonal_covariance_penalty(correlated).item() > 0.0

def test_post_l2_covariance_penalty_is_scale_invariant() -> None:
    torch.manual_seed(13)
    raw = torch.randn(32, 8)
    reference = module.off_diagonal_covariance_penalty(module.F.normalize(raw, p=2, dim=-1))
    scaled = module.off_diagonal_covariance_penalty(module.F.normalize(7.0 * raw, p=2, dim=-1))
    torch.testing.assert_close(reference, scaled, rtol=1e-5, atol=1e-8)


def test_zero_penalty_training_reproduces_historical_readout() -> None:
    module.v9.configure_determinism(31)
    initial = module.v9.ProjectionHead(input_dim=4, embedding_dim=2)
    control = copy.deepcopy(initial)
    candidate = copy.deepcopy(initial)
    features = torch.randn(16, 4)
    labels = torch.tensor([0] * 8 + [1] * 8, dtype=torch.long)
    plan, _digest = module.v9.build_episode_plan(
        labels,
        n_way=2,
        k_shot=2,
        q_queries=2,
        epochs=2,
        episodes_per_epoch=2,
        seed=47,
    )
    kwargs = dict(
        learning_rate=1e-3,
        weight_decay=1e-4,
        temperature=0.1,
        warmup_epochs=1,
        rng_seed=53,
        device=torch.device("cpu"),
    )
    historical = module.v9.train_supervised(control, features, plan, **kwargs)
    regularized = module.train_supervised_covariance(
        candidate,
        features,
        plan,
        covariance_coefficient=0.0,
        **kwargs,
    )
    assert module.v9.state_dict_sha256(control) == module.v9.state_dict_sha256(candidate)
    assert regularized["last_classification_loss"] == historical["last_loss"]


def test_positive_coefficient_is_persisted_as_an_engaged_penalty() -> None:
    module.v9.configure_determinism(61)
    model = module.v9.ProjectionHead(input_dim=4, embedding_dim=2)
    features = torch.randn(16, 4)
    labels = torch.tensor([0] * 8 + [1] * 8, dtype=torch.long)
    plan, _digest = module.v9.build_episode_plan(
        labels,
        n_way=2,
        k_shot=2,
        q_queries=2,
        epochs=1,
        episodes_per_epoch=2,
        seed=67,
    )
    result = module.train_supervised_covariance(
        model,
        features,
        plan,
        learning_rate=1e-3,
        weight_decay=1e-4,
        temperature=0.1,
        warmup_epochs=1,
        rng_seed=71,
        device=torch.device("cpu"),
        covariance_coefficient=0.04,
    )
    assert result["mean_covariance_penalty"] > 0.0
    assert math.isclose(
        result["mean_weighted_covariance_penalty"],
        0.04 * result["mean_covariance_penalty"],
        rel_tol=1e-6,
    )
    assert result["minimum_post_l2_query_batch_mean_std"] >= 0.01
    assert result["penalty_tensor"] == "query_post_l2_shared_with_classification"
    assert math.isfinite(result["mean_weighted_covariance_penalty_over_classification_loss"])
    assert result["mean_weighted_covariance_penalty_over_classification_loss"] > 0.0

def test_runner_exposes_run_command() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "run" in completed.stdout

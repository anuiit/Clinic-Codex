from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/autoresearch_lora_vicreg_v18.py"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "autoresearch_lora_vicreg_v18_tested", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _Attention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qkv = torch.nn.Linear(384, 1152)
        self.proj = torch.nn.Linear(384, 384)


class _MLP(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(384, 1536)
        self.fc2 = torch.nn.Linear(1536, 384)


class _Block(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = _Attention()
        self.mlp = _MLP()


def test_frozen_contract_hashes_validate_without_candidate_work() -> None:
    module = load_module()

    contract = module.validate_contract(
        module.DEFAULT_SPEC,
        module.DEFAULT_EVALUATOR,
        module.DEFAULT_CACHE_SCHEMA,
        module.DEFAULT_FEASIBILITY_AUDIT,
        module.DEFAULT_PREDECESSOR_AUDIT,
        module.DEFAULT_SOURCE_INVENTORY,
    )

    assert contract["spec_sha256"] == module.EXPECTED_SPEC_SHA256
    assert contract["evaluator_sha256"] == module.EXPECTED_EVALUATOR_SHA256
    assert contract["cache_schema_sha256"] == module.EXPECTED_CACHE_SCHEMA_SHA256
    assert contract["feasibility"]["candidate_training_count"] == 0
    assert contract["feasibility"]["candidate_prediction_count"] == 0


def test_paired_sources_and_all_c1_checkpoint_hashes_are_pinned() -> None:
    module = load_module()

    source = module.load_source_rows(
        module.DEFAULT_V9_SUMMARY,
        module.DEFAULT_V9_PREDICTIONS,
        module.DEFAULT_V18_2_PREDICTIONS,
    )

    assert len(source["v9_lookup"]) == module.EXPECTED_PAIRED_ROWS
    for key, diagnostic in source["persisted"]["diagnostics"].items():
        expected = module.EXPECTED_C1_CHECKPOINT_FILE_SHA256[key]
        assert diagnostic["checkpoints"]["candidate"]["sha256"] == expected
        assert module.v9.sha256_file(
            module.expected_c1_checkpoint_path(*key)
        ) == expected


def test_lora_boundary_step_zero_count_and_rng_isolation() -> None:
    module = load_module()
    torch.manual_seed(41)
    block = _Block()
    block.requires_grad_(False)
    probes = {
        name: torch.randn(3, child.in_features)
        for name, child in block.named_modules()
        if isinstance(child, torch.nn.Linear)
    }
    references = {
        name: block.get_submodule(name)(values).detach().clone()
        for name, values in probes.items()
    }
    rng_before = torch.get_rng_state().clone()

    adapters = module.inject_lora(block, fold=2, seed=42)

    assert torch.equal(rng_before, torch.get_rng_state())
    manifest = module.assert_trainable_boundary(
        block, torch.nn.LayerNorm(384).requires_grad_(False), adapters
    )
    assert manifest == module.expected_optimizer_manifest()
    assert manifest["count"] == module.LORA_PARAMETER_COUNT
    for name, values in probes.items():
        assert torch.equal(block.get_submodule(name)(values), references[name])


def test_frozen_base_hash_survives_wrapping_and_lora_update() -> None:
    module = load_module()
    torch.manual_seed(7)
    block = _Block().requires_grad_(False)
    norm = torch.nn.LayerNorm(384).requires_grad_(False)
    before = module.parameter_mapping_sha256(
        module.frozen_backbone_parameters(block, norm)
    )
    adapters = module.inject_lora(block, fold=1, seed=17)
    wrapped = module.parameter_mapping_sha256(
        module.frozen_backbone_parameters(block, norm)
    )
    with torch.no_grad():
        adapters["attn.qkv"].lora_b.add_(0.25)
    after = module.parameter_mapping_sha256(
        module.frozen_backbone_parameters(block, norm)
    )

    assert before == wrapped == after
    assert module.parameter_mapping_sha256(
        module.lora_named_parameters(adapters)
    ) != module.parameter_mapping_sha256(
        [(name, torch.zeros_like(value)) for name, value in module.lora_named_parameters(adapters)]
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA contract probe")
def test_selected_token_resume_preserves_gradients_and_batch_padding() -> None:
    module = load_module()
    device = torch.device("cuda")
    block = torch.nn.Linear(384, 384, bias=False).to(device)
    norm = torch.nn.Identity().to(device)
    tokens = np.random.default_rng(8).normal(
        size=(2, 1, module.TOKEN_COUNT, module.EMBED_DIM)
    ).astype(np.float32)

    values = module.resume_selected(
        tokens,
        [0, 1],
        [0, 0],
        [3, 3],
        block=block,
        norm=norm,
        device=device,
    )
    values.square().mean().backward()

    assert values.shape == (2, module.EMBED_DIM)
    assert block.weight.grad is not None
    assert torch.isfinite(block.weight.grad).all()


def test_tensor_artifact_is_byte_deterministic(tmp_path: Path) -> None:
    module = load_module()
    tensors = {
        "b": torch.tensor([[1.0, 2.0]], dtype=torch.float32),
        "a": torch.tensor([3, 4], dtype=torch.int64),
    }
    metadata = {"fold": 1, "seed": 17}

    first = module.write_tensor_artifact(
        tmp_path / "first.bin", metadata=metadata, tensors=tensors
    )
    second = module.write_tensor_artifact(
        tmp_path / "second.bin", metadata=metadata, tensors=dict(reversed(list(tensors.items())))
    )

    assert first == second
    assert (tmp_path / "first.bin").read_bytes() == (
        tmp_path / "second.bin"
    ).read_bytes()


def test_manifest_loader_rejects_pretraining_oof_exposure(tmp_path: Path) -> None:
    module = load_module()
    contract = {
        "spec_sha256": "spec",
        "evaluator_sha256": "evaluator",
        "cache_schema_sha256": "schema",
        "source_inventory_sha256": "inventory",
    }
    manifest = {
        **contract,
        "runner_sha256": module.v9.sha256_file(SCRIPT),
        "status": "sealed_before_any_active_lora_training",
        "oof_token_file_count": 1,
        "candidate_optimizer_steps": 0,
    }
    path = tmp_path / "manifest.json"
    module.v9.write_json(path, manifest)

    with pytest.raises(ValueError, match="OOF token existed"):
        module.load_train_manifest(path, contract=contract)


def test_oof_state_gate_separates_canonical_and_replay(tmp_path: Path) -> None:
    module = load_module()
    manifest = tmp_path / "token-cache-manifest.json"

    module.assert_initial_oof_state(tmp_path, manifest, reuse_oof=False)
    with pytest.raises(FileNotFoundError, match="replay requires"):
        module.assert_initial_oof_state(tmp_path, manifest, reuse_oof=True)

    for fold in module.EXPECTED_FOLDS:
        paths = module.fold_paths(tmp_path, fold)
        paths["root"].mkdir(parents=True, exist_ok=True)
        for key in ("oof_tokens", "oof_metadata", "oof_features"):
            paths[key].touch()
    manifest.touch()

    module.assert_initial_oof_state(tmp_path, manifest, reuse_oof=True)
    with pytest.raises(FileExistsError, match="physically absent"):
        module.assert_initial_oof_state(tmp_path, manifest, reuse_oof=False)

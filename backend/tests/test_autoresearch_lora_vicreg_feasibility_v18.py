from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/autoresearch_lora_vicreg_feasibility_v18.py"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "autoresearch_lora_vicreg_feasibility_v18_tested", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parser_args(module):
    return module.build_parser().parse_args([])


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[2] / ".omc/autoresearch/elements-baseline-replacement/runs/20260731-council-guided-v8/iteration-0002/provenance-manifest.jsonl").exists(),
    reason="requires unshipped research artifact: iteration-0002/provenance-manifest.jsonl",
)
def test_frozen_input_hashes_and_predecessor_audit_are_valid() -> None:
    module = load_module()

    pins = module.validate_frozen_inputs(parser_args(module))

    assert pins["v9_runner_sha256"] == module.EXPECTED_V9_RUNNER_SHA256
    assert pins["predecessor_replay_audit_sha256"] == (
        module.EXPECTED_PREDECESSOR_AUDIT_SHA256
    )


def test_lora_linear_is_exact_at_step_zero_and_changes_after_update() -> None:
    module = load_module()
    torch.manual_seed(7)
    base = torch.nn.Linear(5, 3)
    values = torch.randn(4, 5)
    reference = base(values).detach()
    adapter = module.LoRALinear(base)

    assert torch.equal(adapter(values), reference)
    assert adapter.lora_a.requires_grad is True
    assert adapter.lora_b.requires_grad is True
    assert all(parameter.requires_grad is False for parameter in base.parameters())

    loss = adapter(values).square().mean()
    loss.backward()
    with torch.no_grad():
        adapter.lora_b.add_(adapter.lora_b.grad, alpha=-0.1)

    assert not torch.equal(adapter(values), reference)


class _Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = torch.nn.Linear(384, 1152)
        self.proj = torch.nn.Linear(384, 384)


class _MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(384, 1536)
        self.fc2 = torch.nn.Linear(1536, 384)


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = _Attention()
        self.mlp = _MLP()


class _Backbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            [torch.nn.Identity() for _ in range(11)] + [_Block()]
        )


def test_last_block_injection_has_exact_targets_boundary_and_count() -> None:
    module = load_module()
    model = _Backbone()
    original = {
        name: child
        for name, child in model.blocks[-1].named_modules()
        if isinstance(child, torch.nn.Linear)
    }

    adapters = module.inject_last_block_lora(model)

    assert list(adapters) == list(module.EXPECTED_LORA_TARGETS)
    assert sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    ) == module.EXPECTED_LORA_PARAMETER_COUNT
    assert {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    } == {
        f"blocks.11.{target}.lora_{suffix}"
        for target in module.EXPECTED_LORA_TARGETS
        for suffix in ("a", "b")
    }
    for name, adapter in adapters.items():
        width = original[name].in_features
        values = torch.randn(2, width)
        assert torch.equal(adapter(values), original[name](values))


def test_source_inventory_hashes_bytes_and_decoded_pixels(tmp_path: Path) -> None:
    module = load_module()
    image = tmp_path / "glyph.png"
    Image.new("RGB", (3, 2), (12, 34, 56)).save(image)
    pixel_sha = module.provenance_v8.decoded_rgb_sha256(image)
    rows = [
        {
            "row_id": "row-a",
            "image_path": str(image),
            "class_label": 1,
            "fold": 1,
            "component_id": "component-a",
            "decoded_pixel_sha256": pixel_sha,
        },
        {
            "row_id": "row-b",
            "image_path": str(image),
            "class_label": 1,
            "fold": 1,
            "component_id": "component-a",
            "decoded_pixel_sha256": pixel_sha,
        },
    ]

    report = module.audit_source_inventory(
        rows, output=tmp_path / "inventory.jsonl", workers=2
    )

    assert report["retained_row_count"] == 2
    assert report["unique_image_path_count"] == 1
    assert report["all_decoded_pixel_hashes_match"] is True
    assert len((tmp_path / "inventory.jsonl").read_text().splitlines()) == 2


def test_source_inventory_rejects_pixel_drift(tmp_path: Path) -> None:
    module = load_module()
    image = tmp_path / "glyph.png"
    Image.new("RGB", (2, 2), (1, 2, 3)).save(image)
    rows = [
        {
            "row_id": "row-a",
            "image_path": str(image),
            "class_label": 1,
            "fold": 1,
            "component_id": "component-a",
            "decoded_pixel_sha256": "wrong",
        }
    ]

    with pytest.raises(ValueError, match="decoded pixel hash mismatch"):
        module.audit_source_inventory(
            rows, output=tmp_path / "inventory.jsonl", workers=1
        )


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[2] / ".omc/autoresearch/elements-baseline-replacement/runs/20260731-council-guided-v8/iteration-0002/provenance-manifest.jsonl").exists(),
    reason="requires unshipped research artifact: iteration-0002/provenance-manifest.jsonl",
)
def test_canonical_cache_metadata_matches_corpus_and_token_estimate() -> None:
    module = load_module()
    rows, report = module.v9.load_corpus(
        module.DEFAULT_CORPUS_MANIFEST,
        module.DEFAULT_COLLECTION_AUDIT,
        strict_counts=True,
    )

    validations = module.validate_cache_metadata(rows)

    assert report["retained_rows"] == 9990
    assert {
        fold: (
            item["train_rows"],
            item["oof_rows"],
            item["estimated_float32_token_cache_bytes"],
        )
        for fold, item in validations.items()
    } == {
        "1": (8091, 48, 28764393984),
        "2": (8139, 22, 28924663296),
        "3": (8021, 32, 28509384192),
        "4": (8041, 438, 28740708864),
        "5": (7668, 113, 27287232000),
    }
    assert all(
        item["row_component_pixel_overlap"] == 0
        for item in validations.values()
    )


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[2] / ".omc/autoresearch/elements-baseline-replacement/runs/20260803-self-supervised-v9/inputs/dinov2-vits14-local-manifest.json").exists(),
    reason="requires unshipped research artifact: inputs/dinov2-vits14-local-manifest.json",
)
def test_backbone_pin_critical_files_and_weights_are_unchanged() -> None:
    module = load_module()

    pin = module.validate_backbone_pin(module.DEFAULT_BACKBONE_MANIFEST)

    assert pin["weights_sha256_verified"] == module.EXPECTED_WEIGHTS_SHA256
    assert pin["critical_source_hashes_verified"] == {
        "hubconf_sha256": module.EXPECTED_HUBCONF_SHA256,
        "vision_transformer_sha256": (
            module.EXPECTED_VISION_TRANSFORMER_SHA256
        ),
    }


def test_stable_source_tree_ignores_generated_python_cache(tmp_path: Path) -> None:
    module = load_module()
    (tmp_path / "module.py").write_text("value = 1\n")
    before = module.stable_source_tree_sha256(tmp_path)
    generated = tmp_path / "__pycache__"
    generated.mkdir()
    (generated / "module.pyc").write_bytes(b"mutable")

    assert module.stable_source_tree_sha256(tmp_path) == before

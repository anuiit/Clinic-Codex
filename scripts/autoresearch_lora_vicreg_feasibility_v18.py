#!/usr/bin/env python3
"""Audit feasibility for one final DINOv2 last-block LoRA-VICReg candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import autoresearch_component_contrastive_projection_v18 as v18_2  # noqa: E402
import autoresearch_provenance_v8 as provenance_v8  # noqa: E402
import autoresearch_self_supervised_v9 as v9  # noqa: E402
import autoresearch_support_aware_readout_v10 as i5  # noqa: E402


V18_RUN = v18_2.V18_RUN
DEFAULT_CORPUS_MANIFEST = v9.DEFAULT_MANIFEST
DEFAULT_COLLECTION_AUDIT = v9.DEFAULT_AUDIT
DEFAULT_BACKBONE_MANIFEST = v9.DEFAULT_BACKBONE_MANIFEST
DEFAULT_PREDECESSOR_AUDIT = V18_RUN / "iteration-0002-replay-audit.json"
DEFAULT_SOURCE_INVENTORY = V18_RUN / "specs/iteration-0003-source-inventory.jsonl"
DEFAULT_OUTPUT = V18_RUN / "iteration-0003-feasibility.json"

EXPECTED_CORPUS_MANIFEST_SHA256 = (
    "e918a190195d15ccfa9ea1fd4902607b84eea0c7d6c267d146a757f1b6547385"
)
EXPECTED_COLLECTION_AUDIT_SHA256 = (
    "e931611a9174fbcb2799c52f6bbb6cc315229d9e198260756ce51f804d032b99"
)
EXPECTED_BACKBONE_MANIFEST_SHA256 = (
    "bfc72fa2d0fbffc786236d591501ff9dfa23f5c316d619a47e891160316f9ab5"
)
EXPECTED_PREDECESSOR_AUDIT_SHA256 = (
    "795d8394671635db210deabb29f7f3ec85dd9f06290eb335493bd51495d8d10b"
)
EXPECTED_V9_RUNNER_SHA256 = (
    "e03e1e29340a1a165ab2f82001236f5aaae4450afcee862dff8cb27ef26c48a9"
)
EXPECTED_PROVENANCE_RUNNER_SHA256 = (
    "9641296db9893f38c12ea3c72ad804ea79d15b5fc2cbe6d275da6946aae26dc5"
)
EXPECTED_BACKBONE = "dinov2_vits14"
EXPECTED_SOURCE_TREE_DECLARATION = (
    "b64f9117250826940b54ed25b65e2869fcdc9222cbaf9c1d4885eeab5f950a3b"
)
EXPECTED_WEIGHTS_SHA256 = (
    "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"
)
EXPECTED_HUBCONF_SHA256 = (
    "c1f5090e78ff940b72c076d2bf9c0310d1707c946b3d10e2d6f2b0bdf56a6f64"
)
EXPECTED_VISION_TRANSFORMER_SHA256 = (
    "7799a260f2d7d0fe197331d08502fb8c542f9b7424723650f6a39b64fa2639ea"
)
EXPECTED_LORA_TARGETS = (
    "attn.qkv",
    "attn.proj",
    "mlp.fc1",
    "mlp.fc2",
)
EXPECTED_FOLDS = (1, 2, 3, 4, 5)
EXPECTED_VIEWS = 8
TOKEN_COUNT = 257
EMBED_DIM = 384
LORA_RANK = 8
LORA_ALPHA = 8
VICREG_BATCH_SIZE = 256
EXPECTED_LORA_PARAMETER_COUNT = 49_152


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_tensor_map(items: Iterable[tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(items):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(v9.canonical_json(list(value.shape)).encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def stable_source_tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    candidates = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and ".git" not in path.parts
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    )
    for path in candidates:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def validate_frozen_inputs(args: argparse.Namespace) -> dict[str, Any]:
    actual = {
        "corpus_manifest_sha256": v9.sha256_file(args.corpus_manifest),
        "collection_audit_sha256": v9.sha256_file(args.collection_audit),
        "backbone_manifest_sha256": v9.sha256_file(args.backbone_manifest),
        "predecessor_replay_audit_sha256": v9.sha256_file(
            args.predecessor_audit
        ),
        "v9_runner_sha256": v9.sha256_file(Path(v9.__file__).resolve()),
        "provenance_runner_sha256": v9.sha256_file(
            Path(provenance_v8.__file__).resolve()
        ),
    }
    expected = {
        "corpus_manifest_sha256": EXPECTED_CORPUS_MANIFEST_SHA256,
        "collection_audit_sha256": EXPECTED_COLLECTION_AUDIT_SHA256,
        "backbone_manifest_sha256": EXPECTED_BACKBONE_MANIFEST_SHA256,
        "predecessor_replay_audit_sha256": EXPECTED_PREDECESSOR_AUDIT_SHA256,
        "v9_runner_sha256": EXPECTED_V9_RUNNER_SHA256,
        "provenance_runner_sha256": EXPECTED_PROVENANCE_RUNNER_SHA256,
    }
    if actual != expected:
        raise ValueError(f"v18.3 feasibility input pin mismatch: {actual}")
    predecessor = read_json(args.predecessor_audit)
    if (
        predecessor.get("pass") is not True
        or predecessor.get("decision") != "not_supported"
        or predecessor.get("final_test_read") is not False
        or predecessor.get("runtime_promotion") is not False
    ):
        raise ValueError("v18.2 predecessor audit does not authorize feasibility")
    return actual


def validate_backbone_pin(path: Path) -> dict[str, Any]:
    manifest = read_json(path)
    if (
        manifest.get("schema_version") != "dinov2-local-pin.v1"
        or manifest.get("backbone") != EXPECTED_BACKBONE
        or manifest.get("source_tree_sha256") != EXPECTED_SOURCE_TREE_DECLARATION
        or manifest.get("weights_sha256") != EXPECTED_WEIGHTS_SHA256
        or manifest.get("hubconf_sha256") != EXPECTED_HUBCONF_SHA256
        or manifest.get("vision_transformer_sha256")
        != EXPECTED_VISION_TRANSFORMER_SHA256
    ):
        raise ValueError("DINOv2-S/14 manifest declaration changed")
    repository = Path(manifest["repository_path"])
    weights = Path(manifest["weights_path"])
    critical = {
        "hubconf_sha256": repository / "hubconf.py",
        "vision_transformer_sha256": (
            repository / "dinov2/models/vision_transformer.py"
        ),
    }
    actual_critical = {
        key: v9.sha256_file(value) for key, value in critical.items()
    }
    if actual_critical != {
        "hubconf_sha256": EXPECTED_HUBCONF_SHA256,
        "vision_transformer_sha256": EXPECTED_VISION_TRANSFORMER_SHA256,
    }:
        raise ValueError("critical DINOv2 source file changed")
    actual_weights = v9.sha256_file(weights)
    if actual_weights != EXPECTED_WEIGHTS_SHA256:
        raise ValueError("DINOv2-S/14 weights changed")
    return {
        **manifest,
        "manifest_sha256": v9.sha256_file(path),
        "weights_sha256_verified": actual_weights,
        "weights_bytes": weights.stat().st_size,
        "critical_source_hashes_verified": actual_critical,
        "stable_source_tree_sha256": stable_source_tree_sha256(repository),
    }


def audit_source_inventory(
    rows: list[dict[str, Any]],
    *,
    output: Path,
    workers: int,
) -> dict[str, Any]:
    paths = sorted({Path(str(row["image_path"])) for row in rows}, key=str)
    if any(path.is_symlink() for path in paths):
        raise ValueError("source inventory contains symbolic links")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing source images: {missing[:5]}")

    def inspect(path: Path) -> tuple[str, int, str, str]:
        return (
            str(path),
            path.stat().st_size,
            v9.sha256_file(path),
            provenance_v8.decoded_rgb_sha256(path),
        )

    with ThreadPoolExecutor(max_workers=max(1, min(workers, 32))) as executor:
        inspected = list(executor.map(inspect, paths))
    by_path = {
        path: {
            "source_bytes": size,
            "source_file_sha256": file_sha,
            "decoded_pixel_sha256": pixel_sha,
        }
        for path, size, file_sha, pixel_sha in inspected
    }
    inventory: list[dict[str, Any]] = []
    mismatches: list[str] = []
    for row in rows:
        path = str(row["image_path"])
        source = by_path[path]
        if source["decoded_pixel_sha256"] != str(row["decoded_pixel_sha256"]):
            mismatches.append(str(row["row_id"]))
        inventory.append(
            {
                "row_id": str(row["row_id"]),
                "image_path": path,
                "class_label": int(row["class_label"]),
                "fold": int(row["fold"]),
                "component_id": str(row["component_id"]),
                "decoded_pixel_sha256": str(row["decoded_pixel_sha256"]),
                "source_bytes": int(source["source_bytes"]),
                "source_file_sha256": str(source["source_file_sha256"]),
            }
        )
    if mismatches:
        raise ValueError(f"decoded pixel hash mismatch: {mismatches[:5]}")
    v9.write_jsonl(output, inventory)
    return {
        "retained_row_count": len(rows),
        "unique_image_path_count": len(paths),
        "all_paths_regular_files": True,
        "all_decoded_pixel_hashes_match": True,
        "source_inventory_sha256": v9.sha256_file(output),
        "source_inventory_rows_sha256": v9.sha256_json(inventory),
        "source_bytes_total": sum(value["source_bytes"] for value in by_path.values()),
    }


def validate_cache_metadata(rows: list[dict[str, Any]]) -> dict[str, Any]:
    validations: dict[str, Any] = {}
    fields = (
        "row_id",
        "class_label",
        "class_name",
        "component_id",
        "decoded_pixel_sha256",
    )
    for fold in EXPECTED_FOLDS:
        path = v9.cache_path(v9.DEFAULT_CACHE_DIR, fold, EXPECTED_VIEWS, None)
        cache, validation = i5.load_validated_cache(
            path, fold=fold, views=EXPECTED_VIEWS
        )
        expected_train_rows = [row for row in rows if int(row["fold"]) != fold]
        train_labels = {int(row["class_label"]) for row in expected_train_rows}
        expected_sections = {
            "train": expected_train_rows,
            # The canonical v9 cache excludes OOF classes absent from the
            # corresponding fold-train partition. Reproduce that sealed
            # rule exactly instead of treating those unevaluable rows as a
            # metadata mismatch.
            "oof": [
                row
                for row in rows
                if int(row["fold"]) == fold
                and int(row["class_label"]) in train_labels
            ],
        }
        for section, expected_rows in expected_sections.items():
            expected = {
                field: [
                    (
                        int(row[field])
                        if field == "class_label"
                        else str(row[field])
                    )
                    for row in expected_rows
                ]
                for field in fields
            }
            actual = {
                field: [
                    (
                        int(value)
                        if field == "class_label"
                        else str(value)
                    )
                    for value in cache[section][field]
                ]
                for field in fields
            }
            if actual != expected:
                raise ValueError(
                    f"canonical cache metadata mismatch: fold={fold} {section}"
                )
        train_ids = set(str(value) for value in cache["train"]["row_id"])
        oof_ids = set(str(value) for value in cache["oof"]["row_id"])
        train_components = set(
            str(value) for value in cache["train"]["component_id"]
        )
        oof_components = set(str(value) for value in cache["oof"]["component_id"])
        train_pixels = set(
            str(value) for value in cache["train"]["decoded_pixel_sha256"]
        )
        oof_pixels = set(
            str(value) for value in cache["oof"]["decoded_pixel_sha256"]
        )
        if train_ids & oof_ids or train_components & oof_components or train_pixels & oof_pixels:
            raise ValueError(f"cache leakage for fold={fold}")
        train_count = len(cache["train"]["row_id"])
        oof_count = len(cache["oof"]["row_id"])
        token_bytes = (
            (train_count * (EXPECTED_VIEWS + 1) + oof_count)
            * TOKEN_COUNT
            * EMBED_DIM
            * 4
        )
        validations[str(fold)] = {
            **validation,
            "train_rows": train_count,
            "oof_rows": oof_count,
            "metadata_matches_corpus": True,
            "row_component_pixel_overlap": 0,
            "estimated_float32_token_cache_bytes": token_bytes,
        }
    return validations


class LoRALinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int = LORA_RANK,
        alpha: int = LORA_ALPHA,
    ):
        super().__init__()
        if rank != LORA_RANK or alpha != LORA_ALPHA:
            raise ValueError("v18.3 LoRA rank/alpha changed")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.lora_a = nn.Parameter(
            torch.empty(rank, base.in_features, device=base.weight.device)
        )
        self.lora_b = nn.Parameter(
            torch.zeros(base.out_features, rank, device=base.weight.device)
        )
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.scaling = alpha / rank

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        update = (values @ self.lora_a.T) @ self.lora_b.T
        return self.base(values) + update * self.scaling


def _replace_module(parent: nn.Module, name: str, replacement: nn.Module) -> None:
    parts = name.split(".")
    target = parent
    for part in parts[:-1]:
        target = getattr(target, part)
    setattr(target, parts[-1], replacement)


def inject_last_block_lora(model: nn.Module) -> dict[str, LoRALinear]:
    model.requires_grad_(False)
    model.eval()
    block = model.blocks[-1]
    targets = [
        (name, module)
        for name, module in block.named_modules()
        if isinstance(module, nn.Linear)
    ]
    if tuple(name for name, _module in targets) != EXPECTED_LORA_TARGETS:
        raise ValueError(
            f"DINOv2 last-block Linear target drift: {[name for name, _ in targets]}"
        )
    adapters: dict[str, LoRALinear] = {}
    for name, module in targets:
        adapter = LoRALinear(module)
        _replace_module(block, name, adapter)
        adapters[name] = adapter
    return adapters


def synthetic_lora_smoke(
    backbone_manifest: Path,
    *,
    device: torch.device,
) -> dict[str, Any]:
    if device.type != "cuda":
        raise ValueError("v18.3 feasibility smoke requires CUDA")
    v9.configure_determinism(20260804)
    before_runtime = v18_2.v18.runtime_hashes(
        v18_2.DEFAULT_RUNTIME_PROJECTION,
        v18_2.DEFAULT_RUNTIME_PROTOTYPES,
        v18_2.DEFAULT_RUNTIME_CONFIG,
    )
    model, provenance = v9.load_backbone(
        EXPECTED_BACKBONE, device, backbone_manifest
    )
    model.requires_grad_(False).eval()
    block = model.blocks[-1]
    base_parameters = {
        name: parameter for name, parameter in model.named_parameters()
    }
    base_hash_before = sha256_tensor_map(base_parameters.items())
    sample_tokens = torch.randn(2, TOKEN_COUNT, EMBED_DIM, device=device)
    with torch.no_grad():
        step_zero_reference = model.norm(block(sample_tokens))[:, 0].clone()
    adapters = inject_last_block_lora(model)
    with torch.no_grad():
        step_zero_candidate = model.norm(model.blocks[-1](sample_tokens))[:, 0]
    step_zero_equal = torch.equal(step_zero_reference, step_zero_candidate)
    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    lora_parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    if lora_parameter_count != EXPECTED_LORA_PARAMETER_COUNT:
        raise ValueError(f"LoRA parameter-count drift: {lora_parameter_count}")
    expected_trainable_suffixes = {
        f"blocks.11.{target}.lora_{suffix}"
        for target in EXPECTED_LORA_TARGETS
        for suffix in ("a", "b")
    }
    if set(trainable_names) != expected_trainable_suffixes:
        raise ValueError(f"LoRA trainable-boundary drift: {trainable_names}")

    projection = v9.ProjectionHead(EMBED_DIM, 128).to(device).train()
    expander = v9.VICRegExpander().to(device).train()
    candidate_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ] + list(projection.parameters()) + list(expander.parameters())
    optimizer = torch.optim.AdamW(
        candidate_parameters,
        lr=3e-4,
        weight_decay=1e-4,
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    tokens = torch.randn(
        VICREG_BATCH_SIZE * 2,
        TOKEN_COUNT,
        EMBED_DIM,
        device=device,
    )
    features = model.norm(model.blocks[-1](tokens))[:, 0]
    quantized = features.to(torch.float16).to(torch.float32)
    projected = expander(projection.net(quantized))
    terms = v9.vicreg_loss(
        projected[:VICREG_BATCH_SIZE],
        projected[VICREG_BATCH_SIZE:],
    )
    optimizer.zero_grad(set_to_none=True)
    terms.total.backward()
    gradient_finite = all(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in candidate_parameters
    )
    base_gradients_absent = all(
        parameter.grad is None for parameter in base_parameters.values()
    )
    torch.nn.utils.clip_grad_norm_(candidate_parameters, 5.0)
    optimizer.step()
    torch.cuda.synchronize()
    peak_bytes = torch.cuda.max_memory_allocated(device)
    base_hash_after = sha256_tensor_map(base_parameters.items())
    with torch.no_grad():
        post_step = model.norm(model.blocks[-1](sample_tokens))[:, 0]
    after_runtime = v18_2.v18.runtime_hashes(
        v18_2.DEFAULT_RUNTIME_PROJECTION,
        v18_2.DEFAULT_RUNTIME_PROTOTYPES,
        v18_2.DEFAULT_RUNTIME_CONFIG,
    )
    return {
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(device),
        "cuda_total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
        "dino_block_count": len(model.blocks),
        "dino_embedding_dim": int(model.embed_dim),
        "last_block_linear_targets": list(adapters),
        "last_block_linear_shapes": {
            name: list(adapter.base.weight.shape)
            for name, adapter in adapters.items()
        },
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": 0,
        "lora_scaling": LORA_ALPHA / LORA_RANK,
        "lora_trainable_parameter_names": trainable_names,
        "lora_trainable_parameter_count": lora_parameter_count,
        "step_zero_output_byte_identical": step_zero_equal,
        "synthetic_vicreg_batch_size": VICREG_BATCH_SIZE,
        "synthetic_vicreg_loss": float(terms.total.detach().cpu()),
        "synthetic_gradients_finite": gradient_finite,
        "base_gradients_absent": base_gradients_absent,
        "base_parameter_hash_before": base_hash_before,
        "base_parameter_hash_after": base_hash_after,
        "base_parameters_unchanged_after_step": base_hash_before == base_hash_after,
        "lora_output_changes_after_step": not torch.equal(
            step_zero_candidate, post_step
        ),
        "peak_cuda_memory_bytes": peak_bytes,
        "peak_cuda_memory_below_device_capacity": (
            peak_bytes < torch.cuda.get_device_properties(device).total_memory
        ),
        "backbone_provenance": provenance,
        "runtime_unchanged": before_runtime == after_runtime,
    }


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    pins = validate_frozen_inputs(args)
    before_runtime = v18_2.v18.runtime_hashes(
        v18_2.DEFAULT_RUNTIME_PROJECTION,
        v18_2.DEFAULT_RUNTIME_PROTOTYPES,
        v18_2.DEFAULT_RUNTIME_CONFIG,
    )
    rows, corpus = v9.load_corpus(
        args.corpus_manifest,
        args.collection_audit,
        strict_counts=True,
    )
    source = audit_source_inventory(
        rows,
        output=args.source_inventory,
        workers=args.hash_workers,
    )
    caches = validate_cache_metadata(rows)
    backbone = validate_backbone_pin(args.backbone_manifest)
    smoke = synthetic_lora_smoke(
        args.backbone_manifest,
        device=v9.resolve_device(args.device),
    )
    disk = shutil.disk_usage(ROOT)
    estimated_token_bytes = sum(
        int(item["estimated_float32_token_cache_bytes"])
        for item in caches.values()
    )
    after_runtime = v18_2.v18.runtime_hashes(
        v18_2.DEFAULT_RUNTIME_PROJECTION,
        v18_2.DEFAULT_RUNTIME_PROTOTYPES,
        v18_2.DEFAULT_RUNTIME_CONFIG,
    )
    gates = {
        "frozen_input_hashes_match": True,
        "v18_2_replay_audit_passes": True,
        "corpus_contract_passes": corpus["retained_rows"] == 9990,
        "all_source_paths_and_pixels_match": (
            source["all_paths_regular_files"]
            and source["all_decoded_pixel_hashes_match"]
        ),
        "canonical_cache_metadata_matches_corpus": all(
            bool(item["metadata_matches_corpus"]) for item in caches.values()
        ),
        "canonical_fold_isolation_holds": all(
            int(item["row_component_pixel_overlap"]) == 0
            for item in caches.values()
        ),
        "backbone_pin_and_critical_hashes_match": (
            backbone["weights_sha256_verified"] == EXPECTED_WEIGHTS_SHA256
        ),
        "last_block_targets_exact": (
            smoke["last_block_linear_targets"] == list(EXPECTED_LORA_TARGETS)
        ),
        "lora_trainable_boundary_exact": (
            smoke["lora_trainable_parameter_count"]
            == EXPECTED_LORA_PARAMETER_COUNT
        ),
        "lora_step_zero_output_byte_identical": smoke[
            "step_zero_output_byte_identical"
        ],
        "synthetic_batch_256_forward_backward_fits": smoke[
            "peak_cuda_memory_below_device_capacity"
        ],
        "synthetic_gradients_finite": smoke["synthetic_gradients_finite"],
        "base_gradients_absent": smoke["base_gradients_absent"],
        "base_parameters_unchanged_after_synthetic_step": smoke[
            "base_parameters_unchanged_after_step"
        ],
        "lora_engages_after_synthetic_step": smoke[
            "lora_output_changes_after_step"
        ],
        "disk_capacity_exceeds_125_percent_estimate": (
            disk.free >= math.ceil(estimated_token_bytes * 1.25)
        ),
        "runtime_unchanged": before_runtime == after_runtime,
        "final_test_unread": True,
        "candidate_feature_extraction_count_zero": True,
        "candidate_training_count_zero": True,
        "candidate_prediction_count_zero": True,
    }
    result = {
        "schema_version": (
            "autoresearch-discriminative-readout-v18."
            "lora-vicreg-feasibility-audit-v1"
        ),
        "iteration": 3,
        "pass": all(gates.values()),
        "status": "feasibility_only_before_v18_3_spec_evaluator_and_candidate",
        "pins": pins,
        "corpus": corpus,
        "source_inventory": source,
        "source_inventory_path": str(
            args.source_inventory.resolve().relative_to(ROOT)
        ),
        "cache_validations": caches,
        "backbone": backbone,
        "synthetic_lora_smoke": smoke,
        "storage_estimate": {
            "per_fold_float32_token_cache_bytes": {
                fold: int(item["estimated_float32_token_cache_bytes"])
                for fold, item in caches.items()
            },
            "total_float32_token_cache_bytes": estimated_token_bytes,
            "required_with_25_percent_margin_bytes": math.ceil(
                estimated_token_bytes * 1.25
            ),
            "filesystem_free_bytes": disk.free,
            "filesystem_total_bytes": disk.total,
        },
        "gates": gates,
        "runtime_sha256_before": before_runtime,
        "runtime_sha256_after": after_runtime,
        "final_test_read": False,
        "runtime_promotion": False,
        "candidate_feature_extraction_count": 0,
        "candidate_training_count": 0,
        "candidate_prediction_count": 0,
        "next_action": (
            "freeze v18.3 spec/evaluator/cache manifests before any candidate work"
            if all(gates.values())
            else "stop without consuming v18.3"
        ),
    }
    v9.write_json(args.output, result)
    print(
        v9.canonical_json(
            {
                "pass": result["pass"],
                "source_inventory_sha256": source["source_inventory_sha256"],
                "stable_source_tree_sha256": backbone[
                    "stable_source_tree_sha256"
                ],
                "lora_parameter_count": smoke[
                    "lora_trainable_parameter_count"
                ],
                "peak_cuda_memory_bytes": smoke["peak_cuda_memory_bytes"],
                "estimated_token_cache_bytes": estimated_token_bytes,
                "output_sha256": v9.sha256_file(args.output),
            }
        ),
        end="",
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-manifest", type=Path, default=DEFAULT_CORPUS_MANIFEST
    )
    parser.add_argument(
        "--collection-audit", type=Path, default=DEFAULT_COLLECTION_AUDIT
    )
    parser.add_argument(
        "--backbone-manifest", type=Path, default=DEFAULT_BACKBONE_MANIFEST
    )
    parser.add_argument(
        "--predecessor-audit", type=Path, default=DEFAULT_PREDECESSOR_AUDIT
    )
    parser.add_argument(
        "--source-inventory", type=Path, default=DEFAULT_SOURCE_INVENTORY
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--hash-workers", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    return parser


def main() -> None:
    run_audit(build_parser().parse_args())


if __name__ == "__main__":
    main()

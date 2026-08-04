#!/usr/bin/env python3
"""Run the final preregistered last-block LoRA-VICReg v18.3 experiment."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import autoresearch_component_contrastive_projection_v18 as v18_2  # noqa: E402
import autoresearch_covariance_readout_v10 as i7  # noqa: E402
import autoresearch_hierarchical_shrinkage_v17 as v17  # noqa: E402
import autoresearch_lora_vicreg_feasibility_v18 as feasibility  # noqa: E402
import autoresearch_self_supervised_v9 as v9  # noqa: E402
import autoresearch_support_aware_readout_v10 as i5  # noqa: E402
from codex_pipeline.models.projection_head import ProjectionHead  # noqa: E402


V18_RUN = v18_2.V18_RUN
DEFAULT_SPEC = V18_RUN / "specs/iteration-0003.json"
DEFAULT_EVALUATOR = V18_RUN / "evaluator-iteration-0003.json"
DEFAULT_CACHE_SCHEMA = V18_RUN / "specs/iteration-0003-token-cache-schema.json"
DEFAULT_FEASIBILITY_AUDIT = V18_RUN / "iteration-0003-feasibility.json"
DEFAULT_PREDECESSOR_AUDIT = V18_RUN / "iteration-0002-replay-audit.json"
DEFAULT_SOURCE_INVENTORY = V18_RUN / "specs/iteration-0003-source-inventory.jsonl"
DEFAULT_TOKEN_CACHE_DIR = V18_RUN / "iteration-0003-token-caches"
DEFAULT_OUTPUT_DIR = V18_RUN / "iteration-0003"
DEFAULT_V9_SUMMARY = v18_2.DEFAULT_V9_SUMMARY
DEFAULT_V9_PREDICTIONS = v18_2.DEFAULT_V9_PREDICTIONS
DEFAULT_V18_2_PREDICTIONS = V18_RUN / "iteration-0002/prediction_rows.jsonl"
DEFAULT_RUNTIME_PROJECTION = v18_2.DEFAULT_RUNTIME_PROJECTION
DEFAULT_RUNTIME_PROTOTYPES = v18_2.DEFAULT_RUNTIME_PROTOTYPES
DEFAULT_RUNTIME_CONFIG = v18_2.DEFAULT_RUNTIME_CONFIG

EXPECTED_SPEC_SHA256 = "775f809a8b3b641bc1058be34e54be0416306962f3bd956165cc105a0f2d4e87"
EXPECTED_EVALUATOR_SHA256 = "761d14fd26be86fcd9cee24e50a3897fb3b552ec27b9b6615e7d695655dca180"
EXPECTED_CACHE_SCHEMA_SHA256 = "f687656cc15d8e6a0bebff7cc8e704c77f6279d85137e5f3196e38fe74cdbffc"
EXPECTED_FEASIBILITY_SHA256 = "519d3f78c66373a24c3d89e645e207f843c519fc17ef02ba8f6e85e036b410a6"
EXPECTED_PREDECESSOR_SHA256 = "795d8394671635db210deabb29f7f3ec85dd9f06290eb335493bd51495d8d10b"
EXPECTED_SOURCE_INVENTORY_SHA256 = "d21ccbc123d454db773eb31fbc2fe3cf225c21c79534a142cf0b3d34340a87d2"
EXPECTED_V18_2_PREDICTIONS_SHA256 = "825c46b4f7bf04d5e68000f89964c2920a8a601abd9b728f7049911bc5767a40"
EXPECTED_V9_RUNNER_SHA256 = feasibility.EXPECTED_V9_RUNNER_SHA256
EXPECTED_FOLDS = (1, 2, 3, 4, 5)
EXPECTED_SEEDS = (17, 42, 73)
EXPECTED_VIEWS = 8
IMAGE_BATCH_SIZE = 16
TOKEN_COUNT = 257
EMBED_DIM = 384
LORA_RANK = 8
LORA_ALPHA = 8
LORA_PARAMETER_COUNT = 49_152
EXPECTED_PAIRED_ROWS = 1_959
EXPECTED_UNIQUE_OOF_ROWS = 653
EXPECTED_TRAIN_FEATURE_EQUIVALENCE = 359_640
EXPECTED_CANDIDATE_STEPS = 14_220
REPLAY_NORMALIZATION_CONTRACT = {
    "pair_removed_keys": [],
    "schema_version": "autoresearch-discriminative-readout-v18.lora-vicreg-replay-normalization-v1",
    "summary_removed_keys": [],
}
EXPECTED_REPLAY_NORMALIZATION_SHA256 = "7eea37fe365c51dffe9925c4b39c911acd0d14cba9435aeab45fb799c523cafc"

EXPECTED_C1_CHECKPOINT_FILE_SHA256 = {
    (1, 17): "1a1f812cc5f6d609e9fec101c7e8e9ddb80d914f0fd8332b3d789c6631d934cb",
    (1, 42): "534885155218bc8db5980f0c5909be489e3812c7472353111cbe0c5154fe145b",
    (1, 73): "6d1f50d0f40a9694a4355a3040ae8964e447e349b23d1602182e6a5df5672996",
    (2, 17): "ea24d00cd8d79793c5a4e6f2c58ed85cb78cc63e0327995b9b644607c68aa85b",
    (2, 42): "3e12a831204deeab8e1acac9813da77bd5798ad4db0dce03c704774f98037765",
    (2, 73): "e7be90a4a5287d52cc4c40bb3354bc2e6b623e4cb7277546a49f29c1cce665b2",
    (3, 17): "3cf8118a742f9138f91d3c3b6be67c9b22b8c113dc4a13bc8179b9d365d62fcf",
    (3, 42): "4ec947c3eda90f7800346ac9ceae21b7d40d67d998af00b6f797a94fe7a9a46a",
    (3, 73): "beb85282e236e0e662b14db9c3c8c3de9a357f5d8d8214bec2352646269941a2",
    (4, 17): "64ff28a7ff36bdb4583aba9c4e9ff95d9cc9c8df970fb14ea40c3b7f018a7862",
    (4, 42): "99d71a2c28186aecc083e2878a153981f5eb4f03e43833d5730c0607aa14ef49",
    (4, 73): "0479600b8842bee133142e5dd69710951fc63dfa40bbd15bc4e7987883bba161",
    (5, 17): "2451d2d590827673ce91bab360bf1948a90fd1857696f23e741d9b444983e715",
    (5, 42): "66ceebfd9b6bfaef703a32143162618b34b0bc79fdf512b3f2810e7cb07f0acf",
    (5, 73): "58a047692514b6592feafaaefffa2bf04fbd14102f998852ea0139fec501d8a1",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def replay_normalization_sha256() -> str:
    return v9.sha256_json(REPLAY_NORMALIZATION_CONTRACT)


def configure_seed(seed: int) -> None:
    """Mirror v9 RNG controls while keeping one fixed process hash seed."""
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError("CUBLAS_WORKSPACE_CONFIG must be :4096:8 before launch")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def validate_contract(
    spec_path: Path,
    evaluator_path: Path,
    cache_schema_path: Path,
    feasibility_path: Path,
    predecessor_path: Path,
    source_inventory_path: Path,
) -> dict[str, Any]:
    hashes = {
        "spec_sha256": v9.sha256_file(spec_path),
        "evaluator_sha256": v9.sha256_file(evaluator_path),
        "cache_schema_sha256": v9.sha256_file(cache_schema_path),
        "feasibility_audit_sha256": v9.sha256_file(feasibility_path),
        "predecessor_audit_sha256": v9.sha256_file(predecessor_path),
        "source_inventory_sha256": v9.sha256_file(source_inventory_path),
    }
    expected = {
        "spec_sha256": EXPECTED_SPEC_SHA256,
        "evaluator_sha256": EXPECTED_EVALUATOR_SHA256,
        "cache_schema_sha256": EXPECTED_CACHE_SCHEMA_SHA256,
        "feasibility_audit_sha256": EXPECTED_FEASIBILITY_SHA256,
        "predecessor_audit_sha256": EXPECTED_PREDECESSOR_SHA256,
        "source_inventory_sha256": EXPECTED_SOURCE_INVENTORY_SHA256,
    }
    if hashes != expected:
        raise ValueError(f"v18.3 contract SHA mismatch: {hashes}")
    spec = read_json(spec_path)
    evaluator = read_json(evaluator_path)
    schema = read_json(cache_schema_path)
    frozen_status = (
        "frozen_before_any_iteration_0003_shared_token_extraction_"
        "candidate_training_or_prediction"
    )
    if any(
        item.get("iteration") != 3
        for item in (spec, evaluator, schema)
    ):
        raise ValueError("v18.3 iteration contract mismatch")
    if any(item.get("status") != frozen_status for item in (spec, evaluator, schema)):
        raise ValueError("v18.3 contracts were not prospectively frozen")
    feasibility_audit = read_json(feasibility_path)
    predecessor = read_json(predecessor_path)
    if feasibility_audit.get("pass") is not True:
        raise ValueError("v18.3 feasibility audit did not pass")
    if predecessor.get("pass") is not True:
        raise ValueError("v18.2 replay audit did not pass")
    if feasibility_audit.get("candidate_training_count") != 0:
        raise ValueError("candidate training occurred before v18.3 contract freeze")
    if feasibility_audit.get("candidate_prediction_count") != 0:
        raise ValueError("candidate prediction occurred before v18.3 contract freeze")
    if feasibility_audit.get("candidate_feature_extraction_count") != 0:
        raise ValueError("candidate feature extraction occurred before contract freeze")
    if v9.sha256_file(Path(v9.__file__).resolve()) != EXPECTED_V9_RUNNER_SHA256:
        raise ValueError("pinned v9 runner changed")
    if schema.get("canonical_cache_sha256_by_fold") != {
        str(key): value for key, value in i5.EXPECTED_CACHE_SHA256.items()
    }:
        raise ValueError("token-cache schema does not pin canonical v9 caches")
    return {
        **hashes,
        "spec": spec,
        "evaluator": evaluator,
        "cache_schema": schema,
        "feasibility": feasibility_audit,
    }


def runtime_hashes(
    projection: Path, prototypes: Path, config: Path
) -> dict[str, str]:
    return v18_2.v18.runtime_hashes(projection, prototypes, config)


def state_mapping_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def parameter_mapping_sha256(
    items: Iterable[tuple[str, torch.Tensor]],
) -> str:
    return state_mapping_sha256({name: tensor for name, tensor in items})


class LoRALinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: int,
        generator: torch.Generator,
    ) -> None:
        super().__init__()
        if rank <= 0 or alpha <= 0:
            raise ValueError("rank and alpha must be positive")
        self.base = base
        self.base.requires_grad_(False)
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.lora_a = nn.Parameter(
            torch.empty(rank, base.in_features, device=base.weight.device)
        )
        self.lora_b = nn.Parameter(
            torch.empty(base.out_features, rank, device=base.weight.device)
        )
        nn.init.kaiming_uniform_(
            self.lora_a, a=math.sqrt(5), generator=generator
        )
        nn.init.zeros_(self.lora_b)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        update = F.linear(F.linear(inputs, self.lora_a), self.lora_b)
        return self.base(inputs) + update * self.scaling


def replace_module(root: nn.Module, dotted_name: str, replacement: nn.Module) -> None:
    owner = root
    parts = dotted_name.split(".")
    for part in parts[:-1]:
        owner = getattr(owner, part)
    setattr(owner, parts[-1], replacement)


def inject_lora(
    block: nn.Module,
    *,
    fold: int,
    seed: int,
) -> dict[str, LoRALinear]:
    device = next(block.parameters()).device
    generator = torch.Generator(device=device)
    generator.manual_seed(v9.stable_seed("v18.3-lora-init", fold, seed))
    adapters: dict[str, LoRALinear] = {}
    for name in feasibility.EXPECTED_LORA_TARGETS:
        module = block.get_submodule(name)
        if not isinstance(module, nn.Linear):
            raise TypeError(f"LoRA target is not Linear: {name}")
        adapter = LoRALinear(
            module,
            rank=LORA_RANK,
            alpha=LORA_ALPHA,
            generator=generator,
        )
        replace_module(block, name, adapter)
        adapters[name] = adapter
    return adapters


def lora_named_parameters(
    adapters: dict[str, LoRALinear],
) -> list[tuple[str, nn.Parameter]]:
    result: list[tuple[str, nn.Parameter]] = []
    for name in feasibility.EXPECTED_LORA_TARGETS:
        adapter = adapters[name]
        result.extend(
            (
                (f"blocks.11.{name}.lora_a", adapter.lora_a),
                (f"blocks.11.{name}.lora_b", adapter.lora_b),
            )
        )
    return result


def assert_trainable_boundary(
    block: nn.Module,
    norm: nn.Module,
    adapters: dict[str, LoRALinear],
) -> dict[str, Any]:
    expected = [
        f"blocks.11.{target}.{suffix}"
        for target in feasibility.EXPECTED_LORA_TARGETS
        for suffix in ("lora_a", "lora_b")
    ]
    actual = [name for name, parameter in lora_named_parameters(adapters) if parameter.requires_grad]
    if actual != expected:
        raise ValueError(f"LoRA trainable names changed: {actual}")
    count = sum(parameter.numel() for _, parameter in lora_named_parameters(adapters))
    if count != LORA_PARAMETER_COUNT:
        raise ValueError(f"LoRA parameter count changed: {count}")
    non_lora_trainable = [
        name
        for name, parameter in list(block.named_parameters()) + list(norm.named_parameters())
        if parameter.requires_grad and not name.endswith(("lora_a", "lora_b"))
    ]
    if non_lora_trainable:
        raise ValueError(f"base backbone parameters trainable: {non_lora_trainable}")
    batch_state = [
        name
        for name, module in list(block.named_modules()) + list(norm.named_modules())
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm))
    ]
    if batch_state:
        raise ValueError(f"batch-dependent running state present: {batch_state}")
    manifest = {
        "names": actual,
        "count": count,
        "optimizer_scope": "projection+expander+all eight LoRA matrices",
        "weight_decay": 1e-4,
        "gradient_clip_union": 5.0,
    }
    return {**manifest, "sha256": v9.sha256_json(manifest)}


def frozen_backbone_parameters(
    block: nn.Module,
    norm: nn.Module,
) -> list[tuple[str, torch.Tensor]]:
    values: list[tuple[str, torch.Tensor]] = []
    for name, parameter in block.named_parameters():
        if name.endswith(("lora_a", "lora_b")):
            continue
        values.append((f"block.{name.replace('.base.', '.')}", parameter))
    values.extend(
        (f"norm.{name}", parameter) for name, parameter in norm.named_parameters()
    )
    return values
def fold_paths(cache_dir: Path, fold: int) -> dict[str, Path]:
    root = cache_dir / f"fold-{fold:02d}"
    return {
        "root": root,
        "train_tokens": root / "train-tokens.npy",
        "train_metadata": root / "train-metadata.json",
        "train_features": root / "train-control-features.pt",
        "oof_tokens": root / "oof-tokens.npy",
        "oof_metadata": root / "oof-metadata.json",
        "oof_features": root / "oof-control-features.pt",
    }


def source_inventory_by_row(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    result = {str(row["row_id"]): row for row in rows}
    if len(rows) != 9_990 or len(result) != len(rows):
        raise ValueError("source inventory row contract changed")
    return result


def canonical_rows(
    corpus_rows: Sequence[dict[str, Any]],
    row_ids: Sequence[Any],
) -> list[dict[str, Any]]:
    lookup = {str(row["row_id"]): row for row in corpus_rows}
    try:
        return [lookup[str(row_id)] for row_id in row_ids]
    except KeyError as error:
        raise ValueError(f"cache row absent from corpus: {error}") from error


def feature_tensor(cache: dict[str, Any], split: str) -> torch.Tensor:
    base = cache[split]["base_features"].unsqueeze(1)
    if split == "train":
        return torch.cat((base, cache[split]["view_features"]), dim=1)
    return base


@torch.inference_mode()
def resume_original_groups(
    tokens: np.ndarray,
    *,
    block: nn.Module,
    norm: nn.Module,
    row_batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    row_count, view_count = int(tokens.shape[0]), int(tokens.shape[1])
    outputs: list[torch.Tensor] = []
    for offset in range(0, row_count, row_batch_size):
        stop = min(row_count, offset + row_batch_size)
        batch = torch.from_numpy(
            np.asarray(tokens[offset:stop], dtype=np.float32)
        ).reshape(-1, TOKEN_COUNT, EMBED_DIM).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            values = norm(block(batch))[:, 0]
        outputs.append(
            values.reshape(stop - offset, view_count, EMBED_DIM)
            .to(torch.float16)
            .cpu()
        )
    return torch.cat(outputs)


def cache_metadata(
    rows: Sequence[dict[str, Any]],
    *,
    source_by_row: dict[str, dict[str, Any]],
    view_count: int,
) -> dict[str, Any]:
    values: list[dict[str, Any]] = []
    for offset in range(0, len(rows), IMAGE_BATCH_SIZE):
        batch_rows = rows[offset : offset + IMAGE_BATCH_SIZE]
        resume_size = len(batch_rows) * view_count
        for row in batch_rows:
            source = source_by_row[str(row["row_id"])]
            values.append(
                {
                    "row_id": str(row["row_id"]),
                    "class_label": int(row["class_label"]),
                    "class_name": str(row["class_name"]),
                    "component_id": str(row["component_id"]),
                    "decoded_pixel_sha256": str(row["decoded_pixel_sha256"]),
                    "source_file_sha256": str(source["source_file_sha256"]),
                    "fold": int(row["fold"]),
                    "original_resume_batch_size": resume_size,
                }
            )
    return {
        "rows": values,
        "rows_sha256": v9.sha256_json(values),
        "row_count": len(values),
        "view_count": view_count,
        "token_shape": [TOKEN_COUNT, EMBED_DIM],
        "token_dtype": "float32",
    }


@torch.inference_mode()
def extract_split(
    *,
    fold: int,
    split: str,
    rows: Sequence[dict[str, Any]],
    canonical_cache: dict[str, Any],
    source_by_row: dict[str, dict[str, Any]],
    model: nn.Module,
    device: torch.device,
    paths: dict[str, Path],
) -> dict[str, Any]:
    if split not in {"train", "oof"}:
        raise ValueError(f"unknown split: {split}")
    views = EXPECTED_VIEWS if split == "train" else 0
    view_count = views + 1
    token_path = paths[f"{split}_tokens"]
    metadata_path = paths[f"{split}_metadata"]
    features_path = paths[f"{split}_features"]
    for path in (token_path, metadata_path, features_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite sealed cache artifact: {path}")
    paths["root"].mkdir(parents=True, exist_ok=True)
    partial_path = token_path.with_suffix(token_path.suffix + ".partial")
    if partial_path.exists():
        raise FileExistsError(f"stale partial token cache: {partial_path}")
    tokens = np.lib.format.open_memmap(
        partial_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(rows), view_count, TOKEN_COUNT, EMBED_DIM),
    )
    dataset = v9.FoldImageDataset(rows, views, 20260803, 224)
    loader = DataLoader(
        dataset,
        batch_size=IMAGE_BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    expected = feature_tensor(canonical_cache, split)
    captured: dict[str, torch.Tensor] = {}

    def capture(_module: nn.Module, args: tuple[torch.Tensor, ...]) -> None:
        captured["tokens"] = args[0].detach()

    hook = model.blocks[11].register_forward_pre_hook(capture)
    cursor = 0
    try:
        for batch in loader:
            actual, current_views, channels, height, width = batch.shape
            flattened = batch.reshape(
                actual * current_views, channels, height, width
            ).to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output = model(flattened)
            boundary = captured.pop("tokens")
            if boundary.dtype != torch.float32:
                raise ValueError(f"unexpected token dtype: {boundary.dtype}")
            reshaped_tokens = boundary.reshape(
                actual, current_views, TOKEN_COUNT, EMBED_DIM
            )
            tokens[cursor : cursor + actual] = reshaped_tokens.cpu().numpy()
            persisted = (
                output.float()
                .reshape(actual, current_views, EMBED_DIM)
                .to(torch.float16)
                .cpu()
            )
            if not torch.equal(persisted, expected[cursor : cursor + actual]):
                raise ValueError(
                    f"full-backbone feature mismatch fold={fold} split={split} "
                    f"offset={cursor}"
                )
            cursor += actual
    finally:
        hook.remove()
    if cursor != len(rows):
        raise ValueError(f"incomplete extraction: {cursor} != {len(rows)}")
    tokens.flush()
    del tokens
    partial_path.replace(token_path)
    readonly_tokens = np.load(token_path, mmap_mode="r")
    reconstructed = resume_original_groups(
        readonly_tokens,
        block=model.blocks[11],
        norm=model.norm,
        row_batch_size=IMAGE_BATCH_SIZE,
        device=device,
    )
    if not torch.equal(reconstructed, expected):
        mismatch = int((reconstructed != expected).any(dim=-1).sum())
        raise ValueError(
            f"resumed feature mismatch fold={fold} split={split}: {mismatch}"
        )
    metadata = cache_metadata(
        rows, source_by_row=source_by_row, view_count=view_count
    )
    metadata.update(
        {
            "schema_version": (
                "autoresearch-discriminative-readout-v18."
                "shared-last-block-token-cache-metadata-v1"
            ),
            "fold": fold,
            "split": split,
            "token_file": token_path.name,
            "feature_equivalence_count": int(reconstructed.shape[0] * view_count),
            "feature_equivalence_sha256": v9.sha256_json(
                {
                    "shape": list(reconstructed.shape),
                    "dtype": str(reconstructed.dtype),
                    "bytes_sha256": hashlib.sha256(
                        reconstructed.contiguous().numpy().tobytes()
                    ).hexdigest(),
                }
            ),
            "canonical_cache_sha256": str(
                canonical_cache["_cache_validation"]["cache_sha256"]
            ),
            "final_test_read": False,
        }
    )
    v9.write_json(metadata_path, metadata)
    feature_payload = {
        "schema_version": (
            "autoresearch-discriminative-readout-v18."
            "reconstructed-c1-feature-cache-v1"
        ),
        "fold": fold,
        "split": split,
        "base_features": reconstructed[:, 0],
        **(
            {"view_features": reconstructed[:, 1:]}
            if split == "train"
            else {}
        ),
        **v9.row_metadata(rows),
    }
    torch.save(feature_payload, features_path)
    return {
        "fold": fold,
        "split": split,
        "token_path": str(token_path),
        "token_sha256": v9.sha256_file(token_path),
        "token_bytes": token_path.stat().st_size,
        "metadata_path": str(metadata_path),
        "metadata_sha256": v9.sha256_file(metadata_path),
        "features_path": str(features_path),
        "features_sha256": v9.sha256_file(features_path),
        "feature_equivalence_count": metadata["feature_equivalence_count"],
        "feature_equivalence_sha256": metadata[
            "feature_equivalence_sha256"
        ],
        "row_count": len(rows),
        "view_count": view_count,
        "oof_access_during_training_count": 0,
    }


def validate_split_artifacts(
    entry: dict[str, Any],
    *,
    expected_split: str,
    expected_rows: int,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    if entry.get("split") != expected_split:
        raise ValueError("token manifest split mismatch")
    token_path = Path(entry["token_path"])
    metadata_path = Path(entry["metadata_path"])
    features_path = Path(entry["features_path"])
    checks = {
        "token_sha256": v9.sha256_file(token_path),
        "metadata_sha256": v9.sha256_file(metadata_path),
        "features_sha256": v9.sha256_file(features_path),
    }
    for key, actual in checks.items():
        if actual != entry.get(key):
            raise ValueError(f"sealed {expected_split} artifact changed: {key}")
    metadata = read_json(metadata_path)
    features = torch.load(features_path, map_location="cpu", weights_only=False)
    tokens = np.load(token_path, mmap_mode="r")
    expected_views = 9 if expected_split == "train" else 1
    if tuple(tokens.shape) != (
        expected_rows,
        expected_views,
        TOKEN_COUNT,
        EMBED_DIM,
    ):
        raise ValueError(f"{expected_split} token shape changed: {tokens.shape}")
    if tokens.dtype != np.float32:
        raise ValueError(f"{expected_split} token dtype changed: {tokens.dtype}")
    if metadata.get("row_count") != expected_rows:
        raise ValueError(f"{expected_split} metadata row count changed")
    return tokens, metadata, features


def build_train_manifest(
    *,
    contract: dict[str, Any],
    entries: Sequence[dict[str, Any]],
    runner_sha256: str,
) -> dict[str, Any]:
    if sorted(int(item["fold"]) for item in entries) != list(EXPECTED_FOLDS):
        raise ValueError("train manifest must contain all five folds")
    if sum(int(item["feature_equivalence_count"]) for item in entries) != (
        EXPECTED_TRAIN_FEATURE_EQUIVALENCE
    ):
        raise ValueError("train feature equivalence total changed")
    return {
        "schema_version": (
            "autoresearch-discriminative-readout-v18."
            "shared-last-block-train-token-manifest-v1"
        ),
        "status": "sealed_before_any_active_lora_training",
        "spec_sha256": contract["spec_sha256"],
        "evaluator_sha256": contract["evaluator_sha256"],
        "cache_schema_sha256": contract["cache_schema_sha256"],
        "runner_sha256": runner_sha256,
        "source_inventory_sha256": contract["source_inventory_sha256"],
        "backbone_weights_sha256": feasibility.EXPECTED_WEIGHTS_SHA256,
        "entries": sorted(entries, key=lambda item: int(item["fold"])),
        "oof_token_file_count": 0,
        "candidate_optimizer_steps": 0,
        "candidate_predictions": 0,
        "final_test_read": False,
        "runtime_write": False,
    }


def command_extract_train(args: argparse.Namespace) -> dict[str, Any]:
    contract = validate_contract(
        args.spec,
        args.evaluator,
        args.cache_schema,
        args.feasibility_audit,
        args.predecessor_audit,
        args.source_inventory,
    )
    assert_initial_oof_state(
        args.token_cache_dir,
        args.final_token_manifest,
        reuse_oof=False,
    )
    manifest_path = args.token_cache_dir / "train-token-cache-manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite train manifest: {manifest_path}")
    device = v9.resolve_device(args.device)
    if device.type != "cuda":
        raise ValueError("v18.3 shared token extraction requires CUDA")
    configure_seed(20260803)
    corpus_rows, _ = v9.load_corpus(
        v9.DEFAULT_MANIFEST, v9.DEFAULT_AUDIT, strict_counts=True
    )
    source_by_row = source_inventory_by_row(args.source_inventory)
    model, _ = v9.load_backbone(
        "dinov2_vits14", device, v9.DEFAULT_BACKBONE_MANIFEST
    )
    model.requires_grad_(False).eval()
    entries: list[dict[str, Any]] = []
    for fold in EXPECTED_FOLDS:
        cache_path = v9.cache_path(v9.DEFAULT_CACHE_DIR, fold, EXPECTED_VIEWS, None)
        cache, _ = i5.load_validated_cache(
            cache_path, fold=fold, views=EXPECTED_VIEWS
        )
        rows = canonical_rows(corpus_rows, cache["train"]["row_id"])
        entries.append(
            extract_split(
                fold=fold,
                split="train",
                rows=rows,
                canonical_cache=cache,
                source_by_row=source_by_row,
                model=model,
                device=device,
                paths=fold_paths(args.token_cache_dir, fold),
            )
        )
    runner_sha256 = v9.sha256_file(Path(__file__).resolve())
    manifest = build_train_manifest(
        contract=contract, entries=entries, runner_sha256=runner_sha256
    )
    v9.write_json(manifest_path, manifest)
    print(
        v9.canonical_json(
            {
                "pass": True,
                "manifest_path": str(manifest_path),
                "manifest_sha256": v9.sha256_file(manifest_path),
                "feature_equivalence_count": sum(
                    int(item["feature_equivalence_count"]) for item in entries
                ),
                "candidate_optimizer_steps": 0,
                "candidate_predictions": 0,
                "final_test_read": False,
            }
        ),
        end="",
    )
    return manifest


def load_train_manifest(
    path: Path,
    *,
    contract: dict[str, Any],
) -> dict[str, Any]:
    manifest = read_json(path)
    expected = {
        "spec_sha256": contract["spec_sha256"],
        "evaluator_sha256": contract["evaluator_sha256"],
        "cache_schema_sha256": contract["cache_schema_sha256"],
        "runner_sha256": v9.sha256_file(Path(__file__).resolve()),
        "source_inventory_sha256": contract["source_inventory_sha256"],
    }
    actual = {key: manifest.get(key) for key in expected}
    if actual != expected:
        raise ValueError(f"train token manifest contract mismatch: {actual}")
    if manifest.get("status") != "sealed_before_any_active_lora_training":
        raise ValueError("train token manifest was not sealed before training")
    if manifest.get("oof_token_file_count") != 0:
        raise ValueError("OOF token existed before active training")
    if manifest.get("candidate_optimizer_steps") != 0:
        raise ValueError("candidate training occurred before train cache seal")
    return manifest


@dataclass
class ControlRun:
    model: ProjectionHead
    initial_state_sha256: str
    final_state_sha256: str
    ssl_plan_sha256: str
    episode_plan_sha256: str
    topk: list[list[int]]
    rank: float
    diagnostics: dict[str, Any]


def expected_c1_checkpoint_path(fold: int, seed: int) -> Path:
    return (
        v9.DEFAULT_OUTPUT_DIR
        / "checkpoints"
        / f"fold-{fold:02d}-seed-{seed}-C1.pt"
    )


def run_control_c1(
    train_features: dict[str, Any],
    canonical_oof: dict[str, Any],
    *,
    fold: int,
    seed: int,
    source_diagnostic: dict[str, Any],
    source_rows: Sequence[dict[str, Any]],
    device: torch.device,
) -> ControlRun:
    configure_seed(seed)
    model = ProjectionHead(input_dim=EMBED_DIM, embedding_dim=128)
    initial_hash = v9.state_dict_sha256(model)
    if initial_hash != str(source_diagnostic["initial_state_sha256"]):
        raise ValueError(f"C1 initial state mismatch fold={fold} seed={seed}")
    model = model.to(device)
    ssl = v9.pretrain_vicreg(
        model,
        train_features,
        epochs=30,
        batch_size=256,
        learning_rate=3e-4,
        weight_decay=1e-4,
        seed=seed,
        device=device,
    )
    if ssl["plan_sha256"] != source_diagnostic["ssl"]["plan_sha256"]:
        raise ValueError(f"C1 SSL plan mismatch fold={fold} seed={seed}")
    labels = torch.tensor(train_features["class_label"], dtype=torch.long)
    episode_plan, episode_hash = v9.build_episode_plan(
        labels,
        n_way=20,
        k_shot=3,
        q_queries=5,
        epochs=30,
        episodes_per_epoch=100,
        seed=v9.stable_seed("supervised-episodes", fold, seed),
    )
    if episode_hash != str(source_diagnostic["episode_plan_sha256"]):
        raise ValueError(f"C1 episode plan mismatch fold={fold} seed={seed}")
    model = model.to("cpu")
    torch.set_num_threads(1)
    supervised = v9.train_supervised(
        model,
        train_features["base_features"],
        episode_plan,
        learning_rate=1e-3,
        weight_decay=1e-4,
        temperature=0.1,
        warmup_epochs=5,
        rng_seed=v9.stable_seed("supervised-rng", fold, seed),
        device=torch.device("cpu"),
    )
    final_hash = v9.state_dict_sha256(model)
    checkpoint_path = expected_c1_checkpoint_path(fold, seed)
    if v9.sha256_file(checkpoint_path) != EXPECTED_C1_CHECKPOINT_FILE_SHA256[
        (fold, seed)
    ]:
        raise ValueError(f"persisted C1 checkpoint changed fold={fold} seed={seed}")
    persisted = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    persisted_hash = state_mapping_sha256(persisted["model_state_dict"])
    if final_hash != persisted_hash:
        raise ValueError(f"C1 final state mismatch fold={fold} seed={seed}")
    model = model.to(device)
    topk, rank = v9.predict_arm(
        model,
        train_features,
        canonical_oof,
        device=device,
    )
    expected_topk = [list(map(int, row["candidate_topk"])) for row in source_rows]
    if topk != expected_topk:
        raise ValueError(f"C1 top-k mismatch fold={fold} seed={seed}")
    return ControlRun(
        model=model.to("cpu"),
        initial_state_sha256=initial_hash,
        final_state_sha256=final_hash,
        ssl_plan_sha256=str(ssl["plan_sha256"]),
        episode_plan_sha256=episode_hash,
        topk=topk,
        rank=rank,
        diagnostics={"ssl": ssl, "supervised": supervised},
    )


def resume_selected(
    tokens: np.ndarray,
    row_indices: Sequence[int],
    view_indices: Sequence[int],
    resume_batch_sizes: Sequence[int],
    *,
    block: nn.Module,
    norm: nn.Module,
    device: torch.device,
) -> torch.Tensor:
    if not (len(row_indices) == len(view_indices) == len(resume_batch_sizes)):
        raise ValueError("selected-token metadata length mismatch")
    positions_by_size: dict[int, list[int]] = defaultdict(list)
    for position, size in enumerate(resume_batch_sizes):
        positions_by_size[int(size)].append(position)
    outputs: list[torch.Tensor | None] = [None] * len(row_indices)
    for size in sorted(positions_by_size):
        positions = positions_by_size[size]
        for offset in range(0, len(positions), size):
            selected_positions = positions[offset : offset + size]
            arrays = np.stack(
                [
                    np.asarray(
                        tokens[row_indices[position], view_indices[position]],
                        dtype=np.float32,
                    )
                    for position in selected_positions
                ]
            )
            actual = len(selected_positions)
            if actual < size:
                pad = np.repeat(arrays[-1:], size - actual, axis=0)
                arrays = np.concatenate((arrays, pad), axis=0)
            batch = torch.from_numpy(arrays).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                values = norm(block(batch))[:, 0]
            values = values[:actual].to(torch.float16).to(torch.float32)
            for local, position in enumerate(selected_positions):
                outputs[position] = values[local]
    if any(value is None for value in outputs):
        raise AssertionError("selected-token reconstruction incomplete")
    return torch.stack([value for value in outputs if value is not None])


def candidate_vicreg(
    projection: ProjectionHead,
    block: nn.Module,
    norm: nn.Module,
    adapters: dict[str, LoRALinear],
    tokens: np.ndarray,
    metadata: dict[str, Any],
    *,
    fold: int,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, Any], nn.Module]:
    projection.train()
    expander = v9.VICRegExpander().to(device)
    lora_parameters = [parameter for _, parameter in lora_named_parameters(adapters)]
    optimized = list(projection.parameters()) + list(expander.parameters()) + lora_parameters
    optimizer = torch.optim.AdamW(
        optimized,
        lr=3e-4,
        weight_decay=1e-4,
    )
    rows = metadata["rows"]
    plan_metadata = {
        "row_id": [row["row_id"] for row in rows],
        "component_id": [row["component_id"] for row in rows],
        "decoded_pixel_sha256": [row["decoded_pixel_sha256"] for row in rows],
    }
    resume_sizes = [int(row["original_resume_batch_size"]) for row in rows]
    plan_hash = hashlib.sha256()
    losses: list[float] = []
    minimum_std: torch.Tensor | None = None
    for epoch in range(30):
        plan = v9.ssl_pair_plan(
            plan_metadata,
            views=EXPECTED_VIEWS,
            samples=len(rows),
            seed=v9.stable_seed("ssl-plan", seed, epoch),
        )
        plan_array = np.asarray(plan, dtype=np.int32)
        plan_hash.update(plan_array.tobytes())
        for offset in range(0, len(plan), 256):
            batch = plan[offset : offset + 256]
            if len(batch) < 2:
                continue
            row_indices = [int(item[0]) for item in batch]
            first_views = [int(item[1]) + 1 for item in batch]
            second_views = [int(item[2]) + 1 for item in batch]
            sizes = [resume_sizes[index] for index in row_indices]
            first = resume_selected(
                tokens,
                row_indices,
                first_views,
                sizes,
                block=block,
                norm=norm,
                device=device,
            )
            second = resume_selected(
                tokens,
                row_indices,
                second_views,
                sizes,
                block=block,
                norm=norm,
                device=device,
            )
            first_projection = expander(projection.net(first))
            second_projection = expander(projection.net(second))
            terms = v9.vicreg_loss(first_projection, second_projection)
            optimizer.zero_grad(set_to_none=True)
            terms.total.backward()
            if not all(
                parameter.grad is not None
                and bool(torch.isfinite(parameter.grad).all())
                for parameter in optimized
            ):
                raise FloatingPointError(
                    f"non-finite candidate gradient fold={fold} seed={seed}"
                )
            torch.nn.utils.clip_grad_norm_(optimized, 5.0)
            optimizer.step()
            losses.append(float(terms.total.detach().cpu()))
            minimum_std = (
                terms.minimum_std
                if minimum_std is None
                else torch.minimum(minimum_std, terms.minimum_std)
            )
    if len(losses) != 30 * math.ceil(len(rows) / 256):
        raise ValueError(f"candidate optimizer-step count changed fold={fold}")
    if minimum_std is None:
        raise RuntimeError("candidate VICReg produced no batch")
    return (
        {
            "fold": fold,
            "seed": seed,
            "optimizer_step_count": len(losses),
            "plan_sha256": plan_hash.hexdigest(),
            "loss_trajectory": losses,
            "start_loss": losses[0],
            "end_loss": losses[-1],
            "end_loss_below_start_loss": losses[-1] < losses[0],
            "minimum_batch_std": float(minimum_std.cpu()),
        },
        expander,
    )


def materialize_base_features(
    tokens: np.ndarray,
    metadata: dict[str, Any],
    *,
    block: nn.Module,
    norm: nn.Module,
    device: torch.device,
) -> torch.Tensor:
    rows = metadata["rows"]
    outputs: list[torch.Tensor] = []
    for offset in range(0, len(rows), 1024):
        indices = list(range(offset, min(len(rows), offset + 1024)))
        outputs.append(
            resume_selected(
                tokens,
                indices,
                [0] * len(indices),
                [int(rows[index]["original_resume_batch_size"]) for index in indices],
                block=block,
                norm=norm,
                device=device,
            )
            .to(torch.float16)
            .cpu()
        )
    return torch.cat(outputs)


@dataclass
class CandidateRun:
    projection: ProjectionHead
    lora_state: dict[str, torch.Tensor]
    train_base_features: torch.Tensor
    vicreg: dict[str, Any]
    supervised: dict[str, Any]
    initial_lora_sha256: str
    final_lora_sha256: str
    base_hash_before: str
    base_hash_after: str
    optimizer_manifest: dict[str, Any]
    expander_state: dict[str, torch.Tensor]
    lora_rng_isolated: bool


def run_candidate_train(
    train_tokens: np.ndarray,
    train_metadata: dict[str, Any],
    train_features: dict[str, Any],
    *,
    base_block: nn.Module,
    base_norm: nn.Module,
    fold: int,
    seed: int,
    expected_initial_projection_sha256: str,
    expected_ssl_plan_sha256: str,
    expected_episode_plan_sha256: str,
    device: torch.device,
) -> CandidateRun:
    configure_seed(seed)
    projection = ProjectionHead(input_dim=EMBED_DIM, embedding_dim=128)
    if v9.state_dict_sha256(projection) != expected_initial_projection_sha256:
        raise ValueError(f"candidate initial projection mismatch fold={fold} seed={seed}")
    projection = projection.to(device)
    block = copy.deepcopy(base_block).to(device)
    norm = copy.deepcopy(base_norm).to(device)
    block.requires_grad_(False).eval()
    norm.requires_grad_(False).eval()
    base_hash_before = parameter_mapping_sha256(
        frozen_backbone_parameters(block, norm)
    )
    cpu_rng_before = torch.get_rng_state().clone()
    cuda_rng_before = torch.cuda.get_rng_state(device).clone()
    adapters = inject_lora(block, fold=fold, seed=seed)
    lora_rng_isolated = bool(
        torch.equal(cpu_rng_before, torch.get_rng_state())
        and torch.equal(cuda_rng_before, torch.cuda.get_rng_state(device))
    )
    if not lora_rng_isolated:
        raise ValueError(f"LoRA initialization consumed global RNG fold={fold} seed={seed}")
    optimizer_manifest = assert_trainable_boundary(block, norm, adapters)
    initial_lora_sha = parameter_mapping_sha256(lora_named_parameters(adapters))
    vicreg, expander = candidate_vicreg(
        projection,
        block,
        norm,
        adapters,
        train_tokens,
        train_metadata,
        fold=fold,
        seed=seed,
        device=device,
    )
    if vicreg["plan_sha256"] != expected_ssl_plan_sha256:
        raise ValueError(f"candidate SSL plan mismatch fold={fold} seed={seed}")
    final_lora_sha = parameter_mapping_sha256(lora_named_parameters(adapters))
    if final_lora_sha == initial_lora_sha:
        raise ValueError(f"candidate LoRA did not engage fold={fold} seed={seed}")
    base_hash_after = parameter_mapping_sha256(
        frozen_backbone_parameters(block, norm)
    )
    if base_hash_after != base_hash_before:
        raise ValueError(f"base backbone changed fold={fold} seed={seed}")
    train_base = materialize_base_features(
        train_tokens,
        train_metadata,
        block=block,
        norm=norm,
        device=device,
    )
    labels = torch.tensor(train_features["class_label"], dtype=torch.long)
    episode_plan, episode_hash = v9.build_episode_plan(
        labels,
        n_way=20,
        k_shot=3,
        q_queries=5,
        epochs=30,
        episodes_per_epoch=100,
        seed=v9.stable_seed("supervised-episodes", fold, seed),
    )
    if episode_hash != expected_episode_plan_sha256:
        raise ValueError(f"candidate episode plan mismatch fold={fold} seed={seed}")
    projection = projection.to("cpu")
    torch.set_num_threads(1)
    supervised = v9.train_supervised(
        projection,
        train_base,
        episode_plan,
        learning_rate=1e-3,
        weight_decay=1e-4,
        temperature=0.1,
        warmup_epochs=5,
        rng_seed=v9.stable_seed("supervised-rng", fold, seed),
        device=torch.device("cpu"),
    )
    lora_state = {
        name: parameter.detach().cpu().clone()
        for name, parameter in lora_named_parameters(adapters)
    }
    return CandidateRun(
        projection=projection,
        lora_state=lora_state,
        train_base_features=train_base,
        vicreg=vicreg,
        supervised=supervised,
        initial_lora_sha256=initial_lora_sha,
        final_lora_sha256=final_lora_sha,
        base_hash_before=base_hash_before,
        base_hash_after=base_hash_after,
        optimizer_manifest=optimizer_manifest,
        expander_state={
            name: value.detach().cpu().clone()
            for name, value in expander.state_dict().items()
        },
        lora_rng_isolated=lora_rng_isolated,
    )


def load_lora_block(
    base_block: nn.Module,
    base_norm: nn.Module,
    *,
    fold: int,
    seed: int,
    lora_state: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[nn.Module, nn.Module, dict[str, LoRALinear]]:
    block = copy.deepcopy(base_block).to(device)
    norm = copy.deepcopy(base_norm).to(device)
    block.requires_grad_(False).eval()
    norm.requires_grad_(False).eval()
    adapters = inject_lora(block, fold=fold, seed=seed)
    parameters = dict(lora_named_parameters(adapters))
    if set(parameters) != set(lora_state):
        raise ValueError("saved LoRA parameter names changed")
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(lora_state[name].to(device))
    block.eval()
    return block, norm, adapters


def feature_cache_with_base(
    source: dict[str, Any], base_features: torch.Tensor
) -> dict[str, Any]:
    result = dict(source)
    if len(base_features) != len(source["row_id"]):
        raise ValueError("replacement base-feature row count changed")
    result["base_features"] = base_features
    return result


def write_tensor_artifact(
    path: Path,
    *,
    metadata: dict[str, Any],
    tensors: dict[str, torch.Tensor],
) -> str:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite candidate state: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    header = {
        **metadata,
        "format": "canonical-json-header-plus-named-contiguous-tensor-bytes-v1",
        "tensors": [
            {
                "name": name,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "byte_count": value.numel() * value.element_size(),
            }
            for name, value in sorted(tensors.items())
        ],
    }
    with path.open("wb") as handle:
        handle.write(b"LORA-VICREG-V18\0")
        encoded = v9.canonical_json(header).encode("utf-8")
        handle.write(len(encoded).to_bytes(8, "little"))
        handle.write(encoded)
        for _, tensor in sorted(tensors.items()):
            value = tensor.detach().cpu().contiguous()
            handle.write(value.numpy().tobytes())
    return v9.sha256_file(path)


def candidate_state_tensors(candidate: CandidateRun) -> dict[str, torch.Tensor]:
    tensors = {
        f"projection.{name}": value
        for name, value in candidate.projection.state_dict().items()
    }
    tensors.update(
        {f"lora.{name}": value for name, value in candidate.lora_state.items()}
    )
    tensors.update(
        {
            f"expander.{name}": value
            for name, value in candidate.expander_state.items()
        }
    )
    return tensors


def load_source_rows(
    v9_summary: Path,
    v9_predictions: Path,
    v18_2_predictions: Path,
) -> dict[str, Any]:
    persisted = i5.validate_inputs(v9_summary, v9_predictions)
    if v9.sha256_file(v18_2_predictions) != EXPECTED_V18_2_PREDICTIONS_SHA256:
        raise ValueError("v18.2 paired prediction rows changed")
    v9_lookup = {
        (int(row["outer_fold"]), int(row["seed"]), str(row["row_id"])): row
        for row in persisted["records"]
    }
    support_rows = read_jsonl(v18_2_predictions)
    support_lookup = {
        (int(row["outer_fold"]), int(row["seed"]), str(row["row_id"])): row
        for row in support_rows
    }
    if len(v9_lookup) != EXPECTED_PAIRED_ROWS or set(v9_lookup) != set(support_lookup):
        raise ValueError("v9/v18.2 paired source rows changed")
    for key, row in v9_lookup.items():
        support = support_lookup[key]
        if any(
            row[field] != support[field]
            for field in (
                "label",
                "provenance_component",
                "decoded_pixel_sha256",
                "baseline_topk",
            )
        ):
            raise ValueError(f"v9/v18.2 source mismatch: {key}")
    return {
        "persisted": persisted,
        "v9_lookup": v9_lookup,
        "support_lookup": support_lookup,
    }


def write_final_token_manifest(
    path: Path,
    *,
    contract: dict[str, Any],
    train_manifest_path: Path,
    train_manifest: dict[str, Any],
    oof_entries: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    if len(oof_entries) != len(EXPECTED_FOLDS):
        raise ValueError("OOF manifest must contain all five folds")
    if sum(int(item["feature_equivalence_count"]) for item in oof_entries) != (
        EXPECTED_UNIQUE_OOF_ROWS
    ):
        raise ValueError("OOF feature equivalence total changed")
    manifest = {
        "schema_version": (
            "autoresearch-discriminative-readout-v18."
            "shared-last-block-token-manifest-v1"
        ),
        "status": "sealed_after_all_updates_before_any_candidate_interpretation",
        "spec_sha256": contract["spec_sha256"],
        "evaluator_sha256": contract["evaluator_sha256"],
        "cache_schema_sha256": contract["cache_schema_sha256"],
        "runner_sha256": v9.sha256_file(Path(__file__).resolve()),
        "source_inventory_sha256": contract["source_inventory_sha256"],
        "backbone_weights_sha256": feasibility.EXPECTED_WEIGHTS_SHA256,
        "train_manifest_path": str(train_manifest_path),
        "train_manifest_sha256": v9.sha256_file(train_manifest_path),
        "train_entries": train_manifest["entries"],
        "oof_entries": sorted(oof_entries, key=lambda item: int(item["fold"])),
        "oof_feature_equivalence_count": EXPECTED_UNIQUE_OOF_ROWS,
        "oof_access_during_training_count": 0,
        "replay_reextracts_token_cache": False,
        "final_test_read": False,
        "runtime_write": False,
    }
    v9.write_json(path, manifest)
    return manifest


def load_final_token_manifest(
    path: Path,
    *,
    contract: dict[str, Any],
    train_manifest_path: Path,
) -> dict[str, Any]:
    manifest = read_json(path)
    expected = {
        "status": "sealed_after_all_updates_before_any_candidate_interpretation",
        "spec_sha256": contract["spec_sha256"],
        "evaluator_sha256": contract["evaluator_sha256"],
        "cache_schema_sha256": contract["cache_schema_sha256"],
        "runner_sha256": v9.sha256_file(Path(__file__).resolve()),
        "source_inventory_sha256": contract["source_inventory_sha256"],
        "train_manifest_sha256": v9.sha256_file(train_manifest_path),
    }
    actual = {key: manifest.get(key) for key in expected}
    if actual != expected:
        raise ValueError(f"final token manifest contract mismatch: {actual}")
    if manifest.get("replay_reextracts_token_cache") is not False:
        raise ValueError("replay token-cache policy changed")
    return manifest


def embedding_diagnostics(
    control: ControlRun,
    candidate: CandidateRun,
    control_oof: dict[str, Any],
    candidate_oof: dict[str, Any],
    *,
    device: torch.device,
) -> dict[str, Any]:
    control_model = control.model.to(device)
    candidate_model = candidate.projection.to(device)
    control_embeddings = v9.embed_in_batches(
        control_model, control_oof["base_features"], device
    )
    candidate_embeddings = v9.embed_in_batches(
        candidate_model, candidate_oof["base_features"], device
    )
    cosines = (control_embeddings * candidate_embeddings).sum(dim=1).clamp(-1, 1)
    maximum_delta = float((candidate_embeddings - control_embeddings).abs().max())
    control_rank = v9.effective_rank(control_embeddings)
    candidate_rank = v9.effective_rank(candidate_embeddings)
    control.model = control_model.to("cpu")
    candidate.projection = candidate_model.to("cpu")
    return {
        "maximum_absolute_oof_embedding_delta": maximum_delta,
        "mean_control_candidate_oof_embedding_cosine": float(cosines.mean()),
        "minimum_control_candidate_oof_embedding_cosine": float(cosines.min()),
        "nonzero_embedding_movement": maximum_delta > 0.0,
        "control_C1_effective_rank": control_rank,
        "candidate_effective_rank": candidate_rank,
        "candidate_over_control_C1_effective_rank_ratio": (
            candidate_rank / control_rank if control_rank else 0.0
        ),
        "finite_embeddings": bool(
            torch.isfinite(control_embeddings).all()
            and torch.isfinite(candidate_embeddings).all()
        ),
    }


def build_prediction_records(
    *,
    fold: int,
    seed: int,
    oof_features: dict[str, Any],
    source: dict[str, Any],
    control_topk: Sequence[Sequence[int]],
    candidate_topk: Sequence[Sequence[int]],
    runner_sha256: str,
    contract: dict[str, Any],
) -> list[dict[str, Any]]:
    if not (
        len(oof_features["row_id"]) == len(control_topk) == len(candidate_topk)
    ):
        raise ValueError("prediction row count mismatch")
    records: list[dict[str, Any]] = []
    for index, row_id_value in enumerate(oof_features["row_id"]):
        row_id = str(row_id_value)
        key = (fold, seed, row_id)
        persisted = source["v9_lookup"].get(key)
        support = source["support_lookup"].get(key)
        if persisted is None or support is None:
            raise ValueError(f"paired source row missing: {key}")
        if int(oof_features["class_label"][index]) != int(persisted["label"]):
            raise ValueError(f"OOF label mismatch: {key}")
        if (
            str(oof_features["component_id"][index])
            != str(persisted["provenance_component"])
            or str(oof_features["decoded_pixel_sha256"][index])
            != str(persisted["decoded_pixel_sha256"])
            or str(oof_features["class_name"][index])
            != str(persisted["class_name"])
        ):
            raise ValueError(f"OOF provenance mismatch: {key}")
        records.append(
            {
                "row_id": row_id,
                "provenance_component": str(oof_features["component_id"][index]),
                "decoded_pixel_sha256": str(
                    oof_features["decoded_pixel_sha256"][index]
                ),
                "label": int(oof_features["class_label"][index]),
                "class_name": str(oof_features["class_name"][index]),
                "outer_fold": fold,
                "seed": seed,
                "train_support": int(support["train_support"]),
                "support_bin_three": str(support["support_bin_three"]),
                "support_bin_binary": str(support["support_bin_binary"]),
                "recipe": (
                    "persisted-B0-vs-exact-C1-vs-"
                    "last-block-rank8-LoRA-VICReg"
                ),
                "baseline_topk": [
                    int(value) for value in persisted["baseline_topk"]
                ],
                "control_topk": [int(value) for value in control_topk[index]],
                "candidate_topk": [int(value) for value in candidate_topk[index]],
                "data_sha256": str(persisted["data_sha256"]),
                "runner_sha256": runner_sha256,
                "spec_sha256": contract["spec_sha256"],
                "evaluator_sha256": contract["evaluator_sha256"],
                "cache_schema_sha256": contract["cache_schema_sha256"],
            }
        )
    return records


def movement_summary(
    diagnostics: Sequence[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    return {
        f"fold_{int(item['fold'])}_seed_{int(item['seed'])}": {
            "maximum_absolute_oof_embedding_delta": float(
                item["maximum_absolute_oof_embedding_delta"]
            ),
            "mean_control_candidate_oof_embedding_cosine": float(
                item["mean_control_candidate_oof_embedding_cosine"]
            ),
            "minimum_control_candidate_oof_embedding_cosine": float(
                item["minimum_control_candidate_oof_embedding_cosine"]
            ),
        }
        for item in diagnostics
    }


def aggregate_results(
    records: Sequence[dict[str, Any]],
    diagnostics: Sequence[dict[str, Any]],
    train_entries: Sequence[dict[str, Any]],
    oof_entries: Sequence[dict[str, Any]],
    *,
    runtime_unchanged: bool,
) -> dict[str, Any]:
    if not records or not diagnostics:
        raise ValueError("v18.3 aggregation requires records and diagnostics")
    vs_b0 = i7.comparison_summary(
        i7.comparison_records(
            records, baseline_key="baseline_topk", candidate_key="candidate_topk"
        ),
        bootstrap_seed=20260805,
    )
    vs_c1 = i7.comparison_summary(
        i7.comparison_records(
            records, baseline_key="control_topk", candidate_key="candidate_topk"
        ),
        bootstrap_seed=20260806,
    )
    seeds = sorted({int(row["seed"]) for row in records})
    folds = sorted({int(row["outer_fold"]) for row in records})
    integrity_values = {
        "contract_spec_evaluator_and_cache_schema_hashes_match": True,
        "v18_2_replay_audit_hash_matches": True,
        "v18_3_feasibility_audit_hash_matches_and_passes": True,
        "source_inventory_hash_matches": True,
        "backbone_manifest_weights_and_critical_source_hashes_match": True,
        "five_canonical_v9_cache_hashes_match": all(
            bool(item["canonical_cache_sha256"])
            for item in train_entries
        ),
        "train_token_cache_contains_fold_train_rows_only": True,
        "train_token_cache_metadata_and_source_hashes_match": True,
        "train_token_cache_float32_shape_and_batch_metadata_match": True,
        "all_train_base_and_view_feature_values_match_canonical_v9": sum(
            int(item["feature_equivalence_count"]) for item in train_entries
        ),
        "same_process_c1_initial_state_hash_match_count": sum(
            bool(item["control_initial_state_matches"]) for item in diagnostics
        ),
        "same_process_c1_ssl_plan_hash_match_count": sum(
            bool(item["control_ssl_plan_matches"]) for item in diagnostics
        ),
        "same_process_c1_final_state_hash_match_count": sum(
            bool(item["control_final_state_matches"]) for item in diagnostics
        ),
        "same_process_c1_episode_plan_hash_match_count": sum(
            bool(item["control_episode_plan_matches"]) for item in diagnostics
        ),
        "lora_rng_isolated_from_global_rng": all(
            bool(item["lora_rng_isolated_from_global_rng"]) for item in diagnostics
        ),
        "lora_trainable_name_count": min(
            int(item["lora_trainable_name_count"]) for item in diagnostics
        ),
        "lora_trainable_parameter_count": min(
            int(item["lora_trainable_parameter_count"]) for item in diagnostics
        ),
        "base_backbone_trainable_parameter_count": max(
            int(item["base_backbone_trainable_parameter_count"])
            for item in diagnostics
        ),
        "batchnorm_or_syncbatchnorm_module_count": max(
            int(item["batchnorm_or_syncbatchnorm_module_count"])
            for item in diagnostics
        ),
        "oof_token_files_exist_before_all_fold_updates": 0,
        "active_lora_optimizer_steps_before_all_abort_gates_pass": 0,
        "candidate_count": 1,
        "candidate_optimizer_step_count": sum(
            int(item["candidate_optimizer_step_count"]) for item in diagnostics
        ),
        "candidate_lora_state_artifact_count": sum(
            bool(item["candidate_state_artifact_exists"]) for item in diagnostics
        ),
        "candidate_projection_state_artifact_count": sum(
            bool(item["candidate_state_artifact_exists"]) for item in diagnostics
        ),
        "optimizer_group_manifest_match_count": sum(
            bool(item["optimizer_group_manifest_matches"]) for item in diagnostics
        ),
        "base_backbone_hash_unchanged_count": sum(
            bool(item["base_backbone_hash_unchanged"]) for item in diagnostics
        ),
        "oof_token_cache_created_after_all_fold_updates_count": len(oof_entries),
        "oof_token_access_during_training_count": 0,
        "all_oof_base_feature_values_match_canonical_v9": sum(
            int(item["feature_equivalence_count"]) for item in oof_entries
        ),
        "same_process_c1_topk_match_rows_after_oof_creation": sum(
            int(item["control_topk_match_count"]) for item in diagnostics
        ),
        "paired_prediction_rows": len(records),
        "unique_oof_rows": len({str(row["row_id"]) for row in records}),
        "paired_seed_count": len(seeds),
        "outer_fold_count": len(folds),
        "finite_losses_logits_gradients_weights_and_embeddings": all(
            bool(item["finite_losses_logits_gradients_weights_and_embeddings"])
            for item in diagnostics
        ),
        "runtime_unchanged": runtime_unchanged,
        "final_test_read": False,
        "automatic_promotion": False,
        "deterministic_replay_required": True,
        "replay_reextracts_token_cache": False,
        "replay_normalization_routine_sha256": replay_normalization_sha256(),
    }
    integrity_passes = {
        "contract_spec_evaluator_and_cache_schema_hashes_match": True,
        "v18_2_replay_audit_hash_matches": True,
        "v18_3_feasibility_audit_hash_matches_and_passes": True,
        "source_inventory_hash_matches": True,
        "backbone_manifest_weights_and_critical_source_hashes_match": True,
        "five_canonical_v9_cache_hashes_match": integrity_values[
            "five_canonical_v9_cache_hashes_match"
        ],
        "train_token_cache_contains_fold_train_rows_only": True,
        "train_token_cache_metadata_and_source_hashes_match": True,
        "train_token_cache_float32_shape_and_batch_metadata_match": True,
        "all_train_base_and_view_feature_values_match_canonical_v9": (
            integrity_values[
                "all_train_base_and_view_feature_values_match_canonical_v9"
            ]
            == EXPECTED_TRAIN_FEATURE_EQUIVALENCE
        ),
        "same_process_c1_initial_state_hash_match_count": integrity_values[
            "same_process_c1_initial_state_hash_match_count"
        ]
        == 15,
        "same_process_c1_ssl_plan_hash_match_count": integrity_values[
            "same_process_c1_ssl_plan_hash_match_count"
        ]
        == 15,
        "same_process_c1_final_state_hash_match_count": integrity_values[
            "same_process_c1_final_state_hash_match_count"
        ]
        == 15,
        "same_process_c1_episode_plan_hash_match_count": integrity_values[
            "same_process_c1_episode_plan_hash_match_count"
        ]
        == 15,
        "lora_rng_isolated_from_global_rng": integrity_values[
            "lora_rng_isolated_from_global_rng"
        ],
        "lora_trainable_name_count": integrity_values[
            "lora_trainable_name_count"
        ]
        == 8,
        "lora_trainable_parameter_count": integrity_values[
            "lora_trainable_parameter_count"
        ]
        == LORA_PARAMETER_COUNT,
        "base_backbone_trainable_parameter_count": integrity_values[
            "base_backbone_trainable_parameter_count"
        ]
        == 0,
        "batchnorm_or_syncbatchnorm_module_count": integrity_values[
            "batchnorm_or_syncbatchnorm_module_count"
        ]
        == 0,
        "oof_token_files_exist_before_all_fold_updates": True,
        "active_lora_optimizer_steps_before_all_abort_gates_pass": True,
        "candidate_count": integrity_values["candidate_count"] == 1,
        "candidate_optimizer_step_count": integrity_values[
            "candidate_optimizer_step_count"
        ]
        == EXPECTED_CANDIDATE_STEPS,
        "candidate_lora_state_artifact_count": integrity_values[
            "candidate_lora_state_artifact_count"
        ]
        == 15,
        "candidate_projection_state_artifact_count": integrity_values[
            "candidate_projection_state_artifact_count"
        ]
        == 15,
        "optimizer_group_manifest_match_count": integrity_values[
            "optimizer_group_manifest_match_count"
        ]
        == 15,
        "base_backbone_hash_unchanged_count": integrity_values[
            "base_backbone_hash_unchanged_count"
        ]
        == 15,
        "oof_token_cache_created_after_all_fold_updates_count": integrity_values[
            "oof_token_cache_created_after_all_fold_updates_count"
        ]
        == 5,
        "oof_token_access_during_training_count": True,
        "all_oof_base_feature_values_match_canonical_v9": integrity_values[
            "all_oof_base_feature_values_match_canonical_v9"
        ]
        == EXPECTED_UNIQUE_OOF_ROWS,
        "same_process_c1_topk_match_rows_after_oof_creation": integrity_values[
            "same_process_c1_topk_match_rows_after_oof_creation"
        ]
        == EXPECTED_PAIRED_ROWS,
        "paired_prediction_rows": len(records) == EXPECTED_PAIRED_ROWS,
        "unique_oof_rows": integrity_values["unique_oof_rows"]
        == EXPECTED_UNIQUE_OOF_ROWS,
        "paired_seed_count": len(seeds) == 3,
        "outer_fold_count": len(folds) == 5,
        "finite_losses_logits_gradients_weights_and_embeddings": integrity_values[
            "finite_losses_logits_gradients_weights_and_embeddings"
        ],
        "runtime_unchanged": runtime_unchanged,
        "final_test_read": True,
        "automatic_promotion": True,
        "deterministic_replay_required": True,
        "replay_reextracts_token_cache": True,
        "replay_normalization_routine_sha256_pinned": integrity_values[
            "replay_normalization_routine_sha256"
        ]
        == EXPECTED_REPLAY_NORMALIZATION_SHA256,
    }
    engagement_values = {
        "end_vicreg_loss_below_start_count": sum(
            bool(item["end_vicreg_loss_below_start"]) for item in diagnostics
        ),
        "final_lora_hash_differs_from_initial_count": sum(
            bool(item["final_lora_hash_differs_from_initial"])
            for item in diagnostics
        ),
        "nonzero_embedding_movement_count": sum(
            bool(item["nonzero_embedding_movement"]) for item in diagnostics
        ),
        "candidate_vs_c1_discordant_prediction_rows": sum(
            int(row["control_topk"][0]) != int(row["candidate_topk"][0])
            for row in records
        ),
    }
    engagement_passes = {
        "end_vicreg_loss_below_start_count": engagement_values[
            "end_vicreg_loss_below_start_count"
        ]
        == 15,
        "final_lora_hash_differs_from_initial_count": engagement_values[
            "final_lora_hash_differs_from_initial_count"
        ]
        == 15,
        "nonzero_embedding_movement_count": engagement_values[
            "nonzero_embedding_movement_count"
        ]
        == 15,
        "candidate_vs_c1_discordant_prediction_rows_gt": engagement_values[
            "candidate_vs_c1_discordant_prediction_rows"
        ]
        > 0,
    }
    b0_values = {
        "delta_top1": float(vs_b0["overall"]["delta_top1"]),
        "delta_top1_component_bootstrap_lower_95": float(
            vs_b0["bootstrap"]["delta_top1_95"][0]
        ),
        "positive_seed_count": int(vs_b0["positive_seed_count"]),
        "delta_macro_top1": float(vs_b0["overall"]["delta_macro_top1"]),
        "delta_top3": float(vs_b0["overall"]["delta_top3"]),
        "minimum_fold_delta_top1": float(vs_b0["minimum_fold_delta_top1"]),
    }
    b0_passes = {
        "delta_top1_gte": b0_values["delta_top1"] >= 0.01,
        "delta_top1_component_bootstrap_lower_95_gt": b0_values[
            "delta_top1_component_bootstrap_lower_95"
        ]
        > 0.0,
        "positive_seed_count_gte": b0_values["positive_seed_count"] >= 2,
        "delta_macro_top1_gte": b0_values["delta_macro_top1"] >= -0.005,
        "delta_top3_gte": b0_values["delta_top3"] >= 0.0,
        "minimum_fold_delta_top1_gte": b0_values["minimum_fold_delta_top1"]
        >= -0.05,
    }
    c1_values = {
        "delta_top1": float(vs_c1["overall"]["delta_top1"]),
        "delta_top1_component_bootstrap_lower_95": float(
            vs_c1["bootstrap"]["delta_top1_95"][0]
        ),
        "positive_seed_count": int(vs_c1["positive_seed_count"]),
        "delta_top3": float(vs_c1["overall"]["delta_top3"]),
        "delta_macro_top1": float(vs_c1["overall"]["delta_macro_top1"]),
    }
    c1_passes = {
        "delta_top1_gte": c1_values["delta_top1"] >= -0.005,
        "delta_top1_component_bootstrap_lower_95_gt": c1_values[
            "delta_top1_component_bootstrap_lower_95"
        ]
        > -0.01,
        "delta_top3_gte": c1_values["delta_top3"] >= -0.01,
        "delta_macro_top1_gte": c1_values["delta_macro_top1"] >= -0.01,
    }
    verdict, claim = v17.classify_verdict(
        integrity_pass=all(integrity_passes.values()),
        engagement_pass=all(engagement_passes.values()),
        b0_anchor_pass=all(b0_passes.values()),
        c1_noninferiority_pass=all(c1_passes.values()),
        c1_delta_top1=c1_values["delta_top1"],
        c1_bootstrap_lower=c1_values[
            "delta_top1_component_bootstrap_lower_95"
        ],
        c1_positive_seed_count=c1_values["positive_seed_count"],
    )
    minimum_rank_c1 = min(
        float(item["candidate_over_control_C1_effective_rank_ratio"])
        for item in diagnostics
    )
    return {
        "pass": verdict in {"supported_strong", "supported_reference"},
        "score": c1_values["delta_top1"],
        "hypothesis_supported": verdict
        in {"supported_strong", "supported_reference"},
        "decision": verdict,
        "claim": claim,
        "failure_interpretation": (
            "last-block LoRA-VICReg did not establish a better model; "
            "retire the instrument without v18.4"
            if verdict in {"neutral_preservation", "not_supported"}
            else None
        ),
        "promotion_eligible": False,
        "comparison_vs_persisted_B0": vs_b0,
        "comparison_vs_exact_C1": vs_c1,
        "integrity_gates": integrity_values,
        "integrity_gate_passes": integrity_passes,
        "engagement_gates": engagement_values,
        "engagement_gate_passes": engagement_passes,
        "b0_mission_anchor_gates": b0_values,
        "b0_mission_anchor_gate_passes": b0_passes,
        "c1_noninferiority_gates": c1_values,
        "c1_noninferiority_gate_passes": c1_passes,
        "support_diagnostics": {
            "three_bin_candidate_vs_c1": v17._metrics_by_group(
                records, group_key="support_bin_three"
            ),
            "binary_candidate_vs_c1": v17._metrics_by_group(
                records, group_key="support_bin_binary"
            ),
            "per_class_candidate_vs_c1": v17._per_class_metrics(records),
        },
        "embedding_movement_diagnostic": movement_summary(diagnostics),
        "rank_diagnostic": {
            "minimum_candidate_over_exact_C1_embedding_rank": minimum_rank_c1,
            "council_escalation_threshold": 0.8,
            "council_escalation_required": minimum_rank_c1 < 0.8,
            "hard_gate": False,
        },
        "final_test_read": False,
        "runtime_promotion": False,
    }


def expected_optimizer_manifest() -> dict[str, Any]:
    value = {
        "names": [
            f"blocks.11.{target}.{suffix}"
            for target in feasibility.EXPECTED_LORA_TARGETS
            for suffix in ("lora_a", "lora_b")
        ],
        "count": LORA_PARAMETER_COUNT,
        "optimizer_scope": "projection+expander+all eight LoRA matrices",
        "weight_decay": 1e-4,
        "gradient_clip_union": 5.0,
    }
    return {**value, "sha256": v9.sha256_json(value)}


def preflight_lora_boundaries(
    base_block: nn.Module,
    base_norm: nn.Module,
    *,
    device: torch.device,
) -> list[dict[str, Any]]:
    expected_manifest = expected_optimizer_manifest()
    expected_base_sha = parameter_mapping_sha256(
        frozen_backbone_parameters(base_block, base_norm)
    )
    audits: list[dict[str, Any]] = []
    for fold in EXPECTED_FOLDS:
        for seed in EXPECTED_SEEDS:
            configure_seed(seed)
            block = copy.deepcopy(base_block).to(device)
            norm = copy.deepcopy(base_norm).to(device)
            block.requires_grad_(False).eval()
            norm.requires_grad_(False).eval()
            cpu_before = torch.get_rng_state().clone()
            cuda_before = torch.cuda.get_rng_state(device).clone()
            adapters = inject_lora(block, fold=fold, seed=seed)
            rng_isolated = bool(
                torch.equal(cpu_before, torch.get_rng_state())
                and torch.equal(cuda_before, torch.cuda.get_rng_state(device))
            )
            manifest = assert_trainable_boundary(block, norm, adapters)
            base_sha = parameter_mapping_sha256(
                frozen_backbone_parameters(block, norm)
            )
            if not rng_isolated:
                raise ValueError(
                    f"preflight LoRA consumed global RNG fold={fold} seed={seed}"
                )
            if manifest != expected_manifest:
                raise ValueError(
                    f"preflight optimizer boundary changed fold={fold} seed={seed}"
                )
            if base_sha != expected_base_sha:
                raise ValueError(
                    f"preflight base backbone changed fold={fold} seed={seed}"
                )
            audits.append(
                {
                    "fold": fold,
                    "seed": seed,
                    "lora_rng_isolated_from_global_rng": True,
                    "optimizer_group_manifest_sha256": manifest["sha256"],
                    "base_backbone_sha256": base_sha,
                    "candidate_optimizer_steps": 0,
                    "oof_token_access_count": 0,
                }
            )
    if len(audits) != 15:
        raise AssertionError("preflight did not cover all fold-seed pairs")
    return audits


def command_validate_contract(args: argparse.Namespace) -> dict[str, Any]:
    contract = validate_contract(
        args.spec,
        args.evaluator,
        args.cache_schema,
        args.feasibility_audit,
        args.predecessor_audit,
        args.source_inventory,
    )
    result = {
        "pass": True,
        "spec_sha256": contract["spec_sha256"],
        "evaluator_sha256": contract["evaluator_sha256"],
        "cache_schema_sha256": contract["cache_schema_sha256"],
        "feasibility_audit_sha256": contract["feasibility_audit_sha256"],
        "predecessor_audit_sha256": contract["predecessor_audit_sha256"],
        "source_inventory_sha256": contract["source_inventory_sha256"],
        "candidate_feature_extraction_count": 0,
        "candidate_optimizer_steps": 0,
        "candidate_predictions": 0,
        "final_test_read": False,
        "runtime_write": False,
    }
    print(v9.canonical_json(result), end="")
    return result


def prepare_train_data(
    *,
    train_manifest: dict[str, Any],
    contract: dict[str, Any],
    token_cache_dir: Path,
) -> dict[int, dict[str, Any]]:
    schema_folds = contract["cache_schema"]["folds"]
    entries = {int(item["fold"]): item for item in train_manifest["entries"]}
    if set(entries) != set(EXPECTED_FOLDS):
        raise ValueError("train manifest fold set changed")
    result: dict[int, dict[str, Any]] = {}
    for fold in EXPECTED_FOLDS:
        expected_rows = int(schema_folds[str(fold)]["train_rows"])
        tokens, metadata, features = validate_split_artifacts(
            entries[fold],
            expected_split="train",
            expected_rows=expected_rows,
        )
        if any(int(value) == fold for value in features["fold"]):
            raise ValueError(f"outer-fold row present in train cache fold={fold}")
        if any(int(row["fold"]) == fold for row in metadata["rows"]):
            raise ValueError(f"outer-fold metadata present in train cache fold={fold}")
        if Path(entries[fold]["token_path"]) != fold_paths(
            token_cache_dir, fold
        )["train_tokens"]:
            raise ValueError(f"unexpected train token path fold={fold}")
        canonical_path = v9.cache_path(
            v9.DEFAULT_CACHE_DIR, fold, EXPECTED_VIEWS, None
        )
        canonical, validation = i5.load_validated_cache(
            canonical_path, fold=fold, views=EXPECTED_VIEWS
        )
        if (
            str(validation["cache_sha256"])
            != str(entries[fold]["canonical_cache_sha256"])
            or str(validation["cache_sha256"])
            != i5.EXPECTED_CACHE_SHA256[fold]
        ):
            raise ValueError(f"canonical cache hash changed fold={fold}")
        result[fold] = {
            "tokens": tokens,
            "metadata": metadata,
            "features": features,
            "canonical": canonical,
            "validation": validation,
            "entry": entries[fold],
        }
    return result


def assert_initial_oof_state(
    token_cache_dir: Path,
    final_manifest_path: Path,
    *,
    reuse_oof: bool,
) -> None:
    artifacts = [
        fold_paths(token_cache_dir, fold)[key]
        for fold in EXPECTED_FOLDS
        for key in ("oof_tokens", "oof_metadata", "oof_features")
    ]
    if reuse_oof:
        missing = [str(path) for path in artifacts if not path.is_file()]
        if missing or not final_manifest_path.is_file():
            raise FileNotFoundError(
                f"replay requires sealed OOF artifacts: {missing}"
            )
    else:
        existing = [str(path) for path in artifacts if path.exists()]
        if existing or final_manifest_path.exists():
            raise FileExistsError(
                f"canonical run requires physically absent OOF artifacts: {existing}"
            )


def command_run(args: argparse.Namespace) -> dict[str, Any]:
    contract = validate_contract(
        args.spec,
        args.evaluator,
        args.cache_schema,
        args.feasibility_audit,
        args.predecessor_audit,
        args.source_inventory,
    )
    source = load_source_rows(
        args.v9_summary, args.v9_predictions, args.v18_2_predictions
    )
    train_manifest = load_train_manifest(
        args.train_manifest, contract=contract
    )
    assert_initial_oof_state(
        args.token_cache_dir,
        args.final_token_manifest,
        reuse_oof=args.reuse_oof,
    )
    for filename in ("prediction_rows.jsonl", "summary.json", "evaluation.json", "audit.json"):
        if (args.output_dir / filename).exists():
            raise FileExistsError(f"refusing to overwrite run output: {filename}")
    for fold in EXPECTED_FOLDS:
        for seed in EXPECTED_SEEDS:
            artifact = (
                args.output_dir
                / "candidate_states"
                / f"fold-{fold:02d}-seed-{seed:02d}.bin"
            )
            if artifact.exists():
                raise FileExistsError(
                    f"refusing to overwrite candidate state: {artifact}"
                )
    device = v9.resolve_device(args.device)
    if device.type != "cuda":
        raise ValueError("v18.3 exact control and candidate require CUDA")
    before_runtime = runtime_hashes(
        args.runtime_projection, args.runtime_prototypes, args.runtime_config
    )
    train_data = prepare_train_data(
        train_manifest=train_manifest,
        contract=contract,
        token_cache_dir=args.token_cache_dir,
    )

    controls: dict[tuple[int, int], ControlRun] = {}
    for fold in EXPECTED_FOLDS:
        canonical = train_data[fold]["canonical"]
        oof_ids = [str(value) for value in canonical["oof"]["row_id"]]
        for seed in EXPECTED_SEEDS:
            source_rows = [
                source["v9_lookup"][(fold, seed, row_id)]
                for row_id in oof_ids
            ]
            source_diagnostic = source["persisted"]["diagnostics"][(fold, seed)]
            controls[(fold, seed)] = run_control_c1(
                train_data[fold]["features"],
                canonical["oof"],
                fold=fold,
                seed=seed,
                source_diagnostic=source_diagnostic,
                source_rows=source_rows,
                device=device,
            )

    configure_seed(20260803)
    backbone, _ = v9.load_backbone(
        "dinov2_vits14", device, v9.DEFAULT_BACKBONE_MANIFEST
    )
    backbone.requires_grad_(False).eval()
    base_block = backbone.blocks[11]
    base_norm = backbone.norm
    preflight_audit = preflight_lora_boundaries(
        base_block, base_norm, device=device
    )
    candidates: dict[tuple[int, int], CandidateRun] = {}
    oof_entries: list[dict[str, Any]] = []
    replay_manifest = (
        load_final_token_manifest(
            args.final_token_manifest,
            contract=contract,
            train_manifest_path=args.train_manifest,
        )
        if args.reuse_oof
        else None
    )
    replay_oof_entries = (
        {int(item["fold"]): item for item in replay_manifest["oof_entries"]}
        if replay_manifest is not None
        else {}
    )
    corpus_rows, _ = v9.load_corpus(
        v9.DEFAULT_MANIFEST, v9.DEFAULT_AUDIT, strict_counts=True
    )
    source_by_row = source_inventory_by_row(args.source_inventory)
    for fold in EXPECTED_FOLDS:
        for seed in EXPECTED_SEEDS:
            control = controls[(fold, seed)]
            candidates[(fold, seed)] = run_candidate_train(
                train_data[fold]["tokens"],
                train_data[fold]["metadata"],
                train_data[fold]["features"],
                base_block=base_block,
                base_norm=base_norm,
                fold=fold,
                seed=seed,
                expected_initial_projection_sha256=control.initial_state_sha256,
                expected_ssl_plan_sha256=control.ssl_plan_sha256,
                expected_episode_plan_sha256=control.episode_plan_sha256,
                device=device,
            )
        if args.reuse_oof:
            entry = replay_oof_entries.get(fold)
            if entry is None:
                raise ValueError(f"replay OOF manifest missing fold={fold}")
        else:
            canonical = train_data[fold]["canonical"]
            rows = canonical_rows(corpus_rows, canonical["oof"]["row_id"])
            entry = extract_split(
                fold=fold,
                split="oof",
                rows=rows,
                canonical_cache=canonical,
                source_by_row=source_by_row,
                model=backbone,
                device=device,
                paths=fold_paths(args.token_cache_dir, fold),
            )
        oof_entries.append(entry)

    if args.reuse_oof:
        final_manifest = replay_manifest
    else:
        final_manifest = write_final_token_manifest(
            args.final_token_manifest,
            contract=contract,
            train_manifest_path=args.train_manifest,
            train_manifest=train_manifest,
            oof_entries=oof_entries,
        )
    if final_manifest is None:
        raise AssertionError("final token manifest missing")
    if {int(item["fold"]) for item in final_manifest["oof_entries"]} != set(
        EXPECTED_FOLDS
    ):
        raise ValueError("final token manifest fold set changed")

    runner_sha256 = v9.sha256_file(Path(__file__).resolve())
    expected_optimizer = expected_optimizer_manifest()
    all_records: list[dict[str, Any]] = []
    all_diagnostics: list[dict[str, Any]] = []
    for fold in EXPECTED_FOLDS:
        expected_rows = int(
            contract["cache_schema"]["folds"][str(fold)]["oof_rows"]
        )
        entry = next(
            item for item in oof_entries if int(item["fold"]) == fold
        )
        oof_tokens, oof_metadata, oof_control_features = validate_split_artifacts(
            entry, expected_split="oof", expected_rows=expected_rows
        )
        control_train = train_data[fold]["features"]
        for seed in EXPECTED_SEEDS:
            control = controls[(fold, seed)]
            candidate = candidates[(fold, seed)]
            block, norm, adapters = load_lora_block(
                base_block,
                base_norm,
                fold=fold,
                seed=seed,
                lora_state=candidate.lora_state,
                device=device,
            )
            assert_trainable_boundary(block, norm, adapters)
            candidate_oof_base = materialize_base_features(
                oof_tokens,
                oof_metadata,
                block=block,
                norm=norm,
                device=device,
            )
            candidate_train = feature_cache_with_base(
                control_train, candidate.train_base_features
            )
            candidate_oof = feature_cache_with_base(
                oof_control_features, candidate_oof_base
            )
            control_model = control.model.to(device)
            control_topk, control_rank = v9.predict_arm(
                control_model, control_train, oof_control_features, device=device
            )
            control.model = control_model.to("cpu")
            expected_rows_source = [
                source["v9_lookup"][(fold, seed, str(row_id))]
                for row_id in oof_control_features["row_id"]
            ]
            expected_control_topk = [
                [int(value) for value in row["candidate_topk"]]
                for row in expected_rows_source
            ]
            if control_topk != expected_control_topk:
                raise ValueError(
                    f"shared-token C1 top-k mismatch fold={fold} seed={seed}"
                )
            if control_rank != control.rank:
                raise ValueError(
                    f"shared-token C1 rank mismatch fold={fold} seed={seed}"
                )
            candidate_model = candidate.projection.to(device)
            candidate_topk, candidate_rank = v9.predict_arm(
                candidate_model, candidate_train, candidate_oof, device=device
            )
            candidate.projection = candidate_model.to("cpu")
            movement = embedding_diagnostics(
                control,
                candidate,
                oof_control_features,
                candidate_oof,
                device=device,
            )
            if candidate_rank != movement["candidate_effective_rank"]:
                raise ValueError(
                    f"candidate rank replay mismatch fold={fold} seed={seed}"
                )
            artifact_path = (
                args.output_dir
                / "candidate_states"
                / f"fold-{fold:02d}-seed-{seed:02d}.bin"
            )
            artifact_sha256 = write_tensor_artifact(
                artifact_path,
                metadata={
                    "schema_version": (
                        "autoresearch-discriminative-readout-v18."
                        "lora-vicreg-candidate-state-v1"
                    ),
                    "fold": fold,
                    "seed": seed,
                    "runner_sha256": runner_sha256,
                    "spec_sha256": contract["spec_sha256"],
                    "evaluator_sha256": contract["evaluator_sha256"],
                },
                tensors=candidate_state_tensors(candidate),
            )
            state_values = list(candidate_state_tensors(candidate).values())
            finite = all(bool(torch.isfinite(value).all()) for value in state_values)
            finite = finite and all(
                math.isfinite(float(value))
                for value in candidate.vicreg["loss_trajectory"]
            )
            records = build_prediction_records(
                fold=fold,
                seed=seed,
                oof_features=oof_control_features,
                source=source,
                control_topk=control_topk,
                candidate_topk=candidate_topk,
                runner_sha256=runner_sha256,
                contract=contract,
            )
            all_records.extend(records)
            source_diagnostic = source["persisted"]["diagnostics"][(fold, seed)]
            all_diagnostics.append(
                {
                    "fold": fold,
                    "seed": seed,
                    "device": str(device),
                    "control_initial_state_sha256": control.initial_state_sha256,
                    "control_initial_state_matches": True,
                    "control_final_state_sha256": control.final_state_sha256,
                    "control_final_state_matches": True,
                    "control_ssl_plan_sha256": control.ssl_plan_sha256,
                    "control_ssl_plan_matches": True,
                    "control_episode_plan_sha256": control.episode_plan_sha256,
                    "control_episode_plan_matches": True,
                    "control_topk_match_count": len(records),
                    "control_topk_matches_persisted_C1": True,
                    "control_C1_effective_rank": control_rank,
                    "persisted_C1_effective_rank": float(
                        source_diagnostic["candidate_effective_rank"]
                    ),
                    "persisted_v9_B0_effective_rank": float(
                        source_diagnostic["baseline_effective_rank"]
                    ),
                    "candidate_optimizer_step_count": int(
                        candidate.vicreg["optimizer_step_count"]
                    ),
                    "candidate_vicreg": candidate.vicreg,
                    "candidate_supervised": candidate.supervised,
                    "end_vicreg_loss_below_start": bool(
                        candidate.vicreg["end_loss_below_start_loss"]
                    ),
                    "initial_lora_sha256": candidate.initial_lora_sha256,
                    "final_lora_sha256": candidate.final_lora_sha256,
                    "final_lora_hash_differs_from_initial": (
                        candidate.final_lora_sha256
                        != candidate.initial_lora_sha256
                    ),
                    "lora_rng_isolated_from_global_rng": candidate.lora_rng_isolated,
                    "lora_trainable_name_count": len(
                        candidate.optimizer_manifest["names"]
                    ),
                    "lora_trainable_parameter_count": int(
                        candidate.optimizer_manifest["count"]
                    ),
                    "base_backbone_trainable_parameter_count": 0,
                    "batchnorm_or_syncbatchnorm_module_count": 0,
                    "optimizer_group_manifest": candidate.optimizer_manifest,
                    "optimizer_group_manifest_matches": (
                        candidate.optimizer_manifest == expected_optimizer
                    ),
                    "base_backbone_hash_before": candidate.base_hash_before,
                    "base_backbone_hash_after": candidate.base_hash_after,
                    "base_backbone_hash_unchanged": (
                        candidate.base_hash_before == candidate.base_hash_after
                    ),
                    "candidate_state_artifact": (
                        f"candidate_states/{artifact_path.name}"
                    ),
                    "candidate_state_artifact_sha256": artifact_sha256,
                    "candidate_state_artifact_exists": artifact_path.is_file(),
                    "candidate_over_persisted_B0_effective_rank_ratio": (
                        candidate_rank
                        / float(source_diagnostic["baseline_effective_rank"])
                    ),
                    **movement,
                    "finite_losses_logits_gradients_weights_and_embeddings": (
                        finite and movement["finite_embeddings"]
                    ),
                    "oof_token_access_during_training_count": 0,
                    "final_test_read": False,
                }
            )

    after_runtime = runtime_hashes(
        args.runtime_projection, args.runtime_prototypes, args.runtime_config
    )
    runtime_unchanged = before_runtime == after_runtime
    if not runtime_unchanged:
        raise RuntimeError("runtime artifacts changed during v18.3")
    fold_assignments = sorted(
        {
            (
                str(row["row_id"]),
                int(row["outer_fold"]),
                str(row["provenance_component"]),
                str(row["decoded_pixel_sha256"]),
            )
            for row in all_records
        }
    )
    folds_sha256 = v9.sha256_json(fold_assignments)
    for row in all_records:
        row["folds_sha256"] = folds_sha256
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "prediction_rows.jsonl"
    v9.write_jsonl(predictions_path, all_records)
    result = aggregate_results(
        all_records,
        all_diagnostics,
        train_manifest["entries"],
        oof_entries,
        runtime_unchanged=runtime_unchanged,
    )
    result.update(
        {
            "schema_version": (
                "autoresearch-discriminative-readout-v18."
                "lora-vicreg-evaluation-v1"
            ),
            "iteration": 3,
            "name": "last-block-rank8-lora-vicreg-participation",
            "paired_predictions_path": predictions_path.name,
            "candidate_state_artifact_directory": "candidate_states",
            "folds": list(EXPECTED_FOLDS),
            "seeds": list(EXPECTED_SEEDS),
            "folds_sha256": folds_sha256,
            "canonical_cache_sha256": {
                str(fold): str(train_data[fold]["validation"]["cache_sha256"])
                for fold in EXPECTED_FOLDS
            },
            "runner_sha256": runner_sha256,
            "spec_sha256": contract["spec_sha256"],
            "evaluator_sha256": contract["evaluator_sha256"],
            "cache_schema_sha256": contract["cache_schema_sha256"],
            "feasibility_audit_sha256": contract["feasibility_audit_sha256"],
            "predecessor_audit_sha256": contract["predecessor_audit_sha256"],
            "source_inventory_sha256": contract["source_inventory_sha256"],
            "train_token_manifest_sha256": v9.sha256_file(args.train_manifest),
            "final_token_manifest_sha256": v9.sha256_file(
                args.final_token_manifest
            ),
            "runtime_checkpoint_sha256": before_runtime,
            "runtime_unchanged": runtime_unchanged,
            "replay_normalization_sha256": replay_normalization_sha256(),
            "pretraining_abort_gate_audit": preflight_audit,
            "train_cache_entries": train_manifest["entries"],
            "oof_cache_entries": oof_entries,
            "diagnostics": all_diagnostics,
        }
    )
    summary_path = args.output_dir / "summary.json"
    evaluation_path = args.output_dir / "evaluation.json"
    v9.write_json(summary_path, result)
    v9.write_json(evaluation_path, result)
    audit = {
        "schema_version": (
            "autoresearch-discriminative-readout-v18.lora-vicreg-audit-v1"
        ),
        "iteration": 3,
        "decision": result["decision"],
        "claim": result["claim"],
        "execution_integrity_pass": all(
            result["integrity_gate_passes"].values()
        ),
        "engagement_pass": all(result["engagement_gate_passes"].values()),
        "summary_sha256": v9.sha256_file(summary_path),
        "evaluation_sha256": v9.sha256_file(evaluation_path),
        "prediction_rows_sha256": v9.sha256_file(predictions_path),
        "candidate_state_artifact_sha256": {
            str(item["candidate_state_artifact"]): str(
                item["candidate_state_artifact_sha256"]
            )
            for item in all_diagnostics
        },
        "train_token_manifest_sha256": v9.sha256_file(args.train_manifest),
        "final_token_manifest_sha256": v9.sha256_file(
            args.final_token_manifest
        ),
        "replay_normalization_sha256": replay_normalization_sha256(),
        "runtime_unchanged": runtime_unchanged,
        "final_test_read": False,
        "runtime_promotion": False,
    }
    v9.write_json(args.output_dir / "audit.json", audit)
    print(
        v9.canonical_json(
            {
                "decision": result["decision"],
                "claim": result["claim"],
                "score": result["score"],
                "integrity_pass": audit["execution_integrity_pass"],
                "engagement_pass": audit["engagement_pass"],
                "summary_sha256": audit["summary_sha256"],
                "prediction_rows_sha256": audit["prediction_rows_sha256"],
            }
        ),
        end="",
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("validate-contract", "extract-train", "run")
    )
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--evaluator", type=Path, default=DEFAULT_EVALUATOR)
    parser.add_argument("--cache-schema", type=Path, default=DEFAULT_CACHE_SCHEMA)
    parser.add_argument(
        "--feasibility-audit", type=Path, default=DEFAULT_FEASIBILITY_AUDIT
    )
    parser.add_argument(
        "--predecessor-audit", type=Path, default=DEFAULT_PREDECESSOR_AUDIT
    )
    parser.add_argument(
        "--source-inventory", type=Path, default=DEFAULT_SOURCE_INVENTORY
    )
    parser.add_argument(
        "--token-cache-dir", type=Path, default=DEFAULT_TOKEN_CACHE_DIR
    )
    parser.add_argument(
        "--train-manifest",
        type=Path,
        default=DEFAULT_TOKEN_CACHE_DIR / "train-token-cache-manifest.json",
    )
    parser.add_argument(
        "--final-token-manifest",
        type=Path,
        default=DEFAULT_TOKEN_CACHE_DIR / "token-cache-manifest.json",
    )
    parser.add_argument("--v9-summary", type=Path, default=DEFAULT_V9_SUMMARY)
    parser.add_argument(
        "--v9-predictions", type=Path, default=DEFAULT_V9_PREDICTIONS
    )
    parser.add_argument(
        "--v18-2-predictions", type=Path, default=DEFAULT_V18_2_PREDICTIONS
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--runtime-projection", type=Path, default=DEFAULT_RUNTIME_PROJECTION
    )
    parser.add_argument(
        "--runtime-prototypes", type=Path, default=DEFAULT_RUNTIME_PROTOTYPES
    )
    parser.add_argument(
        "--runtime-config", type=Path, default=DEFAULT_RUNTIME_CONFIG
    )
    parser.add_argument(
        "--reuse-oof",
        action="store_true",
        help="Replay only: reuse sealed token caches and forbid re-extraction.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "validate-contract":
        command_validate_contract(args)
    elif args.command == "extract-train":
        if args.reuse_oof:
            raise ValueError("--reuse-oof is invalid for train extraction")
        command_extract_train(args)
    else:
        command_run(args)


if __name__ == "__main__":
    main()

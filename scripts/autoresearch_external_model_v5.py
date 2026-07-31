#!/usr/bin/env python3
"""Leakage-safe prototype-only autoresearch for the CODEx Elements runtime.

The historical projection is immutable. Ten deterministic variants optimize
only the 286 runtime prototypes on safe legacy features plus external features
whose normalized source groups do not intersect development or sealed test.
Model selection reads development only. The preserved v4 sealed test is read
once, and only when a development leader exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"
sys.path.insert(0, str(BACKEND_ROOT))

from codex_pipeline.determinism import configure_determinism
from codex_pipeline.models.projection_head import ProjectionHead

CLASS_COUNT = 286
FEATURE_DIM = 384
EMBEDDING_DIM = 128
DEFAULT_EXTERNAL_CACHE = (
    BACKEND_ROOT
    / "model_registry/versions/20260711T220000Z-external286-weak-v3"
    / "training_data/precomputed/features.pt"
)
DEFAULT_LEGACY_CACHE = Path(
    "/tmp/baseline-replacement-e2e/perclass-split/features_train_augmented_safe.pt"
)
DEFAULT_IMPORT_SNAPSHOT = (
    BACKEND_ROOT
    / "training_corpus/external/20260711-approved-external-corpus-v1"
    / "import_snapshot.json"
)
DEFAULT_V4_STRICT_RESULTS = (
    REPO_ROOT / "reports/model-comparison-strict-unseen-20260729/element-results.json"
)
DEFAULT_HISTORICAL_WEIGHTS = BACKEND_ROOT / "codex_model/weights"
DEFAULT_RUNTIME_CONFIG = BACKEND_ROOT / "codex_model/config.json"
DEFAULT_RUN_DIR = (
    REPO_ROOT
    / ".omc/autoresearch/elements-baseline-replacement/runs"
    / "20260729-external-proxy-v5"
)
SNAPSHOT_SCHEMA = "external-corpus-training-snapshot.v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)


def resolved(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve(strict=False))


def normalized_source_group(path: str | Path) -> str:
    """Return case-insensitive ``parent/stem`` without a terminal ``-NN``.

    ``NN`` means a terminal numeric run of any length. Both Windows and POSIX
    separators are canonicalized so snapshot provenance stays portable.
    """

    canonical = str(path).replace("\\", "/")
    source = PurePosixPath(canonical)
    stem = re.sub(r"-\d+$", "", source.stem)
    return f"{source.parent.as_posix().rstrip('/')}/{stem}".casefold()


def load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def class_names(cache: dict[str, Any]) -> dict[int, str]:
    names = cache.get("class_names")
    if not isinstance(names, dict):
        raise ValueError("feature cache must expose class_names")
    return {int(key): str(value) for key, value in names.items()}


def load_cache(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {"features", "labels", "image_paths", "class_names"}
    if not isinstance(payload, dict) or not required <= payload.keys():
        raise ValueError(f"incomplete feature cache: {path}")
    count = len(payload["labels"])
    if len(payload["features"]) != count or len(payload["image_paths"]) != count:
        raise ValueError(f"misaligned feature cache: {path}")
    if payload["features"].ndim != 2 or payload["features"].shape[1] != FEATURE_DIM:
        raise ValueError(f"unexpected feature shape in cache: {path}")
    return payload


@dataclass(frozen=True)
class Runtime:
    weights_dir: Path
    model: ProjectionHead
    prototypes: torch.Tensor
    class_labels: torch.Tensor
    class_names: dict[int, str]
    ordered_names: list[str]
    prototype_payload: dict[str, Any]


def load_runtime(
    weights_dir: Path,
    runtime_config_path: Path,
    device: torch.device,
) -> Runtime:
    payload = torch.load(
        weights_dir / "prototypes.pt", map_location="cpu", weights_only=False
    )
    if not isinstance(payload, dict):
        raise ValueError("runtime prototypes payload must be a mapping")
    prototypes = payload.get("prototypes")
    raw_names = payload.get("class_names")
    class_labels = payload.get("class_labels")
    if not isinstance(prototypes, torch.Tensor) or prototypes.shape != (
        CLASS_COUNT,
        EMBEDDING_DIM,
    ):
        raise ValueError("runtime prototypes must have shape (286, 128)")
    if not isinstance(raw_names, dict) or len(raw_names) != CLASS_COUNT:
        raise ValueError("runtime class_names must contain 286 entries")
    names = {int(key): str(value) for key, value in raw_names.items()}
    sorted_labels = sorted(names)
    expected_labels = torch.tensor(sorted_labels, dtype=torch.long)
    if not isinstance(class_labels, torch.Tensor):
        raise ValueError("runtime prototypes must expose class_labels")
    if class_labels.dtype != torch.long or not torch.equal(
        class_labels.cpu(), expected_labels
    ):
        raise ValueError("runtime class_labels do not match sorted class_names keys")
    ordered_names = [names[label] for label in sorted_labels]
    if len(set(ordered_names)) != CLASS_COUNT:
        raise ValueError("runtime taxonomy contains duplicate class names")

    config = load_json_object(runtime_config_path)
    configured_names = config.get("class_names")
    if (
        config.get("num_classes") != CLASS_COUNT
        or not isinstance(configured_names, list)
        or configured_names != ordered_names
    ):
        raise ValueError("runtime config taxonomy differs from prototype taxonomy")

    model = ProjectionHead(FEATURE_DIM, EMBEDDING_DIM).to(device)
    model.load_state_dict(
        torch.load(
            weights_dir / "projection.pt", map_location=device, weights_only=False
        )
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return Runtime(
        weights_dir=weights_dir,
        model=model,
        prototypes=F.normalize(prototypes.float(), dim=1).to(device),
        class_labels=class_labels.cpu(),
        class_names=names,
        ordered_names=ordered_names,
        prototype_payload=payload,
    )


def validate_cache_taxonomy(
    cache: dict[str, Any],
    expected_names: list[str],
    *,
    cache_name: str,
) -> None:
    names = class_names(cache)
    expected_mapping = {index: name for index, name in enumerate(expected_names)}
    if names != expected_mapping:
        raise ValueError(f"{cache_name} taxonomy differs from runtime taxonomy")
    labels = cache["labels"].long()
    if labels.ndim != 1 or bool(((labels < 0) | (labels >= CLASS_COUNT)).any()):
        raise ValueError(f"{cache_name} labels are outside dense runtime positions")
    if int(labels.unique().numel()) != CLASS_COUNT:
        raise ValueError(f"{cache_name} does not retain all runtime classes")


def load_snapshot(path: Path, expected_names: list[str]) -> dict[str, Any]:
    snapshot = load_json_object(path)
    if snapshot.get("schema_version") != SNAPSHOT_SCHEMA:
        raise ValueError(f"unexpected import snapshot schema: {path}")
    if snapshot.get("ready_for_training") is not True:
        raise ValueError("import snapshot is not ready_for_training")
    if snapshot.get("class_count") != CLASS_COUNT:
        raise ValueError("import snapshot class_count differs from runtime")
    if snapshot.get("taxonomy") != expected_names:
        raise ValueError("import snapshot taxonomy differs from runtime")
    rows = snapshot.get("rows")
    if not isinstance(rows, list) or len(rows) != snapshot.get("image_count"):
        raise ValueError("import snapshot rows/image_count are inconsistent")
    return snapshot


def provenance_for_cache(
    external: dict[str, Any],
    snapshot: dict[str, Any],
) -> list[dict[str, Any]]:
    by_output: dict[str, dict[str, Any]] = {}
    for row in snapshot["rows"]:
        if not isinstance(row, dict):
            raise ValueError("import snapshot row must be a mapping")
        output_path = row.get("output_path")
        source_path = row.get("source_path")
        if not isinstance(output_path, str) or not isinstance(source_path, str):
            raise ValueError("snapshot row lacks output_path/source_path provenance")
        key = resolved(output_path)
        if key in by_output:
            raise ValueError(f"duplicate snapshot output_path: {output_path}")
        by_output[key] = row

    if len(by_output) != len(external["image_paths"]):
        raise ValueError("snapshot and external cache image counts differ")
    dense_names = class_names(external)
    provenance: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, image_path in enumerate(external["image_paths"]):
        key = resolved(image_path)
        row = by_output.get(key)
        if row is None:
            raise ValueError(f"external cache path absent from snapshot: {image_path}")
        if key in seen:
            raise ValueError(f"duplicate external cache path: {image_path}")
        seen.add(key)
        dense_label = int(external["labels"][index])
        if row.get("class_label") != dense_label:
            raise ValueError(f"snapshot/cache class_label mismatch: {image_path}")
        if row.get("class_name") != dense_names[dense_label]:
            raise ValueError(f"snapshot/cache class_name mismatch: {image_path}")
        provenance.append(
            {
                "cache_index": index,
                "output_path": key,
                "source_path": str(row["source_path"]),
                "source_group": normalized_source_group(str(row["source_path"])),
                "class_label": dense_label,
                "class_name": dense_names[dense_label],
            }
        )
    if seen != set(by_output):
        raise ValueError("snapshot contains outputs absent from external cache")
    return provenance


def v4_indices(
    external: dict[str, Any],
    provenance: list[dict[str, Any]],
    strict_results_path: Path,
) -> tuple[list[int], list[int], list[int]]:
    """Reproduce the original v4 299/270 split without reading predictions."""

    strict_value = json.loads(strict_results_path.read_text(encoding="utf-8"))
    if not isinstance(strict_value, list) or len(strict_value) != 569:
        raise ValueError("v4 strict results must contain the audited 569 rows")
    external_by_path = {
        resolved(path): index for index, path in enumerate(external["image_paths"])
    }
    strict_indices: list[int] = []
    for row in strict_value:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            raise ValueError("v4 strict row lacks path")
        index = external_by_path.get(resolved(row["path"]))
        if index is None:
            raise ValueError(f"v4 strict path absent from cache: {row['path']}")
        strict_indices.append(index)
    if len(set(strict_indices)) != len(strict_indices):
        raise ValueError("v4 strict results contain duplicate cache indices")

    by_class_source: dict[int, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for index in strict_indices:
        label = int(external["labels"][index])
        raw_source = provenance[index]["source_path"].casefold()
        by_class_source[label][raw_source].append(index)

    dev: list[int] = []
    test: list[int] = []
    for label, grouped in sorted(by_class_source.items()):
        groups = sorted(
            grouped, key=lambda value: stable_hash(f"{label}:{value}")
        )
        for position, group in enumerate(groups):
            if len(groups) == 1:
                use_dev = stable_hash(f"singleton:{label}:{group}") % 2 == 0
            else:
                use_dev = position % 2 == 0
            (dev if use_dev else test).extend(grouped[group])
    if len(dev) != 299 or len(test) != 270:
        raise ValueError(
            f"failed to reproduce v4 split: dev={len(dev)}, test={len(test)}"
        )
    return sorted(dev), sorted(test), sorted(strict_indices)


def leakage_safe_split(
    external: dict[str, Any],
    provenance: list[dict[str, Any]],
    strict_results_path: Path,
) -> dict[str, Any]:
    old_dev, old_test, strict = v4_indices(
        external, provenance, strict_results_path
    )
    group_by_index = [row["source_group"] for row in provenance]
    test = list(old_test)
    test_groups = {group_by_index[index] for index in test}
    dev = [
        index for index in old_dev if group_by_index[index] not in test_groups
    ]
    dev_groups = {group_by_index[index] for index in dev}
    strict_groups = dev_groups | test_groups
    external_train = [
        index
        for index, group in enumerate(group_by_index)
        if group not in strict_groups
    ]
    train_groups = {group_by_index[index] for index in external_train}

    overlap = {
        "train_dev_group_count": len(train_groups & dev_groups),
        "train_test_group_count": len(train_groups & test_groups),
        "dev_test_group_count": len(dev_groups & test_groups),
        "train_dev_index_count": len(set(external_train) & set(dev)),
        "train_test_index_count": len(set(external_train) & set(test)),
        "dev_test_index_count": len(set(dev) & set(test)),
        "test_missing_from_v4_count": len(set(old_test) - set(test)),
        "test_added_beyond_v4_count": len(set(test) - set(old_test)),
    }
    nonzero = {key: value for key, value in overlap.items() if value != 0}
    if nonzero:
        raise ValueError(f"source-disjoint split failed: {nonzero}")
    if test != old_test:
        raise ValueError("v5 must preserve the complete ordered v4 sealed test")
    if any(group_by_index[index] in strict_groups for index in external_train):
        raise ValueError("external train still contains a strict source group")

    strict_set = set(strict)
    purged_non_strict_count = sum(
        index not in strict_set and group_by_index[index] in strict_groups
        for index in range(len(provenance))
    )
    audit = {
        "schema_version": "autoresearch-external-split-audit.v5",
        "group_rule": "casefold(parent + '/' + stem_without_terminal_-digits)",
        "provenance_source": "import_snapshot.json:rows[].source_path",
        "v4_dev_count": len(old_dev),
        "v4_test_count": len(old_test),
        "v5_dev_count": len(dev),
        "v5_test_count": len(test),
        "v5_external_train_count": len(external_train),
        "removed_v4_dev_count": len(old_dev) - len(dev),
        "purged_non_strict_external_count": purged_non_strict_count,
        "dev_group_count": len(dev_groups),
        "test_group_count": len(test_groups),
        "external_train_group_count": len(train_groups),
        "v4_test_preserved_exactly": True,
        "overlap": overlap,
        "overlap_total": sum(overlap.values()),
    }
    return {
        "external_train": external_train,
        "dev": dev,
        "test": test,
        "v4_dev": old_dev,
        "v4_test": old_test,
        "strict": strict,
        "audit": audit,
    }


@torch.inference_mode()
def project(
    model: ProjectionHead,
    features: torch.Tensor,
    device: torch.device,
    batch_size: int = 2048,
) -> torch.Tensor:
    model.eval()
    return torch.cat(
        [
            model(batch.float().to(device)).cpu()
            for batch in features.split(batch_size)
        ]
    )


def class_balanced_weights(labels: torch.Tensor) -> torch.Tensor:
    counts = torch.bincount(labels.long(), minlength=CLASS_COUNT).float()
    if bool((counts == 0).any()):
        raise ValueError("prototype training requires every runtime class")
    weights = counts.sum() / (CLASS_COUNT * counts)
    return weights / weights.mean()


def train_prototypes(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    historical_prototypes: torch.Tensor,
    spec: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    """Optimize proxies only; the historical projection is never trainable."""

    configure_determinism(int(spec["seed"]))
    proxies = torch.nn.Parameter(
        historical_prototypes.detach().clone().float().to(device)
    )
    optimizer = torch.optim.AdamW(
        [proxies],
        lr=float(spec["lr"]),
        weight_decay=float(spec["weight_decay"]),
    )
    epochs = int(spec["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )
    generator = torch.Generator().manual_seed(int(spec["seed"]))
    weights = class_balanced_weights(labels).to(device)
    anchor = F.normalize(historical_prototypes.detach().float().to(device), dim=1)
    batch_size = int(spec["batch_size"])
    temperature = float(spec["temperature"])
    anchor_weight = float(spec["anchor_weight"])

    for _ in range(epochs):
        permutation = torch.randperm(len(labels), generator=generator)
        for indices in permutation.split(batch_size):
            batch = embeddings[indices].float().to(device)
            target = labels[indices].long().to(device)
            normalized = F.normalize(proxies, dim=1)
            logits = batch @ normalized.T / temperature
            ce = F.cross_entropy(
                logits,
                target,
                weight=weights,
                label_smoothing=float(spec["label_smoothing"]),
            )
            anchor_loss = 1 - F.cosine_similarity(
                normalized, anchor, dim=1
            ).mean()
            loss = ce + anchor_weight * anchor_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([proxies], 5.0)
            optimizer.step()
        scheduler.step()
    return F.normalize(proxies.detach(), dim=1).cpu()


@torch.inference_mode()
def metrics(
    prototypes: torch.Tensor,
    embeddings: torch.Tensor,
    truth: torch.Tensor,
) -> tuple[dict[str, float | int], torch.Tensor]:
    scores = embeddings.float() @ F.normalize(prototypes.float(), dim=1).T
    top3 = scores.topk(3, dim=1).indices
    predictions = top3[:, 0]
    per_class: list[float] = []
    for label in truth.unique(sorted=True):
        mask = truth == label
        per_class.append(
            float((predictions[mask] == truth[mask]).float().mean())
        )
    result: dict[str, float | int] = {
        "top1": float((predictions == truth).float().mean()),
        "macro_top1": sum(per_class) / len(per_class),
        "top3": float((top3 == truth[:, None]).any(1).float().mean()),
        "mean_confidence": float(scores.max(1).values.mean()),
        "count": int(len(truth)),
        "class_count": int(truth.unique().numel()),
    }
    return result, predictions


def exact_mcnemar(
    candidate_correct: torch.Tensor,
    historical_correct: torch.Tensor,
) -> dict[str, int | float]:
    candidate_only = int((candidate_correct & ~historical_correct).sum())
    historical_only = int((historical_correct & ~candidate_correct).sum())
    discordant = candidate_only + historical_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail_end = min(candidate_only, historical_only)
        tail = sum(
            math.comb(discordant, k) for k in range(tail_end + 1)
        ) / (2**discordant)
        p_value = min(1.0, 2 * tail)
    return {
        "candidate_only_correct": candidate_only,
        "historical_only_correct": historical_only,
        "discordant": discordant,
        "exact_two_sided_p": p_value,
    }


def experiment_specs() -> list[dict[str, Any]]:
    common = {
        "objective": "prototype_only_class_balanced_ce_cosine_anchor",
        "projection": "historical_frozen",
        "prototype_initialization": "historical_runtime",
        "batch_size": 2048,
        "temperature": 0.08,
        "weight_decay": 1e-6,
        "label_smoothing": 0.01,
    }
    grid = [
        ("proxy-lr1000-e100-a100-s1", 1.00e-3, 100, 1.00, 6101),
        ("proxy-lr0750-e100-a100-s2", 0.75e-3, 100, 1.00, 6102),
        ("proxy-lr1250-e100-a100-s3", 1.25e-3, 100, 1.00, 6103),
        ("proxy-lr1000-e075-a100-s4", 1.00e-3, 75, 1.00, 6104),
        ("proxy-lr1000-e125-a100-s5", 1.00e-3, 125, 1.00, 6105),
        ("proxy-lr1000-e100-a050-s6", 1.00e-3, 100, 0.50, 6106),
        ("proxy-lr1000-e100-a200-s7", 1.00e-3, 100, 2.00, 6107),
        ("proxy-lr0750-e125-a075-s8", 0.75e-3, 125, 0.75, 6108),
        ("proxy-lr1250-e075-a150-s9", 1.25e-3, 75, 1.50, 6109),
        ("proxy-lr1000-e100-a100-s10", 1.00e-3, 100, 1.00, 6110),
    ]
    return [
        {
            **common,
            "iteration": iteration,
            "name": name,
            "lr": lr,
            "epochs": epochs,
            "anchor_weight": anchor_weight,
            "seed": seed,
        }
        for iteration, (name, lr, epochs, anchor_weight, seed) in enumerate(
            grid, start=1
        )
    ]


@dataclass
class SealedTest:
    external: dict[str, Any]
    indices: list[int]
    truth: torch.Tensor
    projection: ProjectionHead
    historical_prototypes: torch.Tensor
    device: torch.device
    read_count: int = 0

    def evaluate(
        self, candidate_prototypes: torch.Tensor
    ) -> tuple[
        dict[str, float | int],
        torch.Tensor,
        dict[str, float | int],
        torch.Tensor,
    ]:
        if self.read_count != 0:
            raise RuntimeError("sealed test may be read exactly once")
        self.read_count += 1
        index = torch.tensor(self.indices, dtype=torch.long)
        embeddings = project(
            self.projection, self.external["features"][index], self.device
        )
        truth = self.truth[index]
        historical, historical_predictions = metrics(
            self.historical_prototypes.cpu(), embeddings, truth
        )
        candidate, candidate_predictions = metrics(
            candidate_prototypes, embeddings, truth
        )
        return (
            historical,
            historical_predictions,
            candidate,
            candidate_predictions,
        )


def export_runtime(
    run_dir: Path,
    historical: Runtime,
    runtime_config_path: Path,
    prototypes: torch.Tensor,
    selected: dict[str, Any],
    final_path: Path,
) -> dict[str, Any]:
    runtime_dir = run_dir / "candidate-runtime"
    weights_dir = runtime_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    projection_output = weights_dir / "projection.pt"
    prototypes_output = weights_dir / "prototypes.pt"
    shutil.copy2(
        historical.weights_dir / "projection.pt", projection_output
    )
    shutil.copy2(runtime_config_path, runtime_dir / "config.json")
    payload = dict(historical.prototype_payload)
    payload["prototypes"] = F.normalize(prototypes.float(), dim=1).cpu()
    payload["class_labels"] = historical.class_labels.clone()
    payload["class_names"] = dict(historical.class_names)
    payload["embedding_dim"] = EMBEDDING_DIM
    torch.save(payload, prototypes_output)

    checked = torch.load(
        prototypes_output, map_location="cpu", weights_only=False
    )
    if checked["prototypes"].shape != (CLASS_COUNT, EMBEDDING_DIM):
        raise ValueError("exported runtime has invalid prototype shape")
    if not torch.equal(checked["class_labels"], historical.class_labels):
        raise ValueError("export changed runtime class_labels")
    if checked["class_names"] != historical.class_names:
        raise ValueError("export changed runtime class_names")
    if sha256_file(projection_output) != sha256_file(
        historical.weights_dir / "projection.pt"
    ):
        raise ValueError("export changed the frozen historical projection")

    manifest = {
        "schema_version": "autoresearch-external-runtime-export.v5",
        "runtime_compatible": True,
        "run_local_only": True,
        "historical_projection_frozen": True,
        "historical_runtime_untouched": True,
        "selected_iteration": selected["iteration"],
        "selected_name": selected["name"],
        "final_evaluation": str(final_path.resolve()),
        "projection_sha256": sha256_file(projection_output),
        "prototypes_sha256": sha256_file(prototypes_output),
        "config_sha256": sha256_file(runtime_dir / "config.json"),
    }
    dump_json(runtime_dir / "export-manifest.json", manifest)
    return manifest


def completed_without_leader(
    run_dir: Path,
    state_path: Path,
    started_at: str,
    historical_dev: dict[str, float | int],
) -> int:
    final = {
        "schema_version": "autoresearch-external-final.v5",
        "pass": False,
        "score": 0.0,
        "reason": "no development variant passed the historical gate",
        "development": {"historical": historical_dev, "selected": None},
        "sealed_test": None,
        "sealed_test_read_count": 0,
        "iteration_count": len(experiment_specs()),
        "runtime_exported": False,
    }
    dump_json(run_dir / "final-evaluation.json", final)
    dump_json(
        state_path,
        {
            "status": "completed",
            "pass": False,
            "started_at": started_at,
            "completed_at": utc_now(),
            "iteration": len(experiment_specs()),
            "leader": None,
            "sealed_test_read_count": 0,
        },
    )
    print(json.dumps(final, indent=2))
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--external-cache", type=Path, default=DEFAULT_EXTERNAL_CACHE
    )
    parser.add_argument(
        "--legacy-cache", type=Path, default=DEFAULT_LEGACY_CACHE
    )
    parser.add_argument(
        "--import-snapshot", type=Path, default=DEFAULT_IMPORT_SNAPSHOT
    )
    parser.add_argument(
        "--v4-strict-results", type=Path, default=DEFAULT_V4_STRICT_RESULTS
    )
    parser.add_argument(
        "--historical-weights",
        type=Path,
        default=DEFAULT_HISTORICAL_WEIGHTS,
    )
    parser.add_argument(
        "--runtime-config", type=Path, default=DEFAULT_RUNTIME_CONFIG
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the requested device")
    device = torch.device(args.device)
    started_at = utc_now()
    deadline = time.time() + 6 * 60 * 60
    run_dir = args.run_dir.resolve()
    state_path = run_dir / "state.json"
    dump_json(
        state_path,
        {
            "status": "running",
            "run_id": run_dir.name,
            "mission": "elements-baseline-replacement-v5",
            "started_at": started_at,
            "updated_at": started_at,
            "iteration": 0,
            "sealed_test_read_count": 0,
            "max_runtime_seconds": 21600,
        },
    )

    historical = load_runtime(
        args.historical_weights, args.runtime_config, device
    )
    external = load_cache(args.external_cache)
    legacy = load_cache(args.legacy_cache)
    validate_cache_taxonomy(
        external, historical.ordered_names, cache_name="external cache"
    )
    validate_cache_taxonomy(
        legacy, historical.ordered_names, cache_name="safe legacy cache"
    )
    snapshot = load_snapshot(args.import_snapshot, historical.ordered_names)
    provenance = provenance_for_cache(external, snapshot)
    split = leakage_safe_split(
        external, provenance, args.v4_strict_results
    )
    audit = {
        **split["audit"],
        "external_cache_sha256": sha256_file(args.external_cache),
        "legacy_cache_sha256": sha256_file(args.legacy_cache),
        "import_snapshot_sha256": sha256_file(args.import_snapshot),
        "v4_strict_results_sha256": sha256_file(args.v4_strict_results),
        "historical_projection_sha256": sha256_file(
            args.historical_weights / "projection.pt"
        ),
        "historical_prototypes_sha256": sha256_file(
            args.historical_weights / "prototypes.pt"
        ),
    }
    dump_json(run_dir / "split-audit.json", audit)
    dump_json(
        run_dir / "split-indices.json",
        {
            "schema_version": "autoresearch-external-split-indices.v5",
            "external_train": split["external_train"],
            "development": split["dev"],
            "sealed_test_v4_preserved": split["test"],
            "removed_v4_development": sorted(
                set(split["v4_dev"]) - set(split["dev"])
            ),
        },
    )

    external_train_index = torch.tensor(
        split["external_train"], dtype=torch.long
    )
    dev_index = torch.tensor(split["dev"], dtype=torch.long)
    train_features = torch.cat(
        [legacy["features"], external["features"][external_train_index]]
    )
    train_labels = torch.cat(
        [legacy["labels"].long(), external["labels"][external_train_index].long()]
    )
    train_embeddings = project(
        historical.model, train_features, device
    )
    dev_embeddings = project(
        historical.model, external["features"][dev_index], device
    )
    dev_truth = external["labels"][dev_index].long()
    historical_dev, _ = metrics(
        historical.prototypes.cpu(), dev_embeddings, dev_truth
    )
    evaluator = {
        "schema_version": "autoresearch-external-evaluator.v5",
        "mission": "elements-baseline-replacement-v5",
        "primary_metric": "source_group_disjoint_external_dev_top1",
        "secondary_gate": "development macro_top1 >= historical",
        "required_iteration_output": {
            "pass": "boolean",
            "score": "number",
        },
        "historical_development": historical_dev,
        "selection": "development only; sealed test unavailable until a final leader exists",
        "final_gate": {
            "development_top1": "strictly greater than historical",
            "development_macro_top1": "greater than or equal to historical",
            "sealed_test_top1": "strictly greater than historical",
            "sealed_test_macro_top1": "greater than or equal to historical",
            "paired_report": "exact two-sided McNemar",
        },
        "projection": "historical frozen and byte-identical on export",
    }
    dump_json(run_dir / "evaluator.json", evaluator)

    incumbent: dict[str, float | int] | None = None
    leader: dict[str, Any] | None = None
    decision_lines = [
        "# External prototype-only autoresearch v5",
        "",
        f"Started: {started_at}",
        f"Historical DEV top-1: {float(historical_dev['top1']):.6f}",
        f"Historical DEV macro top-1: {float(historical_dev['macro_top1']):.6f}",
        "Sealed TEST read: no",
        "",
    ]
    specs = experiment_specs()
    for spec in specs:
        if time.time() >= deadline:
            raise TimeoutError("autoresearch v5 exceeded its six-hour deadline")
        iteration_started = time.perf_counter()
        prototypes = train_prototypes(
            train_embeddings,
            train_labels,
            historical.prototypes,
            spec,
            device,
        )
        dev_result, _ = metrics(prototypes, dev_embeddings, dev_truth)
        passes_historical_gate = bool(
            float(dev_result["top1"]) > float(historical_dev["top1"])
            and float(dev_result["macro_top1"])
            >= float(historical_dev["macro_top1"])
        )
        improves_incumbent = bool(
            incumbent is None
            or (
                float(dev_result["top1"]),
                float(dev_result["macro_top1"]),
            )
            > (
                float(incumbent["top1"]),
                float(incumbent["macro_top1"]),
            )
        )
        passed = passes_historical_gate and improves_incumbent
        checkpoint_path = (
            run_dir / "checkpoints" / f"iteration-{spec['iteration']:04d}.pt"
        )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "prototypes": prototypes,
                "class_labels": historical.class_labels,
                "class_names": historical.class_names,
                "projection_sha256": audit["historical_projection_sha256"],
                "spec": spec,
            },
            checkpoint_path,
        )
        evaluation = {
            "schema_version": "autoresearch-external-evaluation.v5",
            "iteration": spec["iteration"],
            "name": spec["name"],
            "pass": passed,
            "score": float(dev_result["top1"]),
            "development": dev_result,
            "historical_development": historical_dev,
            "passes_historical_gate": passes_historical_gate,
            "improves_incumbent": improves_incumbent,
            "runtime_compatible": True,
            "projection_frozen": True,
            "optimized_prototypes_retained": True,
            "external_train_count": len(split["external_train"]),
            "safe_legacy_train_count": len(legacy["labels"]),
            "source_group_overlap_total": audit["overlap_total"],
            "sealed_test_evaluated": False,
            "elapsed_seconds": time.perf_counter() - iteration_started,
            "spec": spec,
            "checkpoint_path": str(checkpoint_path),
        }
        dump_json(
            run_dir
            / "evaluations"
            / f"iteration-{spec['iteration']:04d}.json",
            evaluation,
        )
        dump_json(
            run_dir / "specs" / f"iteration-{spec['iteration']:04d}.json",
            spec,
        )
        decision_lines.extend(
            [
                f"## Iteration {spec['iteration']:04d} — {spec['name']}",
                "",
                f"- DEV top-1: {float(dev_result['top1']):.6f}",
                f"- DEV macro top-1: {float(dev_result['macro_top1']):.6f}",
                f"- Decision: {'KEEP' if passed else 'REJECT'}",
                "- Sealed TEST read: no",
                "",
            ]
        )
        if passed:
            incumbent = dev_result
            leader = {
                "spec": spec,
                "prototypes": prototypes,
                "development": dev_result,
                "checkpoint_path": checkpoint_path,
            }
        dump_json(
            state_path,
            {
                "status": "running",
                "run_id": run_dir.name,
                "mission": "elements-baseline-replacement-v5",
                "started_at": started_at,
                "updated_at": utc_now(),
                "iteration": spec["iteration"],
                "leader": leader["spec"]["name"] if leader else None,
                "sealed_test_read_count": 0,
                "max_runtime_seconds": 21600,
            },
        )

    if leader is None:
        (run_dir / "decision-log.md").write_text(
            "\n".join(decision_lines), encoding="utf-8"
        )
        return completed_without_leader(
            run_dir, state_path, started_at, historical_dev
        )

    sealed = SealedTest(
        external=external,
        indices=split["test"],
        truth=external["labels"].long(),
        projection=historical.model,
        historical_prototypes=historical.prototypes,
        device=device,
    )
    (
        historical_test,
        historical_test_predictions,
        leader_test,
        leader_test_predictions,
    ) = sealed.evaluate(leader["prototypes"])
    test_index = torch.tensor(split["test"], dtype=torch.long)
    test_truth = external["labels"][test_index].long()
    paired = exact_mcnemar(
        leader_test_predictions == test_truth,
        historical_test_predictions == test_truth,
    )
    final_pass = bool(
        float(leader["development"]["top1"])
        > float(historical_dev["top1"])
        and float(leader["development"]["macro_top1"])
        >= float(historical_dev["macro_top1"])
        and float(leader_test["top1"]) > float(historical_test["top1"])
        and float(leader_test["macro_top1"])
        >= float(historical_test["macro_top1"])
    )
    final_path = run_dir / "final-evaluation.json"
    final = {
        "schema_version": "autoresearch-external-final.v5",
        "pass": final_pass,
        "score": float(leader_test["top1"])
        - float(historical_test["top1"]),
        "selected": leader["spec"],
        "development": {
            "historical": historical_dev,
            "selected": leader["development"],
        },
        "sealed_test": {
            "historical": historical_test,
            "selected": leader_test,
            "paired_mcnemar": paired,
        },
        "sealed_test_read_count": sealed.read_count,
        "iteration_count": len(specs),
        "strict_holdout_used_for_training": False,
        "source_group_overlap_total": audit["overlap_total"],
        "historical_projection_frozen": True,
        "optimized_prototypes_retained": True,
        "runtime_exported": final_pass,
    }
    dump_json(final_path, final)
    if final_pass:
        export_runtime(
            run_dir,
            historical,
            args.runtime_config,
            leader["prototypes"],
            leader["spec"],
            final_path,
        )

    decision_lines.extend(
        [
            "## Final sealed TEST",
            "",
            f"- Selected: {leader['spec']['name']}",
            f"- Historical top-1: {float(historical_test['top1']):.6f}",
            f"- Selected top-1: {float(leader_test['top1']):.6f}",
            f"- Delta: {float(final['score']):+.6f}",
            f"- Exact McNemar p: {float(paired['exact_two_sided_p']):.6f}",
            f"- Final pass: {final_pass}",
            f"- Sealed TEST read count: {sealed.read_count}",
            "",
        ]
    )
    (run_dir / "decision-log.md").write_text(
        "\n".join(decision_lines), encoding="utf-8"
    )
    dump_json(
        state_path,
        {
            "status": "completed",
            "run_id": run_dir.name,
            "mission": "elements-baseline-replacement-v5",
            "started_at": started_at,
            "updated_at": utc_now(),
            "completed_at": utc_now(),
            "iteration": len(specs),
            "leader": leader["spec"]["name"],
            "pass": final_pass,
            "sealed_test_read_count": sealed.read_count,
            "max_runtime_seconds": 21600,
        },
    )
    print(json.dumps(final, indent=2))
    return 0 if final_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build the frozen full-data R2 candidate without runtime promotion.

This build-only runner performs two independent B/14 feature extractions,
checks semantic equality and cosine agreement with the frozen fold caches,
performs two deterministic R2 refits, and exports only an immutable candidate.
It never reads the final test, writes backend/codex_model, or promotes a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as functional

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from codex_pipeline.models.projection_head import ProjectionHead  # noqa: E402
from codex_pipeline.models.prototypical import compute_prototypes  # noqa: E402
from codex_pipeline.scripts.precompute_embeddings import load_backbone  # noqa: E402
from scripts import autoresearch_self_supervised_v9 as v9  # noqa: E402
from scripts import export_elements_refit as exporter  # noqa: E402

RUN_DIR = ROOT / ".omc/autoresearch/elements-baseline-replacement/runs/20260805-full-data-r2"
BUILD_SPEC_PATH = BACKEND / "model_registry/specs/r2-full-data-production-v1.json"
E2E_SPEC_PATH = BACKEND / "model_registry/specs/r2-e2e-promotion-v1.json"
DEFAULT_CACHE_DIR = RUN_DIR / "cache"
DEFAULT_RESULTS = RUN_DIR / "results"
DEFAULT_REGISTRY = BACKEND / "model_registry"
DEFAULT_FOLD_CACHE_DIR = (
    ROOT / ".omc/autoresearch/elements-baseline-replacement/runs/"
    "20260804-vicreg-backbone-renomination-v19-r1/iteration-0002/caches"
)
DEFAULT_BACKBONE_MANIFEST = (
    ROOT / ".omc/autoresearch/elements-baseline-replacement/runs/"
    "20260803-self-supervised-v10/inputs/dinov2-vitb14-local-manifest.json"
)
DEFAULT_RUNTIME_CONFIG = BACKEND / "codex_model/config.json"
DEFAULT_RUNTIME_PROJECTION = BACKEND / "codex_model/weights/projection.pt"
DEFAULT_RUNTIME_PROTOTYPES = BACKEND / "codex_model/weights/prototypes.pt"

EXPECTED_BUILD_SPEC_SHA256 = "dc67d68e7befc5eb8fea39253f5a7907ba3d4db8f2a34e7701dfc7808ba499ae"
EXPECTED_E2E_SPEC_SHA256 = "95fdeb7691f12edddf3969b54c3bdbccd93a893fd8e5a1f1126b91833b61dcd0"
DEFAULT_VERSION_ID = "20260805T112535Z-vicreg-full-data-r2"
FULL_DATA_SEED = 1706651702
BACKBONE = "dinov2_vitb14"
HIDDEN_DIM = 768
EMBEDDING_DIM = 128
IMAGE_SIZE = 224
VIEWS = 8
VIEW_SEED = 20260803
IMAGE_BATCH_SIZE = 4
NUM_WORKERS = 0
EXPECTED_ROWS = 9990
EXPECTED_CLASSES = 286
EXPECTED_COMPONENTS = 300
COSINE_MIN = 0.9999
COSINE_MEDIAN_MIN = 0.999999
METRIC_CONTEXT = "full_training_cache_fit_diagnostic_not_efficacy"
CACHE_SCHEMA = "autoresearch-full-data-v19r2.cache-v1"
CACHE_SIDECAR_SCHEMA = "autoresearch-full-data-v19r2.cache-provenance-v1"


def sha256_file(path: Path) -> str:
    return v9.sha256_file(path)


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def state_dict_sha256(state_dict: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state_dict.items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor_sha256(tensor).encode("ascii"))
    return digest.hexdigest()


def cache_sidecar_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".prov.json")


def cache_path(cache_dir: Path, replica: int) -> Path:
    if replica not in (1, 2):
        raise ValueError("cache replica must be 1 or 2")
    return cache_dir / f"full-data-extraction-{replica:02d}.pt"


def validate_contract(
    build_spec_path: Path = BUILD_SPEC_PATH,
    e2e_spec_path: Path = E2E_SPEC_PATH,
) -> dict[str, Any]:
    build_hash = sha256_file(build_spec_path)
    e2e_hash = sha256_file(e2e_spec_path)
    if build_hash != EXPECTED_BUILD_SPEC_SHA256:
        raise ValueError(f"R2 build spec hash mismatch: {build_hash}")
    if e2e_hash != EXPECTED_E2E_SPEC_SHA256:
        raise ValueError(f"R2 E2E spec hash mismatch: {e2e_hash}")
    build = json.loads(build_spec_path.read_text(encoding="utf-8"))
    e2e = json.loads(e2e_spec_path.read_text(encoding="utf-8"))
    for key, expected in {
        "version_id": DEFAULT_VERSION_ID,
        "build_only": True,
        "runtime_write_allowed": False,
        "automatic_promotion_allowed": False,
        "final_test_read_allowed": False,
    }.items():
        if build.get(key) != expected:
            raise ValueError(f"R2 build spec contract mismatch for {key}")
    model = build.get("model", {})
    if (
        model.get("backbone") != BACKBONE
        or model.get("hidden_dim") != HIDDEN_DIM
        or model.get("embedding_dim") != EMBEDDING_DIM
        or model.get("image_size") != IMAGE_SIZE
        or model.get("num_classes") != EXPECTED_CLASSES
    ):
        raise ValueError("R2 model contract mismatch")
    training = build.get("training", {})
    if training.get("seed") != FULL_DATA_SEED or training.get("replicas") != 2:
        raise ValueError("R2 refit contract mismatch")
    cache = build.get("cache", {})
    if (
        cache.get("replicas") != 2
        or cache.get("views") != VIEWS
        or cache.get("view_seed") != VIEW_SEED
        or cache.get("row_batch_size") != IMAGE_BATCH_SIZE
        or cache.get("persistent_dtype") != "float16"
        or cache.get("semantic_replica_equality_required") is not True
    ):
        raise ValueError("R2 extraction contract mismatch")
    if e2e.get("model_version_id") != DEFAULT_VERSION_ID:
        raise ValueError("R2 E2E spec targets another model version")
    return {
        "build_spec": build,
        "e2e_spec": e2e,
        "build_spec_sha256": build_hash,
        "e2e_spec_sha256": e2e_hash,
    }


def _class_map(rows: Sequence[dict[str, Any]]) -> dict[int, str]:
    result: dict[int, str] = {}
    for row in rows:
        label = int(row["class_label"])
        name = str(row["class_name"])
        previous = result.setdefault(label, name)
        if previous != name:
            raise ValueError(f"class label {label} maps to multiple names")
    return dict(sorted(result.items()))


def _metadata(rows: Sequence[dict[str, Any]]) -> dict[str, list[Any]]:
    fields = (
        "row_id",
        "class_label",
        "class_name",
        "component_id",
        "decoded_pixel_sha256",
        "fold",
    )
    return {field: [row[field] for row in rows] for field in fields}


def cache_semantic_sha256(payload: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for field in (
        "schema_version",
        "backbone",
        "hidden_dim",
        "views",
        "view_seed",
        "image_size",
        "row_id",
        "class_label",
        "class_name",
        "component_id",
        "decoded_pixel_sha256",
        "fold",
        "class_names",
    ):
        digest.update(v9.sha256_json(payload[field]).encode("ascii"))
    for field in ("features", "view_features", "labels"):
        digest.update(tensor_sha256(payload[field]).encode("ascii"))
    return digest.hexdigest()


def _environment(device: torch.device) -> dict[str, Any]:
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else None
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": str(device),
        "gpu_name": gpu_name,
    }


def precompute_full_cache(
    *,
    cache_path_value: Path,
    extraction_replica: int,
    manifest_path: Path,
    audit_path: Path,
    backbone_manifest: Path,
    views: int,
    view_seed: int,
    image_size: int,
    image_batch_size: int,
    num_workers: int,
    device: torch.device,
    force: bool,
    allow_reuse: bool = False,
) -> dict[str, Any]:
    contract = validate_contract()
    if extraction_replica not in (1, 2):
        raise ValueError("extraction replica must be 1 or 2")
    if (views, view_seed, image_size, image_batch_size, num_workers) != (
        VIEWS,
        VIEW_SEED,
        IMAGE_SIZE,
        IMAGE_BATCH_SIZE,
        NUM_WORKERS,
    ):
        raise ValueError("R2 extraction geometry differs from the frozen build spec")
    if cache_path_value.exists() and not force:
        if not allow_reuse:
            raise ValueError(
                f"R2 full-data cache already exists at {cache_path_value}; "
                "implicit reuse is forbidden for the auditable final build. "
                "Pass --allow-cache-reuse to opt in after provenance validation, "
                "or --force to re-extract."
            )
        payload = load_full_cache(cache_path_value, strict=True)
        return {
            "cache_path": str(cache_path_value.resolve()),
            "cache_sha256": sha256_file(cache_path_value),
            "cache_semantic_sha256": cache_semantic_sha256(payload),
            "rows": len(payload["row_id"]),
            "reused": True,
        }
    environment = _environment(device)
    expected_gpu = contract["build_spec"]["cache"]["expected_gpu_name"]
    if device.type != "cuda":
        raise ValueError("a fresh R2 extraction requires the CUDA device pinned by the build spec")
    if environment["gpu_name"] != expected_gpu:
        raise ValueError(
            f"R2 extraction GPU mismatch: {environment['gpu_name']} != {expected_gpu}"
        )
    rows, corpus = v9.load_corpus(manifest_path, audit_path, strict_counts=True)
    rows = sorted(rows, key=lambda row: str(row["row_id"]))
    if len({str(row["row_id"]) for row in rows}) != len(rows):
        raise ValueError("R2 corpus row_id values must be unique")
    backbone, backbone_provenance = load_backbone(BACKBONE, device, backbone_manifest)
    backbone.requires_grad_(False).eval()
    try:
        base_features, view_features = v9.extract_features(
            backbone,
            rows,
            views=views,
            seed=view_seed,
            image_size=image_size,
            batch_size=image_batch_size,
            num_workers=num_workers,
            device=device,
        )
    finally:
        del backbone
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if view_features is None:
        raise RuntimeError("R2 full-data extraction did not produce SSL views")
    base_features = base_features.detach().cpu().half().contiguous()
    view_features = view_features.detach().cpu().half().contiguous()
    if tuple(base_features.shape) != (len(rows), HIDDEN_DIM):
        raise ValueError("R2 full-data base feature shape mismatch")
    if tuple(view_features.shape) != (len(rows), VIEWS, HIDDEN_DIM):
        raise ValueError("R2 full-data view feature shape mismatch")
    metadata = _metadata(rows)
    labels = torch.tensor(metadata["class_label"], dtype=torch.long)
    payload = {
        "schema_version": CACHE_SCHEMA,
        "build_spec_sha256": contract["build_spec_sha256"],
        "extraction_replica": extraction_replica,
        "backbone": BACKBONE,
        "hidden_dim": HIDDEN_DIM,
        "features": base_features,
        "view_features": view_features,
        "labels": labels,
        "class_names": _class_map(rows),
        "views": views,
        "view_seed": view_seed,
        "image_size": image_size,
        **metadata,
        "provenance": {
            "corpus_validation": corpus,
            "backbone": backbone_provenance,
            "environment": environment,
            "manifest_sha256": sha256_file(manifest_path),
            "audit_sha256": sha256_file(audit_path),
            "backbone_manifest_sha256": sha256_file(backbone_manifest),
            "safe_view_operations": list(v9.SAFE_VIEW_OPERATIONS),
            "forbidden_view_operations": sorted(v9.FORBIDDEN_VIEW_OPERATIONS),
            "labels_used_by_ssl": False,
            "heldout_rows_consumed": 0,
            "external_holdout_consumed": False,
            "final_test_read": False,
            "runtime_write": False,
        },
    }
    semantic_hash = cache_semantic_sha256(payload)
    cache_path_value.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path_value.with_suffix(cache_path_value.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(cache_path_value)
    sidecar = {
        "schema_version": CACHE_SIDECAR_SCHEMA,
        "cache_path": str(cache_path_value.resolve()),
        "cache_sha256": sha256_file(cache_path_value),
        "cache_semantic_sha256": semantic_hash,
        "build_spec_sha256": contract["build_spec_sha256"],
        "extraction_replica": extraction_replica,
        "row_count": len(rows),
        "component_count": len(set(metadata["component_id"])),
        "class_count": len(payload["class_names"]),
        "row_ids_sha256": v9.sha256_json(metadata["row_id"]),
        "labels_sha256": v9.sha256_json(metadata["class_label"]),
        "manifest_sha256": sha256_file(manifest_path),
        "audit_sha256": sha256_file(audit_path),
        "backbone_manifest_sha256": sha256_file(backbone_manifest),
        "environment": payload["provenance"]["environment"],
        "final_test_read": False,
        "runtime_write": False,
    }
    v9.write_json(cache_sidecar_path(cache_path_value), sidecar)
    return {
        "cache_path": str(cache_path_value.resolve()),
        "cache_sha256": sidecar["cache_sha256"],
        "cache_semantic_sha256": semantic_hash,
        "rows": len(rows),
        "reused": False,
    }


def load_full_cache(path: Path, *, strict: bool) -> dict[str, Any]:
    sidecar_path = cache_sidecar_path(path)
    if not sidecar_path.is_file():
        raise ValueError(f"missing R2 full-data cache sidecar: {sidecar_path}")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if sidecar.get("schema_version") != CACHE_SIDECAR_SCHEMA:
        raise ValueError("unexpected R2 full-data cache sidecar schema")
    if sidecar.get("cache_sha256") != sha256_file(path):
        raise ValueError("R2 full-data cache hash does not match sidecar")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != CACHE_SCHEMA:
        raise ValueError("unexpected R2 full-data cache schema")
    required = {
        "features", "view_features", "labels", "class_names", "row_id",
        "component_id", "decoded_pixel_sha256", "class_label", "class_name",
        "fold", "provenance",
    }
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"R2 full-data cache missing keys: {sorted(missing)}")
    row_count = len(payload["row_id"])
    if payload["row_id"] != sorted(payload["row_id"]):
        raise ValueError("R2 full-data cache is not ordered lexicographically by row_id")
    if len(set(payload["row_id"])) != row_count:
        raise ValueError("R2 full-data cache row_id values are not unique")
    for field in ("component_id", "decoded_pixel_sha256", "class_label", "class_name", "fold"):
        if len(payload[field]) != row_count:
            raise ValueError(f"R2 full-data cache metadata length mismatch: {field}")
    features = payload["features"]
    views = payload["view_features"]
    labels = payload["labels"]
    if not isinstance(features, torch.Tensor) or features.shape != (row_count, HIDDEN_DIM):
        raise ValueError("R2 full-data base feature tensor shape mismatch")
    if not isinstance(views, torch.Tensor) or views.shape != (row_count, int(payload["views"]), HIDDEN_DIM):
        raise ValueError("R2 full-data view feature tensor shape mismatch")
    if features.dtype != torch.float16 or views.dtype != torch.float16:
        raise ValueError("R2 persistent feature tensors must be float16")
    if not torch.isfinite(features).all() or not torch.isfinite(views).all():
        raise FloatingPointError("R2 full-data cache contains non-finite features")
    if not isinstance(labels, torch.Tensor) or labels.shape != (row_count,):
        raise ValueError("R2 full-data labels tensor shape mismatch")
    if labels.tolist() != [int(value) for value in payload["class_label"]]:
        raise ValueError("R2 full-data label tensor/metadata mismatch")
    if payload.get("backbone") != BACKBONE or payload.get("hidden_dim") != HIDDEN_DIM:
        raise ValueError("R2 full-data backbone contract mismatch")
    if payload.get("build_spec_sha256") != EXPECTED_BUILD_SPEC_SHA256:
        raise ValueError("R2 full-data cache build spec mismatch")
    if cache_semantic_sha256(payload) != sidecar.get("cache_semantic_sha256"):
        raise ValueError("R2 full-data cache semantic hash mismatch")
    provenance = payload["provenance"]
    if (
        provenance.get("labels_used_by_ssl") is not False
        or provenance.get("heldout_rows_consumed") != 0
        or provenance.get("external_holdout_consumed") is not False
        or provenance.get("final_test_read") is not False
        or provenance.get("runtime_write") is not False
    ):
        raise ValueError("R2 full-data cache provenance violates the build-only contract")
    if strict:
        corpus = provenance.get("corpus_validation", {})
        expected = {
            "retained_rows": EXPECTED_ROWS,
            "components_after_quarantine": EXPECTED_COMPONENTS,
            "classes": EXPECTED_CLASSES,
            "quarantined_rows": 49,
            "conflicting_rgb_hashes": 22,
        }
        mismatches = {
            key: (corpus.get(key), value)
            for key, value in expected.items()
            if corpus.get(key) != value
        }
        if mismatches:
            raise ValueError(f"strict R2 full-data corpus mismatch: {mismatches}")
        if (
            row_count != EXPECTED_ROWS
            or len(payload["class_names"]) != EXPECTED_CLASSES
            or len(set(payload["component_id"])) != EXPECTED_COMPONENTS
            or int(payload["views"]) != VIEWS
            or int(payload["view_seed"]) != VIEW_SEED
            or int(payload["image_size"]) != IMAGE_SIZE
        ):
            raise ValueError("strict R2 full-data geometry mismatch")
    return payload


def _cosines(left: torch.Tensor, right: torch.Tensor, *, chunk_size: int = 2048) -> list[float]:
    if left.shape != right.shape:
        raise ValueError("cosine reference tensors have different shapes")
    values: list[float] = []
    for offset in range(0, len(left), chunk_size):
        similarity = functional.cosine_similarity(
            left[offset : offset + chunk_size].float(),
            right[offset : offset + chunk_size].float(),
            dim=-1,
        )
        values.extend(float(value) for value in similarity.reshape(-1))
    return values


def compare_cache_to_fold_references(
    cache: dict[str, Any],
    fold_cache_dir: Path,
    *,
    strict: bool,
) -> dict[str, Any]:
    paths = sorted(fold_cache_dir.glob("fold-*-views08.pt"))
    if strict and len(paths) != 5:
        raise ValueError(f"R2 fold reference directory must contain five caches, found {len(paths)}")
    full_index = {str(row_id): index for index, row_id in enumerate(cache["row_id"])}
    base_values: list[float] = []
    view_values: list[float] = []
    base_coverage: set[str] = set()
    view_coverage: set[str] = set()
    for path in paths:
        if strict:
            sidecar_path = path.with_suffix(path.suffix + ".prov.json")
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            actual_sha256 = sha256_file(path)
            if sidecar.get("cache_sha256") != actual_sha256:
                raise ValueError(f"fold cache hash does not match its sidecar: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        for split_name in ("train", "oof"):
            split = payload.get(split_name)
            if not isinstance(split, dict):
                raise ValueError(f"fold cache lacks {split_name}: {path}")
            row_ids = [str(value) for value in split["row_id"]]
            try:
                indices = torch.tensor([full_index[row_id] for row_id in row_ids], dtype=torch.long)
            except KeyError as exc:
                raise ValueError(f"fold cache contains an unknown row_id: {exc.args[0]}") from exc
            base_values.extend(_cosines(cache["features"][indices], split["base_features"]))
            base_coverage.update(row_ids)
            if split_name == "train":
                fold_views = split.get("view_features")
                if not isinstance(fold_views, torch.Tensor):
                    raise ValueError(f"fold train cache lacks view_features: {path}")
                view_values.extend(_cosines(cache["view_features"][indices], fold_views))
                view_coverage.update(row_ids)
    if not base_values or not view_values:
        raise ValueError("fold reference comparison produced no cosine samples")
    if strict and (
        len(base_coverage) != len(cache["row_id"])
        or len(view_coverage) != len(cache["row_id"])
    ):
        raise ValueError("fold reference caches do not cover every full-data row")
    base_tensor = torch.tensor(base_values, dtype=torch.float64)
    view_tensor = torch.tensor(view_values, dtype=torch.float64)
    result = {
        "fold_cache_count": len(paths),
        "base_samples": len(base_values),
        "view_samples": len(view_values),
        "base_rows_covered": len(base_coverage),
        "view_rows_covered": len(view_coverage),
        "base_cosine_min": float(base_tensor.min()),
        "base_cosine_median": float(base_tensor.median()),
        "view_cosine_min": float(view_tensor.min()),
        "view_cosine_median": float(view_tensor.median()),
    }
    result["pass"] = (
        result["base_cosine_min"] >= COSINE_MIN
        and result["view_cosine_min"] >= COSINE_MIN
        and result["base_cosine_median"] >= COSINE_MEDIAN_MIN
        and result["view_cosine_median"] >= COSINE_MEDIAN_MIN
    )
    return result


def verify_cache_replicas(
    left_path: Path,
    right_path: Path,
    *,
    fold_cache_dir: Path | None,
    output_path: Path | None,
    strict: bool,
) -> dict[str, Any]:
    left = load_full_cache(left_path, strict=strict)
    right = load_full_cache(right_path, strict=strict)
    metadata_fields = (
        "row_id", "class_label", "class_name", "component_id",
        "decoded_pixel_sha256", "fold", "class_names",
    )
    tensor_gates = {
        f"{field}_exact": torch.equal(left[field], right[field])
        for field in ("features", "view_features", "labels")
    }
    left_semantic = cache_semantic_sha256(left)
    right_semantic = cache_semantic_sha256(right)
    cosine = (
        compare_cache_to_fold_references(left, fold_cache_dir, strict=strict)
        if fold_cache_dir is not None
        else None
    )
    gates = {
        "metadata_exact": all(left[field] == right[field] for field in metadata_fields),
        **tensor_gates,
        "semantic_hash_exact": left_semantic == right_semantic,
        "fold_reference_cosine": cosine is None or cosine["pass"],
        "final_test_unread": left["provenance"].get("final_test_read") is False,
        "runtime_write_forbidden": left["provenance"].get("runtime_write") is False,
    }
    result = {
        "schema_version": "autoresearch-full-data-v19r2.cache-reproducibility-v1",
        "pass": all(gates.values()),
        "gates": gates,
        "left_cache_sha256": sha256_file(left_path),
        "right_cache_sha256": sha256_file(right_path),
        "cache_files_byte_exact": sha256_file(left_path) == sha256_file(right_path),
        "left_semantic_sha256": left_semantic,
        "right_semantic_sha256": right_semantic,
        "fold_reference_cosine": cosine,
        "promotion_eligible": False,
        "final_test_read": False,
    }
    if not result["pass"]:
        raise RuntimeError(f"R2 cache reproducibility gates failed: {gates}")
    if output_path is not None:
        v9.write_json(output_path, result)
    return result


def fit_metrics(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    class_labels: torch.Tensor,
) -> dict[str, Any]:
    return {
        "context": METRIC_CONTEXT,
        "values": exporter.compute_cache_metrics(embeddings, labels, class_labels),
    }


def train_replica(
    *,
    cache_path_value: Path,
    output_dir: Path,
    seed: int,
    ssl_epochs: int,
    ssl_batch_size: int,
    supervised_epochs: int,
    episodes_per_epoch: int,
    n_way: int,
    k_shot: int,
    q_queries: int,
    device: torch.device,
    supervised_device: torch.device,
    strict_cache: bool,
) -> dict[str, Any]:
    contract = validate_contract()
    if strict_cache and (
        seed,
        ssl_epochs,
        ssl_batch_size,
        supervised_epochs,
        episodes_per_epoch,
        n_way,
        k_shot,
        q_queries,
    ) != (
        FULL_DATA_SEED,
        30,
        256,
        30,
        100,
        20,
        3,
        5,
    ):
        raise ValueError("R2 refit recipe differs from the frozen build spec")
    cache = load_full_cache(cache_path_value, strict=strict_cache)
    labels = cache["labels"].long().cpu()
    counts = Counter(int(value) for value in labels.tolist())
    eligible = sorted(
        label for label, count in counts.items()
        if count >= k_shot + q_queries
    )
    if len(eligible) < n_way:
        raise ValueError("too few full-data classes support the R2 episodic contract")
    v9.configure_determinism(seed)
    model = ProjectionHead(input_dim=HIDDEN_DIM, embedding_dim=EMBEDDING_DIM).to(device)
    initial_state_sha256 = v9.state_dict_sha256(model)
    train_payload = {
        "row_id": cache["row_id"],
        "component_id": cache["component_id"],
        "decoded_pixel_sha256": cache["decoded_pixel_sha256"],
        "base_features": cache["features"],
        "view_features": cache["view_features"],
    }
    ssl = v9.pretrain_vicreg(
        model,
        train_payload,
        epochs=ssl_epochs,
        batch_size=ssl_batch_size,
        learning_rate=3e-4,
        weight_decay=1e-4,
        seed=seed,
        device=device,
    )
    episode_seed = v9.stable_seed("supervised-episodes", "full", seed)
    episode_plan, episode_plan_sha256 = v9.build_episode_plan(
        labels,
        n_way=n_way,
        k_shot=k_shot,
        q_queries=q_queries,
        epochs=supervised_epochs,
        episodes_per_epoch=episodes_per_epoch,
        seed=episode_seed,
    )
    supervised_rng_seed = v9.stable_seed("supervised-rng", "full", seed)
    model = model.to(supervised_device)
    supervised = v9.train_supervised(
        model,
        cache["features"],
        episode_plan,
        learning_rate=1e-3,
        weight_decay=1e-4,
        temperature=0.1,
        warmup_epochs=min(5, supervised_epochs),
        rng_seed=supervised_rng_seed,
        device=supervised_device,
    )
    model = model.to(torch.device("cpu"))
    embeddings = v9.embed_in_batches(model, cache["features"], torch.device("cpu"))
    class_labels = labels.unique(sorted=True)
    prototypes = compute_prototypes(embeddings, labels).cpu()
    if strict_cache and len(prototypes) != EXPECTED_CLASSES:
        raise ValueError(f"R2 refit must produce {EXPECTED_CLASSES} prototypes")
    state_dict = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in sorted(model.state_dict().items())
    }
    result = {
        "schema_version": "autoresearch-full-data-v19r2.replica-v1",
        "seed": seed,
        "cache_sha256": sha256_file(cache_path_value),
        "cache_semantic_sha256": cache_semantic_sha256(cache),
        "build_spec_sha256": contract["build_spec_sha256"],
        "e2e_spec_sha256": contract["e2e_spec_sha256"],
        "initial_state_sha256": initial_state_sha256,
        "state_dict_sha256": state_dict_sha256(state_dict),
        "prototypes_sha256": tensor_sha256(prototypes),
        "prototype_count": int(prototypes.shape[0]),
        "eligible_class_labels": eligible,
        "eligible_class_labels_sha256": v9.sha256_json(eligible),
        "episode_seed": episode_seed,
        "episode_plan_sha256": episode_plan_sha256,
        "supervised_rng_seed": supervised_rng_seed,
        "ssl": ssl,
        "supervised": supervised,
        "train_fit_diagnostics": fit_metrics(embeddings, labels, class_labels),
        "effective_rank": v9.effective_rank(embeddings),
        "final_test_read": False,
        "external_holdout_consumed": False,
        "runtime_write": False,
        "promotion_eligible": False,
    }
    promotion_contract = {
        "e2e_report_required": True,
        "e2e_spec_path": str(E2E_SPEC_PATH.relative_to(ROOT)).replace("\\", "/"),
        "e2e_spec_sha256": contract["e2e_spec_sha256"],
        "build_spec_path": str(BUILD_SPEC_PATH.relative_to(ROOT)).replace("\\", "/"),
        "build_spec_sha256": contract["build_spec_sha256"],
    }
    checkpoint_payload = {
        "model_state_dict": state_dict,
        "prototypes": prototypes,
        "class_labels": class_labels.cpu(),
        "spec": {
            "objective": "vicreg_then_supervised_episodic_full_data_r2_build",
            "initialization": "fresh_deterministic",
            "seed": seed,
            "backbone": BACKBONE,
            "hidden_dim": HIDDEN_DIM,
            "embedding_dim": EMBEDDING_DIM,
            "teacher_weight": 0.0,
            "hidden_teacher_weight": 0.0,
        },
        "metric_context": METRIC_CONTEXT,
        "rejection_threshold_status": "legacy_inherited_unvalidated",
        "promotion_contract": promotion_contract,
        "promotion_eligible": False,
        "build": result,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.pt"
    temporary = checkpoint_path.with_suffix(".pt.tmp")
    torch.save(checkpoint_payload, temporary)
    temporary.replace(checkpoint_path)
    summary = {
        **result,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }
    v9.write_json(output_dir / "summary.json", summary)
    return summary


def verify_refit_replicas(
    left_path: Path,
    right_path: Path,
    *,
    output_path: Path | None,
    strict: bool,
) -> dict[str, Any]:
    left = torch.load(left_path, map_location="cpu", weights_only=False)
    right = torch.load(right_path, map_location="cpu", weights_only=False)
    left_state = left["model_state_dict"]
    right_state = right["model_state_dict"]
    state_exact = set(left_state) == set(right_state) and all(
        torch.equal(left_state[name], right_state[name]) for name in left_state
    )
    build_fields = (
        "seed", "cache_semantic_sha256", "build_spec_sha256", "e2e_spec_sha256",
        "initial_state_sha256", "state_dict_sha256", "prototypes_sha256",
        "prototype_count", "eligible_class_labels", "eligible_class_labels_sha256",
        "episode_seed", "episode_plan_sha256", "supervised_rng_seed", "ssl",
        "supervised", "train_fit_diagnostics", "effective_rank", "final_test_read",
        "external_holdout_consumed", "runtime_write", "promotion_eligible",
    )
    diagnostics_exact = all(
        left["build"].get(key) == right["build"].get(key) for key in build_fields
    )
    prototype_count_valid = (
        len(left["prototypes"]) == EXPECTED_CLASSES
        if strict
        else len(left["prototypes"]) == len(left["class_labels"])
    )
    gates = {
        "state_dict_exact": state_exact,
        "prototypes_exact": torch.equal(left["prototypes"], right["prototypes"]),
        "class_labels_exact": torch.equal(left["class_labels"], right["class_labels"]),
        "training_diagnostics_exact": diagnostics_exact,
        "prototype_count_valid": prototype_count_valid,
        "promotion_contract_exact": left.get("promotion_contract") == right.get("promotion_contract"),
        "final_test_unread": left["build"].get("final_test_read") is False,
        "runtime_write_forbidden": left["build"].get("runtime_write") is False,
        "promotion_ineligible": left.get("promotion_eligible") is False,
    }
    result = {
        "schema_version": "autoresearch-full-data-v19r2.refit-reproducibility-v1",
        "pass": all(gates.values()),
        "gates": gates,
        "left_checkpoint_sha256": sha256_file(left_path),
        "right_checkpoint_sha256": sha256_file(right_path),
        "checkpoint_files_byte_exact": sha256_file(left_path) == sha256_file(right_path),
        "state_dict_sha256": state_dict_sha256(left_state),
        "prototypes_sha256": tensor_sha256(left["prototypes"]),
        "prototype_count": int(left["prototypes"].shape[0]),
        "promotion_eligible": False,
        "final_test_read": False,
    }
    if not result["pass"]:
        raise RuntimeError(f"R2 refit reproducibility gates failed: {gates}")
    if output_path is not None:
        v9.write_json(output_path, result)
    return result


def runtime_hashes() -> dict[str, str]:
    return {
        "projection": sha256_file(DEFAULT_RUNTIME_PROJECTION),
        "prototypes": sha256_file(DEFAULT_RUNTIME_PROTOTYPES),
        "config": sha256_file(DEFAULT_RUNTIME_CONFIG),
    }


def export_candidate(
    *,
    checkpoint_path: Path,
    cache_path_value: Path,
    registry_root: Path,
    version_id: str,
    device_name: str,
    result_path: Path | None,
) -> dict[str, Any]:
    contract = validate_contract()
    before = runtime_hashes()
    output = exporter.export_elements_refit(
        checkpoint_path=checkpoint_path,
        feature_cache_path=cache_path_value,
        registry_root=registry_root,
        runtime_config_path=DEFAULT_RUNTIME_CONFIG,
        runtime_prototypes_path=DEFAULT_RUNTIME_PROTOTYPES,
        version_id=version_id,
        device_name=device_name,
    )
    after = runtime_hashes()
    if before != after:
        raise RuntimeError("runtime artifacts changed during R2 registry export")
    manifest = output["manifest"]
    promotion = manifest.get("promotion", {})
    expected_promotion = {
        "requires_manual_review": True,
        "eligible": False,
        "e2e_report_required": True,
        "e2e_spec_path": str(E2E_SPEC_PATH.relative_to(ROOT)).replace("\\", "/"),
        "e2e_spec_sha256": contract["e2e_spec_sha256"],
        "build_spec_path": str(BUILD_SPEC_PATH.relative_to(ROOT)).replace("\\", "/"),
        "build_spec_sha256": contract["build_spec_sha256"],
    }
    if manifest.get("status") != "candidate" or promotion != expected_promotion:
        raise RuntimeError("R2 export manifest does not preserve the frozen promotion contract")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    exported_projection = torch.load(
        Path(output["projection_path"]), map_location="cpu", weights_only=False
    )
    exported_prototypes = torch.load(
        Path(output["prototypes_path"]), map_location="cpu", weights_only=False
    )["prototypes"]
    projection_exact = set(exported_projection) == set(checkpoint["model_state_dict"]) and all(
        torch.equal(exported_projection[name], checkpoint["model_state_dict"][name])
        for name in exported_projection
    )
    prototypes_exact = torch.equal(exported_prototypes, checkpoint["prototypes"])
    result = {
        "schema_version": "autoresearch-full-data-v19r2.candidate-export-v1",
        "pass": projection_exact and prototypes_exact and before == after,
        "version_id": version_id,
        "registry_status": manifest["status"],
        "projection_exact": projection_exact,
        "prototypes_exact": prototypes_exact,
        "runtime_before": before,
        "runtime_after": after,
        "runtime_unchanged": before == after,
        "metrics_context": manifest["metrics_context"],
        "promotion_eligible": False,
        "final_test_read": False,
        "export": output,
    }
    if not result["pass"]:
        raise RuntimeError("R2 registry export did not preserve the verified candidate")
    if result_path is not None:
        v9.write_json(result_path, result)
    return result


def add_extraction_args(parser: argparse.ArgumentParser, *, include_replica: bool) -> None:
    if include_replica:
        parser.add_argument("--replica", type=int, choices=(1, 2), required=True)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--manifest", type=Path, default=v9.DEFAULT_MANIFEST)
    parser.add_argument("--audit", type=Path, default=v9.DEFAULT_AUDIT)
    parser.add_argument("--backbone-manifest", type=Path, default=DEFAULT_BACKBONE_MANIFEST)
    parser.add_argument("--views", type=int, default=VIEWS)
    parser.add_argument("--view-seed", type=int, default=VIEW_SEED)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--image-batch-size", type=int, default=IMAGE_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--device", default="auto")


def add_training_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--replica", type=int, choices=(1, 2), required=True)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--seed", type=int, default=FULL_DATA_SEED)
    parser.add_argument("--ssl-epochs", type=int, default=30)
    parser.add_argument("--ssl-batch-size", type=int, default=256)
    parser.add_argument("--supervised-epochs", type=int, default=30)
    parser.add_argument("--episodes-per-epoch", type=int, default=100)
    parser.add_argument("--n-way", type=int, default=20)
    parser.add_argument("--k-shot", type=int, default=3)
    parser.add_argument("--q-queries", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--supervised-device", default="cpu")
    parser.add_argument("--supervised-cpu-threads", type=int, default=1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    precompute = subparsers.add_parser("precompute")
    add_extraction_args(precompute, include_replica=True)
    precompute.add_argument("--force", action="store_true")
    precompute.add_argument("--allow-cache-reuse", action="store_true")
    verify_cache = subparsers.add_parser("verify-cache")
    verify_cache.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    verify_cache.add_argument("--left-cache", type=Path, default=None)
    verify_cache.add_argument("--right-cache", type=Path, default=None)
    verify_cache.add_argument("--fold-cache-dir", type=Path, default=DEFAULT_FOLD_CACHE_DIR)
    verify_cache.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    refit = subparsers.add_parser("refit")
    add_training_args(refit)
    verify_refit = subparsers.add_parser("verify-refit")
    verify_refit.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    export = subparsers.add_parser("export")
    export.add_argument("--cache", type=Path, default=None)
    export.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    export.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    export.add_argument("--registry-root", type=Path, default=DEFAULT_REGISTRY)
    export.add_argument("--version-id", default=DEFAULT_VERSION_ID)
    export.add_argument("--fold-cache-dir", type=Path, default=DEFAULT_FOLD_CACHE_DIR)
    export.add_argument("--device", default="cpu")
    all_parser = subparsers.add_parser("all")
    add_extraction_args(all_parser, include_replica=False)
    all_parser.add_argument("--force-cache", action="store_true")
    all_parser.add_argument("--allow-cache-reuse", action="store_true")
    all_parser.add_argument("--fold-cache-dir", type=Path, default=DEFAULT_FOLD_CACHE_DIR)
    all_parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    all_parser.add_argument("--seed", type=int, default=FULL_DATA_SEED)
    all_parser.add_argument("--ssl-epochs", type=int, default=30)
    all_parser.add_argument("--ssl-batch-size", type=int, default=256)
    all_parser.add_argument("--supervised-epochs", type=int, default=30)
    all_parser.add_argument("--episodes-per-epoch", type=int, default=100)
    all_parser.add_argument("--n-way", type=int, default=20)
    all_parser.add_argument("--k-shot", type=int, default=3)
    all_parser.add_argument("--q-queries", type=int, default=5)
    all_parser.add_argument("--supervised-device", default="cpu")
    all_parser.add_argument("--supervised-cpu-threads", type=int, default=1)
    all_parser.add_argument("--registry-root", type=Path, default=DEFAULT_REGISTRY)
    all_parser.add_argument("--version-id", default=DEFAULT_VERSION_ID)
    all_parser.add_argument("--export-device", default="cpu")
    return parser


def _resolved_cache(args: argparse.Namespace, replica: int) -> Path:
    return args.cache if args.cache is not None else cache_path(args.cache_dir, replica)


def _training_kwargs(args: argparse.Namespace, replica: int) -> dict[str, Any]:
    supervised_device = v9.resolve_device(args.supervised_device)
    if supervised_device.type == "cpu":
        torch.set_num_threads(args.supervised_cpu_threads)
    return {
        "cache_path_value": _resolved_cache(args, replica),
        "output_dir": args.results_dir / f"replica-{replica:02d}",
        "seed": args.seed,
        "ssl_epochs": args.ssl_epochs,
        "ssl_batch_size": args.ssl_batch_size,
        "supervised_epochs": args.supervised_epochs,
        "episodes_per_epoch": args.episodes_per_epoch,
        "n_way": args.n_way,
        "k_shot": args.k_shot,
        "q_queries": args.q_queries,
        "device": v9.resolve_device(args.device),
        "supervised_device": supervised_device,
        "strict_cache": True,
    }


def _precompute_kwargs(args: argparse.Namespace, replica: int, force: bool) -> dict[str, Any]:
    return {
        "cache_path_value": _resolved_cache(args, replica),
        "extraction_replica": replica,
        "manifest_path": args.manifest,
        "audit_path": args.audit,
        "backbone_manifest": args.backbone_manifest,
        "views": args.views,
        "view_seed": args.view_seed,
        "image_size": args.image_size,
        "image_batch_size": args.image_batch_size,
        "num_workers": args.num_workers,
        "device": v9.resolve_device(args.device),
        "force": force,
        "allow_reuse": getattr(args, "allow_cache_reuse", False),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_contract()
    if args.command == "all" and args.cache is not None:
        raise ValueError("R2 all requires --cache-dir so the two extractions stay independent")
    if args.command == "precompute":
        result = precompute_full_cache(**_precompute_kwargs(args, args.replica, args.force))
    elif args.command == "verify-cache":
        result = verify_cache_replicas(
            args.left_cache or cache_path(args.cache_dir, 1),
            args.right_cache or cache_path(args.cache_dir, 2),
            fold_cache_dir=args.fold_cache_dir,
            output_path=args.results_dir / "cache-reproducibility.json",
            strict=True,
        )
    elif args.command == "refit":
        result = train_replica(**_training_kwargs(args, args.replica))
    elif args.command == "verify-refit":
        result = verify_refit_replicas(
            args.results_dir / "replica-01/checkpoint.pt",
            args.results_dir / "replica-02/checkpoint.pt",
            output_path=args.results_dir / "refit-reproducibility.json",
            strict=True,
        )
    elif args.command == "export":
        verify_cache_replicas(
            cache_path(args.cache_dir, 1),
            cache_path(args.cache_dir, 2),
            fold_cache_dir=args.fold_cache_dir,
            output_path=args.results_dir / "cache-reproducibility.json",
            strict=True,
        )
        verify_refit_replicas(
            args.results_dir / "replica-01/checkpoint.pt",
            args.results_dir / "replica-02/checkpoint.pt",
            output_path=args.results_dir / "refit-reproducibility.json",
            strict=True,
        )
        result = export_candidate(
            checkpoint_path=args.results_dir / "replica-01/checkpoint.pt",
            cache_path_value=args.cache or cache_path(args.cache_dir, 1),
            registry_root=args.registry_root,
            version_id=args.version_id,
            device_name=args.device,
            result_path=args.results_dir / "candidate-export.json",
        )
    else:
        before = runtime_hashes()
        first_cache = precompute_full_cache(**_precompute_kwargs(args, 1, args.force_cache))
        second_cache = precompute_full_cache(**_precompute_kwargs(args, 2, args.force_cache))
        cache_reproducibility = verify_cache_replicas(
            cache_path(args.cache_dir, 1),
            cache_path(args.cache_dir, 2),
            fold_cache_dir=args.fold_cache_dir,
            output_path=args.results_dir / "cache-reproducibility.json",
            strict=True,
        )
        first = train_replica(**_training_kwargs(args, 1))
        second = train_replica(**_training_kwargs(args, 2))
        refit_reproducibility = verify_refit_replicas(
            args.results_dir / "replica-01/checkpoint.pt",
            args.results_dir / "replica-02/checkpoint.pt",
            output_path=args.results_dir / "refit-reproducibility.json",
            strict=True,
        )
        candidate = export_candidate(
            checkpoint_path=args.results_dir / "replica-01/checkpoint.pt",
            cache_path_value=cache_path(args.cache_dir, 1),
            registry_root=args.registry_root,
            version_id=args.version_id,
            device_name=args.export_device,
            result_path=args.results_dir / "candidate-export.json",
        )
        after = runtime_hashes()
        result = {
            "schema_version": "autoresearch-full-data-v19r2.final-v1",
            "pass": (
                cache_reproducibility["pass"]
                and refit_reproducibility["pass"]
                and candidate["pass"]
                and before == after
            ),
            "cache_replica_01": first_cache,
            "cache_replica_02": second_cache,
            "cache_reproducibility": cache_reproducibility,
            "refit_replica_01": first,
            "refit_replica_02": second,
            "refit_reproducibility": refit_reproducibility,
            "candidate": candidate,
            "runtime_before": before,
            "runtime_after": after,
            "runtime_unchanged": before == after,
            "promotion_eligible": False,
            "final_test_read": False,
        }
        v9.write_json(args.results_dir / "final.json", result)
        if not result["pass"]:
            raise RuntimeError("R2 full-data build failed a hard gate")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

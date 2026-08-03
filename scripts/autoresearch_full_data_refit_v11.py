#!/usr/bin/env python3
"""Build a reproducible full-data VICReg candidate without evaluating or promoting it.

This runner materializes the v10 research reference as one runtime-compatible
projection/prototype package.  It never reads a holdout, never writes
``backend/codex_model``, and requires two byte-identical refits before registry
export.  Full-corpus metrics are training-fit diagnostics, not efficacy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import torch


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


RUN_DIR = (
    ROOT
    / ".omc/autoresearch/elements-baseline-replacement/runs/20260803-full-data-v11"
)
SPEC_PATH = RUN_DIR / "specs/iteration-0001.json"
EVALUATOR_PATH = RUN_DIR / "evaluator.json"
DEFAULT_CACHE = RUN_DIR / "cache/full-data-views08.pt"
DEFAULT_RESULTS = RUN_DIR / "results"
DEFAULT_REGISTRY = BACKEND / "model_registry"
DEFAULT_RUNTIME_CONFIG = BACKEND / "codex_model/config.json"
DEFAULT_RUNTIME_PROJECTION = BACKEND / "codex_model/weights/projection.pt"
DEFAULT_RUNTIME_PROTOTYPES = BACKEND / "codex_model/weights/prototypes.pt"
DEFAULT_VERSION_ID = "20260803T202104Z-vicreg-full-data-v11"
FULL_DATA_SEED = 1706649342
EXPECTED_SPEC_SHA256 = "64e0e3422bf2504ed2cf59bacf06f4ef52213010ed22a8579f35b5cdd7531b16"
EXPECTED_EVALUATOR_SHA256 = "d39f0de910ebfd80dd1972d5a9ffb299c7657cb435bb806c84513635231460e5"
METRIC_CONTEXT = "full_training_cache_fit_diagnostic_not_efficacy"


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


def validate_contract(
    spec_path: Path = SPEC_PATH,
    evaluator_path: Path = EVALUATOR_PATH,
) -> dict[str, Any]:
    spec_hash = sha256_file(spec_path)
    evaluator_hash = sha256_file(evaluator_path)
    if spec_hash != EXPECTED_SPEC_SHA256:
        raise ValueError(f"v11 spec hash mismatch: {spec_hash}")
    if evaluator_hash != EXPECTED_EVALUATOR_SHA256:
        raise ValueError(f"v11 evaluator hash mismatch: {evaluator_hash}")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    evaluator = json.loads(evaluator_path.read_text(encoding="utf-8"))
    if spec.get("seed", {}).get("value") != FULL_DATA_SEED:
        raise ValueError("v11 full-data seed contract mismatch")
    if spec.get("registry", {}).get("version_id") != DEFAULT_VERSION_ID:
        raise ValueError("v11 registry version contract mismatch")
    if spec.get("final_test_read_allowed") is not False:
        raise ValueError("v11 must forbid final-test access")
    if spec.get("runtime_write_allowed") is not False:
        raise ValueError("v11 must forbid runtime writes")
    if evaluator.get("promotion_eligible") is not False:
        raise ValueError("v11 build evaluator must forbid promotion")
    return {
        "spec": spec,
        "evaluator": evaluator,
        "spec_sha256": spec_hash,
        "evaluator_sha256": evaluator_hash,
    }


def cache_sidecar_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".prov.json")


def _class_map(rows: Sequence[dict[str, Any]]) -> dict[int, str]:
    result: dict[int, str] = {}
    for row in rows:
        label = int(row["class_label"])
        name = str(row["class_name"])
        previous = result.setdefault(label, name)
        if previous != name:
            raise ValueError(f"class label {label} maps to multiple names")
    return dict(sorted(result.items()))


def _full_metadata(rows: Sequence[dict[str, Any]]) -> dict[str, list[Any]]:
    fields = (
        "row_id",
        "class_label",
        "class_name",
        "component_id",
        "decoded_pixel_sha256",
        "fold",
    )
    return {field: [row[field] for row in rows] for field in fields}


def precompute_full_cache(
    *,
    cache_path: Path,
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
) -> dict[str, Any]:
    if views < 2:
        raise ValueError("VICReg full-data cache requires at least two views")
    if cache_path.exists() and not force:
        payload = load_full_cache(cache_path, strict=True)
        return {
            "cache_path": str(cache_path.resolve()),
            "cache_sha256": sha256_file(cache_path),
            "rows": len(payload["row_id"]),
            "reused": True,
        }

    rows, corpus = v9.load_corpus(manifest_path, audit_path, strict_counts=True)
    backbone, backbone_provenance = load_backbone(
        "dinov2_vits14", device, backbone_manifest
    )
    backbone.requires_grad_(False).eval()
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
    if view_features is None:
        raise RuntimeError("full-data cache did not produce SSL views")
    metadata = _full_metadata(rows)
    labels = torch.tensor(metadata["class_label"], dtype=torch.long)
    payload = {
        "schema_version": "autoresearch-full-data-v11.cache",
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
            "safe_view_operations": list(v9.SAFE_VIEW_OPERATIONS),
            "forbidden_view_operations": sorted(v9.FORBIDDEN_VIEW_OPERATIONS),
            "labels_used_by_ssl": False,
            "heldout_rows_consumed": 0,
            "external_holdout_consumed": False,
            "final_test_read": False,
        },
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(cache_path)
    sidecar = {
        "schema_version": "autoresearch-full-data-v11.cache.provenance",
        "cache_path": str(cache_path.resolve()),
        "cache_sha256": sha256_file(cache_path),
        "row_count": len(rows),
        "component_count": len(set(metadata["component_id"])),
        "class_count": len(payload["class_names"]),
        "views": views,
        "view_seed": view_seed,
        "row_ids_sha256": v9.sha256_json(metadata["row_id"]),
        "component_ids_sha256": v9.sha256_json(metadata["component_id"]),
        "labels_sha256": v9.sha256_json(metadata["class_label"]),
        "manifest_sha256": sha256_file(manifest_path),
        "audit_sha256": sha256_file(audit_path),
        "backbone_manifest_sha256": sha256_file(backbone_manifest),
        "heldout_rows_consumed": 0,
        "external_holdout_consumed": False,
        "final_test_read": False,
    }
    v9.write_json(cache_sidecar_path(cache_path), sidecar)
    return {
        "cache_path": str(cache_path.resolve()),
        "cache_sha256": sidecar["cache_sha256"],
        "rows": len(rows),
        "reused": False,
    }


def load_full_cache(path: Path, *, strict: bool) -> dict[str, Any]:
    sidecar_path = cache_sidecar_path(path)
    if not sidecar_path.is_file():
        raise ValueError(f"missing full-data cache sidecar: {sidecar_path}")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if sidecar.get("cache_sha256") != sha256_file(path):
        raise ValueError("full-data cache hash does not match sidecar")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != "autoresearch-full-data-v11.cache":
        raise ValueError("unexpected full-data cache schema")
    required = {
        "features",
        "view_features",
        "labels",
        "class_names",
        "row_id",
        "component_id",
        "decoded_pixel_sha256",
        "class_label",
        "class_name",
        "fold",
        "provenance",
    }
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"full-data cache missing keys: {sorted(missing)}")
    row_count = len(payload["row_id"])
    for field in (
        "component_id",
        "decoded_pixel_sha256",
        "class_label",
        "class_name",
        "fold",
    ):
        if len(payload[field]) != row_count:
            raise ValueError(f"full-data cache metadata length mismatch: {field}")
    features = payload["features"]
    views = payload["view_features"]
    labels = payload["labels"]
    if not isinstance(features, torch.Tensor) or features.shape != (row_count, 384):
        raise ValueError("full-data base feature tensor shape mismatch")
    if not isinstance(views, torch.Tensor) or views.shape != (
        row_count,
        int(payload["views"]),
        384,
    ):
        raise ValueError("full-data view feature tensor shape mismatch")
    if not isinstance(labels, torch.Tensor) or labels.shape != (row_count,):
        raise ValueError("full-data labels tensor shape mismatch")
    if labels.tolist() != [int(value) for value in payload["class_label"]]:
        raise ValueError("full-data label tensor/metadata mismatch")
    provenance = payload["provenance"]
    if provenance.get("labels_used_by_ssl") is not False:
        raise ValueError("full-data cache does not prove label-free SSL")
    if provenance.get("heldout_rows_consumed") != 0:
        raise ValueError("full-data cache reports heldout consumption")
    if provenance.get("external_holdout_consumed") is not False:
        raise ValueError("full-data cache reports external-holdout consumption")
    if provenance.get("final_test_read") is not False:
        raise ValueError("full-data cache reports final-test access")
    if strict:
        corpus = provenance.get("corpus_validation", {})
        expected = {
            "retained_rows": 9990,
            "components_after_quarantine": 300,
            "classes": 286,
            "quarantined_rows": 49,
            "conflicting_rgb_hashes": 22,
        }
        mismatches = {
            key: (corpus.get(key), value)
            for key, value in expected.items()
            if corpus.get(key) != value
        }
        if mismatches:
            raise ValueError(f"strict full-data corpus mismatch: {mismatches}")
        if row_count != 9990 or len(payload["class_names"]) != 286:
            raise ValueError("strict full-data row/class count mismatch")
        if len(set(payload["component_id"])) != 300:
            raise ValueError("strict full-data component count mismatch")
        if int(payload["views"]) != 8:
            raise ValueError("strict full-data view count mismatch")
    return payload


def fit_metrics(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    class_labels: torch.Tensor,
) -> dict[str, Any]:
    values = exporter.compute_cache_metrics(embeddings, labels, class_labels)
    return {"context": METRIC_CONTEXT, "values": values}


def train_replica(
    *,
    cache_path: Path,
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
    cache = load_full_cache(cache_path, strict=strict_cache)
    labels = cache["labels"].long().cpu()
    counts = Counter(int(value) for value in labels.tolist())
    required = k_shot + q_queries
    eligible = sorted(label for label, count in counts.items() if count >= required)
    if len(eligible) < n_way:
        raise ValueError("too few full-data classes support the episodic contract")

    v9.configure_determinism(seed)
    model = ProjectionHead(input_dim=384, embedding_dim=128).to(device)
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
    prototype_device = torch.device("cpu")
    model = model.to(prototype_device)
    embeddings = v9.embed_in_batches(model, cache["features"], prototype_device)
    class_labels = labels.unique(sorted=True)
    prototypes = compute_prototypes(embeddings, labels).cpu()
    state_dict = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in sorted(model.state_dict().items())
    }
    result = {
        "schema_version": "autoresearch-full-data-v11.replica",
        "seed": seed,
        "cache_sha256": sha256_file(cache_path),
        "initial_state_sha256": initial_state_sha256,
        "state_dict_sha256": state_dict_sha256(state_dict),
        "prototypes_sha256": tensor_sha256(prototypes),
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
    checkpoint_payload = {
        "model_state_dict": state_dict,
        "prototypes": prototypes,
        "class_labels": class_labels.cpu(),
        "spec": {
            "objective": "vicreg_then_supervised_episodic_full_data_build",
            "initialization": "fresh_deterministic",
            "seed": seed,
            "teacher_weight": 0.0,
            "hidden_teacher_weight": 0.0,
        },
        "metric_context": METRIC_CONTEXT,
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


def verify_replicas(
    left_path: Path,
    right_path: Path,
    *,
    output_path: Path | None,
) -> dict[str, Any]:
    left = torch.load(left_path, map_location="cpu", weights_only=False)
    right = torch.load(right_path, map_location="cpu", weights_only=False)
    left_state = left["model_state_dict"]
    right_state = right["model_state_dict"]
    state_exact = set(left_state) == set(right_state) and all(
        torch.equal(left_state[name], right_state[name]) for name in left_state
    )
    prototypes_exact = torch.equal(left["prototypes"], right["prototypes"])
    class_labels_exact = torch.equal(left["class_labels"], right["class_labels"])
    build_fields = (
        "seed",
        "cache_sha256",
        "initial_state_sha256",
        "state_dict_sha256",
        "prototypes_sha256",
        "eligible_class_labels_sha256",
        "episode_seed",
        "episode_plan_sha256",
        "supervised_rng_seed",
        "ssl",
        "supervised",
        "train_fit_diagnostics",
        "effective_rank",
        "final_test_read",
        "external_holdout_consumed",
        "runtime_write",
        "promotion_eligible",
    )
    build_exact = all(left["build"].get(key) == right["build"].get(key) for key in build_fields)
    checkpoint_bytes_exact = sha256_file(left_path) == sha256_file(right_path)
    gates = {
        "state_dict_byte_exact": state_exact,
        "prototypes_byte_exact": prototypes_exact,
        "class_labels_exact": class_labels_exact,
        "training_diagnostics_exact": build_exact,
        "checkpoint_file_byte_exact": checkpoint_bytes_exact,
        "final_test_unread": left["build"].get("final_test_read") is False,
        "external_holdout_unconsumed": left["build"].get("external_holdout_consumed") is False,
        "runtime_write_forbidden": left["build"].get("runtime_write") is False,
        "promotion_ineligible": left.get("promotion_eligible") is False,
    }
    result = {
        "schema_version": "autoresearch-full-data-v11.reproducibility",
        "pass": all(gates.values()),
        "gates": gates,
        "left_checkpoint_sha256": sha256_file(left_path),
        "right_checkpoint_sha256": sha256_file(right_path),
        "state_dict_sha256": state_dict_sha256(left_state),
        "prototypes_sha256": tensor_sha256(left["prototypes"]),
        "episode_plan_sha256": left["build"]["episode_plan_sha256"],
        "eligible_class_labels_sha256": left["build"]["eligible_class_labels_sha256"],
        "promotion_eligible": False,
        "final_test_read": False,
    }
    if not result["pass"]:
        raise RuntimeError(f"v11 reproducibility gates failed: {gates}")
    if output_path:
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
    cache_path: Path,
    registry_root: Path,
    version_id: str,
    device_name: str,
    result_path: Path | None,
) -> dict[str, Any]:
    before = runtime_hashes()
    output = exporter.export_elements_refit(
        checkpoint_path=checkpoint_path,
        feature_cache_path=cache_path,
        registry_root=registry_root,
        runtime_config_path=DEFAULT_RUNTIME_CONFIG,
        runtime_prototypes_path=DEFAULT_RUNTIME_PROTOTYPES,
        version_id=version_id,
        device_name=device_name,
    )
    after = runtime_hashes()
    if before != after:
        raise RuntimeError("runtime artifacts changed during v11 registry export")
    manifest = output["manifest"]
    promotion = manifest.get("promotion", {})
    if manifest.get("status") != "candidate":
        raise RuntimeError("v11 export is not a registry candidate")
    if promotion.get("requires_manual_review") is not True:
        raise RuntimeError("v11 export does not require manual review")
    if promotion.get("eligible") is not False:
        raise RuntimeError("v11 export is not explicitly promotion-ineligible")
    if manifest.get("metrics_context") != METRIC_CONTEXT:
        raise RuntimeError("v11 export metrics are not labeled as training-fit diagnostics")
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
        "schema_version": "autoresearch-full-data-v11.candidate-export",
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
        raise RuntimeError("v11 registry export did not preserve the verified candidate")
    if result_path:
        v9.write_json(result_path, result)
    return result


def add_common_cache_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--manifest", type=Path, default=v9.DEFAULT_MANIFEST)
    parser.add_argument("--audit", type=Path, default=v9.DEFAULT_AUDIT)
    parser.add_argument("--backbone-manifest", type=Path, default=v9.DEFAULT_BACKBONE_MANIFEST)
    parser.add_argument("--views", type=int, default=8)
    parser.add_argument("--view-seed", type=int, default=20260803)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--image-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")


def add_training_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
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
    add_common_cache_args(precompute)
    precompute.add_argument("--force", action="store_true")
    refit = subparsers.add_parser("refit")
    add_training_args(refit)
    refit.add_argument("--replica", type=int, choices=(1, 2), required=True)
    refit.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    export = subparsers.add_parser("export")
    export.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    export.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    export.add_argument("--registry-root", type=Path, default=DEFAULT_REGISTRY)
    export.add_argument("--version-id", default=DEFAULT_VERSION_ID)
    export.add_argument("--device", default="cpu")
    all_parser = subparsers.add_parser("all")
    add_common_cache_args(all_parser)
    all_parser.add_argument("--force-cache", action="store_true")
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
    all_parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    all_parser.add_argument("--registry-root", type=Path, default=DEFAULT_REGISTRY)
    all_parser.add_argument("--version-id", default=DEFAULT_VERSION_ID)
    all_parser.add_argument("--export-device", default="cpu")
    return parser


def _training_kwargs(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    supervised_device = v9.resolve_device(args.supervised_device)
    if supervised_device.type == "cpu":
        torch.set_num_threads(args.supervised_cpu_threads)
    return {
        "cache_path": args.cache,
        "output_dir": output_dir,
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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    contract = validate_contract()
    if args.command == "precompute":
        result = precompute_full_cache(
            cache_path=args.cache,
            manifest_path=args.manifest,
            audit_path=args.audit,
            backbone_manifest=args.backbone_manifest,
            views=args.views,
            view_seed=args.view_seed,
            image_size=args.image_size,
            image_batch_size=args.image_batch_size,
            num_workers=args.num_workers,
            device=v9.resolve_device(args.device),
            force=args.force,
        )
    elif args.command == "refit":
        result = train_replica(
            **_training_kwargs(args, args.results_dir / f"replica-{args.replica:02d}")
        )
    elif args.command == "verify":
        result = verify_replicas(
            args.results_dir / "replica-01/checkpoint.pt",
            args.results_dir / "replica-02/checkpoint.pt",
            output_path=args.results_dir / "reproducibility.json",
        )
    elif args.command == "export":
        verify_replicas(
            args.results_dir / "replica-01/checkpoint.pt",
            args.results_dir / "replica-02/checkpoint.pt",
            output_path=args.results_dir / "reproducibility.json",
        )
        result = export_candidate(
            checkpoint_path=args.results_dir / "replica-01/checkpoint.pt",
            cache_path=args.cache,
            registry_root=args.registry_root,
            version_id=args.version_id,
            device_name=args.device,
            result_path=args.results_dir / "candidate-export.json",
        )
    else:
        before = runtime_hashes()
        precompute = precompute_full_cache(
            cache_path=args.cache,
            manifest_path=args.manifest,
            audit_path=args.audit,
            backbone_manifest=args.backbone_manifest,
            views=args.views,
            view_seed=args.view_seed,
            image_size=args.image_size,
            image_batch_size=args.image_batch_size,
            num_workers=args.num_workers,
            device=v9.resolve_device(args.device),
            force=args.force_cache,
        )
        first = train_replica(
            **_training_kwargs(args, args.results_dir / "replica-01")
        )
        second = train_replica(
            **_training_kwargs(args, args.results_dir / "replica-02")
        )
        reproducibility = verify_replicas(
            args.results_dir / "replica-01/checkpoint.pt",
            args.results_dir / "replica-02/checkpoint.pt",
            output_path=args.results_dir / "reproducibility.json",
        )
        candidate = export_candidate(
            checkpoint_path=args.results_dir / "replica-01/checkpoint.pt",
            cache_path=args.cache,
            registry_root=args.registry_root,
            version_id=args.version_id,
            device_name=args.export_device,
            result_path=args.results_dir / "candidate-export.json",
        )
        after = runtime_hashes()
        result = {
            "schema_version": "autoresearch-full-data-v11.final",
            "pass": all(
                (
                    reproducibility["pass"],
                    candidate["pass"],
                    before == after,
                )
            ),
            "build_only": True,
            "efficacy_claim": False,
            "promotion_eligible": False,
            "final_test_read": False,
            "contract": {
                "spec_sha256": contract["spec_sha256"],
                "evaluator_sha256": contract["evaluator_sha256"],
            },
            "precompute": precompute,
            "replica_01": first,
            "replica_02": second,
            "reproducibility": reproducibility,
            "candidate": candidate,
            "runtime_before": before,
            "runtime_after": after,
            "runtime_unchanged": before == after,
        }
        v9.write_json(args.results_dir / "final.json", result)
    print(v9.canonical_json(result), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

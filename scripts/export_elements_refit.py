#!/usr/bin/env python3
"""Export a refit Elements checkpoint into an immutable registry package.

The exporter is intentionally narrow:
- it consumes an explicit projection-head checkpoint
- it consumes an explicit full Elements feature cache
- it validates that the cache taxonomy matches the deployed runtime taxonomy
- it writes a versioned package under backend/model_registry/versions
- it never writes backend/codex_model
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"
RUNTIME_MODEL_DIR = BACKEND_ROOT / "codex_model"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from backend.services.model_registry import ModelRegistry, atomic_write_json, sha256_file, utc_now_iso
from codex_pipeline.models.projection_head import ProjectionHead, get_device
from codex_pipeline.models.prototypical import compute_prototypes


EXPECTED_STATE_KEYS = {
    "net.0.weight",
    "net.0.bias",
    "net.3.weight",
    "net.3.bias",
}


def _is_within(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_cache(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {"features", "labels", "class_names"}
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"{path} is missing cache keys: {sorted(missing)}")
    features = payload["features"]
    labels = payload["labels"]
    if not isinstance(features, torch.Tensor) or not isinstance(labels, torch.Tensor):
        raise TypeError(f"{path} must store tensor features and labels")
    if len(features) != len(labels):
        raise ValueError(f"{path} has mismatched features and labels")
    if not isinstance(payload["class_names"], (dict, list)):
        raise TypeError(f"{path} class_names must be a dict or list")
    return payload


def _load_runtime_config(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    class_names = data.get("class_names")
    if not isinstance(class_names, list) or not all(isinstance(name, str) for name in class_names):
        raise ValueError(f"{path} must contain a class_names list")
    return data


def _load_runtime_prototypes(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {"prototypes", "class_names", "class_labels", "embedding_dim"}
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"{path} is missing runtime prototype keys: {sorted(missing)}")
    prototypes = payload["prototypes"]
    class_labels = payload["class_labels"]
    class_names = payload["class_names"]
    if not isinstance(prototypes, torch.Tensor) or prototypes.ndim != 2:
        raise TypeError(f"{path} prototypes must be a 2D tensor")
    if not isinstance(class_labels, torch.Tensor) or class_labels.ndim != 1:
        raise TypeError(f"{path} class_labels must be a 1D tensor")
    if not isinstance(class_names, dict):
        raise TypeError(f"{path} class_names must be a dict")
    if len(class_labels) != len(class_names) or len(class_labels) != prototypes.shape[0]:
        raise ValueError(f"{path} runtime prototype lengths do not match")
    if len(set(int(label) for label in class_labels.tolist())) != len(class_labels):
        raise ValueError(f"{path} class_labels must be unique")
    return payload


def _build_training_metadata(checkpoint_payload: dict[str, Any], runtime_teacher_projection_path: Path) -> dict[str, Any]:
    spec = checkpoint_payload.get("spec") if isinstance(checkpoint_payload, dict) else None
    spec = spec if isinstance(spec, dict) else {}
    teacher_weight = float(spec.get("teacher_weight", 0.0) or 0.0)
    hidden_teacher_weight = float(spec.get("hidden_teacher_weight", 0.0) or 0.0)
    teacher_assisted = teacher_weight > 0.0 or hidden_teacher_weight > 0.0
    teacher_temperature = spec.get("teacher_temperature")
    if teacher_temperature is not None:
        teacher_temperature = float(teacher_temperature)
    training = {
        "objective": spec.get("objective"),
        "initialization": spec.get("initialization"),
        "seed": spec.get("seed"),
        "checkpoint_type": "model_state_dict" if "model_state_dict" in checkpoint_payload else "raw_state_dict",
        "teacher_assisted": teacher_assisted,
        "teacher_weight": teacher_weight,
        "hidden_teacher_weight": hidden_teacher_weight,
        "teacher_temperature": teacher_temperature,
    }
    if teacher_assisted:
        if not runtime_teacher_projection_path.is_file():
            raise FileNotFoundError(runtime_teacher_projection_path)
        training["teacher_projection_path"] = str(runtime_teacher_projection_path.resolve())
        training["teacher_projection_sha256"] = sha256_file(runtime_teacher_projection_path)
    return training


def _load_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict"):
            state_dict = checkpoint.get(key)
            if isinstance(state_dict, dict):
                return state_dict
        if all(key in checkpoint for key in EXPECTED_STATE_KEYS):
            return checkpoint
    raise ValueError("checkpoint does not contain a model state dict")


def _validate_state_dict(
    state_dict: dict[str, torch.Tensor],
    *,
    input_dim: int,
    embedding_dim: int,
) -> None:
    if set(state_dict) != EXPECTED_STATE_KEYS:
        raise ValueError("checkpoint state dict is not runtime compatible")
    expected_shapes = {
        "net.0.weight": (input_dim, input_dim),
        "net.0.bias": (input_dim,),
        "net.3.weight": (embedding_dim, input_dim),
        "net.3.bias": (embedding_dim,),
    }
    for key, expected_shape in expected_shapes.items():
        if tuple(state_dict[key].shape) != expected_shape:
            raise ValueError(f"checkpoint tensor shape mismatch for {key}")


def _model_contract(
    cache: dict[str, Any],
    state_dict: dict[str, torch.Tensor],
    runtime_config: dict[str, Any],
) -> dict[str, Any]:
    features = cache["features"]
    if features.ndim != 2:
        raise ValueError("feature cache features must be a 2D tensor")
    input_dim = int(features.shape[1])
    embedding_dim = int(state_dict["net.3.weight"].shape[0])
    if input_dim not in {384, 768}:
        raise ValueError(f"unsupported Elements feature dimension: {input_dim}")
    if embedding_dim != 128:
        raise ValueError(f"unsupported Elements embedding dimension: {embedding_dim}")
    expected_backbone = {384: "dinov2_vits14", 768: "dinov2_vitb14"}[input_dim]
    cache_backbone = cache.get("backbone")
    if cache_backbone is not None and str(cache_backbone) != expected_backbone:
        raise ValueError("feature cache backbone does not match its hidden dimension")
    cache_hidden_dim = cache.get("hidden_dim")
    if cache_hidden_dim is not None and int(cache_hidden_dim) != input_dim:
        raise ValueError("feature cache hidden_dim does not match its feature tensor")
    image_size = int(cache.get("image_size", runtime_config.get("image_size", 224)))
    if image_size != 224:
        raise ValueError(f"unsupported Elements image size: {image_size}")
    _validate_state_dict(state_dict, input_dim=input_dim, embedding_dim=embedding_dim)
    return {
        "backbone": expected_backbone,
        "hidden_dim": input_dim,
        "embedding_dim": embedding_dim,
        "image_size": image_size,
    }


def _normalize_class_names(raw: dict[int, str] | list[str]) -> dict[int, str]:
    if isinstance(raw, list):
        return {index: str(name) for index, name in enumerate(raw)}
    normalized: dict[int, str] = {}
    for label, name in raw.items():
        normalized[int(label)] = str(name)
    return normalized


def _class_order(payload: dict[str, Any]) -> tuple[list[int], list[str]]:
    class_map = _normalize_class_names(payload["class_names"])
    ordered = sorted(class_map.items(), key=lambda item: item[0])
    labels = [label for label, _ in ordered]
    names = [name for _, name in ordered]
    return labels, names


def _ordered_names_by_labels(class_map: dict[int, str], class_labels: torch.Tensor) -> list[str]:
    return [class_map[int(label)] for label in class_labels.tolist()]


def compute_cache_metrics(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    class_labels: torch.Tensor,
) -> dict[str, float]:
    prototypes = compute_prototypes(embeddings, labels)
    similarities = embeddings @ prototypes.T
    top_k = min(3, similarities.shape[1])
    top_indices = similarities.topk(k=top_k, dim=1).indices
    predictions = class_labels[top_indices[:, 0]]
    top3_labels = class_labels[top_indices]

    per_class: list[float] = []
    for label in labels.unique(sorted=True):
        mask = labels == label
        per_class.append(float((predictions[mask] == labels[mask]).float().mean()))

    return {
        "prototype_top1": float((predictions == labels).float().mean()),
        "prototype_macro_top1": sum(per_class) / len(per_class),
        "prototype_top3": float((top3_labels == labels[:, None]).any(dim=1).float().mean()),
    }


def _ensure_registry_root_is_safe(registry_root: Path) -> None:
    registry_root = registry_root.resolve()
    runtime_dir = RUNTIME_MODEL_DIR.resolve()
    if registry_root == runtime_dir or _is_within(registry_root, runtime_dir):
        raise ValueError("refusing to write backend/codex_model; export to backend/model_registry instead")


def export_elements_refit(
    checkpoint_path: Path,
    feature_cache_path: Path,
    registry_root: Path,
    runtime_config_path: Path,
    runtime_prototypes_path: Path,
    version_id: str | None,
    device_name: str,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    feature_cache_path = Path(feature_cache_path)
    registry_root = Path(registry_root)
    runtime_config_path = Path(runtime_config_path)

    _ensure_registry_root_is_safe(registry_root)

    registry = ModelRegistry(registry_root, repo_root=REPO_ROOT, runtime_model_dir=RUNTIME_MODEL_DIR)
    runtime_config = _load_runtime_config(runtime_config_path)
    runtime_prototypes = _load_runtime_prototypes(runtime_prototypes_path)
    cache = _load_cache(feature_cache_path)
    class_labels_list, class_names_list = _class_order(cache)
    runtime_class_names = runtime_config["class_names"]
    runtime_class_map = _normalize_class_names(runtime_prototypes["class_names"])
    runtime_class_labels = runtime_prototypes["class_labels"].long().cpu()
    runtime_class_names_ordered = _ordered_names_by_labels(runtime_class_map, runtime_class_labels)

    if len(runtime_class_names) != 286:
        raise ValueError(f"runtime config must contain 286 class names, got {len(runtime_class_names)}")
    if class_names_list != runtime_class_names:
        raise ValueError("feature cache class order does not match the deployed runtime taxonomy")
    if runtime_class_names_ordered != runtime_class_names:
        raise ValueError("runtime prototype labels/classes do not match the deployed runtime config")

    labels = cache["labels"].long().cpu()
    observed_labels = sorted(set(int(label) for label in labels.tolist()))
    if observed_labels != class_labels_list:
        raise ValueError("feature cache labels do not match its class_names ABI")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = _load_state_dict(checkpoint)
    model_contract = _model_contract(cache, state_dict, runtime_config)
    checkpoint_payload = checkpoint if isinstance(checkpoint, dict) else {}
    metric_context = str(
        checkpoint_payload.get(
            "metric_context",
            "training_feature_cache_fit_diagnostic_not_holdout_efficacy",
        )
    )
    if checkpoint_payload.get("promotion_eligible") is True:
        raise ValueError("refit exporter only accepts promotion-ineligible checkpoints")
    promotion_eligible = False
    runtime_teacher_projection_path = RUNTIME_MODEL_DIR / "weights" / "projection.pt"
    training_metadata = _build_training_metadata(checkpoint_payload, runtime_teacher_projection_path)
    promotion_metadata: dict[str, Any] = {
        "requires_manual_review": True,
        "eligible": False,
    }
    promotion_contract = checkpoint_payload.get("promotion_contract")
    if promotion_contract is not None:
        required_promotion_keys = {
            "e2e_report_required",
            "e2e_spec_path",
            "e2e_spec_sha256",
            "build_spec_path",
            "build_spec_sha256",
        }
        if not isinstance(promotion_contract, dict):
            raise TypeError("checkpoint promotion_contract must be an object")
        if set(promotion_contract) != required_promotion_keys:
            raise ValueError("checkpoint promotion_contract keys do not match the registry ABI")
        if promotion_contract.get("e2e_report_required") is not True:
            raise ValueError("checkpoint promotion_contract must require an E2E report")
        promotion_metadata.update(promotion_contract)

    version_id = version_id or registry.build_version_id(run_id="elements-refit")
    version_dir = registry.version_dir(version_id)
    if version_dir.exists() and any(version_dir.iterdir()):
        raise FileExistsError(f"version directory already exists and is not empty: {version_dir}")
    version_dir.mkdir(parents=True, exist_ok=False)

    device = get_device(device_name)
    model = ProjectionHead(
        model_contract["hidden_dim"],
        model_contract["embedding_dim"],
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    features = cache["features"].float().to(device)
    with torch.no_grad():
        embeddings = model(features).cpu()
    prototypes = compute_prototypes(embeddings, labels)
    metrics = compute_cache_metrics(embeddings, labels, torch.tensor(class_labels_list, dtype=torch.long))

    runtime_dir = version_dir / "runtime"
    weights_dir = runtime_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    prototype_payload = {
        "prototypes": prototypes.cpu(),
        "class_names": runtime_class_map,
        "class_labels": runtime_class_labels,
        "embedding_dim": model_contract["embedding_dim"],
    }
    projection_path = weights_dir / "projection.pt"
    prototypes_path = weights_dir / "prototypes.pt"
    torch.save(state_dict, projection_path)
    torch.save(prototype_payload, prototypes_path)

    runtime_config_out = dict(runtime_config)
    runtime_config_out["num_classes"] = len(class_names_list)
    runtime_config_out["class_names"] = class_names_list
    runtime_config_out["backbone"] = model_contract["backbone"]
    runtime_config_out["embedding_dim"] = model_contract["embedding_dim"]
    runtime_config_out["hidden_dim"] = model_contract["hidden_dim"]
    runtime_config_out["image_size"] = model_contract["image_size"]
    runtime_config_out["rejection_threshold_status"] = str(
        checkpoint_payload.get(
            "rejection_threshold_status",
            "legacy_inherited_unvalidated",
        )
    )
    config_path = runtime_dir / "config.json"
    atomic_write_json(config_path, runtime_config_out)

    provenance = {
        "schema_version": "elements-refit-export.v1",
        "created_at": utc_now_iso(),
        "version_id": version_id,
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "sha256": sha256_file(checkpoint_path),
            "has_model_state_dict": "model_state_dict" in checkpoint_payload or "state_dict" in checkpoint_payload,
            "train_top1": checkpoint_payload.get("train_top1"),
            "train_macro_top1": checkpoint_payload.get("train_macro_top1"),
            "train_top3": checkpoint_payload.get("train_top3"),
            "spec": checkpoint_payload.get("spec"),
        },
        "training": training_metadata,
        "feature_cache": {
            "path": str(feature_cache_path.resolve()),
            "sha256": sha256_file(feature_cache_path),
            "feature_count": len(cache["features"]),
            "class_count": len(class_names_list),
            "class_order_sha256": _sha256_json(class_names_list),
            "class_labels_sha256": _sha256_json(class_labels_list),
            "backbone": model_contract["backbone"],
            "hidden_dim": model_contract["hidden_dim"],
        },
        "runtime": {
            "config_path": str(runtime_config_path.resolve()),
            "sha256": sha256_file(runtime_config_path),
            "class_order_sha256": _sha256_json(runtime_class_names),
            "class_count": len(runtime_class_names),
            "prototype_path": str(runtime_prototypes_path.resolve()),
            "prototype_sha256": sha256_file(runtime_prototypes_path),
            "prototype_label_order_sha256": _sha256_json(runtime_class_labels.tolist()),
            "compatible": True,
        },
        "export": {
            "device": str(device),
            "projection_path": str(projection_path),
            "prototypes_path": str(prototypes_path),
            "config_path": str(config_path),
        },
        "metrics": metrics,
        "metrics_context": metric_context,
        "promotion_eligible": promotion_eligible,
    }
    provenance_path = version_dir / "provenance.json"
    atomic_write_json(provenance_path, provenance)

    artifacts = [prototypes_path, projection_path, config_path, provenance_path]
    metadata = {
        "source": {
            "kind": "elements_refit_export",
            "checkpoint_path": str(checkpoint_path.resolve()),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "feature_cache_path": str(feature_cache_path.resolve()),
            "feature_cache_sha256": sha256_file(feature_cache_path),
            "runtime_config_path": str(runtime_config_path.resolve()),
            "runtime_config_sha256": sha256_file(runtime_config_path),
            "git_commit": registry.git_short(),
        },
        "data": {
            "class_count": len(class_names_list),
            "feature_count": len(cache["features"]),
            "class_order_sha256": _sha256_json(class_names_list),
            "class_labels_sha256": _sha256_json(class_labels_list),
            "runtime_class_labels_sha256": _sha256_json(runtime_class_labels.tolist()),
        },
        "training": training_metadata,
        "base_models": {
            "backbone": model_contract["backbone"],
            "projection_head": (
                f"ProjectionHead({model_contract['hidden_dim']}, "
                f"{model_contract['embedding_dim']})"
            ),
            "prototype_space": "cosine_mean",
        },
        "metrics": metrics,
        "metrics_context": metric_context,
        "promotion": promotion_metadata,
    }
    manifest = registry.write_manifest(version_id, status="candidate", artifact_paths=artifacts, metadata=metadata)

    return {
        "schema_version": "elements-refit-export.v1",
        "version_id": version_id,
        "version_dir": str(version_dir),
        "manifest_path": str(version_dir / "manifest.json"),
        "model_card_path": str(version_dir / "model-card.md"),
        "provenance_path": str(provenance_path),
        "runtime_config_path": str(config_path),
        "projection_path": str(projection_path),
        "prototypes_path": str(prototypes_path),
        "runtime_compatible": True,
        "class_count": len(class_names_list),
        "prototype_count": int(prototypes.shape[0]),
        "metrics": metrics,
        "metrics_context": metric_context,
        "promotion_eligible": promotion_eligible,
        "manifest": manifest,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path, help="Refit checkpoint to export")
    parser.add_argument("--feature-cache", required=True, type=Path, help="Full Elements feature cache")
    parser.add_argument(
        "--registry-root",
        type=Path,
        default=BACKEND_ROOT / "model_registry",
        help="Registry root that contains versions/ and index.json",
    )
    parser.add_argument(
        "--runtime-config",
        type=Path,
        default=RUNTIME_MODEL_DIR / "config.json",
        help="Deployed runtime config used as the taxonomy contract",
    )
    parser.add_argument(
        "--runtime-prototypes",
        type=Path,
        default=RUNTIME_MODEL_DIR / "weights" / "prototypes.pt",
        help="Deployed runtime prototypes used as the label-order contract",
    )
    parser.add_argument("--version-id", default=None, help="Optional stable registry version id")
    parser.add_argument("--device", default="auto", help="Torch device to use for projection evaluation")
    args = parser.parse_args(argv)

    result = export_elements_refit(
        args.checkpoint,
        args.feature_cache,
        args.registry_root,
        args.runtime_config,
        args.runtime_prototypes,
        args.version_id,
        args.device,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

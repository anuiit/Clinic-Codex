"""
Export trained prototypes into a backend-loadable classifier package.

Usage:
    python -m codex_pipeline.scripts.export_model --weights-dir <candidate>/runtime/weights \
      --config-template backend/codex_model/config.json \
      --config-out <candidate>/runtime/config.json

What it does:
    1. Loads trusted local prototypes/prototypes.pt (full training artefact)
    2. Writes runtime/weights/prototypes.pt  — prototypes tensor + class info
    3. Writes runtime/weights/projection.pt  — projection head state_dict only
    4. Writes runtime/config.json            — updates num_classes + class_names

By default this command refuses to write into backend/codex_model runtime paths.
Only bootstrap/install callsites should pass --allow-runtime-write.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch


class RuntimeWriteRefusedError(RuntimeError):
    """Raised when export_model would mutate runtime files without opt-in."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _optional_hash(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    return _sha256_file(path)


def _optional_file_provenance(path: Path | None, *, label: str) -> dict[str, str] | None:
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return {"path": str(path.resolve()), "sha256": _sha256_file(path)}


def _snapshot_provenance(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("schema_version") != "training-snapshot.v2":
        return None
    checksums_path = path.parent / "checksums.json"
    return {
        "schema_version": payload.get("schema_version"),
        "snapshot_id": payload.get("snapshot_id"),
        "content_sha256": payload.get("content_sha256"),
        "class_order_sha256": payload.get("class_order_sha256"),
        "ready_for_training": payload.get("ready_for_training"),
        "promotion_evaluation_ready": payload.get("promotion_evaluation_ready"),
        "manifest_sha256": _sha256_file(path),
        "checksums_sha256": _optional_hash(checksums_path),
        "split_counts": payload.get("split_counts"),
        "live_train_count": payload.get("live_train_count"),
        "live_annotation_usage": payload.get("live_annotation_usage"),
    }


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def _refuse_runtime_write_unless_allowed(
    *,
    weights_dir: Path,
    config_out_path: Path,
    runtime_model_dir: Path,
    allow_runtime_write: bool,
) -> None:
    runtime_weights_dir = runtime_model_dir / "weights"
    runtime_config_path = runtime_model_dir / "config.json"
    writes_runtime_weights = _is_relative_to(weights_dir, runtime_weights_dir)
    writes_runtime_config = config_out_path.resolve() == runtime_config_path.resolve()
    if allow_runtime_write or not (writes_runtime_weights or writes_runtime_config):
        return
    targets = []
    if writes_runtime_weights:
        targets.append(str(weights_dir))
    if writes_runtime_config:
        targets.append(str(config_out_path))
    raise RuntimeWriteRefusedError(
        "export_model refuses to write runtime classifier artifacts without "
        f"--allow-runtime-write: {', '.join(targets)}"
    )


def _prototype_contract(data: dict[str, Any], *, label: str) -> tuple[list[int], dict[int, str]]:
    labels = data.get("class_labels")
    names = data.get("class_names")
    prototypes = data.get("prototypes")
    if not isinstance(prototypes, torch.Tensor) or prototypes.ndim != 2:
        raise ValueError(f"{label} prototypes must be a 2D tensor")
    if not torch.is_floating_point(prototypes) or not torch.isfinite(prototypes).all() or not torch.allclose(prototypes.norm(dim=1), torch.ones(prototypes.size(0)), atol=1e-4):
        raise ValueError(f"{label} prototypes must be finite normalized vectors")
    if not isinstance(labels, torch.Tensor) or labels.ndim != 1 or labels.numel() != prototypes.size(0):
        raise ValueError(f"{label} class_labels must align with prototype rows")
    if labels.dtype not in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}:
        raise ValueError(f"{label} class_labels must be integer values")
    values = [int(value) for value in labels.tolist()]
    if values != sorted(values) or len(set(values)) != len(values):
        raise ValueError(f"{label} class_labels must be unique and sorted")
    if not isinstance(names, dict) or set(names) != set(values):
        raise ValueError(f"{label} class_names and class_labels disagree")
    if any(
        isinstance(key, bool) or not isinstance(key, int) or not isinstance(name, str) or not name
        for key, name in names.items()
    ):
        raise ValueError(f"{label} class_names mapping is invalid")
    if len(set(names.values())) != len(names):
        raise ValueError(f"{label} class_names must be unique")
    return values, dict(names)


def _preserve_base_numeric_contract(
    data: dict[str, Any], *, base_model_dir: Path | None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reorder candidate rows to the active package's name-to-label contract.

    Training labels are dense snapshot indices, while runtime labels may be
    sparse historical ABI values.  Names are the only safe join key.
    """
    source_labels, source_names = _prototype_contract(data, label="candidate")
    if base_model_dir is None:
        return data, {"base_package": None, "preserved": False}
    base_path = base_model_dir / "weights" / "prototypes.pt"
    if not base_path.is_file():
        return data, {"base_package": None, "preserved": False}
    base_data = torch.load(base_path, map_location="cpu", weights_only=True)
    if not isinstance(base_data, dict):
        raise ValueError(f"base runtime prototype artifact must be a mapping: {base_path}")
    base_labels, base_names = _prototype_contract(base_data, label="base runtime")
    source_by_name = {source_names[label]: index for index, label in enumerate(source_labels)}
    base_by_name = {name: label for label, name in base_names.items()}
    if set(source_by_name) != set(base_by_name):
        raise ValueError("candidate and base runtime taxonomies differ; cannot preserve numeric class labels")
    ordered_names = [base_names[label] for label in base_labels]
    row_indices = [source_by_name[name] for name in ordered_names]
    rewritten = dict(data)
    rewritten["prototypes"] = data["prototypes"][row_indices].clone()
    rewritten["class_labels"] = torch.tensor(base_labels, dtype=data["class_labels"].dtype)
    rewritten["class_names"] = {label: base_names[label] for label in base_labels}
    return rewritten, {
        "base_package": str(base_model_dir.resolve()),
        "base_prototypes_path": str(base_path.resolve()),
        "base_prototypes_sha256": _sha256_file(base_path),
        "preserved": True,
    }


def export_model(
    src_path: Path,
    weights_dir: Path,
    config_template_path: Path,
    *,
    config_out_path: Path | None = None,
    allow_runtime_write: bool = False,
    runtime_model_dir: Path | None = None,
    base_model_dir: Path | None = None,
    manifest_out_path: Path | None = None,
    registry_dir: Path | None = None,
    version_id: str | None = None,
    metadata_csv_path: Path | None = None,
    approved_manifest_path: Path | None = None,
    training_config_path: Path | None = None,
    features_path: Path | None = None,
    features_provenance_path: Path | None = None,
    checkpoint_path: Path | None = None,
    init_projection_path: Path | None = None,
    training_manifest_path: Path | None = None,
    checkpoint_selection: str | None = None,
    evaluate_candidate: bool = False,
) -> dict[str, Path]:
    if checkpoint_selection not in {None, "best", "latest"}:
        raise ValueError("checkpoint_selection must be 'best' or 'latest'")
    if evaluate_candidate and (base_model_dir is None or features_path is None or approved_manifest_path is None):
        raise ValueError("candidate evaluation requires base model, features and split manifest")
    project_root = Path(__file__).resolve().parents[2]  # backend/
    runtime_model_dir = runtime_model_dir or project_root / "codex_model"
    config_out_path = config_out_path or config_template_path
    _refuse_runtime_write_unless_allowed(
        weights_dir=weights_dir,
        config_out_path=config_out_path,
        runtime_model_dir=runtime_model_dir,
        allow_runtime_write=allow_runtime_write,
    )
    weights_dir.mkdir(parents=True, exist_ok=True)
    config_out_path.parent.mkdir(parents=True, exist_ok=True)

    data_provenance = {
        "features_cache": _optional_file_provenance(features_path, label="features cache"),
        "features_provenance": _optional_file_provenance(
            features_provenance_path, label="features provenance"
        ),
    }
    training_provenance = {
        "config": _optional_file_provenance(training_config_path, label="training config"),
        "checkpoint": _optional_file_provenance(checkpoint_path, label="training checkpoint"),
        "checkpoint_selection": checkpoint_selection,
        "init_projection": _optional_file_provenance(
            init_projection_path, label="initial projection"
        ),
        "training_manifest": _optional_file_provenance(
            training_manifest_path, label="training manifest"
        ),
    }

    # --------------------------------------------------------- load full artefact
    print(f"Loading: {src_path}")
    if not src_path.exists():
        print(f"ERROR: {src_path} not found. Train the model first.", file=sys.stderr)
        sys.exit(1)

    data = torch.load(src_path, map_location="cpu", weights_only=True)

    required_keys = {"prototypes", "class_names", "class_labels",
                     "embedding_dim", "hidden_dim", "model_state_dict"}
    missing = required_keys - set(data.keys())
    if missing:
        print(f"ERROR: prototypes.pt is missing keys: {missing}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(data, dict):
        raise ValueError("prototypes.pt must contain a mapping")
    data, numeric_label_contract = _preserve_base_numeric_contract(
        data,
        base_model_dir=base_model_dir or runtime_model_dir,
    )

    # -------------------------------------------------------- split and save
    # 1) Prototypes file — everything except the model weights
    proto_out = {
        "prototypes":   data["prototypes"],
        "class_names":  data["class_names"],   # {int: str}
        "class_labels": data["class_labels"],
        "embedding_dim": data["embedding_dim"],
    }
    proto_dest = weights_dir / "prototypes.pt"
    torch.save(proto_out, proto_dest)
    print(f"Saved prototypes  → {proto_dest}  "
          f"(shape {data['prototypes'].shape})")

    # 2) Projection head weights only
    proj_dest = weights_dir / "projection.pt"
    torch.save(data["model_state_dict"], proj_dest)
    print(f"Saved projection  → {proj_dest}  "
          f"({len(data['model_state_dict'])} tensors)")

    # -------------------------------------------------------- update config.json
    class_names_dict: dict = data["class_names"]      # {int: str}
    num_classes = len(class_names_dict)
    # Build ordered list aligned to sorted class labels.
    # class_names keys are class_label integers (can be sparse / not 0..N-1).
    sorted_labels = sorted(class_names_dict.keys())
    class_names_list = [class_names_dict[lbl] for lbl in sorted_labels]

    with open(config_template_path, "r") as f:
        config = json.load(f)

    config["num_classes"] = num_classes
    config["class_names"] = class_names_list
    config["embedding_dim"] = int(data["embedding_dim"])
    config["hidden_dim"] = int(data["hidden_dim"])

    with open(config_out_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"Read config       → {config_template_path}")
    print(f"Wrote config      → {config_out_path}")

    manifest_out: Path | None = None
    if manifest_out_path is not None:
        manifest_out_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "prototypes_source": str(src_path),
            "weights_dir": str(weights_dir),
            "config_template": str(config_template_path),
            "config_out": str(config_out_path),
            "allow_runtime_write": allow_runtime_write,
            "numeric_label_contract": numeric_label_contract,
            "data": data_provenance,
            "training": training_provenance,
            "artifacts": {
                "prototypes": {"path": str(proto_dest), "sha256": _sha256_file(proto_dest)},
                "projection": {"path": str(proj_dest), "sha256": _sha256_file(proj_dest)},
                "config": {"path": str(config_out_path), "sha256": _sha256_file(config_out_path)},
            },
        }
        manifest_out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifest_out = manifest_out_path
        print(f"Wrote manifest    → {manifest_out_path}")

    evaluation_paths = []
    if evaluate_candidate:
        sys.path.insert(0, str(project_root.parent))
        from scripts.benchmark_model_replacement import benchmark_model_replacement

        print("Final exported package metrics (earlier refit diagnostics do not score this candidate):")
        for split in ("dev", "locked_test"):
            output = config_out_path.parent.parent / "evaluation" / f"{split}.json"
            report = benchmark_model_replacement(
                features_path=features_path, split_manifest_path=approved_manifest_path,
                runtime_dir=base_model_dir, candidate_dir=config_out_path.parent,
                output_path=output, prototype_mode="stored", eval_split=split,
            )
            evaluation_paths.append(output)
            print(f"  {split}: active={report['models']['runtime']['top1_micro']:.4f}, candidate={report['models']['candidate']['top1_micro']:.4f}")

    registry_manifest: Path | None = None
    if registry_dir is not None or version_id is not None:
        if registry_dir is None or version_id is None:
            raise RuntimeError("--registry-dir and --version-id must be supplied together")
        registry_artifacts = [src_path, proto_dest, proj_dest, config_out_path, *evaluation_paths]
        if manifest_out_path is not None:
            registry_artifacts.append(manifest_out_path)
        registry_manifest = _write_registry_manifest(
            project_root=project_root,
            registry_dir=registry_dir,
            version_id=version_id,
            artifact_paths=registry_artifacts,
            metadata_csv_path=metadata_csv_path,
            approved_manifest_path=approved_manifest_path,
            config_template_path=config_template_path,
            training_config_path=training_config_path,
            features_path=features_path,
            features_provenance_path=features_provenance_path,
            checkpoint_path=checkpoint_path,
            init_projection_path=init_projection_path,
            training_manifest_path=training_manifest_path,
            checkpoint_selection=checkpoint_selection,
            numeric_label_contract=numeric_label_contract,
        )
        print(f"Wrote registry    → {registry_manifest}")

    # -------------------------------------------------------- summary
    print()
    print("=" * 60)
    print(f"  Export complete")
    print(f"  Classes  : {num_classes}")
    print(f"  Proto dim: {data['prototypes'].shape[1]}")
    print(f"  Examples : {class_names_list[:5]}{'...' if num_classes > 5 else ''}")
    print("=" * 60)
    print()
    if allow_runtime_write:
        print("Next: from codex_model import CodexClassifier; clf = CodexClassifier()")
    else:
        print("Next: inspect the candidate package and promote it with scripts/promote_model.py")
    return {
        "prototypes": proto_dest,
        "projection": proj_dest,
        "config": config_out_path,
        **({"manifest": manifest_out} if manifest_out is not None else {}),
        **({"registry_manifest": registry_manifest} if registry_manifest is not None else {}),
    }


def _write_registry_manifest(
    *,
    project_root: Path,
    registry_dir: Path,
    version_id: str,
    artifact_paths: list[Path],
    metadata_csv_path: Path | None,
    approved_manifest_path: Path | None,
    config_template_path: Path,
    training_config_path: Path | None,
    features_path: Path | None,
    features_provenance_path: Path | None,
    checkpoint_path: Path | None,
    init_projection_path: Path | None,
    training_manifest_path: Path | None,
    checkpoint_selection: str | None,
    numeric_label_contract: dict[str, Any],
) -> Path:
    repo_root = project_root.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from backend.services.model_registry import ModelRegistry  # noqa: WPS433

    registry = ModelRegistry(registry_dir, repo_root=repo_root, runtime_model_dir=project_root / "codex_model")
    artifact_paths = [path for path in artifact_paths if path is not None and path.is_file()]
    version_dir = registry.version_dir(version_id)
    for optional_artifact in (
        training_config_path,
        features_path,
        features_provenance_path,
        checkpoint_path,
        init_projection_path,
        training_manifest_path,
    ):
        if (
            optional_artifact is not None
            and optional_artifact.is_file()
            and _is_relative_to(optional_artifact, version_dir)
        ):
            artifact_paths.append(optional_artifact)
    artifact_paths = list(dict.fromkeys(path.resolve() for path in artifact_paths))
    snapshot = _snapshot_provenance(approved_manifest_path)
    metadata: dict[str, Any] = {
        "source": {
            "git_commit": registry.git_short(),
            "script": "backend/codex_pipeline/scripts/export_model.py",
        },
        "data": {
            "approved_export_manifest_path": str(approved_manifest_path) if approved_manifest_path else None,
            "approved_export_manifest_sha256": _optional_hash(approved_manifest_path),
            "metadata_csv_path": str(metadata_csv_path) if metadata_csv_path else None,
            "metadata_csv_sha256": _optional_hash(metadata_csv_path),
            "training_snapshot": snapshot,
            "features_cache": _optional_file_provenance(features_path, label="features cache"),
            "features_provenance": _optional_file_provenance(
                features_provenance_path, label="features provenance"
            ),
        },
        "training": {
            "command": "scripts/retrain.sh or scripts/retrain.ps1",
            "config_template": str(config_template_path),
            "config": _optional_file_provenance(training_config_path, label="training config"),
            "checkpoint": _optional_file_provenance(checkpoint_path, label="training checkpoint"),
            "checkpoint_selection": checkpoint_selection,
            "init_projection": _optional_file_provenance(
                init_projection_path, label="initial projection"
            ),
            "training_manifest": _optional_file_provenance(
                training_manifest_path, label="training manifest"
            ),
        },
        "numeric_label_contract": numeric_label_contract,
        "metrics": {"prototype_export": "completed"},
    }
    if snapshot is not None:
        metadata["promotion"] = {
            "blocked": True,
            "gate": "p4_locked_test_promotion_contract",
            "reason": (
                "P3 training-snapshot.v2 candidates are not promotable until P4 "
                "seals and validates a complete locked-test promotion contract"
            ),
        }
    registry.write_manifest(
        version_id,
        status="candidate",
        artifact_paths=artifact_paths,
        metadata=metadata,
    )
    return registry.version_dir(version_id) / "manifest.json"


def main() -> None:
    # ------------------------------------------------------------------ paths
    project_root = Path(__file__).resolve().parents[2]  # backend/
    parser = argparse.ArgumentParser(description="Export classifier artifacts for codex_model.")
    parser.add_argument(
        "--prototypes",
        default=str(project_root / "prototypes" / "prototypes.pt"),
        help="Trusted local full training prototype artifact to export.",
    )
    parser.add_argument(
        "--weights-dir",
        default=str(project_root / "codex_model" / "weights"),
        help="Destination runtime/weights directory. Runtime writes require --allow-runtime-write.",
    )
    parser.add_argument(
        "--config-template",
        default=str(project_root / "codex_model" / "config.json"),
        help="Baseline config.json to read before writing candidate config.",
    )
    parser.add_argument(
        "--config-out",
        default=None,
        help="Destination config.json to write. Defaults to --config-template.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Legacy alias for using the same config path as template and output.",
    )
    parser.add_argument(
        "--manifest-out",
        default=None,
        help="Optional JSON summary of exported artifacts and checksums.",
    )
    parser.add_argument(
        "--allow-runtime-write",
        action="store_true",
        help="Allow direct writes to backend/codex_model runtime artifacts (bootstrap only).",
    )
    parser.add_argument("--registry-dir", default=None, help="Optional model registry root for candidate manifest/index.")
    parser.add_argument("--version-id", default=None, help="Version ID to register when --registry-dir is supplied.")
    parser.add_argument("--evaluate-candidate", action="store_true", help="Compare exported weights on dev and locked_test before registering; seal both reports.")
    parser.add_argument("--metadata-csv", default=None, help="Approved metadata CSV path for registry provenance.")
    parser.add_argument(
        "--approved-manifest",
        default=None,
        help="Approved export manifest path for registry provenance.",
    )
    parser.add_argument(
        "--training-config",
        default=None,
        help="Optional training YAML recorded with its SHA-256.",
    )
    parser.add_argument(
        "--features",
        default=None,
        help="Optional cached features.pt recorded with its SHA-256.",
    )
    parser.add_argument(
        "--features-provenance",
        default=None,
        help="Optional features-cache provenance JSON recorded with its SHA-256.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional selected checkpoint recorded with its SHA-256.",
    )
    parser.add_argument(
        "--init-projection",
        default=None,
        help="Optional warm-start projection recorded with its SHA-256.",
    )
    parser.add_argument(
        "--base-model-dir",
        default=None,
        help="Base runtime package whose name-to-numeric-label ABI must be preserved.",
    )
    parser.add_argument(
        "--training-manifest",
        default=None,
        help="Optional training_manifest.json recorded with its SHA-256.",
    )
    parser.add_argument(
        "--checkpoint-selection",
        choices=("best", "latest"),
        default=None,
        help="Fixed checkpoint selection policy used by evaluation and export.",
    )
    args = parser.parse_args()
    config_template = Path(args.config or args.config_template)
    config_out = Path(args.config or args.config_out) if (args.config or args.config_out) else config_template

    try:
        export_model(
            src_path=Path(args.prototypes),
            weights_dir=Path(args.weights_dir),
            config_template_path=config_template,
            config_out_path=config_out,
            allow_runtime_write=args.allow_runtime_write,
            runtime_model_dir=project_root / "codex_model",
            base_model_dir=Path(args.base_model_dir) if args.base_model_dir else None,
            manifest_out_path=Path(args.manifest_out) if args.manifest_out else None,
            registry_dir=Path(args.registry_dir) if args.registry_dir else None,
            version_id=args.version_id,
            evaluate_candidate=args.evaluate_candidate,
            metadata_csv_path=Path(args.metadata_csv) if args.metadata_csv else None,
            approved_manifest_path=Path(args.approved_manifest) if args.approved_manifest else None,
            training_config_path=Path(args.training_config) if args.training_config else None,
            features_path=Path(args.features) if args.features else None,
            features_provenance_path=(
                Path(args.features_provenance) if args.features_provenance else None
            ),
            checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
            init_projection_path=(
                Path(args.init_projection) if args.init_projection else None
            ),
            training_manifest_path=(
                Path(args.training_manifest) if args.training_manifest else None
            ),
            checkpoint_selection=args.checkpoint_selection,
        )
    except (RuntimeWriteRefusedError, RuntimeError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()

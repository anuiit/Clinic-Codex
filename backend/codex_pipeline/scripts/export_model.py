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


def export_model(
    src_path: Path,
    weights_dir: Path,
    config_template_path: Path,
    *,
    config_out_path: Path | None = None,
    allow_runtime_write: bool = False,
    runtime_model_dir: Path | None = None,
    manifest_out_path: Path | None = None,
    registry_dir: Path | None = None,
    version_id: str | None = None,
    metadata_csv_path: Path | None = None,
    approved_manifest_path: Path | None = None,
) -> dict[str, Path]:
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
            "artifacts": {
                "prototypes": {"path": str(proto_dest), "sha256": _sha256_file(proto_dest)},
                "projection": {"path": str(proj_dest), "sha256": _sha256_file(proj_dest)},
                "config": {"path": str(config_out_path), "sha256": _sha256_file(config_out_path)},
            },
        }
        manifest_out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifest_out = manifest_out_path
        print(f"Wrote manifest    → {manifest_out_path}")

    registry_manifest: Path | None = None
    if registry_dir is not None or version_id is not None:
        if registry_dir is None or version_id is None:
            raise RuntimeError("--registry-dir and --version-id must be supplied together")
        registry_artifacts = [src_path, proto_dest, proj_dest, config_out_path]
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
) -> Path:
    repo_root = project_root.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from backend.services.model_registry import ModelRegistry  # noqa: WPS433

    registry = ModelRegistry(registry_dir, repo_root=repo_root, runtime_model_dir=project_root / "codex_model")
    artifact_paths = [path for path in artifact_paths if path is not None and path.is_file()]
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
        },
        "training": {
            "command": "scripts/retrain.sh or scripts/retrain.ps1",
            "config_template": str(config_template_path),
        },
        "metrics": {"prototype_export": "completed"},
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
    parser.add_argument("--metadata-csv", default=None, help="Approved metadata CSV path for registry provenance.")
    parser.add_argument(
        "--approved-manifest",
        default=None,
        help="Approved export manifest path for registry provenance.",
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
            manifest_out_path=Path(args.manifest_out) if args.manifest_out else None,
            registry_dir=Path(args.registry_dir) if args.registry_dir else None,
            version_id=args.version_id,
            metadata_csv_path=Path(args.metadata_csv) if args.metadata_csv else None,
            approved_manifest_path=Path(args.approved_manifest) if args.approved_manifest else None,
        )
    except (RuntimeWriteRefusedError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()

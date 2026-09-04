#!/usr/bin/env python3
"""Record local DINOv2 source and weights identities without vendoring them."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(root: Path) -> str:
    """Hash stable torch.hub source files, excluding generated metadata/caches."""
    digest = hashlib.sha256()
    excluded_dirs = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
    excluded_suffixes = {".pyc", ".pyo"}
    for path in sorted(
        candidate
        for candidate in root.rglob("*")
        if candidate.is_file()
        and not excluded_dirs.intersection(candidate.relative_to(root).parts)
        and candidate.suffix.lower() not in excluded_suffixes
    ):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()



# Fixed upstream revision and checkpoint; normal training never downloads either.
DINOV2_REVISION = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
DINOV2_WEIGHTS_SHA256 = "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"


def download_backbone() -> tuple[Path, Path]:
    import os
    import torch

    cache = Path(torch.hub.get_dir())
    weights = cache / "checkpoints" / "dinov2_vits14_pretrain.pth"
    weights.parent.mkdir(parents=True, exist_ok=True)
    if not weights.is_file() or sha256_file(weights) != DINOV2_WEIGHTS_SHA256:
        temporary = weights.with_name(f".{weights.name}.{os.getpid()}.tmp")
        try:
            torch.hub.download_url_to_file(
                "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth",
                str(temporary), hash_prefix=DINOV2_WEIGHTS_SHA256,
            )
            os.replace(temporary, weights)
        finally:
            temporary.unlink(missing_ok=True)
    if sha256_file(weights) != DINOV2_WEIGHTS_SHA256:
        raise ValueError("DINOv2 checkpoint checksum mismatch; no pin was created")
    torch.hub.load(
        f"facebookresearch/dinov2:{DINOV2_REVISION}", "dinov2_vits14",
        pretrained=False, trust_repo=True, skip_validation=True,
    )
    return cache / f"facebookresearch_dinov2_{DINOV2_REVISION}", weights


def load_backbone(backbone_name: str, device, backbone_manifest: str | Path | None = None):
    import torch

    if backbone_manifest is None:
        print("Loading DINOv2-S/14 backbone from torch.hub...")
        backbone = torch.hub.load(
            "facebookresearch/dinov2",
            backbone_name,
            pretrained=True,
        )
        return backbone.to(device).eval(), {
            "mode": "torch_hub_pretrained",
            "backbone": backbone_name,
        }

    manifest_path = Path(backbone_manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"backbone manifest must be a JSON object: {manifest_path}")
    if manifest.get("schema_version") != "dinov2-local-pin.v1":
        raise ValueError(f"backbone manifest has an unsupported schema: {manifest_path}")
    if manifest.get("backbone") != backbone_name:
        raise ValueError(
            f"backbone manifest targets {manifest.get('backbone')!r}, expected {backbone_name!r}"
        )

    repository_raw = manifest.get("repository_path")
    weights_raw = manifest.get("weights_path")
    if not isinstance(repository_raw, str) or not repository_raw:
        raise ValueError("backbone manifest repository_path must be a non-empty string")
    if not isinstance(weights_raw, str) or not weights_raw:
        raise ValueError("backbone manifest weights_path must be a non-empty string")
    repository_path = Path(repository_raw)
    weights_path = Path(weights_raw)
    if not repository_path.is_dir():
        raise FileNotFoundError(f"backbone repository path missing: {repository_path}")
    if not (repository_path / "hubconf.py").is_file():
        raise FileNotFoundError(f"backbone repository is not a torch.hub checkout: {repository_path}")
    if not weights_path.is_file():
        raise FileNotFoundError(f"backbone weights file missing: {weights_path}")

    expected_weights_sha = manifest.get("weights_sha256")
    actual_weights_sha = sha256_file(weights_path)
    if expected_weights_sha != actual_weights_sha:
        raise ValueError(
            "backbone weights checksum mismatch: "
            f"expected {expected_weights_sha}, got {actual_weights_sha}"
        )
    expected_source_sha = manifest.get("source_tree_sha256")
    actual_source_sha = sha256_tree(repository_path)
    if expected_source_sha != actual_source_sha:
        raise ValueError(
            "backbone source tree checksum mismatch: "
            f"expected {expected_source_sha}, got {actual_source_sha}"
        )

    print("Loading DINOv2-S/14 backbone from local pin...")
    backbone = torch.hub.load(str(repository_path), backbone_name, source="local", pretrained=False)
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    backbone.load_state_dict(state_dict, strict=True)
    backbone = backbone.to(device).eval()
    return backbone, {
        "mode": "local_pin",
        "backbone": backbone_name,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "repository_path": str(repository_path),
        "repository_sha256": actual_source_sha,
        "weights_path": str(weights_path),
        "weights_sha256": actual_weights_sha,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-path", type=Path)
    parser.add_argument("--weights-path", type=Path)
    parser.add_argument("--download", action="store_true", help="Install the fixed public DINOv2 base")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backbone", default="dinov2_vits14")
    args = parser.parse_args()

    if args.download:
        if args.repository_path or args.weights_path or args.backbone != "dinov2_vits14":
            parser.error("--download uses the fixed DINOv2-S/14 source and weights")
        repository, weights = download_backbone()
    else:
        if not args.repository_path or not args.weights_path:
            parser.error("provide --download or both --repository-path and --weights-path")
        repository, weights = args.repository_path, args.weights_path
    repository, weights = repository.resolve(), weights.resolve()
    if not (repository / "hubconf.py").is_file():
        raise SystemExit(f"repository is not a torch.hub checkout: {repository}")
    if not weights.is_file():
        raise SystemExit(f"weights file does not exist: {weights}")

    payload = {
        "schema_version": "dinov2-local-pin.v1",
        "backbone": args.backbone,
        "repository_path": str(repository),
        "weights_path": str(weights),
        "weights_sha256": sha256_file(weights),
        "source_tree_sha256": sha256_tree(repository),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Pinned {args.backbone}: {payload['weights_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

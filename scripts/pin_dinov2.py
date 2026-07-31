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
    """Hash the local torch.hub source excluding its mutable Git metadata."""
    digest = hashlib.sha256()
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file() and ".git" not in candidate.parts):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-path", type=Path, required=True)
    parser.add_argument("--weights-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backbone", default="dinov2_vits14")
    args = parser.parse_args()

    repository = args.repository_path.resolve()
    weights = args.weights_path.resolve()
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

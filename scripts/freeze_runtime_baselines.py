#!/usr/bin/env python3
"""Freeze the deployed white-pad and legacy gray-pad runtime contracts.

The command is deliberately read-only with respect to model weights.  It
records file and tensor digests, validates the sparse taxonomy/ABI, and proves
whether the two runtimes differ only by preprocessing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CURRENT_WEIGHTS = REPO_ROOT / "backend/codex_model/weights"
DEFAULT_LEGACY_WEIGHTS = (
    REPO_ROOT / "_legacy/frontend_integration/codex_model/weights"
)
DEFAULT_CURRENT_CLASSIFIER = REPO_ROOT / "backend/codex_model/classifier.py"
DEFAULT_LEGACY_CLASSIFIER = (
    REPO_ROOT / "_legacy/frontend_integration/codex_model/classifier.py"
)
EXPECTED_PROJECTION_SHAPES = {
    "net.0.weight": (384, 384),
    "net.0.bias": (384,),
    "net.3.weight": (128, 384),
    "net.3.bias": (128,),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape)).encode("ascii"))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def load_projection(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"projection is not a state dict: {path}")
    if set(payload) != set(EXPECTED_PROJECTION_SHAPES):
        raise ValueError(f"projection ABI keys differ: {path}")
    state: dict[str, torch.Tensor] = {}
    for key, expected_shape in EXPECTED_PROJECTION_SHAPES.items():
        tensor = payload[key]
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"projection entry {key!r} is not a tensor: {path}")
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"projection shape for {key!r} is {tuple(tensor.shape)}, "
                f"expected {expected_shape}: {path}"
            )
        state[key] = tensor.detach().cpu()
    return state


def load_prototypes(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"prototype payload is not a mapping: {path}")
    prototypes = payload.get("prototypes")
    class_names = payload.get("class_names")
    class_labels = payload.get("class_labels")
    if not isinstance(prototypes, torch.Tensor) or tuple(prototypes.shape) != (
        286,
        128,
    ):
        raise ValueError(f"expected prototypes with shape (286, 128): {path}")
    if not isinstance(class_names, dict) or len(class_names) != 286:
        raise ValueError(f"expected 286 class_names: {path}")
    if not isinstance(class_labels, torch.Tensor) or class_labels.dtype != torch.long:
        raise ValueError(f"expected sparse int64 class_labels: {path}")
    labels = [int(value) for value in class_labels.tolist()]
    names = {int(key): str(value) for key, value in class_names.items()}
    if labels != sorted(names) or len(labels) != 286:
        raise ValueError(f"class_labels do not match sorted class_names: {path}")
    return {
        "prototypes": prototypes.detach().cpu(),
        "class_names": names,
        "class_labels": class_labels.detach().cpu(),
    }


def tensors_equal(
    left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]
) -> bool:
    return set(left) == set(right) and all(
        torch.equal(left[key], right[key]) for key in left
    )


def assert_padding_contract(path: Path, color: tuple[int, int, int]) -> None:
    source = path.read_text(encoding="utf-8")
    literal = f"({color[0]}, {color[1]}, {color[2]})"
    if literal not in source:
        raise ValueError(f"expected padding color {literal} in {path}")


def build_manifest(
    current_weights: Path,
    legacy_weights: Path,
    current_classifier: Path,
    legacy_classifier: Path,
) -> dict[str, Any]:
    current_projection_path = current_weights / "projection.pt"
    legacy_projection_path = legacy_weights / "projection.pt"
    current_prototypes_path = current_weights / "prototypes.pt"
    legacy_prototypes_path = legacy_weights / "prototypes.pt"

    current_projection = load_projection(current_projection_path)
    legacy_projection = load_projection(legacy_projection_path)
    current_prototypes = load_prototypes(current_prototypes_path)
    legacy_prototypes = load_prototypes(legacy_prototypes_path)

    if not tensors_equal(current_projection, legacy_projection):
        raise ValueError("legacy and deployed projection tensors differ")
    if not torch.equal(
        current_prototypes["prototypes"], legacy_prototypes["prototypes"]
    ):
        raise ValueError("legacy and deployed prototype tensors differ")
    if current_prototypes["class_names"] != legacy_prototypes["class_names"]:
        raise ValueError("legacy and deployed class taxonomies differ")
    if not torch.equal(
        current_prototypes["class_labels"], legacy_prototypes["class_labels"]
    ):
        raise ValueError("legacy and deployed sparse class labels differ")

    assert_padding_contract(current_classifier, (255, 255, 255))
    assert_padding_contract(legacy_classifier, (128, 128, 128))

    projection_tensor_hashes = {
        key: tensor_sha256(current_projection[key])
        for key in sorted(current_projection)
    }
    shared = {
        "projection_tensor_sha256": projection_tensor_hashes,
        "prototypes_tensor_sha256": tensor_sha256(
            current_prototypes["prototypes"]
        ),
        "class_labels_sha256": tensor_sha256(
            current_prototypes["class_labels"]
        ),
        "class_count": 286,
        "feature_dim": 384,
        "embedding_dim": 128,
        "projection_abi": "384-384-128",
        "similarity": "cosine",
    }
    manifest: dict[str, Any] = {
        "schema_version": "codex-runtime-baselines.v1",
        "runtime_weights_tensor_equal": True,
        "behavioral_difference": "resize padding color only",
        "shared_contract": shared,
        "baselines": {
            "deployed_white": {
                "role": "product_reference",
                "padding_rgb": [255, 255, 255],
                "weights_dir": str(current_weights.resolve()),
                "classifier_path": str(current_classifier.resolve()),
                "projection_file_sha256": sha256_file(current_projection_path),
                "prototypes_file_sha256": sha256_file(current_prototypes_path),
            },
            "legacy_gray": {
                "role": "presentation_reference",
                "padding_rgb": [128, 128, 128],
                "weights_dir": str(legacy_weights.resolve()),
                "classifier_path": str(legacy_classifier.resolve()),
                "projection_file_sha256": sha256_file(legacy_projection_path),
                "prototypes_file_sha256": sha256_file(legacy_prototypes_path),
            },
        },
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--current-weights", type=Path, default=DEFAULT_CURRENT_WEIGHTS
    )
    parser.add_argument(
        "--legacy-weights", type=Path, default=DEFAULT_LEGACY_WEIGHTS
    )
    parser.add_argument(
        "--current-classifier", type=Path, default=DEFAULT_CURRENT_CLASSIFIER
    )
    parser.add_argument(
        "--legacy-classifier", type=Path, default=DEFAULT_LEGACY_CLASSIFIER
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = build_manifest(
        args.current_weights,
        args.legacy_weights,
        args.current_classifier,
        args.legacy_classifier,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

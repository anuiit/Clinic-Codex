#!/usr/bin/env python3
"""Filter a cached feature corpus to the paths present in another cache."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def filter_cache(
    input_path: Path,
    allowed_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    payload = torch.load(input_path, map_location="cpu", weights_only=False)
    allowed_payload = torch.load(
        allowed_path,
        map_location="cpu",
        weights_only=False,
    )
    source_paths = [str(path) for path in payload["image_paths"]]
    allowed_paths = {str(path) for path in allowed_payload["image_paths"]}
    mask = torch.tensor(
        [path in allowed_paths for path in source_paths],
        dtype=torch.bool,
    )
    indices = mask.nonzero(as_tuple=True)[0]
    selected_paths = {source_paths[index] for index in indices.tolist()}
    if selected_paths != allowed_paths:
        missing = sorted(allowed_paths - selected_paths)
        raise ValueError(f"{len(missing)} allowed paths are absent from input cache")

    filtered: dict[str, Any] = {}
    source_count = len(source_paths)
    for key, value in payload.items():
        if isinstance(value, torch.Tensor) and value.ndim > 0 and len(value) == source_count:
            filtered[key] = value[indices]
        elif isinstance(value, list) and len(value) == source_count:
            filtered[key] = [value[index] for index in indices.tolist()]
        else:
            filtered[key] = value

    filtered["filtered_by_paths_cache"] = str(allowed_path.resolve())
    filtered["input_cache_sha256"] = sha256_file(input_path)
    filtered["allowed_paths_cache_sha256"] = sha256_file(allowed_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(filtered, output_path)
    return {
        "source_feature_count": source_count,
        "allowed_path_count": len(allowed_paths),
        "filtered_feature_count": len(indices),
        "class_count": int(filtered["labels"].unique().numel()),
        "output": str(output_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-cache", required=True, type=Path)
    parser.add_argument("--allowed-paths-cache", required=True, type=Path)
    parser.add_argument("--output-cache", required=True, type=Path)
    args = parser.parse_args()
    print(
        filter_cache(
            args.input_cache,
            args.allowed_paths_cache,
            args.output_cache,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

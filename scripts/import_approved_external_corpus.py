#!/usr/bin/env python3
"""Materialize an approved external-corpus manifest as a reproducible Elements tree.

The source images remain untouched.  Every imported image is re-encoded to BMP
under a class directory whose numeric prefix is derived from the active runtime
taxonomy.  This preserves the runtime class-label ordering for a candidate
model while recording source and output hashes in a snapshot manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image
import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT_SCHEMA_VERSION = "external-corpus-training-snapshot.v1"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def runtime_class_labels(runtime_config: Path) -> dict[str, int]:
    config = load_json(runtime_config)
    classes = config.get("class_names")
    if not isinstance(classes, list) or not classes or not all(isinstance(name, str) and name for name in classes):
        raise ValueError(f"runtime config has no usable class_names list: {runtime_config}")
    if len(set(classes)) != len(classes):
        raise ValueError(f"runtime config contains duplicate class names: {runtime_config}")
    return {name: label for label, name in enumerate(classes)}


def imported_actions(approved_manifest: dict[str, Any], class_labels: dict[str, int]) -> list[dict[str, Any]]:
    if approved_manifest.get("ready_for_import") is not True:
        raise ValueError("approved manifest is not marked ready_for_import")
    actions = approved_manifest.get("actions")
    if not isinstance(actions, list):
        raise ValueError("approved manifest has no actions list")

    imports = [action for action in actions if action.get("action") == "import"]
    if not imports:
        raise ValueError("approved manifest contains no import actions")

    unknown = sorted({str(action.get("approved_class", "")) for action in imports} - set(class_labels))
    if unknown:
        raise ValueError(f"approved manifest imports classes absent from runtime taxonomy: {unknown}")
    missing = sorted(set(class_labels) - {str(action["approved_class"]) for action in imports})
    if missing:
        raise ValueError(
            "approved manifest does not retain every runtime class; "
            f"missing {len(missing)} classes: {', '.join(missing[:20])}"
        )

    seen_paths: set[str] = set()
    for action in imports:
        path = str(action.get("path", ""))
        expected_hash = str(action.get("sha256_bytes", ""))
        if not path or not expected_hash:
            raise ValueError(f"import action lacks path or source hash: {action!r}")
        if path in seen_paths:
            raise ValueError(f"duplicate import action path: {path}")
        seen_paths.add(path)
    return sorted(imports, key=lambda action: (class_labels[str(action["approved_class"])], str(action["path"])))


def class_directory(elements_dir: Path, label: int, class_name: str) -> Path:
    return elements_dir / f"{label + 1:04d}-{class_name}"


def materialize_snapshot(
    approved_manifest_path: Path,
    output_dir: Path,
    runtime_config: Path,
    training_config_template: Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    approved_manifest = load_json(approved_manifest_path)
    class_labels = runtime_class_labels(runtime_config)
    actions = imported_actions(approved_manifest, class_labels)
    class_counts = Counter(str(action["approved_class"]) for action in actions)
    with training_config_template.open(encoding="utf-8") as handle:
        training_config = yaml.safe_load(handle)
    if not isinstance(training_config, dict) or not isinstance(training_config.get("data"), dict):
        raise ValueError(f"training config has no data section: {training_config_template}")
    configured_minimum = int(training_config["data"].get("min_images_per_class", 1))
    retained_minimum = min(class_counts.values())
    if retained_minimum < configured_minimum:
        training_config["data"]["min_images_per_class"] = retained_minimum

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"output already exists (pass --overwrite to replace it): {output_dir}")
        shutil.rmtree(output_dir)

    elements_dir = output_dir / "Elements"
    elements_dir.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    try:
        for sequence, action in enumerate(actions, start=1):
            source = Path(str(action["path"])).expanduser()
            if not source.is_file():
                raise FileNotFoundError(f"approved source image is missing: {source}")
            source_hash = sha256_file(source)
            if source_hash != action["sha256_bytes"]:
                raise ValueError(f"approved source image changed since audit: {source}")

            class_name = str(action["approved_class"])
            label = class_labels[class_name]
            destination_dir = class_directory(elements_dir, label, class_name)
            destination_dir.mkdir(parents=True, exist_ok=True)
            destination = destination_dir / f"999_000_000-ext-{sequence:06d}-{source_hash}.bmp"
            temporary = destination.with_suffix(".tmp")
            with Image.open(source) as image:
                image.convert("RGB").save(temporary, format="BMP")
            temporary.replace(destination)
            rows.append(
                {
                    "class_name": class_name,
                    "class_label": label,
                    "source_path": str(source),
                    "source_sha256": source_hash,
                    "source_pixel_sha256": action.get("sha256_pixels"),
                    "output_path": str(destination),
                    "output_sha256": sha256_file(destination),
                    "original_status": action.get("original_status"),
                    "decision": action.get("decision"),
                    "reason": action.get("reason"),
                }
            )
    except Exception:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise

    counts = Counter(row["class_name"] for row in rows)
    training_config_path = output_dir / "training_config.yaml"
    training_config_path.write_text(yaml.safe_dump(training_config, sort_keys=False), encoding="utf-8")
    snapshot = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "ready_for_training": True,
        "approved_manifest_path": str(approved_manifest_path),
        "approved_manifest_sha256": sha256_file(approved_manifest_path),
        "approved_manifest_content_sha256": canonical_json_sha(approved_manifest),
        "runtime_config_path": str(runtime_config),
        "runtime_config_sha256": sha256_file(runtime_config),
        "training_config_template_path": str(training_config_template),
        "training_config_template_sha256": sha256_file(training_config_template),
        "training_config_path": str(training_config_path),
        "training_config_sha256": sha256_file(training_config_path),
        "configured_min_images_per_class": configured_minimum,
        "effective_min_images_per_class": int(training_config["data"]["min_images_per_class"]),
        "elements_dir": str(elements_dir),
        "taxonomy": list(class_labels),
        "class_count": len(class_labels),
        "image_count": len(rows),
        "class_counts": dict(sorted(counts.items())),
        "weak_classes_retained": sorted(approved_manifest.get("excluded_weak_classes") or []) == [],
        "rows": rows,
    }
    snapshot_path = output_dir / "import_snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return snapshot


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approved-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="New snapshot directory; contains Elements/ and import_snapshot.json")
    parser.add_argument("--runtime-config", type=Path, default=REPO_ROOT / "backend" / "codex_model" / "config.json")
    parser.add_argument("--training-config-template", type=Path, default=REPO_ROOT / "backend" / "codex_pipeline" / "config" / "default.yaml")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output directory")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        snapshot = materialize_snapshot(
            args.approved_manifest.resolve(),
            args.output.resolve(),
            args.runtime_config.resolve(),
            args.training_config_template.resolve(),
            overwrite=args.overwrite,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(f"Imported {snapshot['image_count']} images across {snapshot['class_count']} runtime classes")
        print(f"Snapshot: {args.output.resolve() / 'import_snapshot.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

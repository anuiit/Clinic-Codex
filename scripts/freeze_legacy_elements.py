#!/usr/bin/env python3
"""Freeze a legacy Elements ZIP into a normalized tree plus manifest JSON.

The tool validates the archive structure, keeps only BMP images under class
folders named ``NNNN-label``, normalizes labels to NFC, drops classes with fewer
than ``min_images_per_class`` images, and records per-file hashes plus split
metadata in a deterministic manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import unicodedata
import zipfile
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.codex_pipeline.data.class_order import load_runtime_class_order


SCHEMA_VERSION = "legacy-elements-freeze.v1"
DEFAULT_MIN_IMAGES_PER_CLASS = 2
DEFAULT_RUNTIME_CONFIG = ROOT / "backend" / "codex_model" / "config.json"
CLASS_DIR_RE = re.compile(r"^(?P<prefix>\d{4})-(?P<label>.+)$")
SOURCE_GROUP_RE = re.compile(r"^(?P<codex>[^_]+)_(?P<folio>[^_]+)_(?P<page>[^-_.]+)(?:[-_].*)?$")
IGNORED_FILENAMES = {".DS_Store"}
IGNORED_PREFIXES = ("__MACOSX/", "._")


@dataclass(frozen=True)
class ImageRecord:
    class_dir: str
    class_name: str
    class_index: int
    image_index: int
    archive_path: str
    output_path: str
    filename: str
    source_group: str
    size_bytes: int
    source_sha256: str
    output_sha256: str


@dataclass(frozen=True)
class ClassRecord:
    class_dir: str
    class_name: str
    class_index: int
    prefix: int
    source_class_dir: str
    source_prefix: int
    image_count: int
    source_group_count: int
    source_groups: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RejectedClassRecord:
    class_dir: str
    class_name: str
    prefix: int
    image_count: int
    reason: str


@dataclass
class FreezeReport:
    schema_version: str
    hash_algorithm: str
    source_zip: str
    source_zip_sha256: str
    output_dir: str
    elements_dir: str
    runtime_config: str
    min_images_per_class: int
    class_order_source: str
    class_order: list[str]
    class_order_sha256: str
    archive_entry_count: int
    ignored_entry_count: int
    class_dir_count: int
    kept_class_count: int
    rejected_class_count: int
    image_count: int
    filtered_image_count: int
    source_group_count: int
    classes: list[ClassRecord]
    rejected_classes: list[RejectedClassRecord]
    images: list[ImageRecord]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def normalize_nfc(value: str) -> str:
    return unicodedata.normalize("NFC", value)


def parse_class_dir(raw_name: str) -> tuple[int, str]:
    name = normalize_nfc(raw_name)
    match = CLASS_DIR_RE.fullmatch(name)
    if not match:
        raise ValueError(f"invalid class directory name (expected NNNN-label): {raw_name!r}")
    prefix = int(match.group("prefix"))
    label = normalize_nfc(match.group("label"))
    if not label:
        raise ValueError(f"class directory label is empty: {raw_name!r}")
    return prefix, label


def source_group_for_path(archive_path: str) -> str:
    normalized = archive_path.replace("\\", "/")
    filename = normalize_nfc(Path(normalized).name)
    stem = Path(filename).stem
    match = SOURCE_GROUP_RE.fullmatch(stem)
    if match:
        codex = normalize_nfc(match.group("codex"))
        folio = normalize_nfc(match.group("folio"))
        page = normalize_nfc(match.group("page"))
        return f"page:{codex}:{folio}:{page}"
    return f"image:{normalized}"


def _is_ignored_entry(archive_path: str) -> bool:
    normalized = archive_path.replace("\\", "/")
    if any(part == "__MACOSX" for part in normalized.split("/")):
        return True
    if Path(normalized).name in IGNORED_FILENAMES:
        return True
    return normalized.endswith("/._") or Path(normalized).name.startswith("._")


def _copy_zip_member(zip_file: zipfile.ZipFile, info: zipfile.ZipInfo, destination: Path) -> tuple[int, str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    with zip_file.open(info, "r") as src, destination.open("wb") as dst:
        for chunk in iter(lambda: src.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
            dst.write(chunk)
    return size, digest.hexdigest()


def freeze_legacy_elements(
    source_zip: Path,
    output_dir: Path,
    *,
    runtime_config: Path = DEFAULT_RUNTIME_CONFIG,
    min_images_per_class: int = DEFAULT_MIN_IMAGES_PER_CLASS,
    overwrite: bool = False,
) -> FreezeReport:
    if min_images_per_class < 1:
        raise ValueError("min_images_per_class must be >= 1")

    source_zip = source_zip.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not source_zip.is_file():
        raise FileNotFoundError(f"source zip does not exist: {source_zip}")
    source_zip_sha256 = sha256_file(source_zip)
    runtime_config = runtime_config.expanduser().resolve()
    if not runtime_config.is_file():
        raise FileNotFoundError(f"runtime config does not exist: {runtime_config}")

    class_buckets: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
    archive_entry_count = 0
    ignored_entry_count = 0

    with zipfile.ZipFile(source_zip, metadata_encoding="utf-8") as zip_file:
        for info in zip_file.infolist():
            if info.is_dir():
                continue
            archive_entry_count += 1
            archive_path = normalize_nfc(info.filename.replace("\\", "/"))
            if _is_ignored_entry(archive_path):
                ignored_entry_count += 1
                continue
            if not archive_path.startswith("Elements/"):
                raise ValueError(f"unexpected archive path outside Elements/: {archive_path!r}")
            parts = archive_path.split("/")
            if len(parts) != 3:
                raise ValueError(f"unexpected archive path depth (expected Elements/class/file.bmp): {archive_path!r}")
            class_dir = normalize_nfc(parts[1])
            filename = normalize_nfc(parts[2])
            if Path(filename).suffix.lower() != ".bmp":
                raise ValueError(f"non-BMP file found in Elements archive: {archive_path!r}")
            prefix, class_name = parse_class_dir(class_dir)
            bucket = class_buckets.setdefault(
                class_dir,
                {
                    "prefix": prefix,
                    "class_name": class_name,
                    "members": [],
                },
            )
            if bucket["class_name"] != class_name or bucket["prefix"] != prefix:
                raise ValueError(f"class directory normalized to conflicting values: {archive_path!r}")
            bucket["members"].append(
                {
                    "archive_path": archive_path,
                    "filename": filename,
                    "zip_info": info,
                    "source_group": source_group_for_path(archive_path),
                }
            )

        kept_class_entries: list[dict[str, Any]] = []
        rejected_classes: list[RejectedClassRecord] = []
        filtered_image_count = 0
        for class_dir, bucket in class_buckets.items():
            member_count = len(bucket["members"])
            if member_count < min_images_per_class:
                filtered_image_count += member_count
                rejected_classes.append(
                    RejectedClassRecord(
                        class_dir=class_dir,
                        class_name=bucket["class_name"],
                        prefix=bucket["prefix"],
                        image_count=member_count,
                        reason=f"below_min_images_per_class:{member_count}<{min_images_per_class}",
                    )
                )
                continue
            kept_class_entries.append(
                {
                    "class_dir": class_dir,
                    "class_name": bucket["class_name"],
                    "prefix": bucket["prefix"],
                    "members": bucket["members"],
                }
            )

        runtime_class_order = [normalize_nfc(name) for name in load_runtime_class_order(runtime_config)]
        if len(runtime_class_order) != len(set(runtime_class_order)):
            raise ValueError(f"runtime class order collides after NFC normalization: {runtime_config}")
        if len(runtime_class_order) != len(kept_class_entries):
            raise ValueError(
                "runtime class order does not match the frozen archive class count after NFC normalization; "
                f"runtime={len(runtime_class_order)} archive={len(kept_class_entries)}"
            )

        class_entry_by_name: dict[str, dict[str, Any]] = {}
        for entry in kept_class_entries:
            class_name = entry["class_name"]
            if class_name in class_entry_by_name:
                raise ValueError(f"duplicate archive class name after NFC normalization: {class_name!r}")
            class_entry_by_name[class_name] = entry

        runtime_class_name_set = set(runtime_class_order)
        missing_classes = [class_name for class_name in runtime_class_order if class_name not in class_entry_by_name]
        extra_classes = [class_name for class_name in class_entry_by_name if class_name not in runtime_class_name_set]
        if missing_classes or extra_classes:
            raise ValueError(
                "archive class set does not match the runtime contract after NFC normalization; "
                f"missing={missing_classes!r} extra={extra_classes!r}"
            )

        kept_class_entries = [class_entry_by_name[class_name] for class_name in runtime_class_order]
        class_order = runtime_class_order
        class_order_sha256 = canonical_json_sha(class_order)
        class_records: list[ClassRecord] = []
        image_records: list[ImageRecord] = []
        source_groups: set[str] = set()

        elements_dir = output_dir / "Elements"
        if output_dir.exists():
            if not overwrite:
                raise FileExistsError(f"output already exists (pass --overwrite to replace it): {output_dir}")
            shutil.rmtree(output_dir)
        elements_dir.mkdir(parents=True, exist_ok=True)

        try:
            for class_index, entry in enumerate(kept_class_entries, start=1):
                source_class_dir = entry["class_dir"]
                class_name = entry["class_name"]
                members = entry["members"]
                output_prefix = class_index
                output_class_dir = f"{output_prefix:04d}-{class_name}"
                destination_class_dir = elements_dir / output_class_dir
                destination_class_dir.mkdir(parents=True, exist_ok=True)
                class_source_groups: list[str] = []
                for image_index, member in enumerate(members, start=1):
                    destination_path = destination_class_dir / member["filename"]
                    size_bytes, source_sha256 = _copy_zip_member(zip_file, member["zip_info"], destination_path)
                    output_sha256 = sha256_file(destination_path)
                    if output_sha256 != source_sha256:
                        raise ValueError(f"output hash mismatch after copy: {destination_path}")
                    source_group = member["source_group"]
                    source_groups.add(source_group)
                    if source_group not in class_source_groups:
                        class_source_groups.append(source_group)
                    image_records.append(
                        ImageRecord(
                            class_dir=output_class_dir,
                            class_name=class_name,
                            class_index=class_index,
                            image_index=image_index,
                            archive_path=member["archive_path"],
                            output_path=str(destination_path),
                            filename=member["filename"],
                            source_group=source_group,
                            size_bytes=size_bytes,
                            source_sha256=source_sha256,
                            output_sha256=output_sha256,
                        )
                    )
                class_records.append(
                    ClassRecord(
                        class_dir=output_class_dir,
                        class_name=class_name,
                        class_index=class_index,
                        prefix=output_prefix,
                        source_class_dir=source_class_dir,
                        source_prefix=entry["prefix"],
                        image_count=len(members),
                        source_group_count=len(class_source_groups),
                        source_groups=class_source_groups,
                    )
                )
        except Exception:
            shutil.rmtree(output_dir, ignore_errors=True)
            raise

    report = FreezeReport(
        schema_version=SCHEMA_VERSION,
        hash_algorithm="sha256",
        source_zip=str(source_zip),
        source_zip_sha256=source_zip_sha256,
        output_dir=str(output_dir),
        elements_dir=str(elements_dir),
        runtime_config=str(runtime_config),
        min_images_per_class=min_images_per_class,
        class_order_source="runtime_config",
        class_order=class_order,
        class_order_sha256=class_order_sha256,
        archive_entry_count=archive_entry_count,
        ignored_entry_count=ignored_entry_count,
        class_dir_count=len(class_buckets),
        kept_class_count=len(class_records),
        rejected_class_count=len(rejected_classes),
        image_count=len(image_records),
        filtered_image_count=filtered_image_count,
        source_group_count=len(source_groups),
        classes=class_records,
        rejected_classes=rejected_classes,
        images=image_records,
    )
    manifest_path = output_dir / "legacy_elements_manifest.json"
    manifest_path.write_text(json.dumps(asdict(report), ensure_ascii=False, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return report


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_zip", type=Path, help="Path to the legacy Elements ZIP archive")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("backend/training_corpus/external/legacy-elements-frozen"),
        help="Directory that will receive Elements/ and legacy_elements_manifest.json",
    )
    parser.add_argument(
        "--runtime-config",
        type=Path,
        default=DEFAULT_RUNTIME_CONFIG,
        help="Path to the runtime config that defines the canonical class order",
    )
    parser.add_argument(
        "--min-images-per-class",
        type=int,
        default=DEFAULT_MIN_IMAGES_PER_CLASS,
        help="Drop classes with fewer than this many BMPs",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    try:
        report = freeze_legacy_elements(
            args.source_zip,
            args.output_dir,
            runtime_config=args.runtime_config,
            min_images_per_class=args.min_images_per_class,
            overwrite=args.overwrite,
        )
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(
        f"Frozen {report.kept_class_count} classes / {report.image_count} images "
        f"into {report.elements_dir} (manifest: {Path(report.output_dir) / 'legacy_elements_manifest.json'})"
    )
    print(
        f"Archive sha256={report.source_zip_sha256} class_order_sha256={report.class_order_sha256} "
        f"filtered_classes={report.rejected_class_count} filtered_images={report.filtered_image_count}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

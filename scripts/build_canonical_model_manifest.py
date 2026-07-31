#!/usr/bin/env python3
"""Build a deterministic canonical manifest from legacy and external sources.

The manifest merges one or more frozen legacy Elements manifests
(`legacy-elements-freeze.v1`) and/or approved external import snapshots
(`external-corpus-training-snapshot.v1`), deduplicates by pixel hash when
available and otherwise by content hash, preserves provenance, and emits a
stable train/dev/locked_test partition while keeping at least one train example
per class whenever the data makes that possible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, OrderedDict
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image, ImageOps


SCHEMA_VERSION = "canonical-model-manifest.v1"
SOURCE_SCHEMA_VERSIONS = {
    "legacy-elements-freeze.v1",
    "external-corpus-training-snapshot.v1",
}
DEFAULT_DEV_FRACTION = 0.15
DEFAULT_LOCKED_TEST_FRACTION = 0.15
PARTITION_ORDER = {"train": 0, "dev": 1, "locked_test": 2}
MANIFEST_KIND_ORDER = {
    "legacy-elements-freeze.v1": 0,
    "external-corpus-training-snapshot.v1": 1,
}


@dataclass(frozen=True)
class SourceManifestSummary:
    schema_version: str
    manifest_kind: str
    manifest_path: str
    manifest_sha256: str
    source_row_count: int
    class_count: int


@dataclass(frozen=True)
class CanonicalRow:
    row_index: int
    class_label: int
    class_name: str
    source_group: str
    source_group_source: str
    partition: str
    partition_reason: str
    manifest_kind: str
    manifest_path: str
    manifest_sha256: str
    manifest_row_index: int
    provenance_path: str
    provenance_kind: str
    source_sha256: str
    source_pixel_sha256: str
    output_sha256: str
    output_path: str
    dedup_sha256: str
    dedup_sha256_kind: str
    dedup_sha256_source: str
    duplicate_count: int


@dataclass(frozen=True)
class DuplicateRow:
    duplicate_index: int
    canonical_row_index: int
    class_label: int
    class_name: str
    source_group: str
    source_group_source: str
    manifest_kind: str
    manifest_path: str
    manifest_sha256: str
    manifest_row_index: int
    provenance_path: str
    provenance_kind: str
    source_sha256: str
    source_pixel_sha256: str
    output_sha256: str
    dedup_sha256: str
    dedup_sha256_kind: str
    dedup_sha256_source: str


@dataclass
class ManifestReport:
    schema_version: str
    hash_algorithm: str
    source_manifest_count: int
    source_row_count: int
    row_count: int
    duplicate_count: int
    class_count: int
    source_group_count: int
    partition_counts: dict[str, int]
    class_partition_counts: dict[str, dict[str, int]]
    source_manifests: list[SourceManifestSummary]
    rows: list[CanonicalRow]
    duplicates: list[DuplicateRow]
    partitions: dict[str, list[int]]
    signatures: dict[str, str]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json_sha(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def normalize_nfc(value: str) -> str:
    import unicodedata

    return unicodedata.normalize("NFC", value)


def normalize_source_group_text(value: str) -> str:
    text = normalize_nfc(value).strip()
    if not text:
        raise ValueError("source_group must not be empty")
    if text.startswith("page:"):
        return text.casefold()
    if text.startswith("image:"):
        prefix, suffix = text.split(":", 1)
        return f"{prefix.casefold()}:{suffix.casefold()}"
    return text.casefold()


def normalized_source_group(path: str | Path) -> str:
    canonical = str(path).replace("\\", "/")
    source = PurePosixPath(canonical)
    stem = source.stem
    import re

    stem = re.sub(r"-\d+$", "", stem)
    return f"{source.parent.as_posix().rstrip('/')}/{stem}".casefold()


def _require_mapping(value: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a JSON object")
    return value


def _require_string(value: Any, *, context: str) -> str:
    if value is None:
        raise ValueError(f"{context} is required")
    text = str(value)
    if not text.strip():
        raise ValueError(f"{context} must not be empty")
    return normalize_nfc(text)


def _require_int(value: Any, *, context: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must be an integer") from exc


def _require_optional_string(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return normalize_nfc(text) if text else ""


def load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return _require_mapping(payload, context=str(path))


def sha256_pixels(path: Path) -> str:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        width, height = image.size
        digest = hashlib.sha256()
        digest.update(f"RGB:{width}x{height}:".encode("ascii"))
        digest.update(image.tobytes())
        return digest.hexdigest()


def _manifest_kind(schema_version: str) -> str:
    if schema_version not in SOURCE_SCHEMA_VERSIONS:
        raise ValueError(f"unsupported source schema: {schema_version!r}")
    return schema_version


def _load_freeze_manifest(path: Path) -> tuple[SourceManifestSummary, list[dict[str, Any]]]:
    manifest = load_json_object(path)
    if manifest.get("schema_version") != "legacy-elements-freeze.v1":
        raise ValueError(f"unexpected freeze schema: {manifest.get('schema_version')!r}")
    images = manifest.get("images")
    classes = manifest.get("classes")
    if not isinstance(images, list) or not isinstance(classes, list):
        raise ValueError("freeze manifest must contain classes and images lists")

    class_names = {
        _require_int(record.get("class_index"), context="class_index"): _require_string(
            record.get("class_name"), context="class_name"
        )
        for record in classes
    }
    records: list[dict[str, Any]] = []
    for index, image in enumerate(images):
        record = _require_mapping(image, context="freeze image record")
        class_label = _require_int(record.get("class_index"), context="class_index")
        class_name = _require_string(record.get("class_name"), context="class_name")
        if class_names.get(class_label) != class_name:
            raise ValueError(f"freeze class metadata mismatch for {class_name!r}")
        output_path = Path(_require_string(record.get("output_path"), context="output_path"))
        source_group = normalize_source_group_text(_require_string(record.get("source_group"), context="source_group"))
        source_pixel_sha256 = sha256_pixels(output_path) if output_path.is_file() else ""
        records.append(
            {
                "class_label": class_label,
                "class_name": class_name,
                "source_group": source_group,
                "source_group_source": _require_string(record.get("archive_path"), context="archive_path"),
                "manifest_kind": "legacy-elements-freeze.v1",
                "manifest_path": str(path),
                "manifest_sha256": sha256_file(path),
                "manifest_row_index": index,
                "provenance_path": _require_string(record.get("archive_path"), context="archive_path"),
                "provenance_kind": "archive_path",
                "source_sha256": _require_string(record.get("source_sha256"), context="source_sha256"),
                "source_pixel_sha256": source_pixel_sha256,
                "output_sha256": _require_string(record.get("output_sha256"), context="output_sha256"),
                "output_path": str(output_path),
            }
        )

    return (
        SourceManifestSummary(
            schema_version=str(manifest.get("schema_version")),
            manifest_kind="legacy-elements-freeze.v1",
            manifest_path=str(path),
            manifest_sha256=sha256_file(path),
            source_row_count=len(records),
            class_count=len(classes),
        ),
        records,
    )


def _load_import_snapshot(path: Path) -> tuple[SourceManifestSummary, list[dict[str, Any]]]:
    snapshot = load_json_object(path)
    if snapshot.get("schema_version") != "external-corpus-training-snapshot.v1":
        raise ValueError(f"unexpected import snapshot schema: {snapshot.get('schema_version')!r}")
    rows = snapshot.get("rows")
    if not isinstance(rows, list):
        raise ValueError("import snapshot must contain a rows list")

    records: list[dict[str, Any]] = []
    class_names: set[str] = set()
    manifest_sha256 = sha256_file(path)
    for index, row in enumerate(rows):
        record = _require_mapping(row, context="snapshot row")
        class_label = _require_int(record.get("class_label"), context="class_label")
        class_name = _require_string(record.get("class_name"), context="class_name")
        source_path = Path(_require_string(record.get("source_path"), context="source_path"))
        source_pixel_sha256 = _require_optional_string(record.get("source_pixel_sha256"))
        output_path = Path(_require_string(record.get("output_path"), context="output_path"))
        source_sha256 = _require_string(record.get("source_sha256"), context="source_sha256")
        output_sha256 = _require_string(record.get("output_sha256"), context="output_sha256")
        if not source_pixel_sha256 and output_path.is_file():
            source_pixel_sha256 = sha256_pixels(output_path)
        source_group = normalized_source_group(source_path)
        records.append(
            {
                "class_label": class_label,
                "class_name": class_name,
                "source_group": source_group,
                "source_group_source": str(source_path),
                "manifest_kind": "external-corpus-training-snapshot.v1",
                "manifest_path": str(path),
                "manifest_sha256": manifest_sha256,
                "manifest_row_index": index,
                "provenance_path": str(source_path),
                "provenance_kind": "source_path",
                "source_sha256": source_sha256,
                "source_pixel_sha256": source_pixel_sha256,
                "output_sha256": output_sha256,
                "output_path": str(output_path),
            }
        )
        class_names.add(class_name)

    return (
        SourceManifestSummary(
            schema_version=str(snapshot.get("schema_version")),
            manifest_kind="external-corpus-training-snapshot.v1",
            manifest_path=str(path),
            manifest_sha256=manifest_sha256,
            source_row_count=len(records),
            class_count=len(class_names),
        ),
        records,
    )


def _preferred_dedup_hash(record: dict[str, Any]) -> tuple[str, str]:
    pixel_hash = _require_optional_string(record.get("source_pixel_sha256"))
    if pixel_hash:
        return pixel_hash, "pixel"
    source_sha256 = _require_optional_string(record.get("source_sha256"))
    if source_sha256:
        return source_sha256, "content"
    output_sha256 = _require_optional_string(record.get("output_sha256"))
    if output_sha256:
        return output_sha256, "content"
    raise ValueError("record lacks both pixel and content hashes")


def _row_sort_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        PARTITION_ORDER[record["partition"]],
        record["class_label"],
        record["source_group"],
        record["dedup_sha256"],
        record["provenance_path"],
        record["manifest_path"],
        record["manifest_row_index"],
    )


def _group_sort_key(group: dict[str, Any]) -> tuple[Any, ...]:
    return (
        group["source_group"],
        group["class_labels"],
        group["first_provenance_path"],
    )


def _can_hold_out(group: dict[str, Any], remaining_train_counts: Counter[str]) -> bool:
    for class_name, count in group["class_counts"].items():
        if remaining_train_counts[class_name] - count < 1:
            return False
    return True


def _choose_split(
    group: dict[str, Any],
    *,
    remaining_train_counts: Counter[str],
    split_row_counts: Counter[str],
    target_dev_rows: int,
    target_locked_rows: int,
) -> str:
    if not _can_hold_out(group, remaining_train_counts):
        return "train"

    dev_deficit = max(0, target_dev_rows - split_row_counts["dev"])
    locked_deficit = max(0, target_locked_rows - split_row_counts["locked_test"])
    if dev_deficit == 0 and locked_deficit == 0:
        return "train"
    if dev_deficit > locked_deficit:
        return "dev"
    if locked_deficit > dev_deficit:
        return "locked_test"
    return "locked_test" if (hashlib.sha256(group["source_group"].encode("utf-8")).digest()[0] % 2) else "dev"


def _move_group(
    group: dict[str, Any],
    *,
    target_split: str,
    assignments: dict[str, str],
    remaining_train_counts: Counter[str],
    split_row_counts: Counter[str],
) -> bool:
    if assignments[group["source_group"]] != "train":
        return False
    if not _can_hold_out(group, remaining_train_counts):
        return False
    assignments[group["source_group"]] = target_split
    split_row_counts["train"] -= len(group["rows"])
    split_row_counts[target_split] += len(group["rows"])
    for class_name, count in group["class_counts"].items():
        remaining_train_counts[class_name] -= count
    return True


def _assign_partitions(rows: list[dict[str, Any]], dev_fraction: float, locked_test_fraction: float) -> tuple[list[dict[str, Any]], dict[str, list[int]], dict[str, int], dict[str, dict[str, int]]]:
    if not rows:
        raise ValueError("no canonical rows available after merging inputs")
    if not 0 <= dev_fraction < 1:
        raise ValueError("dev_fraction must be in [0, 1)")
    if not 0 <= locked_test_fraction < 1:
        raise ValueError("locked_test_fraction must be in [0, 1)")
    if dev_fraction + locked_test_fraction >= 1:
        raise ValueError("dev_fraction + locked_test_fraction must be < 1")

    group_buckets: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
    for row in rows:
        group = group_buckets.setdefault(
            row["source_group"],
            {
                "source_group": row["source_group"],
                "rows": [],
                "class_counts": Counter(),
                "class_labels": set(),
                "first_provenance_path": row["provenance_path"],
            },
        )
        group["rows"].append(row)
        group["class_counts"][row["class_name"]] += 1
        group["class_labels"].add(row["class_label"])
        if row["provenance_path"] < group["first_provenance_path"]:
            group["first_provenance_path"] = row["provenance_path"]

    group_entries = sorted(
        (
            {
                **group,
                "class_labels": tuple(sorted(group["class_labels"])),
            }
            for group in group_buckets.values()
        ),
        key=_group_sort_key,
    )

    total_rows = len(rows)
    target_dev_rows = max(1, int(total_rows * dev_fraction)) if total_rows >= 3 and dev_fraction > 0 else 0
    target_locked_rows = max(1, int(total_rows * locked_test_fraction)) if total_rows >= 3 and locked_test_fraction > 0 else 0

    remaining_train_counts = Counter(row["class_name"] for row in rows)
    split_row_counts: Counter[str] = Counter({"train": total_rows, "dev": 0, "locked_test": 0})
    assignments: dict[str, str] = {group["source_group"]: "train" for group in group_entries}

    for group in group_entries:
        split = _choose_split(
            group,
            remaining_train_counts=remaining_train_counts,
            split_row_counts=split_row_counts,
            target_dev_rows=target_dev_rows,
            target_locked_rows=target_locked_rows,
        )
        if split != "train":
            _move_group(
                group,
                target_split=split,
                assignments=assignments,
                remaining_train_counts=remaining_train_counts,
                split_row_counts=split_row_counts,
            )

    for target_split, target_rows in (("dev", target_dev_rows), ("locked_test", target_locked_rows)):
        if target_rows == 0 or split_row_counts[target_split] > 0:
            continue
        for group in group_entries:
            if _move_group(
                group,
                target_split=target_split,
                assignments=assignments,
                remaining_train_counts=remaining_train_counts,
                split_row_counts=split_row_counts,
            ):
                break

    canonical_rows: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    seen_hashes: dict[str, dict[str, Any]] = {}

    for group in group_entries:
        for row in group["rows"]:
            row = dict(row)
            row["partition"] = assignments[row["source_group"]]
            row["partition_reason"] = "group_assignment"
            row["dedup_sha256"], row["dedup_sha256_kind"] = _preferred_dedup_hash(row)
            row["dedup_sha256_source"] = "source_pixel_sha256" if row["dedup_sha256_kind"] == "pixel" else (
                "source_sha256" if row["source_sha256"] == row["dedup_sha256"] else "output_sha256"
            )
            row["duplicate_count"] = 0
            row["canonical_candidate_key"] = (
                0 if row["dedup_sha256_kind"] == "pixel" else 1,
                row["provenance_path"],
                row["manifest_path"],
                row["manifest_row_index"],
                row["class_label"],
                row["output_path"],
            )
            canonical_rows.append(row)

    clusters: "OrderedDict[str, list[dict[str, Any]]]" = OrderedDict()
    for row in canonical_rows:
        clusters.setdefault(row["dedup_sha256"], []).append(row)

    selected_rows: list[dict[str, Any]] = []
    duplicate_rows: list[dict[str, Any]] = []
    duplicate_index = 0
    for dedup_hash, cluster in clusters.items():
        class_names = {row["class_name"] for row in cluster}
        if len(class_names) > 1:
            raise ValueError(f"dedup hash maps to multiple classes: {dedup_hash}")
        canonical = sorted(cluster, key=lambda row: row["canonical_candidate_key"])[0]
        canonical = dict(canonical)
        canonical.pop("canonical_candidate_key", None)
        canonical["duplicate_count"] = len(cluster) - 1
        canonical["row_index"] = -1
        selected_rows.append(canonical)
        for row in sorted(cluster, key=lambda item: item["canonical_candidate_key"])[1:]:
            duplicate_rows.append(
                {
                    "duplicate_index": duplicate_index,
                    "canonical_row_index": -1,
                    "class_label": row["class_label"],
                    "class_name": row["class_name"],
                    "source_group": row["source_group"],
                    "source_group_source": row["source_group_source"],
                    "manifest_kind": row["manifest_kind"],
                    "manifest_path": row["manifest_path"],
                    "manifest_sha256": row["manifest_sha256"],
                    "manifest_row_index": row["manifest_row_index"],
                    "provenance_path": row["provenance_path"],
                    "provenance_kind": row["provenance_kind"],
                    "source_sha256": row["source_sha256"],
                    "source_pixel_sha256": row["source_pixel_sha256"],
                    "output_sha256": row["output_sha256"],
                    "dedup_sha256": row["dedup_sha256"],
                    "dedup_sha256_kind": row["dedup_sha256_kind"],
                    "dedup_sha256_source": row["dedup_sha256_source"],
                }
            )
            duplicate_index += 1

    selected_rows.sort(key=_row_sort_key)
    for index, row in enumerate(selected_rows):
        row["row_index"] = index

    row_index_by_key = {
        (
            row["dedup_sha256"],
            row["manifest_path"],
            row["manifest_row_index"],
            row["provenance_path"],
        ): row["row_index"]
        for row in selected_rows
    }

    for duplicate in duplicate_rows:
        key = (
            duplicate["dedup_sha256"],
            duplicate["manifest_path"],
            duplicate["manifest_row_index"],
            duplicate["provenance_path"],
        )
        duplicate["canonical_row_index"] = row_index_by_key[
            min(
                (
                    k
                    for k in row_index_by_key
                    if k[0] == duplicate["dedup_sha256"]
                ),
                key=lambda item: row_index_by_key[item],
            )
        ]

    duplicate_rows.sort(
        key=lambda row: (
            row["dedup_sha256"],
            row["class_label"],
            row["source_group"],
            row["provenance_path"],
            row["manifest_path"],
            row["manifest_row_index"],
        )
    )
    for index, row in enumerate(duplicate_rows):
        row["duplicate_index"] = index

    partitions: dict[str, list[int]] = {name: [] for name in ("train", "dev", "locked_test")}
    partition_counts = {name: 0 for name in ("train", "dev", "locked_test")}
    class_partition_counts: dict[str, Counter[str]] = {}

    for row in selected_rows:
        partitions[row["partition"]].append(row["row_index"])
        partition_counts[row["partition"]] += 1
        class_partition_counts.setdefault(row["class_name"], Counter())[row["partition"]] += 1

    class_partition_counts_payload = {
        class_name: {split: counts.get(split, 0) for split in ("train", "dev", "locked_test")}
        for class_name, counts in sorted(class_partition_counts.items())
    }

    return selected_rows, partitions, partition_counts, class_partition_counts_payload, duplicate_rows


def build_canonical_model_manifest(
    *,
    freeze_manifests: list[Path] | None = None,
    import_snapshots: list[Path] | None = None,
    output: Path,
    dev_fraction: float = DEFAULT_DEV_FRACTION,
    locked_test_fraction: float = DEFAULT_LOCKED_TEST_FRACTION,
    overwrite: bool = False,
) -> ManifestReport:
    freeze_manifests = freeze_manifests or []
    import_snapshots = import_snapshots or []
    if not freeze_manifests and not import_snapshots:
        raise ValueError("at least one manifest must be provided")

    source_inputs = sorted(
        [path.expanduser().resolve() for path in [*freeze_manifests, *import_snapshots]],
        key=lambda path: str(path),
    )

    summaries: list[SourceManifestSummary] = []
    raw_records: list[dict[str, Any]] = []
    for path in source_inputs:
        payload = load_json_object(path)
        schema_version = str(payload.get("schema_version", ""))
        if schema_version == "legacy-elements-freeze.v1":
            summary, records = _load_freeze_manifest(path)
        elif schema_version == "external-corpus-training-snapshot.v1":
            summary, records = _load_import_snapshot(path)
        else:
            raise ValueError(f"unsupported manifest schema: {schema_version!r} ({path})")
        summaries.append(summary)
        raw_records.extend(records)

    source_row_count = len(raw_records)

    dedup_buckets: "OrderedDict[str, list[dict[str, Any]]]" = OrderedDict()
    for record in raw_records:
        dedup_hash, dedup_kind = _preferred_dedup_hash(record)
        normalized = dict(record)
        normalized["dedup_sha256"] = dedup_hash
        normalized["dedup_sha256_kind"] = dedup_kind
        normalized["dedup_sha256_source"] = (
            "source_pixel_sha256"
            if dedup_kind == "pixel"
            else ("source_sha256" if normalized["source_sha256"] == dedup_hash else "output_sha256")
        )
        dedup_buckets.setdefault(dedup_hash, []).append(normalized)

    canonical_rows_input: list[dict[str, Any]] = []
    duplicate_rows_input: list[dict[str, Any]] = []
    duplicate_index = 0
    for dedup_hash, cluster in dedup_buckets.items():
        class_names = {row["class_name"] for row in cluster}
        if len(class_names) > 1:
            raise ValueError(f"dedup hash maps to multiple classes: {dedup_hash}")
        canonical = sorted(
            cluster,
            key=lambda row: (
                0 if row["dedup_sha256_kind"] == "pixel" else 1,
                row["provenance_path"],
                row["manifest_path"],
                row["manifest_row_index"],
                row["class_label"],
                row["output_path"],
            ),
        )[0]
        canonical["duplicate_count"] = len(cluster) - 1
        canonical_rows_input.append(canonical)
        for row in sorted(
            cluster,
            key=lambda item: (
                0 if item["dedup_sha256_kind"] == "pixel" else 1,
                item["provenance_path"],
                item["manifest_path"],
                item["manifest_row_index"],
                item["class_label"],
                item["output_path"],
            ),
        )[1:]:
            duplicate_rows_input.append(
                {
                    "duplicate_index": duplicate_index,
                    "canonical_row_index": -1,
                    "class_label": row["class_label"],
                    "class_name": row["class_name"],
                    "source_group": row["source_group"],
                    "source_group_source": row["source_group_source"],
                    "manifest_kind": row["manifest_kind"],
                    "manifest_path": row["manifest_path"],
                    "manifest_sha256": row["manifest_sha256"],
                    "manifest_row_index": row["manifest_row_index"],
                    "provenance_path": row["provenance_path"],
                    "provenance_kind": row["provenance_kind"],
                    "source_sha256": row["source_sha256"],
                    "source_pixel_sha256": row["source_pixel_sha256"],
                    "output_sha256": row["output_sha256"],
                    "dedup_sha256": row["dedup_sha256"],
                    "dedup_sha256_kind": row["dedup_sha256_kind"],
                    "dedup_sha256_source": row["dedup_sha256_source"],
                }
            )
            duplicate_index += 1

    selected_rows, partitions, partition_counts, class_partition_counts, _ = _assign_partitions(
        canonical_rows_input,
        dev_fraction=dev_fraction,
        locked_test_fraction=locked_test_fraction,
    )

    row_index_by_key = {
        (
            row["dedup_sha256"],
            row["manifest_path"],
            row["manifest_row_index"],
            row["provenance_path"],
        ): row["row_index"]
        for row in selected_rows
    }

    for duplicate in duplicate_rows_input:
        matching_keys = sorted(
            [
                key
                for key in row_index_by_key
                if key[0] == duplicate["dedup_sha256"]
            ],
            key=lambda item: row_index_by_key[item],
        )
        duplicate["canonical_row_index"] = row_index_by_key[matching_keys[0]]

    duplicate_rows_input.sort(
        key=lambda row: (
            row["dedup_sha256"],
            row["class_label"],
            row["source_group"],
            row["provenance_path"],
            row["manifest_path"],
            row["manifest_row_index"],
        )
    )
    for index, row in enumerate(duplicate_rows_input):
        row["duplicate_index"] = index

    source_group_count = len({row["source_group"] for row in selected_rows})
    signatures = {
        "source_manifests_sha256": canonical_json_sha([asdict(summary) for summary in summaries]),
        "rows_sha256": canonical_json_sha(selected_rows),
        "duplicates_sha256": canonical_json_sha(duplicate_rows_input),
        "partitions_sha256": canonical_json_sha(partitions),
    }
    manifest_without_signature = {
        "schema_version": SCHEMA_VERSION,
        "hash_algorithm": "sha256",
        "source_manifest_count": len(summaries),
        "source_row_count": source_row_count,
        "row_count": len(selected_rows),
        "duplicate_count": len(duplicate_rows_input),
        "class_count": len(class_partition_counts),
        "source_group_count": source_group_count,
        "partition_counts": partition_counts,
        "class_partition_counts": class_partition_counts,
        "source_manifests": [asdict(summary) for summary in summaries],
        "rows": selected_rows,
        "duplicates": duplicate_rows_input,
        "partitions": partitions,
        "signatures": {**signatures, "manifest_sha256": ""},
    }
    signatures["manifest_sha256"] = canonical_json_sha(manifest_without_signature)
    manifest = dict(manifest_without_signature)
    manifest["signatures"] = signatures

    output = output.expanduser().resolve()
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"output already exists (pass --overwrite to replace it): {output}")
        if output.is_dir():
            raise IsADirectoryError(f"output must be a JSON file, got directory: {output}")
        output.unlink()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return ManifestReport(
        schema_version=SCHEMA_VERSION,
        hash_algorithm="sha256",
        source_manifest_count=len(summaries),
        source_row_count=source_row_count,
        row_count=len(selected_rows),
        duplicate_count=len(duplicate_rows_input),
        class_count=len(class_partition_counts),
        source_group_count=source_group_count,
        partition_counts=partition_counts,
        class_partition_counts=class_partition_counts,
        source_manifests=summaries,
        rows=[CanonicalRow(**row) for row in selected_rows],
        duplicates=[DuplicateRow(**row) for row in duplicate_rows_input],
        partitions=partitions,
        signatures=signatures,
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--freeze-manifest",
        action="append",
        type=Path,
        default=[],
        help="Path to a legacy-elements-freeze.v1 manifest (repeatable)",
    )
    parser.add_argument(
        "--import-snapshot",
        action="append",
        type=Path,
        default=[],
        help="Path to an external-corpus-training-snapshot.v1 import snapshot (repeatable)",
    )
    parser.add_argument("--output", required=True, type=Path, help="Output JSON manifest path")
    parser.add_argument("--dev-fraction", type=float, default=DEFAULT_DEV_FRACTION)
    parser.add_argument("--locked-test-fraction", type=float, default=DEFAULT_LOCKED_TEST_FRACTION)
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output file")
    parser.add_argument("--json", action="store_true", help="Print the manifest JSON to stdout")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    try:
        report = build_canonical_model_manifest(
            freeze_manifests=args.freeze_manifest,
            import_snapshots=args.import_snapshot,
            output=args.output,
            dev_fraction=args.dev_fraction,
            locked_test_fraction=args.locked_test_fraction,
            overwrite=args.overwrite,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.json:
        payload = {
            "schema_version": report.schema_version,
            "hash_algorithm": report.hash_algorithm,
            "source_manifest_count": report.source_manifest_count,
            "source_row_count": report.source_row_count,
            "row_count": report.row_count,
            "duplicate_count": report.duplicate_count,
            "class_count": report.class_count,
            "source_group_count": report.source_group_count,
            "partition_counts": report.partition_counts,
            "class_partition_counts": report.class_partition_counts,
            "source_manifests": [asdict(summary) for summary in report.source_manifests],
            "rows": [asdict(row) for row in report.rows],
            "duplicates": [asdict(row) for row in report.duplicates],
            "partitions": report.partitions,
            "signatures": report.signatures,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(
            f"Wrote {report.row_count} canonical rows (+{report.duplicate_count} duplicates) "
            f"to {args.output.resolve()}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

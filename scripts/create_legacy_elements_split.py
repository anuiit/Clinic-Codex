#!/usr/bin/env python3
"""Create a deterministic train/val/test split manifest from frozen Elements.

The input is the immutable manifest written by ``freeze_legacy_elements.py``.
Rows stay grouped by ``source_group`` and byte-identical ``output_sha256`` so
every member of a page-level or duplicate-connected group is assigned to the
same dataset split. The splitter preserves at least one train example for every
class, keeps the held-out test split nonempty when possible, and records
reproducibility hashes plus summary statistics in the output JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import Counter, OrderedDict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


SOURCE_SCHEMA_VERSION = "legacy-elements-freeze.v1"
SCHEMA_VERSION = "legacy-elements-split.v1"
DEFAULT_VAL_FRACTION = 0.15
DEFAULT_TEST_FRACTION = 0.15
TRAIN_SPLIT = "train"
VAL_SPLIT = "val"
TEST_SPLIT = "test"
DEFAULT_OUTPUT_DIR = Path("backend/training_corpus/external/legacy-elements-split")


@dataclass(frozen=True)
class SplitRow:
    index: int
    dataset_split: st
    split_reason: st
    class_label: int
    class_name: st
    class_dir: st
    image_index: int
    image_count: int
    source_group: st
    output_path: st
    source_sha256: st
    output_sha256: st


@dataclass(frozen=True)
class GroupRecord:
    group_index: int
    source_group: st
    dataset_split: st
    split_reason: st
    image_count: int
    class_count: int
    class_names: list[str] = field(default_factory=list)
    member_paths: list[str] = field(default_factory=list)
    member_sha256: str = ""


@dataclass
class SplitReport:
    schema_version: st
    hash_algorithm: st
    source_manifest: st
    source_manifest_sha256: st
    source_manifest_schema_version: st
    output_dir: st
    split_manifest: st
    val_fraction: float
    test_fraction: float
    class_count: int
    image_count: int
    source_group_count: int
    train_image_count: int
    val_image_count: int
    test_image_count: int
    train_group_count: int
    val_group_count: int
    test_group_count: int
    class_counts: dict[str, int]
    class_split_counts: dict[str, dict[str, int]]
    split_counts: dict[str, int]
    group_counts: dict[str, int]
    rows_sha256: st
    group_assignments_sha256: st
    groups: list[GroupRecord]
    rows: list[SplitRow]


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
    import unicodedata

    return unicodedata.normalize("NFC", value)


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


def load_frozen_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    manifest = _require_mapping(payload, context="frozen manifest")
    if manifest.get("schema_version") != SOURCE_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported frozen manifest schema: {manifest.get('schema_version')!r} "
            f"(expected {SOURCE_SCHEMA_VERSION!r})"
        )
    if not isinstance(manifest.get("classes"), list):
        raise ValueError("frozen manifest must contain a classes list")
    if not isinstance(manifest.get("images"), list):
        raise ValueError("frozen manifest must contain an images list")
    return manifest


def _group_sort_key(group: dict[str, Any]) -> tuple[Any, ...]:
    class_indices = tuple(sorted(group["class_indices"]))
    first_path = group["member_paths"][0] if group["member_paths"] else ""
    return (class_indices, group["source_group"], first_path)


def _merge_duplicate_connected_groups(
    group_buckets: "OrderedDict[str, dict[str, Any]]",
) -> "OrderedDict[str, dict[str, Any]]":
    parent = {source_group: source_group for source_group in group_buckets}

    def find(source_group: str) -> str:
        while parent[source_group] != source_group:
            parent[source_group] = parent[parent[source_group]]
            source_group = parent[source_group]
        return source_group

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        canonical_root, other_root = sorted((left_root, right_root))
        parent[other_root] = canonical_root

    first_group_for_hash: dict[str, str] = {}
    for source_group, group in group_buckets.items():
        for member in group["members"]:
            digest = member["output_sha256"]
            duplicate_group = first_group_for_hash.setdefault(digest, source_group)
            union(source_group, duplicate_group)

    merged: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
    for source_group, group in group_buckets.items():
        root = find(source_group)
        target = merged.setdefault(
            root,
            {
                "source_group": root,
                "members": [],
                "class_counts": Counter(),
                "class_indices": set(),
                "class_names": [],
                "member_paths": [],
            },
        )
        target["members"].extend(group["members"])
        target["class_counts"].update(group["class_counts"])
        target["class_indices"].update(group["class_indices"])
        for class_name in group["class_names"]:
            if class_name not in target["class_names"]:
                target["class_names"].append(class_name)
        target["member_paths"].extend(group["member_paths"])

    return merged


def _can_hold_out(group: dict[str, Any], remaining_train_counts: Counter[str]) -> bool:
    for class_name, count in group["class_counts"].items():
        if remaining_train_counts[class_name] - count < 1:
            return False
    return True


def _choose_split(
    group: dict[str, Any],
    *,
    remaining_train_counts: Counter[str],
    split_image_counts: Counter[str],
    target_test_images: int,
    target_val_images: int,
) -> tuple[str, str]:
    if not _can_hold_out(group, remaining_train_counts):
        return TRAIN_SPLIT, "train_coverage"

    test_deficit = max(0, target_test_images - split_image_counts[TEST_SPLIT])
    val_deficit = max(0, target_val_images - split_image_counts[VAL_SPLIT])

    if split_image_counts[TEST_SPLIT] == 0 and test_deficit > 0:
        return TEST_SPLIT, "held_out_test"
    if split_image_counts[VAL_SPLIT] == 0 and val_deficit > 0 and test_deficit == 0:
        return VAL_SPLIT, "held_out_val"
    if test_deficit > val_deficit and test_deficit > 0:
        return TEST_SPLIT, "held_out_test"
    if val_deficit > test_deficit and val_deficit > 0:
        return VAL_SPLIT, "held_out_val"
    if test_deficit > 0:
        return TEST_SPLIT, "held_out_test"
    if val_deficit > 0:
        return VAL_SPLIT, "held_out_val"
    return TRAIN_SPLIT, "train_reserve"


def create_legacy_elements_split(
    source_manifest: Path,
    output_dir: Path,
    *,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    overwrite: bool = False,
) -> SplitReport:
    if not 0 <= val_fraction < 1:
        raise ValueError("val_fraction must be in [0, 1)")
    if not 0 <= test_fraction < 1:
        raise ValueError("test_fraction must be in [0, 1)")
    if val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction + test_fraction must be < 1")

    source_manifest = source_manifest.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not source_manifest.is_file():
        raise FileNotFoundError(f"source manifest does not exist: {source_manifest}")

    manifest = load_frozen_manifest(source_manifest)
    source_manifest_sha256 = sha256_file(source_manifest)

    class_lookup: dict[str, dict[str, Any]] = {}
    class_order: list[str] = []
    for class_record in manifest["classes"]:
        record = _require_mapping(class_record, context="class record")
        class_name = _require_string(record.get("class_name"), context="class_name")
        class_lookup[class_name] = {
            "class_index": _require_int(record.get("class_index"), context=f"class_index for {class_name}"),
            "class_dir": _require_string(record.get("class_dir"), context=f"class_dir for {class_name}"),
            "image_count": _require_int(record.get("image_count"), context=f"image_count for {class_name}"),
        }
        class_order.append(class_name)

    images = manifest["images"]
    class_image_counts: Counter[str] = Counter()
    group_buckets: "OrderedDict[str, dict[str, Any]]" = OrderedDict()

    for image in images:
        record = _require_mapping(image, context="image record")
        class_name = _require_string(record.get("class_name"), context="image class_name")
        source_group = _require_string(record.get("source_group"), context=f"source_group for {class_name}")
        class_info = class_lookup.get(class_name)
        if class_info is None:
            raise ValueError(f"image references unknown class_name: {class_name}")

        class_dir = _require_string(record.get("class_dir"), context=f"class_dir for {class_name}")
        class_label = _require_int(record.get("class_index"), context=f"class_index for {class_name}")
        if class_label != class_info["class_index"]:
            raise ValueError(
                f"image class_index mismatch for {class_name}: {class_label} != {class_info['class_index']}"
            )
        if class_dir != class_info["class_dir"]:
            raise ValueError(
                f"image class_dir mismatch for {class_name}: {class_dir!r} != {class_info['class_dir']!r}"
            )

        group = group_buckets.setdefault(
            source_group,
            {
                "source_group": source_group,
                "members": [],
                "class_counts": Counter(),
                "class_indices": set(),
                "class_names": [],
                "member_paths": [],
            },
        )

        output_path = _require_string(record.get("output_path"), context=f"output_path for {source_group}")
        source_sha256 = _require_string(record.get("source_sha256"), context=f"source_sha256 for {source_group}")
        output_sha256 = _require_string(record.get("output_sha256"), context=f"output_sha256 for {source_group}")
        image_index = _require_int(record.get("image_index"), context=f"image_index for {source_group}")
        image_count = class_info["image_count"]
        member = {
            "class_label": class_label,
            "class_name": class_name,
            "class_dir": class_dir,
            "image_index": image_index,
            "image_count": image_count,
            "source_group": source_group,
            "output_path": output_path,
            "source_sha256": source_sha256,
            "output_sha256": output_sha256,
        }
        group["members"].append(member)
        group["class_counts"][class_name] += 1
        group["class_indices"].add(class_label)
        if class_name not in group["class_names"]:
            group["class_names"].append(class_name)
        group["member_paths"].append(output_path)
        class_image_counts[class_name] += 1

    split_group_buckets = _merge_duplicate_connected_groups(group_buckets)
    group_entries = sorted(split_group_buckets.values(), key=_group_sort_key)
    if len(group_entries) < 2:
        raise ValueError("need at least two source groups to create a nonempty held-out test split")

    total_images = len(images)
    target_test_images = max(1, int(total_images * test_fraction))
    target_val_images = max(1, int(total_images * val_fraction)) if total_images >= 3 else 0

    remaining_train_counts = Counter(class_image_counts)
    split_image_counts: Counter[str] = Counter()
    split_group_counts: Counter[str] = Counter()
    class_split_counts: dict[str, Counter[str]] = {name: Counter() for name in class_order}
    group_records: list[GroupRecord] = []
    split_rows: list[SplitRow] = []

    for group_index, group in enumerate(group_entries, start=1):
        split, reason = _choose_split(
            group,
            remaining_train_counts=remaining_train_counts,
            split_image_counts=split_image_counts,
            target_test_images=target_test_images,
            target_val_images=target_val_images,
        )

        if split in {TEST_SPLIT, VAL_SPLIT}:
            for class_name, count in group["class_counts"].items():
                remaining_train_counts[class_name] -= count
                if remaining_train_counts[class_name] < 0:
                    raise AssertionError(f"negative remaining train count for {class_name}")

        split_group_counts[split] += 1
        split_image_counts[split] += len(group["members"])

        member_paths = [member["output_path"] for member in group["members"]]
        member_hashes = [member["output_sha256"] for member in group["members"]]
        group_records.append(
            GroupRecord(
                group_index=group_index,
                source_group=group["source_group"],
                dataset_split=split,
                split_reason=reason,
                image_count=len(group["members"]),
                class_count=len(group["class_names"]),
                class_names=list(group["class_names"]),
                member_paths=member_paths,
                member_sha256=canonical_json_sha(member_hashes),
            )
        )

        for member in sorted(group["members"], key=lambda item: (item["class_label"], item["image_index"], item["output_path"])):
            row_index = len(split_rows)
            split_rows.append(
                SplitRow(
                    index=row_index,
                    dataset_split=split,
                    split_reason=reason,
                    class_label=member["class_label"],
                    class_name=member["class_name"],
                    class_dir=member["class_dir"],
                    image_index=member["image_index"],
                    image_count=member["image_count"],
                    source_group=member["source_group"],
                    output_path=member["output_path"],
                    source_sha256=member["source_sha256"],
                    output_sha256=member["output_sha256"],
                )
            )
            class_split_counts[member["class_name"]][split] += 1

    if split_image_counts[TEST_SPLIT] == 0:
        raise ValueError("could not allocate any test image while preserving train coverage")
    if any(count <= 0 for count in remaining_train_counts.values()):
        missing = sorted(name for name, count in remaining_train_counts.items() if count <= 0)
        raise ValueError(f"split removed all train examples for class(es): {missing}")

    split_rows.sort(key=lambda row: (row.class_label, row.source_group, row.image_index, row.output_path))
    for index, row in enumerate(split_rows):
        split_rows[index] = SplitRow(
            index=index,
            dataset_split=row.dataset_split,
            split_reason=row.split_reason,
            class_label=row.class_label,
            class_name=row.class_name,
            class_dir=row.class_dir,
            image_index=row.image_index,
            image_count=row.image_count,
            source_group=row.source_group,
            output_path=row.output_path,
            source_sha256=row.source_sha256,
            output_sha256=row.output_sha256,
        )

    rows_payload = [asdict(row) for row in split_rows]
    groups_payload = [asdict(group) for group in group_records]
    rows_sha256 = canonical_json_sha(rows_payload)
    group_assignments_sha256 = canonical_json_sha(groups_payload)

    split_counts = {name: split_image_counts.get(name, 0) for name in (TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT)}
    group_counts = {name: split_group_counts.get(name, 0) for name in (TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT)}
    class_split_counts_payload = {
        name: {split: counts.get(split, 0) for split in (TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT)}
        for name, counts in sorted(class_split_counts.items())
    }

    report = SplitReport(
        schema_version=SCHEMA_VERSION,
        hash_algorithm="sha256",
        source_manifest=str(source_manifest),
        source_manifest_sha256=source_manifest_sha256,
        source_manifest_schema_version=str(manifest.get("schema_version", "")),
        output_dir=str(output_dir),
        split_manifest=str(output_dir / "legacy_elements_split_manifest.json"),
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        class_count=len(class_order),
        image_count=total_images,
        source_group_count=len(group_entries),
        train_image_count=split_image_counts[TRAIN_SPLIT],
        val_image_count=split_image_counts[VAL_SPLIT],
        test_image_count=split_image_counts[TEST_SPLIT],
        train_group_count=split_group_counts[TRAIN_SPLIT],
        val_group_count=split_group_counts[VAL_SPLIT],
        test_group_count=split_group_counts[TEST_SPLIT],
        class_counts=dict(sorted(class_image_counts.items())),
        class_split_counts=class_split_counts_payload,
        split_counts=split_counts,
        group_counts=group_counts,
        rows_sha256=rows_sha256,
        group_assignments_sha256=group_assignments_sha256,
        groups=group_records,
        rows=split_rows,
    )

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"output already exists (pass --overwrite to replace it): {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "legacy_elements_split_manifest.json"
    manifest_payload = asdict(report)
    manifest_path.write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return report


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_manifest", type=Path, help="Path to legacy_elements_manifest.json")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory that will receive legacy_elements_split_manifest.json",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=DEFAULT_VAL_FRACTION,
        help="Target fraction of images for the validation split",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=DEFAULT_TEST_FRACTION,
        help="Target fraction of images for the test split",
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
        report = create_legacy_elements_split(
            args.source_manifest,
            args.output_dir,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
            overwrite=args.overwrite,
        )
    except (OSError, ValueError, AssertionError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(
        f"Wrote {report.split_manifest} "
        f"(train={report.train_image_count}, val={report.val_image_count}, test={report.test_image_count})"
    )
    print(
        f"source_manifest_sha256={report.source_manifest_sha256} "
        f"rows_sha256={report.rows_sha256} group_assignments_sha256={report.group_assignments_sha256}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cove
    raise SystemExit(main())

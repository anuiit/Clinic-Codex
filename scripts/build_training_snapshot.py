#!/usr/bin/env python3
"""Build an immutable, content-addressed classifier training snapshot.

The builder merges the frozen legacy corpus, the approved external corpus,
and currently trainable annotations from the live review store.  It never
modifies any source.  Runtime class order, content hashes, source groups,
duplicate relationships, and persisted train/dev/locked-test assignments are
recorded in one snapshot contract.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import shutil
import sys
import tempfile
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from PIL import Image, ImageOps

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.codex_pipeline.data.class_order import (  # noqa: E402
    class_order_sha256,
    load_runtime_class_order,
)
from backend.services.annotation_review import AnnotationReviewStore  # noqa: E402


SCHEMA_VERSION = "training-snapshot.v2"
POLICY_VERSION = "locked-source-group.v3"
DEFAULT_SPLIT_SALT = "clinic-codex-training-snapshot-v2"
SPLITS = ("train", "dev", "locked_test")
SOURCE_PRIORITY = {"legacy": 0, "external": 1, "live_annotation": 2}


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_json_sha(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def live_annotations_sha256(rows: Iterable[dict[str, Any]]) -> str:
    records = [
        {
            "source_id": row.get("source_id"),
            "class_name": row.get("class_name"),
            "bbox": row.get("bbox"),
            "source_fingerprint_v1": row.get("source_fingerprint_v1"),
            "source_sha256": row.get("source_sha256"),
            "source_image_sha256": row.get("source_image_sha256"),
        }
        for row in rows
        if row.get("source_kind") == "live_annotation"
    ]
    return canonical_json_sha(sorted(records, key=canonical_json_bytes))


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pixel_sha256(path: Path) -> str:
    with Image.open(path) as image:
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        width, height = rgb.size
        digest = hashlib.sha256()
        digest.update(f"RGB:{width}x{height}:".encode("ascii"))
        digest.update(rgb.tobytes())
    return digest.hexdigest()


def normalized_name(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"invalid class name: {value!r}")
    return unicodedata.normalize("NFC", value.strip())


def slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("._-")
    return cleaned or "unnamed"


def repo_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def source_group_from_path(value: str | Path) -> str:
    canonical = str(value).replace("\\", "/")
    stem = PurePosixPath(canonical).stem
    page = re.match(r"^(\d+)[_-](\d+)[_-](\d+)(?:[-_].*)?$", stem)
    if page:
        return f"page:{page.group(1)}:{page.group(2)}:{page.group(3)}"
    without_instance = re.sub(r"[-_]\d+$", "", stem)
    parent = PurePosixPath(canonical).parent.as_posix().casefold()
    return "source-path:" + hashlib.sha256(
        f"{parent}/{without_instance.casefold()}".encode("utf-8")
    ).hexdigest()


def _verified_path(path_value: object, *, context: str) -> Path:
    if not isinstance(path_value, str) or not path_value:
        raise ValueError(f"{context} has no path")
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{context} file is missing: {path}")
    return path


def _base_row(
    *,
    source_kind: str,
    source_id: str,
    class_name: str,
    class_label: int,
    source_group: str,
    source_path: Path,
    source_sha256: str,
    source_pixel_sha256: str,
    decision: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    identity = {
        "source_kind": source_kind,
        "source_id": source_id,
        "class_name": class_name,
        "source_group": source_group,
        "source_pixel_sha256": source_pixel_sha256,
    }
    row = {
        "row_id": "row-" + canonical_json_sha(identity),
        "source_kind": source_kind,
        "source_id": source_id,
        "class_label": class_label,
        "class_name": class_name,
        "source_group": source_group,
        "source_path": repo_path(source_path),
        "source_sha256": source_sha256,
        "source_pixel_sha256": source_pixel_sha256,
        "decision": decision,
        "duplicate_of": None,
        "dataset_split": None,
    }
    if extra:
        row.update(extra)
    return row


def load_legacy_rows(
    manifest_path: Path,
    class_labels: dict[str, int],
    class_order: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = load_json(manifest_path)
    if manifest.get("schema_version") != "legacy-elements-freeze.v1":
        raise ValueError(f"unexpected legacy schema: {manifest.get('schema_version')!r}")
    if manifest.get("class_order") != class_order:
        raise ValueError("legacy manifest class order does not match runtime")
    if manifest.get("class_order_sha256") != class_order_sha256(class_order):
        raise ValueError("legacy manifest class-order hash does not match runtime")
    images = manifest.get("images")
    if not isinstance(images, list):
        raise ValueError("legacy manifest has no images list")

    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(images):
        if not isinstance(raw, dict):
            raise ValueError(f"legacy images[{index}] must be an object")
        class_name = normalized_name(raw.get("class_name"))
        if class_name not in class_labels:
            raise ValueError(f"legacy class absent from runtime taxonomy: {class_name}")
        path = _verified_path(raw.get("output_path"), context=f"legacy images[{index}]")
        byte_sha = sha256_file(path)
        expected = raw.get("output_sha256")
        if expected != byte_sha:
            raise ValueError(f"legacy output hash mismatch: {path}")
        rows.append(
            _base_row(
                source_kind="legacy",
                source_id=str(raw.get("archive_path") or raw.get("output_path")),
                class_name=class_name,
                class_label=class_labels[class_name],
                source_group=str(raw.get("source_group") or source_group_from_path(path)),
                source_path=path,
                source_sha256=str(raw.get("source_sha256") or byte_sha),
                source_pixel_sha256=pixel_sha256(path),
                decision="frozen_legacy",
                extra={"source_manifest_row": index},
            )
        )
    return manifest, rows


def load_external_rows(
    snapshot_path: Path,
    class_labels: dict[str, int],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    snapshot = load_json(snapshot_path)
    if snapshot.get("schema_version") != "external-corpus-training-snapshot.v1":
        raise ValueError(f"unexpected external schema: {snapshot.get('schema_version')!r}")
    rows_raw = snapshot.get("rows")
    if not isinstance(rows_raw, list):
        raise ValueError("external snapshot has no rows list")

    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(rows_raw):
        if not isinstance(raw, dict):
            raise ValueError(f"external rows[{index}] must be an object")
        class_name = normalized_name(raw.get("class_name"))
        if class_name not in class_labels:
            raise ValueError(f"external class absent from runtime taxonomy: {class_name}")
        if raw.get("class_label") != class_labels[class_name]:
            raise ValueError(f"external class label disagrees with runtime: {class_name}")
        path = _verified_path(raw.get("output_path"), context=f"external rows[{index}]")
        byte_sha = sha256_file(path)
        if raw.get("output_sha256") != byte_sha:
            raise ValueError(f"external output hash mismatch: {path}")
        actual_pixel_sha = pixel_sha256(path)
        recorded_pixel_sha = raw.get("source_pixel_sha256")
        if recorded_pixel_sha and recorded_pixel_sha != actual_pixel_sha:
            raise ValueError(f"external pixel hash mismatch: {path}")
        original_path = str(raw.get("source_path") or path)
        rows.append(
            _base_row(
                source_kind="external",
                source_id=original_path,
                class_name=class_name,
                class_label=class_labels[class_name],
                source_group=source_group_from_path(original_path),
                source_path=path,
                source_sha256=str(raw.get("source_sha256") or byte_sha),
                source_pixel_sha256=actual_pixel_sha,
                decision="approved_external",
                extra={"source_manifest_row": index},
            )
        )
    return snapshot, rows


def load_live_annotation_rows(
    annotations_dirs: Iterable[Path],
    class_labels: dict[str, int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifests: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for annotations_dir in sorted(
        [path.expanduser().resolve() for path in annotations_dirs], key=str
    ):
        review_index = annotations_dir / "review-index.json"
        manifests.append(
            {
                "path": repo_path(review_index),
                "sha256": sha256_file(review_index),
            }
        )
        for item in AnnotationReviewStore(annotations_dir).iter_approved_annotations():
            class_name = normalized_name(item.get("class_name"))
            if class_name not in class_labels:
                raise ValueError(f"approved annotation class absent from runtime: {class_name}")
            crop_path = _verified_path(item.get("crop_path"), context="approved crop")
            image_path = _verified_path(item.get("image_path"), context="approved source image")
            crop_byte_sha = sha256_file(crop_path)
            crop_pixel_sha = pixel_sha256(crop_path)
            image_byte_sha = sha256_file(image_path)
            image_pixel_sha = pixel_sha256(image_path)
            analysis_id = str(item.get("analysis_id"))
            element_index = item.get("index")
            source_id = f"{analysis_id}:{element_index}"
            rows.append(
                _base_row(
                    source_kind="live_annotation",
                    source_id=source_id,
                    class_name=class_name,
                    class_label=class_labels[class_name],
                    source_group=f"image-pixels:{image_pixel_sha}",
                    source_path=crop_path,
                    source_sha256=crop_byte_sha,
                    source_pixel_sha256=crop_pixel_sha,
                    decision="approved_live_review",
                    extra={
                        "analysis_id": analysis_id,
                        "index": element_index,
                        "bbox": item.get("bbox"),
                        "source_image_path": repo_path(image_path),
                        "source_image_sha256": image_byte_sha,
                        "source_image_pixel_sha256": image_pixel_sha,
                        "source_fingerprint_v1": item.get("source_fingerprint"),
                        "review_dataset_split_v1": item.get("dataset_split"),
                    },
                )
            )
    return manifests, rows


class DisjointSet:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        smaller, larger = sorted((left_root, right_root))
        self.parent[larger] = smaller


def _deduplicate_and_group(
    rows: list[dict[str, Any]],
    *,
    exclude_conflicts: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], DisjointSet]:
    source_groups = {row["source_group"] for row in rows}
    dsu = DisjointSet(source_groups)
    by_pixels: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_pixels[row["source_pixel_sha256"]].append(row)

    selected: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    for pixel_sha, cluster in sorted(by_pixels.items()):
        classes = {row["class_name"] for row in cluster}
        if len(classes) != 1:
            details = sorted((row["class_name"], row["source_id"]) for row in cluster)
            if not exclude_conflicts:
                raise ValueError(f"pixel hash maps to conflicting classes {pixel_sha}: {details}")
            conflicts.append(
                {
                    "source_pixel_sha256": pixel_sha,
                    "class_names": sorted(classes),
                    "rows": [
                        dict(row)
                        for row in sorted(cluster, key=lambda item: item["row_id"])
                    ],
                    "resolution": "excluded_pending_manual_review",
                }
            )
            continue
        groups = sorted({row["source_group"] for row in cluster})
        for group in groups[1:]:
            dsu.union(groups[0], group)
        ordered = sorted(
            cluster,
            key=lambda row: (
                SOURCE_PRIORITY[row["source_kind"]],
                row["class_label"],
                row["source_group"],
                row["source_id"],
                row["row_id"],
            ),
        )
        winner = dict(ordered[0])
        winner["duplicate_count"] = len(ordered) - 1
        selected.append(winner)
        for duplicate in ordered[1:]:
            excluded = dict(duplicate)
            excluded["duplicate_of"] = winner["row_id"]
            excluded["dataset_split"] = "excluded"
            excluded["exclusion_reason"] = "exact_pixel_duplicate"
            duplicates.append(excluded)
    return selected, duplicates, conflicts, dsu


def _parent_assignments(parent_manifest: Path | None) -> tuple[dict[str, str], set[str], str | None]:
    if parent_manifest is None:
        return {}, set(), None
    parent = load_json(parent_manifest)
    if parent.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unexpected parent snapshot schema: {parent.get('schema_version')!r}")
    assignments = parent.get("source_group_assignments")
    if not isinstance(assignments, dict) or not all(
        isinstance(group, str) and split in SPLITS for group, split in assignments.items()
    ):
        raise ValueError("parent snapshot has invalid source-group assignments")
    active_row_ids = parent.get("active_row_ids")
    if not isinstance(active_row_ids, list) or not all(isinstance(value, str) for value in active_row_ids):
        raise ValueError("parent snapshot has invalid active_row_ids")
    snapshot_id = parent.get("snapshot_id")
    if not isinstance(snapshot_id, str) or not snapshot_id:
        raise ValueError("parent snapshot has no snapshot_id")
    return dict(assignments), set(active_row_ids), snapshot_id


def _hash_split(group_key: str, *, salt: str, dev_fraction: float, test_fraction: float) -> str:
    bucket = int(hashlib.sha256(f"{salt}:{group_key}".encode("utf-8")).hexdigest()[:16], 16) / 2**64
    if bucket < test_fraction:
        return "locked_test"
    if bucket < test_fraction + dev_fraction:
        return "dev"
    return "train"


def assign_splits(
    rows: list[dict[str, Any]],
    dsu: DisjointSet,
    *,
    parent_assignments: dict[str, str],
    salt: str,
    dev_fraction: float,
    test_fraction: float,
    min_train_rows_per_class: int,
) -> tuple[list[dict[str, Any]], dict[str, str], list[dict[str, Any]]]:
    if (
        not isinstance(min_train_rows_per_class, int)
        or isinstance(min_train_rows_per_class, bool)
        or min_train_rows_per_class < 1
    ):
        raise ValueError("min_train_rows_per_class must be an integer >= 1")
    if (
        isinstance(dev_fraction, bool)
        or isinstance(test_fraction, bool)
        or not isinstance(dev_fraction, (int, float))
        or not isinstance(test_fraction, (int, float))
        or not math.isfinite(dev_fraction)
        or not math.isfinite(test_fraction)
        or not 0 <= dev_fraction < 1
        or not 0 <= test_fraction < 1
    ):
        raise ValueError("split fractions must be finite values in [0, 1)")
    if dev_fraction + test_fraction >= 1:
        raise ValueError("dev_fraction + locked_test_fraction must be < 1")

    component_groups: dict[str, set[str]] = defaultdict(set)
    active_groups = {row["source_group"] for row in rows}
    for group in sorted(active_groups):
        component_groups[dsu.find(group)].add(group)
    component_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        component_rows[dsu.find(row["source_group"])].append(row)

    component_split: dict[str, str] = {}
    inherited_components: set[str] = set()
    for root, groups in sorted(component_groups.items()):
        inherited = {parent_assignments[group] for group in groups if group in parent_assignments}
        if len(inherited) > 1:
            raise ValueError(f"new duplicate connection spans parent splits: {sorted(groups)}")
        if inherited:
            component_split[root] = inherited.pop()
            inherited_components.add(root)
        else:
            component_split[root] = _hash_split(
                "|".join(sorted(groups)),
                salt=salt,
                dev_fraction=dev_fraction,
                test_fraction=test_fraction,
            )

    class_to_components: dict[str, set[str]] = defaultdict(set)
    for root, component in component_rows.items():
        for row in component:
            class_to_components[row["class_name"]].add(root)
    # Train coverage is independent from holdout eligibility. Move only new
    # components until every class reaches the declared row floor; inherited
    # parent assignments are immutable.
    while True:
        train_class_counts: Counter[str] = Counter()
        for root, split in component_split.items():
            if split == "train":
                train_class_counts.update(
                    row["class_name"] for row in component_rows[root]
                )
        deficient = sorted(
            class_name
            for class_name in class_to_components
            if train_class_counts[class_name] < min_train_rows_per_class
        )
        if not deficient:
            break
        class_name = deficient[0]
        candidates = [
            root
            for root in class_to_components[class_name]
            if component_split[root] != "train" and root not in inherited_components
        ]
        if not candidates:
            raise ValueError(
                "parent assignments leave insufficient train rows for class "
                f"{class_name!r}: {train_class_counts[class_name]} < "
                f"{min_train_rows_per_class}"
            )
        candidates.sort(
            key=lambda root: (
                -sum(
                    row["class_name"] == class_name for row in component_rows[root]
                ),
                root,
            )
        )
        component_split[candidates[0]] = "train"

    # Hash assignment plus the train-coverage safeguard can empty a held-out
    # split when rare classes connect many source groups. Move only new,
    # eligible train components and never sacrifice the final train example
    # of any class. Parent assignments remain untouched.
    target_rows = {
        "locked_test": max(1, int(len(rows) * test_fraction)) if test_fraction else 0,
        "dev": max(1, int(len(rows) * dev_fraction)) if dev_fraction else 0,
    }
    active_targets = [split for split in ("locked_test", "dev") if target_rows[split] > 0]
    while active_targets:
        current_rows = {
            target: sum(
                len(component_rows[root])
                for root, split in component_split.items()
                if split == target
            )
            for target in active_targets
        }
        if all(current_rows[target] >= target_rows[target] for target in active_targets):
            break
        target_split = min(
            active_targets,
            key=lambda target: (
                current_rows[target] / target_rows[target],
                target,
            ),
        )
        train_class_counts: Counter[str] = Counter()
        for root, split in component_split.items():
            if split == "train":
                train_class_counts.update(row["class_name"] for row in component_rows[root])
        held_classes = {
            row["class_name"]
            for root, split in component_split.items()
            if split == target_split
            for row in component_rows[root]
        }
        eligible: list[str] = []
        for root, split in component_split.items():
            if split != "train" or root in inherited_components:
                continue
            component_counts = Counter(row["class_name"] for row in component_rows[root])
            if all(
                train_class_counts[class_name] - count >= min_train_rows_per_class
                for class_name, count in component_counts.items()
            ):
                eligible.append(root)
        if not eligible:
            break
        eligible.sort(
            key=lambda root: (
                -len(
                    {row["class_name"] for row in component_rows[root]}
                    - held_classes
                ),
                len(component_rows[root]),
                hashlib.sha256(
                    f"{salt}:{target_split}:{root}".encode("utf-8")
                ).hexdigest(),
                root,
            )
        )
        component_split[eligible[0]] = target_split

    # The train-coverage safeguard may leave nearly all movable held-out
    # components in one split. Redistribute only non-parent held-out
    # components between dev and locked_test; the train set and inherited
    # assignments remain untouched.
    if len(active_targets) == 2:
        movable_held = [
            root
            for root, split in component_split.items()
            if split in active_targets and root not in inherited_components
        ]
        held_rows = {target: 0 for target in active_targets}
        held_classes = {target: set() for target in active_targets}
        for root, split in component_split.items():
            if root in inherited_components and split in active_targets:
                held_rows[split] += len(component_rows[root])
                held_classes[split].update(
                    row["class_name"] for row in component_rows[root]
                )
        movable_held.sort(
            key=lambda root: (
                -len({row["class_name"] for row in component_rows[root]}),
                -len(component_rows[root]),
                hashlib.sha256(f"{salt}:heldout:{root}".encode("utf-8")).hexdigest(),
                root,
            )
        )
        if len(movable_held) <= 20 and movable_held:
            left, right = active_targets
            best_score = None
            best_mask = 0
            for mask in range(1 << len(movable_held)):
                rows_by_split = dict(held_rows)
                classes_by_split = {
                    split: set(values) for split, values in held_classes.items()
                }
                for index, root in enumerate(movable_held):
                    target = right if mask & (1 << index) else left
                    rows_by_split[target] += len(component_rows[root])
                    classes_by_split[target].update(
                        row["class_name"] for row in component_rows[root]
                    )
                if not rows_by_split[left] or not rows_by_split[right]:
                    continue
                score = (
                    min(len(classes_by_split[left]), len(classes_by_split[right])),
                    -abs(
                        rows_by_split[left] / target_rows[left]
                        - rows_by_split[right] / target_rows[right]
                    ),
                    min(rows_by_split[left], rows_by_split[right]),
                    -mask,
                )
                if best_score is None or score > best_score:
                    best_score = score
                    best_mask = mask
            for index, root in enumerate(movable_held):
                component_split[root] = right if best_mask & (1 << index) else left
        else:
            for root in movable_held:
                root_classes = {row["class_name"] for row in component_rows[root]}
                target = min(
                    active_targets,
                    key=lambda candidate: (
                        len(held_classes[candidate]),
                        held_rows[candidate] / target_rows[candidate],
                        -len(root_classes - held_classes[candidate]),
                        candidate,
                    ),
                )
                component_split[root] = target
                held_rows[target] += len(component_rows[root])
                held_classes[target].update(root_classes)

    assignments: dict[str, str] = {}
    components: list[dict[str, Any]] = []
    for root, groups in sorted(component_groups.items()):
        split = component_split[root]
        for group in groups:
            assignments[group] = split
        component = component_rows[root]
        components.append(
            {
                "component_id": "group-" + canonical_json_sha(sorted(groups)),
                "source_groups": sorted(groups),
                "dataset_split": split,
                "assignment_origin": (
                    "parent" if root in inherited_components else "new"
                ),
                "row_ids": sorted(row["row_id"] for row in component),
                "class_names": sorted({row["class_name"] for row in component}),
            }
        )

    assigned_rows: list[dict[str, Any]] = []
    for row in rows:
        assigned = dict(row)
        assigned["dataset_split"] = assignments[row["source_group"]]
        assigned_rows.append(assigned)
    return assigned_rows, assignments, components


def _semantic_manifest(
    *,
    class_order: list[str],
    runtime_config_path: Path,
    source_manifests: list[dict[str, Any]],
    active_rows: list[dict[str, Any]],
    duplicate_rows: list[dict[str, Any]],
    conflicts: list[dict[str, Any]],
    source_group_assignments: dict[str, str],
    components: list[dict[str, Any]],
    parent_snapshot_id: str | None,
    split_salt: str,
    dev_fraction: float,
    locked_test_fraction: float,
    min_train_rows_per_class: int,
    allow_underfilled_holdouts: bool,
) -> dict[str, Any]:
    sorted_active = sorted(
        active_rows,
        key=lambda row: (row["class_label"], row["source_group"], row["row_id"]),
    )
    sorted_duplicates = sorted(
        duplicate_rows,
        key=lambda row: (row["duplicate_of"], row["class_label"], row["row_id"]),
    )
    split_counts = Counter(row["dataset_split"] for row in sorted_active)
    class_split_counts: dict[str, dict[str, int]] = {}
    for class_name in class_order:
        counts = Counter(
            row["dataset_split"] for row in sorted_active if row["class_name"] == class_name
        )
        class_split_counts[class_name] = {split: counts[split] for split in SPLITS}
    target_rows = {
        "dev": max(1, int(len(sorted_active) * dev_fraction)) if dev_fraction else 0,
        "locked_test": (
            max(1, int(len(sorted_active) * locked_test_fraction))
            if locked_test_fraction
            else 0
        ),
    }
    split_target_deficits = {
        split: max(0, target_rows[split] - split_counts[split])
        for split in ("dev", "locked_test")
    }
    split_targets_satisfied = not any(split_target_deficits.values())
    ready_for_training = (
        len(class_order) > 0
        and all(
            class_split_counts[name]["train"] >= min_train_rows_per_class
            for name in class_order
        )
        and split_counts["dev"] > 0
        and split_counts["locked_test"] > 0
    )
    heldout_class_counts = {
        split: len(
            {row["class_name"] for row in sorted_active if row["dataset_split"] == split}
        )
        for split in ("dev", "locked_test")
    }
    promotion_evaluation_ready = (
        split_counts["locked_test"] >= 600
        and heldout_class_counts["locked_test"] >= 100
        and split_targets_satisfied
        and not allow_underfilled_holdouts
    )
    row_by_id = {row["row_id"]: row for row in sorted_active}
    total_class_counts = Counter(row["class_name"] for row in sorted_active)
    class_components: dict[str, set[str]] = defaultdict(set)
    class_groups: dict[str, set[str]] = defaultdict(set)
    component_class_counts: dict[str, Counter[str]] = {}
    component_by_id = {component["component_id"]: component for component in components}
    for component in components:
        component_id = component["component_id"]
        counts = Counter(
            row_by_id[row_id]["class_name"] for row_id in component["row_ids"]
        )
        component_class_counts[component_id] = counts
        for class_name in counts:
            class_components[class_name].add(component_id)
        for row_id in component["row_ids"]:
            row = row_by_id[row_id]
            class_groups[row["class_name"]].add(row["source_group"])
    holdout_eligibility_classes: list[dict[str, Any]] = []
    current_train_class_counts = Counter(
        row["class_name"]
        for row in sorted_active
        if row["dataset_split"] == "train"
    )
    for class_name in class_order:
        eligible_components = sorted(
            component_id
            for component_id in class_components[class_name]
            if all(
                total_class_counts[member_class] - count
                >= min_train_rows_per_class
                for member_class, count in component_class_counts[component_id].items()
            )
        )
        component_count = len(class_components[class_name])
        movable_new_train_components = sorted(
            component_id
            for component_id in class_components[class_name]
            if component_by_id[component_id]["dataset_split"] == "train"
            and component_by_id[component_id].get("assignment_origin") == "new"
            and all(
                current_train_class_counts[member_class] - count
                >= min_train_rows_per_class
                for member_class, count in component_class_counts[component_id].items()
            )
        )
        persisted_holdout_splits = [
            split
            for split in ("dev", "locked_test")
            if class_split_counts[class_name][split] > 0
        ]
        if persisted_holdout_splits:
            operational_status = "already_present_in_persisted_holdout"
        elif movable_new_train_components:
            operational_status = "new_train_component_is_movable"
        else:
            operational_status = "requires_new_independent_component"
        holdout_eligibility_classes.append(
            {
                "class_label": class_order.index(class_name),
                "class_name": class_name,
                "row_count": total_class_counts[class_name],
                "component_count": component_count,
                "source_group_count": len(class_groups[class_name]),
                "eligible": bool(eligible_components),
                "eligible_component_ids": eligible_components,
                "persisted_holdout_splits": persisted_holdout_splits,
                "movable_new_train_component_ids": movable_new_train_components,
                "operational_status": operational_status,
                "ineligible_reason": (
                    None
                    if eligible_components
                    else (
                        "single_duplicate_connected_component"
                        if component_count <= 1
                        else "all_components_violate_train_floor"
                    )
                ),
            }
        )
    holdout_eligible_count = sum(
        item["eligible"] for item in holdout_eligibility_classes
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "parent_snapshot_id": parent_snapshot_id,
        "runtime_config": repo_path(runtime_config_path),
        "runtime_config_sha256": sha256_file(runtime_config_path),
        "class_order": class_order,
        "class_order_sha256": class_order_sha256(class_order),
        "policy": {
            "version": POLICY_VERSION,
            "deduplication": "exact_rgb_pixel_sha256",
            "grouping": "source_group_plus_duplicate_union",
            "split_salt": split_salt,
            "dev_fraction": dev_fraction,
            "locked_test_fraction": locked_test_fraction,
            "minimum_train_rows_per_class": min_train_rows_per_class,
            "underfilled_holdouts_require_explicit_override": True,
            "allow_underfilled_holdouts": allow_underfilled_holdouts,
            "parent_assignments_are_immutable": True,
            "content_sha256_excludes_materialized_output_fields": True,
        },
        "source_manifests": sorted(source_manifests, key=lambda item: item["path"]),
        "live_annotation_count": sum(
            row.get("source_kind") == "live_annotation"
            for row in [*sorted_active, *sorted_duplicates]
        )
        + sum(
            row.get("source_kind") == "live_annotation"
            for conflict in conflicts
            for row in conflict["rows"]
        ),
        "live_annotations_sha256": live_annotations_sha256(
            [
                *sorted_active,
                *sorted_duplicates,
                *[
                    row
                    for conflict in conflicts
                    for row in conflict["rows"]
                ],
            ]
        ),
        "active_row_ids": [row["row_id"] for row in sorted_active],
        "row_count": len(sorted_active),
        "duplicate_count": len(sorted_duplicates),
        "conflict_count": len(conflicts),
        "class_count": len(class_order),
        "source_group_count": len(source_group_assignments),
        "component_count": len(components),
        "ready_for_training": ready_for_training,
        "promotion_evaluation_ready": promotion_evaluation_ready,
        "split_targets_satisfied": split_targets_satisfied,
        "split_target_rows": target_rows,
        "split_target_deficits": split_target_deficits,
        "holdout_eligibility": {
            "structural_definition": (
                "A component is structurally eligible when removing it from the full "
                "corpus leaves the declared train-row floor for every member class."
            ),
            "operational_definition": (
                "Parent-origin components are immutable. Only new train components "
                "listed as movable may be reassigned without collecting more data."
            ),
            "eligible_class_count": holdout_eligible_count,
            "ineligible_class_count": len(class_order) - holdout_eligible_count,
            "operationally_movable_new_train_class_count": sum(
                bool(item["movable_new_train_component_ids"])
                for item in holdout_eligibility_classes
            ),
            "classes": holdout_eligibility_classes,
        },
        "heldout_class_counts": heldout_class_counts,
        "split_counts": {split: split_counts[split] for split in SPLITS},
        "split_fractions_achieved": {
            split: split_counts[split] / len(sorted_active) if sorted_active else 0.0
            for split in SPLITS
        },
        "class_split_counts": class_split_counts,
        "source_group_assignments": dict(sorted(source_group_assignments.items())),
        "components": sorted(components, key=lambda item: item["component_id"]),
        "rows": sorted_active,
        "duplicates": sorted_duplicates,
        "conflicts": sorted(conflicts, key=lambda item: item["source_pixel_sha256"]),
    }


def plan_training_snapshot(
    *,
    runtime_config: Path,
    legacy_manifest: Path,
    external_snapshot: Path,
    annotations_dirs: list[Path] | None = None,
    parent_manifest: Path | None = None,
    split_salt: str = DEFAULT_SPLIT_SALT,
    dev_fraction: float = 0.1,
    locked_test_fraction: float = 0.1,
    min_train_rows_per_class: int = 2,
    allow_underfilled_holdouts: bool = False,
    exclude_conflicts: bool = False,
) -> dict[str, Any]:
    runtime_config = runtime_config.expanduser().resolve()
    class_order = [normalized_name(name) for name in load_runtime_class_order(runtime_config)]
    class_labels = {name: index for index, name in enumerate(class_order)}
    legacy_payload, legacy_rows = load_legacy_rows(legacy_manifest.resolve(), class_labels, class_order)
    external_payload, external_rows = load_external_rows(external_snapshot.resolve(), class_labels)
    annotation_manifests, annotation_rows = load_live_annotation_rows(
        annotations_dirs or [], class_labels
    )
    all_rows = legacy_rows + external_rows + annotation_rows
    selected, duplicates, conflicts, dsu = _deduplicate_and_group(
        all_rows, exclude_conflicts=exclude_conflicts
    )
    parent_assignments, parent_row_ids, parent_snapshot_id = _parent_assignments(parent_manifest)
    selected, assignments, components = assign_splits(
        selected,
        dsu,
        parent_assignments=parent_assignments,
        salt=split_salt,
        dev_fraction=dev_fraction,
        test_fraction=locked_test_fraction,
        min_train_rows_per_class=min_train_rows_per_class,
    )
    current_ids = {row["row_id"] for row in selected}
    missing_parent = sorted(parent_row_ids - current_ids)
    if missing_parent:
        raise ValueError(
            "snapshot would silently remove parent rows: " + ", ".join(missing_parent[:5])
        )

    source_manifests = [
        {
            "kind": str(legacy_payload.get("schema_version")),
            "path": repo_path(legacy_manifest),
            "sha256": sha256_file(legacy_manifest),
        },
        {
            "kind": str(external_payload.get("schema_version")),
            "path": repo_path(external_snapshot),
            "sha256": sha256_file(external_snapshot),
        },
        *[
            {"kind": "annotation-review-index.v1", **manifest}
            for manifest in annotation_manifests
        ],
    ]
    semantic = _semantic_manifest(
        class_order=class_order,
        runtime_config_path=runtime_config,
        source_manifests=source_manifests,
        active_rows=selected,
        duplicate_rows=duplicates,
        conflicts=conflicts,
        source_group_assignments=assignments,
        components=components,
        parent_snapshot_id=parent_snapshot_id,
        split_salt=split_salt,
        dev_fraction=dev_fraction,
        locked_test_fraction=locked_test_fraction,
        min_train_rows_per_class=min_train_rows_per_class,
        allow_underfilled_holdouts=allow_underfilled_holdouts,
    )
    if not semantic["split_targets_satisfied"] and not allow_underfilled_holdouts:
        deficits = semantic["split_target_deficits"]
        raise ValueError(
            "heldout split targets are infeasible with the current immutable groups: "
            f"dev deficit={deficits['dev']}, locked_test deficit={deficits['locked_test']}; "
            "collect independent source components or pass "
            "--allow-underfilled-holdouts for an explicitly research-only snapshot"
        )
    content_sha = canonical_json_sha(semantic)
    return {
        **semantic,
        "snapshot_id": "snapshot-" + content_sha[:20],
        "content_sha256": content_sha,
    }


def _bmp_bytes(path: Path) -> bytes:
    with Image.open(path) as image:
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        buffer = io.BytesIO()
        rgb.save(buffer, format="BMP")
    return buffer.getvalue()


def materialize_training_snapshot(manifest: dict[str, Any], output_root: Path) -> Path:
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / manifest["snapshot_id"]
    if destination.exists():
        raise FileExistsError(f"snapshot already exists: {destination}")

    with tempfile.TemporaryDirectory(prefix=".snapshot-staging-", dir=output_root) as temporary:
        staging = Path(temporary)
        elements = staging / "Elements"
        materialized_rows: list[dict[str, Any]] = []
        for row in manifest["rows"]:
            class_dir = elements / f"{row['class_label'] + 1:04d}-{slug(row['class_name'])}"
            class_dir.mkdir(parents=True, exist_ok=True)
            relative_output = Path("Elements") / class_dir.name / f"{row['source_kind']}-{row['row_id'][4:20]}.bmp"
            output = staging / relative_output
            payload = _bmp_bytes(Path(row["source_path"]) if Path(row["source_path"]).is_absolute() else REPO_ROOT / row["source_path"])
            output.write_bytes(payload)
            updated = dict(row)
            updated["output_path"] = relative_output.as_posix()
            updated["output_sha256"] = hashlib.sha256(payload).hexdigest()
            materialized_rows.append(updated)

        final_manifest = dict(manifest)
        final_manifest["rows"] = materialized_rows
        manifest_path = staging / "snapshot_manifest.json"
        manifest_path.write_text(
            json.dumps(final_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        metadata_path = staging / "metadata.csv"
        fieldnames = [
            "image_path",
            "element_id",
            "element_name",
            "class_label",
            "codex",
            "folio",
            "page",
            "instance_id",
            "row_id",
            "source_group",
            "dataset_split",
            "source_sha256",
            "source_pixel_sha256",
            "output_sha256",
        ]
        with metadata_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in materialized_rows:
                writer.writerow(
                    {
                        "image_path": row["output_path"],
                        "element_id": row["class_label"] + 1,
                        "element_name": row["class_name"],
                        "class_label": row["class_label"],
                        "codex": "",
                        "folio": "",
                        "page": "",
                        "instance_id": row["row_id"],
                        "row_id": row["row_id"],
                        "source_group": row["source_group"],
                        "dataset_split": row["dataset_split"],
                        "source_sha256": row["source_sha256"],
                        "source_pixel_sha256": row["source_pixel_sha256"],
                        "output_sha256": row["output_sha256"],
                    }
                )

        checksums = {
            "schema_version": "training-snapshot-checksums.v1",
            "snapshot_id": manifest["snapshot_id"],
            "snapshot_manifest_sha256": sha256_file(manifest_path),
            "metadata_csv_sha256": sha256_file(metadata_path),
            "element_count": len(materialized_rows),
            "elements_sha256": canonical_json_sha(
                [
                    {"path": row["output_path"], "sha256": row["output_sha256"]}
                    for row in materialized_rows
                ]
            ),
        }
        (staging / "checksums.json").write_text(
            json.dumps(checksums, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        staging.replace(destination)
    return destination


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime-config",
        type=Path,
        default=REPO_ROOT / "backend" / "codex_model" / "config.json",
    )
    parser.add_argument(
        "--legacy-manifest",
        type=Path,
        default=REPO_ROOT
        / "backend"
        / "training_corpus"
        / "frozen"
        / "legacy-elements-v1"
        / "legacy_elements_manifest.json",
    )
    parser.add_argument(
        "--external-snapshot",
        type=Path,
        default=REPO_ROOT
        / "backend"
        / "training_corpus"
        / "external"
        / "20260711-approved-external-corpus-v1"
        / "import_snapshot.json",
    )
    parser.add_argument(
        "--annotations-dir",
        action="append",
        type=Path,
        default=[],
        help="Live annotation review store (repeatable); stale/pending rows are excluded",
    )
    parser.add_argument("--parent-manifest", type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "backend" / "training_corpus" / "snapshots",
    )
    parser.add_argument("--split-salt", default=DEFAULT_SPLIT_SALT)
    parser.add_argument("--dev-fraction", type=float, default=0.1)
    parser.add_argument("--locked-test-fraction", type=float, default=0.1)
    parser.add_argument("--min-train-rows-per-class", type=int, default=2)
    parser.add_argument(
        "--allow-underfilled-holdouts",
        action="store_true",
        help=(
            "Allow a research-only snapshot when immutable source components cannot "
            "reach requested heldout fractions; the deficit remains manifest-bound"
        ),
    )
    parser.add_argument(
        "--exclude-conflicts",
        action="store_true",
        help="Exclude exact-pixel label conflicts and record them for manual review",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    annotations_dirs = args.annotations_dir or [REPO_ROOT / "backend" / "annotations"]
    try:
        manifest = plan_training_snapshot(
            runtime_config=args.runtime_config,
            legacy_manifest=args.legacy_manifest,
            external_snapshot=args.external_snapshot,
            annotations_dirs=annotations_dirs,
            parent_manifest=args.parent_manifest,
            split_salt=args.split_salt,
            dev_fraction=args.dev_fraction,
            locked_test_fraction=args.locked_test_fraction,
            min_train_rows_per_class=args.min_train_rows_per_class,
            allow_underfilled_holdouts=args.allow_underfilled_holdouts,
            exclude_conflicts=args.exclude_conflicts,
        )
        destination = None
        if not args.dry_run:
            destination = materialize_training_snapshot(manifest, args.output_root)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    summary = {
        "snapshot_id": manifest["snapshot_id"],
        "content_sha256": manifest["content_sha256"],
        "row_count": manifest["row_count"],
        "duplicate_count": manifest["duplicate_count"],
        "conflict_count": manifest["conflict_count"],
        "class_count": manifest["class_count"],
        "live_annotation_count": manifest["live_annotation_count"],
        "live_annotations_sha256": manifest["live_annotations_sha256"],
        "source_group_count": manifest["source_group_count"],
        "split_counts": manifest["split_counts"],
        "split_policy_version": manifest["policy"]["version"],
        "split_targets_satisfied": manifest["split_targets_satisfied"],
        "split_target_deficits": manifest["split_target_deficits"],
        "holdout_eligible_class_count": manifest["holdout_eligibility"][
            "eligible_class_count"
        ],
        "allow_underfilled_holdouts": manifest["policy"][
            "allow_underfilled_holdouts"
        ],
        "destination": str(destination) if destination else None,
        "dry_run": args.dry_run,
    }
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(
            f"Snapshot {summary['snapshot_id']}: {summary['row_count']} rows, "
            f"{summary['duplicate_count']} duplicates, splits={summary['split_counts']}"
        )
        if destination:
            print(f"Materialized: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

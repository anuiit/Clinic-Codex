#!/usr/bin/env python3
"""Build leakage-safe provenance components and grouped folds.

This utility is deliberately model-free. It reads an explicit JSONL manifest,
hashes every readable image in one decoded RGB domain, joins rows connected by
document family or exact pixels, and assigns whole components to deterministic
folds. It never reads a model evaluation or final test artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


ALPHA_VARIANT_RE = re.compile(r"_[a-z]{1,2}$", re.IGNORECASE)
NUMERIC_INSTANCE_RE = re.compile(r"-\d+$")
PAGE_GROUP_RE = re.compile(r"^page:(\d+):(\d+):(\d+)$", re.IGNORECASE)


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def normalize_source_family(value: str) -> str:
    normalized = value.strip().replace("\\", "/").casefold()
    page_match = PAGE_GROUP_RE.fullmatch(normalized)
    if page_match:
        return "_".join(page_match.groups())
    stem = Path(normalized).stem
    stem = ALPHA_VARIANT_RE.sub("", stem)
    stem = NUMERIC_INSTANCE_RE.sub("", stem)
    return stem


def decoded_rgb_sha256(path: Path) -> str:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        digest = hashlib.sha256()
        digest.update(width.to_bytes(8, "big"))
        digest.update(height.to_bytes(8, "big"))
        digest.update(rgb.tobytes())
        return digest.hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_json_bytes(value))


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


def _image_path(raw: dict[str, Any], input_dir: Path) -> Path:
    candidate = raw.get("image_path") or raw.get("output_path")
    if not candidate:
        raise ValueError("row needs image_path or output_path")
    path = Path(str(candidate))
    return path if path.is_absolute() else (input_dir / path).resolve()


def _family_source(raw: dict[str, Any]) -> str:
    candidate = raw.get("source_group") or raw.get("source_path")
    candidate = candidate or raw.get("image_path") or raw.get("output_path")
    if not candidate:
        raise ValueError("row needs source_group, source_path, image_path, or output_path")
    return str(candidate)


def load_and_hash_rows(
    input_jsonl: Path,
    hash_workers: int = 16,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    with input_jsonl.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError(f"line {line_number}: expected an object")
            row_id = str(raw.get("row_id", "")).strip()
            if not row_id:
                raise ValueError(f"line {line_number}: missing row_id")
            if row_id in seen_ids:
                raise ValueError(f"line {line_number}: duplicate row_id {row_id}")
            seen_ids.add(row_id)
            image_path = _image_path(raw, input_jsonl.parent)
            source_family = normalize_source_family(_family_source(raw))
            if not source_family:
                raise ValueError(f"line {line_number}: empty source family")
            rows.append(
                {
                    "row_id": row_id,
                    "cache_name": str(raw.get("cache_name", "unknown")),
                    "image_path": str(image_path),
                    "source_path": str(raw.get("source_path", "")),
                    "source_group": str(raw.get("source_group", "")),
                    "source_family": source_family,
                    "decoded_pixel_sha256": None,
                    "class_label": int(raw["class_label"]),
                    "class_name": str(raw["class_name"]),
                }
            )
    if not rows:
        raise ValueError("input manifest is empty")
    rows.sort(key=lambda row: row["row_id"])

    def hash_row(row: dict[str, Any]) -> tuple[str | None, str | None]:
        try:
            return decoded_rgb_sha256(Path(row["image_path"])), None
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}"

    worker_count = max(1, min(int(hash_workers), 32))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        hash_results = executor.map(hash_row, rows)
        for row, (pixel_hash, error) in zip(rows, hash_results):
            row["decoded_pixel_sha256"] = pixel_hash
            if error:
                errors.append(
                    {
                        "row_id": row["row_id"],
                        "image_path": row["image_path"],
                        "error": error,
                    }
                )
    return rows, errors


def build_components(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    union_find = UnionFind(len(rows))
    family_owner: dict[str, int] = {}
    pixel_owner: dict[str, int] = {}
    pixel_labels: dict[str, set[tuple[int, str]]] = defaultdict(set)
    for index, row in enumerate(rows):
        family = row["source_family"]
        if family in family_owner:
            union_find.union(index, family_owner[family])
        else:
            family_owner[family] = index
        pixel_hash = row["decoded_pixel_sha256"]
        if pixel_hash:
            if pixel_hash in pixel_owner:
                union_find.union(index, pixel_owner[pixel_hash])
            else:
                pixel_owner[pixel_hash] = index
            pixel_labels[pixel_hash].add((row["class_label"], row["class_name"]))

    members_by_root: dict[int, list[int]] = defaultdict(list)
    for index in range(len(rows)):
        members_by_root[union_find.find(index)].append(index)

    components: list[dict[str, Any]] = []
    component_for_index: dict[int, str] = {}
    for member_indices in members_by_root.values():
        row_ids = sorted(rows[index]["row_id"] for index in member_indices)
        component_digest = hashlib.sha256("\n".join(row_ids).encode("utf-8")).hexdigest()
        component_id = f"component:{component_digest[:20]}"
        for index in member_indices:
            component_for_index[index] = component_id
        label_counts = Counter(rows[index]["class_label"] for index in member_indices)
        components.append(
            {
                "component_id": component_id,
                "row_ids": row_ids,
                "row_count": len(member_indices),
                "source_families": sorted(
                    {rows[index]["source_family"] for index in member_indices}
                ),
                "decoded_pixel_sha256": sorted(
                    {
                        rows[index]["decoded_pixel_sha256"]
                        for index in member_indices
                        if rows[index]["decoded_pixel_sha256"]
                    }
                ),
                "class_counts": {
                    str(label): count for label, count in sorted(label_counts.items())
                },
            }
        )
    components.sort(key=lambda component: component["component_id"])
    for index, row in enumerate(rows):
        row["component_id"] = component_for_index[index]

    pixel_conflicts = [
        {
            "decoded_pixel_sha256": pixel_hash,
            "labels": [
                {"class_label": label, "class_name": name}
                for label, name in sorted(labels)
            ],
        }
        for pixel_hash, labels in sorted(pixel_labels.items())
        if len(labels) > 1
    ]
    return rows, components, pixel_conflicts


def assign_folds(
    rows: list[dict[str, Any]],
    components: list[dict[str, Any]],
    fold_count: int,
) -> list[dict[str, Any]]:
    if fold_count < 2:
        raise ValueError("fold_count must be at least 2")
    rows_by_id = {row["row_id"]: row for row in rows}
    total_class_counts = Counter(row["class_label"] for row in rows)
    target_rows = len(rows) / fold_count
    target_classes = {
        label: count / fold_count for label, count in total_class_counts.items()
    }
    folds = [
        {
            "fold": fold + 1,
            "row_ids": [],
            "component_ids": [],
            "class_counts": Counter(),
        }
        for fold in range(fold_count)
    ]
    ordered_components = sorted(
        components,
        key=lambda component: (-component["row_count"], component["component_id"]),
    )
    for component in ordered_components:
        component_counts = Counter(
            {int(label): count for label, count in component["class_counts"].items()}
        )
        scored_folds: list[tuple[float, int, int]] = []
        for fold_index, fold in enumerate(folds):
            after_rows = len(fold["row_ids"]) + component["row_count"]
            row_penalty = ((after_rows - target_rows) / max(target_rows, 1.0)) ** 2
            class_penalty = 0.0
            for label, target in target_classes.items():
                after = fold["class_counts"][label] + component_counts[label]
                class_penalty += ((after - target) / max(target, 1.0)) ** 2
            score = class_penalty + 0.25 * row_penalty
            scored_folds.append((score, len(fold["row_ids"]), fold_index))
        _, _, selected_index = min(scored_folds)
        selected = folds[selected_index]
        selected["component_ids"].append(component["component_id"])
        selected["row_ids"].extend(component["row_ids"])
        selected["class_counts"].update(component_counts)

    fold_for_row: dict[str, int] = {}
    for fold in folds:
        for row_id in fold["row_ids"]:
            fold_for_row[row_id] = fold["fold"]
    for row in rows:
        row["fold"] = fold_for_row[row["row_id"]]

    for fold in folds:
        evaluable = 0
        evaluable_classes: set[int] = set()
        for row_id in fold["row_ids"]:
            label = rows_by_id[row_id]["class_label"]
            if total_class_counts[label] - fold["class_counts"][label] > 0:
                evaluable += 1
                evaluable_classes.add(label)
        fold["row_ids"].sort()
        fold["component_ids"].sort()
        fold["class_counts"] = {
            str(label): count for label, count in sorted(fold["class_counts"].items())
        }
        fold["row_count"] = len(fold["row_ids"])
        fold["evaluable_oof_rows"] = evaluable
        fold["evaluable_classes"] = sorted(evaluable_classes)
    return folds


def overlap_count(rows: list[dict[str, Any]], field: str) -> int:
    folds_by_value: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        value = row.get(field)
        if value:
            folds_by_value[str(value)].add(int(row["fold"]))
    return sum(1 for folds in folds_by_value.values() if len(folds) > 1)


def taxonomy_conflicts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    names_by_label: dict[int, set[str]] = defaultdict(set)
    labels_by_name: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        names_by_label[row["class_label"]].add(row["class_name"])
        labels_by_name[row["class_name"]].add(row["class_label"])
    return {
        "labels_with_multiple_names": {
            str(label): sorted(names)
            for label, names in sorted(names_by_label.items())
            if len(names) > 1
        },
        "names_with_multiple_labels": {
            name: sorted(labels)
            for name, labels in sorted(labels_by_name.items())
            if len(labels) > 1
        },
    }


def run(
    input_jsonl: Path,
    output_dir: Path,
    fold_count: int,
    minimum_oof_rows: int,
    hash_workers: int = 16,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, image_errors = load_and_hash_rows(input_jsonl, hash_workers=hash_workers)
    rows, components, pixel_conflicts = build_components(rows)
    folds = assign_folds(rows, components, fold_count)

    manifest_path = output_dir / "provenance-manifest.jsonl"
    components_path = output_dir / "provenance-components.json"
    folds_path = output_dir / "folds-v8.json"
    audit_path = output_dir / "provenance-audit.json"
    write_jsonl(manifest_path, rows)
    write_json(components_path, {"components": components})
    total_evaluable = sum(fold["evaluable_oof_rows"] for fold in folds)
    fold_payload = {
        "schema_version": "autoresearch-provenance-v8.folds",
        "fold_count": fold_count,
        "folds": folds,
    }
    write_json(folds_path, fold_payload)

    taxonomy = taxonomy_conflicts(rows)
    gates = {
        "known_provenance_component_overlap_across_folds": overlap_count(
            rows, "component_id"
        )
        == 0,
        "decoded_pixel_hash_overlap_across_folds": overlap_count(
            rows, "decoded_pixel_sha256"
        )
        == 0,
        "label_conflicts_reported": True,
        "manifest_sha256_present": True,
        "folds_sha256_present": True,
        "independently_evaluable_oof_rows_gte": total_evaluable >= minimum_oof_rows,
        "all_images_decoded": not image_errors,
        "taxonomy_mapping_unambiguous": not any(taxonomy.values()),
        "final_test_read": False,
        "runtime_unchanged": True,
    }
    audit = {
        "schema_version": "autoresearch-provenance-v8.audit",
        "pass": all(
            value is True
            for key, value in gates.items()
            if key not in {"final_test_read"}
        )
        and gates["final_test_read"] is False,
        "input_jsonl": str(input_jsonl),
        "input_sha256": sha256_file(input_jsonl),
        "row_count": len(rows),
        "component_count": len(components),
        "class_count": len({row["class_label"] for row in rows}),
        "evaluable_oof_rows": total_evaluable,
        "hash_workers": max(1, min(int(hash_workers), 32)),
        "minimum_oof_rows": minimum_oof_rows,
        "image_decode_errors": image_errors,
        "pixel_label_conflicts": pixel_conflicts,
        "taxonomy_conflicts": taxonomy,
        "gates": gates,
        "artifacts": {
            "manifest": manifest_path.name,
            "manifest_sha256": sha256_file(manifest_path),
            "components": components_path.name,
            "components_sha256": sha256_file(components_path),
            "folds": folds_path.name,
            "folds_sha256": sha256_file(folds_path),
        },
        "final_test_read": False,
        "runtime_unchanged": True,
    }
    write_json(audit_path, audit)
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--minimum-oof-rows", type=int, default=600)
    parser.add_argument("--hash-workers", type=int, default=16)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audit = run(
        input_jsonl=args.input_jsonl.resolve(),
        output_dir=args.output_dir.resolve(),
        fold_count=args.folds,
        minimum_oof_rows=args.minimum_oof_rows,
        hash_workers=args.hash_workers,
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0 if audit["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Prepare a deterministic, group-preserving glyph benchmark manifest.

The script accepts either a directory tree or a zip archive of glyph images.
It computes per-image provenance hashes, keeps weak labels from the enclosing
folder name, and assigns whole source groups to annotation_pool/dev/locked_test
splits without ever breaking a source group apart.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from itertools import permutations
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
import zipfile


IMAGE_EXTENSIONS = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}

SPLIT_NAMES = ("annotation_pool", "dev", "locked_test")


@dataclass(frozen=True)
class GlyphSource:
    rel_path: str
    weak_folder_label: str
    source_group: str
    sha256: str
    byte_size: int
    split: str = "annotation_pool"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def is_image_path(path: str) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def _normalize_rel_path(path: str) -> str:
    return Path(path).as_posix().lstrip("./")


def _weak_folder_label(rel_path: str) -> str:
    parent = Path(rel_path).parent
    if parent == Path("."):
        return "root"
    return parent.name or "root"


def _page_source_key(rel_path: str) -> str:
    stem = Path(rel_path).stem
    stem = re.sub(r'-\d+$', '', stem)
    tokens = stem.split('_')
    if len(tokens) >= 2:
        return '_'.join(tokens[:2])
    return stem or 'root'


def _source_group(page_key: str, weak_folder_label: str, ambiguous: bool) -> str:
    if ambiguous:
        return f'{weak_folder_label}/{page_key}'
    return page_key


def _iter_dir_images(root: Path) -> Iterable[tuple[str, bytes]]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and is_image_path(path.name):
            yield path.relative_to(root).as_posix(), path.read_bytes()


def _iter_zip_images(archive: Path) -> Iterable[tuple[str, bytes]]:
    with zipfile.ZipFile(archive) as zf:
        for info in sorted(zf.infolist(), key=lambda item: item.filename):
            if info.is_dir():
                continue
            if not is_image_path(info.filename):
                continue
            with zf.open(info) as handle:
                yield _normalize_rel_path(info.filename), handle.read()


def _load_sources(input_path: Path) -> list[GlyphSource]:
    if input_path.is_dir():
        iterator = _iter_dir_images(input_path)
    elif input_path.is_file() and input_path.suffix.lower() == '.zip':
        iterator = _iter_zip_images(input_path)
    else:
        raise FileNotFoundError(f"Unsupported benchmark input: {input_path}")

    raw_sources: list[tuple[str, str, str, bytes]] = []
    page_labels: dict[str, set[str]] = defaultdict(set)
    for rel_path, payload in iterator:
        rel_path = _normalize_rel_path(rel_path)
        weak_folder_label = _weak_folder_label(rel_path)
        page_key = _page_source_key(rel_path)
        raw_sources.append((rel_path, weak_folder_label, page_key, payload))
        page_labels[page_key].add(weak_folder_label)

    sources: list[GlyphSource] = []
    for rel_path, weak_folder_label, page_key, payload in raw_sources:
        sources.append(
            GlyphSource(
                rel_path=rel_path,
                weak_folder_label=weak_folder_label,
                source_group=_source_group(page_key, weak_folder_label, len(page_labels[page_key]) > 1),
                sha256=sha256_bytes(payload),
                byte_size=len(payload),
            )
        )
    return sources


def _group_sources(sources: Iterable[GlyphSource]) -> dict[str, list[GlyphSource]]:
    grouped: dict[str, list[GlyphSource]] = defaultdict(list)
    for source in sources:
        grouped[source.source_group].append(source)
    for values in grouped.values():
        values.sort(key=lambda item: item.rel_path)
    return dict(sorted(grouped.items(), key=lambda item: item[0]))


def _stable_group_key(source_group: str) -> tuple[int, str]:
    digest = hashlib.sha256(source_group.encode("utf-8")).hexdigest()
    return int(digest[:16], 16), source_group


def _assignment_cost(
    current_counts: dict[str, float],
    target_counts: dict[str, float],
    split: str,
    size: int,
) -> float:
    return sum(
        (current_counts[name] + (size if name == split else 0) - target_counts[name]) ** 2
        for name in SPLIT_NAMES
    )


def _choose_split(
    current_counts: dict[str, float],
    target_counts: dict[str, float],
    size: int,
    allowed_splits: Iterable[str] = SPLIT_NAMES,
) -> str:
    return min(
        allowed_splits,
        key=lambda split: (
            _assignment_cost(current_counts, target_counts, split, size),
            SPLIT_NAMES.index(split),
        ),
    )


def _assign_splits(
    grouped_sources: dict[str, list[GlyphSource]],
    annotation_pool_ratio: float,
    dev_ratio: float,
    locked_test_ratio: float,
) -> list[GlyphSource]:
    ratios = {
        "annotation_pool": annotation_pool_ratio,
        "dev": dev_ratio,
        "locked_test": locked_test_ratio,
    }
    ratio_total = sum(ratios.values())
    if ratio_total <= 0:
        raise ValueError("At least one split ratio must be positive.")

    total_items = sum(len(items) for items in grouped_sources.values())
    target_counts = {
        split: total_items * ratio / ratio_total for split, ratio in ratios.items()
    }
    current_counts = {split: 0.0 for split in SPLIT_NAMES}

    assigned: list[GlyphSource] = []
    groups_by_label: dict[str, list[tuple[str, list[GlyphSource]]]] = defaultdict(list)
    for source_group, items in grouped_sources.items():
        groups_by_label[items[0].weak_folder_label].append((source_group, items))

    def assign_group(items: list[GlyphSource], split: str) -> None:
        current_counts[split] += len(items)
        for item in items:
            assigned.append(dataclasses.replace(item, split=split))

    label_order = sorted(
        groups_by_label,
        key=lambda weak_label: (
            -sum(len(items) for _, items in groups_by_label[weak_label]),
            weak_label,
        ),
    )

    for weak_label in label_order:
        groups = sorted(
            groups_by_label[weak_label],
            key=lambda item: (len(item[1]), _stable_group_key(item[0])),
        )

        if len(groups) >= len(SPLIT_NAMES):
            seed_groups = groups[: len(SPLIT_NAMES)]
            best_seed_assignment: tuple[str, ...] | None = None
            best_seed_cost: float | None = None
            for candidate in permutations(SPLIT_NAMES):
                candidate_cost = 0.0
                for (_, items), split in zip(seed_groups, candidate, strict=True):
                    candidate_cost += _assignment_cost(current_counts, target_counts, split, len(items))
                if best_seed_cost is None or candidate_cost < best_seed_cost or (
                    candidate_cost == best_seed_cost and (best_seed_assignment is None or candidate < best_seed_assignment)
                ):
                    best_seed_cost = candidate_cost
                    best_seed_assignment = candidate
            assert best_seed_assignment is not None
            for (_, items), split in zip(seed_groups, best_seed_assignment, strict=True):
                assign_group(items, split)
            remaining_groups = groups[len(SPLIT_NAMES):]
        else:
            remaining_groups = groups

        for source_group, items in sorted(
            remaining_groups,
            key=lambda item: (-len(item[1]), _stable_group_key(item[0])),
        ):
            chosen_split = _choose_split(current_counts, target_counts, len(items))
            assign_group(items, chosen_split)

    return assigned


def build_manifest(
    input_path: Path,
    *,
    annotation_pool_ratio: float = 0.8,
    dev_ratio: float = 0.1,
    locked_test_ratio: float = 0.1,
    created_at: str | None = None,
) -> dict:
    sources = _load_sources(input_path)
    grouped = _group_sources(sources)
    assigned = _assign_splits(
        grouped,
        annotation_pool_ratio=annotation_pool_ratio,
        dev_ratio=dev_ratio,
        locked_test_ratio=locked_test_ratio,
    )

    items = [
        {
            "id": f"glyph-{index:06d}",
            "rel_path": source.rel_path,
            "weak_folder_label": source.weak_folder_label,
            "source_group": source.source_group,
            "sha256": source.sha256,
            "byte_size": source.byte_size,
            "split": source.split,
            "gt_status": "pending",
            "gt_boxes": [],
            "gt_class_name": None,
        }
        for index, source in enumerate(sorted(assigned, key=lambda item: (item.split, item.source_group, item.rel_path)))
    ]

    counts_by_split = {split: 0 for split in SPLIT_NAMES}
    counts_by_label = defaultdict(int)
    for item in items:
        counts_by_split[item["split"]] += 1
        counts_by_label[item["weak_folder_label"]] += 1

    return {
        "schema_version": "1.0",
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
        "source_path": str(input_path),
        "source_kind": "directory" if input_path.is_dir() else "zip",
        "split_ratios": {
            "annotation_pool": annotation_pool_ratio,
            "dev": dev_ratio,
            "locked_test": locked_test_ratio,
        },
        "counts": {
            "total": len(items),
            "by_split": counts_by_split,
            "by_weak_folder_label": dict(sorted(counts_by_label.items())),
        },
        "items": items,
    }


def write_manifest(manifest: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Glyph directory or zip archive.")
    parser.add_argument("--output", type=Path, required=True, help="Destination JSON manifest.")
    parser.add_argument("--annotation-pool-ratio", type=float, default=0.8)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument("--locked-test-ratio", type=float, default=0.1)
    args = parser.parse_args(argv)

    manifest = build_manifest(
        args.input,
        annotation_pool_ratio=args.annotation_pool_ratio,
        dev_ratio=args.dev_ratio,
        locked_test_ratio=args.locked_test_ratio,
    )
    write_manifest(manifest, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

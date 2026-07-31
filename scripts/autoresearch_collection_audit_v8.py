#!/usr/bin/env python3
"""Audit whether business provenance can safely split v8 components.

The utility is model-free and read-only with respect to source data. It
quarantines exact-pixel label conflicts, reconstructs the provenance graph,
and separates positive documentary identities from technical cache or
directory origins. It never reads a model evaluation or final test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable


IDENTITY_FIELDS = frozenset(
    {
        "collection_id",
        "collection_uuid",
        "manuscript_id",
        "manuscript_uuid",
        "document_id",
        "document_uuid",
        "volume_id",
        "volume_uuid",
        "codex_id",
        "codex_uuid",
    }
)
PATH_FIELDS = ("path", "source_path", "archive_path", "output_path", "image_path")
REGRESSION_CLASSES = ("huitzilin", "petlatl", "atl", "piqui", "tetl", "tilmatli")
THRESHOLDS = (2, 3, 5)
TRAILING_NUMERIC_RE = re.compile(r"-[0-9]+$")


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


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_json_bytes(value))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"line {line_number}: expected an object")
            rows.append(dict(row))
    if not rows:
        raise ValueError("provenance manifest is empty")
    row_ids = [str(row.get("row_id", "")) for row in rows]
    if any(not row_id for row_id in row_ids):
        raise ValueError("every provenance row needs row_id")
    if len(set(row_ids)) != len(row_ids):
        raise ValueError("duplicate row_id in provenance manifest")
    return sorted(rows, key=lambda row: str(row["row_id"]))


def normalize_path(value: str) -> str:
    return value.strip().replace("\\", "/").casefold()


IdentityToken = tuple[tuple[str, str], ...]


def collect_path_metadata(
    document: Any,
) -> tuple[dict[str, set[IdentityToken]], dict[str, str]]:
    identities_by_path: dict[str, set[IdentityToken]] = defaultdict(set)
    origins_by_path: dict[str, str] = {}

    def walk(value: Any, inherited: IdentityToken = ()) -> None:
        if isinstance(value, list):
            for child in value:
                walk(child, inherited)
            return
        if not isinstance(value, dict):
            return
        own = {
            (str(key), str(field_value).strip())
            for key, field_value in value.items()
            if key in IDENTITY_FIELDS
            and field_value is not None
            and str(field_value).strip()
        }
        token = tuple(sorted(set(inherited).union(own)))
        paths = {
            normalize_path(str(value[field]))
            for field in PATH_FIELDS
            if value.get(field)
        }
        for path in paths:
            if token:
                identities_by_path[path].add(token)
            origin = str(value.get("source", "")).strip()
            if origin:
                previous = origins_by_path.get(path)
                origins_by_path[path] = (
                    origin if not previous or previous == origin else "ambiguous-origin"
                )
        for child in value.values():
            if isinstance(child, (dict, list)):
                walk(child, token)

    walk(document)
    return identities_by_path, origins_by_path


def merge_path_metadata(
    documents: Iterable[Any],
) -> tuple[dict[str, set[IdentityToken]], dict[str, str]]:
    identities: dict[str, set[IdentityToken]] = defaultdict(set)
    origins: dict[str, str] = {}
    for document in documents:
        document_identities, document_origins = collect_path_metadata(document)
        for path, tokens in document_identities.items():
            identities[path].update(tokens)
        for path, origin in document_origins.items():
            previous = origins.get(path)
            origins[path] = (
                origin if not previous or previous == origin else "ambiguous-origin"
            )
    return identities, origins


def row_path_keys(row: dict[str, Any]) -> list[str]:
    return sorted(
        {
            normalize_path(str(row[field]))
            for field in PATH_FIELDS
            if row.get(field)
        }
    )


def decorate_rows(
    rows: list[dict[str, Any]],
    identities_by_path: dict[str, set[IdentityToken]],
    origins_by_path: dict[str, str],
) -> list[dict[str, Any]]:
    decorated: list[dict[str, Any]] = []
    for original in rows:
        row = dict(original)
        tokens: set[IdentityToken] = set()
        origins: set[str] = set()
        for path in row_path_keys(row):
            tokens.update(identities_by_path.get(path, set()))
            if path in origins_by_path:
                origins.add(origins_by_path[path])
        row["_identity_token"] = next(iter(tokens)) if len(tokens) == 1 else None
        row["_identity_ambiguous"] = len(tokens) > 1
        cache_name = str(row.get("cache_name", "unknown"))
        if len(origins) == 1:
            row["_technical_origin"] = next(iter(origins))
        elif len(origins) > 1:
            row["_technical_origin"] = "ambiguous-origin"
        elif cache_name == "legacy":
            row["_technical_origin"] = "legacy-archive"
        else:
            row["_technical_origin"] = f"unmapped-{cache_name}"
        decorated.append(row)
    return decorated


FamilyKey = Callable[[dict[str, Any]], Any]


def build_components(
    rows: list[dict[str, Any]],
    family_key: FamilyKey,
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    union_find = UnionFind(len(rows))
    family_owner: dict[Any, int] = {}
    pixel_owner: dict[str, int] = {}
    for index, row in enumerate(rows):
        family = family_key(row)
        if family in family_owner:
            union_find.union(index, family_owner[family])
        else:
            family_owner[family] = index
        pixel_hash = str(row.get("decoded_pixel_sha256", ""))
        if pixel_hash:
            if pixel_hash in pixel_owner:
                union_find.union(index, pixel_owner[pixel_hash])
            else:
                pixel_owner[pixel_hash] = index

    indices_by_root: dict[int, list[int]] = defaultdict(list)
    for index in range(len(rows)):
        indices_by_root[union_find.find(index)].append(index)

    row_to_component: dict[str, str] = {}
    components: list[dict[str, Any]] = []
    for indices in indices_by_root.values():
        row_ids = sorted(str(rows[index]["row_id"]) for index in indices)
        digest = hashlib.sha256("\n".join(row_ids).encode("utf-8")).hexdigest()
        component_id = f"component:{digest[:20]}"
        for index in indices:
            row_to_component[str(rows[index]["row_id"])] = component_id
        components.append(
            {
                "component_id": component_id,
                "row_ids": row_ids,
                "row_count": len(indices),
                "classes": sorted(
                    {str(rows[index]["class_name"]) for index in indices}
                ),
                "source_families": sorted(
                    {str(rows[index]["source_family"]) for index in indices}
                ),
                "technical_origins": sorted(
                    {str(rows[index]["_technical_origin"]) for index in indices}
                ),
                "identity_covered_rows": sum(
                    rows[index]["_identity_token"] is not None for index in indices
                ),
            }
        )
    components.sort(key=lambda item: str(item["component_id"]))
    return row_to_component, components


def graph_stats(
    rows: list[dict[str, Any]],
    row_to_component: dict[str, str],
) -> dict[str, Any]:
    components: dict[str, set[str]] = defaultdict(set)
    folds: dict[str, set[int]] = defaultdict(set)
    row_counts = Counter()
    fold_counts: dict[str, Counter[int]] = defaultdict(Counter)
    for row in rows:
        name = str(row["class_name"])
        fold = int(row["fold"])
        components[name].add(row_to_component[str(row["row_id"])])
        folds[name].add(fold)
        row_counts[name] += 1
        fold_counts[name][fold] += 1

    per_class: list[dict[str, Any]] = []
    for name in sorted(row_counts):
        oof_rows = sum(
            count
            for fold, count in fold_counts[name].items()
            if row_counts[name] - count > 0
        )
        per_class.append(
            {
                "class_name": name,
                "row_count": row_counts[name],
                "component_count": len(components[name]),
                "folds": sorted(folds[name]),
                "evaluable_oof_rows": oof_rows,
            }
        )
    return {
        "class_count": len(per_class),
        "classes_at_least_components": {
            str(threshold): sum(
                item["component_count"] >= threshold for item in per_class
            )
            for threshold in THRESHOLDS
        },
        "classes_at_least_two_folds": sum(
            len(item["folds"]) >= 2 for item in per_class
        ),
        "evaluable_class_count": sum(
            item["evaluable_oof_rows"] > 0 for item in per_class
        ),
        "evaluable_oof_rows": sum(item["evaluable_oof_rows"] for item in per_class),
        "per_class": per_class,
    }


def quarantine_conflicts(
    rows: list[dict[str, Any]],
    provenance_audit: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    hashes = {
        str(item["decoded_pixel_sha256"])
        for item in provenance_audit.get("pixel_label_conflicts", [])
        if item.get("decoded_pixel_sha256")
    }
    quarantined = [
        row for row in rows if str(row.get("decoded_pixel_sha256", "")) in hashes
    ]
    retained = [
        row for row in rows if str(row.get("decoded_pixel_sha256", "")) not in hashes
    ]
    return retained, {
        "conflicting_hash_count": len(hashes),
        "quarantined_row_count": len(quarantined),
        "affected_class_count": len(
            {str(row["class_name"]) for row in quarantined}
        ),
        "affected_component_count": len(
            {str(row["component_id"]) for row in quarantined}
        ),
        "affected_folds": sorted({int(row["fold"]) for row in quarantined}),
        "affected_classes": sorted(
            {str(row["class_name"]) for row in quarantined}
        ),
        "row_ids": sorted(str(row["row_id"]) for row in quarantined),
    }


def source_instance_overlap(rows: list[dict[str, Any]]) -> dict[str, int]:
    origins_by_key: dict[tuple[str, str], set[str]] = defaultdict(set)
    rows_by_key = Counter()
    for row in rows:
        source_path = str(row.get("source_path", ""))
        if not source_path:
            continue
        stem = PurePosixPath(source_path.replace("\\", "/")).stem.casefold()
        key = (str(row["class_name"]), stem)
        origins_by_key[key].add(str(row["_technical_origin"]))
        rows_by_key[key] += 1
    overlapping = {
        key for key, origins in origins_by_key.items() if len(origins) > 1
    }
    return {
        "cross_origin_instance_key_count": len(overlapping),
        "rows_on_cross_origin_instance_keys": sum(
            rows_by_key[key] for key in overlapping
        ),
    }


def collapse_nested_numeric_suffixes(value: str) -> str:
    collapsed = value
    while True:
        updated = TRAILING_NUMERIC_RE.sub("", collapsed)
        if updated == collapsed:
            return collapsed
        collapsed = updated


def acquisition_targets(stats: dict[str, Any]) -> dict[str, Any]:
    targets: list[dict[str, Any]] = []
    aggregate = {str(threshold): 0 for threshold in THRESHOLDS}
    for item in stats["per_class"]:
        current = int(item["component_count"])
        gaps = {
            str(threshold): max(0, threshold - current)
            for threshold in THRESHOLDS
        }
        for threshold, gap in gaps.items():
            aggregate[threshold] += gap
        targets.append(
            {
                "class_name": item["class_name"],
                "current_independent_components": current,
                "additional_component_slots_needed": gaps,
            }
        )
    return {
        "schema_version": "autoresearch-acquisition-targets-v8.1",
        "unit": "class-component slots; one acquired page may cover several classes",
        "threshold_meanings": {
            "2": "leave-one-component-out observation only",
            "3": "minimum train, inner selection and external evaluation separation",
            "5": "minimum practical start for per-class variance estimation",
        },
        "aggregate_additional_component_slots_needed": aggregate,
        "priority_regression_classes": [
            item for item in targets if item["class_name"] in REGRESSION_CLASSES
        ],
        "targets": targets,
        "final_test_read": False,
        "runtime_unchanged": True,
    }


def audit(
    rows: list[dict[str, Any]],
    input_components: dict[str, Any],
    provenance_audit: dict[str, Any],
    metadata_documents: list[Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    identities, origins = merge_path_metadata(metadata_documents)
    decorated = decorate_rows(rows, identities, origins)
    retained, quarantine = quarantine_conflicts(decorated, provenance_audit)

    conservative_map, conservative_components = build_components(
        retained, lambda row: str(row["source_family"])
    )
    conservative_stats = graph_stats(retained, conservative_map)

    rows_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in retained:
        rows_by_family[str(row["source_family"])].append(row)
    splittable_families = {
        family
        for family, family_rows in rows_by_family.items()
        if all(row["_identity_token"] is not None for row in family_rows)
        and len({row["_identity_token"] for row in family_rows}) >= 2
    }

    def positive_key(row: dict[str, Any]) -> Any:
        family = str(row["source_family"])
        return (
            (family, row["_identity_token"])
            if family in splittable_families
            else family
        )

    positive_map, positive_components = build_components(retained, positive_key)
    positive_stats = graph_stats(retained, positive_map)
    positive_parts: dict[str, set[str]] = defaultdict(set)
    for row in retained:
        row_id = str(row["row_id"])
        positive_parts[conservative_map[row_id]].add(positive_map[row_id])
    supported_splits = sorted(
        component_id
        for component_id, parts in positive_parts.items()
        if len(parts) > 1
    )

    technical_map, technical_components = build_components(
        retained,
        lambda row: (str(row["_technical_origin"]), str(row["source_family"])),
    )
    technical_stats = graph_stats(retained, technical_map)

    collapsed_map, collapsed_components = build_components(
        retained,
        lambda row: collapse_nested_numeric_suffixes(str(row["source_family"])),
    )
    collapsed_stats = graph_stats(retained, collapsed_map)
    families_by_collapsed: dict[str, set[str]] = defaultdict(set)
    for row in retained:
        family = str(row["source_family"])
        families_by_collapsed[collapse_nested_numeric_suffixes(family)].add(family)
    nested_alias_groups = [
        {"collapsed_family": base, "source_families": sorted(families)}
        for base, families in sorted(families_by_collapsed.items())
        if len(families) > 1
    ]

    covered_rows = sum(row["_identity_token"] is not None for row in decorated)
    origin_counts = Counter(str(row["_technical_origin"]) for row in decorated)
    largest = sorted(
        conservative_components,
        key=lambda item: (-int(item["row_count"]), str(item["component_id"])),
    )[:20]
    gates = {
        "explicit_collection_or_document_identity_coverage_gt": (
            covered_rows / len(decorated) > 0.0
        ),
        "positively_supported_component_splits_gte": len(supported_splits) >= 1,
        "classes_with_at_least_two_components_after_quarantine_gte": (
            positive_stats["classes_at_least_components"]["2"] >= 20
        ),
        "conflicting_hashes_quarantined": (
            len(retained) + quarantine["quarantined_row_count"] == len(decorated)
        ),
        "source_rows_modified": 0,
        "final_test_read": False,
        "runtime_unchanged": True,
    }
    passed = (
        gates["explicit_collection_or_document_identity_coverage_gt"]
        and gates["positively_supported_component_splits_gte"]
        and gates["classes_with_at_least_two_components_after_quarantine_gte"]
        and gates["conflicting_hashes_quarantined"]
    )
    regression_by_name = {
        item["class_name"]: item for item in conservative_stats["per_class"]
    }
    report = {
        "schema_version": "autoresearch-collection-provenance-audit-v8.1",
        "row_count": len(decorated),
        "input_component_count": len(input_components.get("components", [])),
        "identity_evidence": {
            "accepted_fields": sorted(IDENTITY_FIELDS),
            "covered_row_count": covered_rows,
            "coverage": covered_rows / len(decorated),
            "ambiguous_row_count": sum(
                row["_identity_ambiguous"] for row in decorated
            ),
            "positively_splittable_family_count": len(splittable_families),
            "positively_supported_component_split_count": len(supported_splits),
            "positively_supported_component_ids": supported_splits,
            "technical_origin_is_positive_identity": False,
        },
        "technical_origins": {
            "row_counts": dict(sorted(origin_counts.items())),
            **source_instance_overlap(decorated),
        },
        "quarantine": quarantine,
        "conservative_after_quarantine": {
            "component_count": len(conservative_components),
            **conservative_stats,
        },
        "positive_identity_revised": {
            "component_count": len(positive_components),
            **positive_stats,
        },
        "technical_origin_separated_upper_bound": {
            "scientifically_valid": False,
            "reason": (
                "cache and directory roots are not collection or manuscript identities"
            ),
            "component_count": len(technical_components),
            **technical_stats,
        },
        "nested_numeric_suffix_sensitivity": {
            "scientifically_valid": False,
            "reason": (
                "repeated numeric suffixes may be page sub-instances, but no "
                "document grammar proves that interpretation"
            ),
            "alias_group_count": len(nested_alias_groups),
            "alias_groups": nested_alias_groups,
            "component_count": len(collapsed_components),
            **collapsed_stats,
        },
        "largest_conservative_components": largest,
        "regression_class_overlap": [
            {
                "class_name": name,
                **regression_by_name.get(
                    name,
                    {
                        "row_count": 0,
                        "component_count": 0,
                        "folds": [],
                        "evaluable_oof_rows": 0,
                    },
                ),
            }
            for name in REGRESSION_CLASSES
        ],
        "gates_without_replay": gates,
        "pass_without_replay": passed,
        "source_rows_modified": 0,
        "final_test_read": False,
        "runtime_unchanged": True,
        "promotion_eligible": False,
    }
    return report, acquisition_targets(conservative_stats)


def prepare_output_dir(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)


def run(
    provenance_manifest: Path,
    provenance_components: Path,
    provenance_audit: Path,
    legacy_manifest: Path,
    external_snapshot: Path,
    approved_import_manifest: Path,
    output_dir: Path,
) -> dict[str, Any]:
    prepare_output_dir(output_dir)
    input_paths = {
        "provenance_manifest": provenance_manifest,
        "provenance_components": provenance_components,
        "provenance_audit": provenance_audit,
        "legacy_manifest": legacy_manifest,
        "external_snapshot": external_snapshot,
        "approved_import_manifest": approved_import_manifest,
    }
    rows = load_jsonl(provenance_manifest)
    input_components = load_json(provenance_components)
    audit_input = load_json(provenance_audit)
    documents = [
        load_json(legacy_manifest),
        load_json(external_snapshot),
        load_json(approved_import_manifest),
    ]
    report, targets = audit(rows, input_components, audit_input, documents)
    report["inputs"] = {
        name: {"path": str(path), "sha256": sha256_file(path)}
        for name, path in sorted(input_paths.items())
    }
    report["script_sha256"] = sha256_file(Path(__file__))
    write_json(output_dir / "acquisition-targets.json", targets)
    report["acquisition_targets_sha256"] = sha256_file(
        output_dir / "acquisition-targets.json"
    )
    write_json(output_dir / "collection-provenance-audit.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provenance-manifest", type=Path, required=True)
    parser.add_argument("--provenance-components", type=Path, required=True)
    parser.add_argument("--provenance-audit", type=Path, required=True)
    parser.add_argument("--legacy-manifest", type=Path, required=True)
    parser.add_argument("--external-snapshot", type=Path, required=True)
    parser.add_argument("--approved-import-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(
        provenance_manifest=args.provenance_manifest,
        provenance_components=args.provenance_components,
        provenance_audit=args.provenance_audit,
        legacy_manifest=args.legacy_manifest,
        external_snapshot=args.external_snapshot,
        approved_import_manifest=args.approved_import_manifest,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "pass_without_replay": result["pass_without_replay"],
                "identity_coverage": result["identity_evidence"]["coverage"],
                "supported_component_splits": result["identity_evidence"][
                    "positively_supported_component_split_count"
                ],
                "classes_at_least_two": result["positive_identity_revised"][
                    "classes_at_least_components"
                ]["2"],
                "output_dir": str(args.output_dir),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

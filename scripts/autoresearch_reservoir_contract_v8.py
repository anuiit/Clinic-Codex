#!/usr/bin/env python3
"""Audit non-imported image reservoirs under a positive-provenance contract.

This utility is deliberately model-free and read-only with respect to its inputs.
Technical similarity may reject or quarantine a candidate, but it can never prove
that the candidate is an independent documentary source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageOps


POSITIVE_IDENTITY_FIELDS = (
    "collection_id",
    "manuscript_id",
    "document_id",
    "volume_id",
    "codex_id",
)
SECONDARY_DOCUMENT_FIELDS = (
    "manuscript_id",
    "document_id",
    "volume_id",
    "codex_id",
)
REQUIRED_IDENTITY_CONTEXT = (
    "namespaced_page_id",
    "crop_instance_id",
    "scan_batch_id",
    "parent_asset_id",
    "derivation_relation",
    "evidence_reference",
)
ALLOWED_INDEPENDENT_RELATIONS = {
    "independent_acquisition",
    "independent_original_scan",
    "proven_distinct_source",
}
FORBIDDEN_INDEPENDENCE_EVIDENCE = (
    "cache_or_directory_root",
    "filename_or_stem",
    "numeric_or_alphabetic_suffix",
    "different_file_hash",
    "different_decoded_pixel_hash",
    "absence_of_near_duplicate",
)
V7_REGRESSION_CLASSES = {
    "atl",
    "huitzilin",
    "petlatl",
    "piqui",
    "tetl",
    "tilmatli",
}


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_json_bytes(value))


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


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
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            rows.append(row)
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_id(prefix: str, value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]
    return f"{prefix}:{digest}"


def normalized_text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def resolve_image_path(value: Any, manifest_path: Path) -> Path:
    raw = normalized_text(value)
    if not raw:
        raise ValueError("image path is empty")
    path = Path(raw)
    return path if path.is_absolute() else (manifest_path.parent / path).resolve()


def difference_hash(rgb: Image.Image) -> str:
    grayscale = rgb.resize((9, 8), Image.Resampling.LANCZOS).convert("L")
    pixels = list(grayscale.get_flattened_data())
    bits = 0
    for row in range(8):
        offset = row * 9
        for column in range(8):
            bits = (bits << 1) | int(
                pixels[offset + column] > pixels[offset + column + 1]
            )
    return f"{bits:016x}"


def image_fingerprint(path: Path) -> dict[str, Any]:
    try:
        with Image.open(path) as image:
            rgb = ImageOps.exif_transpose(image).convert("RGB")
            width, height = rgb.size
            digest = hashlib.sha256()
            digest.update(width.to_bytes(8, "big"))
            digest.update(height.to_bytes(8, "big"))
            digest.update(rgb.tobytes())
            return {
                "decoded_rgb_sha256": digest.hexdigest(),
                "dhash64": difference_hash(rgb),
                "width": width,
                "height": height,
                "error": None,
            }
    except Exception as exc:
        return {
            "decoded_rgb_sha256": None,
            "dhash64": None,
            "width": None,
            "height": None,
            "error": f"{exc.__class__.__name__}:{exc}",
        }


def fingerprint_paths(paths: Iterable[Path], workers: int) -> dict[str, dict[str, Any]]:
    ordered = sorted({str(path): path for path in paths}.items())

    def worker(item: tuple[str, Path]) -> tuple[str, dict[str, Any]]:
        key, path = item
        return key, image_fingerprint(path)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        return dict(executor.map(worker, ordered))


def extract_identity(row: dict[str, Any]) -> tuple[dict[str, str], bool, list[str]]:
    fields = POSITIVE_IDENTITY_FIELDS + REQUIRED_IDENTITY_CONTEXT
    identity = {field: normalized_text(row.get(field)) for field in fields}
    missing: list[str] = []

    if not identity["collection_id"]:
        missing.append("collection_id")
    if not any(identity[field] for field in SECONDARY_DOCUMENT_FIELDS):
        missing.append("one_of_manuscript_document_volume_codex")
    for field in REQUIRED_IDENTITY_CONTEXT:
        if not identity[field]:
            missing.append(field)
    if (
        identity["derivation_relation"]
        and identity["derivation_relation"] not in ALLOWED_INDEPENDENT_RELATIONS
    ):
        missing.append("independent_derivation_relation")

    return identity, not missing, sorted(set(missing))


def identity_component_token(identity: dict[str, str]) -> str:
    component_fields = {
        key: identity[key]
        for key in (
            "collection_id",
            "manuscript_id",
            "document_id",
            "volume_id",
            "codex_id",
            "namespaced_page_id",
            "scan_batch_id",
            "parent_asset_id",
        )
    }
    return stable_id(
        "provenance-component",
        json.dumps(component_fields, ensure_ascii=False, sort_keys=True),
    )


def prepare_output_dir(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)


def load_candidates(import_manifest: Path) -> list[dict[str, Any]]:
    payload = load_json(import_manifest)
    actions = payload.get("actions")
    if not isinstance(actions, list):
        raise ValueError("approved import manifest needs an actions list")
    candidates: list[dict[str, Any]] = []
    for index, raw in enumerate(actions):
        if not isinstance(raw, dict):
            raise ValueError(f"actions[{index}] must be an object")
        if normalized_text(raw.get("action")) == "import":
            continue
        candidate = dict(raw)
        candidate["_manifest_index"] = index
        candidates.append(candidate)
    return sorted(
        candidates,
        key=lambda row: (
            normalized_text(row.get("source")),
            normalized_text(row.get("path")),
            int(row["_manifest_index"]),
        ),
    )


def imported_references(
    imported_manifest: Path,
    fingerprints: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    by_path: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(imported_manifest):
        raw_path = row.get("image_path") or row.get("output_path") or row.get("source_path")
        if not raw_path:
            continue
        path = resolve_image_path(raw_path, imported_manifest)
        key = str(path)
        reference = by_path.setdefault(
            key,
            {
                "kind": "imported",
                "path": key,
                "classes": set(),
                "row_ids": [],
            },
        )
        class_name = normalized_text(row.get("class_name"))
        if class_name:
            reference["classes"].add(class_name)
        if row.get("row_id"):
            reference["row_ids"].append(normalized_text(row["row_id"]))

    references: list[dict[str, Any]] = []
    for path, raw in sorted(by_path.items()):
        fingerprint = fingerprints[path]
        if fingerprint["error"]:
            continue
        references.append(
            {
                "kind": "imported",
                "path": path,
                "classes": sorted(raw["classes"]),
                "row_ids": sorted(raw["row_ids"])[:5],
                "decoded_rgb_sha256": fingerprint["decoded_rgb_sha256"],
                "dhash64": fingerprint["dhash64"],
            }
        )
    return references


def near_matches(
    dhash64: str,
    references: list[dict[str, Any]],
    threshold: int,
) -> tuple[int | None, int, list[dict[str, Any]]]:
    value = int(dhash64, 16)
    minimum: int | None = None
    matches: list[tuple[int, dict[str, Any]]] = []
    count = 0
    for reference in references:
        distance = (value ^ int(reference["dhash64"], 16)).bit_count()
        if minimum is None or distance < minimum:
            minimum = distance
        if distance <= threshold:
            count += 1
            matches.append((distance, reference))
    matches.sort(key=lambda item: (item[0], item[1]["kind"], item[1]["path"]))
    evidence = [
        {
            "distance": distance,
            "kind": reference["kind"],
            "path": reference["path"],
            "classes": reference.get("classes", []),
        }
        for distance, reference in matches[:5]
    ]
    return minimum, count, evidence


def build_contract(
    input_hashes: dict[str, str],
    expected_candidates: int,
    near_threshold: int,
) -> dict[str, Any]:
    return {
        "schema_version": "positive-provenance-admission-contract.v1",
        "scientific_label": (
            "Provenance cataloging and descriptive corpus-coverage audit; "
            "no model-performance inference."
        ),
        "candidate_population": {
            "definition": "approved import-manifest actions whose action is not import",
            "expected_rows": expected_candidates,
        },
        "positive_identity_fields": list(POSITIVE_IDENTITY_FIELDS),
        "required_identity_context": list(REQUIRED_IDENTITY_CONTEXT),
        "allowed_independent_relations": sorted(ALLOWED_INDEPENDENT_RELATIONS),
        "forbidden_independence_evidence": list(
            FORBIDDEN_INDEPENDENCE_EVIDENCE
        ),
        "technical_checks": {
            "exact": "EXIF-transposed decoded RGB dimensions and bytes",
            "near_duplicate": {
                "algorithm": "dhash64",
                "hamming_distance_lte": near_threshold,
                "scientifically_valid_as_independence_proof": False,
            },
        },
        "decision_lattice": [
            "inadmissible",
            "duplicate",
            "near_duplicate",
            "proven_independent",
            "unknown",
        ],
        "model_contract_separate": True,
        "final_test_read": False,
        "runtime_unchanged": True,
        "input_sha256": input_hashes,
    }


def disposition_for_candidate(
    candidate: dict[str, Any],
    fingerprint: dict[str, Any],
    exact_matches: list[dict[str, Any]],
    near_minimum: int | None,
    near_count: int,
    identity_complete: bool,
) -> tuple[str, str]:
    action = normalized_text(candidate.get("action"))
    reason = normalized_text(candidate.get("reason"))
    original_status = normalized_text(candidate.get("original_status"))
    approved_class = normalized_text(candidate.get("approved_class"))

    if fingerprint["error"]:
        return "inadmissible", "image_unreadable"
    if (
        action == "quarantine"
        or not approved_class
        or "conflict" in reason
        or "conflict" in original_status
        or "unmapped" in reason
        or "unmapped" in original_status
    ):
        return "inadmissible", reason or original_status or "schema_or_taxonomy"
    if exact_matches:
        return "duplicate", "exact_decoded_rgb_match"
    if near_count:
        return "near_duplicate", f"dhash64_hamming_lte:{near_minimum}"
    if identity_complete:
        return "proven_independent", "positive_documentary_identity"
    return "unknown", "positive_documentary_identity_missing"


def revised_acquisition_deficits(
    acquisition_payload: dict[str, Any],
    slot_keys: set[tuple[str, str]],
) -> dict[str, Any]:
    added_by_class = Counter(class_name for class_name, _ in slot_keys)
    before = {
        str(key): int(value)
        for key, value in acquisition_payload.get(
            "aggregate_additional_component_slots_needed", {}
        ).items()
    }
    after = {str(threshold): 0 for threshold in (2, 3, 5)}
    for target in acquisition_payload.get("targets", []):
        class_name = normalized_text(target.get("class_name"))
        current = int(target.get("current_independent_components", 0))
        revised = current + added_by_class[class_name]
        for threshold in (2, 3, 5):
            after[str(threshold)] += max(0, threshold - revised)
    return {
        "before": before,
        "after": after,
        "positively_proven_slots_by_class": dict(sorted(added_by_class.items())),
    }


def audit_reservoir(
    import_manifest: Path,
    external_audit: Path,
    imported_manifest: Path,
    acquisition_targets: Path,
    expected_candidates: int = 897,
    near_threshold: int = 4,
    hash_workers: int = 16,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    candidates = load_candidates(import_manifest)
    imported_rows = load_jsonl(imported_manifest)
    external_payload = load_json(external_audit)
    acquisition_payload = load_json(acquisition_targets)

    imported_paths = [
        resolve_image_path(
            row.get("image_path") or row.get("output_path") or row.get("source_path"),
            imported_manifest,
        )
        for row in imported_rows
        if row.get("image_path") or row.get("output_path") or row.get("source_path")
    ]
    candidate_paths = [
        resolve_image_path(row.get("path"), import_manifest) for row in candidates
    ]
    fingerprints = fingerprint_paths(
        [*imported_paths, *candidate_paths],
        workers=hash_workers,
    )
    imported = imported_references(imported_manifest, fingerprints)

    exact_index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    comparison_references: list[dict[str, Any]] = []
    for reference in imported:
        exact_index[reference["decoded_rgb_sha256"]].append(reference)
        comparison_references.append(reference)

    catalog: list[dict[str, Any]] = []
    slot_keys: set[tuple[str, str]] = set()
    for candidate in candidates:
        path = resolve_image_path(candidate.get("path"), import_manifest)
        path_key = str(path)
        fingerprint = fingerprints[path_key]
        identity, identity_complete, identity_missing = extract_identity(candidate)
        exact = (
            exact_index.get(fingerprint["decoded_rgb_sha256"], [])
            if fingerprint["decoded_rgb_sha256"]
            else []
        )

        near_minimum: int | None = None
        near_count = 0
        near_evidence: list[dict[str, Any]] = []
        if fingerprint["dhash64"] and not exact:
            near_minimum, near_count, near_evidence = near_matches(
                fingerprint["dhash64"],
                comparison_references,
                near_threshold,
            )

        disposition, disposition_reason = disposition_for_candidate(
            candidate,
            fingerprint,
            exact,
            near_minimum,
            near_count,
            identity_complete,
        )
        class_name = normalized_text(candidate.get("approved_class"))
        component_token = (
            identity_component_token(identity)
            if disposition == "proven_independent"
            else None
        )
        if component_token and class_name:
            slot_keys.add((class_name, component_token))

        if disposition == "duplicate":
            evidence_reference = (
                f"decoded_rgb_sha256:{fingerprint['decoded_rgb_sha256']}"
            )
        elif disposition == "near_duplicate":
            evidence_reference = (
                f"dhash64:{fingerprint['dhash64']};distance:{near_minimum}"
            )
        elif disposition == "proven_independent":
            evidence_reference = identity["evidence_reference"]
        else:
            evidence_reference = (
                f"approved-import-manifest:actions[{candidate['_manifest_index']}]"
            )

        row = {
            "candidate_id": stable_id(
                "reservoir-candidate",
                f"{normalized_text(candidate.get('source'))}\0{path_key}",
            ),
            "manifest_index": candidate["_manifest_index"],
            "path": path_key,
            "source": normalized_text(candidate.get("source")),
            "source_folder": normalized_text(candidate.get("source_folder")),
            "action": normalized_text(candidate.get("action")),
            "original_status": normalized_text(candidate.get("original_status")),
            "manifest_reason": normalized_text(candidate.get("reason")),
            "approved_class": class_name,
            "is_v7_regression_class": class_name in V7_REGRESSION_CLASSES,
            "decoded_rgb_sha256": fingerprint["decoded_rgb_sha256"],
            "dhash64": fingerprint["dhash64"],
            "width": fingerprint["width"],
            "height": fingerprint["height"],
            "decode_error": fingerprint["error"],
            "exact_match_count": len(exact),
            "exact_match_evidence": [
                {
                    "kind": reference["kind"],
                    "path": reference["path"],
                    "classes": reference.get("classes", []),
                }
                for reference in sorted(
                    exact, key=lambda item: (item["kind"], item["path"])
                )[:5]
            ],
            "near_duplicate_min_hamming": near_minimum,
            "near_duplicate_match_count": near_count,
            "near_duplicate_evidence": near_evidence,
            "positive_identity": identity,
            "identity_complete": identity_complete,
            "identity_missing": identity_missing,
            "disposition": disposition,
            "disposition_reason": disposition_reason,
            "evidence_reference": evidence_reference,
            "documented_disposition": bool(
                disposition and disposition_reason and evidence_reference
            ),
            "provenance_component_id": component_token,
            "forbidden_independence_evidence_used": False,
        }
        catalog.append(row)

        if fingerprint["decoded_rgb_sha256"] and fingerprint["dhash64"]:
            reference = {
                "kind": "candidate",
                "path": path_key,
                "classes": [class_name] if class_name else [],
                "row_ids": [row["candidate_id"]],
                "decoded_rgb_sha256": fingerprint["decoded_rgb_sha256"],
                "dhash64": fingerprint["dhash64"],
            }
            exact_index[fingerprint["decoded_rgb_sha256"]].append(reference)
            comparison_references.append(reference)

    dispositions = Counter(row["disposition"] for row in catalog)
    documented_count = sum(row["documented_disposition"] for row in catalog)
    coverage = documented_count / len(catalog) if catalog else 0.0
    regression_slots = {
        key for key in slot_keys if key[0] in V7_REGRESSION_CLASSES
    }
    forbidden_admissions = sum(
        row["disposition"] == "proven_independent"
        and row["forbidden_independence_evidence_used"]
        for row in catalog
    )
    duplicate_admissions = sum(
        row["disposition"] == "proven_independent"
        and (row["exact_match_count"] or row["near_duplicate_match_count"])
        for row in catalog
    )
    base_gates = {
        "candidate_row_count_equals_expected": len(catalog) == expected_candidates,
        "documented_disposition_coverage_gte_0_95": coverage >= 0.95,
        "positively_proven_new_component_slots_gte_5": len(slot_keys) >= 5,
        "v7_regression_component_slots_gte_1": len(regression_slots) >= 1,
        "admissions_using_forbidden_evidence_equals_0": forbidden_admissions == 0,
        "exact_or_near_duplicates_admitted_equals_0": duplicate_admissions == 0,
        "source_rows_modified_equals_0": True,
        "final_test_unread": True,
        "runtime_unchanged": True,
    }

    input_hashes = {
        "approved_import_manifest": sha256_file(import_manifest),
        "external_corpus_audit": sha256_file(external_audit),
        "imported_provenance_manifest": sha256_file(imported_manifest),
        "acquisition_targets": sha256_file(acquisition_targets),
    }
    contract = build_contract(input_hashes, expected_candidates, near_threshold)
    audit = {
        "schema_version": "reservoir-positive-provenance-audit.v1",
        "scientific_label": contract["scientific_label"],
        "candidate_rows": len(catalog),
        "imported_reference_paths": len(imported),
        "imported_decode_errors": len(set(map(str, imported_paths))) - len(imported),
        "candidate_decode_errors": sum(bool(row["decode_error"]) for row in catalog),
        "documented_disposition_count": documented_count,
        "documented_disposition_coverage": coverage,
        "disposition_counts": dict(sorted(dispositions.items())),
        "by_manifest_reason": dict(
            sorted(Counter(row["manifest_reason"] for row in catalog).items())
        ),
        "by_source": dict(
            sorted(Counter(row["source"] for row in catalog).items())
        ),
        "positively_proven_row_count": sum(
            row["disposition"] == "proven_independent" for row in catalog
        ),
        "positively_proven_new_component_slots": len(slot_keys),
        "v7_regression_component_slots": len(regression_slots),
        "v7_regression_classes_with_new_slots": sorted(
            {class_name for class_name, _ in regression_slots}
        ),
        "strong_signal": (
            len(slot_keys) >= 15 and len(regression_slots) >= 3
        ),
        "reservoir_stop": len(slot_keys) < 5 or len(regression_slots) == 0,
        "reservoir_stop_reason": (
            "fewer_than_five_proven_slots_or_zero_regression_slots"
            if len(slot_keys) < 5 or len(regression_slots) == 0
            else None
        ),
        "base_gates": base_gates,
        "pass_without_replay": all(base_gates.values()),
        "deterministic_replay_required": True,
        "model_go_triggered": False,
        "promotion_eligible": False,
        "final_test_read": False,
        "runtime_unchanged": True,
        "source_rows_modified": 0,
        "external_audit_context": {
            "schema_version": external_payload.get("schema_version"),
            "inventory_count": external_payload.get("inventory_count"),
            "pixel_duplicate_group_count": len(
                external_payload.get("pixel_duplicate_groups", [])
            ),
        },
        "revised_acquisition_deficits": revised_acquisition_deficits(
            acquisition_payload,
            slot_keys,
        ),
        "input_sha256": input_hashes,
    }
    return contract, catalog, audit


def run(
    import_manifest: Path,
    external_audit: Path,
    imported_manifest: Path,
    acquisition_targets: Path,
    output_dir: Path,
    expected_candidates: int = 897,
    near_threshold: int = 4,
    hash_workers: int = 16,
) -> dict[str, Any]:
    prepare_output_dir(output_dir)
    contract, catalog, audit = audit_reservoir(
        import_manifest=import_manifest,
        external_audit=external_audit,
        imported_manifest=imported_manifest,
        acquisition_targets=acquisition_targets,
        expected_candidates=expected_candidates,
        near_threshold=near_threshold,
        hash_workers=hash_workers,
    )
    contract_path = output_dir / "provenance-admission-contract.json"
    catalog_path = output_dir / "reservoir-catalog.jsonl"
    write_json(contract_path, contract)
    write_jsonl(catalog_path, catalog)
    audit["artifact_sha256"] = {
        "provenance_admission_contract": sha256_file(contract_path),
        "reservoir_catalog": sha256_file(catalog_path),
    }
    write_json(output_dir / "reservoir-audit.json", audit)
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--import-manifest", type=Path, required=True)
    parser.add_argument("--external-audit", type=Path, required=True)
    parser.add_argument("--imported-manifest", type=Path, required=True)
    parser.add_argument("--acquisition-targets", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-candidates", type=int, default=897)
    parser.add_argument("--near-threshold", type=int, default=4)
    parser.add_argument("--hash-workers", type=int, default=16)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audit = run(
        import_manifest=args.import_manifest,
        external_audit=args.external_audit,
        imported_manifest=args.imported_manifest,
        acquisition_targets=args.acquisition_targets,
        output_dir=args.output_dir,
        expected_candidates=args.expected_candidates,
        near_threshold=args.near_threshold,
        hash_workers=args.hash_workers,
    )
    print(
        json.dumps(
            {
                "candidate_rows": audit["candidate_rows"],
                "documented_disposition_coverage": audit[
                    "documented_disposition_coverage"
                ],
                "positively_proven_new_component_slots": audit[
                    "positively_proven_new_component_slots"
                ],
                "v7_regression_component_slots": audit[
                    "v7_regression_component_slots"
                ],
                "pass_without_replay": audit["pass_without_replay"],
                "reservoir_stop": audit["reservoir_stop"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

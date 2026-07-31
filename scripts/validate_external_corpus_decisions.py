#!/usr/bin/env python3
"""Validate human decisions for an external corpus import plan.

The validator is the gate between review and any future import step. It reads an
`import_manifest.preview.json` plus a reviewer decisions CSV. If every required
item has a valid non-contradictory decision, it can write an
`import_manifest.approved.json`. The approved manifest is still read-only; it is
only a contract for a later import script.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

VALIDATION_SCHEMA_VERSION = "external-corpus-review-validation-v1"
APPROVED_SCHEMA_VERSION = "external-corpus-approved-import-manifest.v1"
REQUIRED_HEADERS = {
    "item_type",
    "item_id",
    "current_status",
    "source_dataset",
    "source_folder",
    "path",
    "active_class_name",
    "decision",
    "target_class",
    "notes",
    "reviewer",
}
DECISIONS_BY_TYPE = {
    "image_conflict": {"keep_as", "quarantine"},
    "invalid_image": {"quarantine"},
    "unmapped_class": {"map_to_existing", "create_new_class", "quarantine"},
    "weak_class": {"accept_weak", "exclude_until_more_data"},
}
IMPORTABLE_BASE_STATUSES = {"selected"}
EXCLUDED_BASE_STATUSES = {"duplicate_same_class", "suspected_reencode"}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header")
        missing = REQUIRED_HEADERS - set(reader.fieldnames)
        if missing:
            raise ValueError(f"{path} missing headers: {sorted(missing)}")
        return [dict(row) for row in reader]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def required_items(manifest: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    items: dict[tuple[str, str], dict[str, Any]] = {}
    weak_classes = set((manifest.get("class_counts") or {}).keys())
    # Weak classes are sourced from preview manifest class_counts + weak rows are
    # not embedded in the manifest. The generated template is authoritative for
    # weak rows; validation handles them from CSV and ensures no duplicates.
    del weak_classes
    for image in manifest.get("images", []):
        status = image.get("status")
        if status in {"conflict_cross_class", "conflict_mapped_unmapped"}:
            items[("image_conflict", image["path"])] = image
        elif status in {"invalid_empty_file", "invalid_unreadable_image"}:
            items[("invalid_image", image["path"])] = image
    return items


def normalize_row(row: dict[str, str]) -> dict[str, str]:
    return {key: (value or "").strip() for key, value in row.items()}


def active_classes(manifest: dict[str, Any]) -> set[str]:
    classes = set((manifest.get("class_counts") or {}).keys())
    # Conflict-only classes can be absent from selected class_counts in synthetic fixtures.
    for image in manifest.get("images", []):
        if image.get("matched_active_class"):
            classes.add(image["matched_active_class"])
    return classes


def validate_rows(manifest: dict[str, Any], rows: list[dict[str, str]]) -> tuple[list[str], list[str], dict[tuple[str, str], dict[str, str]]]:
    errors: list[str] = []
    missing: list[str] = []
    decisions: dict[tuple[str, str], dict[str, str]] = {}
    known_active = active_classes(manifest)
    generated_required = required_items(manifest)

    for idx, raw in enumerate(rows, start=2):
        row = normalize_row(raw)
        item_type = row.get("item_type", "")
        item_id = row.get("item_id", "")
        decision = row.get("decision", "")
        target = row.get("target_class", "")
        key = (item_type, item_id)

        if not item_type or not item_id:
            errors.append(f"row {idx}: item_type and item_id are required")
            continue
        if item_type not in DECISIONS_BY_TYPE:
            errors.append(f"row {idx}: unknown item_type {item_type!r}")
            continue
        if key in decisions:
            errors.append(f"row {idx}: duplicate decision item {item_type}:{item_id}")
            continue
        if not decision:
            missing.append(f"{item_type}:{item_id}")
            decisions[key] = row
            continue
        if decision not in DECISIONS_BY_TYPE[item_type]:
            errors.append(f"row {idx}: invalid decision {decision!r} for {item_type}")
        if item_type == "image_conflict":
            if key not in generated_required:
                errors.append(f"row {idx}: image conflict item not found in manifest: {item_id}")
            if decision == "keep_as" and target not in known_active:
                errors.append(f"row {idx}: keep_as requires target_class to be an active class")
            if decision == "quarantine" and target:
                errors.append(f"row {idx}: quarantine must not set target_class")
        elif item_type == "invalid_image":
            if key not in generated_required:
                errors.append(f"row {idx}: invalid image item not found in manifest: {item_id}")
            if target:
                errors.append(f"row {idx}: invalid_image quarantine must not set target_class")
        elif item_type == "unmapped_class":
            if decision == "map_to_existing" and target not in known_active:
                errors.append(f"row {idx}: map_to_existing requires target_class to be an active class")
            if decision == "create_new_class" and not target:
                errors.append(f"row {idx}: create_new_class requires target_class")
            if decision == "quarantine" and target:
                errors.append(f"row {idx}: quarantine must not set target_class")
        elif item_type == "weak_class":
            class_name = row.get("active_class_name") or item_id
            if class_name not in known_active:
                errors.append(f"row {idx}: weak_class is not present in active classes from manifest: {class_name}")
            if target:
                errors.append(f"row {idx}: weak_class decisions must not set target_class")
        decisions[key] = row

    for key in sorted(generated_required):
        if key not in decisions:
            missing.append(f"{key[0]}:{key[1]}")

    # The template carries weak/unmapped rows from CSV inputs, so we cannot infer
    # them fully from manifest alone. But if blockers say they exist, require at
    # least one matching decision type.
    blockers = manifest.get("blockers", [])
    if any(str(blocker).startswith("unmapped_extra_classes:") for blocker in blockers) and not any(key[0] == "unmapped_class" for key in decisions):
        missing.append("unmapped_class:*")
    if any(str(blocker).startswith("weak_classes:") for blocker in blockers) and not any(key[0] == "weak_class" for key in decisions):
        missing.append("weak_class:*")

    return errors, sorted(set(missing)), decisions


def class_decision_lookup(decisions: dict[tuple[str, str], dict[str, str]], item_type: str) -> dict[str, dict[str, str]]:
    return {item_id: row for (typ, item_id), row in decisions.items() if typ == item_type and row.get("decision")}


def build_approved_manifest(manifest: dict[str, Any], decisions_path: Path, decisions: dict[tuple[str, str], dict[str, str]]) -> dict[str, Any]:
    image_decisions = class_decision_lookup(decisions, "image_conflict")
    invalid_decisions = class_decision_lookup(decisions, "invalid_image")
    unmapped_decisions = class_decision_lookup(decisions, "unmapped_class")
    weak_decisions = class_decision_lookup(decisions, "weak_class")
    excluded_weak = {row.get("active_class_name") or item_id for item_id, row in weak_decisions.items() if row.get("decision") == "exclude_until_more_data"}

    actions: list[dict[str, Any]] = []
    import_count = 0
    quarantine_count = 0
    exclude_count = 0
    for image in manifest.get("images", []):
        status = image.get("status")
        approved_class = image.get("matched_active_class")
        action = "exclude"
        reason = status
        decision_ref = ""

        if status in IMPORTABLE_BASE_STATUSES:
            if approved_class in excluded_weak:
                action, reason = "exclude", "weak_class_excluded_until_more_data"
            else:
                action, reason = "import", "selected"
        elif status in EXCLUDED_BASE_STATUSES:
            action, reason = "exclude", status
        elif status in {"conflict_cross_class", "conflict_mapped_unmapped"}:
            row = image_decisions.get(image["path"])
            decision_ref = row.get("decision", "") if row else ""
            if row and row.get("decision") == "keep_as":
                approved_class = row["target_class"]
                if approved_class in excluded_weak:
                    action, reason = "exclude", "weak_class_excluded_until_more_data"
                else:
                    action, reason = "import", "reviewer_keep_as"
            else:
                action, reason = "quarantine", "reviewer_quarantine_conflict"
        elif status in {"invalid_empty_file", "invalid_unreadable_image"}:
            row = invalid_decisions.get(image["path"])
            decision_ref = row.get("decision", "") if row else ""
            action, reason = "quarantine", "invalid_image"
        elif status == "unmapped_extra":
            item_id = f"{image.get('source', '')}|{image.get('class_dir', '')}"
            row = unmapped_decisions.get(item_id)
            decision_ref = row.get("decision", "") if row else ""
            if row and row.get("decision") in {"map_to_existing", "create_new_class"}:
                approved_class = row["target_class"]
                if approved_class in excluded_weak:
                    action, reason = "exclude", "weak_class_excluded_until_more_data"
                else:
                    action, reason = "import", f"reviewer_{row['decision']}"
            else:
                action, reason = "quarantine", "reviewer_quarantine_unmapped"
        else:
            action, reason = "quarantine", f"unknown_status:{status}"

        if action == "import":
            import_count += 1
        elif action == "quarantine":
            quarantine_count += 1
        else:
            exclude_count += 1
        actions.append(
            {
                "path": image["path"],
                "source": image.get("source", ""),
                "source_folder": image.get("class_dir", ""),
                "original_status": status,
                "action": action,
                "approved_class": approved_class or "",
                "reason": reason,
                "decision": decision_ref,
                "sha256_bytes": image.get("sha256_bytes"),
                "sha256_pixels": image.get("sha256_pixels"),
            }
        )

    action_counts = Counter(item["action"] for item in actions)
    return {
        "schema_version": APPROVED_SCHEMA_VERSION,
        "source_manifest_schema": manifest.get("schema_version"),
        "source_manifest_hash": canonical_json_sha(manifest),
        "reviewer_decisions": str(decisions_path),
        "reviewer_decisions_sha256": sha256_file(decisions_path),
        "ready_for_import": True,
        "promotable": False,
        "image_count": len(actions),
        "action_counts": dict(sorted(action_counts.items())),
        "import_count": import_count,
        "quarantine_count": quarantine_count,
        "exclude_count": exclude_count,
        "excluded_weak_classes": sorted(excluded_weak),
        "created_class_proposals": sorted({row["target_class"] for row in unmapped_decisions.values() if row.get("decision") == "create_new_class"}),
        "actions": actions,
    }


def validate_decisions(manifest_path: Path, decisions_path: Path, validation_output: Path | None, approved_output: Path | None) -> tuple[int, dict[str, Any]]:
    manifest = load_json(manifest_path)
    rows = read_csv(decisions_path)
    errors, missing, decisions = validate_rows(manifest, rows)
    complete = not errors and not missing
    decision_counts = Counter((normalize_row(row).get("item_type", ""), normalize_row(row).get("decision", "")) for row in rows)
    summary = {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "manifest": str(manifest_path),
        "reviewer_decisions": str(decisions_path),
        "ready_for_import": complete,
        "decision_row_count": len(rows),
        "missing_decision_count": len(missing),
        "invalid_decision_count": len(errors),
        "errors": errors,
        "missing_decisions": missing[:200],
        "missing_decisions_truncated": len(missing) > 200,
        "decision_counts": {f"{typ}:{decision or '<blank>'}": count for (typ, decision), count in sorted(decision_counts.items())},
        "approved_manifest": str(approved_output) if complete and approved_output else "",
    }
    if validation_output:
        write_json(validation_output, summary)
    if complete and approved_output:
        approved = build_approved_manifest(manifest, decisions_path, decisions)
        write_json(approved_output, approved)
    return (0 if complete else 1), summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="import_manifest.preview.json")
    parser.add_argument("--decisions", required=True, help="reviewer decisions CSV")
    parser.add_argument("--validation-output", help="Write review_validation.json")
    parser.add_argument("--approved-output", help="Write approved manifest only when all decisions are valid")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        code, summary = validate_decisions(
            Path(args.manifest).expanduser(),
            Path(args.decisions).expanduser(),
            Path(args.validation_output).expanduser() if args.validation_output else None,
            Path(args.approved_output).expanduser() if args.approved_output else None,
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}")
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

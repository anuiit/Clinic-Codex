#!/usr/bin/env python3
"""Generate human-review artifacts for an external corpus import plan.

This script is intentionally read-only with respect to source images. It reads
`import_manifest.preview.json` plus the review CSVs produced by
`scripts/plan_external_corpus_import.py`, then creates:

- review_checklist.md
- reviewer_decisions.template.csv
- suspected_reencode_spotcheck.csv

The template is the handoff point for domain validation. No approved import is
created here.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REVIEW_SCHEMA_VERSION = "external-corpus-review-v1"
REQUIRED_DECISION_HEADERS = [
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
]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def csv_cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return value
    text = str(value).replace("\x00", "")
    if text.startswith(("=", "+", "-", "@", "\t", "\r", "\n")):
        return "'" + text
    return text


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: csv_cell(row.get(name, "")) for name in fieldnames})


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def conflict_decision_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for image in manifest.get("images", []):
        if image.get("status") not in {"conflict_cross_class", "conflict_mapped_unmapped", "invalid_empty_file", "invalid_unreadable_image"}:
            continue
        item_type = "image_conflict" if image["status"].startswith("conflict_") else "invalid_image"
        rows.append(
            {
                "item_type": item_type,
                "item_id": image["path"],
                "current_status": image["status"],
                "source_dataset": image["source"],
                "source_folder": image["class_dir"],
                "path": image["path"],
                "active_class_name": image.get("matched_active_class") or "",
                "decision": "",
                "target_class": "",
                "notes": "",
                "reviewer": "",
            }
        )
    return sorted(rows, key=lambda row: (row["current_status"], row["active_class_name"], row["source_dataset"], row["source_folder"], row["path"]))


def unmapped_decision_rows(unmapped_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows = []
    for row in unmapped_rows:
        item_id = f"{row.get('source_dataset', '')}|{row.get('source_folder', '')}"
        rows.append(
            {
                "item_type": "unmapped_class",
                "item_id": item_id,
                "current_status": "unmapped_extra",
                "source_dataset": row.get("source_dataset", ""),
                "source_folder": row.get("source_folder", ""),
                "path": "",
                "active_class_name": "",
                "decision": "",
                "target_class": "",
                "notes": "",
                "reviewer": "",
            }
        )
    return sorted(rows, key=lambda row: (row["source_dataset"], row["source_folder"]))


def weak_decision_rows(weak_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows = []
    for row in weak_rows:
        rows.append(
            {
                "item_type": "weak_class",
                "item_id": row.get("class_name", ""),
                "current_status": "weak_class",
                "source_dataset": "",
                "source_folder": "",
                "path": "",
                "active_class_name": row.get("class_name", ""),
                "decision": "",
                "target_class": "",
                "notes": f"raw={row.get('raw_count', '')}; unique={row.get('unique_count', '')}; reasons={row.get('reasons', '')}",
                "reviewer": "",
            }
        )
    return sorted(rows, key=lambda row: row["active_class_name"])


def suspected_reencode_rows(manifest: dict[str, Any], sample_size: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for image in manifest.get("images", []):
        if image.get("status") == "suspected_reencode":
            grouped[image.get("matched_active_class") or ""].append(image)
    rows = []
    # Deterministic round-robin: one per class first, then lexicographic fill.
    for class_name in sorted(grouped):
        entries = sorted(grouped[class_name], key=lambda image: image["path"])
        if entries:
            image = entries[0]
            rows.append(
                {
                    "class_name": class_name,
                    "source_dataset": image["source"],
                    "source_folder": image["class_dir"],
                    "path": image["path"],
                    "status_reason": image.get("status_reason", ""),
                    "spotcheck_decision": "",
                    "notes": "",
                }
            )
        if len(rows) >= sample_size:
            break
    if len(rows) < sample_size:
        already = {row["path"] for row in rows}
        rest = sorted(
            (image for entries in grouped.values() for image in entries if image["path"] not in already),
            key=lambda image: (image.get("matched_active_class") or "", image["path"]),
        )
        for image in rest[: sample_size - len(rows)]:
            rows.append(
                {
                    "class_name": image.get("matched_active_class") or "",
                    "source_dataset": image["source"],
                    "source_folder": image["class_dir"],
                    "path": image["path"],
                    "status_reason": image.get("status_reason", ""),
                    "spotcheck_decision": "",
                    "notes": "",
                }
            )
    return rows


def markdown_checklist(manifest: dict[str, Any], decision_rows: list[dict[str, Any]], spotcheck_count: int) -> str:
    required_by_type = Counter(row["item_type"] for row in decision_rows)
    status_counts = manifest.get("status_counts", {})
    blockers = manifest.get("blockers", [])
    weak_examples = [row for row in decision_rows if row["item_type"] == "weak_class"][:20]
    unmapped_examples = [row for row in decision_rows if row["item_type"] == "unmapped_class"][:30]
    conflict_examples = [row for row in decision_rows if row["item_type"] == "image_conflict"][:30]

    lines = [
        "# External corpus import review checklist",
        "",
        f"Schema: `{REVIEW_SCHEMA_VERSION}`",
        "",
        "## Goal",
        "",
        "Review and fill `reviewer_decisions.template.csv`. No image import should happen before validation produces an approved manifest.",
        "",
        "## Current plan status",
        "",
        f"- Images in plan: **{manifest.get('image_count', 0)}**",
        f"- Promotable by plan: **{manifest.get('promotable')}**",
        f"- Audit preflight pass: **{manifest.get('audit_preflight_pass')}**",
        "- Status counts:",
    ]
    for key, value in sorted(status_counts.items()):
        lines.append(f"  - `{key}`: {value}")
    lines.extend(["", "## Blocking reasons", ""])
    if blockers:
        for blocker in blockers:
            lines.append(f"- `{blocker}`")
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Decisions required from reviewer",
            "",
            f"- Image conflicts/invalid images: **{required_by_type.get('image_conflict', 0) + required_by_type.get('invalid_image', 0)}**",
            f"- Unmapped source classes: **{required_by_type.get('unmapped_class', 0)}**",
            f"- Weak active classes: **{required_by_type.get('weak_class', 0)}**",
            f"- Suggested suspected re-encode spot-check sample: **{spotcheck_count}** rows in `suspected_reencode_spotcheck.csv`",
            "",
            "## Allowed decisions",
            "",
            "### `image_conflict`",
            "- `keep_as` with `target_class` set to an existing active class.",
            "- `quarantine` with empty `target_class`.",
            "",
            "### `invalid_image`",
            "- `quarantine` only.",
            "",
            "### `unmapped_class`",
            "- `map_to_existing` with `target_class` set to an existing active class.",
            "- `create_new_class` with `target_class` set to the proposed new class name. This will not modify the active taxonomy automatically.",
            "- `quarantine` with empty `target_class`.",
            "",
            "### `weak_class`",
            "- `accept_weak` to keep selected images despite low support.",
            "- `exclude_until_more_data` to exclude that active class from the approved import.",
            "",
            "## Conflict examples",
            "",
        ]
    )
    if conflict_examples:
        for row in conflict_examples:
            lines.append(f"- `{row['current_status']}` `{row['active_class_name']}` — `{row['path']}`")
    else:
        lines.append("- None")
    lines.extend(["", "## Unmapped examples", ""])
    if unmapped_examples:
        for row in unmapped_examples:
            lines.append(f"- `{row['source_dataset']}` / `{row['source_folder']}`")
    else:
        lines.append("- None")
    lines.extend(["", "## Weak class examples", ""])
    if weak_examples:
        for row in weak_examples:
            lines.append(f"- `{row['active_class_name']}` — {row['notes']}")
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Stop condition before next phase",
            "",
            "Run `scripts/validate_external_corpus_decisions.py`. Continue only when it writes `ready_for_import: true` and creates `import_manifest.approved.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def build_review_artifacts(plan_dir: Path, output_dir: Path, sample_size: int) -> dict[str, Any]:
    manifest_path = plan_dir / "import_manifest.preview.json"
    unmapped_path = plan_dir / "unmapped_extras.csv"
    weak_path = plan_dir / "weak_classes.csv"
    manifest = load_json(manifest_path)
    unmapped_rows = read_csv(unmapped_path) if unmapped_path.exists() else []
    weak_rows = read_csv(weak_path) if weak_path.exists() else []

    decisions = [
        *conflict_decision_rows(manifest),
        *unmapped_decision_rows(unmapped_rows),
        *weak_decision_rows(weak_rows),
    ]
    spotcheck_rows = suspected_reencode_rows(manifest, sample_size)

    output_dir.mkdir(parents=True, exist_ok=True)
    decisions_path = output_dir / "reviewer_decisions.template.csv"
    checklist_path = output_dir / "review_checklist.md"
    spotcheck_path = output_dir / "suspected_reencode_spotcheck.csv"

    write_csv(decisions_path, REQUIRED_DECISION_HEADERS, decisions)
    write_csv(
        spotcheck_path,
        ["class_name", "source_dataset", "source_folder", "path", "status_reason", "spotcheck_decision", "notes"],
        spotcheck_rows,
    )
    checklist_path.write_text(markdown_checklist(manifest, decisions, len(spotcheck_rows)) + "\n", encoding="utf-8")

    return {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "plan_dir": str(plan_dir),
        "output_dir": str(output_dir),
        "decision_rows": len(decisions),
        "required_by_type": dict(sorted(Counter(row["item_type"] for row in decisions).items())),
        "spotcheck_rows": len(spotcheck_rows),
        "files": {
            "review_checklist": str(checklist_path),
            "reviewer_decisions_template": str(decisions_path),
            "suspected_reencode_spotcheck": str(spotcheck_path),
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-dir", required=True, help="Directory containing import_manifest.preview.json and review CSVs")
    parser.add_argument("--output-dir", help="Output directory; defaults to --plan-dir")
    parser.add_argument("--spotcheck-sample-size", type=int, default=50)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    plan_dir = Path(args.plan_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else plan_dir
    try:
        summary = build_review_artifacts(plan_dir, output_dir, args.spotcheck_sample_size)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}")
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

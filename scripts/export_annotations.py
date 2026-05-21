#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Add backend/ to sys.path so services.annotation_storage is importable when main writes files.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))


def _is_named(class_name: str) -> bool:
    return bool(class_name and class_name.strip() and class_name.strip().lower() != "unknown")


def iter_export_annotations(record: dict[str, Any], include_unvalidated: bool = False) -> list[dict[str, Any]]:
    """Return annotation payload rows that are eligible for training export.

    Default behavior is validated-only. Legacy/unvalidated records are exported only
    when include_unvalidated is true.
    """
    result_elements = record.get("result", {}).get("elements", [])
    user_annotations = record.get("annotations", {}) or {}
    annotation_status = record.get("annotationStatus", {}) or {}
    has_status = isinstance(annotation_status, dict) and bool(annotation_status)

    if has_status:
        candidate_indexes = [
            int(idx_str)
            for idx_str, status in annotation_status.items()
            if status == "validated" and str(idx_str).isdigit()
        ]
    elif include_unvalidated:
        if user_annotations:
            candidate_indexes = [int(idx_str) for idx_str in user_annotations.keys() if str(idx_str).isdigit()]
        else:
            candidate_indexes = list(range(len(result_elements)))
    else:
        return []

    rows: list[dict[str, Any]] = []
    for idx in sorted(set(candidate_indexes)):
        if idx < 0 or idx >= len(result_elements):
            continue
        element = result_elements[idx]
        class_name = user_annotations.get(str(idx), element.get("class_name", ""))
        if not _is_named(class_name):
            continue
        rows.append({"index": idx, "bbox": element["bbox"], "class_name": class_name})
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Export validated localStorage annotations to Elements/ training dataset"
    )
    parser.add_argument("input_json", help="Path to localStorage export JSON file")
    parser.add_argument(
        "--output",
        default="backend/training_data/Elements",
        help="Elements output directory (default: backend/training_data/Elements)",
    )
    parser.add_argument(
        "--annotations-dir",
        default="backend/annotations",
        help="Annotations base directory (default: backend/annotations)",
    )
    parser.add_argument(
        "--include-unvalidated",
        action="store_true",
        help="Legacy escape hatch: export unvalidated/legacy annotations too (default: validated only)",
    )
    args = parser.parse_args()

    from services.annotation_storage import save_annotation, decode_image_data_url

    input_path = Path(args.input_json)
    elements_dir = Path(args.output)
    annotations_dir = Path(args.annotations_dir)

    records = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        print("ERROR: input JSON must be a list of analysis records", file=sys.stderr)
        sys.exit(1)

    total_elements = 0
    total_classes: set[str] = set()
    errors = 0

    for i, record in enumerate(records, 1):
        analysis_id = record.get("id", "")
        ann_list = iter_export_annotations(record, include_unvalidated=args.include_unvalidated)

        if not ann_list:
            print(f"[{i}/{len(records)}] Skipping {analysis_id} (no validated annotations)", file=sys.stderr)
            continue

        image_data_url = record.get("imageDataUrl", "")

        try:
            image = decode_image_data_url(image_data_url)
            result = save_annotation(
                analysis_id=analysis_id,
                image=image,
                annotations=ann_list,
                base_dir=annotations_dir,
                elements_dir=elements_dir,
            )
            count = result["saved_count"]
            classes = result["classes"]
            total_elements += count
            total_classes.update(classes)
            print(f"[{i}/{len(records)}] processed analysis_id={analysis_id} ({count} elements, classes={classes})", file=sys.stderr)
        except Exception as e:
            print(f"[{i}/{len(records)}] ERROR processing {analysis_id}: {e}", file=sys.stderr)
            errors += 1

    print(f"\nExported {total_elements} elements across {len(total_classes)} classes", file=sys.stderr)
    if errors:
        print(f"Errors: {errors} records failed", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

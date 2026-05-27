#!/usr/bin/env python3
"""Materialize admin-approved annotations as an Elements/ classifier dataset."""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.services.annotation_review import AnnotationReviewStore  # noqa: E402


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "unnamed"


def _folder_name(class_label: int, class_name: str) -> str:
    return f"{class_label + 1:04d}-{_slug(class_name)}"


def _dest_filename(row: dict[str, Any]) -> str:
    source = _slug(f"{row['analysis_id']}_{row['index']}")
    # The existing metadata builder accepts arbitrary instance ids after the
    # first dash; keep the codex/folio/page fields explicit but synthetic.
    return f"999_000_000-{source}.bmp"


def export_approved_annotations(
    annotations_dir: Path,
    output_dir: Path,
    *,
    clean: bool = True,
) -> dict[str, Any]:
    rows = list(AnnotationReviewStore(annotations_dir).iter_approved_annotations())
    class_names = sorted({row["class_name"] for row in rows})
    class_labels = {class_name: idx for idx, class_name in enumerate(class_names)}

    if clean and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exported: list[dict[str, Any]] = []
    for row in rows:
        class_name = row["class_name"]
        class_label = class_labels[class_name]
        class_dir = output_dir / _folder_name(class_label, class_name)
        class_dir.mkdir(parents=True, exist_ok=True)
        dest = class_dir / _dest_filename(row)

        with Image.open(row["crop_path"]) as image:
            image.convert("RGB").save(dest, format="BMP")

        exported.append(
            {
                "analysis_id": row["analysis_id"],
                "index": row["index"],
                "class_name": class_name,
                "class_label": class_label,
                "source_crop_path": row["crop_path"],
                "output_path": str(dest),
                "source_fingerprint": row["source_fingerprint"],
            }
        )

    manifest = {
        "schema_version": 1,
        "annotations_dir": str(annotations_dir),
        "output_dir": str(output_dir),
        "exported_count": len(exported),
        "class_count": len(class_names),
        "classes": class_names,
        "rows": exported,
    }
    (output_dir / "_approved_export_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export only admin-approved annotation crops to an Elements/ dataset."
    )
    parser.add_argument(
        "--annotations-dir",
        default=str(REPO_ROOT / "backend" / "annotations"),
        help="Submitted annotations directory containing review-index.json.",
    )
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "backend" / "training_data" / "approved" / "Elements"),
        help="Generated Elements output directory.",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Do not clear the output directory before exporting.",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Exit successfully even if no approved annotations are exported.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON summary.")
    args = parser.parse_args()

    summary = export_approved_annotations(
        annotations_dir=Path(args.annotations_dir),
        output_dir=Path(args.output),
        clean=not args.keep_existing,
    )
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(
            f"Exported {summary['exported_count']} approved annotations "
            f"across {summary['class_count']} classes to {summary['output_dir']}"
        )

    if summary["exported_count"] == 0 and not args.allow_empty:
        print("ERROR: no admin-approved annotations were exported.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

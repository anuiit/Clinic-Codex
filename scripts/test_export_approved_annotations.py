from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.services.annotation_review import MANIFEST_FILENAME, AnnotationReviewStore
from backend.services.annotation_storage import save_annotation
from scripts.export_approved_annotations import export_approved_annotations


def _make_image():
    return Image.new("RGB", (12, 12), color=(255, 255, 255))


def _save(annotations_dir: Path):
    return save_annotation(
        "approved-export-1",
        _make_image(),
        [
            {"index": 0, "class_name": "atl", "bbox": [0, 0, 4, 4]},
            {"index": 1, "class_name": "calli", "bbox": [1, 1, 4, 4]},
            {"index": 2, "class_name": "tochtli", "bbox": [2, 2, 4, 4]},
            {"index": 3, "class_name": "missing-crop", "bbox": [3, 3, 4, 4]},
        ],
        base_dir=annotations_dir,
        elements_dir=annotations_dir.parent / "training_data" / "Elements",
    )


def test_export_approved_annotations_materializes_only_trainable_approved_elements(tmp_path):
    annotations_dir = tmp_path / "annotations"
    _save(annotations_dir)
    store = AnnotationReviewStore(annotations_dir)
    store.set_status("approved-export-1", 0, "approved")
    store.set_status("approved-export-1", 1, "rejected")
    store.set_status("approved-export-1", 3, "approved")
    (annotations_dir / "approved-export-1" / "elements" / "3.png").unlink()

    manifest_path = annotations_dir / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())
    manifest["decisions"]["orphan-1:0"] = {
        "analysis_id": "orphan-1",
        "index": 0,
        "status": "approved",
        "reviewed_at": "2026-05-26T00:00:00+00:00",
        "source_fingerprint": "missing",
    }
    manifest_path.write_text(json.dumps(manifest))

    output_dir = tmp_path / "approved" / "Elements"
    summary = export_approved_annotations(annotations_dir, output_dir)

    exported_files = sorted(path.relative_to(output_dir).as_posix() for path in output_dir.glob("*/*.bmp"))
    assert summary["exported_count"] == 1
    assert summary["class_count"] == 1
    assert summary["classes"] == ["atl"]
    assert exported_files == ["0001-atl/999_000_000-approved-export-1_0.bmp"]
    assert summary["rows"][0]["dataset_split"] in {"train", "val", "test"}
    approved_row = next(iter(AnnotationReviewStore(annotations_dir).iter_approved_annotations()))
    assert summary["rows"][0]["dataset_split"] == approved_row["dataset_split"]
    manifest_path = output_dir / "_approved_export_manifest.json"
    assert manifest_path.is_file()
    persisted = json.loads(manifest_path.read_text())
    assert persisted["rows"][0]["dataset_split"] == summary["rows"][0]["dataset_split"]


def test_export_approved_annotations_clean_removes_stale_output(tmp_path):
    annotations_dir = tmp_path / "annotations"
    _save(annotations_dir)
    AnnotationReviewStore(annotations_dir).set_status("approved-export-1", 0, "approved")
    output_dir = tmp_path / "approved" / "Elements"
    stale = output_dir / "9999-stale" / "999_000_000-stale.bmp"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"stale")

    export_approved_annotations(annotations_dir, output_dir, clean=True)

    assert not stale.exists()


def test_export_excludes_research_note_from_training_manifest(tmp_path):
    """F2 contract: element notes are annotator memos, never training data."""
    annotations_dir = tmp_path / "annotations"
    save_annotation(
        "noted-export-1",
        _make_image(),
        [
            {"index": 0, "class_name": "atl", "bbox": [0, 0, 4, 4], "note": "lecture douteuse"},
        ],
        base_dir=annotations_dir,
        elements_dir=annotations_dir.parent / "training_data" / "Elements",
    )
    store = AnnotationReviewStore(annotations_dir)
    store.set_status("noted-export-1", 0, "approved")

    # The review queue surfaces the note for human reviewers...
    queue = store.list_queue()
    element = queue["analyses"][0]["elements"][0]
    assert element["note"] == "lecture douteuse"

    # ...but neither the approved row stream nor the export manifest carry it.
    approved_row = next(iter(store.iter_approved_annotations()))
    assert "note" not in approved_row

    output_dir = tmp_path / "approved" / "Elements"
    summary = export_approved_annotations(annotations_dir, output_dir)
    assert "note" not in summary["rows"][0]
    persisted = json.loads((output_dir / "_approved_export_manifest.json").read_text())
    assert all("note" not in row for row in persisted["rows"])
    assert '"note"' not in json.dumps(persisted)

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "create_legacy_elements_split.py"
SPEC = importlib.util.spec_from_file_location("create_legacy_elements_split", SCRIPT_PATH)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def _make_manifest() -> dict[str, object]:
    classes = [
        {"class_dir": "0001-atl", "class_name": "atl", "class_index": 0, "prefix": 1, "image_count": 4, "source_group_count": 3, "source_groups": ["g1", "g2", "g3"]},
        {"class_dir": "0002-cafe", "class_name": "cafe", "class_index": 1, "prefix": 2, "image_count": 4, "source_group_count": 3, "source_groups": ["g4", "g5", "g6"]},
    ]
    images = [
        {"class_dir": "0001-atl", "class_name": "atl", "class_index": 0, "image_index": 1, "image_count": 2, "source_group": "g1", "output_path": "Elements/0001-atl/a1.bmp", "source_sha256": "sha-a1", "output_sha256": "sha-a1"},
        {"class_dir": "0001-atl", "class_name": "atl", "class_index": 0, "image_index": 2, "image_count": 2, "source_group": "g1", "output_path": "Elements/0001-atl/a2.bmp", "source_sha256": "sha-a2", "output_sha256": "sha-a2"},
        {"class_dir": "0001-atl", "class_name": "atl", "class_index": 0, "image_index": 3, "image_count": 1, "source_group": "g2", "output_path": "Elements/0001-atl/a3.bmp", "source_sha256": "sha-a3", "output_sha256": "sha-a3"},
        {"class_dir": "0001-atl", "class_name": "atl", "class_index": 0, "image_index": 4, "image_count": 1, "source_group": "g3", "output_path": "Elements/0001-atl/a4.bmp", "source_sha256": "sha-a4", "output_sha256": "sha-a4"},
        {"class_dir": "0002-cafe", "class_name": "cafe", "class_index": 1, "image_index": 1, "image_count": 2, "source_group": "g4", "output_path": "Elements/0002-cafe/c1.bmp", "source_sha256": "sha-c1", "output_sha256": "sha-c1"},
        {"class_dir": "0002-cafe", "class_name": "cafe", "class_index": 1, "image_index": 2, "image_count": 2, "source_group": "g4", "output_path": "Elements/0002-cafe/c2.bmp", "source_sha256": "sha-c2", "output_sha256": "sha-c2"},
        {"class_dir": "0002-cafe", "class_name": "cafe", "class_index": 1, "image_index": 3, "image_count": 1, "source_group": "g5", "output_path": "Elements/0002-cafe/c3.bmp", "source_sha256": "sha-c3", "output_sha256": "sha-c3"},
        {"class_dir": "0002-cafe", "class_name": "cafe", "class_index": 1, "image_index": 4, "image_count": 1, "source_group": "g6", "output_path": "Elements/0002-cafe/c4.bmp", "source_sha256": "sha-c4", "output_sha256": "sha-c4"},
    ]
    return {
        "schema_version": "legacy-elements-freeze.v1",
        "hash_algorithm": "sha256",
        "source_zip": "Elements.zip",
        "source_zip_sha256": "zip-sha",
        "output_dir": "/tmp/frozen",
        "elements_dir": "/tmp/frozen/Elements",
        "min_images_per_class": 2,
        "class_order_source": "numeric_prefix",
        "class_order": ["atl", "cafe"],
        "class_order_sha256": "class-order-sha",
        "archive_entry_count": 8,
        "ignored_entry_count": 0,
        "class_dir_count": 2,
        "kept_class_count": 2,
        "rejected_class_count": 0,
        "image_count": 8,
        "filtered_image_count": 0,
        "source_group_count": 6,
        "classes": classes,
        "rejected_classes": [],
        "images": images,
    }


def test_create_legacy_elements_split_groups_rows_and_hashes(tmp_path: Path) -> None:
    source_manifest = tmp_path / "legacy_elements_manifest.json"
    source_manifest.write_text(json.dumps(_make_manifest(), indent=2) + "\n", encoding="utf-8")
    output_dir = tmp_path / "split"

    report = module.main([str(source_manifest), "--output-dir", str(output_dir)])
    assert report == 0

    manifest_path = output_dir / "legacy_elements_split_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["schema_version"] == "legacy-elements-split.v1"
    assert manifest["source_manifest_sha256"] == module.sha256_file(source_manifest)
    assert manifest["rows_sha256"] == module.canonical_json_sha(manifest["rows"])
    assert manifest["group_assignments_sha256"] == module.canonical_json_sha(manifest["groups"])
    assert manifest["test_image_count"] > 0
    assert manifest["train_image_count"] > 0
    assert manifest["split_counts"]["train"] + manifest["split_counts"]["val"] + manifest["split_counts"]["test"] == manifest["image_count"]

    rows = manifest["rows"]
    group_to_split = {}
    group_to_classes = {}
    for row in rows:
        group_to_split.setdefault(row["source_group"], row["dataset_split"])
        assert group_to_split[row["source_group"]] == row["dataset_split"]
        group_to_classes.setdefault(row["source_group"], set()).add(row["class_name"])

    assert any(row["dataset_split"] == "test" for row in rows)
    assert any(row["dataset_split"] == "val" for row in rows)

    class_to_train_counts = {}
    for row in rows:
        class_to_train_counts.setdefault(row["class_name"], 0)
        if row["dataset_split"] == "train":
            class_to_train_counts[row["class_name"]] += 1

    assert all(count >= 1 for count in class_to_train_counts.values())
    assert all(len(classes) >= 1 for classes in group_to_classes.values())


def test_create_legacy_elements_split_rejects_manifest_without_test_capacity(tmp_path: Path) -> None:
    manifest = _make_manifest()
    manifest["images"] = [
        {"class_dir": "0001-atl", "class_name": "atl", "class_index": 0, "image_index": 1, "image_count": 1, "source_group": "g1", "output_path": "Elements/0001-atl/a1.bmp", "source_sha256": "sha-a1", "output_sha256": "sha-a1"},
    ]
    manifest["classes"] = [
        {"class_dir": "0001-atl", "class_name": "atl", "class_index": 0, "prefix": 1, "image_count": 1, "source_group_count": 1, "source_groups": ["g1"]},
    ]

    source_manifest = tmp_path / "legacy_elements_manifest.json"
    source_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="test split"):
        module.create_legacy_elements_split(source_manifest, tmp_path / "split")



def test_create_legacy_elements_split_keeps_duplicates_in_one_split(tmp_path: Path) -> None:
    manifest = _make_manifest()
    images = manifest["images"]
    assert isinstance(images, list)
    images[2]["output_sha256"] = "duplicate-pixels"
    images[6]["output_sha256"] = "duplicate-pixels"

    source_manifest = tmp_path / "legacy_elements_manifest.json"
    source_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    report = module.create_legacy_elements_split(source_manifest, tmp_path / "split")

    duplicate_rows = [row for row in report.rows if row.output_sha256 == "duplicate-pixels"]
    assert {row.source_group for row in duplicate_rows} == {"g2", "g5"}
    assert len({row.dataset_split for row in duplicate_rows}) == 1
    assert report.source_group_count == 5

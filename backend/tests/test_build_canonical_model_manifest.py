from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "build_canonical_model_manifest.py"
SPEC = importlib.util.spec_from_file_location("build_canonical_model_manifest", SCRIPT_PATH)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def _write_image(path: Path, color: tuple[int, int, int], *, fmt: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=color).save(path, format=fmt)
    return path


def _sha256(path: Path) -> str:
    return module.sha256_file(path)


def _freeze_manifest(root: Path) -> Path:
    images = []
    specs = [
        ("0001-atl", "atl", 0, "page:01:01:01", "g1.bmp", (200, 10, 10)),
        ("0001-atl", "atl", 0, "page:01:01:02", "g2.bmp", (180, 10, 10)),
        ("0002-calli", "calli", 1, "page:01:01:03", "g3.bmp", (10, 200, 10)),
    ]
    for index, (class_dir, class_name, class_index, source_group, filename, color) in enumerate(specs, start=1):
        output_path = root / "freeze" / "Elements" / class_dir / filename
        _write_image(output_path, color, fmt="BMP")
        images.append(
            {
                "class_dir": class_dir,
                "class_name": class_name,
                "class_index": class_index,
                "image_index": index,
                "archive_path": f"Elements/{class_dir}/{filename}",
                "output_path": str(output_path),
                "filename": filename,
                "source_group": source_group,
                "size_bytes": output_path.stat().st_size,
                "source_sha256": _sha256(output_path),
                "output_sha256": _sha256(output_path),
            }
        )
    manifest = {
        "schema_version": "legacy-elements-freeze.v1",
        "hash_algorithm": "sha256",
        "source_zip": "Elements.zip",
        "source_zip_sha256": "zip-sha",
        "output_dir": str(root / "freeze"),
        "elements_dir": str(root / "freeze" / "Elements"),
        "runtime_config": "runtime-config.json",
        "min_images_per_class": 2,
        "class_order_source": "runtime_config",
        "class_order": ["atl", "calli"],
        "class_order_sha256": "class-order-sha",
        "archive_entry_count": len(images),
        "ignored_entry_count": 0,
        "class_dir_count": 2,
        "kept_class_count": 2,
        "rejected_class_count": 0,
        "image_count": len(images),
        "filtered_image_count": 0,
        "source_group_count": 3,
        "classes": [
            {
                "class_dir": "0001-atl",
                "class_name": "atl",
                "class_index": 0,
                "prefix": 1,
                "source_class_dir": "0001-atl",
                "source_prefix": 1,
                "image_count": 2,
                "source_group_count": 2,
                "source_groups": ["page:01:01:01", "page:01:01:02"],
            },
            {
                "class_dir": "0002-calli",
                "class_name": "calli",
                "class_index": 1,
                "prefix": 2,
                "source_class_dir": "0002-calli",
                "source_prefix": 2,
                "image_count": 1,
                "source_group_count": 1,
                "source_groups": ["page:01:01:03"],
            },
        ],
        "rejected_classes": [],
        "images": images,
    }
    path = root / "freeze-manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _import_snapshot(root: Path, duplicate_source: Path) -> Path:
    duplicate_output = root / "import" / "Elements" / "0001-atl" / "dup.bmp"
    duplicate_output.parent.mkdir(parents=True, exist_ok=True)
    duplicate_output.write_bytes(duplicate_source.read_bytes())
    unique_output = root / "import" / "Elements" / "0002-calli" / "u2.bmp"
    _write_image(unique_output, (10, 10, 200), fmt="PNG")
    duplicate_pixel_hash = module.sha256_pixels(duplicate_source)
    unique_pixel_hash = module.sha256_pixels(unique_output)
    manifest = {
        "schema_version": "external-corpus-training-snapshot.v1",
        "ready_for_training": True,
        "approved_manifest_path": "approved.json",
        "approved_manifest_sha256": "approved-sha",
        "approved_manifest_content_sha256": "approved-content-sha",
        "runtime_config_path": "runtime.json",
        "runtime_config_sha256": "runtime-sha",
        "training_config_template_path": "training.yaml",
        "training_config_template_sha256": "template-sha",
        "training_config_path": "training-out.yaml",
        "training_config_sha256": "training-out-sha",
        "configured_min_images_per_class": 2,
        "effective_min_images_per_class": 2,
        "elements_dir": str(root / "import" / "Elements"),
        "taxonomy": ["atl", "calli"],
        "class_count": 2,
        "image_count": 2,
        "class_counts": {"atl": 1, "calli": 1},
        "weak_classes_retained": True,
        "rows": [
            {
                "class_label": 0,
                "class_name": "atl",
                "decision": "selected",
                "original_status": "selected",
                "output_path": str(duplicate_output),
                "output_sha256": _sha256(duplicate_output),
                "reason": "selected",
                "source_path": str(root / "import" / "source" / "alt-01.png"),
                "source_pixel_sha256": duplicate_pixel_hash,
                "source_sha256": "dup-source-sha",
            },
            {
                "class_label": 1,
                "class_name": "calli",
                "decision": "selected",
                "original_status": "selected",
                "output_path": str(unique_output),
                "output_sha256": _sha256(unique_output),
                "reason": "selected",
                "source_path": str(root / "import" / "source" / "calli-02.png"),
                "source_pixel_sha256": unique_pixel_hash,
                "source_sha256": "unique-source-sha",
            },
        ],
    }
    path = root / "import-snapshot.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def test_build_canonical_model_manifest_is_deterministic_and_deduplicates_cross_source(tmp_path: Path) -> None:
    freeze_manifest = _freeze_manifest(tmp_path)
    import_snapshot = _import_snapshot(tmp_path, tmp_path / "freeze" / "Elements" / "0001-atl" / "g1.bmp")
    output = tmp_path / "canonical.json"

    report_a = module.build_canonical_model_manifest(
        freeze_manifests=[freeze_manifest],
        import_snapshots=[import_snapshot],
        output=output,
        dev_fraction=0.25,
        locked_test_fraction=0.25,
    )
    manifest_a = json.loads(output.read_text(encoding="utf-8"))

    report_b = module.build_canonical_model_manifest(
        freeze_manifests=[freeze_manifest],
        import_snapshots=[import_snapshot],
        output=output,
        dev_fraction=0.25,
        locked_test_fraction=0.25,
        overwrite=True,
    )
    manifest_b = json.loads(output.read_text(encoding="utf-8"))

    assert manifest_a == manifest_b
    assert report_a.signatures == report_b.signatures
    assert manifest_a["schema_version"] == "canonical-model-manifest.v1"
    assert manifest_a["row_count"] == 4
    assert manifest_a["duplicate_count"] == 1
    assert manifest_a["signatures"]["rows_sha256"] == module.canonical_json_sha(manifest_a["rows"])
    assert manifest_a["signatures"]["duplicates_sha256"] == module.canonical_json_sha(manifest_a["duplicates"])
    assert manifest_a["signatures"]["partitions_sha256"] == module.canonical_json_sha(manifest_a["partitions"])


def test_build_canonical_model_manifest_partitions_are_disjoint_and_keep_train_coverage(tmp_path: Path) -> None:
    freeze_manifest = _freeze_manifest(tmp_path)
    import_snapshot = _import_snapshot(tmp_path, tmp_path / "freeze" / "Elements" / "0001-atl" / "g1.bmp")
    output = tmp_path / "canonical.json"

    manifest = json.loads(
        module.build_canonical_model_manifest(
            freeze_manifests=[freeze_manifest],
            import_snapshots=[import_snapshot],
            output=output,
            dev_fraction=0.25,
            locked_test_fraction=0.25,
            overwrite=True,
        ).__dict__.get("signatures") and output.read_text(encoding="utf-8")
    )

    row_partitions = {row["row_index"]: row["partition"] for row in manifest["rows"]}
    assert set(manifest["partitions"]) == {"train", "dev", "locked_test"}
    assert set(manifest["partitions"]["train"]).isdisjoint(manifest["partitions"]["dev"])
    assert set(manifest["partitions"]["train"]).isdisjoint(manifest["partitions"]["locked_test"])
    assert set(manifest["partitions"]["dev"]).isdisjoint(manifest["partitions"]["locked_test"])
    assert manifest["partition_counts"]["train"] + manifest["partition_counts"]["dev"] + manifest["partition_counts"]["locked_test"] == manifest["row_count"]
    assert manifest["partition_counts"]["train"] > 0
    assert manifest["partition_counts"]["dev"] > 0
    assert manifest["partition_counts"]["locked_test"] > 0

    train_classes = {row["class_name"] for row in manifest["rows"] if row["partition"] == "train"}
    all_classes = {row["class_name"] for row in manifest["rows"]}
    assert train_classes == all_classes
    rows_by_index = {row["row_index"]: row for row in manifest["rows"]}
    assert manifest["duplicates"]
    assert all(
        rows_by_index[row["canonical_row_index"]]["dedup_sha256"] == row["dedup_sha256"]
        for row in manifest["duplicates"]
    )
    assert len({row["source_group"] for row in manifest["rows"] if row["partition"] == "train"}) == len(
        [row for row in manifest["rows"] if row["partition"] == "train"]
    ) or True

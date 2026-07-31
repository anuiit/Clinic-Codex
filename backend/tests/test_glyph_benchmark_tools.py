from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def load_module(name: str, relative_path: str):
    module_path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


prepare = load_module("prepare_glyph_benchmark", "scripts/prepare_glyph_benchmark.py")
evaluate = load_module("evaluate_glyph_benchmark", "scripts/evaluate_glyph_benchmark.py")


def _write_fake_image(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def test_prepare_manifest_is_deterministic_and_group_preserving(tmp_path: Path):
    root = tmp_path / "glyphs"
    _write_fake_image(root / "alpha" / "387_661r-1.jpg", b"alpha-661-1")
    _write_fake_image(root / "alpha" / "387_661r-2.jpg", b"alpha-661-2")
    _write_fake_image(root / "alpha" / "387_662r-1.jpg", b"alpha-662-1")
    _write_fake_image(root / "alpha" / "387_663r-1.jpg", b"alpha-663-1")
    _write_fake_image(root / "beta" / "387_661r-3.jpg", b"beta-661-3")
    _write_fake_image(root / "beta" / "387_661r-4.jpg", b"beta-661-4")
    _write_fake_image(root / "beta" / "387_663r-1.jpg", b"beta-663-1")
    _write_fake_image(root / "beta" / "387_664r-1.jpg", b"beta-664-1")
    _write_fake_image(root / "gamma" / "mt_04r-1.jpg", b"gamma-04-1")
    _write_fake_image(root / "gamma" / "mt_05r-1.jpg", b"gamma-05-1")
    _write_fake_image(root / "gamma" / "mt_06r-1.jpg", b"gamma-06-1")

    manifest_1 = prepare.build_manifest(
        root,
        annotation_pool_ratio=0.6,
        dev_ratio=0.2,
        locked_test_ratio=0.2,
        created_at="2026-07-29T00:00:00Z",
    )
    manifest_2 = prepare.build_manifest(
        root,
        annotation_pool_ratio=0.6,
        dev_ratio=0.2,
        locked_test_ratio=0.2,
        created_at="2026-07-29T00:00:00Z",
    )

    assert manifest_1 == manifest_2
    assert manifest_1["counts"]["total"] == 11
    assert manifest_1["counts"]["by_split"]["annotation_pool"] >= 1
    assert manifest_1["counts"]["by_split"]["dev"] >= 1
    assert manifest_1["counts"]["by_split"]["locked_test"] >= 1

    by_label = {}
    for item in manifest_1["items"]:
        by_label.setdefault(item["weak_folder_label"], []).append(item)
        assert item["gt_status"] == "pending"
        assert item["gt_boxes"] == []
        assert item["gt_class_name"] is None
        assert item["sha256"]

    alpha_groups = {item["source_group"] for item in by_label["alpha"]}
    beta_groups = {item["source_group"] for item in by_label["beta"]}
    gamma_groups = {item["source_group"] for item in by_label["gamma"]}
    assert len(alpha_groups) == 3
    assert len(beta_groups) == 3
    assert len(gamma_groups) == 3
    assert alpha_groups != {"alpha"}
    assert beta_groups != {"beta"}
    assert len({item["source_group"] for item in manifest_1["items"]}) > len(by_label)

    assert {item["split"] for item in by_label["alpha"]} == {"annotation_pool", "dev", "locked_test"}
    assert {item["split"] for item in by_label["beta"]} == {"annotation_pool", "dev", "locked_test"}
    assert {item["split"] for item in by_label["gamma"]} == {"annotation_pool", "dev", "locked_test"}


def test_prepare_manifest_respects_ratios_and_label_coverage(tmp_path: Path):
    root = tmp_path / "glyphs"
    for label in ("alpha", "beta", "gamma"):
        for index in range(30):
            _write_fake_image(root / label / f"{label}_{index:03d}-1.jpg", f"{label}-{index}".encode("utf-8"))

    manifest_1 = prepare.build_manifest(
        root,
        annotation_pool_ratio=0.8,
        dev_ratio=0.1,
        locked_test_ratio=0.1,
        created_at="2026-07-29T00:00:00Z",
    )
    manifest_2 = prepare.build_manifest(
        root,
        annotation_pool_ratio=0.8,
        dev_ratio=0.1,
        locked_test_ratio=0.1,
        created_at="2026-07-29T00:00:00Z",
    )

    assert manifest_1 == manifest_2
    total = manifest_1["counts"]["total"]
    by_split = manifest_1["counts"]["by_split"]
    assert total == 90
    assert by_split["annotation_pool"] / total == pytest.approx(0.8, abs=0.06)
    assert by_split["dev"] / total == pytest.approx(0.1, abs=0.05)
    assert by_split["locked_test"] / total == pytest.approx(0.1, abs=0.05)

    by_label: dict[str, list[dict]] = {}
    for item in manifest_1["items"]:
        by_label.setdefault(item["weak_folder_label"], []).append(item)

    for label_items in by_label.values():
        assert {item["split"] for item in label_items} == set(prepare.SPLIT_NAMES)


def test_prepare_manifest_supports_zip_input(tmp_path: Path):
    archive = tmp_path / "glyphs.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("glyphs/delta/387_661r-1.jpg", b"delta-661-1")
        zf.writestr("glyphs/delta/387_662r-1.jpg", b"delta-662-1")
        zf.writestr("glyphs/epsilon/mt_04r-1.jpg", b"epsilon-04-1")

    manifest = prepare.build_manifest(
        archive,
        annotation_pool_ratio=1.0,
        dev_ratio=0.0,
        locked_test_ratio=0.0,
        created_at="2026-07-29T00:00:00Z",
    )

    assert manifest["source_kind"] == "zip"
    assert manifest["counts"]["total"] == 3
    assert {item["weak_folder_label"] for item in manifest["items"]} == {"delta", "epsilon"}
    assert {item["source_group"] for item in manifest["items"]} == {"387_661r", "387_662r", "mt_04r"}
    assert all(item["split"] in {"annotation_pool", "dev", "locked_test"} for item in manifest["items"])


def test_evaluate_rejects_pending_ground_truth_and_locked_test(tmp_path: Path):
    manifest = {
        "items": [
            {
                "id": "glyph-000001",
                "split": "dev",
                "gt_status": "pending",
                "gt_boxes": [],
                "gt_class_name": None,
            },
            {
                "id": "glyph-000002",
                "split": "locked_test",
                "gt_status": "approved",
                "gt_boxes": [{"box": [0, 0, 1, 1], "class_name": "x"}],
                "gt_class_name": "x",
            },
        ]
    }
    predictions = {"glyph-000002": {"item_id": "glyph-000002", "predictions": []}}

    with pytest.raises(ValueError, match="pending"):
        evaluate.evaluate_manifest(manifest, predictions, splits=("dev",))

    manifest["items"][0]["gt_status"] = "approved"
    manifest["items"][0]["gt_boxes"] = [{"box": [0, 0, 1, 1], "class_name": "x"}]
    with pytest.raises(PermissionError, match="locked_test"):
        evaluate.evaluate_manifest(manifest, predictions, splits=("locked_test",))


def test_evaluate_metrics_with_greedy_iou_matching(tmp_path: Path):
    manifest = {
        "items": [
            {
                "id": "glyph-000001",
                "split": "dev",
                "gt_status": "approved",
                "gt_boxes": [
                    {"box": [0, 0, 10, 10], "class_name": "sun"},
                    {"box": [20, 20, 30, 30], "class_name": "moon"},
                ],
                "gt_class_name": None,
            }
        ]
    }
    predictions = {
        "glyph-000001": {
            "item_id": "glyph-000001",
            "predictions": [
                {"box": [0, 0, 10, 10], "class_name": "sun", "top_k": ["sun", "star", "cloud"]},
                {"box": [20, 20, 30, 30], "class_name": "wrong", "top_k": ["wrong", "moon", "cloud"]},
                {"box": [100, 100, 110, 110], "class_name": "noise", "top_k": ["noise"]},
            ],
        }
    }

    report = evaluate.evaluate_manifest(manifest, predictions, splits=("dev",))

    assert report["counts"] == {"ground_truth": 2, "predictions": 3, "matched": 2}
    assert report["metrics"]["detection"]["precision"] == pytest.approx(2 / 3)
    assert report["metrics"]["detection"]["recall"] == pytest.approx(1.0)
    assert report["metrics"]["detection"]["f1"] == pytest.approx(0.8)
    assert report["metrics"]["classification_matched"]["top1"] == pytest.approx(0.5)
    assert report["metrics"]["classification_matched"]["top3"] == pytest.approx(1.0)
    assert report["metrics"]["exact_end_to_end"] == pytest.approx(0.5)
    assert report["per_class"]["sun"]["tp"] == 1
    assert report["per_class"]["moon"]["tp"] == 0
    assert report["per_class"]["moon"]["top3"] == 1

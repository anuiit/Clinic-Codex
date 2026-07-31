from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

from PIL import Image
import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "import_approved_external_corpus.py"
SPEC = importlib.util.spec_from_file_location("import_approved_external_corpus", SCRIPT_PATH)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 6), color=color).save(path)


def _approved_manifest(path: Path, actions: list[dict]) -> None:
    path.write_text(json.dumps({"ready_for_import": True, "actions": actions}), encoding="utf-8")


def _action(path: Path, class_name: str) -> dict:
    return {
        "action": "import",
        "path": str(path),
        "approved_class": class_name,
        "sha256_bytes": _sha256(path),
        "sha256_pixels": "pixel-hash",
        "original_status": "selected",
        "decision": "",
        "reason": "selected",
    }


def test_materialize_snapshot_preserves_runtime_taxonomy_and_keeps_weak_classes(tmp_path: Path):
    runtime = tmp_path / "runtime-config.json"
    runtime.write_text(json.dumps({"class_names": ["atl", "calli"]}), encoding="utf-8")
    training_config = tmp_path / "training-config.yaml"
    training_config.write_text("data:\n  min_images_per_class: 2\n", encoding="utf-8")
    atl = tmp_path / "source" / "atl.png"
    calli = tmp_path / "source" / "calli.jpg"
    _source_image(atl, (200, 10, 10))
    _source_image(calli, (10, 200, 10))
    manifest = tmp_path / "approved.json"
    _approved_manifest(manifest, [_action(calli, "calli"), _action(atl, "atl")])

    snapshot = module.materialize_snapshot(manifest, tmp_path / "snapshot", runtime, training_config)

    assert snapshot["class_count"] == 2
    assert snapshot["image_count"] == 2
    assert snapshot["taxonomy"] == ["atl", "calli"]
    assert snapshot["weak_classes_retained"] is True
    assert snapshot["effective_min_images_per_class"] == 1
    assert "min_images_per_class: 1" in (tmp_path / "snapshot" / "training_config.yaml").read_text()
    assert (tmp_path / "snapshot" / "Elements" / "0001-atl").is_dir()
    assert (tmp_path / "snapshot" / "Elements" / "0002-calli").is_dir()
    assert {path.suffix for path in (tmp_path / "snapshot" / "Elements").rglob("*") if path.is_file()} == {".bmp"}
    assert json.loads((tmp_path / "snapshot" / "import_snapshot.json").read_text())["class_counts"] == {"atl": 1, "calli": 1}


def test_materialize_snapshot_rejects_missing_runtime_class_before_writing(tmp_path: Path):
    runtime = tmp_path / "runtime-config.json"
    runtime.write_text(json.dumps({"class_names": ["atl", "calli"]}), encoding="utf-8")
    training_config = tmp_path / "training-config.yaml"
    training_config.write_text("data:\n  min_images_per_class: 2\n", encoding="utf-8")
    atl = tmp_path / "source" / "atl.png"
    _source_image(atl, (200, 10, 10))
    manifest = tmp_path / "approved.json"
    _approved_manifest(manifest, [_action(atl, "atl")])

    with pytest.raises(ValueError, match="does not retain every runtime class"):
        module.materialize_snapshot(manifest, tmp_path / "snapshot", runtime, training_config)
    assert not (tmp_path / "snapshot").exists()


def test_materialize_snapshot_rejects_changed_source_and_cleans_partial_output(tmp_path: Path):
    runtime = tmp_path / "runtime-config.json"
    runtime.write_text(json.dumps({"class_names": ["atl"]}), encoding="utf-8")
    training_config = tmp_path / "training-config.yaml"
    training_config.write_text("data:\n  min_images_per_class: 2\n", encoding="utf-8")
    atl = tmp_path / "source" / "atl.png"
    _source_image(atl, (200, 10, 10))
    manifest = tmp_path / "approved.json"
    _approved_manifest(manifest, [_action(atl, "atl")])
    _source_image(atl, (10, 10, 200))

    with pytest.raises(ValueError, match="changed since audit"):
        module.materialize_snapshot(manifest, tmp_path / "snapshot", runtime, training_config)
    assert not (tmp_path / "snapshot").exists()

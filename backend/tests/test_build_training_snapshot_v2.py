from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "build_training_snapshot.py"
SPEC = importlib.util.spec_from_file_location("build_training_snapshot", SCRIPT)
assert SPEC and SPEC.loader
snapshot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(snapshot)


def _write_bmp(path: Path, color: tuple[int, int, int]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 7), color).save(path, format="BMP")
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixtures(root: Path) -> tuple[Path, Path, Path]:
    class_order = ["alpha", "beta", "gamma"]
    runtime = root / "runtime.json"
    runtime.write_text(
        json.dumps({"class_names": class_order, "num_classes": len(class_order)}),
        encoding="utf-8",
    )

    legacy_images = []
    for index, (name, color) in enumerate(
        [("alpha", (255, 0, 0)), ("beta", (0, 255, 0)), ("gamma", (0, 0, 255))]
    ):
        path = _write_bmp(root / "legacy" / f"{index}.bmp", color)
        legacy_images.append(
            {
                "class_name": name,
                "archive_path": f"Elements/{index:04d}-{name}/01_01_0{index + 1}-1.bmp",
                "output_path": str(path),
                "source_group": f"page:01:01:0{index + 1}",
                "source_sha256": _sha(path),
                "output_sha256": _sha(path),
            }
        )
    legacy = root / "legacy.json"
    legacy.write_text(
        json.dumps(
            {
                "schema_version": "legacy-elements-freeze.v1",
                "class_order": class_order,
                "class_order_sha256": snapshot.class_order_sha256(class_order),
                "images": legacy_images,
            }
        ),
        encoding="utf-8",
    )

    external_rows = []
    for index, (name, color) in enumerate(
        [("alpha", (128, 0, 0)), ("beta", (0, 128, 0)), ("gamma", (0, 0, 128))]
    ):
        path = _write_bmp(root / "external" / f"{index}.bmp", color)
        external_rows.append(
            {
                "class_label": index,
                "class_name": name,
                "output_path": str(path),
                "output_sha256": _sha(path),
                "source_path": f"/archive/{20 + index}_01_01-1.jpg",
                "source_sha256": _sha(path),
                "source_pixel_sha256": snapshot.pixel_sha256(path),
            }
        )
    external = root / "external.json"
    external.write_text(
        json.dumps(
            {
                "schema_version": "external-corpus-training-snapshot.v1",
                "rows": external_rows,
            }
        ),
        encoding="utf-8",
    )
    return runtime, legacy, external


def _plan(runtime: Path, legacy: Path, external: Path, **kwargs):
    return snapshot.plan_training_snapshot(
        runtime_config=runtime,
        legacy_manifest=legacy,
        external_snapshot=external,
        annotations_dirs=[],
        dev_fraction=0.2,
        locked_test_fraction=0.2,
        min_train_rows_per_class=1,
        **kwargs,
    )


def test_snapshot_v2_is_deterministic_and_uses_runtime_labels(tmp_path: Path) -> None:
    runtime, legacy, external = _fixtures(tmp_path)
    first = _plan(runtime, legacy, external)
    second = _plan(runtime, legacy, external)

    assert first == second
    assert first["schema_version"] == "training-snapshot.v2"
    assert first["class_order"] == ["alpha", "beta", "gamma"]
    assert {row["class_label"] for row in first["rows"]} == {0, 1, 2}
    assert first["snapshot_id"] == f"snapshot-{first['content_sha256'][:20]}"
    assert first["live_annotation_count"] == 0
    assert first["live_annotations_sha256"] == snapshot.live_annotations_sha256([])
    assert first["ready_for_training"] is True
    assert isinstance(first["promotion_evaluation_ready"], bool)
    assert first["policy"]["version"] == "locked-source-group.v3"
    assert first["split_targets_satisfied"] is True
    assert first["split_target_deficits"] == {"dev": 0, "locked_test": 0}
    assert first["holdout_eligibility"]["eligible_class_count"] == 3
    assert first["component_count"] <= first["source_group_count"]
    assert all(counts["train"] >= 1 for counts in first["class_split_counts"].values())
    assignments = first["source_group_assignments"]
    assert set(assignments.values()) <= {"train", "dev", "locked_test"}


def test_live_training_retires_whole_duplicate_component_without_changing_parent(tmp_path, monkeypatch):
    runtime, legacy, external = _fixtures(tmp_path)
    parent = _plan(runtime, legacy, external)
    held = next(row for row in parent["rows"] if row["dataset_split"] == "locked_test")
    parent_path = tmp_path / "parent.json"
    parent_path.write_text(json.dumps(parent))
    parent_bytes = parent_path.read_bytes()
    live = dict(held, row_id="live-duplicate", source_kind="live_annotation",
                source_id="analysis:0", source_group="live-page", duplicate_of=None)
    second_path = _write_bmp(tmp_path / "live.bmp", (123, 45, 67))
    second = dict(live, row_id="live-new", source_id="analysis:1",
                  source_path=str(second_path), source_sha256=_sha(second_path),
                  source_pixel_sha256=snapshot.pixel_sha256(second_path))
    monkeypatch.setattr(snapshot, "load_live_annotation_rows", lambda *_: ([], [live, second]))
    child = _plan(runtime, legacy, external, parent_manifest=parent_path,
                  train_live_annotations=True, allow_underfilled_holdouts=True)
    assert parent_path.read_bytes() == parent_bytes
    assert child["live_annotation_count"] == child["live_train_count"] == 2
    assert child["live_annotation_usage"]["training_row_ids"] == sorted([held["row_id"], "live-new"])
    assert child["source_group_assignments"]["live-page"] == "train"
    assert child["source_group_assignments"][held["source_group"]] == "train"
    for group, split in parent["source_group_assignments"].items():
        if group != held["source_group"]:
            assert child["source_group_assignments"][group] == split
    component = next(c for c in child["components"] if "live-page" in c["source_groups"])
    assert component["retired_holdout_assignments"] == {held["source_group"]: "locked_test"}
    assert child["policy"]["version"] == "approved-live-train.v1"
    assert child == _plan(runtime, legacy, external, parent_manifest=parent_path,
                          train_live_annotations=True, allow_underfilled_holdouts=True)


def test_underfilled_holdouts_fail_closed_without_research_override(
    tmp_path: Path,
) -> None:
    runtime, legacy, external = _fixtures(tmp_path)

    with pytest.raises(ValueError, match="heldout split targets are infeasible"):
        snapshot.plan_training_snapshot(
            runtime_config=runtime,
            legacy_manifest=legacy,
            external_snapshot=external,
            annotations_dirs=[],
            dev_fraction=0.4,
            locked_test_fraction=0.4,
            min_train_rows_per_class=1,
        )

    research = snapshot.plan_training_snapshot(
        runtime_config=runtime,
        legacy_manifest=legacy,
        external_snapshot=external,
        annotations_dirs=[],
        dev_fraction=0.4,
        locked_test_fraction=0.4,
        min_train_rows_per_class=1,
        allow_underfilled_holdouts=True,
    )
    assert research["split_targets_satisfied"] is False
    assert research["policy"]["allow_underfilled_holdouts"] is True
    assert any(research["split_target_deficits"].values())


def test_cli_requires_named_override_for_underfilled_research_snapshot(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runtime, legacy, external = _fixtures(tmp_path)
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    (annotations / "review-index.json").write_text(
        json.dumps({"schema_version": "annotation-review.v1", "decisions": {}}),
        encoding="utf-8",
    )
    common = [
        "--runtime-config",
        str(runtime),
        "--legacy-manifest",
        str(legacy),
        "--external-snapshot",
        str(external),
        "--annotations-dir",
        str(annotations),
        "--dev-fraction",
        "0.4",
        "--locked-test-fraction",
        "0.4",
        "--min-train-rows-per-class",
        "1",
        "--dry-run",
        "--json",
    ]

    assert snapshot.main(common) == 2
    assert "heldout split targets are infeasible" in capsys.readouterr().err
    assert snapshot.main([*common, "--allow-underfilled-holdouts"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert payload["destination"] is None
    assert payload["split_policy_version"] == "locked-source-group.v3"
    assert payload["split_targets_satisfied"] is False
    assert payload["allow_underfilled_holdouts"] is True
    assert any(payload["split_target_deficits"].values())


def test_minimum_train_floor_is_enforced_and_manifest_bound(tmp_path: Path) -> None:
    runtime, legacy, external = _fixtures(tmp_path)
    research = snapshot.plan_training_snapshot(
        runtime_config=runtime,
        legacy_manifest=legacy,
        external_snapshot=external,
        annotations_dirs=[],
        dev_fraction=0.2,
        locked_test_fraction=0.2,
        min_train_rows_per_class=2,
        allow_underfilled_holdouts=True,
    )

    assert research["policy"]["minimum_train_rows_per_class"] == 2
    assert all(
        counts["train"] >= 2 for counts in research["class_split_counts"].values()
    )
    assert research["ready_for_training"] is False


def test_materialization_is_write_once_and_metadata_keeps_split_contract(tmp_path: Path) -> None:
    runtime, legacy, external = _fixtures(tmp_path)
    manifest = _plan(runtime, legacy, external)
    destination = snapshot.materialize_training_snapshot(manifest, tmp_path / "snapshots")

    persisted = json.loads((destination / "snapshot_manifest.json").read_text())
    metadata = pd.read_csv(destination / "metadata.csv")
    assert len(metadata) == persisted["row_count"]
    assert metadata["class_label"].tolist() == [row["class_label"] for row in persisted["rows"]]
    assert metadata["source_group"].tolist() == [row["source_group"] for row in persisted["rows"]]
    assert metadata["dataset_split"].tolist() == [row["dataset_split"] for row in persisted["rows"]]
    assert (destination / "checksums.json").is_file()
    with pytest.raises(FileExistsError, match="snapshot already exists"):
        snapshot.materialize_training_snapshot(manifest, tmp_path / "snapshots")


def test_parent_assignments_are_stable_and_parent_rows_cannot_disappear(tmp_path: Path) -> None:
    runtime, legacy, external = _fixtures(tmp_path)
    parent = _plan(runtime, legacy, external)
    parent_path = tmp_path / "parent.json"
    parent_path.write_text(json.dumps(parent), encoding="utf-8")

    payload = json.loads(external.read_text())
    new_path = _write_bmp(tmp_path / "external" / "new.bmp", (77, 77, 77))
    payload["rows"].append(
        {
            "class_label": 0,
            "class_name": "alpha",
            "output_path": str(new_path),
            "output_sha256": _sha(new_path),
            "source_path": "/archive/99_01_01-1.jpg",
            "source_sha256": _sha(new_path),
            "source_pixel_sha256": snapshot.pixel_sha256(new_path),
        }
    )
    external.write_text(json.dumps(payload), encoding="utf-8")
    child = _plan(runtime, legacy, external, parent_manifest=parent_path)

    for group, split in parent["source_group_assignments"].items():
        assert child["source_group_assignments"][group] == split
    child_components = {item["component_id"]: item for item in child["components"]}
    for component in child_components.values():
        if set(component["source_groups"]) & set(parent["source_group_assignments"]):
            assert component["assignment_origin"] == "parent"
    assert set(parent["active_row_ids"]).issubset(child["active_row_ids"])
    assert child["parent_snapshot_id"] == parent["snapshot_id"]
    assert "Parent-origin components are immutable" in child[
        "holdout_eligibility"
    ]["operational_definition"]
    for item in child["holdout_eligibility"]["classes"]:
        assert all(
            child_components[component_id]["assignment_origin"] == "new"
            for component_id in item["movable_new_train_component_ids"]
        )

    payload["rows"] = payload["rows"][1:]
    external.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="silently remove parent rows"):
        _plan(runtime, legacy, external, parent_manifest=parent_path)


def test_exact_pixel_label_conflict_blocks_or_is_explicitly_excluded(tmp_path: Path) -> None:
    runtime, legacy, external = _fixtures(tmp_path)
    payload = json.loads(external.read_text())
    legacy_payload = json.loads(legacy.read_text())
    conflicting_path = Path(legacy_payload["images"][0]["output_path"])
    payload["rows"][0].update(
        {
            "class_label": 1,
            "class_name": "beta",
            "output_path": str(conflicting_path),
            "output_sha256": _sha(conflicting_path),
            "source_sha256": _sha(conflicting_path),
            "source_pixel_sha256": snapshot.pixel_sha256(conflicting_path),
        }
    )
    external.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="conflicting classes"):
        _plan(runtime, legacy, external)
    excluded = _plan(runtime, legacy, external, exclude_conflicts=True)
    assert excluded["conflict_count"] == 1
    assert excluded["conflicts"][0]["resolution"] == "excluded_pending_manual_review"
    conflict_ids = {row["row_id"] for row in excluded["conflicts"][0]["rows"]}
    assert conflict_ids.isdisjoint(excluded["active_row_ids"])

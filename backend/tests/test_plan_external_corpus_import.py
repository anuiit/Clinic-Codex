import importlib.util
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]


def load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


audit = load_script("audit_external_corpus_for_plan_tests", ROOT / "scripts" / "audit_external_corpus.py")
plan = load_script("plan_external_corpus_import", ROOT / "scripts" / "plan_external_corpus_import.py")


def write_config(path: Path, classes: list[str]) -> Path:
    path.write_text(json.dumps({"class_names": classes}), encoding="utf-8")
    return path


def write_image(path: Path, color=(1, 2, 3)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), color=color).save(path)


def build_audit_files(tmp_path: Path, classes: list[str], source: Path):
    return build_audit_files_multi(tmp_path, classes, [("fixture", source)])


def build_audit_files_multi(tmp_path: Path, classes: list[str], sources: list[tuple[str, Path]]):
    config = write_config(tmp_path / "config.json", classes)
    report, inventory = audit.build_report(
        config,
        [audit.SourceSpec(name=name, path=source) for name, source in sources],
        hash_files=True,
        pixel_hash_files=True,
        k_shot=3,
        q_queries=5,
        min_images_per_class=2,
        val_fraction=0.15,
        n_way=max(1, min(20, len(classes))),
        require_full_active_coverage=True,
    )
    audit_json = tmp_path / "audit.json"
    inventory_json = tmp_path / "inventory.json"
    audit.write_outputs(report, inventory, audit_json, None, inventory_json)
    return config, audit_json, inventory_json


def test_plan_nominal_outputs_manifest_with_status_closure(tmp_path):
    src = tmp_path / "src"
    for cls, color in [("0015-atl", (255, 0, 0)), ("0016-tetl", (0, 255, 0))]:
        for idx in range(8):
            write_image(src / cls / f"{idx}.bmp", color=(color[0], color[1], idx))
    config, audit_json, inventory_json = build_audit_files(tmp_path, ["atl", "tetl"], src)
    output = tmp_path / "plan"

    exit_code = plan.main([
        "--audit-json", str(audit_json),
        "--inventory-json", str(inventory_json),
        "--active-config", str(config),
        "--output-dir", str(output),
        "--k-shot", "3",
        "--q-queries", "5",
        "--force",
    ])

    assert exit_code == 0
    manifest = json.loads((output / "import_manifest.preview.json").read_text())
    assert manifest["image_count"] == 16
    assert manifest["status_counts"] == {"selected": 16}
    assert not manifest["blockers"]
    assert (output / "class_mapping_proposal.csv").exists()


def test_plan_lists_cross_class_conflict_extra_and_weak_class(tmp_path):
    src = tmp_path / "src"
    write_image(src / "0015-atl" / "a.bmp", color=(9, 9, 9))
    write_image(src / "0016-tetl" / "b.bmp", color=(9, 9, 9))
    write_image(src / "9999-extra" / "x.bmp", color=(1, 1, 1))
    config, audit_json, inventory_json = build_audit_files(tmp_path, ["atl", "tetl"], src)
    output = tmp_path / "plan"

    exit_code = plan.main([
        "--audit-json", str(audit_json),
        "--inventory-json", str(inventory_json),
        "--active-config", str(config),
        "--output-dir", str(output),
        "--force",
    ])

    assert exit_code == 1
    manifest = json.loads((output / "import_manifest.preview.json").read_text())
    assert manifest["status_counts"] == {"conflict_cross_class": 2, "unmapped_extra": 1}
    assert any(reason.startswith("conflict_cross_class_images:2") for reason in manifest["blockers"])
    assert any(reason.startswith("unmapped_extra_classes:1") for reason in manifest["blockers"])
    assert "needs_policy" in (output / "weak_classes.csv").read_text()
    assert "quarantine_all" in (output / "duplicate_conflicts.csv").read_text()
    assert "9999-extra" in (output / "unmapped_extras.csv").read_text()


def test_plan_rejects_truncated_duplicate_groups(tmp_path):
    src = tmp_path / "src"
    write_image(src / "0015-atl" / "a.bmp", color=(7, 7, 7))
    write_image(src / "0015-atl" / "b.bmp", color=(7, 7, 7))
    config, audit_json, inventory_json = build_audit_files(tmp_path, ["atl"], src)
    data = json.loads(audit_json.read_text())
    assert data["duplicate_groups"] or data["pixel_duplicate_groups"]
    data["pixel_duplicate_groups"][0]["count"] += 1
    audit_json.write_text(json.dumps(data), encoding="utf-8")

    exit_code = plan.main([
        "--audit-json", str(audit_json),
        "--inventory-json", str(inventory_json),
        "--active-config", str(config),
        "--output-dir", str(tmp_path / "plan"),
        "--force",
    ])

    assert exit_code == 2


def test_plan_rejects_audit_inventory_run_id_mismatch(tmp_path):
    src = tmp_path / "src"
    write_image(src / "0015-atl" / "a.bmp", color=(1, 2, 3))
    config, audit_json, inventory_json = build_audit_files(tmp_path, ["atl"], src)
    inventory = json.loads(inventory_json.read_text())
    inventory["audit_run_id"] = "different"
    inventory_json.write_text(json.dumps(inventory), encoding="utf-8")

    exit_code = plan.main([
        "--audit-json", str(audit_json),
        "--inventory-json", str(inventory_json),
        "--active-config", str(config),
        "--output-dir", str(tmp_path / "plan"),
        "--force",
    ])

    assert exit_code == 2



def test_plan_marks_same_pixels_different_bytes_as_suspected_reencode(tmp_path):
    src = tmp_path / "src"
    write_image(src / "0015-atl" / "a.bmp", color=(20, 30, 40))
    write_image(src / "0015-atl" / "b.png", color=(20, 30, 40))
    for idx in range(7):
        write_image(src / "0015-atl" / f"unique-{idx}.bmp", color=(20, 30, 41 + idx))
    config, audit_json, inventory_json = build_audit_files(tmp_path, ["atl"], src)
    output = tmp_path / "plan"

    exit_code = plan.main([
        "--audit-json", str(audit_json),
        "--inventory-json", str(inventory_json),
        "--active-config", str(config),
        "--output-dir", str(output),
        "--force",
    ])

    assert exit_code == 0
    manifest = json.loads((output / "import_manifest.preview.json").read_text())
    assert manifest["status_counts"] == {"selected": 8, "suspected_reencode": 1}
    suspected = [image for image in manifest["images"] if image["status"] == "suspected_reencode"]
    assert suspected and suspected[0]["status_reason"].startswith("same_pixels_as:")


def test_plan_source_priority_selects_duplicate_primary(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    write_image(first / "0015-atl" / "dupe.bmp", color=(1, 1, 1))
    write_image(second / "0015-atl" / "dupe.bmp", color=(1, 1, 1))
    for idx in range(7):
        write_image(first / "0015-atl" / f"unique-{idx}.bmp", color=(1, 1, 2 + idx))
    config, audit_json, inventory_json = build_audit_files_multi(
        tmp_path,
        ["atl"],
        [("low_priority", first), ("preferred", second)],
    )
    output = tmp_path / "plan"

    exit_code = plan.main([
        "--audit-json", str(audit_json),
        "--inventory-json", str(inventory_json),
        "--active-config", str(config),
        "--output-dir", str(output),
        "--source-priority", "preferred,low_priority",
        "--force",
    ])

    assert exit_code == 0
    manifest = json.loads((output / "import_manifest.preview.json").read_text())
    assert manifest["status_counts"] == {"duplicate_same_class": 1, "selected": 8}
    selected_duplicate = [
        image for image in manifest["images"]
        if image["path"].endswith("dupe.bmp") and image["status"] == "selected"
    ]
    assert len(selected_duplicate) == 1
    assert selected_duplicate[0]["source"] == "preferred"


def test_plan_freshness_ignores_nested_images_like_audit(tmp_path):
    src = tmp_path / "src"
    for idx in range(8):
        write_image(src / "0015-atl" / f"{idx}.bmp", color=(2, 3, idx))
    config, audit_json, inventory_json = build_audit_files(tmp_path, ["atl"], src)
    write_image(src / "0015-atl" / "nested" / "not-audited.bmp", color=(255, 255, 255))

    exit_code = plan.main([
        "--audit-json", str(audit_json),
        "--inventory-json", str(inventory_json),
        "--active-config", str(config),
        "--output-dir", str(tmp_path / "plan"),
        "--force",
    ])

    assert exit_code == 0


def test_plan_allow_stale_bypasses_config_freshness_check(tmp_path):
    src = tmp_path / "src"
    for idx in range(8):
        write_image(src / "0015-atl" / f"{idx}.bmp", color=(3, 4, idx))
    config, audit_json, inventory_json = build_audit_files(tmp_path, ["atl"], src)
    write_config(config, ["atl", "new-class"])

    stale_exit = plan.main([
        "--audit-json", str(audit_json),
        "--inventory-json", str(inventory_json),
        "--active-config", str(config),
        "--output-dir", str(tmp_path / "stale"),
        "--force",
    ])
    allow_stale_exit = plan.main([
        "--audit-json", str(audit_json),
        "--inventory-json", str(inventory_json),
        "--active-config", str(config),
        "--output-dir", str(tmp_path / "allow-stale"),
        "--allow-stale",
        "--force",
    ])

    assert stale_exit == 2
    assert allow_stale_exit == 0


def test_plan_empty_files_are_invalid_not_selected(tmp_path):
    src = tmp_path / "src"
    for idx in range(8):
        write_image(src / "0015-atl" / f"{idx}.bmp", color=(4, 5, idx))
    empty = src / "0015-atl" / "empty.bmp"
    empty.parent.mkdir(parents=True, exist_ok=True)
    empty.write_bytes(b"")
    config, audit_json, inventory_json = build_audit_files(tmp_path, ["atl"], src)
    output = tmp_path / "plan"

    exit_code = plan.main([
        "--audit-json", str(audit_json),
        "--inventory-json", str(inventory_json),
        "--active-config", str(config),
        "--output-dir", str(output),
        "--force",
    ])

    assert exit_code == 1
    manifest = json.loads((output / "import_manifest.preview.json").read_text())
    assert manifest["status_counts"] == {"invalid_empty_file": 1, "selected": 8}
    assert "invalid_empty_files:1" in manifest["blockers"]
    assert not any(image["path"] == str(empty) and image["status"] == "selected" for image in manifest["images"])


def test_plan_requires_force_to_overwrite_outputs(tmp_path):
    src = tmp_path / "src"
    for idx in range(8):
        write_image(src / "0015-atl" / f"{idx}.bmp", color=(5, 6, idx))
    config, audit_json, inventory_json = build_audit_files(tmp_path, ["atl"], src)
    output = tmp_path / "plan"
    args = [
        "--audit-json", str(audit_json),
        "--inventory-json", str(inventory_json),
        "--active-config", str(config),
        "--output-dir", str(output),
    ]

    assert plan.main(args) == 0
    assert plan.main(args) == 2
    assert plan.main([*args, "--force"]) == 0


def test_plan_invalid_unreadable_images_are_not_selected(tmp_path):
    src = tmp_path / "src"
    for idx in range(8):
        write_image(src / "0015-atl" / f"{idx}.bmp", color=(6, 7, idx))
    bad = src / "0015-atl" / "not-really-an-image.bmp"
    bad.write_text("this is not a bitmap", encoding="utf-8")
    config, audit_json, inventory_json = build_audit_files(tmp_path, ["atl"], src)
    output = tmp_path / "plan"

    exit_code = plan.main([
        "--audit-json", str(audit_json),
        "--inventory-json", str(inventory_json),
        "--active-config", str(config),
        "--output-dir", str(output),
        "--force",
    ])

    assert exit_code == 1
    manifest = json.loads((output / "import_manifest.preview.json").read_text())
    assert manifest["status_counts"] == {"invalid_unreadable_image": 1, "selected": 8}
    assert "invalid_unreadable_images:1" in manifest["blockers"]
    assert not any(image["path"] == str(bad) and image["status"] == "selected" for image in manifest["images"])


def test_plan_surfaces_mapped_unmapped_duplicate_conflicts(tmp_path):
    src = tmp_path / "src"
    write_image(src / "0015-atl" / "mapped-dupe.bmp", color=(7, 8, 9))
    write_image(src / "9999-extra" / "unmapped-dupe.bmp", color=(7, 8, 9))
    for idx in range(8):
        write_image(src / "0015-atl" / f"unique-{idx}.bmp", color=(7, 8, 10 + idx))
    config, audit_json, inventory_json = build_audit_files(tmp_path, ["atl"], src)
    output = tmp_path / "plan"

    exit_code = plan.main([
        "--audit-json", str(audit_json),
        "--inventory-json", str(inventory_json),
        "--active-config", str(config),
        "--output-dir", str(output),
        "--force",
    ])

    assert exit_code == 1
    manifest = json.loads((output / "import_manifest.preview.json").read_text())
    assert manifest["status_counts"] == {"conflict_mapped_unmapped": 2, "selected": 8}
    assert "conflict_mapped_unmapped_images:2" in manifest["blockers"]
    assert "conflict_mapped_unmapped" in (output / "duplicate_conflicts.csv").read_text()


def test_plan_carries_audit_preflight_blockers_into_manifest(tmp_path):
    src = tmp_path / "src"
    for idx in range(8):
        write_image(src / "0015-atl" / f"{idx}.bmp", color=(8, 9, idx))
    config, audit_json, inventory_json = build_audit_files(tmp_path, ["atl", "tetl"], src)
    output = tmp_path / "plan"

    exit_code = plan.main([
        "--audit-json", str(audit_json),
        "--inventory-json", str(inventory_json),
        "--active-config", str(config),
        "--output-dir", str(output),
        "--force",
    ])

    assert exit_code == 1
    manifest = json.loads((output / "import_manifest.preview.json").read_text())
    assert not manifest["audit_preflight_pass"]
    assert manifest["audit_preflight_blocking_reasons"]
    assert any(blocker.startswith("audit_preflight:") for blocker in manifest["blockers"])

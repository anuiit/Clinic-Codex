import csv
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


generate = load_script("generate_external_corpus_review", ROOT / "scripts" / "generate_external_corpus_review.py")
validate = load_script("validate_external_corpus_decisions", ROOT / "scripts" / "validate_external_corpus_decisions.py")
generate_ui = load_script("generate_external_corpus_review_ui", ROOT / "scripts" / "generate_external_corpus_review_ui.py")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_plan_dir(tmp_path: Path) -> Path:
    plan_dir = tmp_path / "plan"
    plan_dir.mkdir()
    manifest = {
        "schema_version": "external-corpus-import-plan.v1",
        "image_count": 6,
        "promotable": False,
        "audit_preflight_pass": False,
        "blockers": ["conflict_cross_class_images:1", "unmapped_extra_classes:1", "weak_classes:1"],
        "status_counts": {
            "selected": 2,
            "duplicate_same_class": 1,
            "conflict_cross_class": 1,
            "unmapped_extra": 1,
            "suspected_reencode": 1,
        },
        "class_counts": {
            "atl": {"raw_count": 3, "selected_count": 2},
            "tetl": {"raw_count": 1, "selected_count": 0},
        },
        "images": [
            {
                "source": "fixture",
                "class_dir": "0015-atl",
                "matched_active_class": "atl",
                "path": "/tmp/selected-a.bmp",
                "status": "selected",
                "status_reason": "primary_unique_or_no_duplicate",
                "sha256_bytes": "a",
                "sha256_pixels": "pa",
            },
            {
                "source": "fixture",
                "class_dir": "0015-atl",
                "matched_active_class": "atl",
                "path": "/tmp/selected-b.bmp",
                "status": "selected",
                "status_reason": "primary_unique_or_no_duplicate",
                "sha256_bytes": "b",
                "sha256_pixels": "pb",
            },
            {
                "source": "fixture",
                "class_dir": "0015-atl",
                "matched_active_class": "atl",
                "path": "/tmp/dupe.bmp",
                "status": "duplicate_same_class",
                "status_reason": "same_byte_or_pixel_as:/tmp/selected-a.bmp",
                "sha256_bytes": "a",
                "sha256_pixels": "pa",
            },
            {
                "source": "fixture",
                "class_dir": "0016-tetl",
                "matched_active_class": "tetl",
                "path": "/tmp/conflict.bmp",
                "status": "conflict_cross_class",
                "status_reason": "duplicate_hash_maps_to_multiple_active_classes",
                "sha256_bytes": "c",
                "sha256_pixels": "pc",
            },
            {
                "source": "fixture",
                "class_dir": "9999-extra",
                "matched_active_class": None,
                "path": "/tmp/unmapped.bmp",
                "status": "unmapped_extra",
                "status_reason": "source_class_not_in_active_taxonomy",
                "sha256_bytes": "d",
                "sha256_pixels": "pd",
            },
            {
                "source": "fixture",
                "class_dir": "0015-atl",
                "matched_active_class": "atl",
                "path": "/tmp/reencode.png",
                "status": "suspected_reencode",
                "status_reason": "same_pixels_as:/tmp/selected-b.bmp",
                "sha256_bytes": "e",
                "sha256_pixels": "pb",
            },
        ],
    }
    (plan_dir / "import_manifest.preview.json").write_text(json.dumps(manifest), encoding="utf-8")
    write_csv(
        plan_dir / "unmapped_extras.csv",
        ["source_dataset", "source_folder", "normalized_name", "image_count", "decision", "reviewer", "notes"],
        [{"source_dataset": "fixture", "source_folder": "9999-extra", "normalized_name": "extra", "image_count": "1", "decision": "quarantine", "reviewer": "", "notes": ""}],
    )
    write_csv(
        plan_dir / "weak_classes.csv",
        ["class_name", "raw_count", "unique_count", "train_count", "val_count", "decision", "reasons", "reviewer", "notes"],
        [{"class_name": "atl", "raw_count": "3", "unique_count": "2", "train_count": "1", "val_count": "1", "decision": "needs_policy", "reasons": "train_split_below_k_shot", "reviewer": "", "notes": ""}],
    )
    write_csv(
        plan_dir / "duplicate_conflicts.csv",
        ["status", "status_reason", "sha256_bytes", "sha256_pixels", "source_dataset", "source_folder", "active_class_name", "path", "decision", "reviewer", "notes"],
        [{"status": "conflict_cross_class", "status_reason": "duplicate_hash_maps_to_multiple_active_classes", "sha256_bytes": "c", "sha256_pixels": "pc", "source_dataset": "fixture", "source_folder": "0016-tetl", "active_class_name": "tetl", "path": "/tmp/conflict.bmp", "decision": "quarantine_all", "reviewer": "", "notes": ""}],
    )
    return plan_dir


def test_generate_review_artifacts_creates_decision_template_and_checklist(tmp_path):
    plan_dir = build_plan_dir(tmp_path)

    summary = generate.build_review_artifacts(plan_dir, plan_dir, sample_size=10)

    assert summary["decision_rows"] == 3
    assert summary["required_by_type"] == {"image_conflict": 1, "unmapped_class": 1, "weak_class": 1}
    assert (plan_dir / "review_checklist.md").exists()
    assert "External corpus import review checklist" in (plan_dir / "review_checklist.md").read_text()
    assert len(read_csv(plan_dir / "reviewer_decisions.template.csv")) == 3
    assert len(read_csv(plan_dir / "suspected_reencode_spotcheck.csv")) == 1


def test_validate_blank_template_is_not_ready_and_writes_validation(tmp_path):
    plan_dir = build_plan_dir(tmp_path)
    generate.build_review_artifacts(plan_dir, plan_dir, sample_size=10)

    code, summary = validate.validate_decisions(
        plan_dir / "import_manifest.preview.json",
        plan_dir / "reviewer_decisions.template.csv",
        plan_dir / "review_validation.json",
        plan_dir / "import_manifest.approved.json",
    )

    assert code == 1
    assert not summary["ready_for_import"]
    assert summary["missing_decision_count"] == 3
    assert (plan_dir / "review_validation.json").exists()
    assert not (plan_dir / "import_manifest.approved.json").exists()


def test_validate_rejects_bad_target_class(tmp_path):
    plan_dir = build_plan_dir(tmp_path)
    generate.build_review_artifacts(plan_dir, plan_dir, sample_size=10)
    rows = read_csv(plan_dir / "reviewer_decisions.template.csv")
    for row in rows:
        if row["item_type"] == "image_conflict":
            row["decision"] = "keep_as"
            row["target_class"] = "missing-class"
        elif row["item_type"] == "unmapped_class":
            row["decision"] = "quarantine"
        elif row["item_type"] == "weak_class":
            row["decision"] = "accept_weak"
    write_csv(plan_dir / "reviewer_decisions.csv", generate.REQUIRED_DECISION_HEADERS, rows)

    code, summary = validate.validate_decisions(
        plan_dir / "import_manifest.preview.json",
        plan_dir / "reviewer_decisions.csv",
        None,
        None,
    )

    assert code == 1
    assert summary["invalid_decision_count"] == 1
    assert "active class" in summary["errors"][0]


def test_validate_complete_decisions_writes_approved_manifest(tmp_path):
    plan_dir = build_plan_dir(tmp_path)
    generate.build_review_artifacts(plan_dir, plan_dir, sample_size=10)
    rows = read_csv(plan_dir / "reviewer_decisions.template.csv")
    for row in rows:
        if row["item_type"] == "image_conflict":
            row["decision"] = "keep_as"
            row["target_class"] = "tetl"
        elif row["item_type"] == "unmapped_class":
            row["decision"] = "map_to_existing"
            row["target_class"] = "atl"
        elif row["item_type"] == "weak_class":
            row["decision"] = "exclude_until_more_data"
    write_csv(plan_dir / "reviewer_decisions.csv", generate.REQUIRED_DECISION_HEADERS, rows)

    code, summary = validate.validate_decisions(
        plan_dir / "import_manifest.preview.json",
        plan_dir / "reviewer_decisions.csv",
        plan_dir / "review_validation.json",
        plan_dir / "import_manifest.approved.json",
    )

    assert code == 0
    assert summary["ready_for_import"]
    approved = json.loads((plan_dir / "import_manifest.approved.json").read_text())
    assert approved["ready_for_import"]
    assert approved["action_counts"] == {"exclude": 5, "import": 1}
    assert approved["created_class_proposals"] == []
    imported_paths = {action["path"] for action in approved["actions"] if action["action"] == "import"}
    assert imported_paths == {"/tmp/conflict.bmp"}


def test_generate_review_ui_html(tmp_path):
    plan_dir = build_plan_dir(tmp_path)
    generate.build_review_artifacts(plan_dir, plan_dir, sample_size=10)
    output = plan_dir / "review_ui.html"

    summary = generate_ui.generate(plan_dir, plan_dir / "reviewer_decisions.template.csv", output)

    assert summary["decision_rows"] == 3
    assert summary["conflict_groups"] == 1
    assert summary["unmapped_groups"] == 1
    assert summary["weak_classes"] == 1
    html = output.read_text(encoding="utf-8")
    assert "External corpus review UI" in html
    assert "Exporter CSV" in html
    assert "file:///tmp/conflict.bmp" in html
    assert "reviewer_decisions.csv" in html

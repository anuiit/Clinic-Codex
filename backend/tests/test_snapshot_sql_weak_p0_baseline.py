import csv
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "snapshot_sql_weak_p0_baseline.py"
spec = importlib.util.spec_from_file_location("snapshot_sql_weak_p0_baseline", SCRIPT)
assert spec is not None
assert spec.loader is not None
snapshot = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = snapshot
spec.loader.exec_module(snapshot)


def write_config(path: Path, classes: list[str], num_classes: int | None = None) -> Path:
    path.write_text(json.dumps({"num_classes": len(classes) if num_classes is None else num_classes, "class_names": classes}), encoding="utf-8")
    return path


def write_manifest(path: Path, config_path: Path) -> Path:
    path.write_text(json.dumps({"active_config_sha256": snapshot.sha256_file(config_path), "schema_version": "fixture"}), encoding="utf-8")
    return path


def write_candidate_csv(path: Path, rows: list[dict[str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "class_name",
        "active_model_class",
        "source_image_path",
        "sha256_bytes",
        "sha256_pixels",
        "status",
        "existing_plan_duplicate_classes",
        "decode_error",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_v2(path_json: Path, path_csv: Path, candidate_rows=3, conflict_groups=1, affected=2, under=None) -> None:
    if under is None:
        under = [{"class_name": "a", "projected_unique_if_conflicts_excluded": 7}]
    path_json.write_text(
        json.dumps(
            {
                "candidate_rows": candidate_rows,
                "internal_cross_class_pixel_conflict_groups": conflict_groups,
                "classes_affected_by_internal_conflicts": affected,
                "classes_still_under_8_if_internal_conflicts_excluded": under,
            }
        ),
        encoding="utf-8",
    )
    path_csv.write_text("class_name,current_unique_count\na,2\n", encoding="utf-8")


def write_sql(path: Path) -> Path:
    path.write_text(
        """-- phpMyAdmin SQL Dump
-- Generation Time: Fixture
INSERT INTO `ai_codex` (`id`, `titre`) VALUES
(1, 'VALUES in string, not keyword'),
(2, 'paren ),( inside');
/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
-- comment with (parentheses); and VALUES
INSERT INTO `ai_element` (`id`, `element`) VALUES
(1, 'a'),
(2, 'b\\' quote'),
(3, 'c'' doubled'),
(4, NULL);
INSERT INTO `ai_element` (`id`, `values`) VALUES
(5, 'weird),( value');
INSERT INTO `ai_glyphe` (`id`, `element`) VALUES
(1, 'x'),
(2, 'multi\nline');
INSERT INTO `ai_glyphe` (`id`, `element`) VALUES
(3, 'second block') ON DUPLICATE KEY UPDATE `element` = VALUES(`element`);
INSERT INTO `ai_plate` (`id`) VALUES (1);
INSERT INTO `ai_zone` (`id`) VALUES (1),(2);
""",
        encoding="utf-8",
    )
    return path


def fixture_args(tmp_path, monkeypatch, *, manifest=True, bad_config=False, bad_candidate=False):
    monkeypatch.setattr(snapshot, "EXPECTED_CANDIDATE_ROWS", 3)
    monkeypatch.setattr(snapshot, "EXPECTED_CONFLICT_GROUPS", 1)
    monkeypatch.setattr(snapshot, "EXPECTED_CONFLICT_AFFECTED_CLASSES", 2)
    monkeypatch.setattr(snapshot, "DEFAULT_UNDER_8_CLASSES", ["a"])
    sql = write_sql(tmp_path / "fixture.sql")
    expected = tmp_path / "expected.json"
    expected.write_text(json.dumps({"ai_codex": 2, "ai_element": 5, "ai_glyphe": 3, "ai_plate": 1, "ai_zone": 2}), encoding="utf-8")
    classes = ["a", "b"] + [f"filler_{i}" for i in range(284)]
    if bad_config:
        classes = ["a", "b"]
    config = write_config(tmp_path / "config.json", classes)
    manifest_path = tmp_path / "manifest.json"
    if manifest:
        write_manifest(manifest_path, config)
    rows = [
        {"class_name": "a", "active_model_class": "a", "source_image_path": "/src/1.jpg", "sha256_bytes": "b1", "sha256_pixels": "p1", "status": "candidate_new_for_weak_class", "existing_plan_duplicate_classes": "", "decode_error": ""},
        {"class_name": "b", "active_model_class": "b", "source_image_path": "/src/2.jpg", "sha256_bytes": "b2", "sha256_pixels": "p1", "status": "candidate_new_for_weak_class", "existing_plan_duplicate_classes": "", "decode_error": ""},
        {"class_name": "a", "active_model_class": "a", "source_image_path": "/src/3.jpg", "sha256_bytes": "b3", "sha256_pixels": "p3", "status": "candidate_new_for_weak_class", "existing_plan_duplicate_classes": "", "decode_error": ""},
    ]
    if bad_candidate:
        rows[0].pop("sha256_pixels")
    candidate = write_candidate_csv(tmp_path / "candidates.csv", rows)
    v2_json = tmp_path / "v2.json"
    v2_csv = tmp_path / "v2.csv"
    write_v2(v2_json, v2_csv)
    output = tmp_path / "snapshot.json"
    return [
        "--sql-dump", str(sql),
        "--active-config", str(config),
        "--external-preview-manifest", str(manifest_path),
        "--candidate-csv", str(candidate),
        "--v2-summary-json", str(v2_json),
        "--v2-summary-csv", str(v2_csv),
        "--expected-counts", str(expected),
        "--secondary-dump", str(tmp_path / "secondary.sql"),
        "--output", str(output),
    ], output


def test_tuple_counter_handles_multiline_strings_comments_and_on_duplicate(tmp_path):
    sql = write_sql(tmp_path / "fixture.sql")
    counts, statements = snapshot.count_sql_rows(sql)
    assert counts == {"ai_codex": 2, "ai_element": 5, "ai_glyphe": 3, "ai_plate": 1, "ai_zone": 2}
    assert statements["ai_glyphe"] == 2


def test_happy_path_writes_ready_snapshot(tmp_path, monkeypatch):
    args, output = fixture_args(tmp_path, monkeypatch)
    assert snapshot.main(args) == 0
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["ready_for_next_phase"] is True
    assert data["blockers"] == []
    assert data["sql_row_counts"]["tables"]["ai_element"]["computed"] == 5
    assert data["inputs_identity"]["candidate_csv"]["internal_cross_class_pixel_conflict_groups"] == 1
    assert data["inputs_identity"]["v2_conflict_summary"]["classes_still_under_8_if_conflicts_excluded"] == ["a"]


def test_deterministic_output_to_two_paths(tmp_path, monkeypatch):
    args, output = fixture_args(tmp_path, monkeypatch)
    output2 = tmp_path / "snapshot2.json"
    assert snapshot.main(args) == 0
    args2 = list(args)
    args2[args2.index("--output") + 1] = str(output2)
    assert snapshot.main(args2) == 0
    assert output.read_bytes() == output2.read_bytes()


def test_row_count_mismatch_blocks_and_exits_nonzero(tmp_path, monkeypatch):
    args, output = fixture_args(tmp_path, monkeypatch)
    expected = Path(args[args.index("--expected-counts") + 1])
    expected.write_text(json.dumps({"ai_codex": 2, "ai_element": 99, "ai_glyphe": 3, "ai_plate": 1, "ai_zone": 2}), encoding="utf-8")
    assert snapshot.main(args) == 1
    data = json.loads(output.read_text(encoding="utf-8"))
    assert "sql_row_count_mismatch" in data["blockers"]
    assert data["ready_for_next_phase"] is False


def test_missing_external_manifest_writes_blocked_snapshot(tmp_path, monkeypatch):
    args, output = fixture_args(tmp_path, monkeypatch, manifest=False)
    assert snapshot.main(args) == 1
    data = json.loads(output.read_text(encoding="utf-8"))
    assert "missing_external_preview_manifest" in data["blockers"]
    assert data["inputs_identity"]["existing_preview_manifest"]["present"] is False


def test_candidate_missing_column_blocks(tmp_path, monkeypatch):
    args, output = fixture_args(tmp_path, monkeypatch)
    candidate = Path(args[args.index("--candidate-csv") + 1])
    candidate.write_text(
        "class_name,active_model_class,source_image_path,sha256_bytes,status,existing_plan_duplicate_classes,decode_error\n"
        "a,a,/src/1.jpg,b1,candidate_new_for_weak_class,,\n",
        encoding="utf-8",
    )
    assert snapshot.main(args) == 1
    data = json.loads(output.read_text(encoding="utf-8"))
    assert "candidate_csv_missing_columns" in data["blockers"]


def test_active_config_count_and_duplicates_block(tmp_path, monkeypatch):
    args, output = fixture_args(tmp_path, monkeypatch, bad_config=True)
    assert snapshot.main(args) == 1
    data = json.loads(output.read_text(encoding="utf-8"))
    assert "active_config_class_count_mismatch" in data["blockers"]


def test_class_outside_active_taxonomy_blocks(tmp_path, monkeypatch):
    args, output = fixture_args(tmp_path, monkeypatch)
    candidate = Path(args[args.index("--candidate-csv") + 1])
    rows = list(csv.DictReader(candidate.open(newline="", encoding="utf-8")))
    rows[0]["active_model_class"] = "outside"
    write_candidate_csv(candidate, rows)
    assert snapshot.main(args) == 1
    data = json.loads(output.read_text(encoding="utf-8"))
    assert "candidate_csv_classes_outside_active_config" in data["blockers"]


def test_overwrite_refused_without_ack_allowed_with_drift_history(tmp_path, monkeypatch):
    args, output = fixture_args(tmp_path, monkeypatch)
    assert snapshot.main(args) == 0
    before = output.read_bytes()
    assert snapshot.main(args) == 2
    assert output.read_bytes() == before
    assert snapshot.main(args + ["--ack-baseline-drift"]) == 0
    data = json.loads(output.read_text(encoding="utf-8"))
    assert len(data["drift_history"]) == 1
    assert data["drift_history"][0]["acknowledged"] is True


def test_v2_under_8_mismatch_blocks(tmp_path, monkeypatch):
    args, output = fixture_args(tmp_path, monkeypatch)
    v2_json = Path(args[args.index("--v2-summary-json") + 1])
    v2_csv = Path(args[args.index("--v2-summary-csv") + 1])
    write_v2(v2_json, v2_csv, under=[{"class_name": "b", "projected_unique_if_conflicts_excluded": 7}])
    assert snapshot.main(args) == 1
    data = json.loads(output.read_text(encoding="utf-8"))
    assert "v2_summary_under_8_classes_mismatch" in data["blockers"]


def test_external_manifest_config_mismatch_blocks(tmp_path, monkeypatch):
    args, output = fixture_args(tmp_path, monkeypatch)
    manifest = Path(args[args.index("--external-preview-manifest") + 1])
    manifest.write_text(json.dumps({"active_config_sha256": "wrong"}), encoding="utf-8")
    assert snapshot.main(args) == 1
    data = json.loads(output.read_text(encoding="utf-8"))
    assert "external_preview_manifest_config_mismatch" in data["blockers"]


def test_duplicate_active_classes_block(tmp_path, monkeypatch):
    args, output = fixture_args(tmp_path, monkeypatch)
    config = Path(args[args.index("--active-config") + 1])
    classes = ["a", "a"] + [f"filler_{i}" for i in range(284)]
    write_config(config, classes, num_classes=286)
    write_manifest(Path(args[args.index("--external-preview-manifest") + 1]), config)
    assert snapshot.main(args) == 1
    data = json.loads(output.read_text(encoding="utf-8"))
    assert "active_config_duplicate_classes" in data["blockers"]


def test_external_manifest_hash_is_raw_bytes(tmp_path, monkeypatch):
    args, output = fixture_args(tmp_path, monkeypatch)
    manifest = Path(args[args.index("--external-preview-manifest") + 1])
    raw_hash = snapshot.sha256_file(manifest)
    assert snapshot.main(args) == 0
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["inputs_identity"]["existing_preview_manifest"]["sha256"] == raw_hash
    assert data["inputs_identity"]["existing_preview_manifest"]["hash_method"] == "raw_bytes_not_canonical_json"


def test_secondary_dumps_recorded(tmp_path, monkeypatch):
    args, output = fixture_args(tmp_path, monkeypatch)
    secondary = tmp_path / "secondary.sql"
    secondary.write_text("-- secondary", encoding="utf-8")
    assert snapshot.main(args) == 0
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["inputs_identity"]["secondary_dumps"][0]["path"] == str(secondary)
    assert data["inputs_identity"]["secondary_dumps"][0]["non_primary"] is True
    assert data["inputs_identity"]["secondary_dumps"][0]["sha256"] == snapshot.sha256_file(secondary)


def test_no_writes_under_source_root(tmp_path, monkeypatch):
    args, output = fixture_args(tmp_path, monkeypatch)
    source_root = tmp_path / "source_root"
    source_root.mkdir()
    sql = source_root / "fixture.sql"
    write_sql(sql)
    args[args.index("--sql-dump") + 1] = str(sql)
    original_open = Path.open

    def guarded_open(self, mode="r", *open_args, **open_kwargs):
        if str(self).startswith(str(source_root)) and any(flag in mode for flag in ("w", "a", "x", "+")):
            raise AssertionError(f"write attempted under source root: {self} {mode}")
        return original_open(self, mode, *open_args, **open_kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    assert snapshot.main(args) == 0
    assert output.exists()


def test_output_under_protected_root_is_rejected(tmp_path, monkeypatch):
    args, _output = fixture_args(tmp_path, monkeypatch)
    protected = tmp_path / "protected"
    protected.mkdir()
    forbidden = protected / "snapshot.json"
    args[args.index("--output") + 1] = str(forbidden)
    args.extend(["--protected-root", str(protected)])
    assert snapshot.main(args) == 2
    assert not forbidden.exists()


def test_default_source_root_remains_protected_with_custom_root(tmp_path, monkeypatch):
    args, _output = fixture_args(tmp_path, monkeypatch)
    protected = tmp_path / "also_protected"
    protected.mkdir()
    forbidden = Path("/mnt/f/CODEX/AI_clinic_class/p0.json")
    args[args.index("--output") + 1] = str(forbidden)
    args.extend(["--protected-root", str(protected)])
    assert snapshot.main(args) == 2


def test_v2_malformed_projection_blocks_without_crashing(tmp_path, monkeypatch):
    args, output = fixture_args(tmp_path, monkeypatch)
    v2_json = Path(args[args.index("--v2-summary-json") + 1])
    v2_json.write_text(
        json.dumps(
            {
                "candidate_rows": 3,
                "internal_cross_class_pixel_conflict_groups": 1,
                "classes_affected_by_internal_conflicts": 2,
                "classes_still_under_8_if_internal_conflicts_excluded": [{"class_name": "a"}],
            }
        ),
        encoding="utf-8",
    )
    assert snapshot.main(args) == 1
    data = json.loads(output.read_text(encoding="utf-8"))
    assert "invalid_v2_conflict_summary_schema" in data["blockers"]
    assert data["checks"]["v2_conflict_summary"] == "fail"

import importlib.util
import json
import sys
from pathlib import Path

from PIL import Image

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "audit_external_corpus.py"
spec = importlib.util.spec_from_file_location("audit_external_corpus", SCRIPT_PATH)
audit = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = audit
assert spec.loader is not None
spec.loader.exec_module(audit)


def write_image(path: Path, payload: bytes = b"fake-bmp") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def write_real_image(path: Path, color=(12, 34, 56)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (3, 2), color=color).save(path)


def write_config(path: Path, classes: list[str]) -> Path:
    path.write_text(json.dumps({"class_names": classes}), encoding="utf-8")
    return path


def test_normalize_class_name_strips_prefix_suffix_and_accents():
    assert audit.normalize_class_name("0015-cacahuatl") == "cacahuatl"
    assert audit.normalize_class_name("atl-element") == "atl"
    assert audit.normalize_class_name("0015-Atl-Element") == "atl"
    assert audit.normalize_class_name("adorno de nuca") == "adorno_de_nuca"
    assert audit.normalize_class_name("cáliz") == "caliz"


def test_audit_maps_sources_deduplicates_and_fails_partial_active_coverage(tmp_path):
    config = write_config(tmp_path / "config.json", ["atl", "tetl", "calli"])
    src = tmp_path / "Main_Elements"
    write_image(src / "atl-element" / "001_001_001-1.bmp", b"same")
    write_image(src / "atl-element" / "001_001_001-2.bmp", b"same")
    write_image(src / "tetl-element" / "001_001_001-3.bmp", b"unique")
    write_image(src / "unknown-element" / "001_001_001-4.bmp", b"unknown")

    report, inventory = audit.build_report(
        config,
        [audit.SourceSpec(name="fixture", path=src)],
        hash_files=True,
        pixel_hash_files=True,
        k_shot=3,
        q_queries=5,
        min_images_per_class=2,
        val_fraction=0.15,
        n_way=20,
        require_full_active_coverage=True,
    )

    assert report.aggregate_counts_by_active_class == {"atl": 2, "tetl": 1}
    assert report.inventory_count == 4
    assert len(inventory) == 4
    assert report.sources[0].matched_class_count == 2
    assert report.sources[0].unmatched_class_count == 1
    assert report.duplicate_groups
    assert report.duplicate_groups[0]["count"] == len(report.duplicate_groups[0]["entries"])
    assert report.preflight.status == "fail"
    assert "active_taxonomy_not_fully_covered:2/3" in report.preflight.blocking_reasons
    assert any(reason.startswith("insufficient_episode_classes") for reason in report.preflight.blocking_reasons)


def test_preflight_can_pass_for_complete_feasible_fixture(tmp_path):
    config = write_config(tmp_path / "config.json", ["atl", "tetl"])
    src = tmp_path / "Elements"
    for class_name in ["0015-atl", "0016-tetl"]:
        for index in range(8):
            write_image(src / class_name / f"001_001_001-{index}.bmp", f"{class_name}-{index}".encode())

    report, _inventory = audit.build_report(
        config,
        [audit.SourceSpec(name="fixture", path=src)],
        hash_files=True,
        pixel_hash_files=True,
        k_shot=3,
        q_queries=5,
        min_images_per_class=2,
        val_fraction=0.15,
        n_way=2,
        require_full_active_coverage=True,
    )

    assert report.preflight.status == "pass"
    assert report.preflight.promotable_against_active is True
    assert report.preflight.covered_active_class_count == 2
    assert not report.duplicate_groups


def test_inventory_output_and_pixel_hash_detect_reencoded_duplicates(tmp_path):
    config = write_config(tmp_path / "config.json", ["atl"])
    src = tmp_path / "src"
    write_real_image(src / "0015-atl" / "same.bmp")
    write_real_image(src / "0015-atl" / "same.png")

    report, inventory = audit.build_report(
        config,
        [audit.SourceSpec(name="fixture", path=src)],
        hash_files=True,
        pixel_hash_files=True,
        k_shot=1,
        q_queries=1,
        min_images_per_class=1,
        val_fraction=0.5,
        n_way=1,
        require_full_active_coverage=True,
    )

    assert len(inventory) == 2
    assert inventory[0].sha256_bytes != inventory[1].sha256_bytes
    assert inventory[0].sha256_pixels == inventory[1].sha256_pixels
    assert not report.duplicate_groups
    assert report.pixel_duplicate_groups
    assert report.pixel_duplicate_groups[0]["count"] == 2
    payload = audit.inventory_payload(inventory, report)
    assert payload["image_count"] == 2
    assert len(payload["images"]) == 2


def test_duplicate_groups_are_not_truncated_above_twenty_entries(tmp_path):
    config = write_config(tmp_path / "config.json", ["atl"])
    src = tmp_path / "src"
    payload = b"same-bytes"
    for index in range(25):
        write_image(src / "0015-atl" / f"{index}.bmp", payload)

    report, inventory = audit.build_report(
        config,
        [audit.SourceSpec(name="fixture", path=src)],
        hash_files=True,
        pixel_hash_files=False,
        k_shot=1,
        q_queries=1,
        min_images_per_class=1,
        val_fraction=0.15,
        n_way=1,
        require_full_active_coverage=True,
    )

    assert len(inventory) == 25
    assert report.duplicate_groups[0]["count"] == 25
    assert len(report.duplicate_groups[0]["entries"]) == 25

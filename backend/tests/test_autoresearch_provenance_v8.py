from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "autoresearch_provenance_v8.py"
SPEC = importlib.util.spec_from_file_location("autoresearch_provenance_v8", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _image(path: Path, color: tuple[int, int, int]) -> None:
    Image.new("RGB", (4, 3), color).save(path)


def _write_input(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")


def test_normalize_source_family_unifies_page_and_variants() -> None:
    assert MODULE.normalize_source_family("page:04:04:01") == "04_04_01"
    assert MODULE.normalize_source_family("04_04_01-1_a.jpg") == "04_04_01"
    assert MODULE.normalize_source_family("04_04_01-27_AH.bmp") == "04_04_01"


def test_components_union_family_and_pixels_and_report_pixel_conflict(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.bmp"
    second = tmp_path / "second.png"
    third = tmp_path / "third.bmp"
    _image(first, (10, 20, 30))
    _image(second, (10, 20, 30))
    _image(third, (200, 100, 50))
    input_path = tmp_path / "input.jsonl"
    _write_input(
        input_path,
        [
            {
                "row_id": "a",
                "image_path": str(first),
                "source_path": "03_04_22-1.jpg",
                "cache_name": "legacy",
                "class_label": 1,
                "class_name": "one",
            },
            {
                "row_id": "b",
                "image_path": str(second),
                "source_path": "99_99_99-1.jpg",
                "cache_name": "external",
                "class_label": 2,
                "class_name": "two",
            },
            {
                "row_id": "c",
                "image_path": str(third),
                "source_path": "03_04_22-9_b.jpg",
                "cache_name": "external",
                "class_label": 1,
                "class_name": "one",
            },
        ],
    )
    rows, errors = MODULE.load_and_hash_rows(input_path)
    assert errors == []
    rows, components, conflicts = MODULE.build_components(rows)
    assert len(components) == 1
    assert {row["component_id"] for row in rows} == {components[0]["component_id"]}
    assert len(conflicts) == 1
    assert [label["class_label"] for label in conflicts[0]["labels"]] == [1, 2]


def test_run_is_deterministic_and_has_no_cross_fold_component_or_pixel_overlap(
    tmp_path: Path,
) -> None:
    rows: list[dict[str, object]] = []
    for index in range(12):
        image_path = tmp_path / f"{index}.bmp"
        _image(image_path, (index, index * 2, index * 3))
        rows.append(
            {
                "row_id": f"row-{index:02d}",
                "image_path": str(image_path),
                "source_path": f"0{index // 4}_01_01-{index}_a.jpg",
                "cache_name": "synthetic",
                "class_label": index % 2,
                "class_name": f"class-{index % 2}",
            }
        )
    input_path = tmp_path / "input.jsonl"
    _write_input(input_path, rows)
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first = MODULE.run(input_path, first_dir, fold_count=3, minimum_oof_rows=1)
    second = MODULE.run(input_path, second_dir, fold_count=3, minimum_oof_rows=1)

    assert first["pass"] is True
    assert second["pass"] is True
    assert first["artifacts"] == second["artifacts"]
    assert (
        first["gates"]["known_provenance_component_overlap_across_folds"] is True
    )
    assert first["gates"]["decoded_pixel_hash_overlap_across_folds"] is True

    manifest = [
        json.loads(line)
        for line in (first_dir / "provenance-manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    component_folds: dict[str, set[int]] = {}
    pixel_folds: dict[str, set[int]] = {}
    for row in manifest:
        component_folds.setdefault(row["component_id"], set()).add(row["fold"])
        pixel_folds.setdefault(row["decoded_pixel_sha256"], set()).add(row["fold"])
    assert all(len(folds) == 1 for folds in component_folds.values())
    assert all(len(folds) == 1 for folds in pixel_folds.values())


def test_unreadable_image_fails_gate_but_is_reported(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    _write_input(
        input_path,
        [
            {
                "row_id": "missing",
                "image_path": str(tmp_path / "missing.bmp"),
                "source_path": "01_01_01-1.jpg",
                "cache_name": "synthetic",
                "class_label": 0,
                "class_name": "zero",
            }
        ],
    )
    result = MODULE.run(
        input_path,
        tmp_path / "output",
        fold_count=2,
        minimum_oof_rows=0,
    )
    assert result["pass"] is False
    assert result["gates"]["all_images_decoded"] is False
    assert result["image_decode_errors"][0]["row_id"] == "missing"

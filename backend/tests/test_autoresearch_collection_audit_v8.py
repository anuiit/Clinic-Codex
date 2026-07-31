from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "autoresearch_collection_audit_v8.py"
SPEC = importlib.util.spec_from_file_location("autoresearch_collection_audit_v8", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _row(
    row_id: str,
    family: str,
    pixel_hash: str,
    class_name: str = "one",
    fold: int = 1,
    source_path: str | None = None,
) -> dict[str, object]:
    return {
        "row_id": row_id,
        "cache_name": "external",
        "class_label": 0,
        "class_name": class_name,
        "component_id": "old-component",
        "decoded_pixel_sha256": pixel_hash,
        "fold": fold,
        "image_path": f"/images/{row_id}.bmp",
        "source_family": family,
        "source_group": "",
        "source_path": source_path or f"/source/{row_id}.jpg",
    }


def _audit(conflict_hashes: list[str] | None = None) -> dict[str, object]:
    return {
        "pixel_label_conflicts": [
            {"decoded_pixel_sha256": value, "labels": []}
            for value in (conflict_hashes or [])
        ]
    }


def test_nested_numeric_suffix_sensitivity_collapses_repeated_instances() -> None:
    assert MODULE.collapse_nested_numeric_suffixes("03_02_05-1-01") == "03_02_05"
    assert MODULE.collapse_nested_numeric_suffixes("03_02_05") == "03_02_05"


def test_directory_origins_are_not_positive_identity() -> None:
    rows = [
        _row("a", "page", "hash-a", source_path="/a/page-1.jpg"),
        _row("b", "page", "hash-b", source_path="/b/page-2.jpg"),
    ]
    approved = {
        "actions": [
            {"path": "/a/page-1.jpg", "source": "root-a"},
            {"path": "/b/page-2.jpg", "source": "root-b"},
        ]
    }
    report, _ = MODULE.audit(rows, {"components": []}, _audit(), [approved])
    assert report["identity_evidence"]["coverage"] == 0.0
    assert (
        report["identity_evidence"]["positively_supported_component_split_count"]
        == 0
    )
    assert report["conservative_after_quarantine"]["component_count"] == 1
    assert report["technical_origin_separated_upper_bound"]["component_count"] == 2
    assert (
        report["technical_origin_separated_upper_bound"]["scientifically_valid"]
        is False
    )


def test_explicit_collection_ids_can_support_a_split() -> None:
    rows = [
        _row("a", "page", "hash-a", source_path="/a/page-1.jpg"),
        _row("b", "page", "hash-b", source_path="/b/page-2.jpg"),
    ]
    approved = {
        "actions": [
            {
                "path": "/a/page-1.jpg",
                "source": "root-a",
                "collection_id": "collection-a",
            },
            {
                "path": "/b/page-2.jpg",
                "source": "root-b",
                "collection_id": "collection-b",
            },
        ]
    }
    report, _ = MODULE.audit(rows, {"components": []}, _audit(), [approved])
    assert report["identity_evidence"]["coverage"] == 1.0
    assert (
        report["identity_evidence"]["positively_supported_component_split_count"]
        == 1
    )
    assert report["positive_identity_revised"]["component_count"] == 2


def test_conflicting_pixels_are_quarantined_before_graph_rebuild() -> None:
    rows = [
        _row("a", "family-a", "bridge", class_name="one", fold=1),
        _row("b", "family-b", "bridge", class_name="two", fold=1),
        _row("c", "family-a", "clean-a", class_name="one", fold=2),
        _row("d", "family-b", "clean-b", class_name="two", fold=2),
    ]
    report, _ = MODULE.audit(
        rows,
        {"components": [{"component_id": "old-component"}]},
        _audit(["bridge"]),
        [],
    )
    assert report["quarantine"]["quarantined_row_count"] == 2
    assert report["conservative_after_quarantine"]["component_count"] == 2
    assert report["source_rows_modified"] == 0
    assert report["final_test_read"] is False


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_run_is_byte_deterministic_and_rejects_nonempty_output(
    tmp_path: Path,
) -> None:
    rows = [
        _row("a", "family-a", "hash-a", fold=1),
        _row("b", "family-b", "hash-b", fold=2),
    ]
    manifest = tmp_path / "manifest.jsonl"
    components = tmp_path / "components.json"
    provenance_audit = tmp_path / "audit.json"
    legacy = tmp_path / "legacy.json"
    external = tmp_path / "external.json"
    approved = tmp_path / "approved.json"
    _write_jsonl(manifest, rows)
    _write_json(components, {"components": []})
    _write_json(provenance_audit, _audit())
    _write_json(legacy, {"images": []})
    _write_json(external, {"rows": []})
    _write_json(approved, {"actions": []})

    first = tmp_path / "first"
    second = tmp_path / "second"
    MODULE.run(manifest, components, provenance_audit, legacy, external, approved, first)
    MODULE.run(manifest, components, provenance_audit, legacy, external, approved, second)
    assert (first / "collection-provenance-audit.json").read_bytes() == (
        second / "collection-provenance-audit.json"
    ).read_bytes()
    assert (first / "acquisition-targets.json").read_bytes() == (
        second / "acquisition-targets.json"
    ).read_bytes()

    with pytest.raises(ValueError, match="not empty"):
        MODULE.run(
            manifest,
            components,
            provenance_audit,
            legacy,
            external,
            approved,
            first,
        )

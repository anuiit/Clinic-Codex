from __future__ import annotations

import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "autoresearch_reservoir_contract_v8.py"
SPEC = importlib.util.spec_from_file_location(
    "autoresearch_reservoir_contract_v8",
    SCRIPT,
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _pattern(path: Path, kind: str) -> None:
    image = Image.new("RGB", (32, 24))
    pixels = image.load()
    for y in range(image.height):
        for x in range(image.width):
            if kind == "forward":
                value = x * 7
            elif kind == "reverse":
                value = 255 - x * 7
            elif kind == "alternating":
                value = 240 if x % 2 else 15
            else:
                value = (x * 31 + y * 17) % 256
            pixels[x, y] = (value, (value + y * 3) % 256, 255 - value)
    image.save(path)


def _inputs(tmp_path: Path) -> dict[str, Path]:
    imported = tmp_path / "imported.png"
    exact = tmp_path / "a-exact.png"
    inadmissible = tmp_path / "b-inadmissible.png"
    near = tmp_path / "c-near.png"
    positive = tmp_path / "d-positive.png"
    unknown = tmp_path / "e-unknown.png"

    _pattern(imported, "forward")
    _pattern(exact, "forward")
    _pattern(inadmissible, "forward")
    _pattern(near, "forward")
    with Image.open(near) as image:
        edited = image.copy()
    edited.putpixel((0, 0), (1, 2, 3))
    edited.save(near)
    _pattern(positive, "reverse")
    _pattern(unknown, "alternating")

    import_manifest = tmp_path / "approved.json"
    actions: list[dict[str, object]] = [
        {
            "action": "exclude",
            "approved_class": "atl",
            "original_status": "duplicate_same_class",
            "path": str(exact),
            "reason": "duplicate_same_class",
            "source": "root-a",
            "source_folder": "atl",
        },
        {
            "action": "quarantine",
            "approved_class": "",
            "original_status": "conflict_cross_class",
            "path": str(inadmissible),
            "reason": "reviewer_quarantine_conflict",
            "source": "root-b",
            "source_folder": "conflict",
        },
        {
            "action": "exclude",
            "approved_class": "petlatl",
            "original_status": "suspected_reencode",
            "path": str(near),
            "reason": "suspected_reencode",
            "source": "root-c",
            "source_folder": "petlatl",
        },
        {
            "action": "exclude",
            "approved_class": "huitzilin",
            "original_status": "candidate",
            "path": str(positive),
            "reason": "not_yet_cataloged",
            "source": "root-d",
            "source_folder": "huitzilin",
            "collection_id": "collection-new",
            "manuscript_id": "manuscript-new",
            "namespaced_page_id": "collection-new:folio-1r",
            "crop_instance_id": "crop-1",
            "scan_batch_id": "scan-new",
            "parent_asset_id": "asset-new",
            "derivation_relation": "independent_acquisition",
            "evidence_reference": "catalog://collection-new/manuscript-new",
        },
        {
            "action": "exclude",
            "approved_class": "piqui",
            "original_status": "candidate",
            "path": str(unknown),
            "reason": "different_root_and_hash_only",
            "source": "totally-different-root",
            "source_folder": "different-stem",
        },
    ]
    _write_json(import_manifest, {"actions": actions})

    imported_manifest = tmp_path / "imported.jsonl"
    _write_jsonl(
        imported_manifest,
        [
            {
                "row_id": "imported:1",
                "class_name": "atl",
                "source_path": str(imported),
            }
        ],
    )

    external_audit = tmp_path / "external-audit.json"
    _write_json(
        external_audit,
        {
            "schema_version": "external-corpus-audit.test",
            "inventory_count": 6,
            "pixel_duplicate_groups": [],
        },
    )

    acquisition_targets = tmp_path / "acquisition-targets.json"
    _write_json(
        acquisition_targets,
        {
            "aggregate_additional_component_slots_needed": {
                "2": 2,
                "3": 4,
                "5": 8,
            },
            "targets": [
                {
                    "class_name": "huitzilin",
                    "current_independent_components": 1,
                },
                {
                    "class_name": "piqui",
                    "current_independent_components": 1,
                },
            ],
        },
    )
    return {
        "import_manifest": import_manifest,
        "external_audit": external_audit,
        "imported_manifest": imported_manifest,
        "acquisition_targets": acquisition_targets,
    }


def _run(tmp_path: Path, output_name: str = "out") -> tuple[dict, list[dict]]:
    inputs = _inputs(tmp_path)
    output_dir = tmp_path / output_name
    audit = MODULE.run(
        **inputs,
        output_dir=output_dir,
        expected_candidates=5,
        near_threshold=4,
        hash_workers=1,
    )
    catalog = [
        json.loads(line)
        for line in (output_dir / "reservoir-catalog.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    return audit, catalog


def test_decision_lattice_is_conservative_and_deterministic(tmp_path: Path) -> None:
    audit, catalog = _run(tmp_path, "first")
    by_class = {row["approved_class"]: row for row in catalog if row["approved_class"]}

    assert by_class["atl"]["disposition"] == "duplicate"
    assert by_class["petlatl"]["disposition"] == "near_duplicate"
    assert by_class["huitzilin"]["disposition"] == "proven_independent"
    assert by_class["piqui"]["disposition"] == "unknown"
    assert all(
        not row["forbidden_independence_evidence_used"] for row in catalog
    )
    assert audit["disposition_counts"]["inadmissible"] == 1
    assert audit["positively_proven_new_component_slots"] == 1
    assert audit["v7_regression_component_slots"] == 1
    assert audit["pass_without_replay"] is False
    assert audit["reservoir_stop"] is True
    assert audit["model_go_triggered"] is False

    inputs = {
        "import_manifest": tmp_path / "approved.json",
        "external_audit": tmp_path / "external-audit.json",
        "imported_manifest": tmp_path / "imported.jsonl",
        "acquisition_targets": tmp_path / "acquisition-targets.json",
    }
    MODULE.run(
        **inputs,
        output_dir=tmp_path / "replay",
        expected_candidates=5,
        near_threshold=4,
        hash_workers=1,
    )
    for artifact in (
        "provenance-admission-contract.json",
        "reservoir-catalog.jsonl",
        "reservoir-audit.json",
    ):
        assert (tmp_path / "first" / artifact).read_bytes() == (
            tmp_path / "replay" / artifact
        ).read_bytes()


def test_technical_proxies_alone_never_prove_independence(tmp_path: Path) -> None:
    _, catalog = _run(tmp_path)
    row = next(item for item in catalog if item["approved_class"] == "piqui")
    assert row["source"] == "totally-different-root"
    assert row["manifest_reason"] == "different_root_and_hash_only"
    assert row["disposition"] == "unknown"
    assert row["identity_complete"] is False
    assert row["provenance_component_id"] is None


def test_positive_identity_requires_the_full_documentary_chain() -> None:
    row = {
        "collection_id": "collection",
        "manuscript_id": "manuscript",
        "namespaced_page_id": "page",
        "crop_instance_id": "crop",
        "scan_batch_id": "batch",
        "parent_asset_id": "parent",
        "derivation_relation": "independent_acquisition",
    }
    _, complete, missing = MODULE.extract_identity(row)
    assert complete is False
    assert missing == ["evidence_reference"]

    row["evidence_reference"] = "catalog://record"
    _, complete, missing = MODULE.extract_identity(row)
    assert complete is True
    assert missing == []


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[2] / ".omx/reviews/data-discovery/import-plan-20260707/import_manifest.approved.json").exists(),
    reason="requires unshipped research artifact: import-plan-20260707/import_manifest.approved.json",
)
def test_real_manifest_has_the_preregistered_candidate_population() -> None:
    manifest = (
        ROOT
        / ".omx"
        / "reviews"
        / "data-discovery"
        / "import-plan-20260707"
        / "import_manifest.approved.json"
    )
    candidates = MODULE.load_candidates(manifest)
    assert len(candidates) == 897
    assert Counter(row["reason"] for row in candidates) == {
        "suspected_reencode": 764,
        "reviewer_quarantine_unmapped": 59,
        "reviewer_quarantine_conflict": 45,
        "duplicate_same_class": 29,
    }
    assert Counter(row["source"] for row in candidates) == {
        "AI_clinic_class__Main_Elements": 792,
        "AI_clinic_class__MainElem_Original": 25,
        "Clinic-Codex__data__Elements": 80,
    }


def test_nonempty_output_directory_is_refused(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    output_dir = tmp_path / "occupied"
    output_dir.mkdir()
    (output_dir / "keep.txt").write_text("do not overwrite", encoding="utf-8")
    with pytest.raises(FileExistsError):
        MODULE.run(
            **inputs,
            output_dir=output_dir,
            expected_candidates=5,
            hash_workers=1,
        )
    assert (output_dir / "keep.txt").read_text(encoding="utf-8") == "do not overwrite"

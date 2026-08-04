from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/autoresearch_member_attribution_v14.py"
spec = importlib.util.spec_from_file_location(
    "autoresearch_member_attribution_v14", SCRIPT
)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def _source_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "row_id": "external:1",
        "attribution_status": "unique_archive_without_unique_codex",
        "class_label": 1,
        "class_name": "atl",
        "component_id": "component:test",
        "development_byte_sha256": "byte-hash",
        "decoded_pixel_sha256": "rgb-hash",
        "unique_archive": "t_sample.zip",
    }
    row.update(overrides)
    return row


def _member(
    archive: str, index: int, codex_ids: list[int], name: str | None = None
) -> dict[str, object]:
    return {
        "archive_name": archive,
        "member_index": index,
        "member_name": name or f"A.{index:02d}.bmp",
        "normalized_stem": f"a.{index:02d}",
        "exact_codex_ids": codex_ids,
        "exact_codex_titles": [f"CODEX {codex_id}" for codex_id in codex_ids],
    }


def test_v14_contract_is_hash_pinned_descriptive_and_model_free() -> None:
    contract = module.validate_contract()
    assert contract["spec_sha256"] == module.EXPECTED_SPEC_SHA256
    assert contract["evaluator_sha256"] == module.EXPECTED_EVALUATOR_SHA256
    assert contract["spec"]["hypothesis_supported"] is None
    assert contract["evaluator"]["numeric_success_threshold"] is None
    assert contract["spec"]["model_inference_allowed"] is False
    assert contract["spec"]["training_allowed"] is False
    assert contract["evaluator"]["promotion_eligible"] is False


def test_member_descriptor_uses_exact_casefolded_stem() -> None:
    descriptor = module.member_descriptor(
        {
            "archive_name": "t_sample.zip",
            "member_index": 3,
            "member_name": "nested/Á.01.BMP",
        },
        cote_to_codex={"á.01": {7}},
        codex_titles={7: "SEVEN"},
    )
    assert descriptor["normalized_stem"] == "á.01"
    assert descriptor["exact_codex_ids"] == [7]
    assert descriptor["exact_codex_titles"] == ["SEVEN"]


def test_resolve_row_accepts_only_full_unanimity() -> None:
    row = _source_row()
    first = _member("t_one.zip", 1, [7])
    second = _member("t_two.zip", 2, [7])
    result = module.resolve_row(
        row,
        byte_lookup={"byte-hash": [first]},
        rgb_lookup={"rgb-hash": [first, second]},
        codex_titles={7: "SEVEN"},
    )
    assert result["resolved"] is True
    assert result["unique_codex_id"] == 7
    assert result["exact_colliding_member_count"] == 2
    assert result["unresolved_reasons"] == []


def test_resolve_row_rejects_unmapped_member_despite_unique_mapped_vote() -> None:
    row = _source_row()
    result = module.resolve_row(
        row,
        byte_lookup={"byte-hash": [_member("t_one.zip", 1, [7])]},
        rgb_lookup={"rgb-hash": [_member("t_two.zip", 2, [])]},
        codex_titles={7: "SEVEN"},
    )
    assert result["resolved"] is False
    assert result["unique_codex_id"] is None
    assert result["unresolved_reasons"] == ["unmapped_member"]


def test_resolve_row_rejects_divergent_byte_and_rgb_codices() -> None:
    row = _source_row()
    result = module.resolve_row(
        row,
        byte_lookup={"byte-hash": [_member("t_one.zip", 1, [7])]},
        rgb_lookup={"rgb-hash": [_member("t_two.zip", 2, [8])]},
        codex_titles={7: "SEVEN", 8: "EIGHT"},
    )
    assert result["resolved"] is False
    assert result["unresolved_reasons"] == [
        "byte_rgb_codex_conflict",
        "divergent_codex_ids",
    ]


def test_population_contract_freezes_all_487_rows_and_subset_432() -> None:
    rows: list[dict[str, object]] = []
    for index in range(37):
        rows.append(
            _source_row(
                row_id=f"ambiguous:{index}",
                attribution_status="ambiguous_archive",
                unique_archive=None,
            )
        )
    for index in range(450):
        rows.append(
            _source_row(
                row_id=f"archive-only:{index}",
                unique_archive=(
                    "t_mappable.zip" if index < 432 else "t_tepeuc.zip"
                ),
            )
        )
    rows.extend(
        _source_row(
            row_id=f"unique:{index}",
            attribution_status="unique_codex",
            unique_codex_id=1,
        )
        for index in range(4680)
    )
    rows.extend(
        _source_row(
            row_id=f"unmatched:{index}",
            attribution_status="unmatched",
            unique_archive=None,
        )
        for index in range(4823)
    )
    population, counts = module.select_population(rows)
    assert len(rows) == 9990
    assert len(population) == 487
    assert counts == {
        "ambiguous_archive": 37,
        "unique_archive_without_unique_codex": 450,
        "population_rows": 487,
        "subset_432_rows": 432,
    }


def test_write_outputs_is_deterministic(tmp_path: Path) -> None:
    resolutions = [{"row_id": "a", "resolved": False}]
    merged = [{"row_id": "a", "attribution_status": "ambiguous_archive"}]
    audit = {"schema_version": "test", "hypothesis_supported": None}
    first = module.write_outputs(
        tmp_path / "first",
        resolutions=resolutions,
        merged=merged,
        audit=audit,
    )
    replay = module.write_outputs(
        tmp_path / "replay",
        resolutions=resolutions,
        merged=merged,
        audit=audit,
    )
    assert first["member_attribution_sha256"] == replay[
        "member_attribution_sha256"
    ]
    assert first["development_provenance_sha256"] == replay[
        "development_provenance_sha256"
    ]
    assert first["audit_sha256"] == replay["audit_sha256"]

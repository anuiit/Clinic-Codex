from __future__ import annotations

import importlib.util
import io
import json
import sys
import zipfile
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/autoresearch_archive_attribution_v13.py"
spec = importlib.util.spec_from_file_location(
    "autoresearch_archive_attribution_v13", SCRIPT
)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def _bmp_bytes(color: tuple[int, int, int] = (20, 40, 60)) -> bytes:
    image = Image.new("RGB", (3, 2), color)
    output = io.BytesIO()
    image.save(output, format="BMP")
    return output.getvalue()


def _archive(path: Path, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, value in members.items():
            archive.writestr(name, value)
    return path


def test_v13_contract_is_hash_pinned_and_model_free() -> None:
    contract = module.validate_contract()
    assert contract["spec_sha256"] == module.EXPECTED_SPEC_SHA256
    assert contract["evaluator_sha256"] == module.EXPECTED_EVALUATOR_SHA256
    assert contract["spec"]["model_inference_allowed"] is False
    assert contract["spec"]["automatic_promotion_allowed"] is False
    assert contract["evaluator"]["exclusion_change_allowed"] is False


def test_rgb_hash_matches_v8_width_height_pixel_contract() -> None:
    value = _bmp_bytes()
    digest, width, height = module.decoded_rgb_sha256_bytes(value)
    expected = __import__("hashlib").sha256()
    expected.update((3).to_bytes(8, "big"))
    expected.update((2).to_bytes(8, "big"))
    expected.update(bytes((20, 40, 60)) * 6)
    assert (width, height) == (3, 2)
    assert digest == expected.hexdigest()


def test_archive_index_is_complete_and_reusable(tmp_path: Path) -> None:
    archive_path = _archive(
        tmp_path / "t_sample.zip",
        {"A.01.bmp": _bmp_bytes(), "metadata.txt": b"evidence"},
    )
    archive_hash = module.sha256_file(archive_path)
    first = module.index_archive(
        archive_path,
        archive_sha256=archive_hash,
        index_dir=tmp_path / "index",
        supported_extensions={".bmp", ".jpg"},
        max_member_bytes=1024 * 1024,
        force=False,
    )
    second = module.index_archive(
        archive_path,
        archive_sha256=archive_hash,
        index_dir=tmp_path / "index",
        supported_extensions={".bmp", ".jpg"},
        max_member_bytes=1024 * 1024,
        force=False,
    )
    assert first["non_directory_entries"] == 2
    assert first["supported_raster_entries"] == 1
    assert first["raster_decode_successes"] == 1
    assert first["non_raster_entries"] == 1
    assert second["reused"] is True
    assert second["index_sha256"] == first["index_sha256"]


def test_exact_cote_and_rgb_collision_yield_unique_codex(tmp_path: Path) -> None:
    value = _bmp_bytes()
    archive_path = _archive(tmp_path / "t_sample.zip", {"A.01.bmp": value})
    summary = module.index_archive(
        archive_path,
        archive_sha256=module.sha256_file(archive_path),
        index_dir=tmp_path / "index",
        supported_extensions={".bmp"},
        max_member_bytes=1024 * 1024,
        force=False,
    )
    codex_csv = tmp_path / "codex.csv"
    codex_csv.write_text(
        "id,titre,glyphes,personnages,elements\n"
        + "\n".join(f"{index},CODEX {index},0,0,0" for index in range(1, 51))
        + "\n",
        encoding="utf-8",
    )
    elements_csv = tmp_path / "elements.csv"
    elements_csv.write_text(
        "id,codexid,cote,element,element_id,theme\n"
        "1,7,A.01,atl,1,01.01.01\n",
        encoding="utf-8",
    )
    titles, cotes = module.load_codex_catalog(codex_csv, elements_csv)
    archive_map = module.build_archive_to_codex_map(
        [summary],
        index_dir=tmp_path / "index",
        codex_titles=titles,
        cote_to_codex=cotes,
    )
    byte_lookup, rgb_lookup = module.build_collision_lookups(
        [summary], index_dir=tmp_path / "index"
    )
    artifact = tmp_path / "artifact.bmp"
    artifact.write_bytes(value)
    pixel_hash, _, _ = module.decoded_rgb_sha256_bytes(value)
    rows = [
        {
            "row_id": "legacy:1",
            "cache_name": "legacy",
            "class_label": 0,
            "class_name": "atl",
            "component_id": "component:test",
            "image_path": str(artifact),
            "decoded_pixel_sha256": pixel_hash,
        }
    ]
    output, result = module.build_attribution(
        rows,
        development_byte_hashes={str(artifact): module.sha256_file(artifact)},
        byte_to_archives=byte_lookup,
        rgb_to_archives=rgb_lookup,
        archive_map=archive_map,
    )
    assert output[0]["attribution_status"] == "unique_codex"
    assert output[0]["unique_codex_id"] == 7
    assert result["unique_codex_attributed_row_coverage"] == 1.0


def test_same_pixels_in_two_archives_are_ambiguous(tmp_path: Path) -> None:
    value = _bmp_bytes()
    pixel_hash, _, _ = module.decoded_rgb_sha256_bytes(value)
    artifact = tmp_path / "artifact.bmp"
    artifact.write_bytes(value)
    archive_map = {
        "archives": [
            {
                "archive_name": "t_one.zip",
                "unique_codex_id": 1,
                "unique_codex_title": "ONE",
            },
            {
                "archive_name": "t_two.zip",
                "unique_codex_id": 2,
                "unique_codex_title": "TWO",
            },
        ]
    }
    rows = [
        {
            "row_id": "legacy:1",
            "cache_name": "legacy",
            "class_label": 0,
            "class_name": "atl",
            "component_id": "component:test",
            "image_path": str(artifact),
            "decoded_pixel_sha256": pixel_hash,
        }
    ]
    output, result = module.build_attribution(
        rows,
        development_byte_hashes={str(artifact): module.sha256_file(artifact)},
        byte_to_archives={},
        rgb_to_archives={pixel_hash: {"t_one.zip", "t_two.zip"}},
        archive_map=archive_map,
    )
    assert output[0]["attribution_status"] == "ambiguous_archive"
    assert result["ambiguous_rows"] == 1
    assert result["unique_codex_attributed_rows"] == 0

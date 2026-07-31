from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import zipfile
from pathlib import Path

from PIL import Image
import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "freeze_legacy_elements.py"
SPEC = importlib.util.spec_from_file_location("freeze_legacy_elements", SCRIPT_PATH)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def _bmp_bytes(color: tuple[int, int, int], size: tuple[int, int] = (4, 4)) -> bytes:
    from io import BytesIO

    buf = BytesIO()
    Image.new("RGB", size, color=color).save(buf, format="BMP")
    return buf.getvalue()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_zip(path: Path, entries: list[tuple[str, bytes]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for archive_path, payload in entries:
            zf.writestr(archive_path, payload)
    return path


def test_freeze_legacy_elements_builds_manifest_filters_small_classes_and_emits_source_groups(tmp_path: Path):
    runtime_config = tmp_path / "runtime-config.json"
    runtime_config.write_text(json.dumps({"class_names": ["café", "atl"]}, ensure_ascii=False), encoding="utf-8")
    source_zip = _write_zip(
        tmp_path / "Elements.zip",
        [
            ("Elements/", b""),
            ("Elements/.DS_Store", b"noise"),
            ("__MACOSX/._Elements", b"noise"),
            ("Elements/0002-café/10_11_12-1.bmp", _bmp_bytes((10, 200, 10))),
            ("Elements/0002-café/10_11_12-2.bmp", _bmp_bytes((10, 180, 10))),
            ("Elements/0001-atl/01_02_03-1.bmp", _bmp_bytes((200, 10, 10))),
            ("Elements/0001-atl/01_02_03-2.bmp", _bmp_bytes((180, 10, 10))),
            ("Elements/0001-atl/loose.bmp", _bmp_bytes((160, 10, 10))),
            ("Elements/0003-dropme/99_99_99-1.bmp", _bmp_bytes((10, 10, 200))),
        ],
    )
    output_dir = tmp_path / "frozen"

    report = module.freeze_legacy_elements(source_zip, output_dir, runtime_config=runtime_config, min_images_per_class=2)

    manifest_path = output_dir / "legacy_elements_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert report.schema_version == "legacy-elements-freeze.v1"
    assert manifest["schema_version"] == "legacy-elements-freeze.v1"
    assert manifest["source_zip_sha256"] == _sha256_bytes(source_zip.read_bytes())
    assert manifest["class_order"] == ["café", "atl"]
    assert manifest["class_order_source"] == "runtime_config"
    assert manifest["class_order_sha256"] == module.canonical_json_sha(["café", "atl"])
    assert manifest["kept_class_count"] == 2
    assert manifest["rejected_class_count"] == 1
    assert manifest["image_count"] == 5
    assert manifest["filtered_image_count"] == 1
    assert manifest["source_group_count"] == 3
    assert manifest["ignored_entry_count"] == 2
    assert [item["class_name"] for item in manifest["classes"]] == ["café", "atl"]
    assert manifest["classes"][0]["class_dir"] == "0001-café"
    assert manifest["classes"][0]["prefix"] == 1
    assert manifest["classes"][0]["source_class_dir"] == "0002-café"
    assert manifest["classes"][0]["source_prefix"] == 2
    assert manifest["classes"][0]["source_groups"] == ["page:10:11:12"]
    assert manifest["classes"][1]["class_dir"] == "0002-atl"
    assert manifest["classes"][1]["prefix"] == 2
    assert manifest["classes"][1]["source_class_dir"] == "0001-atl"
    assert manifest["classes"][1]["source_prefix"] == 1
    assert manifest["classes"][1]["source_groups"] == ["page:01:02:03", "image:Elements/0001-atl/loose.bmp"]
    assert manifest["rejected_classes"][0]["reason"] == "below_min_images_per_class:1<2"

    cafe_dir = output_dir / "Elements" / "0001-café"
    atl_dir = output_dir / "Elements" / "0002-atl"
    assert cafe_dir.is_dir()
    assert atl_dir.is_dir()
    assert not (output_dir / "Elements" / "0003-dropme").exists()
    assert {path.suffix.lower() for path in output_dir.rglob("*") if path.is_file()} == {".bmp", ".json"}

    image_records = manifest["images"]
    assert len(image_records) == 5
    assert image_records[0]["source_group"] == "page:10:11:12"
    assert image_records[1]["source_group"] == "page:10:11:12"
    assert image_records[2]["source_group"] == "page:01:02:03"
    assert image_records[4]["source_group"] == "image:Elements/0001-atl/loose.bmp"
    assert image_records[0]["source_sha256"] == image_records[0]["output_sha256"]
    assert Path(image_records[0]["output_path"]).as_posix().endswith("Elements/0001-café/10_11_12-1.bmp")
    assert (cafe_dir / "10_11_12-1.bmp").is_file()
    assert (cafe_dir / "10_11_12-2.bmp").is_file()
    assert (atl_dir / "01_02_03-1.bmp").is_file()
    assert (atl_dir / "loose.bmp").is_file()

def test_freeze_legacy_elements_rejects_non_bmp_payloads(tmp_path: Path):
    source_zip = _write_zip(
        tmp_path / "Elements.zip",
        [
            ("Elements/0001-atl/01_02_03-1.bmp", _bmp_bytes((200, 10, 10))),
            ("Elements/0001-atl/readme.txt", b"not allowed"),
        ],
    )

    with pytest.raises(ValueError, match="non-BMP file found"):
        module.freeze_legacy_elements(source_zip, tmp_path / "frozen")

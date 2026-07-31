from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from codex_pipeline.data.metadata import build_metadata  # noqa: E402


def test_build_metadata_can_write_absolute_image_paths(tmp_path: Path):
    image_path = tmp_path / "Elements" / "0001-atl" / "01_02_03-sample.bmp"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (5, 5)).save(image_path)

    metadata = build_metadata(str(tmp_path / "Elements"), absolute_paths=True)

    assert metadata.iloc[0]["image_path"] == str(image_path.resolve())

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_release_version_is_valid_and_synchronized() -> None:
    release_version = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
    with (ROOT / "backend" / "pyproject.toml").open("rb") as handle:
        backend_version = tomllib.load(handle)["project"]["version"]
    frontend_version = json.loads(
        (ROOT / "frontend" / "package.json").read_text(encoding="utf-8")
    )["version"]

    assert re.fullmatch(r"\d+\.\d+\.\d+", release_version)
    assert backend_version == release_version
    assert frontend_version == release_version

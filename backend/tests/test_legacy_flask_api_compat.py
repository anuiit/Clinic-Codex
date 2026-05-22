from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_legacy_module_exposes_app():
    import backend.examples.flask_api as module

    assert hasattr(module, "app")
    assert hasattr(module, "create_app")


def test_legacy_module_imports_in_subprocess():
    result = subprocess.run(
        [sys.executable, "-c", "import backend.examples.flask_api as m; assert hasattr(m, 'app'); print('compat-ok')"],
        cwd=".",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "compat-ok" in result.stdout


def test_legacy_file_is_thin_shim_without_route_decorators_or_endpoint_logic():
    source = Path("backend/examples/flask_api.py").read_text()
    assert "@app.route" not in source
    assert "@bp.route" not in source
    for name in ["def classify(", "def classify_batch(", "def segment(", "def similar(", "def trust(", "def save_annotation_route("]:
        assert name not in source

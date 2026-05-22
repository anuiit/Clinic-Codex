from __future__ import annotations

import subprocess
import sys
import textwrap


def test_factory_and_wsgi_import_do_not_import_heavy_ml_modules():
    script = textwrap.dedent(
        """
        import builtins
        import sys

        forbidden = {"torch", "mobile_sam", "codex_model", "codex_pipeline.segmentation"}
        original_import = builtins.__import__

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name in forbidden or name.split('.')[0] in {"torch", "mobile_sam", "codex_model"}:
                raise AssertionError(f"forbidden import during app creation: {name}")
            if name.startswith("codex_pipeline.segmentation"):
                raise AssertionError(f"forbidden import during app creation: {name}")
            return original_import(name, globals, locals, fromlist, level)

        builtins.__import__ = guarded_import
        from backend.app.factory import create_app
        app = create_app()
        import backend.wsgi
        assert app is not None
        assert backend.wsgi.app is not None
        print("cheap-import-ok")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=".",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "cheap-import-ok" in result.stdout

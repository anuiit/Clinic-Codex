"""Native Windows regressions, including Windows PowerShell 5.1."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(os.name != "nt", reason="native Windows scripts")


@pytest.fixture(params=["powershell", "pwsh"])
def shell(request):
    executable = shutil.which(request.param)
    if not executable:
        pytest.skip(f"{request.param} is not installed")
    return executable


def test_powershell_snapshot_dry_run_preserves_space_paths(shell, tmp_path):
    config = tmp_path / "recipe with spaces.yaml"
    config.write_text("training:\n  checkpoint_selection: latest\n", encoding="utf-8")
    result = subprocess.run([
        shell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
        str(ROOT / "scripts/retrain.ps1"), "-DryRun",
        "-TrainingConfigOverride", str(config),
        "-ElementsDirOverride", str(tmp_path / "Elements with spaces"),
        "-ApprovedManifestOverride", str(tmp_path / "manifest.json"),
        "-MetadataCsvOverride", str(tmp_path / "metadata.csv"),
        "-BackboneManifestOverride", str(tmp_path / "pin.json"),
        "-InitProjection", str(tmp_path / "weights/projection.pt"),
        "-UpdateAnnotatedPrototypes",
    ], env={**os.environ, "PYTHON": sys.executable, "DEVICE": "cpu"},
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert str(config) in result.stdout
    assert "--checkpoint-selection latest" in result.stdout
    assert "--eval-only" in result.stdout
    assert "--evaluate-candidate" in result.stdout


def test_powershell_and_local_jobs_share_crash_safe_lock(shell, tmp_path):
    from scripts.retrain_local import training_lock
    scripts = tmp_path / "scripts"
    backend = tmp_path / "backend"
    pipeline = backend / "codex_pipeline/scripts"
    scripts.mkdir()
    pipeline.mkdir(parents=True)
    shutil.copy2(ROOT / "scripts/retrain.ps1", scripts / "retrain.ps1")
    # Stall at the first real pipeline step, with the production script holding its lock.
    (scripts / "export_approved_annotations.py").write_text(
        "import time\nprint('LOCK_HELD', flush=True)\ntime.sleep(30)\n", encoding="utf-8")
    config = tmp_path / "recipe.yaml"
    config.write_text("training: {}\n", encoding="utf-8")
    command = [shell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
               str(scripts / "retrain.ps1"), "-TrainingConfigOverride", str(config)]
    env = {**os.environ, "PYTHON": sys.executable, "DEVICE": "cpu", "GIT_SHORT": "test"}
    with training_lock(backend):
        winner = (backend / ".retrain.lock").read_bytes()
        loser = subprocess.run(command, env=env, capture_output=True, timeout=30)
        assert loser.returncode != 0
        assert (backend / ".retrain.lock").read_bytes() == winner
    log = tmp_path / "job.log"
    with log.open("w") as output:
        process = subprocess.Popen(command, env=env, stdout=output, stderr=subprocess.STDOUT)
    try:
        deadline = time.monotonic() + 25
        while "LOCK_HELD" not in log.read_text(errors="replace"):
            assert process.poll() is None, log.read_text(errors="replace")
            assert time.monotonic() < deadline, log.read_text(errors="replace")
            time.sleep(0.1)
        with pytest.raises((OSError, RuntimeError)):
            with training_lock(backend):
                pytest.fail("Python stole the PowerShell job lock")
        winner = (backend / ".retrain.lock").read_bytes()
        loser = subprocess.run(command, env=env, capture_output=True, timeout=15)
        assert loser.returncode != 0
        assert (backend / ".retrain.lock").read_bytes() == winner
    finally:
        # Exact test-owned process tree; no production processes are targeted.
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                       capture_output=True, timeout=15)
        process.wait(timeout=10)
    with training_lock(backend):
        assert (backend / ".retrain.lock").read_text() == str(os.getpid())
    assert not (backend / ".retrain.lock").exists()

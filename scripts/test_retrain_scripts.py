from __future__ import annotations

import shutil
import subprocess
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_retrain_sh_dry_run_uses_approved_only_explicit_paths():
    env = {
        **os.environ,
        "MODEL_VERSION_ID": "20260527T010203Z-test-deadbeef",
        "MODEL_VERSION_RUN_ID": "deadbeef",
        "GIT_SHORT": "test",
    }
    result = subprocess.run(
        ["bash", "scripts/retrain.sh", "--dry-run"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )
    output = result.stdout
    version_fragment = "backend/model_registry/versions/20260527T010203Z-test-deadbeef"

    expected_fragments = [
        "[1/6] export_approved_annotations",
        "scripts/export_approved_annotations.py",
        "--annotations-dir",
        "backend/annotations",
        "--output",
        "backend/training_data/approved/Elements",
        "[2/6] build_metadata",
        "--elements-dir",
        "backend/training_data/approved/Elements",
        "--output",
        "backend/training_data/approved/metadata.csv",
        "[3/6] precompute_embeddings",
        "--metadata-csv",
        "backend/training_data/approved/metadata.csv",
        "--output-dir",
        "backend/training_data/approved/precomputed",
        "[4/6] train",
        "--features",
        "backend/training_data/approved/precomputed/features.pt",
        "--checkpoint-dir",
        f"{version_fragment}/checkpoints",
        "[5/6] evaluate_export_prototypes",
        "--export-prototypes",
        "--prototype-dir",
        f"{version_fragment}/prototypes",
        "[6/6] export_model",
        "--prototypes",
        f"{version_fragment}/prototypes/prototypes.pt",
        "--weights-dir",
        f"{version_fragment}/runtime/weights",
        "--config-template",
        "backend/codex_model/config.json",
        "--config-out",
        f"{version_fragment}/runtime/config.json",
        "--manifest-out",
        f"{version_fragment}/export_model_manifest.json",
        "--registry-dir",
        "backend/model_registry",
        "--version-id",
        "20260527T010203Z-test-deadbeef",
        "--metadata-csv",
        "backend/training_data/approved/metadata.csv",
        "--approved-manifest",
        "backend/training_data/approved/Elements/_approved_export_manifest.json",
        "Candidate model version created: 20260527T010203Z-test-deadbeef",
    ]
    for fragment in expected_fragments:
        assert fragment in output

    assert "features_aug.pt" not in output
    assert "sam_full_test" not in output.lower()
    assert "--allow-runtime-write" not in output
    assert " --weights-dir backend/codex_model/weights" not in output
    assert " --config-out backend/codex_model/config.json" not in output


def test_retrain_sh_rejects_unsafe_model_version_id_before_paths_are_used():
    result = subprocess.run(
        ["bash", "scripts/retrain.sh", "--dry-run"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        env={**os.environ, "MODEL_VERSION_ID": "../../codex_model"},
    )

    assert result.returncode == 2
    assert "invalid MODEL_VERSION_ID" in result.stderr
    assert "model_registry/versions/../../codex_model" not in result.stdout


def test_retrain_ps1_contains_matching_approved_only_stage_contract():
    text = (REPO_ROOT / "scripts/retrain.ps1").read_text()
    for fragment in [
        "[1/6] export_approved_annotations",
        "scripts/export_approved_annotations.py",
        "--annotations-dir",
        "--output",
        "[2/6] build_metadata",
        "--elements-dir",
        "[3/6] precompute_embeddings",
        "--metadata-csv",
        "--output-dir",
        "[4/6] train",
        "--features",
        "--checkpoint-dir",
        "model_registry",
        "versions",
        "[5/6] evaluate_export_prototypes",
        "--export-prototypes",
        "--prototype-dir",
        "[6/6] export_model",
        "--prototypes",
        "--weights-dir",
        "--config-template",
        "--config-out",
        "--registry-dir",
        "--version-id",
        "--metadata-csv",
        "--approved-manifest",
        "codex_model/config.json",
        "promote_model.py",
    ]:
        assert fragment in text

    assert "features_aug.pt" not in text
    assert "sam_full_test" not in text.lower()
    assert "--allow-runtime-write" not in text


def test_retrain_ps1_dry_run_when_powershell_is_available():
    pwsh = shutil.which("pwsh") or shutil.which("powershell")
    if pwsh is None:
        return

    result = subprocess.run(
        [pwsh, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "scripts/retrain.ps1", "-DryRun"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
        env={**os.environ, "MODEL_VERSION_ID": "20260527T010203Z-test-deadbeef"},
    )
    assert "[1/6] export_approved_annotations" in result.stdout
    assert "--metadata-csv" in result.stdout
    assert "model_registry" in result.stdout
    assert "--allow-runtime-write" not in result.stdout


def test_retrain_ps1_rejects_unsafe_model_version_id_when_powershell_is_available():
    pwsh = shutil.which("pwsh") or shutil.which("powershell")
    if pwsh is None:
        return

    result = subprocess.run(
        [pwsh, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "scripts/retrain.ps1", "-DryRun"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        env={**os.environ, "MODEL_VERSION_ID": "..\\..\\codex_model"},
    )
    # Windows PowerShell and PowerShell Core can normalize explicit script exit
    # codes differently after a terminating parameter-validation error. The
    # stable contract is that the unsafe value is rejected before paths are
    # used and the process exits non-zero.
    assert result.returncode in {1, 2}
    assert "invalid MODEL_VERSION_ID" in (result.stderr + result.stdout)

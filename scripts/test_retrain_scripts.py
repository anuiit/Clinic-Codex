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
        "--training-config",
        "backend/codex_pipeline/config/default.yaml",
        "--features-provenance",
        "backend/training_data/approved/precomputed/features.pt.prov.json",
        "--checkpoint",
        f"{version_fragment}/checkpoints/best.pt",
        "--training-manifest",
        f"{version_fragment}/checkpoints/training_manifest.json",
        "--checkpoint-selection",
        "best",
        "Candidate model version created: 20260527T010203Z-test-deadbeef",
    ]
    for fragment in expected_fragments:
        assert fragment in output

    assert "features_aug.pt" not in output
    assert "sam_full_test" not in output.lower()
    assert "--allow-runtime-write" not in output
    assert " --weights-dir backend/codex_model/weights" not in output
    assert " --config-out backend/codex_model/config.json" not in output


def test_retrain_defaults_to_cuda_with_an_explicit_cpu_override():
    shell = (REPO_ROOT / "scripts/retrain.sh").read_text()
    powershell = (REPO_ROOT / "scripts/retrain.ps1").read_text()

    assert 'DEVICE="${DEVICE:-cuda}"' in shell
    assert "require_cuda" in shell
    assert "$Device = if ($env:DEVICE) { $env:DEVICE } else { 'cuda' }" in powershell
    assert "Assert-CudaAvailable" in powershell


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


def test_retrain_sh_dry_run_accepts_prepared_snapshot_with_explicit_provenance():
    result = subprocess.run(
        [
            "bash",
            "scripts/retrain.sh",
            "--dry-run",
            "--elements-dir",
            "/tmp/external-snapshot/Elements",
            "--approved-manifest",
            "/tmp/external-snapshot/snapshot_manifest.json",
            "--metadata-csv",
            "/tmp/external-snapshot/metadata.csv",
            "--backbone-manifest",
            "/tmp/external-snapshot/dinov2-vits14-pin.json",
        ],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
        env={**os.environ, "MODEL_VERSION_ID": "20260527T010203Z-test-external"},
    )
    assert "[1/6] use_prepared_elements_snapshot" in result.stdout
    assert "scripts/export_approved_annotations.py" not in result.stdout
    assert "/tmp/external-snapshot/Elements" in result.stdout
    assert "/tmp/external-snapshot/snapshot_manifest.json" in result.stdout
    assert "/tmp/external-snapshot/metadata.csv" in result.stdout
    assert "use_snapshot_metadata" in result.stdout
    assert "codex_pipeline/config/snapshot.yaml" in result.stdout
    assert "--runtime-config" in result.stdout
    assert "--backbone-manifest /tmp/external-snapshot/dinov2-vits14-pin.json" in result.stdout
    assert "--split-strategy persisted" in result.stdout
    assert "--prototype-split train" in result.stdout
    assert "--skip-few-shot" in result.stdout
    assert "--training-config" in result.stdout
    assert "--features /home" in result.stdout or "--features " in result.stdout
    assert "features.pt.prov.json" in result.stdout
    assert "--checkpoint" in result.stdout


def test_retrain_sh_prepared_snapshot_requires_explicit_metadata():
    result = subprocess.run(
        [
            "bash",
            "scripts/retrain.sh",
            "--dry-run",
            "--elements-dir",
            "/tmp/external-snapshot/Elements",
            "--approved-manifest",
            "/tmp/external-snapshot/snapshot_manifest.json",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        env={**os.environ, "MODEL_VERSION_ID": "20260527T010203Z-test-external"},
    )

    assert result.returncode == 2
    assert "requires --metadata-csv" in result.stderr


def test_retrain_sh_wires_eval_only_warmstart_and_fixed_latest_selection():
    result = subprocess.run(
        [
            "bash", "scripts/retrain.sh", "--dry-run",
            "--init-projection", "/tmp/historical-projection.pt",
            "--eval-only", "--checkpoint-selection", "latest",
        ],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
        env={**os.environ, "MODEL_VERSION_ID": "snapshot-anchor-test"},
    )

    assert "train.py" in result.stdout
    assert "--init-projection /tmp/historical-projection.pt" in result.stdout
    assert "--eval-only" in result.stdout
    assert result.stdout.count("checkpoints/latest.pt") == 2
    assert "--checkpoint-selection latest" in result.stdout
    assert "--training-manifest" in result.stdout


def test_retrain_sh_adopts_preregistered_checkpoint_selection(tmp_path):
    config = tmp_path / "latest.yaml"
    config.write_text("training:\n  checkpoint_selection: latest\n", encoding="utf-8")
    result = subprocess.run(
        ["bash", "scripts/retrain.sh", "--dry-run", "--config", str(config)],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
        env={**os.environ, "MODEL_VERSION_ID": "snapshot-selection-test"},
    )
    assert result.stdout.count("checkpoints/latest.pt") == 2
    assert "--checkpoint-selection latest" in result.stdout


def test_retrain_sh_rejects_checkpoint_selection_recipe_mismatch(tmp_path):
    config = tmp_path / "latest.yaml"
    config.write_text("training:\n  checkpoint_selection: latest\n", encoding="utf-8")
    result = subprocess.run(
        [
            "bash", "scripts/retrain.sh", "--dry-run", "--config", str(config),
            "--checkpoint-selection", "best",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        env={**os.environ, "MODEL_VERSION_ID": "snapshot-selection-test"},
    )
    assert result.returncode != 0
    assert "conflicts with preregistered training.checkpoint_selection=latest" in result.stderr


def test_retrain_sh_rejects_eval_only_without_warmstart():
    result = subprocess.run(
        ["bash", "scripts/retrain.sh", "--dry-run", "--eval-only"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        env={**os.environ, "MODEL_VERSION_ID": "snapshot-anchor-test"},
    )
    assert result.returncode == 2
    assert "requires --init-projection" in result.stderr


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
        "--training-config",
        "--features-provenance",
        "--checkpoint",
        "--init-projection",
        "--eval-only",
        "--checkpoint-selection",
        "--training-manifest",
        "codex_model/config.json",
        "promote_model.py",
    ]:
        assert fragment in text

    assert "features_aug.pt" not in text
    assert "sam_full_test" not in text.lower()
    assert "--allow-runtime-write" not in text
    assert "__UNSET__" in text
    assert "yaml.safe_load" in text
    assert "training.checkpoint_selection" in text


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
    assert "best.pt" in result.stdout
    assert "--checkpoint-selection best" in result.stdout
    assert "--allow-runtime-write" not in result.stdout


def test_retrain_ps1_wires_warmstart_latest_when_powershell_is_available():
    pwsh = shutil.which("pwsh") or shutil.which("powershell")
    if pwsh is None:
        return

    result = subprocess.run(
        [
            pwsh, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
            "scripts/retrain.ps1", "-DryRun",
            "-InitProjection", "C:\\tmp\\historical-projection.pt",
            "-EvalOnly", "-CheckpointSelection", "latest",
        ],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
        env={**os.environ, "MODEL_VERSION_ID": "snapshot-anchor-test"},
    )
    assert "--init-projection" in result.stdout
    assert "--eval-only" in result.stdout
    assert "latest.pt" in result.stdout
    assert "--checkpoint-selection latest" in result.stdout


def test_retrain_ps1_adopts_and_enforces_recipe_selection_when_powershell_is_available(tmp_path):
    pwsh = shutil.which("pwsh") or shutil.which("powershell")
    if pwsh is None:
        return
    config = tmp_path / "latest.yaml"
    config.write_text("training:\n  checkpoint_selection: latest\n", encoding="utf-8")
    base = [
        pwsh, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
        "scripts/retrain.ps1", "-DryRun", "-TrainingConfigOverride", str(config),
    ]
    env = {**os.environ, "MODEL_VERSION_ID": "snapshot-selection-test"}

    adopted = subprocess.run(
        base, cwd=REPO_ROOT, check=True, text=True, capture_output=True, env=env
    )
    assert "latest.pt" in adopted.stdout
    assert "--checkpoint-selection latest" in adopted.stdout

    mismatch = subprocess.run(
        [*base, "-CheckpointSelection", "best"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        env=env,
    )
    assert mismatch.returncode != 0
    assert "training.checkpoint_selection=latest" in (mismatch.stdout + mismatch.stderr)


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

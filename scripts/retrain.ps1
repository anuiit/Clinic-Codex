<#
.SYNOPSIS
Run classifier retraining from approved data or a cumulative snapshot.

.DESCRIPTION
PowerShell counterpart to scripts/retrain.sh. Without snapshot arguments it
exports admin-approved annotation crops. With snapshot arguments it validates
and trains from the prepared cumulative corpus, optionally warm-starting the
projection. Both paths write immutable candidate artifacts under
backend/model_registry/versions/<id>.

No segmentation/MobileSAM retraining is performed.
No runtime backend/codex_model artifacts are modified; run scripts/promote_model.py
to activate a candidate.
#>
[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [switch]$DryRun,
    [string]$ElementsDirOverride,
    [string]$ApprovedManifestOverride,
    [string]$MetadataCsvOverride,
    [string]$BackboneManifestOverride,
    [string]$TrainingConfigOverride,
    [string]$InitProjection,
    [switch]$EvalOnly,
    [switch]$UpdateAnnotatedPrototypes,
    [ValidateSet('best', 'latest')]
    [string]$CheckpointSelection
)

$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
$BackendDir = Join-Path $RepoRoot 'backend'
$LockFile = Join-Path $BackendDir '.retrain.lock'
$Pipeline = Join-Path $BackendDir 'codex_pipeline/scripts'
$AnnotationsDir = Join-Path $BackendDir 'annotations'
$ApprovedRoot = Join-Path $BackendDir 'training_data/approved'
$ElementsDir = Join-Path $ApprovedRoot 'Elements'
$MetadataCsv = Join-Path $ApprovedRoot 'metadata.csv'
$PrecomputedDir = Join-Path $ApprovedRoot 'precomputed'
$FeaturesFile = Join-Path $PrecomputedDir 'features.pt'
$ModelRegistryDir = if ($env:MODEL_REGISTRY_DIR) { $env:MODEL_REGISTRY_DIR } else { Join-Path $BackendDir 'model_registry' }
try {
    $GitShort = if ($env:GIT_SHORT) { $env:GIT_SHORT } else { (git -C $RepoRoot rev-parse --short=8 HEAD 2>$null).Trim() }
    if (-not $GitShort) { $GitShort = 'nogit' }
} catch {
    $GitShort = 'nogit'
}
$RunSource = if ($env:MODEL_VERSION_RUN_ID) { $env:MODEL_VERSION_RUN_ID } else { ((Get-Date).ToUniversalTime().ToString('HHmmss') + '-' + $PID) }
$RunPrefix = if ($RunSource.Length -gt 8) { $RunSource.Substring(0, 8) } else { $RunSource }
$ModelVersionId = if ($env:MODEL_VERSION_ID) {
    $env:MODEL_VERSION_ID
} else {
    ((Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ') + '-' + $GitShort + '-' + $RunPrefix)
}
if ([string]::IsNullOrWhiteSpace($ModelVersionId) -or
    $ModelVersionId.Contains('/') -or
    $ModelVersionId.Contains('\') -or
    $ModelVersionId.Contains('..') -or
    ($ModelVersionId -notmatch '^[A-Za-z0-9_.-]+$')) {
    Write-Error "ERROR: invalid MODEL_VERSION_ID: $ModelVersionId. MODEL_VERSION_ID may contain only letters, numbers, underscore, dot, and dash; path separators and '..' are forbidden."
    exit 2
}
$VersionDir = Join-Path $ModelRegistryDir (Join-Path 'versions' $ModelVersionId)
$CheckpointDir = Join-Path $VersionDir 'checkpoints'
$PrototypeDir = Join-Path $VersionDir 'prototypes'
$PrototypeFile = Join-Path $PrototypeDir 'prototypes.pt'
$WeightsDir = Join-Path $VersionDir 'runtime/weights'
$ClassifierConfigTemplate = Join-Path $BackendDir 'codex_model/config.json'
$ClassifierConfig = Join-Path $VersionDir 'runtime/config.json'
$ExportManifest = Join-Path $VersionDir 'export_model_manifest.json'
$Config = Join-Path $BackendDir 'codex_pipeline/config/default.yaml'
$BatchSize = if ($env:BATCH_SIZE) { $env:BATCH_SIZE } else { '16' }
# Training is GPU-first. Set DEVICE=cpu only for an intentional CPU fallback.
$Device = if ($env:DEVICE) { $env:DEVICE } else { 'cuda' }

if ($UpdateAnnotatedPrototypes) {
    if (-not $ElementsDirOverride -or -not $InitProjection) {
        throw 'UpdateAnnotatedPrototypes requires ElementsDirOverride and InitProjection.'
    }
    $EvalOnly = $true
}

if (($ApprovedManifestOverride -or $MetadataCsvOverride -or $BackboneManifestOverride) -and -not $ElementsDirOverride) {
    Write-Error 'ERROR: snapshot provenance options require -ElementsDirOverride.'
    exit 2
}
if ($EvalOnly -and -not $InitProjection) {
    Write-Error 'ERROR: -EvalOnly requires -InitProjection.'
    exit 2
}
if ($ElementsDirOverride -and -not $ApprovedManifestOverride) {
    Write-Error 'ERROR: -ElementsDirOverride requires -ApprovedManifestOverride for candidate provenance.'
    exit 2
}
if ($ElementsDirOverride -and -not $MetadataCsvOverride) {
    Write-Error 'ERROR: -ElementsDirOverride requires -MetadataCsvOverride from the immutable snapshot.'
    exit 2
}
if ($ElementsDirOverride -and -not $BackboneManifestOverride) {
    Write-Error 'ERROR: -ElementsDirOverride requires -BackboneManifestOverride for reproducible DINOv2 features.'
    exit 2
}
if ($ElementsDirOverride) {
    $ElementsDir = [System.IO.Path]::GetFullPath($ElementsDirOverride)
    $ApprovedManifest = [System.IO.Path]::GetFullPath($ApprovedManifestOverride)
    $TrainingWorkDir = Join-Path $VersionDir 'training_data'
    $MetadataCsv = [System.IO.Path]::GetFullPath($MetadataCsvOverride)
    $BackboneManifest = [System.IO.Path]::GetFullPath($BackboneManifestOverride)
    $PrecomputedDir = Join-Path $TrainingWorkDir 'precomputed'
    $FeaturesFile = Join-Path $PrecomputedDir 'features.pt'
    if (-not $TrainingConfigOverride) {
        $Config = Join-Path $BackendDir 'codex_pipeline/config/snapshot.yaml'
    }
} else {
    $ApprovedManifest = Join-Path $ElementsDir '_approved_export_manifest.json'
    $TrainingWorkDir = $ApprovedRoot
}
if ($TrainingConfigOverride) {
    $Config = [System.IO.Path]::GetFullPath($TrainingConfigOverride)
}

function Write-Step([string]$Message) {
    Write-Host $Message
}

function Get-PythonPath {
    $candidates = @(
        $env:PYTHON,
        (Join-Path $RepoRoot 'backend/.venv/Scripts/python.exe'),
        (Join-Path $RepoRoot 'backend/.venv/bin/python3'),
        (Join-Path $RepoRoot 'backend/.venv/bin/python'),
        'python'
    )

    foreach ($candidate in $candidates) {
        if (-not $candidate) { continue }
        try {
            if ((Test-Path $candidate) -or (Get-Command $candidate -ErrorAction SilentlyContinue)) {
                return $candidate
            }
        } catch {}
    }

    throw 'Python not found. Run scripts/install.ps1 first or ensure python is on PATH.'
}

function Resolve-CheckpointSelection(
    [string]$PythonPath,
    [string]$ConfigPath,
    [string]$CliSelection
) {
    $sentinel = '__UNSET__'
    $cliArgument = if ($CliSelection) { $CliSelection } else { $sentinel }
    $resolver = @'
import sys
from pathlib import Path

import yaml

config_path = Path(sys.argv[1])
cli_selection = None if sys.argv[2] == "__UNSET__" else sys.argv[2]
try:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
except (OSError, yaml.YAMLError) as exc:
    raise SystemExit(f"ERROR: could not read training config {config_path}: {exc}") from exc
if not isinstance(config, dict):
    raise SystemExit(f"ERROR: training config must be a YAML mapping: {config_path}")
training = config.get("training", {})
if not isinstance(training, dict):
    raise SystemExit(f"ERROR: training config 'training' must be a mapping: {config_path}")
config_selection = training.get("checkpoint_selection")
if config_selection is not None and config_selection not in {"best", "latest"}:
    raise SystemExit(
        "ERROR: training.checkpoint_selection must be best or latest, "
        f"got {config_selection!r}"
    )
if cli_selection and config_selection and cli_selection != config_selection:
    raise SystemExit(
        "ERROR: -CheckpointSelection conflicts with preregistered "
        f"training.checkpoint_selection={config_selection}"
    )
print(cli_selection or config_selection or "best")
'@
    $selection = $resolver | & $PythonPath - $ConfigPath $cliArgument
    if ($LASTEXITCODE -ne 0) {
        throw 'Training checkpoint selection resolution failed.'
    }
    return ([string]$selection).Trim()
}

function Test-ProcessRunning([int]$ProcessId) {
    try {
        $null = Get-Process -Id $ProcessId -ErrorAction Stop
        return $true
    } catch {
        return $false
    }
}

function Join-CommandLine([string[]]$Parts) {
    return ($Parts | ForEach-Object {
        if ($_ -match '\s') { '"' + ($_ -replace '"', '\"') + '"' } else { $_ }
    }) -join ' '
}

function Invoke-PipelineStep([string]$Label, [string[]]$CommandParts) {
    Write-Step "=== $Label ==="
    Write-Host ('+ ' + (Join-CommandLine $CommandParts))
    if ($DryRun) {
        return
    }
    if ($PSCmdlet.ShouldProcess($CommandParts[1], 'run classifier retraining step')) {
        & $CommandParts[0] @($CommandParts[1..($CommandParts.Count - 1)])
        if ($LASTEXITCODE -ne 0) {
            throw "Step failed: $Label"
        }
    }
}

function Assert-CudaAvailable([string]$PythonPath, [string]$RequestedDevice) {
    if ($RequestedDevice -notmatch '^cuda(:[0-9]+)?$') {
        return
    }

    @'
import sys
import torch
if not torch.cuda.is_available():
    sys.exit("ERROR: DEVICE=cuda was requested, but this Python environment cannot use CUDA. Install the CUDA PyTorch build with scripts/install_gpu_training.sh, or explicitly override with DEVICE=cpu.")
print(f"CUDA training device: {torch.cuda.get_device_name(0)}")
'@ | & $PythonPath -
    if ($LASTEXITCODE -ne 0) {
        throw 'CUDA preflight failed.'
    }
}

Set-Location $RepoRoot
$Python = Get-PythonPath
$CheckpointSelection = Resolve-CheckpointSelection $Python $Config $CheckpointSelection
$SelectedCheckpoint = Join-Path $CheckpointDir ($CheckpointSelection + '.pt')

$Guard = $null
$OwnsLock = $false
try {
if (-not $DryRun -and -not $WhatIfPreference) {
    Assert-CudaAvailable $Python $Device
    # Same first-byte OS lock as retrain_local.py; also released after a crash.
    $Guard = [System.IO.File]::Open((Join-Path $BackendDir '.retrain.guard'), 'OpenOrCreate', 'ReadWrite', 'ReadWrite')
    $Guard.Lock(0, 1)
    if (Test-Path $LockFile) {
        $oldPidText = (Get-Content $LockFile -Raw).Trim()
        $oldPid = 0
        if ([int]::TryParse($oldPidText, [ref]$oldPid) -and (Test-ProcessRunning $oldPid)) {
            Write-Error "ERROR: retrain already running (PID $oldPid). Aborting."
            exit 1
        }
        Write-Warning "stale lockfile (PID $oldPidText not running). Removing."
        Remove-Item $LockFile -Force
    }

    Set-Content -Path $LockFile -Value $PID -Encoding ascii
    $OwnsLock = $true
    New-Item -ItemType Directory -Force -Path $TrainingWorkDir | Out-Null
}

    if ($ElementsDirOverride) {
        Write-Step '=== [1/6] use_prepared_elements_snapshot ==='
        Write-Host ("+ prepared Elements: " + $ElementsDir)
        Write-Host ("+ approved manifest: " + $ApprovedManifest)
        if (-not $DryRun) {
            if (-not (Test-Path -Path $ElementsDir -PathType Container)) { throw "Elements directory does not exist: $ElementsDir" }
            if (-not (Test-Path -Path $ApprovedManifest -PathType Leaf)) { throw "Approved manifest does not exist: $ApprovedManifest" }
            if (-not (Test-Path -Path $MetadataCsv -PathType Leaf)) { throw "Snapshot metadata does not exist: $MetadataCsv" }
            if (-not (Test-Path -Path $BackboneManifest -PathType Leaf)) { throw "Backbone manifest does not exist: $BackboneManifest" }
            if (-not (Test-Path -Path $Config -PathType Leaf)) { throw "Training config does not exist: $Config" }
        }
    } else {
        Invoke-PipelineStep '[1/6] export_approved_annotations' @(
            $Python,
            (Join-Path $RepoRoot 'scripts/export_approved_annotations.py'),
            '--annotations-dir', $AnnotationsDir,
            '--output', $ElementsDir
        )
    }
    if ($ElementsDirOverride) {
        Write-Step '=== [2/6] use_snapshot_metadata ==='
        Write-Host ("+ snapshot metadata: " + $MetadataCsv)
    } else {
        $MetadataCommand = @(
            $Python,
            (Join-Path $Pipeline 'build_metadata.py'),
            '--elements-dir', $ElementsDir,
            '--output', $MetadataCsv
        )
        Invoke-PipelineStep '[2/6] build_metadata' $MetadataCommand
    }
    $PrecomputeCommand = @(
        $Python,
        (Join-Path $Pipeline 'precompute_embeddings.py'),
        '--config', $Config,
        '--metadata-csv', $MetadataCsv,
        '--runtime-config', $ClassifierConfigTemplate,
        '--batch-size', $BatchSize,
        '--device', $Device,
        '--output-dir', $PrecomputedDir
    )
    if ($ElementsDirOverride) {
        $PrecomputeCommand += @(
            '--snapshot-manifest', $ApprovedManifest,
            '--backbone-manifest', $BackboneManifest
        )
    }
    Invoke-PipelineStep '[3/6] precompute_embeddings' $PrecomputeCommand
    $TrainCommand = @(
        $Python,
        (Join-Path $Pipeline 'train.py'),
        '--config', $Config,
        '--features', $FeaturesFile,
        '--checkpoint-dir', $CheckpointDir
    )
    if ($InitProjection) {
        $TrainCommand += @('--init-projection', $InitProjection)
    }
    if ($EvalOnly) {
        $TrainCommand += '--eval-only'
    }
    Invoke-PipelineStep '[4/6] train' $TrainCommand
    $EvaluateCommand = @(
        $Python,
        (Join-Path $Pipeline 'evaluate.py'),
        '--checkpoint', $SelectedCheckpoint,
        '--features', $FeaturesFile
    )
    if ($ElementsDirOverride) {
        $EvaluateCommand += @(
            '--split-strategy', 'persisted',
            '--prototype-split', 'train',
            '--skip-few-shot'
        )
    }
    $EvaluateCommand += @('--export-prototypes', '--prototype-dir', $PrototypeDir)
    if ($UpdateAnnotatedPrototypes) {
        $EvaluateCommand += @('--base-prototypes', (Join-Path (Split-Path -Parent $InitProjection) 'prototypes.pt'), '--snapshot-manifest', $ApprovedManifest)
    }
    Invoke-PipelineStep '[5/6] evaluate_export_prototypes' $EvaluateCommand
    $ExportCommand = @(
        $Python,
        (Join-Path $Pipeline 'export_model.py'),
        '--prototypes', $PrototypeFile,
        '--weights-dir', $WeightsDir,
        '--config-template', $ClassifierConfigTemplate,
        '--config-out', $ClassifierConfig,
        '--manifest-out', $ExportManifest,
        '--registry-dir', $ModelRegistryDir,
        '--version-id', $ModelVersionId,
        '--metadata-csv', $MetadataCsv,
        '--approved-manifest', $ApprovedManifest,
        '--training-config', $Config,
        '--features', $FeaturesFile,
        '--features-provenance', ($FeaturesFile + '.prov.json'),
        '--checkpoint', $SelectedCheckpoint,
        '--training-manifest', (Join-Path $CheckpointDir 'training_manifest.json'),
        '--checkpoint-selection', $CheckpointSelection
    )
    if ($InitProjection) {
        $ExportCommand += @('--init-projection', $InitProjection)
        $ExportCommand += @('--base-model-dir', (Split-Path -Parent (Split-Path -Parent $InitProjection)))
    }
    if ($UpdateAnnotatedPrototypes) {
        $ExportCommand += '--evaluate-candidate'
    }
    Invoke-PipelineStep '[6/6] export_model' $ExportCommand
    Write-Step "=== Candidate model version created: $ModelVersionId ==="
    Write-Step "=== Inspect: $VersionDir ==="
    Write-Step "=== Promote explicitly: $Python scripts/promote_model.py $ModelVersionId --dry-run ==="
} finally {
    if ($OwnsLock -and (Test-Path -LiteralPath $LockFile)) {
        if ((Get-Content -LiteralPath $LockFile -Raw).Trim() -eq [string]$PID) {
            Remove-Item -LiteralPath $LockFile -Force -ErrorAction SilentlyContinue
        }
    }
    if ($null -ne $Guard) { $Guard.Dispose() }
}

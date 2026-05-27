<#
.SYNOPSIS
Run approved-only classifier retraining with repo-root anchored explicit paths.

.DESCRIPTION
PowerShell counterpart to scripts/retrain.sh. The pipeline exports only
admin-approved annotation crops, builds metadata from that generated dataset,
precomputes features, trains the projection/classifier checkpoint, exports
prototypes, and writes backend-loadable immutable candidate artifacts under
backend/model_registry/versions/<id>.

No segmentation/MobileSAM retraining is performed.
No runtime backend/codex_model artifacts are modified; run scripts/promote_model.py
to activate a candidate.
#>
[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'

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
$Device = if ($env:DEVICE) { $env:DEVICE } else { 'auto' }

function Write-Step([string]$Message) {
    Write-Host $Message
}

function Get-PythonPath {
    $candidates = @(
        (Join-Path $RepoRoot 'backend/.venv/Scripts/python.exe'),
        (Join-Path $RepoRoot 'backend/.venv/bin/python3'),
        (Join-Path $RepoRoot 'backend/.venv/bin/python'),
        'python'
    )

    foreach ($candidate in $candidates) {
        try {
            if ((Test-Path $candidate) -or (Get-Command $candidate -ErrorAction SilentlyContinue)) {
                return $candidate
            }
        } catch {}
    }

    throw 'Python not found. Run scripts/install.ps1 first or ensure python is on PATH.'
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
    if ($PSCmdlet.ShouldProcess($CommandParts[1], 'run approved-only classifier retraining step')) {
        & $CommandParts[0] @($CommandParts[1..($CommandParts.Count - 1)])
        if ($LASTEXITCODE -ne 0) {
            throw "Step failed: $Label"
        }
    }
}

Set-Location $RepoRoot
$Python = Get-PythonPath

if (-not $DryRun -and -not $WhatIfPreference) {
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
}

try {
    Invoke-PipelineStep '[1/6] export_approved_annotations' @(
        $Python,
        (Join-Path $RepoRoot 'scripts/export_approved_annotations.py'),
        '--annotations-dir', $AnnotationsDir,
        '--output', $ElementsDir
    )
    Invoke-PipelineStep '[2/6] build_metadata' @(
        $Python,
        (Join-Path $Pipeline 'build_metadata.py'),
        '--elements-dir', $ElementsDir,
        '--output', $MetadataCsv
    )
    Invoke-PipelineStep '[3/6] precompute_embeddings' @(
        $Python,
        (Join-Path $Pipeline 'precompute_embeddings.py'),
        '--config', $Config,
        '--metadata-csv', $MetadataCsv,
        '--batch-size', $BatchSize,
        '--device', $Device,
        '--output-dir', $PrecomputedDir
    )
    Invoke-PipelineStep '[4/6] train' @(
        $Python,
        (Join-Path $Pipeline 'train.py'),
        '--config', $Config,
        '--features', $FeaturesFile,
        '--checkpoint-dir', $CheckpointDir
    )
    Invoke-PipelineStep '[5/6] evaluate_export_prototypes' @(
        $Python,
        (Join-Path $Pipeline 'evaluate.py'),
        '--checkpoint', (Join-Path $CheckpointDir 'best.pt'),
        '--features', $FeaturesFile,
        '--export-prototypes',
        '--prototype-dir', $PrototypeDir
    )
    Invoke-PipelineStep '[6/6] export_model' @(
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
        '--approved-manifest', (Join-Path $ElementsDir '_approved_export_manifest.json')
    )
    Write-Step "=== Candidate model version created: $ModelVersionId ==="
    Write-Step "=== Inspect: $VersionDir ==="
    Write-Step "=== Promote explicitly: $Python scripts/promote_model.py $ModelVersionId --dry-run ==="
} finally {
    if (-not $DryRun -and -not $WhatIfPreference -and (Test-Path $LockFile)) {
        Remove-Item $LockFile -Force -ErrorAction SilentlyContinue
    }
}

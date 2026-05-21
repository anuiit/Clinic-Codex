<#
.SYNOPSIS
Run the full retraining pipeline with a lockfile guard.

.DESCRIPTION
PowerShell counterpart to scripts/retrain.sh. It runs build_metadata,
precompute_embeddings, train, and export_model from backend/codex_pipeline/scripts.
Use -WhatIf to print the planned steps without running the pipeline.
#>
[CmdletBinding(SupportsShouldProcess = $true)]
param()

$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
$LockFile = Join-Path $RepoRoot 'backend/.retrain.lock'
$Pipeline = Join-Path $RepoRoot 'backend/codex_pipeline/scripts'

function Write-Step([string]$Message) {
    Write-Host $Message
}

function Get-PythonPath {
    $candidates = @(
        (Join-Path $RepoRoot 'backend/.venv/Scripts/python.exe'),
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

function Invoke-PipelineStep([string]$Label, [string]$ScriptName) {
    $scriptPath = Join-Path $Pipeline $ScriptName
    Write-Step "=== $Label ==="
    if ($PSCmdlet.ShouldProcess($scriptPath, 'run retraining step')) {
        & $Python $scriptPath
        if ($LASTEXITCODE -ne 0) {
            throw "Step failed: $ScriptName"
        }
    }
}

Set-Location $RepoRoot
$Python = Get-PythonPath

if (-not $WhatIfPreference) {
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
    Invoke-PipelineStep '[1/4] build_metadata' 'build_metadata.py'
    Invoke-PipelineStep '[2/4] precompute_embeddings' 'precompute_embeddings.py'
    Invoke-PipelineStep '[3/4] train' 'train.py'
    Invoke-PipelineStep '[4/4] export_model' 'export_model.py'
    Write-Step '=== Retraining complete ==='
} finally {
    if (-not $WhatIfPreference -and (Test-Path $LockFile)) {
        Remove-Item $LockFile -Force -ErrorAction SilentlyContinue
    }
}

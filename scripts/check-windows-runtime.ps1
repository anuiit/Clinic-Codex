<#
.SYNOPSIS
Runs the native Windows installer and runtime smoke, then classifies the result.

Classification contract:
- PASS: install.ps1 succeeds and run-dev.ps1 -Smoke confirms backend + frontend respond.
- EXPECTED BLOCKER: known prerequisite/project-artifact blocker that the script cannot fix.
- ENVIRONMENT FAILURE: host prerequisites or local resources are missing/unavailable.
- SCRIPT FAILURE: install/runtime scripts failed after prerequisites appeared satisfiable.
#>
param(
    [int]$SmokeTimeoutSeconds = 30
)

$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
Set-Location $RepoRoot

function Write-Classification([string]$status, [string]$reason) {
    Write-Host "WINDOWS_RUNTIME_RESULT=$status"
    Write-Host "WINDOWS_RUNTIME_REASON=$reason"
}

function Invoke-Captured([scriptblock]$Command) {
    $output = & $Command 2>&1 | ForEach-Object { "$PSItem" }
    $exit = if ($null -eq $LASTEXITCODE) { 0 } else { $LASTEXITCODE }
    [pscustomobject]@{ ExitCode = $exit; Output = ($output -join "`n") }
}

function Classify-Failure([string]$phase, [int]$exitCode, [string]$output) {
    if ($output -match 'Model artefacts missing|prototypes\.pt not found') {
        Write-Classification 'EXPECTED BLOCKER' "$phase blocked by missing bundled model artefacts."
        return 2
    }
    if ($output -match 'Need Python|Python reports version|Need >=2GB free disk|npm install failed|npm(.cmd)?\s*:.*not recognized|Port \d+ already in use|BACKEND_PORT must be|FRONTEND_PORT must be') {
        Write-Classification 'ENVIRONMENT FAILURE' "$phase failed because the host prerequisites/resources are not ready."
        return 3
    }
    if ($phase -eq 'runtime smoke' -and $output -match 'backend/.venv missing|venv incomplete|frontend/node_modules missing') {
        Write-Classification 'SCRIPT FAILURE' 'install completed but runtime pre-flight still found missing dependencies.'
        return 4
    }
    Write-Classification 'SCRIPT FAILURE' "$phase failed with exit code $exitCode."
    return 4
}

$IsNativeWindows = ($IsWindows -or $PSVersionTable.PSEdition -eq 'Desktop')
if (-not $IsNativeWindows) {
    Write-Classification 'EXPECTED BLOCKER' 'native Windows PowerShell is required; use bash scripts on Linux/macOS.'
    exit 2
}

$install = Invoke-Captured { & (Join-Path $ScriptDir 'install.ps1') }
Write-Host $install.Output
if ($install.ExitCode -ne 0) {
    exit (Classify-Failure 'install' $install.ExitCode $install.Output)
}

$smoke = Invoke-Captured { & (Join-Path $ScriptDir 'run-dev.ps1') -Smoke -SmokeTimeoutSeconds $SmokeTimeoutSeconds }
Write-Host $smoke.Output
if ($smoke.ExitCode -ne 0) {
    exit (Classify-Failure 'runtime smoke' $smoke.ExitCode $smoke.Output)
}

Write-Classification 'PASS' 'install succeeded and runtime smoke confirmed backend and frontend readiness.'
exit 0

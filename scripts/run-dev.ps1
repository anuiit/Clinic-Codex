#requires -Version 5.1
<#
.SYNOPSIS
Clinic Codex dev launcher for native Windows PowerShell 5.1 or PowerShell 7+ - backend on :7117, frontend on :7118.
#>
param(
    [switch]$Smoke,
    [int]$SmokeTimeoutSeconds = 30
)

$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot  = Split-Path -Parent $ScriptDir
Set-Location $RepoRoot

function Log([string]$msg)  { Write-Host "[run-dev] $msg" }
function Fail([string]$msg) { Write-Host "[run-dev] ERROR: $msg" -ForegroundColor Red; exit 1 }
function Join-PathParts([string]$Base, [Parameter(ValueFromRemainingArguments = $true)][string[]]$ChildPath) {
    $path = $Base
    foreach ($child in $ChildPath) {
        $path = Join-Path $path $child
    }
    return $path
}
function Restore-Env([string]$name, $value) {
    if ($null -eq $value) {
        Remove-Item "Env:$name" -ErrorAction SilentlyContinue
    } else {
        Set-Item "Env:$name" -Value $value
    }
}
function Test-PortNumber([string]$port) {
    $number = 0
    return [int]::TryParse($port, [ref]$number) -and $number -ge 1 -and $number -le 65535
}
function Import-BackendEnv([string]$path) {
    if (-not (Test-Path $path)) { return }
    Log "loading backend env from backend/.env"
    foreach ($line in Get-Content $path) {
        $trimmed = $line.Trim()
        if ($trimmed.Length -eq 0 -or $trimmed.StartsWith('#') -or -not $trimmed.Contains('=')) { continue }
        $parts = $trimmed.Split('=', 2)
        $name = $parts[0].Trim()
        $value = $parts[1].Trim().Trim('"').Trim("'")
        if ($name -match '^[A-Za-z_][A-Za-z0-9_]*$' -and -not (Test-Path "Env:$name")) {
            Set-Item "Env:$name" -Value $value
        }
    }
}
function Test-AuthenticationEnabled {
    if (-not (Test-Path 'Env:AUTH_REQUIRED')) { return $true }
    return @('0', 'false', 'no', 'off', '') -notcontains $env:AUTH_REQUIRED.Trim().ToLowerInvariant()
}
function Stop-ProcessTree([int]$processId) {
    if ($processId -le 0) { return }
    if (Get-Command Get-CimInstance -ErrorAction SilentlyContinue) {
        $children = @(Get-CimInstance Win32_Process -Filter "ParentProcessId=$processId" -ErrorAction SilentlyContinue)
        foreach ($child in $children) {
            Stop-ProcessTree -processId ([int]$child.ProcessId)
        }
    }
    Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
}
function Assert-ProcessesRunning([hashtable]$processes) {
    foreach ($name in $processes.Keys) {
        $proc = $processes[$name]
        if ($null -ne $proc) {
            $proc.Refresh()
            if ($proc.HasExited) {
                Fail "$name process exited before services became ready - see logs above"
            }
        }
    }
}
function Wait-HttpReady([string]$label, [string]$uri, [hashtable]$processes, [int]$timeoutSeconds = 30) {
    for ($i = 1; $i -le $timeoutSeconds; $i++) {
        Start-Sleep -Seconds 1
        try {
            $response = Invoke-WebRequest -Uri $uri -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
            if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500) {
                Log "$label ready: $uri"
                return
            }
        } catch {}
        Assert-ProcessesRunning $processes
    }
    Fail "$label did not start within ${timeoutSeconds}s - see logs above"
}

$IsNativeWindows = ($PSVersionTable.PSEdition -eq 'Desktop') -or
    ($PSVersionTable.Platform -eq 'Win32NT') -or
    ($env:OS -eq 'Windows_NT')
if (-not $IsNativeWindows) {
    Fail 'scripts/run-dev.ps1 is intended for native Windows PowerShell. Use bash scripts/run-dev.sh on Linux/macOS.'
}

$RepoProviderPath = (Resolve-Path -LiteralPath $RepoRoot -ErrorAction Stop).ProviderPath
if ($RepoProviderPath.StartsWith('\\')) {
    Fail 'Windows PowerShell cannot launch npm reliably inside a UNC/WSL path. Open WSL and run: bash scripts/run-dev.sh, or clone the repository to a local Windows drive.'
}

Import-BackendEnv (Join-PathParts $RepoRoot 'backend' '.env')
if ((Test-AuthenticationEnabled) -and [string]::IsNullOrWhiteSpace($env:AUTH_SECRET_KEY)) {
    Fail 'Authentication is enabled but AUTH_SECRET_KEY is missing. Run scripts/install.ps1 to create backend/.env, or add a long random AUTH_SECRET_KEY to the existing file.'
}
if (-not (Test-Path 'Env:AUTH_COOKIE_SECURE')) {
    Log 'AUTH_COOKIE_SECURE is not configured - using false for this loopback HTTP development session'
}

# --- Pre-flight: venv ---
$VenvDir = Join-PathParts $RepoRoot 'backend' '.venv'
$VenvPy  = Join-PathParts $VenvDir 'Scripts' 'python.exe'
if (-not (Test-Path $VenvPy)) {
    Fail "backend/.venv missing. Run: powershell -ExecutionPolicy Bypass -File .\scripts\install.ps1"
}
& $VenvPy -c "import flask, torch, mobile_sam" 2>$null
if ($LASTEXITCODE -ne 0) { Fail "venv incomplete. Run: powershell -ExecutionPolicy Bypass -File .\scripts\install.ps1" }

# --- Pre-flight: frontend deps ---
$NodeModules = Join-PathParts $RepoRoot 'frontend' 'node_modules'
if (-not (Test-Path $NodeModules)) {
    Fail "frontend/node_modules missing. Run: powershell -ExecutionPolicy Bypass -File .\scripts\install.ps1"
}
$NpmCommand = Get-Command npm.cmd -ErrorAction SilentlyContinue
if ($null -eq $NpmCommand) {
    Fail 'npm.cmd was not found on PATH. Install Node.js 22.22 or newer, reopen PowerShell, then retry.'
}
$NodeCommand = Get-Command node.exe -ErrorAction SilentlyContinue
if ($null -eq $NodeCommand) {
    Fail 'node.exe was not found on PATH. Install Node.js 22.22 or newer, reopen PowerShell, then retry.'
}
try {
    $NodeVersion = [version]((& $NodeCommand.Source --version 2>&1).ToString().Trim().TrimStart('v'))
} catch {
    Fail "Unable to read the Node.js version: $($_.Exception.Message)"
}
if ($NodeVersion -lt [version]'22.22.0') {
    Fail "Node.js 22.22 or newer is required. Found $NodeVersion."
}

# --- Pre-flight: prototypes (auto-export if missing) ---
$ProtoDerived = Join-PathParts $RepoRoot 'backend' 'codex_model' 'weights' 'prototypes.pt'
$ProtoSource  = Join-PathParts $RepoRoot 'backend' 'prototypes' 'prototypes.pt'
if (-not (Test-Path $ProtoDerived)) {
    if (-not (Test-Path $ProtoSource)) {
        Fail "Model artefacts missing: backend\prototypes\prototypes.pt not found. See backend\README.md."
    }
    Log "prototypes.pt missing - running bootstrap export_model with explicit runtime-write opt-in"
    Push-Location (Join-Path $RepoRoot 'backend')
    & $VenvPy -m codex_pipeline.scripts.export_model `
        --allow-runtime-write `
        --prototypes prototypes/prototypes.pt `
        --weights-dir codex_model/weights `
        --config-template codex_model/config.json `
        --config-out codex_model/config.json
    $exportExit = $LASTEXITCODE
    Pop-Location
    if ($exportExit -ne 0) { Fail "export_model failed - see error above" }
    if (-not (Test-Path $ProtoDerived)) { Fail "export_model ran but prototypes.pt still missing" }
}

# --- Pre-flight: ports ---
$BackendPort  = if ($env:BACKEND_PORT)  { $env:BACKEND_PORT }  else { '7117' }
$FrontendPort = if ($env:FRONTEND_PORT) { $env:FRONTEND_PORT } else { '7118' }
if (-not (Test-PortNumber $BackendPort)) { Fail "BACKEND_PORT must be a TCP port number (1-65535), got: $BackendPort" }
if (-not (Test-PortNumber $FrontendPort)) { Fail "FRONTEND_PORT must be a TCP port number (1-65535), got: $FrontendPort" }
foreach ($port in @($BackendPort, $FrontendPort)) {
    $inUse = $false
    try {
        $tcp = New-Object Net.Sockets.TcpClient
        $tcp.Connect('127.0.0.1', [int]$port)
        $tcp.Close()
        $inUse = $true
    } catch {}
    if ($inUse) { Fail "Port $port already in use. Stop other process or override BACKEND_PORT/FRONTEND_PORT env." }
}
$BackendUrl = "http://localhost:$BackendPort"
$FrontendUrl = "http://localhost:$FrontendPort"
$BackendCorsOrigins = if ($env:CORS_ORIGINS) { $env:CORS_ORIGINS } else { "$FrontendUrl,http://127.0.0.1:$FrontendPort" }
$FrontendApiBaseUrl = if ($env:VITE_API_BASE_URL) { $env:VITE_API_BASE_URL } else { $BackendUrl }

$backendProc = $null
$frontendProc = $null
$processes = @{}

try {
    # --- Launch backend ---
    Log "starting backend on :$BackendPort"
    $oldPort = $env:PORT
    $oldCorsOrigins = $env:CORS_ORIGINS
    $oldAuthCookieSecure = $env:AUTH_COOKIE_SECURE
    try {
        $env:PORT = $BackendPort
        $env:CORS_ORIGINS = $BackendCorsOrigins
        if (-not (Test-Path 'Env:AUTH_COOKIE_SECURE')) { $env:AUTH_COOKIE_SECURE = 'false' }
        $backendProc = Start-Process -FilePath $VenvPy `
            -ArgumentList @('-m', 'flask', '--app', 'backend.wsgi', 'run', '--host', '127.0.0.1', '--port', $BackendPort) `
            -PassThru -NoNewWindow
    } finally {
        Restore-Env 'PORT' $oldPort
        Restore-Env 'CORS_ORIGINS' $oldCorsOrigins
        Restore-Env 'AUTH_COOKIE_SECURE' $oldAuthCookieSecure
    }
    $processes['backend'] = $backendProc

    # --- Launch frontend ---
    Log "starting frontend on :$FrontendPort"
    $oldViteApiBaseUrl = $env:VITE_API_BASE_URL
    try {
        $env:VITE_API_BASE_URL = $FrontendApiBaseUrl
        $frontendProc = Start-Process -FilePath $NpmCommand.Source `
            -ArgumentList @('run', 'dev', '--', '--host', '127.0.0.1', '--port', $FrontendPort, '--strictPort') `
            -WorkingDirectory (Join-Path $RepoRoot 'frontend') `
            -PassThru -NoNewWindow
    } finally {
        Restore-Env 'VITE_API_BASE_URL' $oldViteApiBaseUrl
    }
    $processes['frontend'] = $frontendProc

    # --- Wait for services ready ---
    Wait-HttpReady 'backend' "http://127.0.0.1:$BackendPort/classes" $processes $SmokeTimeoutSeconds
    Wait-HttpReady 'frontend' "http://127.0.0.1:$FrontendPort/" $processes $SmokeTimeoutSeconds

    if ($Smoke) {
        Write-Host ""
        Write-Host "[run-dev] smoke PASS: backend and frontend responded successfully."
        return
    }

    Write-Host ""
    Write-Host "[run-dev] both services up."
    Write-Host "[run-dev]   backend:  http://localhost:$BackendPort"
    Write-Host "[run-dev]   frontend: http://localhost:$FrontendPort"
    Write-Host "[run-dev] Ctrl-C to stop."
    Write-Host ""

    Wait-Process -Id @($backendProc.Id, $frontendProc.Id)
} finally {
    Log "shutting down"
    if ($null -ne $backendProc) { Stop-ProcessTree -processId $backendProc.Id }
    if ($null -ne $frontendProc) { Stop-ProcessTree -processId $frontendProc.Id }
}

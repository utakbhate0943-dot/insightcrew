<#
run-dev.ps1

PowerShell helper to load env vars from `.env`, pick a virtual environment
(prefers `.venv311` then `.venv`), and run the Streamlit app.

Usage:
  .\run-dev.ps1                # reads .env, uses .venv311 or .venv, runs Streamlit
  .\run-dev.ps1 -LogFile out.txt
  .\run-dev.ps1 -VenvName .venv

Security:
  - Do NOT commit `.env` with real secrets. This script reads `.env` in the
    project folder and sets them in the current process for the Streamlit run.
#>

param(
    [string]$LogFile = "streamlit_runtime_log.txt",
    [string]$VenvName = "",
    [switch]$NoTee
)

function Load-EnvFile {
    param([string]$Path)
    if (-not (Test-Path $Path)) { return @{} }
    $pairs = @{}
    Get-Content $Path | ForEach-Object {
        $_ = $_.Trim()
        if ([string]::IsNullOrWhiteSpace($_)) { return }
        if ($_.StartsWith('#')) { return }
        $parts = $_ -split '=', 2
        if ($parts.Count -ne 2) { return }
        $key = $parts[0].Trim()
        $val = $parts[1].Trim()
        # Remove optional surrounding quotes
        if ($val.StartsWith('"') -and $val.EndsWith('"')) { $val = $val.Substring(1, $val.Length - 2) }
        if ($val.StartsWith("'") -and $val.EndsWith("'")) { $val = $val.Substring(1, $val.Length - 2) }
        $pairs[$key] = $val
    }
    return $pairs
}

Write-Host "[run-dev] Project folder: $PSScriptRoot"

# Load .env if present
$envPath = Join-Path $PSScriptRoot '.env'
if (Test-Path $envPath) {
    Write-Host "[run-dev] Loading environment variables from .env"
    $envPairs = Load-EnvFile -Path $envPath
    foreach ($k in $envPairs.Keys) {
        $v = $envPairs[$k]
        # Mask sensitive-looking values in output
        if ($k.ToUpper().Contains('KEY') -or $k.ToUpper().Contains('PASSWORD') -or $k.ToUpper().Contains('SECRET')) {
            Write-Host "  - $k = ***"
        } else {
            Write-Host "  - $k = $v"
        }
        $env:$k = $v
    }
} else {
    Write-Host "[run-dev] No .env found. Make sure required env vars are set in the session."
}

# Determine virtualenv to use
if ($VenvName -ne '') {
    $venvPath = Join-Path $PSScriptRoot $VenvName
} else {
    if (Test-Path (Join-Path $PSScriptRoot '.venv311')) { $venvPath = Join-Path $PSScriptRoot '.venv311' }
    elseif (Test-Path (Join-Path $PSScriptRoot '.venv')) { $venvPath = Join-Path $PSScriptRoot '.venv' }
    else { $venvPath = '' }
}

if ($venvPath -and (Test-Path $venvPath)) {
    Write-Host "[run-dev] Using virtual environment: $venvPath"
    $pythonExe = Join-Path $venvPath 'Scripts\python.exe'
    if (-not (Test-Path $pythonExe)) {
        Write-Host "[run-dev] Warning: python.exe not found in $venvPath\Scripts. Falling back to system python."
        $pythonExe = 'python'
    }
} else {
    Write-Host "[run-dev] No virtual environment found. Using system python."
    $pythonExe = 'python'
}

# Run Streamlit
$scriptPath = Join-Path $PSScriptRoot 'app.py'
if (-not (Test-Path $scriptPath)) {
    Write-Error "app.py not found in project root ($PSScriptRoot). Run this script from the repository root."
    exit 1
}

$cmd = "$pythonExe -m streamlit run `"$scriptPath`""
Write-Host "[run-dev] Running: $cmd"

if ($NoTee) {
    & $pythonExe -m streamlit run $scriptPath
} else {
    # Send both stdout and stderr to tee so we capture logs while still seeing output live
    & $pythonExe -m streamlit run $scriptPath 2>&1 | Tee-Object -FilePath $LogFile
}

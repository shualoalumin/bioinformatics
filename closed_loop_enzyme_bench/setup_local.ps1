# Local setup script (PowerShell)
# Creates/uses a project-local virtual environment and installs dependencies.

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Local Setup (PowerShell)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

Write-Host "`n[1/3] Checking Python..." -ForegroundColor Yellow
try {
    $py = Get-Command python -ErrorAction Stop
    $ver = & python --version 2>&1
    Write-Host "OK: $ver" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Python not found in PATH." -ForegroundColor Red
    Write-Host "Install Python from: https://www.python.org/downloads/" -ForegroundColor White
    Write-Host "Make sure 'Add Python to PATH' is checked." -ForegroundColor White
    exit 1
}

Write-Host "`n[2/3] Creating virtual environment (.venv)..." -ForegroundColor Yellow
if (-not (Test-Path ".venv")) {
    & python -m venv .venv
}

Write-Host "Activating .venv..." -ForegroundColor Yellow
. .\.venv\Scripts\Activate.ps1

Write-Host "`n[3/3] Installing dependencies..." -ForegroundColor Yellow
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

Write-Host "`nRunning environment check..." -ForegroundColor Yellow
python check_environment.py

Write-Host "`nDone." -ForegroundColor Green
Write-Host "Next: python run_experiment_01.py" -ForegroundColor Cyan

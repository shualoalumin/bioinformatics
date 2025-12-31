# Quick Setup Script - Run after Python installation
# This script will set up the project environment

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Quick Project Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Check Python first
Write-Host "`n[1/4] Verifying Python installation..." -ForegroundColor Yellow
try {
    $version = & python --version 2>&1
    if ($LASTEXITCODE -eq 0 -or $version -match 'Python') {
        Write-Host "OK: $version" -ForegroundColor Green
    } else {
        Write-Host "X: Python not found. Please install Python first." -ForegroundColor Red
        Write-Host "   Run: .\install_python.ps1" -ForegroundColor Yellow
        exit 1
    }
} catch {
    Write-Host "X: Python not found. Please install Python first." -ForegroundColor Red
    exit 1
}

# Upgrade pip
Write-Host "`n[2/4] Upgrading pip..." -ForegroundColor Yellow
try {
    & python -m pip install --upgrade pip --quiet
    Write-Host "OK: pip upgraded" -ForegroundColor Green
} catch {
    Write-Host "Warning: pip upgrade failed" -ForegroundColor Yellow
}

# Install packages
Write-Host "`n[3/4] Installing project packages..." -ForegroundColor Yellow
Write-Host "  (This may take a few minutes...)" -ForegroundColor Gray

if (Test-Path "requirements.txt") {
    try {
        & python -m pip install -r requirements.txt
        if ($LASTEXITCODE -eq 0) {
            Write-Host "OK: Packages installed" -ForegroundColor Green
        } else {
            Write-Host "Warning: Some packages may have failed to install" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "Error: Package installation failed" -ForegroundColor Red
        Write-Host "  Try manually: python -m pip install -r requirements.txt" -ForegroundColor Yellow
    }
} else {
    Write-Host "X: requirements.txt not found" -ForegroundColor Red
}

# Check environment
Write-Host "`n[4/4] Checking environment..." -ForegroundColor Yellow
if (Test-Path "check_environment.py") {
    & python check_environment.py
} else {
    Write-Host "Warning: check_environment.py not found" -ForegroundColor Yellow
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Setup complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan

Write-Host "`nYou can now run experiments:" -ForegroundColor Yellow
Write-Host "  python run_experiment_01.py" -ForegroundColor Cyan
Write-Host "  python start_experiment.py" -ForegroundColor Cyan

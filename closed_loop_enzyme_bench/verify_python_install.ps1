# Verify Python Installation Script
# Run this after installing Python

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Python Installation Verification" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

Write-Host "`nChecking Python installation..." -ForegroundColor Yellow

# Check Python
$pythonFound = $false
$pythonVersion = $null

$commands = @('python', 'python3', 'py')
foreach ($cmd in $commands) {
    try {
        $version = & $cmd --version 2>&1
        if ($LASTEXITCODE -eq 0 -or $version -match 'Python') {
            Write-Host "OK: Python found!" -ForegroundColor Green
            Write-Host "  Command: $cmd" -ForegroundColor White
            Write-Host "  Version: $version" -ForegroundColor White
            $pythonFound = $true
            $pythonVersion = $version
            break
        }
    } catch {
        continue
    }
}

if (-not $pythonFound) {
    Write-Host "X: Python not found in PATH" -ForegroundColor Red
    Write-Host "`nPlease make sure:" -ForegroundColor Yellow
    Write-Host "  1. Python installation completed" -ForegroundColor White
    Write-Host "  2. 'Add Python to PATH' was checked during installation" -ForegroundColor White
    Write-Host "  3. PowerShell was restarted after installation" -ForegroundColor White
    Write-Host "  4. Or restart your computer" -ForegroundColor White
    exit 1
}

# Check pip
Write-Host "`nChecking pip..." -ForegroundColor Yellow
try {
    $pipVersion = & python -m pip --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "OK: pip found" -ForegroundColor Green
        Write-Host "  $pipVersion" -ForegroundColor White
    } else {
        Write-Host "Warning: pip check failed" -ForegroundColor Yellow
    }
} catch {
    Write-Host "Warning: pip check failed" -ForegroundColor Yellow
}

# Check Python executable path
Write-Host "`nPython Details:" -ForegroundColor Yellow
try {
    $pythonPath = & python -c "import sys; print(sys.executable)" 2>&1
    Write-Host "  Executable: $pythonPath" -ForegroundColor White
} catch {
    Write-Host "  Could not get executable path" -ForegroundColor Yellow
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Installation verified!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan

Write-Host "`nNext steps:" -ForegroundColor Yellow
Write-Host "  1. Upgrade pip:" -ForegroundColor White
Write-Host "     python -m pip install --upgrade pip" -ForegroundColor Cyan
Write-Host "  2. Install project packages:" -ForegroundColor White
Write-Host "     python -m pip install -r requirements.txt" -ForegroundColor Cyan
Write-Host "  3. Check environment:" -ForegroundColor White
Write-Host "     python check_environment.py" -ForegroundColor Cyan

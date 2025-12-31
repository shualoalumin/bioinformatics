# Python Installation Guide Script
# Check if Python is installed and provide installation instructions

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Python Installation Guide" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

Write-Host "`n[Step 1] Checking for Python installation..." -ForegroundColor Yellow

# Check for Python
$pythonFound = $false
$pythonVersion = $null

$commands = @('python', 'python3', 'py')
foreach ($cmd in $commands) {
    try {
        $result = Get-Command $cmd -ErrorAction Stop 2>$null
        $version = & $cmd --version 2>&1
        if ($LASTEXITCODE -eq 0 -or $version -match 'Python') {
            Write-Host "OK: Python is already installed!" -ForegroundColor Green
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
    Write-Host "X: Python is not installed." -ForegroundColor Red
    Write-Host "`n[Step 2] Python Installation Instructions:" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Method 1: Official Python Website (Recommended)" -ForegroundColor Cyan
    Write-Host "  1. Open browser and go to:" -ForegroundColor White
    Write-Host "     https://www.python.org/downloads/" -ForegroundColor Yellow
    Write-Host "  2. Click 'Download Python 3.12.x' button" -ForegroundColor White
    Write-Host "  3. Run the downloaded .exe file" -ForegroundColor White
    Write-Host "  4. IMPORTANT: Check this option:" -ForegroundColor White
    Write-Host "     [X] Add Python to PATH" -ForegroundColor Green
    Write-Host "  5. Click 'Install Now'" -ForegroundColor White
    Write-Host ""
    Write-Host "Method 2: Microsoft Store" -ForegroundColor Cyan
    Write-Host "  1. Open Microsoft Store" -ForegroundColor White
    Write-Host "  2. Search for 'Python 3.11' or 'Python 3.12'" -ForegroundColor White
    Write-Host "  3. Click Install" -ForegroundColor White
    Write-Host ""
    Write-Host "After installation, restart PowerShell and run this script again." -ForegroundColor Yellow
    Write-Host ""
    
    # Open browser to Python download page
    $openBrowser = Read-Host "Open Python download page in browser? (y/n)"
    if ($openBrowser -eq 'y' -or $openBrowser -eq 'Y') {
        Start-Process "https://www.python.org/downloads/"
        Write-Host "`nBrowser opened. Please download and install Python." -ForegroundColor Green
        Write-Host "After installation, restart PowerShell and run this script again." -ForegroundColor Yellow
    }
    exit 1
}

# Python is installed
Write-Host "`n[Step 3] Checking pip..." -ForegroundColor Yellow
try {
    $pipVersion = & python -m pip --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "OK: pip found: $pipVersion" -ForegroundColor Green
    } else {
        Write-Host "Warning: pip check failed" -ForegroundColor Yellow
    }
} catch {
    Write-Host "Warning: pip check failed" -ForegroundColor Yellow
}

Write-Host "`n[Step 4] Next Steps:" -ForegroundColor Yellow
Write-Host "  1. Upgrade pip:" -ForegroundColor White
Write-Host "     python -m pip install --upgrade pip" -ForegroundColor Cyan
Write-Host "  2. Install project packages:" -ForegroundColor White
Write-Host "     python -m pip install -r requirements.txt" -ForegroundColor Cyan
Write-Host "  3. Check environment:" -ForegroundColor White
Write-Host "     python check_environment.py" -ForegroundColor Cyan
Write-Host "  4. Start experiment:" -ForegroundColor White
Write-Host "     python run_experiment_01.py" -ForegroundColor Cyan

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Python installation check complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan

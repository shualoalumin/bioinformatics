# Remove Microsoft Store Python and guide for official Python installation

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Microsoft Store Python Removal" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

Write-Host "`n[Step 1] Checking for Microsoft Store Python..." -ForegroundColor Yellow

# Check for Microsoft Store Python
$storePython = Get-AppxPackage -Name "*Python*" -ErrorAction SilentlyContinue

if ($storePython) {
    Write-Host "Found Microsoft Store Python packages:" -ForegroundColor Yellow
    $storePython | ForEach-Object {
        Write-Host "  - $($_.Name) (Version: $($_.Version))" -ForegroundColor White
    }
    
    Write-Host "`n[Step 2] Removing Microsoft Store Python..." -ForegroundColor Yellow
    Write-Host "This requires administrator privileges." -ForegroundColor Yellow
    Write-Host ""
    
    # Check if running as administrator
    $isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
    
    if (-not $isAdmin) {
        Write-Host "Warning: Not running as administrator." -ForegroundColor Yellow
        Write-Host "You can either:" -ForegroundColor White
        Write-Host "  1. Run PowerShell as Administrator and run this script again" -ForegroundColor Cyan
        Write-Host "  2. Remove manually from Settings > Apps > Python" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "To run as Administrator:" -ForegroundColor Yellow
        Write-Host "  1. Right-click PowerShell" -ForegroundColor White
        Write-Host "  2. Select 'Run as Administrator'" -ForegroundColor White
        Write-Host "  3. Navigate to this directory and run this script" -ForegroundColor White
    } else {
        Write-Host "Running as Administrator. Proceeding with removal..." -ForegroundColor Green
        
        foreach ($package in $storePython) {
            try {
                Write-Host "  Removing $($package.Name)..." -ForegroundColor White
                Remove-AppxPackage -Package $package.PackageFullName -ErrorAction Stop
                Write-Host "    OK: Removed successfully" -ForegroundColor Green
            } catch {
                Write-Host "    Error: Failed to remove $($package.Name)" -ForegroundColor Red
                Write-Host "    Error: $_" -ForegroundColor Red
            }
        }
        
        Write-Host "`nMicrosoft Store Python removal completed." -ForegroundColor Green
    }
} else {
    Write-Host "No Microsoft Store Python packages found." -ForegroundColor Green
    Write-Host "You can proceed with official Python installation." -ForegroundColor Green
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Next Steps: Install Official Python" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

Write-Host "`n[Step 3] Install Official Python:" -ForegroundColor Yellow
Write-Host "  1. Open browser and go to:" -ForegroundColor White
Write-Host "     https://www.python.org/downloads/" -ForegroundColor Yellow
Write-Host "  2. Click 'Download Python 3.12.x' (or 3.11.x)" -ForegroundColor White
Write-Host "  3. Run the downloaded .exe file" -ForegroundColor White
Write-Host "  4. IMPORTANT: Check this option:" -ForegroundColor White
Write-Host "     [X] Add Python to PATH" -ForegroundColor Green
Write-Host "  5. Click 'Install Now'" -ForegroundColor White
Write-Host "  6. After installation, RESTART PowerShell" -ForegroundColor Yellow
Write-Host "  7. Verify installation:" -ForegroundColor White
Write-Host "     python --version" -ForegroundColor Cyan

Write-Host "`nWould you like to open the Python download page now?" -ForegroundColor Yellow
$openBrowser = Read-Host "Open browser? (y/n)"

if ($openBrowser -eq 'y' -or $openBrowser -eq 'Y') {
    Start-Process "https://www.python.org/downloads/"
    Write-Host "`nBrowser opened. Please download and install Python." -ForegroundColor Green
    Write-Host "After installation, restart PowerShell and verify with: python --version" -ForegroundColor Yellow
}

Write-Host "`n========================================" -ForegroundColor Cyan

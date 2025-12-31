# 로컬 환경 설정 스크립트 (PowerShell)
# 이 스크립트는 Python 설치 확인 및 패키지 설치를 도와줍니다.

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "로컬 환경 설정 스크립트" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# 1. Python 확인
Write-Host "`n[1/4] Python 설치 확인 중..." -ForegroundColor Yellow

$pythonFound = $false
$pythonCmd = $null

# 여러 방법으로 Python 찾기
$commands = @('python', 'python3', 'py')
foreach ($cmd in $commands) {
    try {
        $result = Get-Command $cmd -ErrorAction Stop
        $version = & $cmd --version 2>&1
        if ($LASTEXITCODE -eq 0 -or $version -match 'Python') {
            Write-Host "✓ Python 발견: $cmd" -ForegroundColor Green
            Write-Host "  버전: $version" -ForegroundColor Gray
            $pythonCmd = $cmd
            $pythonFound = $true
            break
        }
    } catch {
        continue
    }
}

if (-not $pythonFound) {
    Write-Host "✗ Python이 설치되어 있지 않거나 PATH에 없습니다." -ForegroundColor Red
    Write-Host "`nPython 설치 방법:" -ForegroundColor Yellow
    Write-Host "  1. https://www.python.org/downloads/ 방문" -ForegroundColor White
    Write-Host "  2. Python 3.10 이상 다운로드" -ForegroundColor White
    Write-Host "  3. 설치 시 'Add Python to PATH' 체크 필수!" -ForegroundColor White
    Write-Host "`n또는 Microsoft Store에서 'Python 3.11' 설치" -ForegroundColor White
    exit 1
}

# 2. pip 확인
Write-Host "`n[2/4] pip 확인 중..." -ForegroundColor Yellow
try {
    $pipVersion = & $pythonCmd -m pip --version 2>&1
    Write-Host "✓ pip 발견: $pipVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ pip가 없습니다. Python 재설치가 필요할 수 있습니다." -ForegroundColor Red
    exit 1
}

# 3. 패키지 설치
Write-Host "`n[3/4] 필수 패키지 설치 중..." -ForegroundColor Yellow
Write-Host "  (이 작업은 몇 분 걸릴 수 있습니다)" -ForegroundColor Gray

$requirementsFile = "requirements.txt"
if (Test-Path $requirementsFile) {
    Write-Host "  requirements.txt 파일 발견" -ForegroundColor Gray
    & $pythonCmd -m pip install --upgrade pip
    & $pythonCmd -m pip install -r $requirementsFile
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ 패키지 설치 완료" -ForegroundColor Green
    } else {
        Write-Host "⚠ 패키지 설치 중 일부 오류가 발생했습니다." -ForegroundColor Yellow
        Write-Host "  수동으로 설치해보세요: $pythonCmd -m pip install -r requirements.txt" -ForegroundColor White
    }
} else {
    Write-Host "✗ requirements.txt 파일을 찾을 수 없습니다." -ForegroundColor Red
    Write-Host "  현재 디렉토리: $(Get-Location)" -ForegroundColor Gray
}

# 4. 환경 확인
Write-Host "`n[4/4] 환경 확인 중..." -ForegroundColor Yellow
& $pythonCmd check_environment.py
if ($LASTEXITCODE -eq 0) {
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "✓ 환경 설정 완료!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "`n다음 단계:" -ForegroundColor Yellow
    Write-Host "  $pythonCmd run_experiment_01.py" -ForegroundColor White
    Write-Host "  $pythonCmd start_experiment.py" -ForegroundColor White
} else {
    Write-Host "`n⚠ 환경 확인 실패. 일부 패키지가 누락되었을 수 있습니다." -ForegroundColor Yellow
}

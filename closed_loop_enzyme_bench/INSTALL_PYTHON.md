# Python 설치 가이드 (Windows)

## 현재 상황 확인

먼저 현재 Python 설치 상태를 확인하세요:

```powershell
# Python 설치 경로 확인
$pythonPaths = @(
    "$env:LOCALAPPDATA\Programs\Python",
    "$env:PROGRAMFILES\Python*",
    "${env:ProgramFiles(x86)}\Python*"
)

Get-ChildItem -Path $pythonPaths -ErrorAction SilentlyContinue | 
    Where-Object { $_.Name -like "Python*" } |
    Select-Object FullName
```

## Python 설치 방법

### 1단계: Microsoft Store Python 제거 (있는 경우)

**방법 A: 설정에서 제거**
1. Windows 설정 열기 (Win + I)
2. "앱" 클릭
3. 검색창에 "Python" 입력
4. Python 앱 선택 → "제거" 클릭

**방법 B: PowerShell에서 제거**
```powershell
# 관리자 권한 PowerShell에서 실행
Get-AppxPackage *Python* | Remove-AppxPackage
```

### 2단계: 공식 Python 설치

1. **Python 다운로드**
   - https://www.python.org/downloads/ 방문
   - "Download Python 3.12.x" (또는 3.11.x) 클릭
   - 다운로드된 `.exe` 파일 실행

2. **설치 옵션 설정** (중요!)
   - ✅ **"Add Python to PATH"** 체크 필수!
   - ✅ "Install launcher for all users" (선택)
   - "Install Now" 클릭

3. **설치 완료 확인**
   - 새 PowerShell 창 열기 (기존 창 닫고 새로 열기)
   ```powershell
   python --version
   # 출력: Python 3.12.x (또는 3.11.x)
   ```

### 3단계: pip 업그레이드

```powershell
python -m pip install --upgrade pip
```

### 4단계: 프로젝트 패키지 설치

```powershell
# 프로젝트 디렉토리로 이동
cd "C:\Users\Mr.Josh\Documents\GitHub\bioinformatics\closed_loop_enzyme_bench"

# 패키지 설치
python -m pip install -r requirements.txt
```

### 5단계: 환경 확인

```powershell
python check_environment.py
```

## 문제 해결

### Python이 여전히 인식되지 않는 경우

**PATH 확인:**
```powershell
$env:PATH -split ';' | Where-Object { $_ -like '*python*' -or $_ -like '*Python*' }
```

**수동으로 PATH 추가:**
1. Windows 검색에서 "환경 변수" 검색
2. "시스템 환경 변수 편집" 선택
3. "환경 변수" 버튼 클릭
4. "Path" 선택 → "편집"
5. Python 설치 경로 추가 (예: `C:\Users\Mr.Josh\AppData\Local\Programs\Python\Python312`)
6. `Scripts` 폴더도 추가 (예: `C:\Users\Mr.Josh\AppData\Local\Programs\Python\Python312\Scripts`)

**또는 PowerShell에서 임시로 추가:**
```powershell
$pythonPath = "C:\Users\Mr.Josh\AppData\Local\Programs\Python\Python312"
$env:PATH += ";$pythonPath;$pythonPath\Scripts"
```

### 설치 후에도 작동하지 않는 경우

1. **PowerShell 재시작** (중요!)
2. **시스템 재부팅** (가장 확실함)
3. **Python 재설치** (PATH 체크 확인)

## 설치 확인 체크리스트

- [ ] Python 다운로드 완료
- [ ] 설치 시 "Add Python to PATH" 체크
- [ ] 새 PowerShell 창에서 `python --version` 작동
- [ ] `pip --version` 작동
- [ ] `python check_environment.py` 실행 성공

## 다음 단계

Python 설치가 완료되면:

```powershell
cd "C:\Users\Mr.Josh\Documents\GitHub\bioinformatics\closed_loop_enzyme_bench"
python run_experiment_01.py
```

## 대안: Colab 사용

로컬 설치가 복잡하다면 Google Colab 사용을 권장합니다:
- Python 자동 설치
- GPU 무료 제공
- 모든 패키지 자동 설치
- `colab/` 디렉토리의 노트북 사용

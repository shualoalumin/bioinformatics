# 로컬 실행 문제 해결 가이드 (Troubleshooting)

## 문제 진단

로컬에서 실험이 실행되지 않는 주요 원인과 해결 방법을 정리했습니다.

## 1. Python이 설치되지 않았거나 PATH에 없음

### 증상
```
'python'은(는) 내부 또는 외부 명령, 실행할 수 있는 프로그램, 또는 배치 파일이 아닙니다.
```

### 해결 방법

**Windows에서 Python 설치 확인:**
```powershell
# 방법 1: py 런처 사용
py --version

# 방법 2: python3 시도
python3 --version

# 방법 3: 전체 경로 확인
Get-Command python -ErrorAction SilentlyContinue
```

**Python 설치:**
1. [Python 공식 사이트](https://www.python.org/downloads/)에서 다운로드
2. 설치 시 **"Add Python to PATH"** 체크 필수!
3. 또는 Microsoft Store에서 "Python 3.11" 설치

**PATH 확인:**
```powershell
$env:PATH -split ';' | Select-String python
```

## 2. 필요한 패키지가 설치되지 않음

### 증상
```
ModuleNotFoundError: No module named 'torch'
ModuleNotFoundError: No module named 'transformers'
```

### 해결 방법

```powershell
cd closed_loop_enzyme_bench

# pip 업그레이드
py -m pip install --upgrade pip

# 의존성 설치
py -m pip install -r requirements.txt

# 또는 개별 설치
py -m pip install transformers>=4.35 accelerate torch biopython pandas numpy matplotlib tqdm scikit-learn pyyaml
```

**가상환경 사용 (권장):**
```powershell
# 가상환경 생성
py -m venv venv

# 가상환경 활성화
.\venv\Scripts\Activate.ps1

# 패키지 설치
pip install -r requirements.txt
```

## 3. CUDA/GPU 문제

### 증상
```
CUDA out of memory
RuntimeError: CUDA error
```

### 해결 방법

**CPU 모드로 실행:**
코드에서 `device="cpu"`로 변경:
```python
# esmfold_eval.py 또는 실행 스크립트에서
device = "cpu"  # "cuda" 대신
```

**GPU 확인:**
```python
import torch
print(torch.cuda.is_available())  # False면 CPU만 사용 가능
```

## 4. ProteinMPNN이 없음

### 증상
```
FileNotFoundError: ProteinMPNN script not found
```

### 해결 방법

**Option 1: ProteinMPNN 설치**
```powershell
git clone https://github.com/dauparas/ProteinMPNN.git
cd ProteinMPNN
pip install -r requirements.txt
cd ..
```

**Option 2: ProteinMPNN 없이 테스트**
```powershell
# 랜덤 mutation으로 테스트
py quick_test_02.py
```

## 5. 경로 문제

### 증상
```
FileNotFoundError: [Errno 2] No such file or directory
```

### 해결 방법

**올바른 디렉토리에서 실행:**
```powershell
# 프로젝트 루트로 이동
cd C:\Users\Mr.Josh\Documents\GitHub\bioinformatics\closed_loop_enzyme_bench

# 현재 경로 확인
pwd

# 스크립트 실행
py run_experiment_01.py
```

**상대 경로 문제:**
- 모든 스크립트는 `closed_loop_enzyme_bench/` 디렉토리에서 실행해야 함
- `src/` 모듈을 import하기 때문

## 6. 권한 문제 (Windows)

### 증상
```
PermissionError: [WinError 5] 액세스가 거부되었습니다
```

### 해결 방법

**PowerShell을 관리자 권한으로 실행:**
1. Windows 검색에서 "PowerShell" 검색
2. "관리자 권한으로 실행" 선택
3. 다시 시도

**또는 사용자 디렉토리 사용:**
```powershell
# 사용자 디렉토리로 이동
cd $env:USERPROFILE\Documents
```

## 7. Microsoft Store Python 문제

### 증상
Python이 Microsoft Store에서 설치되었지만 실행이 안 됨

### 해결 방법

**공식 Python 설치:**
1. Microsoft Store Python 제거
2. [python.org](https://www.python.org/downloads/)에서 공식 버전 설치
3. 설치 시 PATH 추가 확인

**또는 py 런처 사용:**
```powershell
# py 런처는 보통 작동함
py run_experiment_01.py
```

## 빠른 진단 스크립트

다음 스크립트로 환경을 확인하세요:

```powershell
# 환경 확인
Write-Host "=== Python 확인 ==="
py --version
python --version
python3 --version

Write-Host "`n=== pip 확인 ==="
py -m pip --version

Write-Host "`n=== 패키지 확인 ==="
py -c "import torch; print('PyTorch:', torch.__version__)"
py -c "import transformers; print('Transformers:', transformers.__version__)"
py -c "import Bio; print('BioPython:', Bio.__version__)"
```

## 권장 해결 방법

### 가장 확실한 방법: Colab 사용

로컬 환경 문제가 계속되면 **Google Colab 사용을 강력히 권장**합니다:

1. **장점:**
   - Python 자동 설치
   - GPU 무료 제공
   - 모든 패키지 자동 설치
   - ProteinMPNN 자동 클론

2. **사용법:**
   - `colab/` 디렉토리의 노트북 사용
   - 또는 `COLAB_QUICKSTART.md` 참고

### 로컬에서 꼭 실행해야 한다면

1. **가상환경 사용:**
```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. **py 런처 사용:**
```powershell
py run_experiment_01.py
```

3. **경로 확인:**
```powershell
# 반드시 closed_loop_enzyme_bench 디렉토리에서 실행
cd closed_loop_enzyme_bench
py check_environment.py
```

## 추가 도움

문제가 계속되면:
1. `check_environment.py` 실행 결과 공유
2. 에러 메시지 전체 내용 공유
3. Python 버전 및 OS 정보 공유

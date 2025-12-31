# 공식 Python 설치 가이드

## 현재 상태
✅ Microsoft Store Python이 없습니다. 공식 Python을 설치하면 됩니다.

## 설치 단계

### 1단계: Python 다운로드

1. **브라우저에서 Python 다운로드 페이지 열기**
   - https://www.python.org/downloads/
   - 또는 위 스크립트가 자동으로 열었을 수 있습니다.

2. **Python 버전 선택**
   - Python 3.12.x (최신 버전, 권장)
   - 또는 Python 3.11.x (안정 버전)

3. **다운로드**
   - "Download Python 3.12.x" 버튼 클릭
   - `.exe` 파일이 다운로드됩니다 (예: `python-3.12.0-amd64.exe`)

### 2단계: Python 설치

1. **다운로드된 파일 실행**
   - 다운로드 폴더에서 `.exe` 파일 더블클릭

2. **설치 옵션 설정** (매우 중요!)
   
   설치 화면에서:
   - ✅ **"Add Python to PATH"** 체크박스 반드시 체크!
   - ✅ "Install launcher for all users" (선택사항)
   - "Install Now" 클릭

   ⚠️ **경고**: "Add Python to PATH"를 체크하지 않으면 Python이 작동하지 않습니다!

3. **설치 완료 대기**
   - 설치가 완료될 때까지 기다립니다 (1-2분 소요)
   - "Setup was successful" 메시지 확인

### 3단계: 설치 확인

**중요**: PowerShell을 완전히 닫고 새로 열어야 합니다!

1. **현재 PowerShell 창 닫기**
2. **새 PowerShell 창 열기**
3. **다음 명령어 실행:**

```powershell
python --version
```

**예상 출력:**
```
Python 3.12.0
```

4. **pip 확인:**

```powershell
python -m pip --version
```

**예상 출력:**
```
pip 24.0 from C:\Users\Mr.Josh\AppData\Local\Programs\Python\Python312\lib\site-packages\pip (python 3.12)
```

### 4단계: 프로젝트 설정

Python이 정상 작동하면:

```powershell
# 1. 프로젝트 디렉토리로 이동
cd "C:\Users\Mr.Josh\Documents\GitHub\bioinformatics\closed_loop_enzyme_bench"

# 2. pip 업그레이드
python -m pip install --upgrade pip

# 3. 패키지 설치 (몇 분 걸릴 수 있음)
python -m pip install -r requirements.txt

# 4. 환경 확인
python check_environment.py

# 5. 실험 시작
python run_experiment_01.py
```

## 문제 해결

### Python이 여전히 인식되지 않는 경우

**방법 1: PowerShell 재시작**
- 모든 PowerShell 창을 닫고 새로 열기

**방법 2: 시스템 재부팅**
- 가장 확실한 방법

**방법 3: PATH 수동 확인**
```powershell
# PATH에 Python이 있는지 확인
$env:PATH -split ';' | Where-Object { $_ -like '*Python*' }
```

**방법 4: Python 재설치**
- 제어판 > 프로그램 제거에서 Python 제거
- 위 단계를 다시 따라 설치
- "Add Python to PATH" 반드시 체크!

## 설치 확인 체크리스트

- [ ] Python 다운로드 완료
- [ ] 설치 시 "Add Python to PATH" 체크
- [ ] PowerShell 재시작 (또는 시스템 재부팅)
- [ ] `python --version` 명령어 작동
- [ ] `python -m pip --version` 명령어 작동
- [ ] `python check_environment.py` 실행 성공

## 다음 단계

Python 설치가 완료되면:

```powershell
cd "C:\Users\Mr.Josh\Documents\GitHub\bioinformatics\closed_loop_enzyme_bench"
python run_experiment_01.py
```

## 참고

- Python 공식 사이트: https://www.python.org/
- Python 문서: https://docs.python.org/
- 설치 문제 해결: `TROUBLESHOOTING.md` 참고

# 실험 시작 가이드 (Setup Guide)

## 1. 환경 준비

### Python 설치 확인
```bash
python --version
# 또는
python3 --version
```

### 의존성 설치
```bash
cd closed_loop_enzyme_bench
pip install -r requirements.txt
```

또는 개별 설치:
```bash
pip install transformers>=4.35 accelerate torch biopython pandas numpy matplotlib tqdm scikit-learn pyyaml
```

### 환경 확인
```bash
python check_environment.py
```

## 2. 실험 시작

### 방법 1: Python 스크립트로 실행

**Experiment 01: Scaffold 다운로드**
```bash
python start_experiment.py
```

또는 직접:
```bash
python run_experiment_01.py
```

### 방법 2: Colab 노트북 사용 (권장)

1. Google Colab 열기: https://colab.research.google.com/
2. 프로젝트를 Google Drive에 업로드
3. Drive 마운트:
```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/bioinformatics/closed_loop_enzyme_bench
```
4. 노트북 순서대로 실행:
   - `colab/01_scaffold_preprocess.ipynb`
   - `colab/02_single_shot_esmf.ipynb`
   - 등등...

### 방법 3: Python 인터랙티브 모드

```python
from pathlib import Path
from src.data.scaffolds import load_scaffold

# Scaffold 다운로드
OUT = Path("results")
sc = load_scaffold("1AKL", "A", OUT / "scaffolds")
print(f"Sequence length: {len(sc.sequence)}")
print(sc.sequence[:100])
```

## 3. ProteinMPNN 설정 (Experiment 02+ 필요)

### Colab에서 (자동)
노트북이 자동으로 ProteinMPNN을 클론합니다.

### 로컬에서
```bash
git clone https://github.com/dauparas/ProteinMPNN.git
cd ProteinMPNN
pip install -r requirements.txt
```

## 4. 문제 해결

### Python을 찾을 수 없음
- Python이 PATH에 있는지 확인
- `py` 명령어 시도 (Windows)
- Python 3.8+ 설치 확인

### 패키지 설치 오류
```bash
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir
```

### CUDA/GPU 문제
- CPU 모드로 실행: 코드에서 `device="cpu"` 사용
- Colab 사용 권장 (무료 GPU 제공)

### ProteinMPNN 오류
- ProteinMPNN이 올바른 경로에 있는지 확인
- `ProteinMPNN/protein_mpnn_run.py` 파일 존재 확인

## 5. 실험 순서

1. ✅ **Experiment 01**: Scaffold 다운로드 및 전처리
2. ⏳ **Experiment 02**: Single-shot baseline (ProteinMPNN + ESMFold)
3. ⏳ **Experiment 03**: Closed-loop optimization
4. ⏳ **Experiment 04**: Surrogate-guided active learning
5. ⏳ **Experiment 05**: 결과 분석 및 시각화

각 실험은 이전 실험의 결과를 사용합니다.

# Colab 빠른 시작 가이드

## 1단계: Colab 열기 및 설정

1. **Google Colab 열기**
   - https://colab.research.google.com/

2. **새 노트북 생성** 또는 기존 노트북 열기

3. **프로젝트 업로드 및 설정**
```python
# Drive 마운트
from google.colab import drive
drive.mount('/content/drive')

# 프로젝트로 이동
%cd /content/drive/MyDrive/bioinformatics/closed_loop_enzyme_bench
# 또는 GitHub에서 클론
# !git clone https://github.com/yourusername/bioinformatics.git
# %cd bioinformatics/closed_loop_enzyme_bench
```

## 2단계: 의존성 설치

```python
# 필수 패키지 설치
!pip -q install transformers accelerate biopython pandas numpy matplotlib tqdm scikit-learn pyyaml

# ProteinMPNN 클론
!git clone -q https://github.com/dauparas/ProteinMPNN.git
!pip -q install -r ProteinMPNN/requirements.txt

# 환경 확인
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

## 3단계: 실험 실행

### 옵션 A: 개별 실험 실행 (권장)

**Experiment 01: Scaffold**
```python
from pathlib import Path
from src.data.scaffolds import load_scaffold

OUT = Path("results")
sc = load_scaffold("1AKL", "A", OUT / "scaffolds")
print(f"Sequence length: {len(sc.sequence)}")
print(sc.sequence[:100])
```

**Experiment 02: Single-shot**
- `colab/02_single_shot_esmf.ipynb` 열기
- 또는 아래 코드 실행:

```python
from pathlib import Path
import pandas as pd
from src.data.scaffolds import load_scaffold
from src.generate.proteinmpnn import run_proteinmpnn, read_fasta_sequences
from src.evaluate.esmfold_eval import evaluate_batch

OUT = Path("results")
sc = load_scaffold("1AKL", "A", OUT / "scaffolds")

fasta = run_proteinmpnn(sc.pdb_path, OUT / "mpnn_single", Path("ProteinMPNN"), 
                        num_seqs=50, sampling_temp=0.2, seed=42)
seqs = read_fasta_sequences(fasta)

fold_res = evaluate_batch(seqs[:30], "facebook/esmfold_v1", "cuda", 
                          OUT / "pdb" / "single_shot")
df = pd.DataFrame([{"sequence": r.sequence, "mean_plddt": r.mean_plddt} 
                   for r in fold_res])
df.sort_values("mean_plddt", ascending=False).head(10)
```

**Experiment 03: Closed-loop**
- `colab/03_closed_loop_esmf.ipynb` 열기
- 또는 `run_experiment_03.py` 실행:

```python
!python run_experiment_03.py
```

### 옵션 B: 전체 실험 한번에 실행

```python
!python run_all_experiments_colab.py
```

## 4단계: 결과 확인

```python
import pandas as pd
from pathlib import Path

# Single-shot 결과
df1 = pd.read_csv("results/tables/single_shot.csv")
print("Single-shot results:")
print(df1.describe())

# Closed-loop 결과
df2 = pd.read_csv("results/tables/closed_loop.csv")
print("\nClosed-loop results:")
print(df2)

# 그래프 확인
from IPython.display import Image
Image("results/figures/closed_loop_round_curves.png")
```

## 문제 해결

### CUDA out of memory
```python
# 평가할 서열 수 줄이기
fold_res = evaluate_batch(seqs[:10], ...)  # 10개만
```

### ProteinMPNN 오류
```python
# 이미 클론되어 있는지 확인
!ls ProteinMPNN
# 없으면 다시 클론
!git clone https://github.com/dauparas/ProteinMPNN.git
```

### 모델 다운로드 느림
- 첫 실행 시 ESMFold 모델 다운로드 (약 1GB)
- 인터넷 연결 확인

## 예상 실행 시간

- **Experiment 01**: 1-2분
- **Experiment 02**: 15-30분 (30 sequences)
- **Experiment 03**: 30-60분 (4 rounds)

**총 시간**: 약 1-2시간 (GPU 사용 시)

## 다음 단계

- Experiment 04: Surrogate-guided optimization
- Experiment 05: 결과 분석 및 시각화
- `colab/04_surrogate_active_learning.ipynb` 실행

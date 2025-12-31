# Experiment 02: Single-shot Baseline 가이드

## 목표
ProteinMPNN으로 서열을 생성하고, ESMFold로 평가하여 baseline 성능을 측정합니다.

## 방법 1: Colab 사용 (권장 ⭐)

### 장점
- 무료 GPU 제공
- ProteinMPNN 자동 설치
- 모든 의존성 자동 처리

### 단계

1. **Google Colab 열기**
   - https://colab.research.google.com/

2. **프로젝트 업로드**
   - Google Drive에 `closed_loop_enzyme_bench` 폴더 업로드
   - 또는 GitHub에 푸시 후 Colab에서 클론

3. **Drive 마운트 및 설정**
```python
from google.colab import drive
drive.mount('/content/drive')

# 프로젝트 경로로 이동
%cd /content/drive/MyDrive/bioinformatics/closed_loop_enzyme_bench
# 또는 GitHub에서 클론한 경우
# !git clone https://github.com/yourusername/bioinformatics.git
# %cd bioinformatics/closed_loop_enzyme_bench
```

4. **노트북 실행**
   - `colab/02_single_shot_esmf.ipynb` 열기
   - 셀을 순서대로 실행 (Runtime → Run all)

## 방법 2: 로컬 실행

### 사전 요구사항

1. **ProteinMPNN 설치**
```bash
git clone https://github.com/dauparas/ProteinMPNN.git
cd ProteinMPNN
pip install -r requirements.txt
cd ..
```

2. **환경 확인**
```bash
python check_environment.py
```

3. **실험 실행**
```bash
python run_experiment_02.py
```

### 예상 실행 시간
- ProteinMPNN: 2-5분 (50 sequences)
- ESMFold: 10-30분 (10 sequences, GPU) 또는 1-2시간 (CPU)

## 방법 3: ProteinMPNN 없이 실행 (테스트용)

ProteinMPNN이 없어도 랜덤 mutation으로 테스트할 수 있습니다:

```python
from pathlib import Path
from src.data.scaffolds import load_scaffold
from src.generate.mutations import make_mutant_pool
from src.evaluate.esmfold_eval import evaluate_batch
import pandas as pd

OUT = Path("results")
sc = load_scaffold("1AKL", "A", OUT/"scaffolds")

# 랜덤 mutation으로 서열 생성
seqs = make_mutant_pool([sc.sequence], n=20, rate=0.05, seed=42)

# ESMFold로 평가 (처음 5개만)
fold_res = evaluate_batch(
    seqs[:5],
    model_id="facebook/esmfold_v1",
    device="cuda",  # 또는 "cpu"
    out_dir=OUT/"pdb"/"single_shot",
    max_n=5
)

# 결과 저장
df = pd.DataFrame([
    {"sequence": r.sequence, "mean_plddt": r.mean_plddt}
    for r in fold_res
])
df.to_csv(OUT/"tables"/"single_shot.csv", index=False)
print(df.sort_values("mean_plddt", ascending=False))
```

## 예상 결과

### 출력 파일
- `results/tables/single_shot.csv` - 평가 결과 테이블
- `results/pdb/single_shot/pred_*.pdb` - 예측된 구조 파일들

### 결과 해석
- **mean_plddt**: 높을수록 좋음 (80 이상이면 좋은 품질)
- **best pLDDT**: 최고 점수
- **mean pLDDT**: 평균 점수

### 성공 기준
- Best pLDDT > 70: 양호
- Best pLDDT > 80: 우수
- Mean pLDDT > 60: 양호

## 문제 해결

### ProteinMPNN 오류
```
✗ ProteinMPNN not found!
```
**해결**: 
- Colab 사용 (자동 설치)
- 또는 `git clone https://github.com/dauparas/ProteinMPNN.git`

### CUDA/GPU 오류
```
CUDA out of memory
```
**해결**:
- `max_n` 파라미터 줄이기 (예: 5개만 평가)
- `device="cpu"` 사용 (느리지만 작동)

### ESMFold 다운로드 느림
- 첫 실행 시 모델 다운로드 (약 1GB)
- 인터넷 연결 확인

## 다음 단계

Experiment 02 완료 후:
- **Experiment 03**: Closed-loop optimization
- `colab/03_closed_loop_esmf.ipynb` 실행

## 참고

- ESMFold는 느리므로 처음에는 적은 수의 서열로 테스트
- GPU 사용 시 훨씬 빠름 (Colab 권장)
- 결과는 `results/tables/single_shot.csv`에 저장됨

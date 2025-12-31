# Experiment 03: Closed-loop Optimization 가이드

## 목표
반복적 최적화를 통해 단백질 서열을 개선합니다:
1. **Propose**: 새로운 서열 생성 (ProteinMPNN 또는 mutation)
2. **Fold**: ESMFold로 구조 예측 및 평가
3. **Select**: 상위 k개 서열 선택
4. **Mutate**: 선택된 서열을 변이하여 다음 라운드 준비
5. **Repeat**: 4라운드 반복

## 방법 1: Colab 사용 (권장 ⭐)

### 장점
- 무료 GPU 제공 (ESMFold 평가 빠름)
- ProteinMPNN 자동 설치
- 실행 시간: 30-60분

### 단계

1. **Google Colab 열기**
   - https://colab.research.google.com/

2. **프로젝트 설정**
```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/bioinformatics/closed_loop_enzyme_bench
```

3. **노트북 실행**
   - `colab/03_closed_loop_esmf.ipynb` 열기
   - Runtime → Run all

## 방법 2: 로컬 실행

### 사전 요구사항

1. **Experiment 01 완료** (Scaffold 필요)
```bash
python run_experiment_01.py
```

2. **선택사항: Experiment 02 완료** (더 나은 초기화)
```bash
python run_experiment_02.py  # 또는 quick_test_02.py
```

3. **실험 실행**
```bash
python run_experiment_03.py
```

### 예상 실행 시간
- **GPU (CUDA)**: 30-60분
  - Round 0: ~15분 (15 sequences)
  - Round 1-3: ~10분 each (10 sequences each)
  
- **CPU**: 2-4시간
  - Round 0: ~60분 (5 sequences)
  - Round 1-3: ~30분 each (5 sequences each)

## 실험 구성

### 기본 설정
- **Rounds**: 4
- **Sequences per round**: 20 (생성)
- **Evaluations per round**: 10-15 (ESMFold 평가)
- **Top-k selection**: 5

### 라운드별 동작

**Round 0**:
- ProteinMPNN으로 20개 서열 생성 (또는 랜덤 mutation)
- ESMFold로 15개 평가
- 상위 5개 선택

**Round 1-3**:
- 이전 라운드 상위 5개 서열을 변이하여 20개 생성
- ESMFold로 10개 평가
- 상위 5개 선택

## 결과 해석

### 출력 파일
- `results/tables/closed_loop.csv` - 라운드별 결과
- `results/figures/closed_loop_round_curves.png` - 진행 곡선
- `results/pdb/round_*/pred_*.pdb` - 예측된 구조들

### 결과 테이블 컬럼
- `round`: 라운드 번호 (0-3)
- `n`: 평가된 서열 수
- `best`: 최고 pLDDT 점수
- `mean`: 평균 pLDDT 점수
- `success_rate_80`: pLDDT > 80인 서열 비율
- `avg_pairwise_hamming(top_k)`: 상위 k개 서열의 다양성

### 성공 지표
- **Best pLDDT 증가**: 라운드가 진행될수록 증가해야 함
- **Mean pLDDT 증가**: 평균 품질 개선
- **Success rate 증가**: 고품질 서열 비율 증가

### 예상 결과
```
Round 0: best=72.5, mean=65.3
Round 1: best=75.2, mean=68.1  ← 개선
Round 2: best=78.5, mean=70.2  ← 계속 개선
Round 3: best=81.2, mean=72.5  ← 최고점 도달
```

## 빠른 테스트 (소규모)

시간이 부족한 경우 소규모로 테스트:

```python
from pathlib import Path
from src.data.scaffolds import load_scaffold
from src.generate.mutations import make_mutant_pool
from src.evaluate.esmfold_eval import evaluate_batch
from src.loop.closed_loop import Candidate, run_closed_loop

OUT = Path("results")
sc = load_scaffold("1AKL", "A", OUT/"scaffolds")
seed_seq = sc.sequence[:150]  # 짧은 서열

def propose_fn(seeds, n, r):
    return make_mutant_pool(seeds, n=n, rate=0.05, seed=42+r)

def eval_fn(seqs, r):
    res = evaluate_batch(
        seqs[:3],  # 3개만 평가
        model_id="facebook/esmfold_v1",
        device="cuda",  # 또는 "cpu"
        out_dir=OUT/f"pdb/round_{r:02d}",
        max_n=3
    )
    return [Candidate(sequence=x.sequence, score=x.mean_plddt) for x in res]

# 2라운드만 실행
df, best = run_closed_loop([seed_seq], propose_fn, eval_fn, rounds=2, per_round=10, top_k=3)
print(df)
```

## 문제 해결

### 메모리 부족
```
CUDA out of memory
```
**해결**:
- `max_n` 줄이기 (예: 5개만 평가)
- `device="cpu"` 사용
- 서열 길이 줄이기 (`seed_seq[:150]`)

### 실행 시간이 너무 김
**해결**:
- 라운드 수 줄이기 (`rounds=2`)
- 평가 수 줄이기 (`max_n=5`)
- Colab GPU 사용

### 결과가 개선되지 않음
**가능한 원인**:
- 변이율이 너무 높음/낮음 (`rate=0.03` 조정)
- 선택 압력이 너무 약함 (`top_k` 증가)
- 초기 서열이 이미 최적

## 다음 단계

Experiment 03 완료 후:
- **Experiment 04**: Surrogate-guided optimization
  - ESM2 임베딩으로 빠른 예측
  - ESMFold 호출 수 감소
  - 더 효율적인 탐색

## 참고

- Closed-loop는 탐색 전략에 따라 결과가 달라질 수 있음
- 여러 번 실행하여 재현성 확인 권장
- GPU 사용 시 훨씬 빠름 (Colab 권장)

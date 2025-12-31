# Closed-loop Enzyme Design Benchmark (ProteinMPNN + ESMFold)

Colab-friendly benchmark for **enzyme/protein redesign**:
- Generator: ProteinMPNN
- Evaluator: ESMFold (Transformers) `facebook/esmfold_v1`
- Optional: Surrogate model (ESM2 embeddings -> predicted fold score)

## Quick Start

### 1. 환경 확인 및 설치
```bash
# 의존성 설치
pip install -r requirements.txt

# 환경 확인
python check_environment.py
```

### 2. 실험 시작
```bash
# 전체 실험 시작 (권장)
python start_experiment.py

# 또는 개별 실험 실행
python run_experiment_01.py  # Experiment 01: Scaffold
python run_experiment_02.py   # Experiment 02: Single-shot (ProteinMPNN 필요)
python quick_test_02.py       # Experiment 02 빠른 테스트 (ProteinMPNN 불필요)
python run_experiment_03.py   # Experiment 03: Closed-loop optimization
```

**참고**: Experiment 02는 ProteinMPNN이 필요합니다. 
- **Colab 사용 권장** (자동 설치): `colab/02_single_shot_esmf.ipynb`
- 또는 빠른 테스트: `python quick_test_02.py` (랜덤 mutation 사용)

자세한 가이드는 `SETUP.md`를 참고하세요.

### Colab Setup
1. Upload this repository to Google Drive or clone in Colab
2. Run notebooks in `colab/` directory sequentially:
   - `01_scaffold_preprocess.ipynb` - Download and preprocess PDB scaffolds
   - `02_single_shot_esmf.ipynb` - Single-shot baseline experiment
   - `03_closed_loop_esmf.ipynb` - Closed-loop optimization
   - `04_surrogate_active_learning.ipynb` - Surrogate-guided active learning
   - `05_figures_tables.ipynb` - Generate figures and summary tables

### Generate Notebooks (if needed)
If notebooks are missing, run:
```bash
python create_notebooks.py
```

## Project Structure

```
closed_loop_enzyme_bench/
├── README.md
├── requirements.txt
├── configs/
│   └── example.yaml          # Configuration template
├── src/
│   ├── data/                 # Scaffold fetching and parsing
│   ├── generate/            # ProteinMPNN wrappers, mutations
│   ├── evaluate/            # ESMFold evaluation
│   ├── loop/                 # Closed-loop algorithms
│   ├── models/               # Surrogate model training
│   └── metrics/              # Metrics and plotting
├── colab/                    # Colab notebooks
└── docs/                     # Methods documentation

```

## Outputs
- `results/tables/*.csv` - Experimental results tables
- `results/figures/*.png` - Visualization plots
- `results/pdb/*.pdb` - Predicted protein structures

## Experiments

1. **E1: Single-shot baseline** - Generate N sequences once, evaluate with ESMFold
2. **E2: Closed-loop** - Iterative optimization (propose → fold → select → mutate)
3. **E3: Surrogate-guided** - Use ESM2 embeddings to predict fold quality, reduce expensive ESMFold calls

## Metrics

- **mean pLDDT**: Average confidence score across all residues
- **success rate**: Fraction of sequences with pLDDT > 80
- **diversity**: Pairwise Hamming distance between sequences
- **best/mean improvement**: Comparison across rounds

## Requirements

- Python 3.8+
- PyTorch (with CUDA for GPU acceleration)
- transformers >= 4.35
- biopython, pandas, numpy, matplotlib
- ProteinMPNN (cloned automatically in Colab)

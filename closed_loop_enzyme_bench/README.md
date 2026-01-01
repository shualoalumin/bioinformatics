# Closed-loop Enzyme Design Benchmark (ProteinMPNN + ESMFold)

Colab-friendly benchmark for **enzyme/protein redesign**:
- **Generator**: ProteinMPNN
- **Evaluator**: ESMFold (Transformers) `facebook/esmfold_v1`
- **Optional**: Surrogate model (ESM2 embeddings -> predicted fold score)

## Quick Start (Local, recommended setup)

This project supports a local workflow via a project-local virtual environment (**`.venv/`**).

### 1) Create and use the virtual environment

```bash
# from closed_loop_enzyme_bench/
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Then install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2) Verify your environment

```bash
python check_environment.py
```

### 3) Run experiments

```bash
# Experiment 01: Scaffold download + sequence extraction
python run_experiment_01.py

# Experiment 02: Single-shot baseline (uses ProteinMPNN if available; otherwise falls back)
python run_experiment_02.py

# Experiment 03: Closed-loop optimization
python run_experiment_03.py
```

## Quick Start (Colab)

1. Upload the repository to Google Drive or clone it in Colab.
2. Run notebooks in `colab/` sequentially:
   - `01_scaffold_preprocess.ipynb`
   - `02_single_shot_esmf.ipynb`
   - `03_closed_loop_esmf.ipynb`
   - `04_surrogate_active_learning.ipynb`
   - `05_figures_tables.ipynb`

See `COLAB_QUICKSTART.md` for a copy/paste setup snippet.

## Project Structure

```
closed_loop_enzyme_bench/
├── README.md
├── requirements.txt
├── configs/
│   └── example.yaml
├── src/
│   ├── data/        # PDB download + chain sequence extraction
│   ├── generate/    # ProteinMPNN wrapper + simple mutation baselines
│   ├── evaluate/    # ESMFold evaluation + PDB writing
│   ├── loop/        # Closed-loop algorithms
│   ├── metrics/     # Metrics + plotting
│   └── models/      # Surrogate model (ESM2 embeddings -> MLP)
├── colab/           # Colab notebooks
└── docs/            # Methods note template
```

## Outputs

- `results/tables/*.csv` - experiment tables
- `results/figures/*.png` - plots
- `results/pdb/*.pdb` - predicted structures

## Experiments

1. **Experiment 02 (Single-shot)**: Generate N sequences once, evaluate with ESMFold
2. **Experiment 03 (Closed-loop)**: Iterative optimization (propose -> fold -> select -> mutate)
3. **Experiment 04 (Surrogate-guided)**: Use ESM2 embeddings to reduce expensive ESMFold calls

## Metrics

- **mean pLDDT**: average confidence across residues
- **success rate**: fraction with mean pLDDT >= 80
- **diversity**: average pairwise Hamming distance
- **round curves**: best/mean pLDDT by round

## Requirements

- Python 3.10+ recommended (works with newer Windows builds)
- PyTorch (CPU works; CUDA recommended for speed)
- transformers, biopython, numpy, matplotlib, scikit-learn
- ProteinMPNN (clone from `https://github.com/dauparas/ProteinMPNN` or run in Colab)

# Bioinformatics Research Repository

This repository contains computational biology and bioinformatics research projects, with a focus on protein design and enzyme engineering.

## Repository Structure

```
bioinformatics/
├── README.md                              # Repository overview and structure
├── bootstrap_closed_loop_enzyme_bench.py # Bootstrap script to generate project structure
│
└── closed_loop_enzyme_bench/             # Main project: Closed-loop enzyme design benchmark
    │
    ├── README.md                          # Project documentation and quick start
    ├── requirements.txt                   # Python package dependencies
    ├── .gitignore                         # Git ignore patterns
    │
    ├── src/                                # Core source code modules
    │   ├── __init__.py                    # Package initialization
    │   │
    │   ├── data/                          # Data loading and preprocessing
    │   │   ├── __init__.py
    │   │   └── scaffolds.py              # PDB download, chain extraction, Scaffold dataclass
    │   │
    │   ├── generate/                      # Sequence generation
    │   │   ├── __init__.py
    │   │   ├── proteinmpnn.py            # ProteinMPNN wrapper and FASTA parsing
    │   │   └── mutations.py              # Random mutation utilities (make_mutant_pool)
    │   │
    │   ├── evaluate/                      # Structure evaluation
    │   │   ├── __init__.py
    │   │   ├── esmfold_eval.py           # ESMFold model loading, folding, pLDDT scoring
    │   │   └── pdb_utils.py             # PDB format conversion utilities
    │   │
    │   ├── loop/                          # Closed-loop optimization
    │   │   ├── __init__.py
    │   │   └── closed_loop.py           # Candidate dataclass, run_closed_loop() algorithm
    │   │
    │   ├── metrics/                       # Metrics and visualization
    │   │   ├── __init__.py
    │   │   ├── metrics.py                # success_rate(), avg_pairwise_hamming()
    │   │   └── figures.py                # plot_round_curves() for visualization
    │   │
    │   └── models/                        # Surrogate models (ESM2-based active learning)
    │       ├── __init__.py
    │       └── surrogate.py              # ESM2 embeddings extraction, MLP training,
    │                                      # SurrogateConfig, get_embeddings(), train_surrogate()
    │
    ├── colab/                             # Google Colab notebooks (sequential workflow)
    │   ├── 01_scaffold_preprocess.ipynb   # Download and preprocess PDB scaffolds
    │   ├── 02_single_shot_esmf.ipynb      # Single-shot baseline (ProteinMPNN → ESMFold)
    │   ├── 03_closed_loop_esmf.ipynb      # Closed-loop optimization (4 rounds)
    │   ├── 04_surrogate_active_learning.ipynb  # Surrogate-guided active learning
    │   └── 05_figures_tables.ipynb        # Results analysis and visualization
    │
    ├── configs/                           # Configuration files
    │   └── example.yaml                   # Experiment configuration template
    │
    ├── docs/                              # Documentation
    │   └── methods_note_template.md       # Methods documentation template (2-4 pages)
    │
    ├── run_experiment_01.py              # Experiment 01: Scaffold fetch & preprocess
    ├── run_experiment_02.py              # Experiment 02: Single-shot baseline
    ├── run_experiment_03.py              # Experiment 03: Closed-loop optimization
    ├── quick_test_02.py                  # Quick test for Experiment 02 (no ProteinMPNN)
    ├── run_all_experiments_colab.py      # Complete experiment runner for Colab
    ├── start_experiment.py                # Main experiment runner (starts from Exp 01)
    ├── check_environment.py              # Environment and dependency checker
    ├── create_notebooks.py               # Script to regenerate Colab notebooks
    │
    ├── README.md                          # Project documentation
    ├── SETUP.md                           # Detailed setup instructions
    ├── QUICKSTART.md                      # Quick start guide
    ├── COLAB_QUICKSTART.md                # Colab-specific quick start
    ├── EXPERIMENT_02_GUIDE.md            # Single-shot experiment detailed guide
    └── EXPERIMENT_03_GUIDE.md            # Closed-loop experiment detailed guide
    │
    └── results/                           # Generated results (gitignored)
        ├── scaffolds/                     # Downloaded PDB files and sequences
        ├── tables/                        # CSV files with experimental results
        ├── figures/                       # PNG plots (round curves, comparisons)
        └── pdb/                           # Predicted protein structures (PDB format)
```

## Main Project: Closed-loop Enzyme Design Benchmark

### Overview

A comprehensive benchmark for **enzyme/protein redesign** using:
- **Generator**: ProteinMPNN for sequence generation
- **Evaluator**: ESMFold (Transformers) for structure prediction and quality assessment
- **Optimization**: Closed-loop iterative refinement with optional surrogate-guided active learning

### Key Features

- **Reproducible Pipeline**: Complete workflow from scaffold selection to final evaluation
- **Multiple Experiments**: Single-shot baseline, closed-loop optimization, and surrogate-guided search
- **Colab-Ready**: All experiments can be run in Google Colab with GPU support
- **Quantitative Metrics**: pLDDT scores, success rates, sequence diversity tracking

### Quick Start

See the [project README](closed_loop_enzyme_bench/README.md) for detailed instructions.

**For Colab users:**
1. Upload the `closed_loop_enzyme_bench/` folder to Google Drive
2. Open Colab and mount Drive
3. Run notebooks in `colab/` directory sequentially

**For local users:**
```bash
cd closed_loop_enzyme_bench
pip install -r requirements.txt
python start_experiment.py
```

### Project Components

#### Source Code (`src/`)

- **`data/`**: PDB scaffold downloading and sequence extraction
- **`generate/`**: ProteinMPNN integration and random mutation utilities
- **`evaluate/`**: ESMFold model wrapper for structure prediction and pLDDT scoring
- **`loop/`**: Closed-loop optimization framework (propose → fold → select → mutate)
- **`metrics/`**: Success rate, diversity (Hamming distance), and visualization
- **`models/`**: Surrogate model training using ESM2 embeddings

#### Experiments

1. **Experiment 01**: Scaffold fetch & preprocess
   - Downloads PDB structures
   - Extracts chain sequences
   - Script: `run_experiment_01.py`

2. **Experiment 02**: Single-shot baseline
   - Generates sequences with ProteinMPNN
   - Evaluates with ESMFold
   - Establishes baseline performance
   - Script: `run_experiment_02.py` or `quick_test_02.py`

3. **Experiment 03**: Closed-loop optimization
   - Iterative refinement over 4 rounds
   - Propose → Fold → Select → Mutate cycle
   - Tracks improvement metrics
   - Script: `run_experiment_03.py`

4. **Experiment 04**: Surrogate-guided active learning
   - Uses ESM2 embeddings to predict fold quality
   - Reduces expensive ESMFold calls
   - More efficient exploration
   - Notebook: `colab/04_surrogate_active_learning.ipynb`

5. **Experiment 05**: Results analysis and visualization
   - Generates summary tables and figures
   - Compares single-shot vs closed-loop performance
   - Notebook: `colab/05_figures_tables.ipynb`

#### Colab Notebooks (`colab/`)

Five sequential notebooks for complete workflow:
- `01_scaffold_preprocess.ipynb` - Data preparation
- `02_single_shot_esmf.ipynb` - Baseline experiment
- `03_closed_loop_esmf.ipynb` - Optimization loop
- `04_surrogate_active_learning.ipynb` - Active learning
- `05_figures_tables.ipynb` - Results visualization

### Output Structure

All experiments generate results in the `results/` directory:
```
results/
├── scaffolds/          # Downloaded PDB files and sequences
├── tables/             # CSV files with experimental results
├── figures/            # PNG plots (round curves, comparisons)
└── pdb/                # Predicted protein structures (PDB format)
```

### Requirements

- Python 3.8+
- PyTorch (CUDA recommended for GPU acceleration)
- transformers >= 4.35
- biopython, pandas, numpy, matplotlib, scikit-learn
- ProteinMPNN (auto-installed in Colab, or clone from GitHub)

### Documentation

- **`README.md`**: Main project documentation
- **`SETUP.md`**: Detailed setup instructions
- **`QUICKSTART.md`**: Quick start guide
- **`COLAB_QUICKSTART.md`**: Colab-specific guide
- **`EXPERIMENT_02_GUIDE.md`**: Single-shot experiment guide
- **`EXPERIMENT_03_GUIDE.md`**: Closed-loop experiment guide
- **`docs/methods_note_template.md`**: Methods documentation template

## Bootstrap Script

The `bootstrap_closed_loop_enzyme_bench.py` script can regenerate the entire project structure:

```bash
python bootstrap_closed_loop_enzyme_bench.py --out closed_loop_enzyme_bench
```

## Research Goals

This benchmark addresses:
- **Reproducibility**: Standardized pipeline for enzyme redesign experiments
- **Comparison**: Quantitative metrics for single-shot vs closed-loop methods
- **Efficiency**: Surrogate models to reduce computational cost
- **Accessibility**: Colab-ready notebooks for easy experimentation

## Citation

If you use this code in your research, please cite the relevant papers:
- ProteinMPNN: [Dauparas et al., 2022](https://www.science.org/doi/10.1126/science.add2187)
- ESMFold: [Lin et al., 2022](https://www.science.org/doi/10.1126/science.ade2574)

## License

[Add your license here]

## Contact

[Add contact information here]
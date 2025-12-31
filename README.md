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

## References and Citations

### Core Methods

If you use this code in your research, please cite the relevant papers:

#### ProteinMPNN
- **Paper**: [Dauparas et al., 2022](https://www.science.org/doi/10.1126/science.add2187) - "Robust deep learning–based protein sequence design using ProteinMPNN"
- **GitHub**: [dauparas/ProteinMPNN](https://github.com/dauparas/ProteinMPNN)
- **DOI**: 10.1126/science.add2187

#### ESMFold
- **Paper**: [Lin et al., 2022](https://www.science.org/doi/10.1126/science.ade2574) - "Evolutionary-scale prediction of atomic-level protein structure with a language model"
- **GitHub**: [facebookresearch/esm](https://github.com/facebookresearch/esm)
- **Hugging Face**: [facebook/esmfold_v1](https://huggingface.co/facebook/esmfold_v1)
- **DOI**: 10.1126/science.ade2574

#### ESM2 (for Surrogate Models)
- **Paper**: [Rives et al., 2021](https://www.pnas.org/doi/10.1073/pnas.2016239118) - "Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences"
- **GitHub**: [facebookresearch/esm](https://github.com/facebookresearch/esm)
- **Hugging Face**: [facebook/esm2_t12_35M_UR50D](https://huggingface.co/facebook/esm2_t12_35M_UR50D)
- **DOI**: 10.1073/pnas.2016239118

### Related Methods

#### AlphaFold2
- **Paper**: [Jumper et al., 2021](https://www.nature.com/articles/s41586-021-03819-2) - "Highly accurate protein structure prediction with AlphaFold"
- **GitHub**: [deepmind/alphafold](https://github.com/deepmind/alphafold)
- **DOI**: 10.1038/s41586-021-03819-2

#### ColabFold
- **Paper**: [Mirdita et al., 2022](https://www.nature.com/articles/s41592-022-01488-1) - "ColabFold: making protein folding accessible to all"
- **GitHub**: [sokrypton/ColabFold](https://github.com/sokrypton/ColabFold)
- **DOI**: 10.1038/s41592-022-01488-1

### Additional Resources

#### Protein Design and Engineering
- **Protein Design Review**: [Huang et al., 2016](https://www.nature.com/articles/nature19946) - "The coming of age of de novo protein design"
- **Enzyme Engineering**: [Arnold, 2018](https://www.nature.com/articles/s41586-018-0174-3) - "Directed Evolution: Bringing New Chemistry to Life"

#### Active Learning and Surrogate Models
- **Active Learning Review**: [Settles, 2009](https://www.morganclaypool.com/doi/abs/10.2200/S00429ED1V01Y200906AIM006) - "Active Learning Literature Survey"
- **Bayesian Optimization**: [Frazier, 2018](https://arxiv.org/abs/1807.02811) - "A Tutorial on Bayesian Optimization"

#### Datasets and Benchmarks
- **PDB (Protein Data Bank)**: [Berman et al., 2000](https://www.nucleicacidsresearch.org/article/10.1093/nar/28.1.235) - [rcsb.org](https://www.rcsb.org/)
- **UniProt**: [UniProt Consortium, 2023](https://www.nucleicacidsresearch.org/article/10.1093/nar/gkac1052) - [uniprot.org](https://www.uniprot.org/)

### Software and Tools

- **PyTorch**: [Paszke et al., 2019](https://papers.neurips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library) - [pytorch.org](https://pytorch.org/)
- **Transformers (Hugging Face)**: [Wolf et al., 2020](https://www.aclweb.org/anthology/2020.emnlp-demos.6/) - [huggingface.co](https://huggingface.co/)
- **BioPython**: [Cock et al., 2009](https://bioinformatics.oxfordjournals.org/content/25/11/1422) - [biopython.org](https://biopython.org/)

### Tutorials and Guides

- **ProteinMPNN Tutorial**: [ProteinMPNN Documentation](https://github.com/dauparas/ProteinMPNN)
- **ESMFold Usage**: [ESM Documentation](https://github.com/facebookresearch/esm)
- **Google Colab**: [Colab Documentation](https://colab.research.google.com/notebooks/intro.ipynb)

## License

[Add your license here]

## Contact

[Add contact information here]
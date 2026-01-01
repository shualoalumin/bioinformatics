# Colab Quickstart

This project is designed to run smoothly on Google Colab (recommended for GPU speed).

## Step 1) Open Colab and set the project directory

1. Open Colab: `https://colab.research.google.com/`
2. Mount Google Drive and `cd` into the repo:

```python
from google.colab import drive
drive.mount("/content/drive")
%cd /content/drive/MyDrive/bioinformatics/closed_loop_enzyme_bench
```

## Step 2) Install dependencies

```python
!pip -q install transformers accelerate biopython pandas numpy matplotlib tqdm scikit-learn pyyaml
!git clone -q https://github.com/dauparas/ProteinMPNN.git
!pip -q install -r ProteinMPNN/requirements.txt

import torch
print("torch", torch.__version__, "cuda?", torch.cuda.is_available())
```

## Step 3) Run experiments

Recommended: run notebooks in order:

- `colab/01_scaffold_preprocess.ipynb`
- `colab/02_single_shot_esmf.ipynb`
- `colab/03_closed_loop_esmf.ipynb`
- `colab/04_surrogate_active_learning.ipynb`
- `colab/05_figures_tables.ipynb`

Alternatively, run scripts:

```python
!python run_experiment_01.py
!python run_experiment_02.py
!python run_experiment_03.py
```

## Step 4) Review results

```python
import pandas as pd

df1 = pd.read_csv("results/tables/single_shot.csv")
print("Single-shot summary:")
print(df1.describe())

df2 = pd.read_csv("results/tables/closed_loop.csv")
print("Closed-loop rounds:")
print(df2)
```

## Common issues

- **First run is slow**: ESMFold downloads large weights on first use.
- **CUDA OOM**: reduce evaluation counts (fold fewer sequences per round).

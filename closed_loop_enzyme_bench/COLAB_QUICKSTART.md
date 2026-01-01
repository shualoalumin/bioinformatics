# Colab Quickstart

This project is designed to run smoothly on Google Colab (recommended for GPU speed).

## Step 1) Open Colab and set the project directory

**Option A: Clone from GitHub (Recommended)**

```python
!git clone https://github.com/YOUR_USERNAME/bioinformatics.git
%cd bioinformatics/closed_loop_enzyme_bench

# Git configuration (one-time setup)
!git config user.name "Your Name"
!git config user.email "your.email@example.com"
```

**Option B: Mount Google Drive**

```python
from google.colab import drive
drive.mount("/content/drive")
%cd /content/drive/MyDrive/bioinformatics/closed_loop_enzyme_bench
```

**GitHub Token Setup (for automatic result saving):**

1. Create a GitHub Personal Access Token:
   - Go to: GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
   - Generate new token with `repo` scope
   
2. Add to Colab Secrets:
   - Click 🔑 icon in Colab sidebar
   - "Add a secret"
   - Name: `GITHUB_TOKEN`
   - Value: Your token (starts with `ghp_`)

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

## Step 5) Automatic GitHub saving (optional)

Each experiment notebook automatically saves results to GitHub if:
- Repository is cloned (not just mounted from Drive)
- GitHub token is set in Colab Secrets (see Step 1)

Results saved:
- `results/tables/*.csv` - Experimental data
- `results/figures/*.png` - Visualization plots

Large files (PDB structures) are excluded to keep repository size small.

**Manual save (if automatic save fails):**

```python
from src.utils.github_save import save_results_to_github
save_results_to_github("exp02_single_shot")
```

## Common issues

- **First run is slow**: ESMFold downloads large weights on first use.
- **CUDA OOM**: reduce evaluation counts (fold fewer sequences per round).
- **GitHub save fails**: Check that token is set in Colab Secrets and repository is cloned (not just mounted).
# Setup Guide

This guide covers local setup (recommended via `.venv/`) and Colab setup.

## 1) Prerequisites

### Verify Python

```bash
python --version
```

Recommended: Python 3.10+.

## 2) Local setup (recommended)

From `closed_loop_enzyme_bench/`:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Verify:

```bash
python check_environment.py
```

## 3) Run experiments (local)

```bash
python run_experiment_01.py
python run_experiment_02.py
python run_experiment_03.py
```

Notes:
- On CPU, start with small evaluation counts (ESMFold can be slow).
- For GPU acceleration, Colab is usually the fastest path.

## 4) Colab setup

1. Open Colab: `https://colab.research.google.com/`
2. Upload the repo to Drive (or clone it)
3. Mount Drive:

```python
from google.colab import drive
drive.mount("/content/drive")
%cd /content/drive/MyDrive/bioinformatics/closed_loop_enzyme_bench
```

4. Run notebooks in `colab/` sequentially.

## 5) ProteinMPNN (needed for full Experiment 02/03)

Colab notebooks automatically clone ProteinMPNN.

Local:

```bash
git clone https://github.com/dauparas/ProteinMPNN.git
python -m pip install -r ProteinMPNN/requirements.txt
```

## 6) Troubleshooting

See `TROUBLESHOOTING.md`.

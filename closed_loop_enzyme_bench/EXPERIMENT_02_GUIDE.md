# Experiment 02: Single-shot Baseline

## Goal

Generate sequences (ProteinMPNN) and evaluate foldability/quality with ESMFold to establish a baseline.

## Option A (recommended): Run on Colab

Why:
- Free GPU acceleration
- Easy setup
- Fast iteration

Steps:
1. Open Colab: `https://colab.research.google.com/`
2. Clone/upload the repo
3. Run `colab/02_single_shot_esmf.ipynb` from top to bottom

## Option B: Run locally

Prerequisites:

```bash
cd closed_loop_enzyme_bench
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # PowerShell
python -m pip install -r requirements.txt
python check_environment.py
```

Run:

```bash
python run_experiment_02.py
```

Notes:
- On CPU, start with 1-3 sequences (ESMFold is slow).
- The script saves results to `results/tables/single_shot.csv`.

## Option C: Without ProteinMPNN (fallback / quick test)

If ProteinMPNN is not installed, the script can fall back to a mutation baseline automatically.
You can also run:

```bash
python quick_test_02.py
```

## Outputs

- `results/tables/single_shot.csv`
- `results/pdb/single_shot/pred_*.pdb`

## Interpreting results

- **mean_plddt**: higher is better (>= 80 is typically very good)
- Track **best** and **mean** pLDDT as the baseline for comparisons in Experiment 03.

## Next step

Proceed to Experiment 03 (Closed-loop optimization): `run_experiment_03.py` or `colab/03_closed_loop_esmf.ipynb`.

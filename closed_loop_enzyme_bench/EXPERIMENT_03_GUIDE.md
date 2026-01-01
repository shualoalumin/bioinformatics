# Experiment 03: Closed-loop Optimization

## Goal

Improve sequence quality via an iterative loop:
1. **Propose** sequences (ProteinMPNN or mutations)
2. **Fold/Evaluate** with ESMFold (mean pLDDT)
3. **Select** top-k
4. **Mutate/Redesign** around top-k
5. Repeat for multiple rounds

## Option A (recommended): Colab

Colab is recommended because ESMFold is much faster with a GPU.

Run: `colab/03_closed_loop_esmf.ipynb`

## Option B: Local

Prerequisites:
- Experiment 01 completed (scaffold downloaded)
- Dependencies installed in `.venv/` (see `SETUP.md`)

Run:

```bash
python run_experiment_03.py
```

## Outputs

- `results/tables/closed_loop.csv` (round-by-round summary)
- `results/figures/closed_loop_round_curves.png` (best/mean curve)
- `results/pdb/round_*/pred_*.pdb`

## How to interpret

A strong closed-loop run typically shows:
- increasing **best** pLDDT across rounds
- increasing **mean** pLDDT across rounds
- stable or controlled diversity (Hamming distance)

## Tips

- Start small on CPU (fewer sequences per round).
- Use Colab for full-budget runs and publication-quality figures.

## Next step

Proceed to Experiment 04 (surrogate-guided search).

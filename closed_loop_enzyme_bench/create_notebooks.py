#!/usr/bin/env python3
"""Script to create Colab notebooks for the closed-loop enzyme benchmark."""
import json
from pathlib import Path

def md_cell(text: str):
    return {"cell_type":"markdown","metadata":{},"source":text.splitlines(True)}

def code_cell(code: str):
    return {"cell_type":"code","metadata":{},"source":code.splitlines(True),"execution_count":None,"outputs":[]}

def notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "colab": {"name": "Closed-loop Enzyme Bench", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

INSTALL = """!pip -q install transformers accelerate biopython pandas numpy matplotlib tqdm scikit-learn pyyaml
import torch, platform
print("torch", torch.__version__, "cuda?", torch.cuda.is_available(), "python", platform.python_version())
"""

CLONE_MPNN = """!git clone -q https://github.com/dauparas/ProteinMPNN.git
!pip -q install -r ProteinMPNN/requirements.txt
"""

if __name__ == "__main__":
    # Use current directory as base
    out_dir = Path(__file__).parent
    colab_dir = out_dir / "colab"
    colab_dir.mkdir(parents=True, exist_ok=True)

    # Notebook 1
    nb1 = notebook([
        md_cell("# 01 — Scaffold fetch & preprocess"),
        code_cell(INSTALL),
        code_cell("""from pathlib import Path
from src.data.scaffolds import load_scaffold
OUT = Path("results")
sc = load_scaffold("1AKL","A", OUT/"scaffolds")  # change PDB/chain
print(sc.pdb_path)
print("length:", len(sc.sequence))
print(sc.sequence[:120] + "...")
"""),
    ])

    # Notebook 2
    nb2 = notebook([
        md_cell("# 02 — Single-shot baseline (ProteinMPNN → ESMFold)"),
        code_cell(INSTALL),
        code_cell(CLONE_MPNN),
        code_cell("""from pathlib import Path
import pandas as pd
from src.data.scaffolds import load_scaffold
from src.generate.proteinmpnn import run_proteinmpnn, read_fasta_sequences
from src.evaluate.esmfold_eval import evaluate_batch

OUT = Path("results")
PDB_ID, CHAIN = "1AKL","A"
sc = load_scaffold(PDB_ID, CHAIN, OUT/"scaffolds")

fasta = run_proteinmpnn(sc.pdb_path, OUT/"mpnn_single", Path("ProteinMPNN"), num_seqs=50, sampling_temp=0.2, seed=42)
seqs = read_fasta_sequences(fasta)

fold_res = evaluate_batch(seqs[:30], model_id="facebook/esmfold_v1", device="cuda", out_dir=OUT/"pdb"/"single_shot")
df = pd.DataFrame([{"sequence":r.sequence,"mean_plddt":r.mean_plddt,"pdb":str(r.pdb_path)} for r in fold_res])
df.sort_values("mean_plddt", ascending=False).head(10)
"""),
        code_cell("""from pathlib import Path
OUT = Path("results")
(OUT/"tables").mkdir(parents=True, exist_ok=True)
df.to_csv(OUT/"tables"/"single_shot.csv", index=False)
print("saved", OUT/"tables"/"single_shot.csv")
"""),
    ])

    # Notebook 3
    nb3 = notebook([
        md_cell("# 03 — Closed-loop (propose → fold → select → mutate)"),
        code_cell(INSTALL),
        code_cell(CLONE_MPNN),
        code_cell("""from pathlib import Path
from src.data.scaffolds import load_scaffold
from src.generate.proteinmpnn import run_proteinmpnn, read_fasta_sequences
from src.generate.mutations import make_mutant_pool
from src.evaluate.esmfold_eval import evaluate_batch
from src.loop.closed_loop import Candidate, run_closed_loop
import pandas as pd

OUT = Path("results")
PDB_ID, CHAIN = "1AKL","A"
sc = load_scaffold(PDB_ID, CHAIN, OUT/"scaffolds")
seed_seq = sc.sequence[:250]

def propose_fn(seeds, n, r):
    if r == 0:
        fasta = run_proteinmpnn(sc.pdb_path, OUT/"mpnn_round0", Path("ProteinMPNN"), num_seqs=n, sampling_temp=0.2, seed=42)
        return read_fasta_sequences(fasta)[:n]
    return make_mutant_pool(seeds, n=n, rate=0.03, seed=42+r)

def eval_fn(seqs, r):
    res = evaluate_batch(seqs, model_id="facebook/esmfold_v1", device="cuda", out_dir=OUT/"pdb"/f"round_{r:02d}", max_n=25)
    return [Candidate(sequence=x.sequence, score=x.mean_plddt) for x in res]

df, best = run_closed_loop([seed_seq], propose_fn, eval_fn, rounds=4, per_round=50, top_k=10)
df
"""),
        code_cell("""from pathlib import Path
from src.metrics.figures import plot_round_curves
OUT = Path("results")
(OUT/"tables").mkdir(parents=True, exist_ok=True)
df.to_csv(OUT/"tables"/"closed_loop.csv", index=False)
plot_round_curves(df, OUT/"figures"/"closed_loop_round_curves.png")
print("saved tables/figures in", OUT)
"""),
    ])

    # Notebook 4
    nb4 = notebook([
        md_cell("# 04 — Surrogate-guided closed-loop (DL / active learning)"),
        code_cell(INSTALL),
        code_cell(CLONE_MPNN),
        code_cell("""from pathlib import Path
import pandas as pd
import torch
from src.data.scaffolds import load_scaffold
from src.generate.proteinmpnn import run_proteinmpnn, read_fasta_sequences
from src.evaluate.esmfold_eval import evaluate_batch
from src.generate.mutations import make_mutant_pool
from src.models.surrogate import SurrogateConfig, get_embeddings, train_surrogate, predict_surrogate

OUT = Path("results")
sc = load_scaffold("1AKL","A", OUT/"scaffolds")

# labeled set
fasta = run_proteinmpnn(sc.pdb_path, OUT/"mpnn_surrogate_r0", Path("ProteinMPNN"), num_seqs=60, sampling_temp=0.2, seed=42)
seqs0 = read_fasta_sequences(fasta)[:40]
lab = evaluate_batch(seqs0, model_id="facebook/esmfold_v1", device="cuda", out_dir=OUT/"pdb"/"surrogate_r0", max_n=20)

train_seqs = [r.sequence for r in lab]
y = torch.tensor([r.mean_plddt for r in lab], dtype=torch.float32)

cfg = SurrogateConfig(esm2_model_id="facebook/esm2_t12_35M_UR50D", pool="mean", epochs=8, lr=1e-3, device="cuda")
X = get_embeddings(train_seqs, cfg.esm2_model_id, cfg.pool, cfg.device)
m = train_surrogate(X, y, cfg)

# propose many, fold few
cands = make_mutant_pool(train_seqs, n=200, rate=0.03, seed=123)
Xc = get_embeddings(cands, cfg.esm2_model_id, cfg.pool, cfg.device)
pred = predict_surrogate(m, Xc, cfg.device).numpy()
top_idx = pred.argsort()[::-1][:30]
to_fold = [cands[i] for i in top_idx]
res = evaluate_batch(to_fold, model_id="facebook/esmfold_v1", device="cuda", out_dir=OUT/"pdb"/"surrogate_r1", max_n=20)

df = pd.DataFrame([{"sequence":r.sequence, "mean_plddt":r.mean_plddt} for r in res]).sort_values("mean_plddt", ascending=False)
df.head(10)
"""),
        code_cell("""from pathlib import Path
OUT = Path("results")
(OUT/"tables").mkdir(parents=True, exist_ok=True)
df.to_csv(OUT/"tables"/"surrogate_guided.csv", index=False)
print("saved", OUT/"tables"/"surrogate_guided.csv")
"""),
    ])

    # Notebook 5
    nb5 = notebook([
        md_cell("# 05 — Figures & tables"),
        code_cell(INSTALL),
        code_cell("""from pathlib import Path
import pandas as pd
from src.metrics.figures import plot_round_curves
OUT = Path("results")
df = pd.read_csv(OUT/"tables"/"closed_loop.csv")
plot_round_curves(df, OUT/"figures"/"round_curves.png")
print("saved", OUT/"figures"/"round_curves.png")
df
"""),
    ])

    (colab_dir/"01_scaffold_preprocess.ipynb").write_text(json.dumps(nb1, indent=2), encoding="utf-8")
    (colab_dir/"02_single_shot_esmf.ipynb").write_text(json.dumps(nb2, indent=2), encoding="utf-8")
    (colab_dir/"03_closed_loop_esmf.ipynb").write_text(json.dumps(nb3, indent=2), encoding="utf-8")
    (colab_dir/"04_surrogate_active_learning.ipynb").write_text(json.dumps(nb4, indent=2), encoding="utf-8")
    (colab_dir/"05_figures_tables.ipynb").write_text(json.dumps(nb5, indent=2), encoding="utf-8")

    print(f"✓ Created 5 notebooks in {colab_dir}")
    print("  - 01_scaffold_preprocess.ipynb")
    print("  - 02_single_shot_esmf.ipynb")
    print("  - 03_closed_loop_esmf.ipynb")
    print("  - 04_surrogate_active_learning.ipynb")
    print("  - 05_figures_tables.ipynb")

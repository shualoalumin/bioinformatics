# bootstrap_closed_loop_enzyme_bench.py
from __future__ import annotations
from pathlib import Path
import json
import argparse
import zipfile

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

FILES: dict[str, str] = {
"README.md": """# Closed-loop Enzyme Design Benchmark (ProteinMPNN + ESMFold)

Colab-friendly benchmark for **enzyme/protein redesign**:
- Generator: ProteinMPNN
- Evaluator: ESMFold (Transformers) `facebook/esmfold_v1`
- Optional: Surrogate model (ESM2 embeddings -> predicted fold score)

Outputs:
- results/tables/*.csv
- results/figures/*.png
- results/pdb/*.pdb
""",
"requirements.txt": "\n".join([
    "transformers>=4.35","accelerate","torch","biopython","pandas","numpy",
    "matplotlib","tqdm","scikit-learn","pyyaml"
]) + "\n",
".gitignore": "__pycache__/\n*.pyc\n.ipynb_checkpoints/\nresults/\n.venv/\n.DS_Store\n",
"configs/example.yaml": """scaffold:
  pdb_id: "1AKL"
  chain_id: "A"
  max_len: 250

generator:
  mode: "proteinmpnn"
  sampling_temp: 0.2
  seed: 42

closed_loop:
  rounds: 4
  per_round: 50
  top_k: 10
  mutate_rate: 0.03
  diversify_penalty: 0.2

evaluator:
  model_id: "facebook/esmfold_v1"
  device: "cuda"

surrogate:
  enabled: true
  esm2_model_id: "facebook/esm2_t12_35M_UR50D"
  hidden_pool: "mean"
  epochs: 8
  lr: 1e-3
""",
"docs/methods_note_template.md": """# Methods Note (2–4 pages)
Goal: single-shot vs closed-loop enzyme redesign benchmark (ProteinMPNN + ESMFold)

Metrics:
- mean pLDDT
- success rate (pLDDT>80)
- diversity (pairwise Hamming)

Experiments:
E1 single-shot, E2 closed-loop, E3 surrogate-guided closed-loop (+ ablation)
""",
"src/__init__.py": "__version__='0.1.0'\n",

"src/data/scaffolds.py": """from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import requests
from Bio.PDB import PDBParser, PPBuilder

RCSB_PDB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"

@dataclass
class Scaffold:
    pdb_id: str
    chain_id: str
    pdb_path: Path
    sequence: str

def download_pdb(pdb_id: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    pdb_id = pdb_id.upper()
    url = RCSB_PDB_URL.format(pdb_id=pdb_id)
    out_path = out_dir / f"{pdb_id}.pdb"
    r = requests.get(url, timeout=60); r.raise_for_status()
    out_path.write_bytes(r.content)
    return out_path

def extract_chain_sequence(pdb_path: Path, chain_id: str) -> str:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("scaffold", str(pdb_path))
    ppb = PPBuilder()
    for model in structure:
        for chain in model:
            if chain.id == chain_id:
                peptides = ppb.build_peptides(chain)
                return "".join(str(p.get_sequence()) for p in peptides)
    raise ValueError(f"Chain {chain_id} not found in {pdb_path}")

def load_scaffold(pdb_id: str, chain_id: str, out_dir: Path) -> Scaffold:
    pdb_path = download_pdb(pdb_id, out_dir)
    seq = extract_chain_sequence(pdb_path, chain_id)
    return Scaffold(pdb_id.upper(), chain_id, pdb_path, seq)
""",

"src/generate/mutations.py": """from __future__ import annotations
import random
from typing import List, Sequence
AA=list("ACDEFGHIKLMNPQRSTVWY")

def mutate_sequence(seq: str, rate: float, rng: random.Random) -> str:
    out=list(seq)
    for i,ch in enumerate(out):
        if rng.random()<rate:
            out[i]=rng.choice([a for a in AA if a!=ch])
    return "".join(out)

def make_mutant_pool(seeds: Sequence[str], n: int, rate: float, seed: int=0) -> List[str]:
    rng=random.Random(seed)
    return [mutate_sequence(rng.choice(seeds), rate, rng) for _ in range(n)]
""",

"src/generate/proteinmpnn.py": """from __future__ import annotations
import subprocess
from pathlib import Path
from typing import List

def run_proteinmpnn(pdb_path: Path, out_dir: Path, proteinmpnn_dir: Path,
                    num_seqs: int=50, sampling_temp: float=0.2, seed: int=0) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd=[
        "python", str(proteinmpnn_dir/"protein_mpnn_run.py"),
        "--pdb_path", str(pdb_path),
        "--out_folder", str(out_dir),
        "--num_seq_per_target", str(num_seqs),
        "--sampling_temp", str(sampling_temp),
        "--seed", str(seed),
        "--batch_size","1",
    ]
    subprocess.check_call(cmd)
    seq_dir=out_dir/"seqs"
    fas=sorted(seq_dir.glob("*.fa"))+sorted(seq_dir.glob("*.fasta"))
    if not fas: raise FileNotFoundError(f"No FASTA in {seq_dir}")
    return fas[0]

def read_fasta_sequences(fp: Path) -> List[str]:
    seqs=[]; cur=[]
    for line in fp.read_text().splitlines():
        line=line.strip()
        if not line: continue
        if line.startswith(">"):
            if cur: seqs.append("".join(cur)); cur=[]
        else:
            cur.append(line)
    if cur: seqs.append("".join(cur))
    seen=set(); out=[]
    for s in seqs:
        if s not in seen: out.append(s); seen.add(s)
    return out
""",

"src/evaluate/pdb_utils.py": """from __future__ import annotations
def convert_outputs_to_pdb(outputs, sequence: str) -> str:
    pos=getattr(outputs,"positions",None)
    if pos is None:
        raise RuntimeError("No positions in outputs; upgrade transformers or use infer_pdbs().")
    pos=pos[0].detach().cpu().numpy()
    ca=pos[:,1,:]
    lines=[]; atom_id=1
    for res_id,(x,y,z) in enumerate(ca, start=1):
        lines.append(f"ATOM  {atom_id:5d}  CA  ALA A{res_id:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C")
        atom_id+=1
    lines.append("END")
    return "\\n".join(lines)+"\\n"
""",

"src/evaluate/esmfold_eval.py": """from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import torch
from transformers import AutoTokenizer, EsmForProteinFolding
from .pdb_utils import convert_outputs_to_pdb

@dataclass
class FoldResult:
    sequence: str
    mean_plddt: float
    pdb_path: Path

def load_esmfold(model_id: str, device: str):
    tok=AutoTokenizer.from_pretrained(model_id)
    model=EsmForProteinFolding.from_pretrained(model_id)
    if device=="cuda" and not torch.cuda.is_available(): device="cpu"
    model=model.to(device); model.eval()
    return tok, model

@torch.no_grad()
def fold_sequence(tok, model, seq: str, out_pdb: Path) -> Tuple[float, Path]:
    device=next(model.parameters()).device
    inp=tok([seq], return_tensors="pt", add_special_tokens=False)
    inp={k:v.to(device) for k,v in inp.items()}
    out=model(**inp)
    plddt=out.plddt.squeeze(0).float().detach().cpu()
    mean=float(plddt.mean().item())
    try:
        pdb_str=model.infer_pdbs(**inp)[0]
    except Exception:
        pdb_str=convert_outputs_to_pdb(out, seq)
    out_pdb.parent.mkdir(parents=True, exist_ok=True)
    out_pdb.write_text(pdb_str)
    return mean, out_pdb

def evaluate_batch(seqs: List[str], model_id: str, device: str, out_dir: Path, max_n: int|None=None) -> List[FoldResult]:
    tok, model=load_esmfold(model_id, device)
    out=[]
    for i,s in enumerate(seqs[: (max_n or len(seqs))]):
        score, pdb=fold_sequence(tok, model, s, out_dir/f"pred_{i:04d}.pdb")
        out.append(FoldResult(s, score, pdb))
    return out
""",

"src/metrics/metrics.py": """from __future__ import annotations
from typing import List
import numpy as np

def success_rate(scores: List[float], thr: float=80.0)->float:
    if not scores: return float("nan")
    return float(np.mean([s>=thr for s in scores]))

def avg_pairwise_hamming(seqs: List[str], max_pairs: int=2000)->float:
    if len(seqs)<2: return 0.0
    total=0; pairs=0
    for i in range(len(seqs)):
        for j in range(i+1,len(seqs)):
            a,b=seqs[i],seqs[j]
            if len(a)!=len(b): continue
            total += sum(x!=y for x,y in zip(a,b))
            pairs += 1
            if pairs>=max_pairs: return float(total/pairs)
    return float(total/pairs)
""",

"src/metrics/figures.py": """from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

def plot_round_curves(df: pd.DataFrame, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(df["round"], df["best"], label="best")
    plt.plot(df["round"], df["mean"], label="mean")
    plt.xlabel("round"); plt.ylabel("mean pLDDT")
    plt.legend(); plt.tight_layout()
    plt.savefig(out_path, dpi=200); plt.close()
""",

"src/loop/closed_loop.py": """from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple
import pandas as pd
from ..metrics.metrics import success_rate, avg_pairwise_hamming

@dataclass
class Candidate:
    sequence: str
    score: float

def select_top(cands: Sequence[Candidate], k: int)->List[Candidate]:
    return sorted(cands, key=lambda c:c.score, reverse=True)[:k]

def run_closed_loop(seed_sequences: List[str],
                    propose_fn: Callable[[List[str],int,int],List[str]],
                    eval_fn: Callable[[List[str],int],List[Candidate]],
                    rounds: int, per_round: int, top_k: int)->Tuple[pd.DataFrame, List[Candidate]]:
    hist=[]; best=None; seeds=seed_sequences[:]
    for r in range(rounds):
        props=propose_fn(seeds, per_round, r)
        cands=eval_fn(props, r)
        top=select_top(cands, top_k)
        scores=[c.score for c in cands]
        hist.append({
            "round": r, "n": len(cands),
            "best": max(scores) if scores else None,
            "mean": float(sum(scores)/len(scores)) if scores else None,
            "success_rate_80": success_rate(scores, 80.0),
            "avg_pairwise_hamming(top_k)": avg_pairwise_hamming([c.sequence for c in top]),
        })
        if top and (best is None or top[0].score>best.score): best=top[0]
        seeds=[c.sequence for c in top]
    return pd.DataFrame(hist), ([best] if best else [])
""",

"src/models/surrogate.py": """from __future__ import annotations
from dataclasses import dataclass
from typing import List, Literal
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, EsmModel
PoolMode=Literal["mean","cls"]

@dataclass
class SurrogateConfig:
    esm2_model_id: str="facebook/esm2_t12_35M_UR50D"
    pool: PoolMode="mean"
    lr: float=1e-3
    epochs: int=8
    batch_size: int=16
    device: str="cuda"

def get_embeddings(seqs: List[str], model_id: str, pool: PoolMode, device: str)->torch.Tensor:
    if device=="cuda" and not torch.cuda.is_available(): device="cpu"
    tok=AutoTokenizer.from_pretrained(model_id)
    model=EsmModel.from_pretrained(model_id).to(device); model.eval()
    vecs=[]
    with torch.no_grad():
        for s in seqs:
            inp=tok(s, return_tensors="pt", add_special_tokens=False)
            inp={k:v.to(device) for k,v in inp.items()}
            hs=model(**inp).last_hidden_state.squeeze(0)
            vec = hs[0] if pool=="cls" else hs.mean(0)
            vecs.append(vec.detach().cpu())
    return torch.stack(vecs,0)

class MLP(nn.Module):
    def __init__(self,d:int):
        super().__init__()
        self.net=nn.Sequential(nn.LayerNorm(d), nn.Linear(d,256), nn.GELU(), nn.Dropout(0.1),
                               nn.Linear(256,64), nn.GELU(), nn.Linear(64,1))
    def forward(self,x): return self.net(x).squeeze(-1)

def train_surrogate(X: torch.Tensor, y: torch.Tensor, cfg: SurrogateConfig)->MLP:
    device=cfg.device if (cfg.device=="cpu" or torch.cuda.is_available()) else "cpu"
    X,y=X.to(device),y.to(device)
    m=MLP(X.shape[1]).to(device)
    opt=torch.optim.AdamW(m.parameters(), lr=cfg.lr)
    loss_fn=nn.MSELoss()
    dl=DataLoader(TensorDataset(X,y), batch_size=cfg.batch_size, shuffle=True)
    m.train()
    for _ in range(cfg.epochs):
        for xb,yb in dl:
            pred=m(xb); loss=loss_fn(pred,yb)
            opt.zero_grad(); loss.backward(); opt.step()
    m.eval(); return m

@torch.no_grad()
def predict_surrogate(m: MLP, X: torch.Tensor, device: str)->torch.Tensor:
    if device=="cuda" and not torch.cuda.is_available(): device="cpu"
    m=m.to(device); X=X.to(device)
    return m(X).detach().cpu()
""",
}

def write_file(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

def build_notebooks(out_dir: Path):
    (out_dir/"colab").mkdir(parents=True, exist_ok=True)

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

    (out_dir/"colab"/"01_scaffold_preprocess.ipynb").write_text(json.dumps(nb1, indent=2), encoding="utf-8")
    (out_dir/"colab"/"02_single_shot_esmf.ipynb").write_text(json.dumps(nb2, indent=2), encoding="utf-8")
    (out_dir/"colab"/"03_closed_loop_esmf.ipynb").write_text(json.dumps(nb3, indent=2), encoding="utf-8")
    (out_dir/"colab"/"04_surrogate_active_learning.ipynb").write_text(json.dumps(nb4, indent=2), encoding="utf-8")
    (out_dir/"colab"/"05_figures_tables.ipynb").write_text(json.dumps(nb5, indent=2), encoding="utf-8")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="closed_loop_enzyme_bench")
    ap.add_argument("--zip", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for rel, text in FILES.items():
        write_file(out_dir/rel, text)

    build_notebooks(out_dir)

    if args.zip:
        zip_path = out_dir.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for p in out_dir.rglob("*"):
                if p.is_file():
                    z.write(p, arcname=str(p.relative_to(out_dir.parent)))
        print("ZIP:", zip_path)

    print("DONE:", out_dir)

if __name__ == "__main__":
    main()

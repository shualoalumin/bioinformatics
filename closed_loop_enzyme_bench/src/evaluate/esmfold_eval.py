from __future__ import annotations
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
    """Fold a single sequence using ESMFold and return mean pLDDT score and PDB path."""
    device = next(model.parameters()).device
    inp = tok([seq], return_tensors="pt", add_special_tokens=False)
    inp = {k: v.to(device) for k, v in inp.items()}
    out = model(**inp)
    
    # Extract pLDDT scores
    plddt = out.plddt.squeeze(0).float().detach().cpu()
    mean = float(plddt.mean().item())
    
    # Try to get PDB string - use multiple fallback methods
    pdb_str = None
    try:
        # Method 1: Use infer_pdbs if available (newer transformers)
        if hasattr(model, 'infer_pdbs'):
            pdb_str = model.infer_pdbs(**inp)[0]
    except Exception:
        pass
    
    if pdb_str is None:
        try:
            # Method 2: Use convert_outputs_to_pdb fallback
            pdb_str = convert_outputs_to_pdb(out, seq)
        except Exception as e:
            # Method 3: Create minimal PDB with just CA atoms from positions
            raise RuntimeError(f"Failed to generate PDB: {e}")
    
    out_pdb.parent.mkdir(parents=True, exist_ok=True)
    out_pdb.write_text(pdb_str)
    return mean, out_pdb

def evaluate_batch(seqs: List[str], model_id: str, device: str, out_dir: Path, max_n: int|None=None) -> List[FoldResult]:
    tok, model=load_esmfold(model_id, device)
    out=[]
    n = (max_n or len(seqs))
    for i,s in enumerate(seqs[:n]):
        try:
            print(f"[ESMFold] Folding {i+1}/{n} (L={len(s)}) ...", flush=True)
            score, pdb=fold_sequence(tok, model, s, out_dir/f"pred_{i:04d}.pdb")
            print(f"[ESMFold] Done {i+1}/{n}: mean_pLDDT={score:.2f}", flush=True)
            out.append(FoldResult(s, score, pdb))
        except Exception as e:
            print(f"[ESMFold][ERROR] Failed on {i+1}/{n}: {e}", flush=True)
            continue
    return out

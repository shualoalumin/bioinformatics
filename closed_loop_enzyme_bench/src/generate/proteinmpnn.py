from __future__ import annotations
import subprocess
from pathlib import Path
from typing import List

def run_proteinmpnn(pdb_path: Path, out_dir: Path, proteinmpnn_dir: Path,
                    num_seqs: int=50, sampling_temp: float=0.2, seed: int=0) -> Path:
    """Run ProteinMPNN to generate sequences from a PDB structure."""
    import sys
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Try to find the protein_mpnn_run.py script
    script_path = proteinmpnn_dir / "protein_mpnn_run.py"
    if not script_path.exists():
        # Try alternative locations
        alt_paths = [
            proteinmpnn_dir / "helper_scripts" / "protein_mpnn_run.py",
            proteinmpnn_dir / "run_protein_mpnn.py",
        ]
        for alt in alt_paths:
            if alt.exists():
                script_path = alt
                break
        else:
            raise FileNotFoundError(f"ProteinMPNN script not found in {proteinmpnn_dir}")
    
    cmd = [
        sys.executable, str(script_path),
        "--pdb_path", str(pdb_path),
        "--out_folder", str(out_dir),
        "--num_seq_per_target", str(num_seqs),
        "--sampling_temp", str(sampling_temp),
        "--seed", str(seed),
        "--batch_size", "1",
    ]
    
    try:
        subprocess.check_call(cmd, cwd=str(proteinmpnn_dir))
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ProteinMPNN failed: {e}")
    
    # Find output FASTA file
    seq_dir = out_dir / "seqs"
    if not seq_dir.exists():
        seq_dir = out_dir  # Sometimes output is directly in out_dir
    
    fas = sorted(seq_dir.glob("*.fa")) + sorted(seq_dir.glob("*.fasta"))
    if not fas:
        raise FileNotFoundError(f"No FASTA files found in {seq_dir}")
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

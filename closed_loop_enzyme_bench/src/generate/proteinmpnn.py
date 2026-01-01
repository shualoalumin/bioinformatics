from __future__ import annotations
import subprocess
from pathlib import Path
from typing import List

def run_proteinmpnn(pdb_path: Path, out_dir: Path, proteinmpnn_dir: Path,
                    num_seqs: int=50, sampling_temp: float=0.2, seed: int=0) -> Path:
    """Run ProteinMPNN to generate sequences from a PDB structure."""
    import sys
    import os
    import shutil
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to absolute paths
    proteinmpnn_dir = proteinmpnn_dir.resolve()
    pdb_path = pdb_path.resolve()
    out_dir = out_dir.resolve()
    
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
    
    # Determine model weights path (vanilla model by default)
    model_weights_path = proteinmpnn_dir / "vanilla_model_weights"
    if not model_weights_path.exists():
        # Try other locations
        alt_weights = [
            proteinmpnn_dir / "soluble_model_weights",
            proteinmpnn_dir / "ca_model_weights",
        ]
        for alt in alt_weights:
            if alt.exists():
                model_weights_path = alt
                break
    
    # ProteinMPNN has bugs with paths - copy PDB directly to ProteinMPNN root (no subdirectory)
    # Use a temp name to avoid conflicts
    temp_pdb_name = f"_temp_{pdb_path.stem}.pdb"
    temp_pdb = proteinmpnn_dir / temp_pdb_name
    shutil.copy2(pdb_path, temp_pdb)
    
    # Create temp output dir inside ProteinMPNN
    temp_output_dir = proteinmpnn_dir / "_temp_outputs"
    temp_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use just the filename (no directory) for pdb_path
    cmd = [
        sys.executable, "protein_mpnn_run.py",
        "--pdb_path", temp_pdb_name,
        "--out_folder", "_temp_outputs",
        "--num_seq_per_target", str(num_seqs),
        "--sampling_temp", str(sampling_temp),
        "--seed", str(seed),
        "--batch_size", "1",
        "--path_to_model_weights", str(model_weights_path),
    ]
    
    try:
        result = subprocess.run(cmd, cwd=str(proteinmpnn_dir), capture_output=True, text=True, check=False)
        if result.returncode != 0:
            error_msg = f"ProteinMPNN failed with exit code {result.returncode}\n"
            error_msg += f"STDOUT:\n{result.stdout}\n"
            error_msg += f"STDERR:\n{result.stderr}\n"
            error_msg += f"Command: {' '.join(cmd)}\n"
            error_msg += f"Working directory: {proteinmpnn_dir}\n"
            raise RuntimeError(error_msg)
    except FileNotFoundError as e:
        raise RuntimeError(f"Could not execute ProteinMPNN: {e}")
    
    # Copy output files from temp location to actual output directory
    temp_seq_dir = temp_output_dir / "seqs"
    if not temp_seq_dir.exists():
        temp_seq_dir = temp_output_dir
    
    # Find FASTA files in temp output
    fas = sorted(temp_seq_dir.glob("*.fa")) + sorted(temp_seq_dir.glob("*.fasta"))
    if not fas:
        raise FileNotFoundError(f"No FASTA files found in {temp_seq_dir}")
    
    # Copy to actual output directory with original name (remove _temp_ prefix)
    final_seq_dir = out_dir / "seqs"
    final_seq_dir.mkdir(parents=True, exist_ok=True)
    
    # Rename from _temp_XXX.fa to XXX.fa (original PDB name)
    original_fasta_name = pdb_path.stem + ".fa"
    final_fasta = final_seq_dir / original_fasta_name
    shutil.copy2(fas[0], final_fasta)
    
    # Clean up temp files
    try:
        temp_pdb.unlink(missing_ok=True)
        shutil.rmtree(temp_output_dir)
    except Exception:
        pass  # Ignore cleanup errors
    
    return final_fasta

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

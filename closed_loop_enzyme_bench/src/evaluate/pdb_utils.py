from __future__ import annotations
import numpy as np

def convert_outputs_to_pdb(outputs, sequence: str) -> str:
    """Convert ESMFold model outputs to PDB format string."""
    # Try to get positions from outputs
    pos = getattr(outputs, "positions", None)
    if pos is None:
        # Try alternative attribute names
        pos = getattr(outputs, "aatype", None)
        if pos is None:
            raise RuntimeError("No positions in outputs; upgrade transformers or use infer_pdbs().")
    
    # Handle tensor format
    if hasattr(pos, 'detach'):
        pos = pos.detach().cpu().numpy()
    
    # Extract CA positions (assuming shape [batch, length, atoms, 3])
    # For ESMFold, positions are typically [batch, length, atoms, 3] where atoms[1] is CA
    if len(pos.shape) == 4:
        ca = pos[0, :, 1, :]  # [length, 3] - CA atoms
    elif len(pos.shape) == 3:
        ca = pos[0, :, :]  # [length, 3] - assume already CA
    else:
        raise ValueError(f"Unexpected positions shape: {pos.shape}")
    
    lines = []
    atom_id = 1
    aa_3letter = {
        'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
        'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
        'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
        'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'
    }
    
    for res_id, (x, y, z) in enumerate(ca, start=1):
        aa = sequence[res_id - 1] if res_id <= len(sequence) else 'A'
        aa_name = aa_3letter.get(aa, 'ALA')
        lines.append(f"ATOM  {atom_id:5d}  CA  {aa_name} A{res_id:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C")
        atom_id += 1
    lines.append("END")
    return "\n".join(lines) + "\n"

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
    
    # Extract CA positions
    # ESMFold outputs can have different shapes depending on version:
    # - (batch, length, atoms, 3) - older versions
    # - (num_recycles, batch, length, atoms, 3) - newer versions with recycling
    # CA atom is at index 1 in the atoms dimension
    
    if len(pos.shape) == 5:
        # Shape: (num_recycles, batch, length, atoms, 3)
        # Use the last recycling iteration
        ca = pos[-1, 0, :, 1, :]  # [length, 3] - CA atoms from last iteration
    elif len(pos.shape) == 4:
        ca = pos[0, :, 1, :]  # [length, 3] - CA atoms
    elif len(pos.shape) == 3:
        # Could be (batch, length, 3) OR (length, atoms, 3)
        if pos.shape[-1] != 3:
            raise ValueError(f"Unexpected positions shape: {pos.shape}")
        if pos.shape[0] == 1 and pos.shape[2] == 3:
            # (1, length, 3) -> already coordinates
            ca = pos[0, :, :]
        elif pos.shape[-2] in (14, 37):
            # (length, atoms, 3) -> take CA at atom index 1
            ca = pos[:, 1, :]
        else:
            # Fallback: assume first dim is batch
            ca = pos[0, :, :]
    elif len(pos.shape) == 2:
        ca = pos  # Already [length, 3]
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

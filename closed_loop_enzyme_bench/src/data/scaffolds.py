from __future__ import annotations
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

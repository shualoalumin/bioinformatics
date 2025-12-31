#!/usr/bin/env python3
"""
Experiment 01: Scaffold fetch & preprocess
Download PDB structure and extract sequence.
"""
from pathlib import Path
from src.data.scaffolds import load_scaffold

def main():
    print("=" * 60)
    print("Experiment 01: Scaffold Fetch & Preprocess")
    print("=" * 60)
    
    OUT = Path("results")
    PDB_ID = "1AKL"
    CHAIN = "A"
    
    print(f"\nDownloading PDB: {PDB_ID}, Chain: {CHAIN}")
    print("This may take a moment...")
    
    try:
        sc = load_scaffold(PDB_ID, CHAIN, OUT / "scaffolds")
        
        print(f"\n✓ Successfully loaded scaffold!")
        print(f"  PDB ID: {sc.pdb_id}")
        print(f"  Chain: {sc.chain_id}")
        print(f"  PDB Path: {sc.pdb_path}")
        print(f"  Sequence Length: {len(sc.sequence)}")
        print(f"\nSequence (first 120 chars):")
        print(f"  {sc.sequence[:120]}...")
        print(f"\nFull sequence:")
        print(f"  {sc.sequence}")
        
        # Save sequence to file for reference
        seq_file = OUT / "scaffolds" / f"{PDB_ID}_{CHAIN}_sequence.txt"
        seq_file.write_text(sc.sequence)
        print(f"\n✓ Sequence saved to: {seq_file}")
        
        return sc
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()

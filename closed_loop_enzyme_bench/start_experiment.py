#!/usr/bin/env python3
"""
Main experiment runner - starts from Experiment 01.
"""
import sys
from pathlib import Path

def run_experiment_01():
    """Run Experiment 01: Scaffold fetch & preprocess."""
    print("\n" + "=" * 60)
    print("Starting Experiment 01: Scaffold Fetch & Preprocess")
    print("=" * 60)
    
    try:
        from src.data.scaffolds import load_scaffold
        
        OUT = Path("results")
        PDB_ID = "1AKL"
        CHAIN = "A"
        
        print(f"\nDownloading PDB: {PDB_ID}, Chain: {CHAIN}")
        sc = load_scaffold(PDB_ID, CHAIN, OUT / "scaffolds")
        
        print(f"\n✓ Successfully loaded scaffold!")
        print(f"  PDB ID: {sc.pdb_id}")
        print(f"  Chain: {sc.chain_id}")
        print(f"  Sequence Length: {len(sc.sequence)}")
        print(f"\nSequence preview: {sc.sequence[:80]}...")
        
        # Save sequence
        seq_file = OUT / "scaffolds" / f"{PDB_ID}_{CHAIN}_sequence.txt"
        seq_file.write_text(sc.sequence)
        print(f"✓ Sequence saved to: {seq_file}")
        
        return sc
        
    except Exception as e:
        print(f"\n✗ Error in Experiment 01: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print("=" * 60)
    print("Closed-loop Enzyme Design Benchmark")
    print("Experiment Runner")
    print("=" * 60)
    
    # Check environment first
    print("\n[Step 0] Checking environment...")
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, "check_environment.py"],
            capture_output=True,
            text=True
        )
        print(result.stdout)
        if result.returncode != 0:
            print("\n⚠ Environment check failed. Please install missing packages.")
            print("  Run: pip install -r requirements.txt")
            return
    except Exception as e:
        print(f"⚠ Could not run environment check: {e}")
        print("  Continuing anyway...")
    
    # Run Experiment 01
    print("\n[Step 1] Running Experiment 01...")
    scaffold = run_experiment_01()
    
    if scaffold:
        print("\n" + "=" * 60)
        print("✓ Experiment 01 completed successfully!")
        print("=" * 60)
        print("\nNext steps:")
        print("  1. Review the scaffold sequence")
        print("  2. Run Experiment 02: Single-shot baseline")
        print("     (Requires ProteinMPNN - see README.md)")
        print("\nTo continue, run:")
        print("  python -m src.generate.proteinmpnn  # (if implemented)")
        print("  Or use the Colab notebooks in colab/ directory")
    else:
        print("\n✗ Experiment 01 failed. Please check the error messages above.")

if __name__ == "__main__":
    main()

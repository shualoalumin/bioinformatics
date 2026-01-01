#!/usr/bin/env python3
"""
Experiment 02: Single-shot baseline (ProteinMPNN → ESMFold)
Generate sequences with ProteinMPNN and evaluate with ESMFold.
"""
from pathlib import Path
import sys
import pandas as pd

def check_proteinmpnn():
    """Check if ProteinMPNN is available."""
    mpnn_dir = Path("ProteinMPNN")
    if not mpnn_dir.exists():
        print("[WARNING] ProteinMPNN not found!")
        print("\nOptions:")
        print("  1. Clone ProteinMPNN:")
        print("     git clone https://github.com/dauparas/ProteinMPNN.git")
        print("  2. Use Colab (ProteinMPNN auto-installed)")
        print("  3. Skip ProteinMPNN and use random mutations instead")
        return False
    
    script = mpnn_dir / "protein_mpnn_run.py"
    if not script.exists():
        alt_script = mpnn_dir / "helper_scripts" / "protein_mpnn_run.py"
        if not alt_script.exists():
            print(f"[WARNING] ProteinMPNN script not found in {mpnn_dir}")
            return False
    
    print(f"[OK] ProteinMPNN found: {mpnn_dir}")
    return True

def run_with_proteinmpnn():
    """Run experiment with ProteinMPNN."""
    from src.data.scaffolds import load_scaffold
    from src.generate.proteinmpnn import run_proteinmpnn, read_fasta_sequences
    from src.evaluate.esmfold_eval import evaluate_batch
    
    OUT = Path("results")
    PDB_ID, CHAIN = "1AKL", "A"
    
    print(f"\n[Step 1] Loading scaffold: {PDB_ID}, Chain {CHAIN}")
    sc = load_scaffold(PDB_ID, CHAIN, OUT / "scaffolds")
    print(f"[OK] Scaffold loaded: {len(sc.sequence)} residues")
    
    print(f"\n[Step 2] Generating sequences with ProteinMPNN...")
    print("  This may take a few minutes...")
    
    try:
        fasta = run_proteinmpnn(
            sc.pdb_path, 
            OUT / "mpnn_single", 
            Path("ProteinMPNN"),
            num_seqs=50, 
            sampling_temp=0.2, 
            seed=42
        )
        seqs = read_fasta_sequences(fasta)
        print(f"[OK] Generated {len(seqs)} sequences")
        
    except Exception as e:
        print(f"[ERROR] ProteinMPNN failed: {e}")
        print("\nFalling back to random mutations...")
        return run_with_mutations(sc)
    
    print(f"\n[Step 3] Evaluating with ESMFold...")
    print("  This will take several minutes (ESMFold is slow)...")
    print("  Tip: on CPU, start with 1-3 sequences, then scale up.")
    
    try:
        # Auto-scale eval count for CPU vs GPU
        dev = "cuda" if __import__("torch").cuda.is_available() else "cpu"
        n_eval = 10 if dev == "cuda" else 1
        fold_res = evaluate_batch(
            seqs[:max(1, n_eval)],
            model_id="facebook/esmfold_v1", 
            device=dev,
            out_dir=OUT / "pdb" / "single_shot",
            max_n=n_eval
        )
        
        rows = [
            {"sequence": r.sequence, "mean_plddt": r.mean_plddt, "pdb": str(r.pdb_path)}
            for r in fold_res
        ]

        if len(rows) == 0:
            raise RuntimeError("No fold results were produced. See logs above (ESMFold failures).")

        # Convert to DataFrame for easier handling
        df = pd.DataFrame(rows)
        df_sorted = df.sort_values("mean_plddt", ascending=False)
        
        print(f"\n[OK] Evaluation complete!")
        print(f"\nTop 5 sequences by pLDDT:")
        for _, r in df_sorted.head(5).iterrows():
            print(f"  mean_pLDDT={r['mean_plddt']:.2f}  pdb={r['pdb']}")
        
        print(f"\nStatistics:")
        print(f"  Best pLDDT: {df['mean_plddt'].max():.2f}")
        print(f"  Mean pLDDT: {df['mean_plddt'].mean():.2f}")
        print(f"  Min pLDDT: {df['mean_plddt'].min():.2f}")
        
        # Save results
        (OUT / "tables").mkdir(parents=True, exist_ok=True)
        csv_path = OUT / "tables" / "single_shot.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n[OK] Results saved to: {csv_path}")
        
        return rows
        
    except Exception as e:
        print(f"[ERROR] ESMFold evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_with_mutations(scaffold):
    """Fallback: Use random mutations instead of ProteinMPNN."""
    from src.generate.mutations import make_mutant_pool
    from src.evaluate.esmfold_eval import evaluate_batch
    
    print("\n[Alternative] Using random mutations...")
    print("  Generating 20 mutant sequences...")
    
    seqs = make_mutant_pool([scaffold.sequence], n=20, rate=0.05, seed=42)
    print(f"[OK] Generated {len(seqs)} mutant sequences")
    
    print(f"\n[Step 3] Evaluating with ESMFold...")
    try:
        dev = "cuda" if __import__("torch").cuda.is_available() else "cpu"
        n_eval = 10 if dev == "cuda" else 1
        fold_res = evaluate_batch(
            seqs[:max(1, n_eval)],
            model_id="facebook/esmfold_v1",
            device=dev,
            out_dir=Path("results") / "pdb" / "single_shot",
            max_n=n_eval
        )
        
        rows = [
            {"sequence": r.sequence, "mean_plddt": r.mean_plddt, "pdb": str(r.pdb_path)}
            for r in fold_res
        ]

        if len(rows) == 0:
            raise RuntimeError("No fold results were produced. See logs above (ESMFold failures).")
        
        # Convert to DataFrame for easier handling
        df = pd.DataFrame(rows)
        df_sorted = df.sort_values("mean_plddt", ascending=False)
        
        print(f"\n[OK] Evaluation complete!")
        print(f"\nTop 5 sequences:")
        for _, r in df_sorted.head(5).iterrows():
            print(f"  mean_pLDDT={r['mean_plddt']:.2f}  pdb={r['pdb']}")
        
        OUT = Path("results")
        (OUT / "tables").mkdir(parents=True, exist_ok=True)
        csv_path = OUT / "tables" / "single_shot.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n[OK] Results saved to: {csv_path}")
        
        return rows
        
    except Exception as e:
        print(f"[ERROR] Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print("=" * 60)
    print("Experiment 02: Single-shot Baseline")
    print("ProteinMPNN -> ESMFold Evaluation")
    print("=" * 60)
    
    # Check if scaffold exists
    scaffold_file = Path("results/scaffolds/1AKL.pdb")
    if not scaffold_file.exists():
        print("\n[WARNING] Scaffold not found. Running Experiment 01 first...")
        try:
            from run_experiment_01 import main as run_exp01
            scaffold = run_exp01()
            if not scaffold:
                print("\n[ERROR] Failed to load scaffold. Please run Experiment 01 first.")
                return
        except Exception as e:
            print(f"[ERROR] Error: {e}")
            print("Please run: python run_experiment_01.py")
            return
    
    # Check ProteinMPNN
    has_mpnn = check_proteinmpnn()
    
    if has_mpnn:
        print("\nUsing ProteinMPNN for sequence generation...")
        result = run_with_proteinmpnn()
    else:
        print("\n[WARNING] ProteinMPNN not available.")
        # Auto-fallback for non-interactive mode
        print("Continuing with random mutations instead.")
        from src.data.scaffolds import load_scaffold
        OUT = Path("results")
        sc = load_scaffold("1AKL", "A", OUT / "scaffolds")
        result = run_with_mutations(sc)
            
    if result is not None:
        print("\n" + "=" * 60)
        print("[OK] Experiment 02 completed successfully!")
        print("=" * 60)
        print("\nNext steps:")
        print("  - Review results in results/tables/single_shot.csv")
        print("  - Run Experiment 03: Closed-loop optimization")
        print("  - Or use Colab notebook: colab/03_closed_loop_esmf.ipynb")
    else:
        print("\n[ERROR] Experiment 02 failed. Check error messages above.")

if __name__ == "__main__":
    main()

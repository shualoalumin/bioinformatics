#!/usr/bin/env python3
"""
Quick test for Experiment 02 (without ProteinMPNN)
Uses random mutations to test ESMFold evaluation.
"""
from pathlib import Path
import pandas as pd

def main():
    print("=" * 60)
    print("Quick Test: Experiment 02 (Random Mutations)")
    print("Testing ESMFold evaluation without ProteinMPNN")
    print("=" * 60)
    
    try:
        from src.data.scaffolds import load_scaffold
        from src.generate.mutations import make_mutant_pool
        from src.evaluate.esmfold_eval import evaluate_batch
        
        OUT = Path("results")
        
        # Load scaffold
        print("\n[1/4] Loading scaffold...")
        sc = load_scaffold("1AKL", "A", OUT / "scaffolds")
        print(f"✓ Scaffold loaded: {len(sc.sequence)} residues")
        
        # Generate mutants
        print("\n[2/4] Generating random mutants...")
        seqs = make_mutant_pool([sc.sequence], n=10, rate=0.05, seed=42)
        print(f"✓ Generated {len(seqs)} mutant sequences")
        
        # Check device
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\n[3/4] Using device: {device}")
        if device == "cpu":
            print("  ⚠ CPU mode - this will be slow (5-10 min for 5 sequences)")
        
        # Evaluate with ESMFold (small batch for testing)
        print(f"\n[4/4] Evaluating with ESMFold (first 5 sequences)...")
        print("  This may take several minutes...")
        
        fold_res = evaluate_batch(
            seqs[:5],  # Only 5 for quick test
            model_id="facebook/esmfold_v1",
            device=device,
            out_dir=OUT / "pdb" / "single_shot",
            max_n=5
        )
        
        # Create results dataframe
        df = pd.DataFrame([
            {
                "sequence": r.sequence,
                "mean_plddt": r.mean_plddt,
                "pdb": str(r.pdb_path)
            }
            for r in fold_res
        ])
        
        df_sorted = df.sort_values("mean_plddt", ascending=False)
        
        print("\n" + "=" * 60)
        print("✓ Evaluation Complete!")
        print("=" * 60)
        
        print("\nResults:")
        print(df_sorted.to_string(index=False))
        
        print(f"\nStatistics:")
        print(f"  Best pLDDT:  {df['mean_plddt'].max():.2f}")
        print(f"  Mean pLDDT:  {df['mean_plddt'].mean():.2f}")
        print(f"  Min pLDDT:   {df['mean_plddt'].min():.2f}")
        
        # Save results
        (OUT / "tables").mkdir(parents=True, exist_ok=True)
        csv_path = OUT / "tables" / "single_shot.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n✓ Results saved to: {csv_path}")
        
        print("\n" + "=" * 60)
        print("Next steps:")
        print("  1. Review results in results/tables/single_shot.csv")
        print("  2. For full experiment with ProteinMPNN:")
        print("     - Use Colab: colab/02_single_shot_esmf.ipynb")
        print("     - Or install ProteinMPNN and run: python run_experiment_02.py")
        print("=" * 60)
        
    except ImportError as e:
        print(f"\n✗ Import error: {e}")
        print("\nPlease install dependencies:")
        print("  pip install -r requirements.txt")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

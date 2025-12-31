#!/usr/bin/env python3
"""
Complete experiment runner for Colab
Runs all experiments sequentially.
"""
from pathlib import Path
import pandas as pd
import sys

def run_experiment_01():
    """Experiment 01: Scaffold fetch & preprocess"""
    print("\n" + "="*60)
    print("EXPERIMENT 01: Scaffold Fetch & Preprocess")
    print("="*60)
    
    from src.data.scaffolds import load_scaffold
    OUT = Path("results")
    sc = load_scaffold("1AKL", "A", OUT / "scaffolds")
    print(f"✓ Scaffold loaded: {len(sc.sequence)} residues")
    return sc

def run_experiment_02():
    """Experiment 02: Single-shot baseline"""
    print("\n" + "="*60)
    print("EXPERIMENT 02: Single-shot Baseline")
    print("="*60)
    
    from src.data.scaffolds import load_scaffold
    from src.generate.proteinmpnn import run_proteinmpnn, read_fasta_sequences
    from src.evaluate.esmfold_eval import evaluate_batch
    
    OUT = Path("results")
    sc = load_scaffold("1AKL", "A", OUT / "scaffolds")
    
    # Generate with ProteinMPNN
    print("Generating sequences with ProteinMPNN...")
    fasta = run_proteinmpnn(
        sc.pdb_path, OUT / "mpnn_single", Path("ProteinMPNN"),
        num_seqs=50, sampling_temp=0.2, seed=42
    )
    seqs = read_fasta_sequences(fasta)
    print(f"✓ Generated {len(seqs)} sequences")
    
    # Evaluate with ESMFold
    print("Evaluating with ESMFold...")
    fold_res = evaluate_batch(
        seqs[:30], "facebook/esmfold_v1", "cuda",
        OUT / "pdb" / "single_shot", max_n=30
    )
    
    df = pd.DataFrame([
        {"sequence": r.sequence, "mean_plddt": r.mean_plddt, "pdb": str(r.pdb_path)}
        for r in fold_res
    ])
    
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "tables" / "single_shot.csv", index=False)
    print(f"✓ Results saved. Best pLDDT: {df['mean_plddt'].max():.2f}")
    return df

def run_experiment_03():
    """Experiment 03: Closed-loop optimization"""
    print("\n" + "="*60)
    print("EXPERIMENT 03: Closed-loop Optimization")
    print("="*60)
    
    from src.data.scaffolds import load_scaffold
    from src.generate.proteinmpnn import run_proteinmpnn, read_fasta_sequences
    from src.generate.mutations import make_mutant_pool
    from src.evaluate.esmfold_eval import evaluate_batch
    from src.loop.closed_loop import Candidate, run_closed_loop
    from src.metrics.figures import plot_round_curves
    
    OUT = Path("results")
    sc = load_scaffold("1AKL", "A", OUT / "scaffolds")
    seed_seq = sc.sequence[:250]
    
    has_mpnn = Path("ProteinMPNN").exists()
    
    def propose_fn(seeds, n, r):
        if r == 0:
            if has_mpnn:
                fasta = run_proteinmpnn(
                    sc.pdb_path, OUT / f"mpnn_round{r}", Path("ProteinMPNN"),
                    num_seqs=n, sampling_temp=0.2, seed=42
                )
                return read_fasta_sequences(fasta)[:n]
            return make_mutant_pool(seeds, n=n, rate=0.05, seed=42)
        return make_mutant_pool(seeds, n=n, rate=0.03, seed=42+r)
    
    def eval_fn(seqs, r):
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        max_eval = 15 if r == 0 else 10
        res = evaluate_batch(
            seqs[:max_eval], "facebook/esmfold_v1", device,
            OUT / "pdb" / f"round_{r:02d}", max_n=max_eval
        )
        return [Candidate(sequence=x.sequence, score=x.mean_plddt) for x in res]
    
    print("Starting closed-loop optimization (4 rounds)...")
    df, best = run_closed_loop([seed_seq], propose_fn, eval_fn, rounds=4, per_round=20, top_k=5)
    
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "tables" / "closed_loop.csv", index=False)
    
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    plot_round_curves(df, OUT / "figures" / "closed_loop_round_curves.png")
    
    print(f"✓ Optimization complete. Best pLDDT: {best[0].score:.2f}" if best else "✓ Complete")
    return df

def main():
    print("="*60)
    print("CLOSED-LOOP ENZYME DESIGN BENCHMARK")
    print("Complete Experiment Runner")
    print("="*60)
    
    print("\nThis will run all experiments sequentially.")
    print("Estimated time: 1-2 hours (GPU) or 4-6 hours (CPU)")
    
    response = input("\nContinue? (y/n): ").strip().lower()
    if response != 'y':
        print("Cancelled.")
        return
    
    try:
        # Experiment 01
        scaffold = run_experiment_01()
        
        # Experiment 02
        single_shot_df = run_experiment_02()
        
        # Experiment 03
        closed_loop_df = run_experiment_03()
        
        print("\n" + "="*60)
        print("✓ ALL EXPERIMENTS COMPLETED!")
        print("="*60)
        print("\nResults:")
        print("  - results/tables/single_shot.csv")
        print("  - results/tables/closed_loop.csv")
        print("  - results/figures/closed_loop_round_curves.png")
        print("\nNext: Run Experiment 04 (Surrogate-guided) or analyze results")
        
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

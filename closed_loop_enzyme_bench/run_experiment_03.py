#!/usr/bin/env python3
"""
Experiment 03: Closed-loop Optimization
Propose → Fold → Select → Mutate (iterative optimization)
"""
from pathlib import Path
import pandas as pd
import sys

def check_prerequisites():
    """Check if prerequisites are met."""
    issues = []
    
    # Check scaffold
    scaffold_file = Path("results/scaffolds/1AKL.pdb")
    if not scaffold_file.exists():
        issues.append("Scaffold not found. Run Experiment 01 first.")
    
    # Check single-shot results (optional but recommended)
    single_shot_file = Path("results/tables/single_shot.csv")
    if not single_shot_file.exists():
        print("⚠ Single-shot results not found. Will start from scaffold sequence.")
        print("  (You can run Experiment 02 first for better initialization)")
    
    return issues

def run_closed_loop_experiment():
    """Run the closed-loop optimization experiment."""
    from src.data.scaffolds import load_scaffold
    from src.generate.proteinmpnn import run_proteinmpnn, read_fasta_sequences
    from src.generate.mutations import make_mutant_pool
    from src.evaluate.esmfold_eval import evaluate_batch
    from src.loop.closed_loop import Candidate, run_closed_loop
    from src.metrics.figures import plot_round_curves
    
    OUT = Path("results")
    PDB_ID, CHAIN = "1AKL", "A"
    
    print("\n" + "=" * 60)
    print("Experiment 03: Closed-loop Optimization")
    print("=" * 60)
    
    # Load scaffold
    print("\n[Step 1] Loading scaffold...")
    sc = load_scaffold(PDB_ID, CHAIN, OUT / "scaffolds")
    seed_seq = sc.sequence[:250]  # Limit length for faster processing
    print(f"✓ Scaffold loaded: {len(seed_seq)} residues")
    print(f"  Seed sequence: {seed_seq[:50]}...")
    
    # Check for ProteinMPNN
    has_mpnn = Path("ProteinMPNN").exists()
    
    # Define proposal function
    def propose_fn(seeds, n, r):
        """Propose new sequences for round r."""
        if r == 0:
            # Round 0: Use ProteinMPNN if available, else random mutations
            if has_mpnn:
                print(f"\n  [Round {r}] Using ProteinMPNN to generate {n} sequences...")
                try:
                    fasta = run_proteinmpnn(
                        sc.pdb_path, 
                        OUT / f"mpnn_round{r}", 
                        Path("ProteinMPNN"),
                        num_seqs=n, 
                        sampling_temp=0.2, 
                        seed=42
                    )
                    seqs = read_fasta_sequences(fasta)[:n]
                    print(f"  ✓ Generated {len(seqs)} sequences")
                    return seqs
                except Exception as e:
                    print(f"  ⚠ ProteinMPNN failed: {e}")
                    print(f"  → Falling back to random mutations")
            
            # Fallback: random mutations
            print(f"\n  [Round {r}] Using random mutations to generate {n} sequences...")
            seqs = make_mutant_pool(seeds, n=n, rate=0.05, seed=42)
            print(f"  ✓ Generated {len(seqs)} sequences")
            return seqs
        else:
            # Subsequent rounds: mutate top sequences
            print(f"\n  [Round {r}] Mutating top {len(seeds)} sequences...")
            seqs = make_mutant_pool(seeds, n=n, rate=0.03, seed=42+r)
            print(f"  ✓ Generated {len(seqs)} mutant sequences")
            return seqs
    
    # Define evaluation function
    def eval_fn(seqs, r):
        """Evaluate sequences with ESMFold."""
        print(f"  [Round {r}] Evaluating {len(seqs)} sequences with ESMFold...")
        print(f"    (This may take several minutes...)")
        
        # Limit number of evaluations per round for speed
        max_eval = 15 if r == 0 else 10  # More in round 0, fewer in later rounds
        
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            max_eval = 5  # Even fewer on CPU
        
        res = evaluate_batch(
            seqs[:max_eval],  # Limit evaluations
            model_id="facebook/esmfold_v1",
            device=device,
            out_dir=OUT / "pdb" / f"round_{r:02d}",
            max_n=max_eval
        )
        
        print(f"  ✓ Evaluated {len(res)} sequences")
        return [Candidate(sequence=x.sequence, score=x.mean_plddt) for x in res]
    
    # Run closed-loop
    print("\n[Step 2] Starting closed-loop optimization...")
    print("  Configuration:")
    print("    - Rounds: 4")
    print("    - Sequences per round: 20")
    print("    - Top-k selection: 5")
    
    rounds = 4
    per_round = 20
    top_k = 5
    
    df, best_candidates = run_closed_loop(
        [seed_seq],
        propose_fn,
        eval_fn,
        rounds=rounds,
        per_round=per_round,
        top_k=top_k
    )
    
    # Display results
    print("\n" + "=" * 60)
    print("✓ Closed-loop optimization complete!")
    print("=" * 60)
    
    print("\nResults by round:")
    print(df.to_string(index=False))
    
    if best_candidates:
        best = best_candidates[0]
        print(f"\nBest sequence found:")
        print(f"  pLDDT: {best.score:.2f}")
        print(f"  Sequence: {best.sequence[:80]}...")
    
    # Save results
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "tables" / "closed_loop.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Results saved to: {csv_path}")
    
    # Generate plot
    try:
        (OUT / "figures").mkdir(parents=True, exist_ok=True)
        plot_path = OUT / "figures" / "closed_loop_round_curves.png"
        plot_round_curves(df, plot_path)
        print(f"✓ Plot saved to: {plot_path}")
    except Exception as e:
        print(f"⚠ Could not generate plot: {e}")
    
    return df, best_candidates

def main():
    print("=" * 60)
    print("Experiment 03: Closed-loop Optimization")
    print("Propose → Fold → Select → Mutate")
    print("=" * 60)
    
    # Check prerequisites
    issues = check_prerequisites()
    if issues:
        print("\n⚠ Prerequisites not met:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nPlease run:")
        print("  python run_experiment_01.py  # For scaffold")
        return
    
    # Check if single-shot results exist
    single_shot = Path("results/tables/single_shot.csv")
    if single_shot.exists():
        print("\n✓ Found single-shot results (Experiment 02)")
        print("  Will use scaffold sequence as starting point")
    else:
        print("\n⚠ Single-shot results not found")
        print("  Will start from scaffold sequence")
        print("  (Run Experiment 02 first for better initialization)")
    
    # Ask for confirmation (since this takes time)
    print("\n⚠ This experiment will:")
    print("  - Run 4 rounds of optimization")
    print("  - Evaluate ~40-60 sequences with ESMFold")
    print("  - Take 30-60 minutes (GPU) or 2-4 hours (CPU)")
    
    response = input("\nContinue? (y/n): ").strip().lower()
    if response != 'y':
        print("\nCancelled. Use Colab for faster execution with GPU.")
        return
    
    # Run experiment
    try:
        df, best = run_closed_loop_experiment()
        
        print("\n" + "=" * 60)
        print("✓ Experiment 03 completed successfully!")
        print("=" * 60)
        print("\nNext steps:")
        print("  - Review results in results/tables/closed_loop.csv")
        print("  - Check plot in results/figures/closed_loop_round_curves.png")
        print("  - Run Experiment 04: Surrogate-guided optimization")
        print("  - Or use Colab: colab/04_surrogate_active_learning.ipynb")
        
    except KeyboardInterrupt:
        print("\n\n⚠ Experiment interrupted by user")
        print("Partial results may be saved in results/ directory")
    except Exception as e:
        print(f"\n✗ Experiment failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

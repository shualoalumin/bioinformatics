from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

def plot_round_curves(df: pd.DataFrame, out_path: Path):
    """Plot best and mean pLDDT scores across rounds."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    
    # Filter out None values
    df_plot = df[df["best"].notna() & df["mean"].notna()].copy()
    
    if len(df_plot) > 0:
        plt.plot(df_plot["round"], df_plot["best"], marker="o", label="best", linewidth=2)
        plt.plot(df_plot["round"], df_plot["mean"], marker="s", label="mean", linewidth=2)
        plt.xlabel("Round", fontsize=12)
        plt.ylabel("Mean pLDDT", fontsize=12)
        plt.title("Closed-loop Optimization Progress", fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
    else:
        plt.text(0.5, 0.5, "No data to plot", ha="center", va="center", transform=plt.gca().transAxes)
    
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

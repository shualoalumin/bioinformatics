from __future__ import annotations
from typing import List
import numpy as np

def success_rate(scores: List[float], thr: float=80.0)->float:
    """Calculate success rate (fraction of scores >= threshold)."""
    if not scores: return float("nan")
    return float(np.mean([s>=thr for s in scores]))

def avg_pairwise_hamming(seqs: List[str], max_pairs: int=2000)->float:
    """Calculate average pairwise Hamming distance between sequences."""
    if len(seqs)<2: return 0.0
    total=0; pairs=0
    for i in range(len(seqs)):
        for j in range(i+1,len(seqs)):
            a,b=seqs[i],seqs[j]
            if len(a)!=len(b): continue
            total += sum(x!=y for x,y in zip(a,b))
            pairs += 1
            if pairs>=max_pairs: return float(total/pairs)
    return float(total/pairs) if pairs > 0 else 0.0

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple
import pandas as pd
from ..metrics.metrics import success_rate, avg_pairwise_hamming

@dataclass
class Candidate:
    sequence: str
    score: float

def select_top(cands: Sequence[Candidate], k: int)->List[Candidate]:
    return sorted(cands, key=lambda c:c.score, reverse=True)[:k]

def run_closed_loop(seed_sequences: List[str],
                    propose_fn: Callable[[List[str],int,int],List[str]],
                    eval_fn: Callable[[List[str],int],List[Candidate]],
                    rounds: int, per_round: int, top_k: int)->Tuple[pd.DataFrame, List[Candidate]]:
    hist=[]; best=None; seeds=seed_sequences[:]
    for r in range(rounds):
        props=propose_fn(seeds, per_round, r)
        cands=eval_fn(props, r)
        top=select_top(cands, top_k)
        scores=[c.score for c in cands]
        hist.append({
            "round": r, "n": len(cands),
            "best": max(scores) if scores else None,
            "mean": float(sum(scores)/len(scores)) if scores else None,
            "success_rate_80": success_rate(scores, 80.0),
            "avg_pairwise_hamming(top_k)": avg_pairwise_hamming([c.sequence for c in top]),
        })
        if top and (best is None or top[0].score>best.score): best=top[0]
        seeds=[c.sequence for c in top]
    return pd.DataFrame(hist), ([best] if best else [])

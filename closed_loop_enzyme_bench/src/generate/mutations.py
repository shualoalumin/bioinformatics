from __future__ import annotations
import random
from typing import List, Sequence
AA=list("ACDEFGHIKLMNPQRSTVWY")

def mutate_sequence(seq: str, rate: float, rng: random.Random) -> str:
    out=list(seq)
    for i,ch in enumerate(out):
        if rng.random()<rate:
            out[i]=rng.choice([a for a in AA if a!=ch])
    return "".join(out)

def make_mutant_pool(seeds: Sequence[str], n: int, rate: float, seed: int=0) -> List[str]:
    rng=random.Random(seed)
    return [mutate_sequence(rng.choice(seeds), rate, rng) for _ in range(n)]

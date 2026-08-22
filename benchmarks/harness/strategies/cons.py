"""Consensus (majority vote over extracted, normalised answers) and the
unbiased cons@k curve derived from ONE run of K samples per item."""

from __future__ import annotations

import itertools
import math
import random
from collections import Counter
from typing import Callable, Dict, List, Optional, Sequence, Tuple


def majority(normalized: Sequence[Optional[str]]) -> Tuple[Optional[str], int]:
    """Most common non-None answer; ties broken by first occurrence."""
    votes = Counter(a for a in normalized if a is not None)
    if not votes:
        return None, 0
    best = max(votes.values())
    for a in normalized:
        if a is not None and votes[a] == best:
            return a, best
    return None, 0


def cons_correct(normalized: Sequence[Optional[str]], correct: Sequence[bool]) -> bool:
    """Is the majority answer a correct one? (any sample with that normalised
    answer is graded correct)."""
    sel, _ = majority(normalized)
    if sel is None:
        return False
    return any(c for a, c in zip(normalized, correct) if a == sel)


def cons_at_k_curve(per_item: List[Tuple[List[Optional[str]], List[bool]]],
                    ks: Sequence[int], n_sub: int = 200, seed: int = 1234) -> Dict[int, float]:
    """per_item: [(normalized answers, correct flags)] with K samples each.
    For k < K average over random k-subsets (all subsets if few)."""
    rng = random.Random(seed)
    out: Dict[int, float] = {}
    for k in ks:
        accs = []
        for norm, corr in per_item:
            K = len(norm)
            if k >= K:
                accs.append(float(cons_correct(norm, corr)))
                continue
            idxs = list(range(K))
            n_comb = math.comb(K, k)
            if n_comb <= n_sub:
                subs = list(itertools.combinations(idxs, k))
            else:
                subs = [tuple(rng.sample(idxs, k)) for _ in range(n_sub)]
            accs.append(sum(cons_correct([norm[i] for i in s], [corr[i] for i in s]) for s in subs) / len(subs))
        out[k] = sum(accs) / max(len(accs), 1)
    return out


def avg_at_1(per_item: List[Tuple[List[Optional[str]], List[bool]]]) -> float:
    vals = [sum(c) / len(c) for _, c in per_item if c]
    return sum(vals) / max(len(vals), 1)


def pass_at_k(per_item: List[Tuple[List[Optional[str]], List[bool]]], k: int) -> float:
    """Chen et al. unbiased estimator (diagnostic only: an oracle selector)."""
    vals = []
    for _, corr in per_item:
        n, c = len(corr), sum(corr)
        if n - c < k:
            vals.append(1.0)
        else:
            vals.append(1.0 - math.comb(n - c, k) / math.comb(n, k))
    return sum(vals) / max(len(vals), 1)

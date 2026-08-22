"""Statistics and aggregate reporting for harness runs."""

from __future__ import annotations

import math
import random
from typing import Dict, List, Sequence, Tuple


def wilson(p: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_mean_ci(values: Sequence[float], n_boot: int = 10_000, seed: int = 1234,
                      alpha: float = 0.05) -> Tuple[float, float]:
    """Percentile bootstrap CI of the mean over items (item-level resampling)."""
    vals = list(values)
    if not vals:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(vals)
    means = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += vals[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    lo = means[int(alpha / 2 * n_boot)]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return (lo, hi)


def paired_diff_ci(a: Sequence[float], b: Sequence[float], n_boot: int = 10_000,
                   seed: int = 1234) -> Tuple[float, Tuple[float, float]]:
    """Mean of (a_i - b_i) with an item-level bootstrap CI (paired)."""
    diffs = [x - y for x, y in zip(a, b)]
    return (sum(diffs) / max(len(diffs), 1), bootstrap_mean_ci(diffs, n_boot, seed))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from discordant counts b (A right, B
    wrong) and c (A wrong, B right)."""
    from scipy.stats import binomtest
    n = b + c
    if n == 0:
        return 1.0
    return float(binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue)


def pct(x: float, nd: int = 1) -> str:
    return f"{100 * x:.{nd}f}"


def three_way_row(d: Dict) -> str:
    """One markdown row: accuracy (CI) | compute | latency/memory."""
    return ("| {model} {quant} | k={k} | {acc} [{lo}, {hi}] | {tok_item:,.0f} / p95 {tok_p95:,.0f} | "
            "{tok_correct:,.0f} | {wall:.0f}s | {agg_tps:.0f} tok/s | {peak:.1f} GB |").format(
        model=d.get("model_short"), quant=d.get("quant"), k=d.get("k"),
        acc=pct(d["accuracy"]), lo=pct(d["accuracy_ci"][0]), hi=pct(d["accuracy_ci"][1]),
        tok_item=d.get("completion_tokens_per_item_mean", 0), tok_p95=d.get("completion_tokens_per_item_p95", 0),
        tok_correct=d.get("tokens_per_correct", 0), wall=d.get("wall_s_per_item", 0),
        agg_tps=d.get("aggregate_tok_s", 0), peak=d.get("peak_rss_gb", 0) or 0)

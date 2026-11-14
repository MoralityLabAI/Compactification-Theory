"""Paired statistics, standard library only.

Clustered intervals resample whole clusters (source family x topology cells)
with replacement, keeping each case's pairing intact.
"""
from __future__ import annotations

from math import comb, sqrt
import random

BOOT = 10000
BOOT_SEED = 2026111104


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact(b, c):
    """Two-sided exact binomial test on discordant pairs (b: A only, c: B only)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def quantile(values, q):
    xs = sorted(values)
    if not xs:
        return None
    pos = (len(xs) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def paired(a, b, clusters, *, boot=BOOT, seed=BOOT_SEED):
    """a, b: 0/1 outcomes per case (same order); clusters: label per case.

    Returns the difference in totals (a - b), discordant counts, exact McNemar
    p and a cluster-bootstrap 95% interval on the per-case mean difference.
    """
    if not (len(a) == len(b) == len(clusters)):
        raise ValueError("paired vectors differ in length")
    n = len(a)
    d = [x - y for x, y in zip(a, b)]
    groups = {}
    for i, g in enumerate(clusters):
        groups.setdefault(g, []).append(i)
    keys = sorted(groups)
    sums = {g: (sum(d[i] for i in groups[g]), len(groups[g])) for g in keys}
    rng = random.Random(seed)
    stats = []
    for _ in range(boot):
        tot = cnt = 0
        for _ in keys:
            s, m = sums[keys[rng.randrange(len(keys))]]
            tot += s
            cnt += m
        stats.append(tot / cnt)
    stats.sort()
    b_only = sum(1 for x, y in zip(a, b) if x and not y)
    c_only = sum(1 for x, y in zip(a, b) if y and not x)
    return {"n": n, "a_total": sum(a), "b_total": sum(b), "difference": sum(d),
            "mean_difference": sum(d) / n if n else 0.0,
            "a_only": b_only, "b_only": c_only, "mcnemar_p": mcnemar_exact(b_only, c_only),
            "cluster_ci95": [stats[int(0.025 * boot)], stats[int(0.975 * boot) - 1]],
            "clusters": len(keys), "bootstrap": boot, "seed": seed}


def proportion(k, n):
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": k / n if n else 0.0, "wilson95": [lo, hi]}

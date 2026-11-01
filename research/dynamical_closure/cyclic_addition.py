"""Exact Z_p quotient and one-hot risk; numerical phase closure sanity checks."""
from collections import defaultdict
from fractions import Fraction
from math import cos, pi, sin


def states(p):
    if p < 2:
        raise ValueError("modulus must be at least two")
    return tuple((a, b) for a in range(p) for b in range(p))


def task(a, b, p):
    return (a + b) % p


def shift(a, b, u, v, p):
    return ((a + u) % p, (b + v) % p)


def phase(s, j, p):
    """(cos, sin) phase; proof concerns exact roots of unity, not floating points."""
    angle = 2 * pi * j * s / p
    return (cos(angle), sin(angle))


def rotate(z, c, j, p):
    ct, st = phase(c, j, p)
    x, y = z
    return (ct * x - st * y, st * x + ct * y)


def onehot_risk(codes, labels, weights=None):
    """Exact E||e_Y-E[e_Y|C]||^2, with fixed one-hot target scale."""
    if len(codes) != len(labels) or not labels:
        raise ValueError("matching nonempty codes and labels required")
    if weights is None:
        weights = (1,) * len(labels)
    if len(weights) != len(labels):
        raise ValueError("weight count mismatch")
    ws = tuple(Fraction(w) for w in weights)
    if any(w < 0 for w in ws) or not sum(ws):
        raise ValueError("weights must be nonnegative with positive total")
    groups = defaultdict(lambda: defaultdict(Fraction))
    for z, y, w in zip(codes, labels, ws):
        groups[z][y] += w
    # Within a fiber of total mass m, squared-error optimum has loss
    # m - sum_y (mass_y)^2/m; normalize by total probability mass.
    return (sum(sum(counts.values()) - sum(w * w for w in counts.values()) / sum(counts.values())
                for counts in groups.values() if sum(counts.values())) / sum(ws))


def minimum_state_bits(p):
    return (p - 1).bit_length()

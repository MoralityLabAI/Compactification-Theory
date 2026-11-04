"""UG-LAT-02: exact recursive q=2 gadget and all-linear-lift Bellman audit.

Bits encode F_2^3. Every parent has all 7*168=1176 children (A,J).
Finite transfer kernels integrate this fixed tree exactly at any depth.
The arbitrary-alphabet latent-gadget theorem remains outside this module.
"""
from __future__ import annotations

from collections import Counter
from fractions import Fraction as F
from functools import cache
from itertools import combinations, permutations, product
import argparse
import hashlib
import json
from pathlib import Path


def word(x: int, width: int = 3) -> int:
    if type(x) is not int or not 0 <= x < 1 << width:
        raise ValueError("invalid binary vector")
    return x


def linear(columns: tuple[int, ...], x: int) -> int:
    out = 0
    for i, column in enumerate(columns):
        if x >> i & 1:
            out ^= column
    return out


def span(columns: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(sorted({linear(columns, x) for x in range(1 << len(columns))}))


def basis(space: tuple[int, ...]) -> tuple[int, ...]:
    out: tuple[int, ...] = ()
    for x in sorted(space):
        if x not in span(out):
            out += (x,)
    return out


def dot(x: int, y: int) -> int:
    return (x & y).bit_count() & 1


def quadratic(x: int) -> int:
    word(x)
    a, b, c = (x >> i & 1 for i in range(3))
    return (b * c) | (a * c << 1) | (a * b << 2)


@cache
def spaces() -> tuple[tuple[int, ...], ...]:
    return tuple(sorted({span(s) for r in range(4)
                         for s in combinations(range(1, 8), r)},
                        key=lambda s: (len(s), s)))


@cache
def gl3() -> tuple[tuple[int, ...], ...]:
    return tuple(g for g in permutations(range(1, 8), 3) if len(span(g)) == 8)


@cache
def update_basis(v: int) -> tuple[int, ...]:
    word(v)
    if not v:
        raise ValueError("zero cannot name a projective line")
    return basis(tuple(a | ((quadratic(a) ^ w) << 3)
                       for a in (0, v) for w in range(8) if dot(v, w) == 0))


@cache
def children() -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(linear(update_basis(v), col) for col in g)
                 for v in range(1, 8) for g in gl3())


def aggregate(values: tuple[int, ...]) -> int:
    if len(values) != len(children()):
        raise ValueError("height-one input needs exactly 1176 child symbols")
    total = 0
    for j, value in zip(children(), values):
        total ^= linear(j, word(value))
    return total


def height_one(values: tuple[int, ...]) -> int:
    total = aggregate(values)
    return (total >> 3) ^ quadratic(total & 7)


@cache
def noise_row() -> tuple[F, ...]:
    """Conditional output-difference law for any nonzero child difference.

    Uniform J sends it uniformly onto U_A minus zero. The initial x is
    independently uniform; initial y cancels. Enumerate all 392 outcomes.
    """
    counts: Counter[int] = Counter()
    for v, x in product(range(1, 8), range(8)):
        for u in span(update_basis(v)):
            if u:
                a, b = u & 7, u >> 3
                counts[b ^ quadratic(x ^ a) ^ quadratic(x)] += 1
    total = sum(counts.values())
    return tuple(F(counts[y], total) for y in range(8))


def restriction_rows(s: tuple[int, ...], gamma: tuple[int, ...], v: int):
    sb = basis(s)
    if len(gamma) != len(sb):
        raise ValueError("gamma must specify one image for each domain basis vector")
    for g in gamma:
        word(g)
    return tuple(sum((dot(g, u & 7) ^ dot(z, u >> 3)) << j
                     for j, u in enumerate(update_basis(v)))
                 for z, g in zip(sb, gamma))


@cache
def rank_options(s: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    """Every outgoing-rank histogram over the 7 lines, for all linear gamma.

    Gamma is selected BEFORE the outgoing line. Enumerating separately at
    each line would give an inadmissible future-informed optimizer.
    """
    if s not in spaces():
        raise ValueError("expected a complete subspace of F_2^3")
    r = len(basis(s))
    options = set()
    for gamma in product(range(8), repeat=r):
        hist = [0] * 4
        for v in range(1, 8):
            k = len(basis(restriction_rows(s, gamma, v)))
            hist[k] += 1
        options.add(tuple(hist))
    return tuple(sorted(options))


def evolution(depth: int = 16) -> list[dict]:
    if type(depth) is not int or not 0 <= depth <= 128:
        raise ValueError("depth must be an integer from 0 through 128")
    ss = spaces()
    ranks = {s: len(basis(s)) for s in ss}
    values = {s: F(1) - F(1, len(s)) for s in ss}
    distribution = (F(1, 8),) * 8
    result = []
    for t in range(depth + 1):
        result.append({"depth": t, "nonlinear_change": str(1 - distribution[0]),
                       "minimum_full_lift_detection": str(values[tuple(range(8))]),
                       "rank_mean_detection": [str(sum(values[s] for s in ss if ranks[s] == r)
                                                   / sum(ranks[s] == r for s in ss))
                                               for r in range(4)]})
        means = [sum(values[s] for s in ss if ranks[s] == r)
                 / sum(ranks[s] == r for s in ss) for r in range(4)]
        values = {s: min(sum(F(h[k], 7) * means[k] for k in range(4))
                         for h in rank_options(s)) for s in ss}
        moving = 1 - distribution[0]
        distribution = tuple((distribution[0] if y == 0 else F(0))
                             + moving * noise_row()[y] for y in range(8))
    return result


def height_one_lift_detection(gamma: tuple[int, ...]) -> F:
    if len(gamma) != 3:
        raise ValueError("a full lift requires three gamma images")
    for g in gamma:
        word(g)
    count = 0
    for j, delta in product(children(), range(8)):
        u = linear(j, delta)
        observed = (u >> 3) ^ sum(dot(g, u & 7) << i for i, g in enumerate(gamma))
        count += observed != 0
    return F(count, len(children()) * 8)


def receipt() -> dict:
    here = Path(__file__)
    return {"schema": "UG-LAT-02/q2-recursion-v1",
            "source_sha256": hashlib.sha256(here.read_bytes()).hexdigest(),
            "upstream_latent_blob": "8e1c7ad0b873bd91f77e0611dae3d689b7c180a9",
            "field_size": 2, "subspaces": len(spaces()), "isomorphisms_per_line": len(gl3()),
            "children_per_parent": len(children()),
            "linear_extension_maps": sum(8 ** len(basis(s)) for s in spaces()),
            "nonzero_child_noise_kernel": [str(x) for x in noise_row()],
            "rank_histogram_options": [{"subspace": list(s), "options": rank_options(s)} for s in spaces()],
            "depths": evolution(16),
            "interpretation": "Exact q=2 fixed-tree recursion; general high-rank theorem excluded."}


def verify_receipt(candidate: dict) -> None:
    # Normalize tuple/list representation before checking a JSON round trip.
    if candidate != json.loads(json.dumps(receipt())):
        raise ValueError("receipt differs from exact regeneration")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    parser.add_argument("--check", type=Path)
    args = parser.parse_args()
    if args.check:
        verify_receipt(json.loads(args.check.read_text()))
        print("UG-LAT-02 receipt verified")
    else:
        text = json.dumps(receipt(), indent=2) + "\n"
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(text)
        else:
            print(text, end="")

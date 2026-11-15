"""NOR-02: complete Crouzeix filtered-orbit geometry, exact disk envelope.

An explicitly noncommuting matrix-coefficient polynomial C(z) is combined
with (z-q)**order to suppress the unique maximal-modulus boundary point
of the numerical range. The curvature-tail theorem is written separately
in MathCan; this is a stdlib rational/float experimental reference.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
from math import comb, exp, fsum, log, sqrt
import json
from pathlib import Path

from orbit_sketch import BlockLoop, JordanBlock, gaussian_lp_sketch, norm, subtract, window_distortion


A = Fraction(3, 8)
R = Fraction(1, 2)
Q = A + R


def nested_disk_model(block_count: int = 32) -> BlockLoop:
    if not 1 <= block_count <= 32:
        raise ValueError("block count must be 1..32 to keep positive coefficients")
    blocks = tuple(
        JordanBlock(A - Fraction(i, 1024),
                    2 * (R - Fraction(i, 1024)) - Fraction((5 * i) % 7, 256))
        for i in range(block_count)
    )
    model = BlockLoop(blocks)
    if any(abs(b.a - A) + b.b / 2 > R for b in model.blocks):
        raise AssertionError("the rational disk containment certificate failed")
    if model.blocks[0].a != A or model.blocks[0].b / 2 != R:
        raise AssertionError("the first block must realize the outer disk")
    if model.q != Q:
        raise AssertionError("the numerical radius is wrong")
    return model


def curvature_gain(t: int, order: int, a: Fraction = A,
                   radius: Fraction = R) -> float:
    """Majorant of sup |z|**t |z-q|**order on D(a,radius).

    (radius/a)^(order/2) * q^(t+order) *
      sqrt(t^t order^order / (t+order)^(t+order)).
    """
    if t < 0 or order < 1 or a <= 0 or radius <= 0 or a + radius >= 1:
        raise ValueError("required: t>=0, order>=1, a,radius>0, a+radius<1")
    lg = (t * log(t) if t else 0.0) + order * log(order) - (t + order) * log(t + order)
    return (float(radius / a) ** (order / 2) *
            float(a + radius) ** (t + order) * exp(lg / 2))


def filter_coefficient(order: int, k: int) -> Fraction:
    if not 0 <= k <= order:
        return Fraction(0)
    return Fraction(comb(order, k)) * (-Q) ** (order - k)


def scalar_power_filter(model: BlockLoop, x: list[float], t: int, order: int,
                        extra_power: int = 0) -> list[float]:
    """Closed Jordan functional calculus for z^(t+extra_power)(z-q)^order."""
    if order < 1 or t < 0 or extra_power < 0 or len(x) != model.dimension:
        raise ValueError("invalid filter input")
    s = t + extra_power
    ans = []
    for i, block in enumerate(model.blocks):
        aa, bb, qq = float(block.a), float(block.b), float(Q)
        zero = aa - qq
        f = aa ** s * zero ** order
        derivative = ((s * aa ** (s - 1) * zero ** order) if s else 0.0)
        derivative += order * aa ** s * zero ** (order - 1)
        xx, yy = x[2 * i:2 * i + 2]
        ans.extend((f * xx + bb * derivative * yy, f * yy))
    return ans


def matrix_filtered_state(model: BlockLoop, x: list[float], t: int,
                          order: int, u: tuple[float, float]) -> list[float]:
    """(A^t tensor I)P[A](x tensor u) for P(z)=(z-q)^r [[0,1],[z,0]].

    Output is base-coordinate-major tensor ordering. The two coefficient
    matrices [[0,1],[0,0]] and [[0,0],[1,0]] do not commute.
    """
    v0 = scalar_power_filter(model, x, t, order)
    v1 = scalar_power_filter(model, x, t, order, 1)
    return [z for c0, c1 in zip(v0, v1, strict=True)
            for z in (u[1] * c0, u[0] * c1)]


def convolution_filtered_state(model: BlockLoop, x: list[float], t: int,
                               order: int, u: tuple[float, float]) -> list[float]:
    """Independent binomial-coefficient evaluation of the tensor polynomial."""
    n = model.dimension
    ans = [0.0] * (2 * n)
    for j in range(order + 1):
        weight = float(filter_coefficient(order, j))
        f0 = model.exact_power_action(x, t + j)
        f1 = model.exact_power_action(x, t + j + 1)
        for i in range(n):
            ans[2 * i] += weight * u[1] * f0[i]
            ans[2 * i + 1] += weight * u[0] * f1[i]
    return ans


def run_case(*, block_count: int = 32, cutoff: int = 18, order: int = 2,
             k: int = 24, p: int = 4, seed: int = 325094,
             tail_checks: int = 18) -> dict:
    if cutoff < 1 or tail_checks < 1 or k < 1:
        raise ValueError("cutoff, tail_checks, and k must be positive")
    model = nested_disk_model(block_count)
    x = [v for _ in model.blocks for v in (0.0, 1 / sqrt(block_count))]
    u = (1 / sqrt(2), 1 / sqrt(2))
    vnorm = norm(x) * norm(list(u))
    source = [[0.0] * (2 * model.dimension)] + [
        matrix_filtered_state(model, x, t, order, u) for t in range(cutoff)]
    embedded = gaussian_lp_sketch(source, k, p, seed)
    error, relative = window_distortion(source, embedded, p)
    anchor_error = max(abs(norm(v) - norm(im, p)) for v, im in zip(source, embedded, strict=True))
    # Complete matrix-valued Crouzeix with a single factor 2, then the
    # numerical-range disk's exposed-point curvature inequality.
    radius = 2 * curvature_gain(cutoff, order) * vnorm
    new_bound = max(error, anchor_error + radius, (1 + float(Q)) * radius)
    old_stitching_bound = error + 2 * radius
    # Dropping curvature still uses a single complete Crouzeix estimate:
    # sup_{disk}||(z-q)^r C(z)|| = (2*R)^r; ||C(z)||=1 since |z|<1.
    disk_only_radius = 2 * float(Q) ** cutoff * (2 * float(R)) ** order * vnorm
    # Triangle summing coefficient monomials (loses the cancellations):
    triangle_radius = (2 * float(Q) ** cutoff * (1 + float(Q)) *
                       (2 * float(Q)) ** order * vnorm)
    probes = source + [matrix_filtered_state(model, x, t, order, u)
                       for t in range(cutoff, cutoff + tail_checks)]
    clips = embedded + [[0.0] * k for _ in range(tail_checks)]
    max_probe = 0.0
    for i in range(len(probes)):
        for j in range(i + 1, len(probes)):
            d = norm(subtract(probes[i], probes[j]))
            dh = norm(subtract(clips[i], clips[j]), p)
            max_probe = max(max_probe, abs(d-dh))
    result = {
        "family": "NOR-02", "source_papers": ["OpenAI Math #094", "OpenAI Math #325"],
        "block_count": block_count, "ambient_hilbert_dimension": 2 * model.dimension,
        "sketch_dimension": k, "coefficient_dimension": 2, "order": order,
        "cutoff": cutoff, "additional_tail_probe_states": tail_checks,
        "p": p, "seed": seed,
        "enclosing_disk_a": str(A), "enclosing_disk_radius": str(R),
        "numerical_radius": str(Q), "matrix_coefficients_noncommute": True,
        "prefix_max_absolute_error": error,
        "prefix_max_relative_error": relative,
        "zero_anchor_max_absolute_error": anchor_error,
        "curvature_tail_radius": radius,
        "complete_crouzeix_disk_only_tail_radius": disk_only_radius,
        "coefficient_triangle_tail_radius": triangle_radius,
        "sharp_stitch_bound": new_bound,
        "legacy_stitch_bound": old_stitching_bound,
        "finite_tail_probe_max_error": max_probe,
        "finite_probe_below_bound": max_probe <= new_bound + 1e-10,
        "scope": "exact rational disk geometry, paper-level analytic filtered tail, sampled Gaussian Lp finite prefix; reported errors use floating point",
    }
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--order", type=int, default=2)
    ap.add_argument("--cutoff", type=int, default=18)
    ap.add_argument("--k", type=int, default=24)
    ap.add_argument("--p", type=int, choices=(2, 4), default=4)
    ap.add_argument("--seed", type=int, default=325094)
    args = ap.parse_args()
    receipt = run_case(order=args.order, cutoff=args.cutoff, k=args.k,
                       p=args.p, seed=args.seed)
    if args.check:
        assert receipt["finite_probe_below_bound"]
        assert receipt["sharp_stitch_bound"] <= receipt["legacy_stitch_bound"] + 1e-10
        assert receipt["curvature_tail_radius"] < receipt["complete_crouzeix_disk_only_tail_radius"]
        assert receipt["sketch_dimension"] < receipt["ambient_hilbert_dimension"]
    blob = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(blob)
    else:
        print(blob, end="")


if __name__ == "__main__":
    main()

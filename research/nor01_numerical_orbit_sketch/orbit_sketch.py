"""NOR-01: finite-window Lp sketches with an exact numerical-radius tail gate.

Standard library only. This is a deterministic experiment on explicit block-Jordan
linear recurrences, not a neural-model performance result and not a verifier for
OpenAI's full dimension reduction or Crouzeix theorems.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction
import json
import math
from pathlib import Path
import random


@dataclass(frozen=True)
class JordanBlock:
    a: Fraction
    b: Fraction

    def __post_init__(self) -> None:
        if self.a < 0 or self.b < 0:
            raise ValueError("NOR-01 uses nonnegative rational Jordan entries")

    @property
    def numerical_radius(self) -> Fraction:
        # W([[a,b],[0,a]]) is the complex disk centered at a of radius b/2.
        return self.a + self.b / 2

    @property
    def spectral_radius(self) -> Fraction:
        return self.a

    @property
    def operator_norm(self) -> float:
        a, b = float(self.a), float(self.b)
        return math.sqrt(a * a + b * b / 2 + b * math.sqrt(b * b + 4 * a * a) / 2)


@dataclass(frozen=True)
class BlockLoop:
    blocks: tuple[JordanBlock, ...]

    def __post_init__(self) -> None:
        if not self.blocks:
            raise ValueError("at least one block is required")

    @property
    def dimension(self) -> int:
        return 2 * len(self.blocks)

    @property
    def q(self) -> Fraction:
        # Block-diagonal W(A) is the convex hull of individual W(block).
        return max(b.numerical_radius for b in self.blocks)

    @property
    def eigenvalue_radius(self) -> Fraction:
        return max(b.spectral_radius for b in self.blocks)

    @property
    def operator_norm(self) -> float:
        return max(b.operator_norm for b in self.blocks)

    def require_tail_certificate(self) -> None:
        if self.q >= 1:
            raise ValueError("numerical radius is not strictly below one")

    def step(self, state: list[float]) -> list[float]:
        if len(state) != self.dimension:
            raise ValueError("state has wrong width")
        out = []
        for i, b in enumerate(self.blocks):
            x, y = state[2 * i:2 * i + 2]
            a, c = float(b.a), float(b.b)
            out.extend((a * x + c * y, a * y))
        return out

    def exact_power_action(self, state: list[float], t: int) -> list[float]:
        if t < 0 or len(state) != self.dimension:
            raise ValueError("invalid time or state width")
        out = []
        for i, b in enumerate(self.blocks):
            x, y = state[2 * i:2 * i + 2]
            a, c = float(b.a), float(b.b)
            diagonal = a ** t
            off_diagonal = 0.0 if t == 0 else t * c * a ** (t - 1)
            out.extend((diagonal * x + off_diagonal * y, diagonal * y))
        return out


def norm(v: list[float], p: int = 2) -> float:
    if p not in (2, 4):
        raise ValueError("supported measured p values are 2 and 4")
    return sum(abs(x) ** p for x in v) ** (1.0 / p)


def subtract(a: list[float], b: list[float]) -> list[float]:
    return [x - y for x, y in zip(a, b, strict=True)]


def gaussian_lp_sketch(points: list[list[float]], k: int, p: int,
                       seed: int) -> list[list[float]]:
    """Common linear Gaussian map; p=4 uses E|N(0,1)|^4 = 3.

    For p=4 this is an empirical version of the isometric Gaussian
    inclusion of real Hilbert space into L4, *not* the general Lp theorem.
    """
    if k <= 0 or p not in (2, 4) or not points:
        raise ValueError("positive k, p in {2,4} and points required")
    width = len(points[0])
    if any(len(x) != width for x in points):
        raise ValueError("inconsistent point dimensions")
    rng = random.Random(seed)
    scale = (k * (1 if p == 2 else 3)) ** (-1.0 / p)
    out = [[] for _ in points]
    for _ in range(k):
        weights = [rng.gauss(0.0, 1.0) for _ in range(width)]
        for image, v in zip(out, points, strict=True):
            image.append(scale * math.fsum(w * z for w, z in zip(weights, v, strict=True)))
    return out


def window_distortion(original: list[list[float]], mapped: list[list[float]],
                      p: int) -> tuple[float, float]:
    """Actual maximal additive and relative errors over *all* prefix pairs."""
    abs_error = rel_error = 0.0
    for i in range(len(original)):
        for j in range(i + 1, len(original)):
            source = norm(subtract(original[i], original[j]))
            image = norm(subtract(mapped[i], mapped[j]), p)
            abs_error = max(abs_error, abs(source - image))
            if source > 1e-14:
                rel_error = max(rel_error, abs(source - image) / source)
    return abs_error, rel_error


def run_case(*, block_count: int = 32, cutoff: int = 18, k: int = 24,
             p: int = 4, seed: int = 325094, tail_checks: int = 16) -> dict:
    if block_count < 1 or cutoff < 1 or tail_checks < 1:
        raise ValueError("positive block_count, cutoff and tail_checks required")
    blocks = tuple(JordanBlock(Fraction(3, 8) - Fraction(i, 1024),
                               Fraction(1) - Fraction((i * 5) % 7, 256))
                   for i in range(block_count))
    model = BlockLoop(blocks)
    model.require_tail_certificate()
    x0 = [z for _ in range(block_count) for z in (0.0, 1 / math.sqrt(block_count))]
    original = [[0.0] * model.dimension] + [model.exact_power_action(x0, t)
                                               for t in range(cutoff)]
    mapped = gaussian_lp_sketch(original, k, p, seed)
    e_window, rel_window = window_distortion(original, mapped, p)
    # The zero point is index 0. The tail at index cutoff+1 starts at time cutoff.
    tail_radius = 2 * float(model.q) ** cutoff * norm(x0)
    global_bound = e_window + 2 * tail_radius
    true_points = original + [model.exact_power_action(x0, t)
                              for t in range(cutoff, cutoff + tail_checks)]
    clipped_points = mapped + [[0.0] * k for _ in range(tail_checks)]
    worst_global = 0.0
    for i in range(len(true_points)):
        for j in range(i + 1, len(true_points)):
            source = norm(subtract(true_points[i], true_points[j]))
            image = norm(subtract(clipped_points[i], clipped_points[j]), p)
            worst_global = max(worst_global, abs(source - image))
    first_step_gain = norm(model.step(x0)) / norm(x0)
    spectral_shortcut_fails = norm(model.step(x0)) > 2 * float(model.eigenvalue_radius) * norm(x0)
    result = {
        "family": "NOR-01", "papers": ["OpenAI Math 094", "OpenAI Math 325"],
        "model": "distinct rational 2x2 Jordan blocks", "block_count": block_count,
        "ambient_dimension": model.dimension, "sketch_dimension": k,
        "p": p, "seed": seed, "cutoff": cutoff, "tail_checks": tail_checks,
        "exact_numerical_radius_q": str(model.q),
        "spectral_radius": str(model.eigenvalue_radius),
        "operator_norm": model.operator_norm, "first_step_gain": first_step_gain,
        "spectral_shortcut_fails": spectral_shortcut_fails,
        "window_max_absolute_error": e_window,
        "window_max_relative_error": rel_window,
        "tail_radius_bound": tail_radius,
        "all_horizon_absolute_error_bound": global_bound,
        "finite_probe_max_absolute_error": worst_global,
        "finite_probe_within_theorem_bound": worst_global <= global_bound + 1e-10,
        "certificate_scope": "analytic all-t bound conditional on exact finite-window error; measured error uses unverified floating-point arithmetic",
    }
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--p", type=int, choices=(2, 4), default=4)
    ap.add_argument("--seed", type=int, default=325094)
    ap.add_argument("--k", type=int, default=24)
    ap.add_argument("--cutoff", type=int, default=18)
    args = ap.parse_args()
    result = run_case(p=args.p, seed=args.seed, k=args.k, cutoff=args.cutoff)
    if args.check:
        assert result["finite_probe_within_theorem_bound"]
        assert result["spectral_shortcut_fails"]
        assert result["ambient_dimension"] > result["sketch_dimension"]
    body = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(body)
    else:
        print(body, end="")


if __name__ == "__main__":
    main()

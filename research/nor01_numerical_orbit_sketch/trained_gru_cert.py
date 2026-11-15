"""NOR-03: source-locked interval certificate for a trained recurrent GRU.

Python standard library only. The checkpoint's JSON decimal literals and the
public float feature literals (via str(float)) are treated as exact rationals.
Rational Taylor-series enclosures of exp provide verified real-analytic gates.
This is a pointwise derivative certificate for the *true block transition*
D(F^L)(h), including all intermediate Jacobians. It is NOT a certificate
for every nearby nonlinear orbit or for native ranking-task performance.

No outward claims are based on finite differences, double-precision spectral
radii, or sampled numerical-range directions.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction as Q
from hashlib import sha256
from math import isqrt, sqrt
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.rrc_compactification import bench25, compound25, model25

N = model25.HIDDEN
DIGITS = 16
SCALE = 10 ** DIGITS
EXP_TERMS = 72
ZERO = Q(0)
ONE = Q(1)


@dataclass(frozen=True)
class Interval:
    lo: Q
    hi: Q

    def __post_init__(self):
        if self.lo > self.hi:
            raise ValueError("inverted interval")
        # Every interval operation is outward-rounded to a fixed rational
        # denominator. This prevents exact exp Taylor denominators from
        # growing exponentially through recurrent layers.
        low, high = self.lo * SCALE, self.hi * SCALE
        lower = low.numerator // low.denominator
        upper = -((-high.numerator) // high.denominator)
        object.__setattr__(self, "lo", Q(lower, SCALE))
        object.__setattr__(self, "hi", Q(upper, SCALE))

    @staticmethod
    def point(value):
        q = value if isinstance(value, Q) else Q(str(value))
        return Interval(q, q)

    def __add__(self, other):
        rhs = as_interval(other)
        return Interval(self.lo + rhs.lo, self.hi + rhs.hi)

    __radd__ = __add__

    def __neg__(self):
        return Interval(-self.hi, -self.lo)

    def __sub__(self, other):
        return self + (-as_interval(other))

    def __rsub__(self, other):
        return as_interval(other) + (-self)

    def __mul__(self, other):
        rhs = as_interval(other)
        values = (self.lo * rhs.lo, self.lo * rhs.hi,
                  self.hi * rhs.lo, self.hi * rhs.hi)
        return Interval(min(values), max(values))

    __rmul__ = __mul__


def as_interval(x):
    return x if isinstance(x, Interval) else Interval.point(x)


def exponential_bounds(x: Q) -> Interval:
    """Rigorous exp(x) bounds via positive rational Taylor and geometric tail.

    For x>=0, sum(k=0..n) x^k/k! <= exp(x) <= S_n +
      [x^(n+1)/(n+1)!]/[1 - x/(n+2)] when n+2>x.
    For negative x, invert the bounds for exp(-x).
    """
    if x < 0:
        b = exponential_bounds(-x)
        return Interval(ONE / b.hi, ONE / b.lo)
    n = max(EXP_TERMS, int(x) + 55)
    term = total = ONE
    for k in range(1, n + 1):
        term *= x / k
        total += term
    next_term = term * x / (n + 1)
    denom = ONE - x / (n + 2)
    assert denom > 0
    return Interval(total, total + next_term / denom)


def sigmoid_at(x: Q) -> Interval:
    e = exponential_bounds(-x)
    return Interval(ONE / (ONE + e.hi), ONE / (ONE + e.lo))


def sigmoid(x: Interval) -> Interval:
    lo = sigmoid_at(x.lo)
    hi = sigmoid_at(x.hi)
    return Interval(lo.lo, hi.hi)


def tanh(x: Interval) -> Interval:
    s = sigmoid(2 * x)
    return Interval(2 * s.lo - ONE, 2 * s.hi - ONE)


def get_weights(ck: dict):
    assert ck["parameters"] == 1000 and ck["kind"] == "recurrent"
    s = ck["state_dict"]
    return {
        "ih": tuple(tuple(Q(str(v)) for v in row) for row in s["rnn.weight_ih_l0"]),
        "hh": tuple(tuple(Q(str(v)) for v in row) for row in s["rnn.weight_hh_l0"]),
        "bi": tuple(Q(str(v)) for v in s["rnn.bias_ih_l0"]),
        "bh": tuple(Q(str(v)) for v in s["rnn.bias_hh_l0"]),
    }


def affine_interval(rows, biases, values):
    return [sum((w * h for w, h in zip(row, values, strict=True)),
                Interval.point(bias))
            for row, bias in zip(rows, biases, strict=True)]


def gates(weights, hidden, input_row=None):
    if len(hidden) != N:
        raise ValueError("wrong hidden-state width")
    x = list(input_row) if input_row is not None else [Interval.point(0)] * 8
    if len(x) != 8:
        raise ValueError("wrong input width")
    wi = affine_interval(weights["ih"], weights["bi"], x)
    wh = affine_interval(weights["hh"], weights["bh"], hidden)
    r, z, candidate = [], [], []
    one = Interval.point(1)
    for i in range(N):
        r.append(sigmoid(wi[i] + wh[i]))
        z.append(sigmoid(wi[N + i] + wh[N + i]))
        candidate.append(tanh(wi[2 * N + i] + r[i] * wh[2 * N + i]))
    state = [(one - z[i]) * candidate[i] + z[i] * hidden[i] for i in range(N)]
    return state, r, z, candidate, wh


def jacobian_interval(weights, center):
    """Interval enclosure of analytic D_h F(h; zero_input) at exact rational center."""
    h = [Interval.point(c) for c in center]
    _, r, z, candidate, wh = gates(weights, h)
    one = Interval.point(1)
    hh = weights["hh"]
    out = []
    for i in range(N):
        dr = r[i] * (one - r[i])
        dz = z[i] * (one - z[i])
        dn = one - candidate[i] * candidate[i]
        row = []
        for j in range(N):
            c = (Interval.point(center[i]) - candidate[i]) * dz * hh[N + i][j]
            c += (one - z[i]) * dn * (
                r[i] * hh[2 * N + i][j] + wh[2 * N + i] * dr * hh[i][j])
            if i == j:
                c += z[i]
            row.append(c)
        out.append(row)
    return out


def round_rational(x: Q, scale: int = SCALE) -> Q:
    return Q((x * scale + Q(1, 2)).numerator //
             (x * scale + Q(1, 2)).denominator, scale)


def ceil_rational(x: Q, scale: int = SCALE) -> Q:
    y = x * scale
    return Q(-((-y.numerator) // y.denominator), scale)


def sqrt_upper(x: Q, scale: int = SCALE) -> Q:
    if x < 0:
        raise ValueError("negative square root")
    numerator = (x.numerator * scale * scale + x.denominator - 1) // x.denominator
    n = isqrt(numerator)
    return Q(n + (n * n < numerator), scale)


def rational_center(box):
    return [round_rational((x.lo + x.hi) / 2) for x in box]


def radius_upper(box, center):
    sq = sum((max(abs(x.lo - c), abs(x.hi - c)) ** 2
              for x, c in zip(box, center, strict=True)), ZERO)
    return sqrt_upper(sq)


def matmul(a, b):
    m = len(a)
    return [[sum((a[i][k] * b[k][j] for k in range(m)), ZERO)
             for j in range(m)] for i in range(m)]


def frobenius_squared(a):
    return sum((v * v for row in a for v in row), ZERO)


def hessian_bound(weights):
    """Global bound on ||D^2_h F(h)[u,v]||_2/(||u|| ||v||), h in cube.

    Uses |sigmoid'|<=1/4, |sigmoid''|<=1/4,
    |tanh'|<=1, |tanh''|<=2, |h_i-n_i|<=2, and
    the full row-2-norms, rounded upward. The sum of component
    bounds dominates their Euclidean norm.
    """
    hh, bh = weights["hh"], weights["bh"]
    out = ZERO
    for i in range(N):
        nr = sqrt_upper(sum((z*z for z in hh[i]), ZERO))
        nz = sqrt_upper(sum((z*z for z in hh[N+i]), ZERO))
        nn = sqrt_upper(sum((z*z for z in hh[2*N+i]), ZERO))
        v = abs(bh[2*N+i]) + sum(map(abs, hh[2*N+i]), ZERO)
        first = nn + v * nr / 4
        second = v * nr*nr/4 + nr*nn/2
        out += 2*first*first + second + nz*nz/2 + nz*(1+first)/2
    return out


def native_zero_step(hidden, checkpoint):
    s = checkpoint["state_dict"]
    a = model25.affine(s["rnn.weight_ih_l0"], s["rnn.bias_ih_l0"], [0.0]*8)
    v = model25.affine(s["rnn.weight_hh_l0"], s["rnn.bias_hh_l0"], hidden)
    out = []
    for i in range(N):
        r = model25.sigmoid(a[i] + v[i])
        z = model25.sigmoid(a[N+i] + v[N+i])
        candidate = __import__("math").tanh(a[2*N+i] + r*v[2*N+i])
        out.append(z*hidden[i] + (1-z)*candidate)
    return out


def interval_forward(weights, inputs):
    h = [Interval.point(0) for _ in range(N)]
    for row in inputs:
        x = [Interval.point(v) for v in row]
        h = gates(weights, h, x)[0]
    return h


def build_certificate(weights, anchor_box, length=12, center=Q(1, 32),
                      perturbation=Q(1, 10**9), label="zero_8"):
    if length < 1 or center <= 0 or perturbation <= 0:
        raise ValueError("length, disk center and perturbation must be positive")
    if any(z.lo < -1 or z.hi > 1 for z in anchor_box):
        raise ValueError("GRU state must lie inside invariant unit cube")
    M = hessian_bound(weights)
    forward = anchor_box
    nominal = []
    uncertainty = []
    norm_bounds = []
    centers = []
    radii = []
    for _ in range(length):
        midpoint = rational_center(forward)
        centers.append(midpoint)
        radius = radius_upper(forward, midpoint)
        radii.append(radius)
        ji = jacobian_interval(weights, midpoint)
        jm = [[round_rational((z.lo+z.hi)/2) for z in row] for row in ji]
        err_at_center = sqrt_upper(sum(
            (max(abs(z.lo - f), abs(z.hi - f)) ** 2
             for row, estimates in zip(ji, jm, strict=True)
             for z, f in zip(row, estimates, strict=True)), ZERO))
        # Matrix operator error: ||J(h)-Jmid||_2 <= err_at_center + M*radius.
        err = err_at_center + M * radius
        uncertainty.append(err)
        norm_bounds.append(sqrt_upper(frobenius_squared(jm)) + err)
        nominal.append(jm)
        forward = gates(weights, forward)[0]
        if any(z.lo < -1 or z.hi > 1 for z in forward):
            raise AssertionError("GRU invariant cube enclosure failed")
    product = [[Q(int(i == j)) for j in range(N)] for i in range(N)]
    for jm in nominal:
        product = matmul(jm, product)
    maxnorm = max(norm_bounds)
    error = length * maxnorm ** (length - 1) * max(uncertainty)
    f_sq = frobenius_squared([
        [product[i][j] - (center if i==j else ZERO) for j in range(N)]
        for i in range(N)
    ])
    b = ceil_rational(sqrt_upper(f_sq) + error, 10**12)
    q = center + b
    # Ball perturbation and remainder recurrence. Holds for every
    # ||e||_2 <= perturbation with h+e inside invariant cube.
    delta, rem = perturbation, ZERO
    for k in range(length):
        rem = norm_bounds[k] * rem + M * delta * delta / 2
        delta = norm_bounds[k] * delta + M * delta * delta / 2
    digest = sha256("\n".join(",".join(str(v) for v in row)
                              for row in product).encode()).hexdigest()
    return {
        "label": label,
        "block_depth": length,
        "disk_center_fraction": str(center),
        "disk_radius_fraction": str(b),
        "disk_q_float": float(q),
        "contracting_disk": q < 1,
        "analytic_step": "exact trained GRU real-analytic zero-input map",
        "certifies": "W(D F^L(h_exact)) subset closed complex disk, for exact hidden-state interval",
        "anchor_box_max_radius": float(radius_upper(anchor_box, rational_center(anchor_box))),
        "max_tangent_jacobian_error_upper": float(max(uncertainty)),
        "block_derivative_perturbation_upper": float(error),
        "block_nominal_frobenius": float(sqrt_upper(frobenius_squared(product))),
        "block_product_sha256": digest,
        "global_one_step_hessian_upper": float(M),
        "perturbation_radius": float(perturbation),
        "block_nonlinear_remainder_upper": float(rem),
        "block_nonlinear_output_difference_upper": float(delta),
        "rounding": "certified rational outward enclosure (exp Taylor + exact fractions + upward sqrt)",
        "global_guarantee": "pointwise block Jacobian disk and finite-block nonlinear Taylor budget only",
    }


def run(*, include_heldout=True, warmup=8):
    if not 0 <= warmup <= 64:
        raise ValueError("warmup must be between 0 and 64")
    models, checksums = bench25.checkpoints()
    ck = models["recurrent"]
    weights = get_weights(ck)
    # Certified analytic continuation from exact zero initial hidden vector.
    anchor_box = interval_forward(weights, [[0]*8 for _ in range(8)])
    zero_cert = build_certificate(weights, anchor_box, length=12,
                                  label="trained_8_zero_steps")
    # A negative control: very early tangent is amplifying. Disk with a=1/32
    # and its derivative bound should *not* be claimed contractive.
    early = build_certificate(weights, [Interval.point(0)]*N,
                              length=8, label="trained_initial_zero")
    out = {
        "schema": "nor03.trained.gru.interval.block.v1",
        "checkpoint_sha256": checksums["recurrent"],
        "checkpoint_parameters": ck["parameters"],
        "source_hashes_verified": ck["source_sha256"],
        "model": "RRC25 trained 1000-parameter 12D GRU",
        "frozen_model_uses": "original learned weights; zero future inputs after source sequence",
        "zero_origin_positive": zero_cert,
        "early_negative_control": early,
        "pinned_prior_experiments": ["RRC25", "FHC01C", "NOR01", "NOR02"],
    }
    if include_heldout:
        # Build six *new* source-family cases without the benchmark roster's
        # 12 negative complete-search scans and 6 exhaustive future-collision
        # searches. Those unrelated exact-program controls can dominate this
        # matrix-analysis experiment's CPU budget.
        import random
        cases, seen = [], set()
        for ix, family in enumerate(compound25.TEST_FAMILIES):
            rng = random.Random(compound25.SEED + 903900 + 1009 * ix)
            for attempt in range(32):
                candidate = compound25.make_case(
                    rng, family, f"nor03-{family}-{attempt:02d}")
                if candidate.visible not in seen and compound25.possible_headers(candidate):
                    seen.add(candidate.visible)
                    cases.append(candidate)
                    break
            else:
                raise AssertionError(f"could not construct NOR-03 family: {family}")
        heldout = []
        for case in cases:
            headers = compound25.possible_headers(case)
            if not headers:
                raise AssertionError("no public header for heldout case")
            features = compound25.features(case, headers[0])
            native = [0.0]*N
            for row in features:
                # Portable trained GRU: identical gate order as model25.
                s = ck["state_dict"]
                a = model25.affine(s["rnn.weight_ih_l0"], s["rnn.bias_ih_l0"], row)
                v = model25.affine(s["rnn.weight_hh_l0"], s["rnn.bias_hh_l0"], native)
                nxt = []
                for i in range(N):
                    r = model25.sigmoid(a[i]+v[i])
                    z = model25.sigmoid(a[N+i]+v[N+i])
                    q = __import__("math").tanh(a[2*N+i]+r*v[2*N+i])
                    nxt.append((1-z)*q+z*native[i])
                native = nxt
            direct = model25.logits(case, headers[0], ck)
            logits = model25.affine(ck["state_dict"]["head.weight"],
                                   ck["state_dict"]["head.bias"], native)
            parity = max(abs(u-v) for u,v in zip(direct, logits))
            if parity > 1e-10:
                raise AssertionError("source native trained logits parity failed")
            # Public feature float literals are interpreted as exact decimals
            # during certification. The historical FP32 training and Python
            # inference paths remain separate.
            box = interval_forward(weights, features)
            for _ in range(warmup):
                box = gates(weights, box)[0]
                native = native_zero_step(native, ck)
            # Interval dependency can enlarge an enclosure across 17 GRU
            # steps even when floating replay is inside that enclosure.
            # Check violation *outside* the interval, not distance to a
            # particular endpoint (which confounds width with disagreement).
            max_delta = max(max(float(z.lo)-v, v-float(z.hi), 0.0)
                            for z,v in zip(box,native))
            max_width = max(float(z.hi-z.lo) for z in box)
            if max_delta > 1e-9:
                raise AssertionError(
                    f"native continuation lies outside rational enclosure: "
                    f"family={case.family}, gap={max_delta}, width={max_width}")
            cert = build_certificate(weights, box, length=12,
                                     label="heldout_" + case.family)
            heldout.append({
                "family": case.family,
                "case_id_sha256": sha256(case.identifier.encode()).hexdigest(),
                "public_header_candidates": len(headers),
                "post_nine_inputs_native_logits_parity": parity,
                "native_interval_replay_discrepancy": max_delta,
                "native_interval_max_width": max_width,
                "certificate": cert,
            })
        out["case_sampling_protocol"] = "NOR03 independent post-training family seeds SEED+903900+1009*i; no negative/collision roster"
        out["heldout"] = heldout
        out["heldout_contracting_disks"] = sum(
            row["certificate"]["contracting_disk"] for row in heldout)
        out["heldout_count"] = len(heldout)
        out["heldout_warmup"] = warmup
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--skip-heldout", action="store_true")
    parser.add_argument("--warmup", type=int, default=8)
    args = parser.parse_args()
    receipt = run(include_heldout=not args.skip_heldout, warmup=args.warmup)
    if args.check:
        assert receipt["checkpoint_parameters"] == 1000
        assert receipt["early_negative_control"]["contracting_disk"] is False
        assert receipt["zero_origin_positive"]["contracting_disk"] is True
        if not args.skip_heldout:
            assert receipt["heldout_count"] == 6
    body = json.dumps(receipt, sort_keys=True, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(body)
    else:
        print(body, end="")


if __name__ == "__main__":
    main()

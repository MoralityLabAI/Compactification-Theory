"""OIC-20: exact labeled marker observability and finite sensor design.

A hidden state evolves as z[t+1]=(z[t]+d) mod n. Its unknown initial
phase is uniform on Z_n. A sensor has labeled mark positions M; at each
active probe it returns an |M|-bit vector [z[t]==m] for every m in M.
The all-active word of n probes exposes a full n-periodic emission block.
This is an exact standard-library reference, not a neural experiment.
"""
from __future__ import annotations
from collections import Counter
from itertools import combinations
from math import gcd


def markers_ok(n: int, marks: tuple[int, ...]) -> None:
    if not isinstance(n, int) or n < 2:
        raise ValueError('n must be integer >= 2')
    if not marks or not all(isinstance(x, int) and 0 <= x < n for x in marks):
        raise ValueError('nonempty valid integer marker positions required')
    if len(set(marks)) != len(marks):
        raise ValueError('marker labels must identify distinct positions')


def trace_distribution(n: int, shift: int, marks: tuple[int, ...], horizon: int | None = None) -> dict[tuple[int, ...], int]:
    """Counts among n equiprobable initial phases; normalize by n if needed."""
    markers_ok(n, marks)
    if not isinstance(shift, int):
        raise ValueError('integer transition shift required')
    if horizon is None:
        horizon = n
    if not isinstance(horizon, int) or horizon < 0:
        raise ValueError('nonnegative integer horizon required')
    emissions = [sum(1 << j for j, m in enumerate(marks) if z == m) for z in range(n)]
    return dict(Counter(
        tuple(emissions[(initial + shift * t) % n] for t in range(horizon))
        for initial in range(n)
    ))


def shift_signature(n: int, shift: int, k: int) -> tuple[int, int | None]:
    """Exact trace-law invariant for the two labeled markers (0,k)."""
    markers_ok(n, (0, k))
    g = gcd(n, shift)
    if k % g:
        return g, None
    period = n // g
    reduced_step = (shift // g) % period
    reduced_offset = k // g
    return g, (reduced_offset * pow(reduced_step, -1, period)) % period


def two_marker_equivalent(n: int, a: int, b: int, k: int) -> bool:
    return shift_signature(n, a, k) == shift_signature(n, b, k)


def partitions(n: int, marks: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    """Classes of shifts with identical distributions, ordered by minimum d."""
    signatures = {}
    for d in range(n):
        histogram = trace_distribution(n, d, marks)
        key = tuple(sorted(histogram.items()))
        signatures.setdefault(key, []).append(d)
    return tuple(sorted(map(tuple, signatures.values()), key=lambda a: a[0]))


def identifies_every_shift(n: int, marks: tuple[int, ...]) -> bool:
    return len(partitions(n, marks)) == n


def sufficient_divisor_markers(n: int) -> tuple[int, ...]:
    """At most tau(n) labels, one anchor plus every proper divisor."""
    if not isinstance(n, int) or n < 2:
        raise ValueError('n must be integer >= 2')
    return (0,) + tuple(g for g in range(1, n) if n % g == 0)


def prime_power(n: int) -> tuple[int, int] | None:
    for p in range(2, n + 1):
        if n % p:
            continue
        if any(p % k == 0 for k in range(2, int(p ** .5) + 1)):
            continue
        power = n
        exponent = 0
        while power % p == 0:
            exponent += 1
            power //= p
        if power == 1:
            return p, exponent
    return None


def prime_power_minimum_markers(p: int, exponent: int) -> tuple[int, tuple[int, ...]]:
    if not isinstance(p, int) or p < 2 or not isinstance(exponent, int) or exponent < 1:
        raise ValueError('prime p and exponent >= 1 required')
    if any(p % q == 0 for q in range(2, int(p ** .5) + 1)):
        raise ValueError('p must be prime')
    last = exponent - 1 if p > 2 else exponent - 2
    marks = (0,) + tuple(p ** j for j in range(last + 1))
    return len(marks), marks


def exhaustive_minimum_markers(n: int, max_size: int | None = None) -> tuple[int, tuple[int, ...]] | None:
    """Find smallest labeled marker set, quotienting global translations by fixing 0."""
    if n < 2:
        raise ValueError('n>=2')
    if max_size is None:
        max_size = n
    for m in range(1, min(n, max_size) + 1):
        for others in combinations(range(1, n), m - 1):
            marks = (0,) + others
            if identifies_every_shift(n, marks):
                return m, marks
    return None


def reference(ns: tuple[int, ...] = tuple(range(2, 17))) -> dict[str, object]:
    cases = []
    for n in ns:
        sufficient = sufficient_divisor_markers(n)
        assert identifies_every_shift(n, sufficient)
        exact = exhaustive_minimum_markers(n, max_size=len(sufficient))
        if exact is None:
            raise AssertionError('divisor construction must be sufficient')
        pp = prime_power(n)
        if pp:
            m, marks = prime_power_minimum_markers(*pp)
            assert exact[0] == m, (n, exact, m)
            assert identifies_every_shift(n, marks)
        cases.append({
            'n':n, 'minimal_label_count':exact[0],
            'minimal_example':list(exact[1]),
            'divisor_construction':list(sufficient),
            'prime_power_parameters':list(pp) if pp else None,
        })
    return {'status':'exact_finite_multi_marker_reference','cases':cases,
            'family':'constant_modular_shifts_uniform_hidden_initial_phase',
            'sensor':'n_labeled_active_marker_bits_per_step',
            'horizon':'n_continuous_probes'}

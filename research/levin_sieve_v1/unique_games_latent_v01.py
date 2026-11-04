"""UG-LAT-01: exact finite-field quadratic latent block, CPU-only.

Replicates the algebra of the *one-block* gadget in OpenAI's UGC preprint,
not the high-depth recursive gadget or its rank-detection theorem.

Inputs are elements of F_(2^d)^3 represented by tuples of d-bit integers.
The irreducible reduction polynomial is pinned per supported d.
"""
from __future__ import annotations

from fractions import Fraction
from itertools import product

MODULI = {1: 0b11, 2: 0b111, 3: 0b1011, 4: 0b10011}


def mul(x: int, y: int, d: int) -> int:
    """Polynomial multiplication reduced modulo a fixed irreducible polynomial."""
    if d not in MODULI:
        raise ValueError("unsupported field degree")
    q = 1 << d
    if not 0 <= x < q or not 0 <= y < q:
        raise ValueError("field operands must be d-bit integers")
    result = 0
    a, b = x, y
    while b:
        if b & 1:
            result ^= a
        b >>= 1
        a <<= 1
        if a & q:
            a ^= MODULI[d]
    return result


def add(x: tuple[int, int, int], y: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple(a ^ b for a, b in zip(x, y))


def vectors(d: int):
    if d not in MODULI:
        raise ValueError("unsupported field degree")
    return product(range(1 << d), repeat=3)


def dot(x: tuple[int, int, int], y: tuple[int, int, int], d: int) -> int:
    return mul(x[0], y[0], d) ^ mul(x[1], y[1], d) ^ mul(x[2], y[2], d)


def quadratic(x: tuple[int, int, int], d: int) -> tuple[int, int, int]:
    return (mul(x[1], x[2], d), mul(x[0], x[2], d),
            mul(x[0], x[1], d))


def polar(x: tuple[int, int, int], a: tuple[int, int, int],
          d: int) -> tuple[int, int, int]:
    return add(add(quadratic(add(x, a), d), quadratic(x, d)),
               quadratic(a, d))


def scalar(t: int, v: tuple[int, int, int], d: int) -> tuple[int, int, int]:
    return tuple(mul(t, x, d) for x in v)


def projective_lines(d: int) -> tuple[tuple[int, int, int], ...]:
    """One canonical nonzero generator per F_(2^d)-line in F^3."""
    q = 1 << d
    seen = set()
    representatives = []
    for v in vectors(d):
        if v == (0, 0, 0):
            continue
        line = frozenset(scalar(t, v, d) for t in range(1, q))
        if line not in seen:
            seen.add(line)
            representatives.append(min(line))
    return tuple(representatives)


def hyperplane(v: tuple[int, int, int], d: int) -> tuple[tuple[int, int, int], ...]:
    return tuple(w for w in vectors(d) if dot(v, w, d) == 0)


def update_space(v: tuple[int, int, int], d: int):
    """U_A={(a,Q(a)+w): a=t v, v.w=0}, an F2-space of size q^3."""
    if v == (0, 0, 0):
        raise ValueError("a projective line needs a nonzero vector")
    q = 1 << d
    H = hyperplane(v, d)
    return tuple((a, add(quadratic(a, d), w))
                 for t in range(q) for a in (scalar(t, v, d),) for w in H)


def block_output(x: tuple[int, int, int],
                 y: tuple[int, int, int], d: int):
    return add(y, quadratic(x, d))


def response(x: tuple[int, int, int], y: tuple[int, int, int],
             a: tuple[int, int, int], b: tuple[int, int, int], d: int):
    return add(block_output(add(x, a), add(y, b), d),
               block_output(x, y, d))


def exact_block_change(v: tuple[int, int, int], d: int,
                       x: tuple[int, int, int] = (0, 0, 0)) -> Fraction:
    """Response chance for uniform nonzero update in U_A.

    The y-coordinate cancels from every response, so y=0 suffices.
    This exact enumeration is independent of the closed-form formula.
    """
    zero = (0, 0, 0)
    updates = tuple((a, b) for a, b in update_space(v, d)
                    if a != zero or b != zero)
    changes = sum(response(x, zero, a, b, d) != zero for a, b in updates)
    return Fraction(changes, len(updates))


def block_formula(d: int) -> Fraction:
    """1 - theta, theta=1/(q^2+q+1), from the manuscript."""
    if d not in MODULI:
        raise ValueError("unsupported field degree")
    q = 1 << d
    return Fraction(q * (q + 1), q * q + q + 1)


def recursive_change_formula(d: int, depth: int) -> Fraction:
    """Manuscript's p_t, as an algebraic prediction, without a recursive build."""
    if depth < 0:
        raise ValueError("depth must be nonnegative")
    q = 1 << d
    return (Fraction(q ** 3 - 1, q ** 3)
            * block_formula(d) ** depth)

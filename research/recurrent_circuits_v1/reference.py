"""Exact finite references for recurrent-circuit resource accounting.

All state-space algebra is over GF(2). The noise checker uses Fraction.
This module does not load a language model or implement a transformer compiler.
"""
from dataclasses import dataclass
from fractions import Fraction
from math import comb
from typing import Iterable


def integer(value: int, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, excluding bool")
    return value


def exact(value: int | Fraction) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, Fraction)):
        raise TypeError("int or Fraction required")
    return Fraction(value)


def bitvector(value: int, width: int) -> int:
    integer(width, "width")
    integer(value, "bitvector")
    if value >= (1 << width):
        raise ValueError("bitvector exceeds declared width")
    return value


def apply_rows(rows: Iterable[int], state: int) -> int:
    return sum(((row & state).bit_count() & 1) << i for i, row in enumerate(rows))


@dataclass(frozen=True)
class Affine:
    """m-bit affine transition x -> A*x + bias, bit 0 as first coordinate."""
    rows: tuple[int, ...]
    bias: int = 0

    def __post_init__(self):
        object.__setattr__(self, "rows", tuple(self.rows))
        for row in self.rows:
            bitvector(row, len(self.rows))
        bitvector(self.bias, len(self.rows))

    @property
    def width(self) -> int:
        return len(self.rows)

    def __call__(self, state: int) -> int:
        bitvector(state, self.width)
        return apply_rows(self.rows, state) ^ self.bias


def row_basis(rows: Iterable[int], width: int) -> tuple[int, ...]:
    """Canonical reduced row-echelon basis, with low-bit pivots first."""
    integer(width, "width")
    checked = tuple(bitvector(row, width) for row in rows)
    work = sorted(set(row for row in checked if row))
    rank = 0
    for column in range(width):
        pivot = next((i for i in range(rank, len(work)) if work[i] & (1 << column)), None)
        if pivot is None:
            continue
        work[rank], work[pivot] = work[pivot], work[rank]
        for i in range(len(work)):
            if i != rank and work[i] & (1 << column):
                work[i] ^= work[rank]
        rank += 1
    return tuple(work[:rank])


def row_times_matrix(row: int, transition: Affine) -> int:
    bitvector(row, transition.width)
    out = 0
    for i, a in enumerate(transition.rows):
        if row & (1 << i):
            out ^= a
    return out


def coordinates(row: int, basis: tuple[int, ...], width: int) -> int:
    """Express a row in the supplied canonical basis; reject absent rows."""
    remainder = bitvector(row, width)
    out = 0
    for i, b in enumerate(basis):
        pivot = b & -b
        if remainder & pivot:
            remainder ^= b
            out |= 1 << i
    if remainder:
        raise ValueError("row lies outside the declared span")
    return out


@dataclass(frozen=True)
class Quotient:
    width: int
    basis: tuple[int, ...]
    transitions: tuple[Affine, ...]
    readout_rows: tuple[int, ...]

    @property
    def bits(self) -> int:
        return len(self.basis)

    def encode(self, state: int) -> int:
        bitvector(state, self.width)
        return apply_rows(self.basis, state)

    def readout(self, state: int) -> int:
        bitvector(state, self.bits)
        return apply_rows(self.readout_rows, state)


def minimal_affine_quotient(width: int, transitions: Iterable[Affine],
                            output_rows: Iterable[int]) -> Quotient:
    """Exact behavioral quotient on ALL GF(2)^width states and action words.

Translations cancel in state differences. Closure includes every supplied
transition, including interventions. A restricted reachable set can be smaller.
"""
    integer(width, "width")
    transitions = tuple(transitions)
    output_rows = tuple(bitvector(c, width) for c in output_rows)
    if any(f.width != width for f in transitions):
        raise ValueError("inconsistent state width")
    basis = row_basis(output_rows, width)
    while True:
        next_basis = row_basis((*basis, *(row_times_matrix(v, f)
                            for v in basis for f in transitions)), width)
        if next_basis == basis:
            break
        basis = next_basis
    compressed = tuple(Affine(tuple(coordinates(row_times_matrix(v, f), basis, width)
                                      for v in basis), apply_rows(basis, f.bias))
                       for f in transitions)
    readout = tuple(coordinates(c, basis, width) for c in output_rows)
    return Quotient(width, basis, compressed, readout)


def finite_partition(width: int, transitions: Iterable[Affine],
                     output_rows: Iterable[int]) -> tuple[int, ...]:
    """Independent exponential-state partition refinement oracle; tiny widths only."""
    integer(width, "width")
    if width > 12:
        raise ValueError("finite oracle is limited to 12 state bits")
    transitions = tuple(transitions)
    output_rows = tuple(bitvector(c, width) for c in output_rows)
    if any(f.width != width for f in transitions):
        raise ValueError("inconsistent state width")
    states = range(1 << width)

    def numbered(signatures):
        labels = {}
        return tuple(labels.setdefault(s, len(labels)) for s in signatures)

    blocks = numbered(apply_rows(output_rows, x) for x in states)
    while True:
        updated = numbered((blocks[x], tuple(blocks[f(x)] for f in transitions))
                           for x in states)
        if updated == blocks:
            return blocks
        blocks = updated


def label_coverage_bound(labels: int, description_bits: int,
                         choices: int = 1, allowed_errors: int = 0) -> Fraction:
    """Union bound for fair independent labels and a fixed finite decoder family."""
    integer(labels, "labels")
    integer(description_bits, "description_bits")
    integer(choices, "choices", 1)
    integer(allowed_errors, "allowed_errors")
    if allowed_errors > labels:
        raise ValueError("too many allowed errors")
    volume = sum(comb(labels, j) for j in range(allowed_errors + 1))
    return min(Fraction(1), Fraction(choices * (1 << description_bits) * volume, 1 << labels))


def restore(value: int | Fraction, a: int | Fraction = Fraction(1, 4)) -> Fraction:
    """Continuous ReLU clipped-ramp restoration to binary codewords."""
    value, a = exact(value), exact(a)
    if not 0 < a < Fraction(1, 2):
        raise ValueError("restoration margin must lie in (0,1/2)")
    return (max(Fraction(), value - a) - max(Fraction(), value - (1-a))) / (1-2*a)


def restored_nand(x: int | Fraction, y: int | Fraction) -> Fraction:
    return 1 - restore(max(Fraction(), exact(x) + exact(y) - 1))


def error_envelope(initial: int | Fraction, local_error: int | Fraction,
                   gain: int | Fraction, steps: int) -> Fraction:
    initial, local_error, gain = map(exact, (initial, local_error, gain))
    integer(steps, "steps")
    if min(initial, local_error, gain) < 0:
        raise ValueError("nonnegative error terms required")
    error = initial
    for _ in range(steps):
        error = gain * error + local_error
    return error


def invariant_tube(gain: int | Fraction, local_error: int | Fraction,
                   radius: int | Fraction, separation: int | Fraction = 1) -> bool:
    gain, local_error, radius, separation = map(exact, (gain, local_error, radius, separation))
    if min(gain, local_error, radius) < 0 or separation <= 0:
        raise ValueError("invalid tube parameters")
    return gain < 1 and 2 * radius < separation and gain * radius + local_error <= radius


def roundtrip_bytes(state_bytes: int, loops: int, resident_chunk: int) -> int:
    """Explicit spill/reload after each chunk, INCLUDING the final chunk."""
    integer(state_bytes, "state_bytes")
    integer(loops, "loops", 1)
    integer(resident_chunk, "resident_chunk", 1)
    return 2 * state_bytes * ((loops + resident_chunk - 1) // resident_chunk)

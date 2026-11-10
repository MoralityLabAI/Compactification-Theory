"""Prompt rendering from ARG-01 public records only.

Input rows have exactly the fields `external.public_records` exports. The
query series is public but plays no part in identifying the program, so it is
not rendered. Case identifiers are never rendered. Every prompt ends with the
fixed tail "\\nProgram:"; the three answer positions follow it.
"""
from __future__ import annotations

PUBLIC_FIELDS = {"case_id", "topology", "observed_degree", "demos", "query", "public_sha256"}
SHOT_FIELDS = PUBLIC_FIELDS | {"program"}
OPS = ("I", "N", "S", "R")
TAIL = "\nProgram:"
OP_TEXT = {op: " " + op for op in OPS}

HEADER = (
    "Each line lists the coefficients of x^0..x^12 of an integer power series f.\n"
    "Operators: I(f)=f, N(f)=f(-x), S(f)=f(x^2), R(f)=1/f.\n"
    "A program 'a b c' maps an input f to an output y by its topology:\n"
    "fork: y=c(a(f)*b(f)); cascade: y=c(a(f)*b(a(f))); additive: y=c(a(f)+b(f)-1); "
    "hadamard: y=c(1+sum_{n>=1} a(f)_n*b(f)_n*x^n).\n"
    "Find the program that maps both inputs to their outputs.\n"
)

VARIANTS = ("canonical", "swap", "space", "swap_space")


def _check_public(row, allowed):
    if not isinstance(row, dict):
        raise TypeError("public record must be a dict")
    extra = set(row) - allowed
    if extra:
        raise ValueError(f"non-public fields in model input: {sorted(extra)}")
    if row.get("observed_degree") != 12 or len(row.get("demos", ())) != 2:
        raise ValueError("unexpected public record shape")


def _series(values, sep):
    if not values or values[0] != 1 or any(type(v) is not int for v in values):
        raise ValueError("bad public series")
    return sep.join(str(v) for v in values)


def case_block(row, *, variant="canonical", program=None, feedback=()):
    if variant not in VARIANTS:
        raise ValueError("unknown prompt variant")
    sep = " " if variant in ("space", "swap_space") else ", "
    demos = list(row["demos"])
    if variant in ("swap", "swap_space"):
        demos = demos[::-1]
    lines = [f"\nTopology: {row['topology']}"]
    for i, demo in enumerate(demos, 1):
        lines.append(f"Input {i}: {_series(demo['input'], sep)}")
        lines.append(f"Output {i}: {_series(demo['output'], sep)}")
    for item in feedback:
        lines.append(feedback_line(item, swapped=variant in ("swap", "swap_space")))
    text = "\n".join(lines) + TAIL
    if program is not None:
        text += "".join(OP_TEXT[op] for op in program) + "\n"
    return text


def feedback_line(item, *, swapped=False):
    """One rejected candidate and its first mismatching degree in each demo."""
    program, faults = item
    if tuple(program) not in {(a, b, c) for a in OPS for b in OPS for c in OPS}:
        raise ValueError("feedback program outside grammar")
    faults = list(faults)[::-1] if swapped else list(faults)
    parts = []
    for i, fault in enumerate(faults, 1):
        parts.append(f"example {i} ok" if fault is None else f"example {i} differs at x^{int(fault)}")
    return f"Rejected: {' '.join(program)} ({', '.join(parts)})"


def render(row, *, shots=(), variant="canonical", feedback=()):
    """Full prompt text for one public record. `shots` are solved shot records."""
    _check_public(row, PUBLIC_FIELDS)
    text = HEADER
    for shot in shots:
        _check_public(shot, SHOT_FIELDS)
        if "program" not in shot:
            raise ValueError("shot without program")
        text += case_block(shot, variant="canonical", program=shot["program"])
    text += case_block(row, variant=variant, feedback=feedback)
    if not text.endswith(TAIL):
        raise AssertionError("prompt tail drift")
    return text

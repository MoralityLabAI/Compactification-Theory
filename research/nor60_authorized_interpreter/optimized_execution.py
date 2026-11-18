"""Exact NOR45 interpreter with source-derived support and local unary reuse.

The signed verifier invokes this fixed implementation only after its original
authority, session, precommit, provider and query checks. A new local memo is
created for every invocation. No query, unary result or accepted output is
retained between calls.
"""
from __future__ import annotations
from typing import Sequence

from research.nor45_authorized_program import independent_execution_verifier as nor45
from research.nor59_authorized_merkle import authorized_root

POLICY = "support_memoized"


def _unit_series(query: Sequence[int]) -> tuple[int, ...]:
    if not isinstance(query, (tuple, list)):
        raise nor45.ProgramAttestationError("exact full source sequence required")
    source = tuple(query)
    if not source or any(type(v) is not int for v in source) or source[0] != 1:
        raise nor45.ProgramAttestationError("normalized exact integer source required")
    return source


def _support_stride(source: tuple[int, ...]) -> int:
    for stride in (8, 4):
        if all(source[i] == 0 for i in range(1, len(source)) if i % stride):
            return stride
    return 1


def supported_inverse(query: Sequence[int]) -> tuple[int, ...]:
    """Compute the finite unit inverse, deriving support from every input entry.

    If F(x) = H(x**s), the unique inverse modulo x**m is the expansion of
    H**(-1) modulo x**ceil(m/s). The generic function also supports short odd
    lengths for source-free algebra tests; the signed entry point separately
    enforces NOR45's original D512/D1024 geometry.
    """
    source = _unit_series(query)
    stride = _support_stride(source)
    reduced = source[::stride]
    inverse = nor45.unary(reduced, "R")
    return tuple(inverse[i // stride] if i % stride == 0 else 0
                 for i in range(len(source)))


def execution_structure(program: dict, query: Sequence[int],
                        observed_degree: int) -> dict:
    """Source-only operation counts. This function executes no unary transform."""
    source = authorized_root._checked_exact_full_source(query, observed_degree)
    nor45._validate_program(program, observed_degree, len(source))
    operations = [program[k] for k in ("base", "u", "v")]
    if program["hidden_beta"]:
        operations.append(program["hidden_transform"])
    stride = _support_stride(source)
    distinct = list(dict.fromkeys(operations))
    return {
        "source_derived_stride": stride,
        "reduced_inverse_length": (len(source) - 1) // stride + 1,
        "original_unary_call_count": len(operations),
        "optimized_unary_call_count": len(distinct),
        "original_R_call_count": operations.count("R"),
        "optimized_R_call_count": int("R" in operations),
        "R_calls_eliminated": max(0, operations.count("R") - 1),
        "distinct_unary_tokens": distinct,
        "memo_scope": "one_authenticated_program_invocation",
    }


def execute_authorized_program(program: dict, query: Sequence[int],
                               observed_degree: int) -> tuple[int, ...]:
    """Recompute the complete original output with invocation-local reuse."""
    source = authorized_root._checked_exact_full_source(query, observed_degree)
    m = len(source)
    nor45._validate_program(program, observed_degree, m)
    operations = [program[k] for k in ("base", "u", "v")]
    beta = program["hidden_beta"]
    if beta:
        operations.append(program["hidden_transform"])
    memo = {}
    for op in operations:
        if op not in memo:
            memo[op] = supported_inverse(source) if op == "R" else nor45.unary(source, op)
    base, left, right = (memo[program[k]] for k in ("base", "u", "v"))
    hidden = memo[program["hidden_transform"]] if beta else None
    delay, spacing, amplitude = (program[k] for k in ("delay", "spacing", "amplitude"))
    onset = observed_degree + 32
    values = []
    for index in range(m):
        value = base[index]
        if index == delay:
            value += amplitude
        if index >= delay + spacing:
            value += amplitude * left[index - delay - spacing]
        if index >= delay + 2 * spacing:
            value += amplitude * right[index - delay - 2 * spacing]
        if beta and index >= onset:
            value += beta * hidden[index - onset]
        values.append(value)
    return tuple(values)

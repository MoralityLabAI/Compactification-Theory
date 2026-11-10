"""OIC-01: exact finite hidden-cycle and intervention-horizon reference.

Pure standard-library finite arithmetic; the experiment makes no assertion about
consciousness, physical spacetime, or neural-network model behavior.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable


@dataclass(frozen=True, order=True)
class State:
    visible: int
    hidden: int


def _validate(n: int, variant: str, state: State, probe: int) -> None:
    if not isinstance(n, int) or n < 2:
        raise ValueError("n must be an integer >= 2")
    if variant not in ("fixed", "rotating"):
        raise ValueError("variant must be 'fixed' or 'rotating'")
    if state.visible not in (0, 1) or not 0 <= state.hidden < n:
        raise ValueError("state must belong to Z_2 x Z_n")
    if probe not in (0, 1):
        raise ValueError("probe must be 0 or 1")


def step(n: int, variant: str, state: State, probe: int = 0) -> State:
    """x' = x xor 1 xor (probe and z=0); z' = z + delta (mod n)."""
    _validate(n, variant, state, probe)
    return State(
        state.visible ^ 1 ^ int(bool(probe) and state.hidden == 0),
        (state.hidden + int(variant == "rotating")) % n,
    )


def trajectory(
    n: int, variant: str, hidden_initial: int,
    probes: Iterable[int] = (), visible_initial: int = 0,
) -> tuple[int, ...]:
    """Measured outputs y_0..y_T for a T-transition open-loop action word."""
    s = State(visible_initial, hidden_initial)
    _validate(n, variant, s, 0)
    outputs = [s.visible]
    for probe in probes:
        s = step(n, variant, s, probe)
        outputs.append(s.visible)
    return tuple(outputs)


def trace_set(n: int, variant: str, probes: tuple[int, ...], visible_initial: int = 0) -> frozenset[tuple[int, ...]]:
    """All traces possible given an unknown hidden phase and known initial readout."""
    return frozenset(
        trajectory(n, variant, z, probes, visible_initial)
        for z in range(n)
    )


def output_classes(n: int, variant: str, horizon: int) -> dict[tuple[int, ...], tuple[State, ...]]:
    """Passive finite-horizon trace equivalence, exhaustively on 2n states."""
    if horizon < 0:
        raise ValueError("horizon must be nonnegative")
    result: dict[tuple[int, ...], list[State]] = {}
    for x, z in product(range(2), range(n)):
        s = State(x, z)
        signature = trajectory(n, variant, z, (0,) * horizon, x)
        result.setdefault(signature, []).append(s)
    return {signature: tuple(states) for signature, states in result.items()}


def passive_cycle_lengths(n: int, variant: str) -> tuple[int, ...]:
    """Cycle decomposition of the bijection F_0 on Z_2 x Z_n."""
    states = {State(x, z) for x, z in product(range(2), range(n))}
    seen: set[State] = set()
    lengths: list[int] = []
    for source in sorted(states):
        if source in seen:
            continue
        cursor = source
        length = 0
        while cursor not in seen:
            seen.add(cursor)
            length += 1
            cursor = step(n, variant, cursor)
        if cursor != source:
            raise AssertionError("unexpected noncycle transient; F_0 must be bijective")
        lengths.append(length)
    if seen != states or sum(lengths) != 2 * n:
        raise AssertionError("cycle decomposition failed to cover state space")
    return tuple(sorted(lengths))


def freeze_count(outputs: tuple[int, ...]) -> int:
    return sum(int(x == y) for x, y in zip(outputs, outputs[1:]))


def first_separating_fixed_word(n: int, max_horizon: int | None = None) -> tuple[int, tuple[int, ...]] | None:
    """Exhaustive minimum nonadaptive word separating all unknown phases.

    This independently verifies the constructive upper bound. Theorem OIC-T02's
    adaptive-policy lower bound is proved analytically in THEOREMS.md.
    """
    if n < 2:
        raise ValueError("n must be >= 2")
    if max_horizon is None:
        max_horizon = n
    if max_horizon < 0:
        raise ValueError("max_horizon must be nonnegative")
    for t in range(max_horizon + 1):
        for word in product((0, 1), repeat=t):
            if trace_set(n, "fixed", word).isdisjoint(trace_set(n, "rotating", word)):
                return (t, word)
    return None


def odd_triangle_contextuality() -> dict[str, object]:
    """Separate C3 parity-context example: locally consistent, globally unsatisfiable.

    Each edge context has a uniform distribution on its two satisfying binary
    assignments. Its one-variable marginals are therefore uniform on overlaps.
    """
    edges = ((0, 1, 0), (1, 2, 0), (2, 0, 1))
    allowed_by_edge = [
        tuple((a, b) for a, b in product((0, 1), repeat=2) if (a ^ b) == parity)
        for _, _, parity in edges
    ]
    global_sections = [
        bits for bits in product((0, 1), repeat=3)
        if all((bits[i] ^ bits[j]) == parity for i, j, parity in edges)
    ]
    local_uniform_marginals = all(
        sum(pair[pos] == value for pair in allowed) * 2 == len(allowed)
        for allowed in allowed_by_edge for pos in (0, 1) for value in (0, 1)
    )
    holonomy = 0
    for _, _, parity in edges:
        holonomy ^= parity
    return {
        "context_count": len(edges),
        "local_sections_per_context": [len(allowed) for allowed in allowed_by_edge],
        "uniform_overlap_marginals": local_uniform_marginals,
        "global_section_count": len(global_sections),
        "z2_cycle_holonomy": holonomy,
    }


def benchmark(n: int) -> dict[str, object]:
    if n < 3:
        raise ValueError("topology benchmark requires n >= 3")
    a = passive_cycle_lengths(n, "fixed")
    b = passive_cycle_lengths(n, "rotating")
    classes_a = output_classes(n, "fixed", 2 * n)
    classes_b = output_classes(n, "rotating", 2 * n)
    found = first_separating_fixed_word(n)
    if found is None:
        raise AssertionError("no distinguishing intervention found within n steps")
    t, word = found
    fixed_freezes = sorted({freeze_count(y) for y in trace_set(n, "fixed", word)})
    rotating_freezes = sorted({freeze_count(y) for y in trace_set(n, "rotating", word)})
    return {
        "hidden_phases": n,
        "states": 2 * n,
        "passive_cycle_lengths_fixed": list(a),
        "passive_cycle_lengths_rotating": list(b),
        "passive_graph_betti_0_fixed": len(a),
        "passive_graph_betti_0_rotating": len(b),
        "passive_graph_betti_1_fixed": len(a),
        "passive_graph_betti_1_rotating": len(b),
        "passive_observable_classes_fixed": len(classes_a),
        "passive_observable_classes_rotating": len(classes_b),
        "earliest_separating_fixed_action_horizon": t,
        "first_separating_fixed_action_word": "".join(map(str, word)),
        "possible_freeze_counts_fixed": fixed_freezes,
        "possible_freeze_counts_rotating": rotating_freezes,
    }


def full_reference(ns: Iterable[int] = (3, 4, 5, 6, 7, 8)) -> dict[str, object]:
    return {
        "status": "finite_exact_reference_and_written_proofs_only",
        "model": "Z2_x_Zn_hidden_phase",
        "intervention": "probe_couples_hidden_zero_to_visible_bit",
        "results": [benchmark(n) for n in ns],
        "contextuality_separate_example": odd_triangle_contextuality(),
        "model_training_or_consciousness_claim": False,
    }

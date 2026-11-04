"""Finite transition algebras and exact semantic decoders.

A program is a word over instruction indices; each instruction acts as a total
finite-state transition. Semiconjugacy is checked exhaustively on all hidden
states before program-level claims are made.
"""
from __future__ import annotations

from itertools import product
from typing import Sequence


def validate_transition(t: Sequence[int]) -> None:
    if not t or any(type(v) is not int or not 0 <= v < len(t) for v in t):
        raise ValueError("transition must be a nonempty finite endomorphism")


def run_word(transitions: Sequence[Sequence[int]], word: Sequence[int], start: int) -> int:
    if not transitions:
        raise ValueError("instruction set must be nonempty")
    width = len(transitions[0])
    if not 0 <= start < width:
        raise ValueError("invalid initial state")
    for t in transitions:
        validate_transition(t)
        if len(t) != width:
            raise ValueError("instruction transitions must share a state space")
    s = start
    for letter in word:
        if type(letter) is not int or not 0 <= letter < len(transitions):
            raise ValueError("invalid instruction")
        s = transitions[letter][s]
    return s


def semiconjugacy_defects(hidden: Sequence[Sequence[int]],
                          symbolic: Sequence[Sequence[int]], pi: Sequence[int]):
    """All (letter, hidden_state) where pi(F h) differs from T(pi h)."""
    if not symbolic or len(symbolic) != len(hidden) or not pi:
        raise ValueError("missing or unmatched instructions")
    for f in hidden:
        validate_transition(f)
        if len(f) != len(pi):
            raise ValueError("hidden transition width differs from pi")
    for t in symbolic:
        validate_transition(t)
    n = len(symbolic[0])
    if any(len(t) != n for t in symbolic) or any(type(v) is not int or not 0 <= v < n for v in pi):
        raise ValueError("decoder and target have incompatible sizes")
    return [(letter, h) for letter, (f, t) in enumerate(zip(hidden, symbolic))
            for h in range(len(pi)) if pi[f[h]] != t[pi[h]]]


def congruence_defects(hidden: Sequence[Sequence[int]], pi: Sequence[int]):
    """Pairs sharing one decoded state whose successors decode differently."""
    return [(letter, h, g) for letter, f in enumerate(hidden)
            for h in range(len(pi)) for g in range(h + 1, len(pi))
            if pi[h] == pi[g] and pi[f[h]] != pi[f[g]]]


def words(alphabet_size: int, max_length: int):
    if alphabet_size < 1 or max_length < 0:
        raise ValueError("invalid alphabet / length")
    for length in range(max_length + 1):
        yield from product(range(alphabet_size), repeat=length)


def lifted_actions(symbolic: Sequence[Sequence[int]], tag_count: int = 2):
    """A transparent, *constructed* semiconjugate H=S×{0,...,tags-1} example."""
    if not symbolic or tag_count < 1:
        raise ValueError("bad symbolic action or tag count")
    n = len(symbolic[0])
    for t in symbolic:
        validate_transition(t)
        if len(t) != n:
            raise ValueError("incompatible transitions")
    pi = tuple(s for s in range(n) for _ in range(tag_count))
    hidden = []
    for f in symbolic:
        hidden.append(tuple(f[s] * tag_count + ((tag + 1) % tag_count)
                            for s in range(n) for tag in range(tag_count)))
    return tuple(hidden), pi


def exhaustive_word_check(hidden, symbolic, pi, max_length=5):
    """Validate finite-word action equality on a fixed finite example."""
    defects = semiconjugacy_defects(hidden, symbolic, pi)
    count = 0
    violations = 0
    for word in words(len(symbolic), max_length):
        for h in range(len(pi)):
            count += 1
            if pi[run_word(hidden, word, h)] != run_word(symbolic, word, pi[h]):
                violations += 1
    return {"one_step_defects": len(defects), "program_checks": count,
            "program_violations": violations}


def minimal_period(t: Sequence[int], start: int) -> int | None:
    """Period of start if on a cycle; transient points return None."""
    validate_transition(t)
    x = t[start]
    k = 1
    while x != start and k <= len(t):
        x = t[x]
        k += 1
    return k if x == start else None

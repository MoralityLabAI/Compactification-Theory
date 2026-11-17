"""HC-01: finite, exact certificates for stateful chunk abstractions.

Standard-library oracle. Finite deterministic machines only; this does not
train or inspect an H-Net checkpoint and does not infer a neural guarantee.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
from typing import Mapping


@dataclass(frozen=True)
class Machine:
    states: tuple[str, ...]
    alphabet: tuple[str, ...]
    output: Mapping[str, str]
    delta: Mapping[tuple[str, str], str]

    def __post_init__(self) -> None:
        if not self.states or not self.alphabet:
            raise ValueError("nonempty states and operation alphabet required")
        if len(set(self.states)) != len(self.states):
            raise ValueError("duplicate states")
        if len(set(self.alphabet)) != len(self.alphabet):
            raise ValueError("duplicate operations")
        if set(self.output) != set(self.states):
            raise ValueError("exactly one output per state required")
        expected = {(s, a) for s in self.states for a in self.alphabet}
        if set(self.delta) != expected:
            raise ValueError("transition table must be total, with no extra entries")
        if any(t not in self.output for t in self.delta.values()):
            raise ValueError("transition exits declared state domain")

    def step(self, state: str, word: tuple[str, ...]) -> str:
        if state not in self.output:
            raise ValueError("unknown initial state")
        for action in word:
            if action not in self.alphabet:
                raise ValueError("unknown action")
            state = self.delta[(state, action)]
        return state

    def observe(self, state: str, word: tuple[str, ...] = ()) -> str:
        return self.output[self.step(state, word)]


def _canonical(states: tuple[str, ...], signatures: Mapping[str, tuple]) -> dict[str, int]:
    code_by_signature: dict[tuple, int] = {}
    result: dict[str, int] = {}
    for state in states:
        signature = signatures[state]
        if signature not in code_by_signature:
            code_by_signature[signature] = len(code_by_signature)
        result[state] = code_by_signature[signature]
    return result


def _same_partition(states: tuple[str, ...], left: Mapping[str, int],
                    right: Mapping[str, int]) -> bool:
    return all((left[s] == left[t]) == (right[s] == right[t])
               for s in states for t in states)


def horizon_partition(machine: Machine, horizon: int) -> dict[str, int]:
    """Exact observation classes through at most `horizon` future actions.

    Horizon zero includes the current output; a transition spends one unit of
    horizon. This partition need not be a fixed-horizon transition congruence.
    """
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 0:
        raise ValueError("horizon must be a nonnegative integer")
    states = machine.states
    part = _canonical(states, {s: (machine.output[s],) for s in states})
    for _ in range(horizon):
        part = _canonical(states, {
            s: (machine.output[s], *(part[machine.delta[(s, a)]] for a in machine.alphabet))
            for s in states
        })
    return part


def behavioral_partition(machine: Machine) -> dict[str, int]:
    """Minimal right-congruent partition for every admissible future word."""
    states = machine.states
    part = horizon_partition(machine, 0)
    for _ in range(len(states)):
        nxt = _canonical(states, {
            s: (machine.output[s], *(part[machine.delta[(s, a)]] for a in machine.alphabet))
            for s in states
        })
        if _same_partition(states, part, nxt):
            return nxt
        part = nxt
    raise AssertionError("finite partition refinement failed to converge")


def distinguishing_suffix(machine: Machine, left: str, right: str) -> tuple[str, ...] | None:
    """Shortest word exposing different terminal observations, if one exists."""
    if left not in machine.output or right not in machine.output:
        raise ValueError("unknown state")
    queue = deque([(left, right, ())])
    seen = {(left, right)}
    while queue:
        x, y, word = queue.popleft()
        if machine.output[x] != machine.output[y]:
            return word
        for action in machine.alphabet:
            pair = (machine.delta[(x, action)], machine.delta[(y, action)])
            if pair not in seen:
                seen.add(pair)
                queue.append((*pair, (*word, action)))
    return None


def audit(machine: Machine, encoding: Mapping[str, str]) -> dict:
    """Distinguish observational sufficiency from transition factorization.

    A finite encoding has a deterministic quotient transition only if each
    encoding fiber has the same output and the same encoded successor for each
    action. Observational sufficiency alone can permit non-congruent refinement.
    """
    if set(encoding) != set(machine.states):
        raise ValueError("exactly one code for each reachable state required")
    codes = {encoding[s] for s in machine.states}
    canonical = behavioral_partition(machine)
    min_states = len(set(canonical.values()))
    pairs = [(s, t) for i, s in enumerate(machine.states)
             for t in machine.states[i + 1:] if encoding[s] == encoding[t]]
    witness = None
    factor_failure = None
    for s, t in pairs:
        suffix = distinguishing_suffix(machine, s, t)
        if suffix is not None:
            candidate = {"states": [s, t], "suffix": list(suffix),
                         "observations": [machine.observe(s, suffix), machine.observe(t, suffix)]}
            if witness is None or (len(suffix), suffix) < (len(witness["suffix"]), tuple(witness["suffix"])):
                witness = candidate
        if factor_failure is None:
            if machine.output[s] != machine.output[t]:
                factor_failure = {"states": [s, t], "kind": "readout"}
            else:
                for action in machine.alphabet:
                    cs = encoding[machine.delta[(s, action)]]
                    ct = encoding[machine.delta[(t, action)]]
                    if cs != ct:
                        factor_failure = {"states": [s, t], "kind": "transition",
                                          "action": action, "successor_codes": [cs, ct]}
                        break
    return {
        "reachable_states": len(machine.states),
        "behavioral_classes": min_states,
        "lower_bound_bits_for_exact_future_observation": (min_states - 1).bit_length(),
        "encoding_codes": len(codes),
        "minimum_bits_for_this_codebook": (len(codes) - 1).bit_length(),
        "observation_sufficient": witness is None,
        "transition_factorable": factor_failure is None,
        "observation_collision": witness,
        "factorization_failure": factor_failure,
    }


def shift_machine(allow_shift: bool = True) -> Machine:
    """Two raw bits, first-bit readout; shift exposes the buffered second bit."""
    states = ("00", "01", "10", "11")
    alphabet = ("hold", "shift") if allow_shift else ("hold",)
    transitions = {}
    for s in states:
        for a in alphabet:
            transitions[(s, a)] = s[1] + s[0] if a == "shift" else s
    return Machine(states, alphabet, {s: s[0] for s in states}, transitions)


def main() -> None:
    for shift in (False, True):
        machine = shift_machine(shift)
        result = audit(machine, {s: s[0] for s in machine.states})
        print(json.dumps({"shift_allowed": shift, **result}, sort_keys=True))


if __name__ == "__main__":
    main()

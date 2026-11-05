"""SIML-01 finite recurrent *learned-table* student and task-graph oracle.

A deterministic learning/reference environment, not a neural model.  The teacher
supplies supervised one-step successor codes in the training partition; the
held-out T1 words are compositions absent from that partition.  Tables are
fitted from counts, then rolled out autoregressively in the learned state.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import json
import random

ACTIONS = ("identity", "flip_y", "flip_b", "reveal_b", "reveal_c")
T1_WORDS = (("flip_b", "reveal_b"), ("flip_y", "reveal_b"))
WEIGHTS = {"T0": 0.25, "T1": 0.45, "T4": 0.30}
GRAPH = {
    "schema": "siml01.graph.v1", "families": WEIGHTS,
    "nodes": {
        "T0": {"interventions": [], "horizon": 0, "domain": "bit3"},
        "T1": {"interventions": [list(w) for w in T1_WORDS], "horizon": 2, "domain": "bit3"},
        "T4": {"interventions": ["shear"], "horizon": 1, "domain": "Z31-square"},
    },
    "edges": [
        {"source": "T0", "target": "T1", "kind": "prerequisite_hypothesis"},
        {"source": "T1", "target": "T4", "kind": "transfer_hypothesis"},
    ],
    "loss": "unit_correctness", "sampling_unit": "independent initial world",
}


def canonical(data: object) -> bytes:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def digest(data: object) -> str:
    return hashlib.sha256(canonical(data)).hexdigest()


GRAPH_SHA = digest(GRAPH)


def apply_bit_action(x: tuple[int, ...], action: str) -> tuple[int, ...]:
    y, b, c = x
    if action == "identity":
        return x
    if action == "flip_y":
        return (y ^ 1, b, c)
    if action == "flip_b":
        return (y, b ^ 1, c)
    if action == "reveal_b":
        return (b, b, c)
    if action == "reveal_c":
        return (c, b, c)
    raise ValueError("unknown intervention")


def all_bit_states():
    return [(y, b, c) for y in range(2) for b in range(2) for c in range(2)]


def _majority(counts: Counter) -> tuple[int, ...]:
    """Deterministic tie break, recorded as part of the model specification."""
    return min(counts, key=lambda x: (-counts[x], x))


@lru_cache(maxsize=None)
def fit_bit_transitions(retained_bits: int) -> dict:
    if retained_bits not in (1, 2, 3):
        raise ValueError("retained_bits must be 1, 2 or 3")
    counts = defaultdict(Counter)
    for x in all_bit_states():
        z = x[:retained_bits]
        for action in ACTIONS:
            counts[(z, action)][apply_bit_action(x, action)[:retained_bits]] += 1
    return {key: _majority(v) for key, v in counts.items()}


@lru_cache(maxsize=None)
def fit_shear_transitions(joint_state: bool) -> dict:
    counts = defaultdict(Counter)
    p = 31
    for a in range(p):
        for b in range(p):
            s = (a + b) % p
            z = (s, a) if joint_state else (s,)
            t = (s + a) % p
            z_next = (t, a) if joint_state else (t,)
            counts[z][z_next] += 1
    return {key: _majority(v) for key, v in counts.items()}


@dataclass(frozen=True)
class MicroLearner:
    retained_bits: int = 1
    joint_shear: bool = False

    def __post_init__(self):
        if type(self.retained_bits) is not int or self.retained_bits not in (1, 2, 3):
            raise ValueError("retained_bits must be 1..3")
        if type(self.joint_shear) is not bool:
            raise ValueError("joint_shear must be a bool")

    @property
    def checkpoint(self) -> str:
        return digest({"schema": "siml01.tabular.v1", **asdict(self)})

    def update(self, operator: str) -> "MicroLearner":
        if operator == "retain_b" and self.retained_bits == 1:
            return MicroLearner(2, self.joint_shear)
        if operator == "retain_c" and self.retained_bits == 2:
            return MicroLearner(3, self.joint_shear)
        if operator == "drop_b" and self.retained_bits >= 2:
            return MicroLearner(1, self.joint_shear)
        if operator == "install_shear" and not self.joint_shear:
            return MicroLearner(self.retained_bits, True)
        raise ValueError("inapplicable update operator")

    def predict(self, node: str, world: tuple[int, ...], variant: int = 0) -> int:
        if node == "T0":
            return world[0]
        if node == "T1":
            z = world[:self.retained_bits]
            transitions = fit_bit_transitions(self.retained_bits)
            for action in T1_WORDS[variant]:
                z = transitions[(z, action)]
            return z[0]
        if node == "T4":
            a, b = world
            s = (a + b) % 31
            z = (s, a) if self.joint_shear else (s,)
            return fit_shear_transitions(self.joint_shear)[z][0]
        raise ValueError("unknown node")


def sample(node: str, rng: random.Random) -> tuple[tuple[int, ...], int, int]:
    """An independent block's world, variant and answer; one draw per node."""
    if node == "T0":
        x = tuple(rng.randrange(2) for _ in range(3))
        return x, 0, x[0]
    if node == "T1":
        x = tuple(rng.randrange(2) for _ in range(3))
        variant = rng.randrange(2)
        state = x
        for action in T1_WORDS[variant]:
            state = apply_bit_action(state, action)
        return x, variant, state[0]
    if node == "T4":
        a, b = rng.randrange(31), rng.randrange(31)
        return (a, b), 0, (2 * a + b) % 31
    raise ValueError("unknown node")


def exact_accuracy(model: MicroLearner, node: str) -> float:
    if node == "T4":
        cases = [((a, b), 0, (2 * a + b) % 31) for a in range(31) for b in range(31)]
    elif node == "T0":
        cases = [(x, 0, x[0]) for x in all_bit_states()]
    elif node == "T1":
        cases = []
        for x in all_bit_states():
            for variant, word in enumerate(T1_WORDS):
                state = x
                for action in word:
                    state = apply_bit_action(state, action)
                cases.append((x, variant, state[0]))
    else:
        raise ValueError("unknown node")
    return sum(model.predict(node, x, variant) == y for x, variant, y in cases) / len(cases)


def diagnose(model: MicroLearner) -> dict:
    """Exact finite diagnostic for this declared world, not model introspection."""
    b_pair = ((0, 0, 0), (0, 1, 0))
    shear_pair = ((0, 0), (1, 30))  # same present sum, distinct future output
    return {
        "checkpoint": model.checkpoint,
        "source": "finite-exact privileged world oracle",
        "development_accuracy": {n: exact_accuracy(model, n) for n in WEIGHTS},
        "b_future_collision": (b_pair[0][:model.retained_bits] == b_pair[1][:model.retained_bits]
                               and b_pair[0][1] != b_pair[1][1]),
        "shear_future_collision": (not model.joint_shear
                                   and sum(shear_pair[0]) % 31 == sum(shear_pair[1]) % 31
                                   and (2*shear_pair[0][0]+shear_pair[0][1]) % 31
                                   != (2*shear_pair[1][0]+shear_pair[1][1]) % 31),
        "live_state_bits": model.retained_bits + (10 if model.joint_shear else 5),
    }


def heuristic_proposal(model: MicroLearner) -> dict | None:
    d = diagnose(model)
    if d["b_future_collision"]:
        op, node, predicted = "retain_b", "T1", 0.225
    elif d["shear_future_collision"]:
        op, node, predicted = "install_shear", "T4", 0.29
    else:
        return None
    return {"operator": op, "target": node, "predicted_potential_gain": predicted,
            "diagnostics": d, "competing_explanation": "finite-state label-table overfitting",
            "source": "v0 deterministic rule-based proposer"}

"""Algernon-02: semantic pointer-machine IR with an exact BN254 R1CS mirror.

Python R1CS evaluator and Circom source emitter are *distinct backends*.  The
Circom source has not been compiled by Circom in this environment.  This package
produces no zk proof and contains no Qwen/LOTUS inference.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any

# Circom default BN254 *scalar* field; see https://docs.circom.io/background/background/
P = 21888242871839275222246405745257275088548364400416034343698204186575808495617


def canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def digest(obj: Any) -> str:
    return hashlib.sha256(canonical(obj)).hexdigest()


@dataclass(frozen=True)
class PointerTask:
    """Total deterministic machine T: state -> successor(state); fixed step count."""

    successors: tuple[int, ...]
    initial: int
    steps: int

    def __post_init__(self):
        n = len(self.successors)
        if not 2 <= n <= 16 or any(type(s) is not int or not 0 <= s < n for s in self.successors):
            raise ValueError("successors must form a total function on 2..16 states")
        if type(self.initial) is not int or not 0 <= self.initial < n:
            raise ValueError("invalid initial state")
        if type(self.steps) is not int or not 0 <= self.steps <= 64:
            raise ValueError("steps must be in 0..64")

    @property
    def n(self):
        return len(self.successors)

    def public(self):
        return {"edge": [[int(self.successors[i] == j) for j in range(self.n)]
                         for i in range(self.n)],
                "start": [int(self.initial == i) for i in range(self.n)]}

    def trace(self):
        out = [self.initial]
        for _ in range(self.steps):
            out.append(self.successors[out[-1]])
        return out

    def description(self):
        return {"successors": list(self.successors), "initial": self.initial,
                "steps": self.steps, "trace": self.trace()}


def case(family: str, seed: int, ident: int, n: int = 4, steps: int = 8) -> PointerTask:
    """Distinct generator families for training, calibration, and depth testing."""
    if not isinstance(ident, int) or ident < 0:
        raise ValueError("ident must be nonnegative")
    rng = random.Random(f"{family}:{seed}:{ident}:{n}:{steps}")
    if family == "random_function":
        f = tuple(rng.randrange(n) for _ in range(n))
    elif family == "permutation":
        a = list(range(n)); rng.shuffle(a); f = tuple(a)
    elif family == "long_cycle":
        a = list(range(n)); rng.shuffle(a)
        f_list = [0] * n
        for i in range(n):
            f_list[a[i]] = a[(i + 1) % n]
        f = tuple(f_list)
    elif family == "sink":
        end = rng.randrange(n)
        f = tuple(end if i != end else end for i in range(n))
    else:
        raise ValueError(f"unknown generator family {family}")
    return PointerTask(f, rng.randrange(n), steps)


def onehot(index: int, n: int) -> list[int]:
    return [int(index == i) for i in range(n)]


class R1CS:
    """Constraint triples (A·w)(B·w) - C·w == 0 mod p.

    Wire 0 is constant 1. `public_values` are checked *separately*: an exported
    constraint list is insufficient to bind a statement to specific public data.
    """

    def __init__(self):
        self.names = ["ONE"]
        self.indices = {"ONE": 0}
        self.constraints: list[tuple[dict[int, int], dict[int, int], dict[int, int]]] = []
        self.public_names: list[str] = []

    def wire(self, name: str, public: bool = False) -> int:
        if name in self.indices:
            raise ValueError(f"duplicate signal {name}")
        idx = len(self.names)
        self.names.append(name)
        self.indices[name] = idx
        if public:
            self.public_names.append(name)
        return idx

    def add(self, a: dict[int, int], b: dict[int, int], c: dict[int, int]):
        self.constraints.append((a, b, c))

    def boolean(self, x: int):
        self.add({x: 1}, {x: 1, 0: -1}, {})

    def eqzero(self, terms: dict[int, int]):
        self.add(terms, {0: 1}, {})

    def eval_linear(self, expr: dict[int, int], w: list[int]) -> int:
        return sum(coeff * w[idx] for idx, coeff in expr.items()) % P

    def check(self, witness: list[int], public_values: dict[str, int], *, collect: int = 8):
        if len(witness) != len(self.names) or witness[0] != 1 or any(
            type(v) is not int or not 0 <= v < P for v in witness
        ):
            return {"valid": False, "error": "invalid witness length or field element", "bad": []}
        for name in self.public_names:
            if name not in public_values or witness[self.indices[name]] != public_values[name]:
                return {"valid": False, "error": "public input mismatch", "bad": [name]}
        bad = []
        count = 0
        for i, (a, b, c) in enumerate(self.constraints):
            residual = (self.eval_linear(a, witness) * self.eval_linear(b, witness)
                        - self.eval_linear(c, witness)) % P
            if residual:
                count += 1
                if len(bad) < collect:
                    bad.append({"constraint_index": i, "residual": str(residual)})
        return {"valid": count == 0, "violations": count, "bad": bad}

    def export_circom_constraints(self):
        """Circom-like JSON constraint shape; wire numbering is THIS emitter's own."""
        def terms(obj):
            return {str(k): str(v % P) for k, v in sorted(obj.items()) if v % P}
        return {"constraints": [[terms(a), terms(b), terms(c)]
                                for a, b, c in self.constraints]}


def build(task: PointerTask):
    """Fully constrained onehot machine trace and all intermediate mul wires."""
    n, steps = task.n, task.steps
    r = R1CS()
    edge = [[r.wire(f"edge[{i}][{j}]", public=True) for j in range(n)] for i in range(n)]
    start = [r.wire(f"start[{i}]", public=True) for i in range(n)]
    state = [[r.wire(f"state[{t}][{i}]") for i in range(n)] for t in range(steps + 1)]
    prod = [[[r.wire(f"prod[{t}][{i}][{j}]") for j in range(n)]
             for i in range(n)] for t in range(steps)]
    final = [r.wire(f"final[{i}]") for i in range(n)]
    for i in range(n):
        r.boolean(start[i])
        r.eqzero({state[0][i]: 1, start[i]: -1})
        for j in range(n):
            r.boolean(edge[i][j])
        r.eqzero({**{edge[i][j]: 1 for j in range(n)}, 0: -1})
    r.eqzero({**{start[i]: 1 for i in range(n)}, 0: -1})
    for t in range(steps + 1):
        for i in range(n):
            r.boolean(state[t][i])
        r.eqzero({**{state[t][i]: 1 for i in range(n)}, 0: -1})
    for t in range(steps):
        for i in range(n):
            for j in range(n):
                r.add({state[t][i]: 1}, {edge[i][j]: 1}, {prod[t][i][j]: 1})
        for j in range(n):
            r.eqzero({state[t + 1][j]: 1,
                      **{prod[t][i][j]: -1 for i in range(n)}})
    for i in range(n):
        r.eqzero({final[i]: 1, state[steps][i]: -1})
    return r


def witness_for_trace(task: PointerTask, circuit: R1CS, history: list[int]):
    """Construct a candidate trace witness; may be invalid under the public machine.

    Product wires follow the *claimed* predecessor, so the checker catches
    incorrect successor states instead of silently replacing them with oracle.
    """
    n, steps = task.n, task.steps
    if len(history) != steps + 1 or any(type(v) is not int or v < 0 or v >= n for v in history):
        raise ValueError("candidate history must have a valid state per step")
    w = [0] * len(circuit.names)
    w[0] = 1
    for i in range(n):
        w[circuit.indices[f"start[{i}]"]] = int(task.initial == i)
        for j in range(n):
            w[circuit.indices[f"edge[{i}][{j}]"]] = int(task.successors[i] == j)
    for t, index in enumerate(history):
        for i in range(n):
            w[circuit.indices[f"state[{t}][{i}]"]] = int(i == index)
    for t in range(steps):
        for i in range(n):
            for j in range(n):
                w[circuit.indices[f"prod[{t}][{i}][{j}]"]] = int(
                    i == history[t] and j == task.successors[i])
    for i in range(n):
        w[circuit.indices[f"final[{i}]"]] = int(i == history[-1])
    public = {name: w[circuit.indices[name]] for name in circuit.public_names}
    return w, public


def witness_for(task: PointerTask, circuit: R1CS):
    return witness_for_trace(task, circuit, task.trace())


def emit_circom(n: int, steps: int):
    """One-port explicit Circom source; compilation requires external circom binary."""
    PointerTask(tuple(range(n)), 0, steps)  # validates parameters
    return f'''pragma circom 2.1.6;
// Generated from Algernon-02 pointer-machine semantics, N={n}, STEPS={steps}.
// Signal arrays are public only through main's public annotation.
template PointerTrace(N, STEPS) {{
    signal input edge[N][N];
    signal input start[N];
    signal state[STEPS + 1][N];
    signal product[STEPS][N][N];
    signal output final[N];

    var startSum = 0;
    for (var i = 0; i < N; i++) {{
        start[i] * (start[i] - 1) === 0;
        state[0][i] <== start[i];
        startSum += start[i];
        var edgeSum = 0;
        for (var j = 0; j < N; j++) {{
            edge[i][j] * (edge[i][j] - 1) === 0;
            edgeSum += edge[i][j];
        }}
        edgeSum === 1;
    }}
    startSum === 1;

    for (var t = 0; t < STEPS; t++) {{
        for (var i = 0; i < N; i++) {{
            for (var j = 0; j < N; j++) {{
                product[t][i][j] <== state[t][i] * edge[i][j];
            }}
        }}
        for (var j = 0; j < N; j++) {{
            var nextSum = 0;
            for (var i = 0; i < N; i++) {{
                nextSum += product[t][i][j];
            }}
            state[t + 1][j] <== nextSum;
        }}
    }}
    for (var t = 0; t <= STEPS; t++) {{
        var stateSum = 0;
        for (var i = 0; i < N; i++) {{
            state[t][i] * (state[t][i] - 1) === 0;
            stateSum += state[t][i];
        }}
        stateSum === 1;
    }}
    for (var i = 0; i < N; i++) {{
        final[i] <== state[STEPS][i];
    }}
}}
component main {{public [edge, start]}} = PointerTrace({n}, {steps});
'''

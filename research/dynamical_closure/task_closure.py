"""Exact finite-state task quotients and loss certificates. Standard library only."""
from collections import deque
from dataclasses import dataclass
from fractions import Fraction
from typing import Hashable, Iterable, Sequence


def canonical(values: Iterable[Hashable]) -> tuple[int, ...]:
    """Label equal values by their first occurrence; values need stable equality."""
    labels = {}
    result = []
    try:
        for value in values:
            if value != value:
                raise ValueError("labels must have reflexive equality")
            if value not in labels:
                labels[value] = len(labels)
            result.append(labels[value])
    except TypeError as exc:
        raise ValueError("labels must be hashable") from exc
    return tuple(result)


@dataclass(frozen=True)
class FiniteSystem:
    """State x has output task[x]; transitions[a][x] applies intervention a."""
    transitions: tuple[tuple[int, ...], ...]
    task: tuple[Hashable, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "task", tuple(self.task))
        object.__setattr__(self, "transitions", tuple(tuple(f) for f in self.transitions))
        n = len(self.task)
        if n == 0:
            raise ValueError("at least one state is required")
        canonical(self.task)
        for f in self.transitions:
            if len(f) != n or any(type(x) is not int or not 0 <= x < n for x in f):
                raise ValueError("every intervention must map the whole state domain to itself")


def _partition(values: Sequence[Hashable], n: int) -> tuple[int, ...]:
    if len(values) != n:
        raise ValueError("partition length must match the state domain")
    return canonical(values)


def refines(fine: Sequence[Hashable], coarse: Sequence[Hashable]) -> bool:
    """Whether equality of fine codes implies equality of coarse codes."""
    p = canonical(fine)
    q = _partition(coarse, len(p))
    return all(p[x] != p[y] or q[x] == q[y]
               for x in range(len(p)) for y in range(x))


def refinement_history(system: FiniteSystem) -> tuple[tuple[int, ...], ...]:
    """P_h identifies equal task futures through length h; stop when stable."""
    p = canonical(system.task)
    history = [p]
    while True:
        q = canonical((p[x], tuple(p[f[x]] for f in system.transitions))
                      for x in range(len(p)))
        if q == p:
            return tuple(history)
        history.append(q)
        p = q


def is_task_closed(system: FiniteSystem, partition: Sequence[Hashable]) -> bool:
    """Test task factorization and closure under every declared intervention."""
    p = _partition(partition, len(system.task))
    return refines(p, system.task) and all(
        refines(p, tuple(p[f[x]] for x in range(len(p))))
        for f in system.transitions)


def quotient(system: FiniteSystem, partition: Sequence[Hashable]) -> FiniteSystem:
    """Construct the autonomous task-preserving quotient, or raise ValueError."""
    p = _partition(partition, len(system.task))
    if not is_task_closed(system, p):
        raise ValueError("partition does not preserve the task and all transitions")
    representatives = tuple(p.index(z) for z in range(max(p) + 1))
    return FiniteSystem(
        tuple(tuple(p[f[x]] for x in representatives) for f in system.transitions),
        tuple(system.task[x] for x in representatives))


def run_word(system: FiniteSystem, state: int, word: Sequence[int]) -> int:
    """Execute an externally supplied action word, including the empty word."""
    if type(state) is not int or not 0 <= state < len(system.task):
        raise ValueError("invalid initial state")
    for action in word:
        if type(action) is not int or not 0 <= action < len(system.transitions):
            raise ValueError("invalid intervention index")
        state = system.transitions[action][state]
    return state


def shortest_witness(system: FiniteSystem, x: int, y: int) -> tuple[int, ...] | None:
    """Shortest word separating task outputs; None means behavioral equivalence."""
    run_word(system, x, ())
    run_word(system, y, ())
    start = (x, y)
    parents = {start: None}
    pending = deque([start])
    while pending:
        pair = pending.popleft()
        u, v = pair
        if system.task[u] != system.task[v]:
            word = []
            while parents[pair] is not None:
                pair, action = parents[pair]
                word.append(action)
            return tuple(reversed(word))
        for action, f in enumerate(system.transitions):
            successor = (f[u], f[v])
            if successor not in parents:
                parents[successor] = (pair, action)
                pending.append(successor)
    return None



def task_loss_witness(system: FiniteSystem, partition: Sequence[Hashable]
                      ) -> tuple[int, int, tuple[int, ...]] | None:
    """An aliased pair with its shortest separating word, or None if sufficient."""
    p = _partition(partition, len(system.task))
    for x in range(len(p)):
        for y in range(x):
            if p[x] == p[y]:
                word = shortest_witness(system, x, y)
                if word is not None:
                    return x, y, word
    return None


def minimum_bits(system: FiniteSystem) -> int:
    """Minimum fixed-length bits for an exact, state-only behavioral code."""
    states = len(set(refinement_history(system)[-1]))
    return (states - 1).bit_length()


def _rational(value: int | Fraction) -> Fraction:
    if not isinstance(value, (int, Fraction)):
        raise ValueError("exact certificates require integer or Fraction values")
    return Fraction(value)


def bayes_risk(partition: Sequence[Hashable], targets: Sequence[int | Fraction],
               weights: Sequence[int | Fraction] | None = None) -> Fraction:
    """Exact inf_g E[(Y-g(C(X)))^2] for a fixed scalar target and measure."""
    n = len(targets)
    if n == 0:
        raise ValueError("at least one target is required")
    p = _partition(partition, n)
    ys = tuple(_rational(y) for y in targets)
    ws = tuple(_rational(w) for w in (weights if weights is not None else (1,) * n))
    if len(ws) != n or any(w < 0 for w in ws) or sum(ws) == 0:
        raise ValueError("weights must have matching length and positive total mass")
    groups = {}
    for z, y, w in zip(p, ys, ws):
        if w:
            groups.setdefault(z, []).append((y, w))
    total = Fraction(0)
    for values in groups.values():
        mass = sum(w for _, w in values)
        mean = sum(y * w for y, w in values) / mass
        total += sum(w * (y - mean) ** 2 for y, w in values)
    return total / sum(ws)


def minimax_risk(partition: Sequence[Hashable], targets: Sequence[int | Fraction]) -> Fraction:
    """Exact inf_g max_x (Y(x)-g(C(x)))^2, via scalar fiber midpoints."""
    if not targets:
        raise ValueError("at least one target is required")
    p = _partition(partition, len(targets))
    groups = {}
    for z, y in zip(p, targets):
        groups.setdefault(z, []).append(_rational(y))
    return max((max(ys) - min(ys)) ** 2 / 4 for ys in groups.values())

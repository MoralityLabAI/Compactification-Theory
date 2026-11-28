"""PGC-01 exact intervention experiments on finite Boolean register systems.

Register addresses, update ticks and the intervention class are supplied inputs.
The recovered quantities describe finite response functions.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
import json
from pathlib import Path
from typing import Iterable

Expr = tuple
Polynomial = frozenset[int]


def var(index: int) -> Expr:
    return ("var", index)


def const(value: int) -> Expr:
    if value not in (0, 1):
        raise ValueError("Boolean constant required")
    return ("const", value)


def xor(*terms: Expr) -> Expr:
    return ("xor", *terms)


def conjunction(*terms: Expr) -> Expr:
    return ("and", *terms)


def evaluate(expr: Expr, state: int) -> int:
    op = expr[0]
    if op == "var":
        return (state >> expr[1]) & 1
    if op == "const":
        return expr[1]
    if op == "xor":
        return sum(evaluate(x, state) for x in expr[1:]) % 2
    if op == "and":
        return int(all(evaluate(x, state) for x in expr[1:]))
    if op == "or":
        return int(any(evaluate(x, state) for x in expr[1:]))
    if op == "not":
        return 1 - evaluate(expr[1], state)
    raise ValueError(f"unknown operation {op}")


def rename_expr(expr: Expr, permutation: tuple[int, ...]) -> Expr:
    if expr[0] == "var":
        return var(permutation[expr[1]])
    if expr[0] == "const":
        return expr
    return (expr[0], *(rename_expr(x, permutation) for x in expr[1:]))


@dataclass(frozen=True)
class BooleanSystem:
    name: str
    updates: tuple[Expr, ...]

    @property
    def width(self) -> int:
        return len(self.updates)

    def step(self, state: int) -> int:
        if not 0 <= state < (1 << self.width):
            raise ValueError("state outside finite register alphabet")
        return sum(evaluate(expr, state) << j for j, expr in enumerate(self.updates))

    def table(self) -> tuple[int, ...]:
        return tuple(self.step(x) for x in range(1 << self.width))

    def relabel(self, permutation: tuple[int, ...]) -> BooleanSystem:
        if sorted(permutation) != list(range(self.width)):
            raise ValueError("register permutation required")
        result = [const(0)] * self.width
        for old, expr in enumerate(self.updates):
            result[permutation[old]] = rename_expr(expr, permutation)
        return BooleanSystem(self.name + "_relabeled", tuple(result))

    def as_dict(self) -> dict:
        return {"name": self.name, "width": self.width, "updates": self.updates}


def relabel_state(state: int, permutation: tuple[int, ...]) -> int:
    return sum(((state >> old) & 1) << new for old, new in enumerate(permutation))


def simulate(system: BooleanSystem, state: int, lag: int,
             controls: dict[int, dict[int, int]] | None = None) -> int:
    """Optional controls overwrite chosen registers after a numbered update."""
    if lag < 0:
        raise ValueError("nonnegative lag required")
    controls = controls or {}
    for tick in range(1, lag + 1):
        state = system.step(state)
        for register, value in controls.get(tick, {}).items():
            if not 0 <= register < system.width or value not in (0, 1):
                raise ValueError("invalid register control")
            state = (state & ~(1 << register)) | (value << register)
    return state


@dataclass(frozen=True)
class Profile:
    lag: int
    width: int
    outputs: tuple[int, ...]
    maximum: tuple[tuple[int, ...], ...]
    absolute_counts: tuple[tuple[int, ...], ...]
    signed_counts: tuple[tuple[int, ...], ...]

    @property
    def contexts_per_source(self) -> int:
        return 1 << (self.width - 1)

    def as_dict(self) -> dict:
        return {"lag": self.lag, "matrix_orientation": "row source, column target",
                "maximum": self.maximum, "absolute_counts": self.absolute_counts,
                "signed_counts": self.signed_counts,
                "contexts_per_source": self.contexts_per_source}


def profile_from_outputs(outputs: tuple[int, ...], width: int, lag: int) -> Profile:
    absolute = [[0] * width for _ in range(width)]
    signed = [[0] * width for _ in range(width)]
    for source in range(width):
        for context in range(1 << width):
            if (context >> source) & 1:
                continue
            low, high = outputs[context], outputs[context | (1 << source)]
            for target in range(width):
                difference = ((high >> target) & 1) - ((low >> target) & 1)
                absolute[source][target] += abs(difference)
                signed[source][target] += difference
    return Profile(lag, width, outputs,
                   tuple(tuple(int(value > 0) for value in row) for row in absolute),
                   tuple(map(tuple, absolute)), tuple(map(tuple, signed)))


def profiles(system: BooleanSystem, horizon: int) -> tuple[Profile, ...]:
    if system.width < 1 or horizon < 1:
        raise ValueError("positive register count and horizon required")
    table = system.table()
    current = tuple(range(1 << system.width))
    result = []
    for lag in range(1, horizon + 1):
        current = tuple(table[value] for value in current)
        result.append(profile_from_outputs(current, system.width, lag))
    return tuple(result)


def poly_add(*items: Polynomial) -> Polynomial:
    answer: set[int] = set()
    for item in items:
        answer.symmetric_difference_update(item)
    return frozenset(answer)


def poly_multiply(left: Polynomial, right: Polynomial) -> Polynomial:
    """Boolean quotient ring: coefficients mod 2 and x_i squared equals x_i."""
    answer: set[int] = set()
    for a in left:
        for b in right:
            monomial = a | b
            if monomial in answer:
                answer.remove(monomial)
            else:
                answer.add(monomial)
    return frozenset(answer)


def to_anf(expr: Expr) -> Polynomial:
    op = expr[0]
    one = frozenset({0})
    if op == "var":
        return frozenset({1 << expr[1]})
    if op == "const":
        return one if expr[1] else frozenset()
    terms = tuple(to_anf(x) for x in expr[1:])
    if op == "xor":
        return poly_add(*terms)
    if op == "and":
        answer = one
        for term in terms:
            answer = poly_multiply(answer, term)
        return answer
    if op == "or":
        product = one
        for term in terms:
            product = poly_multiply(product, poly_add(one, term))
        return poly_add(one, product)
    if op == "not":
        return poly_add(one, terms[0])
    raise ValueError(f"unknown operation {op}")


def poly_evaluate(poly: Polynomial, state: int) -> int:
    return sum((state & monomial) == monomial for monomial in poly) % 2


def poly_substitute(poly: Polynomial, substitutions: tuple[Polynomial, ...]) -> Polynomial:
    result = frozenset()
    for monomial in poly:
        term = frozenset({0})
        for source, replacement in enumerate(substitutions):
            if (monomial >> source) & 1:
                term = poly_multiply(term, replacement)
        result = poly_add(result, term)
    return result


def derivative(poly: Polynomial, source: int) -> Polynomial:
    bit = 1 << source
    return frozenset(monomial ^ bit for monomial in poly if monomial & bit)


def anf_powers(system: BooleanSystem, horizon: int) -> tuple[tuple[Polynomial, ...], ...]:
    base = tuple(to_anf(x) for x in system.updates)
    current = tuple(frozenset({1 << i}) for i in range(system.width))
    result = []
    for _ in range(horizon):
        current = tuple(poly_substitute(poly, current) for poly in base)
        result.append(current)
    return tuple(result)


def relation_diagnostics(matrix: Iterable[Iterable[int]]) -> dict:
    """Inspect the relation after collapsing update times to register labels."""
    matrix = tuple(tuple(row) for row in matrix)
    n = len(matrix)
    relation = {(i, j) for i in range(n) for j in range(n) if i != j and matrix[i][j]}
    missing = [(i, j, k) for i in range(n) for j in range(n) for k in range(n)
               if (i, j) in relation and (j, k) in relation and (i, k) not in relation]
    reach = set(relation)
    for middle in range(n):
        reach |= {(i, j) for i in range(n) for j in range(n)
                  if (i, middle) in reach and (middle, j) in reach}
    cycles = sorted(i for i in range(n) if (i, i) in reach)
    return {
        "scope": "register relation with update times collapsed",
        "self_responses": [i for i in range(n) if matrix[i][i]],
        "strict_edges": sorted(relation), "transitive": not missing,
        "acyclic": not cycles, "strict_partial_order": not missing and not cycles,
        "transitivity_witnesses": missing, "cycle_registers": cycles,
        "closure_adds_pairs": sorted(reach - relation),
    }


def accumulated_relation(items: tuple[Profile, ...]) -> tuple[tuple[int, ...], ...]:
    n = items[0].width
    return tuple(tuple(int(any(p.maximum[i][j] for p in items))
                       for j in range(n)) for i in range(n))


def affine_system(width: int, matrix: int, bias: int) -> BooleanSystem:
    updates = []
    for target in range(width):
        terms = [var(source) for source in range(width)
                 if (matrix >> (width * target + source)) & 1]
        if (bias >> target) & 1:
            terms.append(const(1))
        updates.append(xor(*terms))
    return BooleanSystem(f"affine_{width}_{matrix}_{bias}", tuple(updates))


def fixture_systems() -> tuple[BooleanSystem, ...]:
    def expression(data):
        return tuple(expression(item) if isinstance(item, list) else item for item in data)

    catalog = json.loads(Path(__file__).with_name("FIXTURES.json").read_text())
    return tuple(BooleanSystem(item["name"], tuple(expression(expr) for expr in item["updates"]))
                 for item in catalog["systems"])


def all_relabelings(system: BooleanSystem):
    for permutation in permutations(range(system.width)):
        yield permutation, system.relabel(permutation)

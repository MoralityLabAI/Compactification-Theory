"""Small exact certificate checks for RefutationBench.

These checks consume explicit JSON data.  They neither parse Lean source nor
verify mathematical interpretations, imported libraries, or Lean proof terms.
The graph checker verifies declaration metadata only; its result must retain
that scope when reported.

Propositional problem schema::

    {"kind": "propositional", "premises": [formula, ...], "conclusion": formula}

A formula is an atom string or one of ``{"not": formula}``,
``{"and": [formula, ...]}``, ``{"or": [formula, ...]}``, or
``{"implies": [antecedent, consequent]}``.  Empty conjunctions and
disjunctions denote True and False respectively.  Python bools are not atoms.
The countermodel certificate is ``{"kind": "countermodel", "assignment":
{"P": True, ...}}``.  Its assignment must contain exactly the problem's atoms,
and every value must be a bool, not an integer or a truthy object.

Declaration graph schema::

    {"kind": "dependency_graph", "target": "result", "declarations": {
        "A": {"kind": "axiom", "dependencies": []},
        "h": {"kind": "hypothesis", "dependencies": []},
        "result": {"kind": "theorem", "dependencies": ["A", "h"]}}}

Node kinds are ``axiom``, ``hypothesis``, ``theorem``, and ``definition``.
Edges include whichever dependencies the supplied metadata explicitly records.
The certificate names ``used_axioms``, ``local_hypotheses``, and
``unused_axioms`` as duplicate-free lists, with kind ``dependency_closure``.
Every reference and every cycle is checked, including unreachable components.
Hypotheses are reported as premises, not silently promoted to global axioms.

All public entry points reject malformed or oversized data with
``CertificateError``.  Fixed caps bound recursion and exhaustive work.  Formula
limits apply to the whole problem, including every premise and the conclusion.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any


MAX_VARIABLES = 12
MAX_FORMULA_NODES = 2048
MAX_FORMULA_DEPTH = 64
MAX_PREMISES = 32
MAX_SUBSET_PREMISES = 12
MAX_DECLARATIONS = 256
MAX_DEPENDENCY_EDGES = 2048
MAX_NAME_LENGTH = 128


class CertificateError(ValueError):
    """Malformed JSON-shaped input or a documented resource limit was exceeded."""


@dataclass(frozen=True)
class _Node:
    operation: str
    children: tuple[_Node, ...] = ()
    atom: str = ""


@dataclass
class _ParseState:
    variables: set[str]
    nodes: int = 0


def _object(value: object, fields: set[str], location: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise CertificateError(f"{location} must be an object")
    if set(value) != fields:
        raise CertificateError(f"{location} must have exactly: {', '.join(sorted(fields))}")
    return value


def _name(value: object, location: str) -> str:
    if type(value) is not str or not value.strip() or len(value) > MAX_NAME_LENGTH:
        raise CertificateError(
            f"{location} must be a nonblank string of at most {MAX_NAME_LENGTH} characters"
        )
    return value


def _parse_formula(value: object, state: _ParseState, depth: int = 1) -> _Node:
    if depth > MAX_FORMULA_DEPTH:
        raise CertificateError(f"formula depth exceeds {MAX_FORMULA_DEPTH}")
    state.nodes += 1
    if state.nodes > MAX_FORMULA_NODES:
        raise CertificateError(f"problem contains more than {MAX_FORMULA_NODES} formula nodes")
    if type(value) is str:
        atom = _name(value, "atom")
        state.variables.add(atom)
        if len(state.variables) > MAX_VARIABLES:
            raise CertificateError(f"problem contains more than {MAX_VARIABLES} atoms")
        return _Node("atom", atom=atom)
    if type(value) is not dict or len(value) != 1:
        raise CertificateError("formula must be an atom string or a single-operator object")
    operation, operand = next(iter(value.items()))
    if operation == "not":
        return _Node(operation, (_parse_formula(operand, state, depth + 1),))
    if operation not in ("and", "or", "implies"):
        raise CertificateError("supported formula operators are: not, and, or, implies")
    if type(operand) is not list:
        raise CertificateError(f"{operation} operands must be a list")
    if operation == "implies" and len(operand) != 2:
        raise CertificateError("implies requires exactly two operands")
    if len(operand) > MAX_FORMULA_NODES:
        raise CertificateError("formula operand list exceeds the node limit")
    return _Node(
        operation, tuple(_parse_formula(child, state, depth + 1) for child in operand)
    )


def _parse_problem(
    premises: object, conclusion: object
) -> tuple[tuple[_Node, ...], _Node, tuple[str, ...]]:
    if type(premises) is not list:
        raise CertificateError("premises must be a list")
    if len(premises) > MAX_PREMISES:
        raise CertificateError(f"problem contains more than {MAX_PREMISES} premises")
    state = _ParseState(set())
    parsed_premises = tuple(_parse_formula(premise, state) for premise in premises)
    parsed_conclusion = _parse_formula(conclusion, state)
    return parsed_premises, parsed_conclusion, tuple(sorted(state.variables))


def _assignment(value: object, variables: tuple[str, ...]) -> dict[str, bool]:
    if type(value) is not dict:
        raise CertificateError("assignment must be an object")
    if len(value) > MAX_VARIABLES:
        raise CertificateError("assignment exceeds the atom limit")
    for key, truth in value.items():
        _name(key, "assignment key")
        if type(truth) is not bool:
            raise CertificateError("assignment values must be bools")
    if set(value) != set(variables):
        raise CertificateError("assignment keys must match exactly the problem's atoms")
    return value


def _evaluate(node: _Node, assignment: dict[str, bool]) -> bool:
    if node.operation == "atom":
        return assignment[node.atom]
    if node.operation == "not":
        return not _evaluate(node.children[0], assignment)
    if node.operation == "and":
        return all(_evaluate(child, assignment) for child in node.children)
    if node.operation == "or":
        return any(_evaluate(child, assignment) for child in node.children)
    antecedent, consequent = node.children
    return not _evaluate(antecedent, assignment) or _evaluate(consequent, assignment)


def evaluate_formula(formula: object, assignment: object) -> bool:
    """Evaluate one validated formula under a complete, strictly Boolean assignment."""
    state = _ParseState(set())
    parsed = _parse_formula(formula, state)
    values = _assignment(assignment, tuple(sorted(state.variables)))
    return _evaluate(parsed, values)


def verify_countermodel(
    premises: object, conclusion: object, assignment: object
) -> dict[str, Any]:
    """Check all premises are true and the conclusion false for one assignment.

    A well-formed but unsuccessful assignment returns ``verified=False``.
    Invalid ASTs, extra/missing atoms, and non-Boolean values raise
    ``CertificateError`` instead of receiving a mathematical verdict.
    """
    parsed_premises, parsed_conclusion, variables = _parse_problem(premises, conclusion)
    values = _assignment(assignment, variables)
    premise_values = [_evaluate(premise, values) for premise in parsed_premises]
    conclusion_value = _evaluate(parsed_conclusion, values)
    return {
        "verified": all(premise_values) and not conclusion_value,
        "scope": "finite_propositional_logic",
        "details": {
            "variables": list(variables),
            "premise_values": premise_values,
            "conclusion_value": conclusion_value,
        },
    }


def _truth_tables(
    variables: tuple[str, ...],
) -> tuple[dict[str, int], int]:
    rows = 1 << len(variables)
    universe = (1 << rows) - 1
    tables = {}
    for position, variable in enumerate(variables):
        run = 1 << (len(variables) - position - 1)
        true_run = (1 << run) - 1
        table = 0
        for start in range(run, rows, 2 * run):
            table |= true_run << start
        tables[variable] = table
    return tables, universe


def _table(node: _Node, atoms: dict[str, int], universe: int) -> int:
    if node.operation == "atom":
        return atoms[node.atom]
    if node.operation == "not":
        return universe ^ _table(node.children[0], atoms, universe)
    if node.operation == "and":
        result = universe
        for child in node.children:
            result &= _table(child, atoms, universe)
        return result
    if node.operation == "or":
        result = 0
        for child in node.children:
            result |= _table(child, atoms, universe)
        return result
    antecedent, consequent = node.children
    return (universe ^ _table(antecedent, atoms, universe)) | _table(
        consequent, atoms, universe
    )


def find_countermodel(premises: object, conclusion: object) -> dict[str, bool] | None:
    """Return the first countermodel, or None after exhaustive finite checking.

    Atoms sort lexicographically and assignments order False before True.
    No result means entailment only in this bounded propositional problem.
    """
    parsed_premises, parsed_conclusion, variables = _parse_problem(premises, conclusion)
    tables, universe = _truth_tables(variables)
    witnesses = universe ^ _table(parsed_conclusion, tables, universe)
    for premise in parsed_premises:
        witnesses &= _table(premise, tables, universe)
    if not witnesses:
        return None
    row = (witnesses & -witnesses).bit_length() - 1
    return {
        variable: bool((row >> (len(variables) - position - 1)) & 1)
        for position, variable in enumerate(variables)
    }


def minimal_entailing_subset(premises: object, conclusion: object) -> list[int] | None:
    """Find a minimum-cardinality premise subset entailing the conclusion.

    Returned indices refer to the input order.  Ties break lexicographically;
    [] means the conclusion is a tautology, and None means even the full set
    does not entail it.  At most 12 premises are accepted for subset search.
    The result describes propositional entailment, not a proof's actual
    dependency set or the assumptions needed for the source mathematics.
    """
    parsed_premises, parsed_conclusion, variables = _parse_problem(premises, conclusion)
    if len(parsed_premises) > MAX_SUBSET_PREMISES:
        raise CertificateError(
            f"subset search accepts at most {MAX_SUBSET_PREMISES} premises"
        )
    tables, universe = _truth_tables(variables)
    false_conclusion = universe ^ _table(parsed_conclusion, tables, universe)
    premise_tables = [_table(premise, tables, universe) for premise in parsed_premises]
    for size in range(len(parsed_premises) + 1):
        for subset in combinations(range(len(parsed_premises)), size):
            possible_countermodels = false_conclusion
            for index in subset:
                possible_countermodels &= premise_tables[index]
            if not possible_countermodels:
                return list(subset)
    return None


def _name_list(value: object, location: str, maximum: int = MAX_DECLARATIONS) -> list[str]:
    if type(value) is not list or len(value) > maximum:
        raise CertificateError(f"{location} must be a list with at most {maximum} names")
    names = [_name(name, location) for name in value]
    if len(set(names)) != len(names):
        raise CertificateError(f"{location} contains duplicate names")
    return names


def dependency_closure(problem: object) -> dict[str, Any]:
    """Compute exact reachability in an explicitly supplied declaration DAG.

    Validate the entire graph first.  Unknown references and cycles in unused
    components are rejected too.  Axioms are globally declared unproved
    assumptions in this metadata; hypothesis nodes denote explicit local
    premises.  This function does not establish that the graph accurately
    describes any Lean source, proof term, or kernel dependency closure.
    """
    data = _object(problem, {"kind", "target", "declarations"}, "dependency problem")
    if data["kind"] != "dependency_graph":
        raise CertificateError("dependency problem kind must be dependency_graph")
    target = _name(data["target"], "target")
    declarations = data["declarations"]
    if type(declarations) is not dict or not 0 < len(declarations) <= MAX_DECLARATIONS:
        raise CertificateError(
            f"declarations must be an object with 1 to {MAX_DECLARATIONS} entries"
        )
    graph: dict[str, tuple[str, list[str]]] = {}
    edge_count = 0
    for name, declaration in declarations.items():
        _name(name, "declaration name")
        node = _object(declaration, {"kind", "dependencies"}, "declaration")
        if node["kind"] not in ("axiom", "hypothesis", "theorem", "definition"):
            raise CertificateError("unknown declaration kind")
        dependencies = _name_list(node["dependencies"], "dependencies")
        edge_count += len(dependencies)
        if edge_count > MAX_DEPENDENCY_EDGES:
            raise CertificateError(f"graph exceeds {MAX_DEPENDENCY_EDGES} dependency edges")
        graph[name] = (node["kind"], dependencies)
    if target not in graph:
        raise CertificateError("target is not a declared name")
    for _, dependencies in graph.values():
        if any(dependency not in graph for dependency in dependencies):
            raise CertificateError("dependency refers to an undeclared name")

    # Iterative depth-first traversal avoids Python recursion limits even when
    # a valid graph reaches the maximum permitted declaration count.
    state: dict[str, int] = {}
    for start in sorted(graph):
        if state.get(start) == 2:
            continue
        stack = [(start, False)]
        while stack:
            name, leaving = stack.pop()
            if leaving:
                state[name] = 2
                continue
            if state.get(name) == 1:
                raise CertificateError("declaration graph contains a cycle")
            if state.get(name) == 2:
                continue
            state[name] = 1
            stack.append((name, True))
            stack.extend((dependency, False) for dependency in reversed(graph[name][1]))

    reachable: set[str] = set()
    pending = [target]
    while pending:
        name = pending.pop()
        if name not in reachable:
            reachable.add(name)
            pending.extend(graph[name][1])
    used_axioms = sorted(name for name in reachable if graph[name][0] == "axiom")
    return {
        "scope": "declaration_metadata_only",
        "target": target,
        "reachable_declarations": sorted(reachable),
        "used_axioms": used_axioms,
        "local_hypotheses": sorted(
            name for name in reachable if graph[name][0] == "hypothesis"
        ),
        "unused_axioms": sorted(
            name for name in graph if graph[name][0] == "axiom" and name not in reachable
        ),
    }


def verify_certificate(problem: object, certificate: object) -> dict[str, Any]:
    """Verify a countermodel or a declaration-closure certificate.

    Returns ``{"verified": bool, "scope": str, "details": object}``.
    Certificate name lists are compared as sets after rejecting duplicates and
    unknown names.  Extra fields are rejected for both supported schemas.
    """
    if type(problem) is not dict:
        raise CertificateError("problem must be an object")
    if problem.get("kind") == "propositional":
        data = _object(problem, {"kind", "premises", "conclusion"}, "propositional problem")
        proof = _object(certificate, {"kind", "assignment"}, "countermodel certificate")
        if proof["kind"] != "countermodel":
            raise CertificateError("propositional certificate kind must be countermodel")
        return verify_countermodel(data["premises"], data["conclusion"], proof["assignment"])
    if problem.get("kind") == "dependency_graph":
        closure = dependency_closure(problem)
        fields = {"kind", "used_axioms", "local_hypotheses", "unused_axioms"}
        proof = _object(certificate, fields, "dependency certificate")
        if proof["kind"] != "dependency_closure":
            raise CertificateError("dependency certificate kind must be dependency_closure")
        matches = {}
        declared_names = set(problem["declarations"])
        for field in sorted(fields - {"kind"}):
            claimed = _name_list(proof[field], field)
            if not set(claimed) <= declared_names:
                raise CertificateError("certificate refers to an undeclared name")
            matches[field] = set(claimed) == set(closure[field])
        return {
            "verified": all(matches.values()),
            "scope": "declaration_metadata_only",
            "details": {"computed": closure, "matching_fields": matches},
        }
    raise CertificateError("problem kind must be propositional or dependency_graph")

"""Exhaustive, deterministic NCG-01 verification with an independent oracle."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from fractions import Fraction
from hashlib import sha256
from itertools import combinations, product
import json
from pathlib import Path
from time import perf_counter

from ncg01 import (Graph, amplitude, amplitude_table, convolve,
                   glue_shared_boundary, predicted_dimension, solve)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def oracle_charge(graph: Graph, p: int, flow: tuple[int, ...]) -> tuple[int, ...]:
    """Independent endpoint accumulation; never reads the incidence matrix."""
    charges = [0] * graph.vertices
    for (tail, head), x in zip(graph.edges, flow):
        charges[tail] -= x
        charges[head] += x
    return tuple(x % p for x in charges)


def oracle_sectors(graph: Graph, p: int) -> dict[tuple[int, ...], set[tuple[int, ...]]]:
    result: dict[tuple[int, ...], set[tuple[int, ...]]] = defaultdict(set)
    for flow in product(range(p), repeat=len(graph.edges)):
        result[oracle_charge(graph, p, flow)].add(flow)
    return dict(result)


def oracle_weights(graph: Graph, p: int, weights: list[list[int]]) -> dict[tuple[int, ...], int]:
    result: dict[tuple[int, ...], int] = defaultdict(int)
    for flow in product(range(p), repeat=len(graph.edges)):
        value = 1
        for j, x in enumerate(flow):
            value *= weights[j][x]
        result[oracle_charge(graph, p, flow)] += value
    return {k: v for k, v in result.items() if v}


def all_oriented_simple_graphs(n: int):
    pairs = tuple(combinations(range(n), 2))
    for choices in product(range(3), repeat=len(pairs)):
        edges = []
        for (u, v), choice in zip(pairs, choices):
            if choice == 1:
                edges.append((u, v))
            elif choice == 2:
                edges.append((v, u))
        yield Graph(n, tuple(edges))


def fixtures():
    for p, largest in ((2, 4), (3, 3)):
        for n in range(largest + 1):
            for graph in all_oriented_simple_graphs(n):
                yield "all_orientations", p, graph
    special = [
        Graph(4, ((0, 1), (1, 2), (2, 3), (3, 0), (0, 2))),
        Graph(4, ((1, 0), (2, 1), (2, 3), (0, 3), (2, 0))),
        Graph(5, ((0, 1), (1, 2), (2, 0), (3, 4))),
        Graph(4, ((0, 1), (0, 1), (1, 0), (2, 2))),
        Graph(3, ((0, 0), (0, 1), (1, 2), (2, 0))),
        Graph(4, ((0, 0), (1, 1), (2, 2), (3, 3))),
    ]
    for p in (2, 3):
        for graph in special:
            yield "loops_parallel_disconnected_cycles", p, graph
    for graph in special[:2] + [special[3]]:
        yield "prime_five", 5, graph


def verify_constraints() -> dict:
    counts: Counter = Counter()
    by_prime: dict[int, Counter] = defaultdict(Counter)
    digest = sha256()
    for family, p, graph in fixtures():
        counts["graphs"] += 1
        counts[family] += 1
        by_prime[p]["graphs"] += 1
        reference = oracle_sectors(graph, p)
        counts["enumerated_ambient_basis_states"] += p ** len(graph.edges)
        by_prime[p]["enumerated_ambient_basis_states"] += p ** len(graph.edges)
        observed_rank = None
        for charges in product(range(p), repeat=graph.vertices):
            code = solve(graph, p, charges)
            expected = reference.get(charges, set())
            actual = list(code.flows())
            require(code.rank == graph.vertices - len(graph.components()), "incidence rank theorem")
            require(code.dimension == predicted_dimension(graph, p, charges), "dimension formula")
            require(code.dimension == len(expected), "independent state count")
            require(len(actual) == len(set(actual)), "decoder injectivity")
            require(set(actual) == expected, "decoder surjectivity")
            counts["charge_sectors"] += 1
            by_prime[p]["charge_sectors"] += 1
            counts["compatible_sectors" if expected else "incompatible_sectors"] += 1
            if observed_rank is None:
                observed_rank = code.rank
            require(observed_rank == code.rank, "rank must be independent of charges")
            for flow in sorted(expected):
                coordinates = code.encode(flow)
                require(code.decode(coordinates) == flow, "encode-decode round trip")
                require(oracle_charge(graph, p, flow) == charges, "independent constraint oracle")
                counts["lossless_round_trips"] += 1
            digest.update(json.dumps([p, graph.vertices, graph.edges, charges,
                                      code.rank, code.dimension, code.free],
                                     separators=(",", ":")).encode())
        require(sum(len(x) for x in reference.values()) == p ** len(graph.edges),
                "sectors partition ambient state labels")
    return {"counts": dict(sorted(counts.items())),
            "by_prime": {str(p): dict(sorted(c.items())) for p, c in sorted(by_prime.items())},
            "case_digest_sha256": digest.hexdigest()}


def verify_gluing() -> dict:
    counts: Counter = Counter()
    digest = sha256()
    for p in (2, 3):
        for graph in (
            Graph(3, ((0, 1), (1, 2), (2, 0))),
            Graph(4, ((0, 1), (1, 2), (2, 3), (3, 0), (0, 2))),
            Graph(4, ((0, 1), (1, 0), (2, 3), (2, 2))),
        ):
            for pattern in ("counts", "signed"):
                weights = [[1] * p for _ in graph.edges] if pattern == "counts" else [
                    [((e + 2) * (x + 1) + x * x) % 7 - 3 for x in range(p)]
                    for e in range(len(graph.edges))]
                global_table = oracle_weights(graph, p, weights)
                m = len(graph.edges)
                # Every ordered nonempty bipartition of the edge set.
                for mask in range(1, (1 << m) - 1):
                    left_ids = [e for e in range(m) if mask & (1 << e)]
                    right_ids = [e for e in range(m) if not mask & (1 << e)]
                    left = Graph(graph.vertices, tuple(graph.edges[e] for e in left_ids))
                    right = Graph(graph.vertices, tuple(graph.edges[e] for e in right_ids))
                    lw = [weights[e] for e in left_ids]
                    rw = [weights[e] for e in right_ids]
                    lt = amplitude_table(left, p, lw)
                    rt = amplitude_table(right, p, rw)
                    require(convolve(lt, rt, p) == global_table, "full charge convolution")
                    boundary = sorted(set(v for edge in left.edges for v in edge)
                                      & set(v for edge in right.edges for v in edge))
                    for charges in product(range(p), repeat=graph.vertices):
                        value = glue_shared_boundary(left, right, p, charges, boundary, lw, rw)
                        require(value == global_table.get(charges, 0), "shared boundary gluing")
                        digest.update(json.dumps([p, graph.edges, pattern, mask, charges, value],
                                                 separators=(",", ":")).encode())
                        counts["boundary_charge_contractions"] += 1
                    counts["edge_bipartitions"] += 1
                # Every edge is assigned to exactly one of three named regions.
                # Allowing an empty region checks the identity as well.
                for labels in product(range(3), repeat=m):
                    groups = [[e for e in range(m) if labels[e] == j] for j in range(3)]
                    tables = [amplitude_table(Graph(graph.vertices, tuple(graph.edges[e] for e in g)),
                                              p, [weights[e] for e in g]) for g in groups]
                    ab_c = convolve(convolve(tables[0], tables[1], p), tables[2], p)
                    a_bc = convolve(tables[0], convolve(tables[1], tables[2], p), p)
                    require(ab_c == a_bc == global_table, "gluing associativity")
                    counts["three_region_associativity_cases"] += 1
    return {"counts": dict(sorted(counts.items())), "case_digest_sha256": digest.hexdigest()}


def verify_orientation() -> dict:
    graph = Graph(4, ((0, 1), (1, 2), (2, 3), (3, 0), (0, 2)))
    reverse = Graph(graph.vertices, tuple((v, u) for u, v in graph.edges))
    permutation = (4, 2, 0, 3, 1)
    permuted = Graph(graph.vertices, tuple(graph.edges[i] for i in permutation))
    count = 0
    for p in (3, 5):
        for flow in product(range(p), repeat=len(graph.edges)):
            charge = oracle_charge(graph, p, flow)
            require(reverse.divergence(tuple(-x % p for x in flow), p) == charge,
                    "orientation covariance")
            require(permuted.divergence(tuple(flow[i] for i in permutation), p) == charge,
                    "edge-order covariance")
            count += 1
    return {"basis_states_checked_for_both_transformations": count}


def must_raise(fn):
    try:
        fn()
    except ValueError:
        return
    raise RuntimeError("expected a ValueError")


def verify_invalid_inputs() -> dict:
    graph = Graph(3, ((0, 1), (1, 2), (2, 0)))
    code = solve(graph, 3, (0, 0, 0))
    invalid = [
        lambda: Graph(-1, ()),
        lambda: Graph(2, ((0, 2),)),
        lambda: solve(graph, 4, (0, 0, 0)),
        lambda: solve(graph, 3, (0, 0)),
        lambda: solve(graph, 3, (0, -1, 1)),
        lambda: solve(graph, 3, (1, 0, 0)).decode((0,)),
        lambda: code.decode(()),
        lambda: code.decode((3,)),
        lambda: code.encode((1, 0, 0)),
        lambda: code.encode((0, 0)),
        lambda: code.encode((3, 3, 3)),
        lambda: amplitude(graph, 3, (0, 0, 0), [[1, 1, 1]]),
        lambda: glue_shared_boundary(Graph(3, ((0, 1),)), Graph(3, ((1, 2),)),
                                      3, (0, 0, 0), [], [[1, 1, 1]], [[1, 1, 1]]),
    ]
    for case in invalid:
        must_raise(case)
    return {"rejected_invalid_inputs": len(invalid)}


def verify_charge_multiplicities() -> dict:
    a = (2, 1, 3)
    b = (1, 4, 2)
    labels_a = [(q, j) for q, d in enumerate(a) for j in range(d)]
    labels_b = [(q, j) for q, d in enumerate(b) for j in range(d)]
    admitted = [(la, lb) for la in labels_a for lb in labels_b if (la[0] + lb[0]) % 3 == 0]
    formula = sum(a[q] * b[-q % 3] for q in range(3))
    require(len(admitted) == formula == 16, "charge-sector multiplicity formula")
    return {"p": 3, "multiplicities_A": a, "multiplicities_B": b,
            "ambient_dimension": len(labels_a) * len(labels_b),
            "physical_dimension": len(admitted),
            "formula": "sum_q d_A(q) d_B(-q)"}


def verify_normalization() -> dict:
    a = [1, 3]
    b = [2, 1]
    total = sum(x * y for x, y in zip(a, b))
    normalized_local = sum(Fraction(x, sum(a)) * Fraction(y, sum(b)) for x, y in zip(a, b))
    require(total == 5 and normalized_local == Fraction(5, 12), "discarded local scales")
    distribution = [Fraction(x * y, total) for x, y in zip(a, b)]
    sector_unit = [Fraction(1, 2), Fraction(1, 2)]
    require(distribution != sector_unit, "per-sector normalization changes charge weights")
    return {"positive_weight_example_A": a, "positive_weight_example_B": b,
            "unnormalized_contraction": total,
            "contraction_after_discarding_local_scales": str(normalized_local),
            "conditioned_charge_distribution": [str(x) for x in distribution],
            "distribution_after_setting_each_sector_weight_to_one": [str(x) for x in sector_unit],
            "interpretation": "These are positive measure weights. Squaring amplitudes is a separate choice."}


def verify_infinite_rank_counterexample() -> dict:
    rows = []
    for cutoff in (0, 1, 2, 4, 8, 16, 32, 64, 128):
        pairs = [(n, -n) for n in range(-cutoff, cutoff + 1)]
        require(all(n + m == 0 for n, m in pairs), "integer charge matching")
        dimension = len(pairs)
        require(dimension == 2 * cutoff + 1, "integer cutoff dimension")
        trace = sum((Fraction(1, 2 ** abs(n)) for n, _ in pairs), Fraction())
        tail = Fraction(2, 2 ** cutoff)
        require(trace + tail == 3, "finite geometric Gibbs trace")
        rows.append({"cutoff_K": cutoff, "dimension": dimension,
                     "gibbs_trace_partial": str(trace), "gibbs_trace_tail": str(tail)})
    for budget in range(17):
        states = [(n, -n) for n in range(-20, 21) if 2 * abs(n) <= budget]
        require(len(states) == 2 * (budget // 2) + 1, "coercive budget finite projector")
    return {
        "model": "l2(Z) tensor l2(Z), Q_A|n>=n|n>, Q_B|m>=m|m>",
        "physical_basis": "{|n,-n>: n in Z}",
        "full_physical_rank": "countably infinite",
        "cutoff_sequence": rows,
        "gibbs_weights_on_physical_basis": "2^(-abs(n))",
        "gibbs_trace_limit": "3",
        "normalized_density_rank": "countably infinite",
        "density_entropy_nats": "log(3)+(4/3)*log(2)",
        "coercive_operator": "A|n,m>=(abs(n)+abs(m))|n,m>",
        "budget_projector_rank": "2*floor(A_max/2)+1",
        "budget_cases_checked": 17,
    }


def examples() -> list[dict]:
    cases = [
        ("triangle", Graph(3, ((0, 1), (1, 2), (2, 0))), 3, (0, 0, 0)),
        ("square_with_diagonal", Graph(4, ((0, 1), (1, 2), (2, 3), (3, 0), (0, 2))), 3, (0, 0, 0, 0)),
        ("charged_tree", Graph(4, ((0, 1), (1, 2), (1, 3))), 3, (1, 1, 1, 0)),
        ("disconnected_incompatible", Graph(4, ((0, 1), (2, 3))), 3, (1, 0, 2, 0)),
    ]
    result = []
    for name, graph, p, b in cases:
        code = solve(graph, p, b)
        result.append({"name": name, "p": p, "vertices": graph.vertices, "edges": graph.edges,
                       "charges": b, "components": graph.components(), "rank": code.rank,
                       "ambient_dimension": p ** len(graph.edges), "physical_dimension": code.dimension,
                       "ambient_field_symbols": len(graph.edges),
                       "free_field_symbols": len(code.free) if code.compatible else None,
                       "free_edge_indices": code.free,
                       "physical_basis": list(code.flows())})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("receipt.json"))
    parser.add_argument("--check", type=Path, help="require exact equality to a previous receipt")
    args = parser.parse_args()
    started = perf_counter()
    source_dir = Path(__file__).resolve().parent
    receipt = {
        "experiment": "NCG-01", "schema_version": 1,
        "scope": "Finite abelian constraint and gluing toy. No validation of quantum gravity.",
        "constraints": verify_constraints(),
        "gluing": verify_gluing(),
        "orientation": verify_orientation(),
        "invalid_inputs": verify_invalid_inputs(),
        "charge_multiplicities": verify_charge_multiplicities(),
        "normalization": verify_normalization(),
        "counterexample": verify_infinite_rank_counterexample(),
        "examples": examples(),
        "source_sha256": {name: sha256((source_dir / name).read_bytes()).hexdigest()
                          for name in ("ncg01.py", "verify_ncg01.py")},
        "status": "PASS",
    }
    # Canonical JSON values also make tuple/list representation irrelevant when
    # replaying a receipt loaded from disk.
    receipt = json.loads(json.dumps(receipt))
    if args.check:
        require(receipt == json.loads(args.check.read_text()), "receipt replay differs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    elapsed = perf_counter() - started
    args.output.with_suffix(".runtime.json").write_text(
        json.dumps({"elapsed_seconds": elapsed, "timing_is_noncanonical": True}, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "receipt": str(args.output),
                      "constraints": receipt["constraints"]["counts"],
                      "gluing": receipt["gluing"]["counts"],
                      "elapsed_seconds": elapsed}, indent=2))


if __name__ == "__main__":
    main()

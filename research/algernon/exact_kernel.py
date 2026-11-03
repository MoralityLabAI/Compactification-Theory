"""CPU-only exact lattice-reachability microkernel for Algernon Phase B.

This is a reference bitset processor, NOT an in-weight Qwenputer/RV32 execution.
It has a separately implemented graph-search oracle and a seeded, disjoint-family
workload, so a future compiled neural circuit can be tested against the same cases.

Run from the repository root:
python -m research.algernon.exact_kernel --output runs/algernon/cpu_reference.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path


def canonical_bytes(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256(obj):
    return hashlib.sha256(canonical_bytes(obj)).hexdigest()


@dataclass(frozen=True)
class Task:
    n: int
    source: int
    goal: int
    edges: tuple[tuple[int, int], ...]

    def __post_init__(self):
        if not 2 <= self.n <= 16:
            raise ValueError("node count outside [2, 16]")
        if not (0 <= self.source < self.n and 0 <= self.goal < self.n):
            raise ValueError("source or goal out of bounds")
        if any(len(e) != 2 or min(e) < 0 or max(e) >= self.n for e in self.edges):
            raise ValueError("invalid edge")

    def json(self):
        return {"n": self.n, "source": self.source, "goal": self.goal,
                "edges": [list(edge) for edge in self.edges]}


def compile_task(task):
    """Compile edges into per-node outgoing bitmasks."""
    rows = [0] * task.n
    for src, dst in task.edges:
        rows[src] |= 1 << dst
    return tuple(rows)


def run_compiled(task, rows):
    if len(rows) != task.n or any(not isinstance(row, int) or row < 0 or row >= (1 << task.n)
                                  for row in rows):
        raise ValueError("invalid compiled rows")
    visited = 1 << task.source
    history = []
    operations = 0
    for iteration in range(1, task.n + 1):
        expanded = visited
        for vertex in range(task.n):
            operations += 1
            if visited & (1 << vertex):
                expanded |= rows[vertex]
        history.append([iteration, visited, expanded])
        if expanded == visited:
            break
        visited = expanded
    return {
        "reachable": bool(visited & (1 << task.goal)),
        "visited_mask": visited,
        "iterations": len(history),
        "bit_operations": operations,
        "trace_sha256": sha256(history),
    }


def oracle_bfs(task):
    """Independent queue-based graph search, using edges rather than bitmasks."""
    adjacency = {i: [] for i in range(task.n)}
    for u, v in task.edges:
        adjacency[u].append(v)
    visited = {task.source}
    q = deque([task.source])
    while q:
        for nxt in adjacency[q.popleft()]:
            if nxt not in visited:
                visited.add(nxt)
                q.append(nxt)
    return bool(task.goal in visited), sum(1 << v for v in visited)


SPLIT_FAMILIES = {
    "train": ("path", "branch"),
    "calibration": ("cycle",),
    "test": ("random", "disconnected"),
}


def make_case(family, seed, case_index):
    rng = random.Random(f"{family}:{seed}:{case_index}")
    n = rng.randrange(2, 13)
    if family == "path":
        edges = [(i, i + 1) for i in range(n - 1)]
    elif family == "branch":
        edges = [(0, i) for i in range(1, n)]
    elif family == "cycle":
        edges = [(i, (i + 1) % n) for i in range(n)]
    elif family == "random":
        edges = [(i, j) for i in range(n) for j in range(n)
                 if i != j and rng.random() < 0.19]
    elif family == "disconnected":
        m = max(1, n // 2)
        edges = [(i, j) for i in range(n) for j in range(n)
                 if i != j and (i < m) == (j < m) and rng.random() < 0.3]
    else:
        raise ValueError(f"unknown family: {family}")
    return Task(n, rng.randrange(n), rng.randrange(n), tuple(sorted(set(edges))))


def run_suite(seed=20261103, cases_per_family=50):
    if cases_per_family <= 0:
        raise ValueError("cases_per_family must be positive")
    out = []
    for split, families in SPLIT_FAMILIES.items():
        for family in families:
            for i in range(cases_per_family):
                task = make_case(family, seed, i)
                got = run_compiled(task, compile_task(task))
                expected, expected_mask = oracle_bfs(task)
                if (got["reachable"], got["visited_mask"]) != (expected, expected_mask):
                    raise AssertionError(f"reference mismatch: {split}/{family}/{i}")
                out.append({
                    "id": f"{split}:{family}:{seed}:{i}",
                    "split": split, "family": family, "task": task.json(),
                    "task_sha256": sha256(task.json()),
                    "reference_reachable": expected, **got,
                })
    split_hashes = {
        s: sha256([r["task_sha256"] for r in out if r["split"] == s])
        for s in SPLIT_FAMILIES
    }
    return {
        "schema_version": "0.1.0",
        "execution_class": "cpu_python_reference_only",
        "description": "monotone bitset closure compared with independent BFS",
        "seed": seed,
        "cases_per_family": cases_per_family,
        "n": len(out),
        "split_sha256": split_hashes,
        "family_assignment": {s: list(v) for s, v in SPLIT_FAMILIES.items()},
        "reference_mismatches": 0,
        "records": out,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=20261103)
    p.add_argument("--cases-per-family", type=int, default=50)
    p.add_argument("--output", type=Path, default=Path("runs/algernon/cpu_reference.json"))
    a = p.parse_args()
    r = run_suite(a.seed, a.cases_per_family)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(r, sort_keys=True, indent=2) + "\n")
    print(f"CPU-only: {r['n']} checks, {r['reference_mismatches']} mismatches, {a.output}")


if __name__ == "__main__":
    main()

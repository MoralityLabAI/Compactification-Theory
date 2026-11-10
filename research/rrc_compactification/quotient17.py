"""RRC-17: probe-conditioned semantic quotient for finite formal-series graphs.

The quotient is built offline from public interpreter operations and random field
probes, without demonstration targets or model labels.  Membership of one
candidate is always tested by *exact integer* degree-12 verification.  Probe
equality is a probabilistic identification heuristic, not a formal proof of
universal algebraic equivalence; see RRC17_THEORY.md for the bound.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import random

try:
    from . import topology16 as base, graph11 as graphs
except ImportError:
    import topology16 as base
    import graph11 as graphs

SEED = 2026111017
PRIME = 2**31 - 1
PROBES = 4
PREFIX = base.PREFIX
HORIZON = base.HORIZON
BUDGET = 12
TOPOLOGIES = base.TOPOLOGIES
COUNTS = {
    'hadamard_uniform': 96,
    'hadamard_novel': 48,
    'hadamard_product': 48,
    'fork_uniform': 48,
    'cascade_uniform': 48,
    'additive_uniform': 48,
    'hadamard_negative': 16,
    'hadamard_collision': 16,
}
SPLITS = {
    'hadamard_uniform': 'hadamard', 'hadamard_novel': 'hadamard',
    'hadamard_product': 'hadamard', 'fork_uniform': 'fork',
    'cascade_uniform': 'cascade', 'additive_uniform': 'additive',
    'hadamard_negative': 'hadamard', 'hadamard_collision': 'hadamard',
}


def random_probes(*, seed: int = SEED, count: int = PROBES) -> tuple[tuple[int, ...], ...]:
    """Independent F_p formal series; no target-program or roster access."""
    if count < 1:
        raise ValueError('need at least one probe')
    rng = random.Random(seed + 19011)
    return tuple((1,) + tuple(rng.randrange(PRIME) for _ in range(PREFIX))
                 for _ in range(count))


@dataclass(frozen=True)
class Quotient:
    topology: str
    classes: tuple[tuple[tuple[str, str, str], ...], ...]
    probes_hash: str
    total_probe_graph_executions: int
    total_probe_integer_work: int

    @property
    def representatives(self):
        """Cheapest representative of each class; tie-break by DSL order."""
        return tuple(min(group, key=lambda p: (base.graph_cost(p, self.topology),
                                                graphs.GRAPHS.index(p)))
                     for group in self.classes)

    def sorted_classes(self, *, mode: str = 'mass'):
        if mode == 'mass':
            return tuple(sorted(self.classes, key=lambda group:
                                (-len(group), base.graph_cost(min(group,
                                    key=lambda p: base.graph_cost(p, self.topology)),
                                    self.topology), graphs.GRAPHS.index(group[0]))))
        if mode == 'mass_per_work':
            return tuple(sorted(self.classes, key=lambda group: (
                -len(group) / (2 * min(base.graph_cost(p, self.topology) for p in group)),
                -len(group), graphs.GRAPHS.index(group[0]))))
        if mode == 'canonical':
            return tuple(sorted(self.classes, key=lambda group: graphs.GRAPHS.index(group[0])))
        raise ValueError('invalid quotient ordering')


def build_quotient(topology: str, *, seed: int = SEED,
                   count: int = PROBES) -> Quotient:
    if topology not in TOPOLOGIES:
        raise ValueError('unknown topology')
    probes = random_probes(seed=seed, count=count)
    groups = {}
    work = 0
    for graph in graphs.GRAPHS:
        signature = tuple(tuple(x % PRIME for x in base.execute(f, graph, topology))
                          for f in probes)
        groups.setdefault(signature, []).append(graph)
        work += count * base.graph_cost(graph, topology)
    classes = tuple(tuple(group) for group in groups.values())
    return Quotient(topology, classes,
                    sha256(json.dumps(probes, separators=(',', ':')).encode()).hexdigest(),
                    count * len(graphs.GRAPHS), work)


def choose_representatives(quotient: Quotient, mode: str) -> tuple[tuple[str, str, str], ...]:
    return tuple(min(group, key=lambda gr: (base.graph_cost(gr, quotient.topology),
                        graphs.GRAPHS.index(gr))) for group in quotient.sorted_classes(mode=mode))


def roster():
    """Independent seeded test roster. Positive graphs sampled uniformly over 64 syntax.

    Novel Hadamard eligibility uses only degree-12 visible demos (no test future).
    Negatives are selected after finite 64-graph public-prefix rejection.
    """
    groups = {name: [] for name in COUNTS}
    seen = set()
    for split in ('hadamard_uniform', 'hadamard_novel', 'hadamard_product',
                  'fork_uniform', 'cascade_uniform', 'additive_uniform',
                  'hadamard_negative'):
        rng = random.Random(SEED + 1000 + list(COUNTS).index(split) * 211)
        attempts = 0
        while len(groups[split]) < COUNTS[split]:
            attempts += 1
            if attempts > 100000:
                raise RuntimeError('roster exhaustion ' + split)
            topology = SPLITS[split]
            gr = rng.choice(graphs.GRAPHS)
            case = base._case(rng, gr, topology, split,
                              f'{split}-{len(groups[split]):04d}',
                              product=split.endswith('product'),
                              mutation=split.endswith('negative'))
            if case.visible in seen:
                continue
            if split == 'hadamard_novel' and any(base.matching_graphs(case, t)
                                                  for t in base.TRAIN_TOPOLOGIES):
                continue
            if split == 'hadamard_negative' and base.matching_graphs(case):
                continue
            seen.add(case.visible)
            groups[split].append(case)
    rng = random.Random(SEED + 24680)
    for index in range(COUNTS['hadamard_collision'] // 2):
        for _ in range(10000):
            gr = rng.choice(graphs.GRAPHS)
            a = base._case(rng, gr, 'hadamard', 'hadamard_collision',
                           f'hadamard_collision-{2*index:04d}',
                           pair=f'pair-{index:02d}')
            if a.visible not in seen:
                break
        else:
            raise RuntimeError('collision case generation exhausted')
        seen.add(a.visible)
        mutated = list(a.future)
        mutated[PREFIX + 3] += 1 if index % 2 else -1
        b = base.Case(f'hadamard_collision-{2*index+1:04d}',
                      'hadamard_collision', 'hadamard', a.demos, a.query,
                      tuple(mutated), None, a.pair)
        groups['hadamard_collision'].extend((a, b))
    return groups


def search(case, order, *, budget=BUDGET):
    """One charged exact *two-demo* execution for each new graph; no free preview."""
    if not 1 <= budget <= len(graphs.GRAPHS):
        raise ValueError('budget out of range')
    trace = []
    work = 0
    accepted_graph = None
    for graph in order:
        if len(trace) >= budget:
            break
        if graph in trace:
            raise AssertionError('duplicate paid query')
        signal = base.feedback(case, graph)
        trace.append(graph)
        work += 2 * base.graph_cost(graph, case.topology)
        if signal['accepted']:
            accepted_graph = graph
            break
    return {'graph': accepted_graph, 'trace': tuple(trace),
            'queries': len(trace), 'primitive_calls': 8 * len(trace),
            'integer_loop_proxy': work, 'model_macs': 0}


def hash_roster(cases):
    payload = [(c.identifier, c.family, c.topology, c.demos, c.query,
                c.future, c.hidden_graph, c.pair) for c in cases]
    return sha256(json.dumps(payload, separators=(',', ':')).encode()).hexdigest()


def verify_probe_stability(quotient: Quotient, *, seeds=(SEED+1,SEED+2)):
    """Independent public-data stress: every inferred alias passes new field probes."""
    n = 0
    for seed in seeds:
        probes = random_probes(seed=seed)
        for group in quotient.classes:
            representative = group[0]
            for graph in group[1:]:
                for f in probes:
                    assert base.execute(f, representative, quotient.topology) == base.execute(f, graph, quotient.topology)
                    n += 1
    return n

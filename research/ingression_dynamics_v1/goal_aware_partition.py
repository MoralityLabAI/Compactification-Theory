"""QO-01: exact goal-aware state minimization and finite stochastic Pareto search.

Dependency-free reference. All reported costs are exact Fractions. This is
model-based *offline* enumeration, not a trained neural quotient.
"""

from fractions import Fraction as Q
from itertools import product


def state_signatures(states, goals):
    assert states and all(set(g) == set(states) for g in goals)
    return {s: tuple(bool(goal[s]) for goal in goals) for s in states}


def _canonical(states, signatures):
    identifiers = {}
    return {s: identifiers.setdefault(signatures[s], len(identifiers)) for s in states}


def minimum_goal_partition(states, transitions, goals):
    """Coarsest deterministic action-stable goal-visible quotient (Moore refinement)."""
    states = tuple(states)
    assert len(set(states)) == len(states) and states
    assert transitions and all(set(f) == set(states) and
                               set(f.values()) <= set(states) for f in transitions.values())
    labels = _canonical(states, state_signatures(states, goals))
    steps = 0
    while True:
        new = _canonical(states, {s: (labels[s], tuple(labels[f[s]] for f in transitions.values()))
                                  for s in states})
        if new == labels:
            return labels, steps
        labels, steps = new, steps + 1
        assert steps < len(states)


def all_goal_words_signature(states, transitions, goals, max_length):
    """Independent oracle by explicit enumeration of action-word output traces."""
    states = tuple(states)
    signs = state_signatures(states, goals)
    actions = tuple(transitions)
    all_words = [()]
    for length in range(1, max_length + 1):
        all_words.extend(product(actions, repeat=length))
    answers = {}
    for s in states:
        signature = []
        for word in all_words:
            end = s
            for action in word:
                end = transitions[action][end]
            signature.append(signs[end])
        answers[s] = tuple(signature)
    return _canonical(states, answers)


def partitions(states):
    """Enumerate each finite set partition once as a restricted-growth string."""
    states = tuple(states)
    assert states
    labels = [0] * len(states)
    def extend(index, max_label):
        if index == len(states):
            yield {s: labels[i] for i, s in enumerate(states)}
            return
        for label in range(max_label + 2):
            labels[index] = label
            yield from extend(index + 1, max(max_label, label))
    yield from extend(1, 0)


def valid_deterministic_quotient(states, transitions, goals, partition):
    signs = state_signatures(states, goals)
    for s in states:
        for t in states:
            if partition[s] != partition[t]:
                continue
            if signs[s] != signs[t]:
                return False
            if any(partition[f[s]] != partition[f[t]] for f in transitions.values()):
                return False
    return True


def partition_bits(partition):
    return (len(set(partition.values())) - 1).bit_length()


def _project(row, partition):
    out = {}
    for s, w in row.items():
        z = partition[s]
        out[z] = out.get(z, Q(0)) + w
    return out


def _tv(p, q):
    return sum((abs(p.get(z, 0) - q.get(z, 0)) for z in set(p) | set(q)), Q(0)) / 2


def goal_worst_error(states, partition, goals):
    # A binary goal with probabilistic per-class readout has radius 0 or 1/2.
    for goal in goals:
        fibre_signatures = {}
        for s in states:
            fibre_signatures.setdefault(partition[s], set()).add(bool(goal[s]))
        if any(len(signs) > 1 for signs in fibre_signatures.values()):
            return Q(1, 2)
    return Q(0)


def goal_mean_bayes_error(states, partition, goals, mass):
    """Uniformly weighted sum of optimal per-goal 0/1 classification errors."""
    assert sum(mass.values(), Q(0)) == 1
    assert set(mass) == set(states) and all(w >= 0 for w in mass.values())
    out = Q(0)
    for g in goals:
        by_class = {}
        for s in states:
            entry = by_class.setdefault(partition[s], [Q(0), Q(0)])
            entry[int(bool(g[s]))] += mass[s]
        out += sum((min(v) for v in by_class.values()), Q(0))
    return out


def _min_tv_center(laws, alphabet):
    # Each singleton or pair admits a closed-form optimum in any finite simplex.
    if len(laws) == 1:
        return Q(0), laws[0].copy()
    if len(laws) == 2:
        p, q = laws
        midpoint = {z: (p.get(z, Q(0)) + q.get(z, Q(0))) / 2
                    for z in alphabet}
        return _tv(p, q) / 2, midpoint
    # For >=3 laws use the independently developed exact-rational LP vertex oracle.
    # This reference is bounded to <=3 macro labels and >=3 rows.
    if len(alphabet) > 3:
        raise ValueError('exact LP vertex oracle accepts <=3 macro labels for >=3 laws')
    from research.ingression_dynamics_v1.minimax_autonomy import tv_minimax_center
    return tv_minimax_center(laws, alphabet)


def optimum_tv_for_partition(states, partition, kernels):
    """Minimax one-step TV error and a constructive optimal macro law for tiny systems."""
    alphabet = tuple(sorted(set(partition.values())))
    groups = {z: tuple(s for s in states if partition[s] == z) for z in alphabet}
    macro = {}
    worst = Q(0)
    for a, kernel in kernels.items():
        assert set(kernel) == set(states)
        for s in states:
            assert set(kernel[s]) <= set(states)
            assert all(p >= 0 for p in kernel[s].values())
            assert sum(kernel[s].values(), Q(0)) == 1
        macro[a] = {}
        for z, fibre in groups.items():
            laws = [_project(kernel[s], partition) for s in fibre]
            radius, center = _min_tv_center(laws, alphabet)
            macro[a][z] = center
            worst = max(worst, radius)
            assert all(_tv(law, center) <= radius for law in laws)
    return worst, macro


def pareto_frontier(states, goals, kernels, mass=None):
    """Non-dominated candidates on (bit cost, uniform TV, mean goal error).

    Same-coordinate solutions are coalesced to a lexicographically canonical
    representative. Required macro-state cardinality is recorded separately.
    """
    states = tuple(states)
    if mass is None:
        mass = {s: Q(1, len(states)) for s in states}
    candidates = {}
    for p in partitions(states):
        eps, macro = optimum_tv_for_partition(states, p, kernels)
        record = {
            'partition': tuple(p[s] for s in states),
            'classes': len(set(p.values())),
            'bits': partition_bits(p),
            'tv_error': eps,
            'goal_worst': goal_worst_error(states, p, goals),
            'goal_mean': goal_mean_bayes_error(states, p, goals, mass),
            'macro': macro,
        }
        key = (record['bits'], record['tv_error'], record['goal_mean'])
        old = candidates.get(key)
        if old is None or record['partition'] < old['partition']:
            candidates[key] = record
    entries = list(candidates.values())
    def dominates(x, y):
        a = (x['bits'], x['tv_error'], x['goal_mean'])
        b = (y['bits'], y['tv_error'], y['goal_mean'])
        return all(i <= j for i, j in zip(a, b)) and any(i < j for i, j in zip(a, b))
    kept = [x for x in entries if not any(dominates(y, x) for y in entries)]
    kept.sort(key=lambda x: (x['bits'], x['tv_error'], x['goal_mean'], x['partition']))
    return kept


def optimal_rate(states, goals, kernels, bits, max_goal_mean, mass=None):
    """Exact finite rate-distortion objective with constraints on bits and goal error."""
    states = tuple(states)
    if mass is None:
        mass = {s: Q(1, len(states)) for s in states}
    options = []
    for p in partitions(states):
        if partition_bits(p) > bits or goal_mean_bayes_error(states, p, goals, mass) > max_goal_mean:
            continue
        value, macro = optimum_tv_for_partition(states, p, kernels)
        options.append((value, tuple(p[s] for s in states), macro))
    if not options:
        return None
    return min(options, key=lambda row: (row[0], row[1]))


def fixture():
    """Four transient states, one absorbing goal, plus a benign wait action."""
    states = tuple(range(5))
    ps = (Q(1, 5), Q(2, 5), Q(3, 5), Q(4, 5))
    hit = {s: {s: 1 - p, 4: p} for s, p in enumerate(ps)}
    hit[4] = {4: Q(1)}
    idle = {s: {s: Q(1)} for s in states}
    goals = [{s: int(s == 4) for s in states}]
    return states, goals, {'hit': hit, 'idle': idle}


def deterministic_fixture():
    states = tuple(range(6))
    goal = {s: int(s == 4) for s in states}
    tick = {0: 2, 1: 3, 2: 4, 3: 4, 4: 4, 5: 5}
    reset = {0: 0, 1: 0, 2: 0, 3: 0, 4: 4, 5: 5}
    return states, [goal], {'tick': tick, 'reset': reset}


def fixed_macro_tv_defect(states, partition, kernels, macro):
    """Uniform TV error for a *frozen* quotient and macro action kernels."""
    return max(_tv(_project(kernels[a][s], partition), macro[a][partition[s]])
               for a in kernels for s in states)


def kernel_shift_tv(states, old_kernels, new_kernels):
    """Maximum preprojection TV shift for unchanged microstate and action labels."""
    assert set(old_kernels) == set(new_kernels)
    return max(_tv(old_kernels[a][s], new_kernels[a][s])
               for a in old_kernels for s in states)


def hit_probability(kernel, initial, goal_states, horizon):
    """Finite-horizon reachability for one fixed action, exact Fractions."""
    assert horizon >= 0
    values = {s: Q(int(s in goal_states)) for s in kernel}
    for _ in range(horizon):
        values = {s: Q(1) if s in goal_states else
                  sum((w * values[t] for t, w in row.items()), Q(0))
                  for s, row in kernel.items()}
    return values[initial]

"""Finite deterministic autonomy defect and stochastic TV diagnostics, exact rationals."""
from fractions import Fraction as Q
from itertools import combinations, product

def fibres(states, projection):
    result = {}
    for s in states:
        result.setdefault(projection[s], []).append(s)
    return result

def action_successors(states, projection, updates):
    parts = fibres(states, projection)
    return {(z, a): frozenset(projection[updates[a][s]] for s in fibre)
            for z, fibre in parts.items() for a in updates}

def radius(outcomes, centers, distance):
    """Exact Chebyshev radius, witness center; finite nonempty sets."""
    assert outcomes and centers
    candidates = [(max(distance(y, c) for y in outcomes), c) for c in centers]
    return min(candidates, key=lambda x: (x[0], str(x[1])))

def optimum(states, projection, updates, macro_states, distance):
    targets = action_successors(states, projection, updates)
    opt = {key: radius(values, macro_states, distance) for key, values in targets.items()}
    return max(v[0] for v in opt.values()), {key: v[1] for key, v in opt.items()}

def direct_optimum(states, projection, updates, macro_states, distance):
    """Independent exhaustive oracle over all macro transition tables."""
    keys = list(action_successors(states, projection, updates))
    best = None
    for centers in product(macro_states, repeat=len(keys)):
        macro = dict(zip(keys, centers))
        loss = max(distance(projection[updates[a][s]], macro[projection[s], a])
                   for a in updates for s in states)
        if best is None or loss < best:
            best = loss
    return best

def tv(p, q):
    keys = set(p) | set(q)
    assert sum(p.values()) == 1 and sum(q.values()) == 1
    return sum((abs(p.get(z, Q(0)) - q.get(z, Q(0))) for z in keys), Q(0)) / 2

def midpoint(p, q):
    return {z: (p.get(z, Q(0)) + q.get(z, Q(0))) / 2 for z in set(p) | set(q)}

def project_kernel(row, projection):
    res = {}
    for s, probability in row.items():
        z = projection[s]
        res[z] = res.get(z, Q(0)) + probability
    return res

def stochastic_fibre_diameters(states, projection, kernels):
    groups = fibres(states, projection)
    return {(z, a): max((tv(project_kernel(kernels[a][s], projection),
                             project_kernel(kernels[a][t], projection))
                         for s in group for t in group), default=Q(0))
            for z, group in groups.items() for a in kernels}

def composed_defect(eps_first, eps_second, downstream_lipschitz):
    assert downstream_lipschitz >= 0 and eps_first >= 0 and eps_second >= 0
    return downstream_lipschitz * eps_first + eps_second

def tv_path_disagreement_bound(epsilon, steps):
    assert Q(0) <= epsilon <= Q(1) and steps >= 0
    return 1 - (1 - epsilon) ** steps

def tv_return_bound(epsilon, reward_gap, horizon):
    assert 0 <= reward_gap <= 1
    return sum((reward_gap + (1 - reward_gap) * tv_path_disagreement_bound(epsilon, t)
                for t in range(horizon)), Q(0))

def _rational_linear_solve(rows, rhs):
    """Unique solution of square rational linear system or None."""
    n = len(rows)
    augmented = [list(map(Q, row)) + [Q(val)] for row, val in zip(rows, rhs)]
    for k in range(n):
        pivot = next((i for i in range(k, n) if augmented[i][k]), None)
        if pivot is None:
            return None
        augmented[k], augmented[pivot] = augmented[pivot], augmented[k]
        leading = augmented[k][k]
        augmented[k] = [v / leading for v in augmented[k]]
        for i in range(n):
            if i != k:
                factor = augmented[i][k]
                augmented[i] = [a - factor * b for a, b in zip(augmented[i], augmented[k])]
    return tuple(augmented[i][-1] for i in range(n))

def tv_minimax_center(distributions, alphabet):
    """Exact LP-vertex oracle for tiny TV centers (alphabet <= 3). Exponential."""
    alphabet = tuple(alphabet)
    assert 1 <= len(alphabet) <= 3 and distributions
    assert all(set(row) <= set(alphabet) and sum(row.values()) == 1 and
               all(v >= 0 for v in row.values()) for row in distributions)
    n = len(alphabet)
    constraints = []  # coefficient vector, minimum right side; variables q_early,r
    for i in range(n-1):
        coefficients = [Q(0)]*n
        coefficients[i] = Q(1)
        constraints.append((tuple(coefficients), Q(0)))
    constraints.append((tuple([Q(-1)]*(n-1)+[Q(0)]), Q(-1)))
    constraints.append((tuple([Q(0)]*(n-1)+[Q(1)]), Q(0)))
    for p in distributions:
        for mask in range(1 << n):
            B = {alphabet[i] for i in range(n) if mask & (1 << i)}
            includes_last = alphabet[-1] in B
            coeffs = ([Q(-int(alphabet[j] not in B)) if includes_last
                       else Q(int(alphabet[j] in B)) for j in range(n-1)]
                      + [Q(1)])
            bound = sum((p.get(z, Q(0)) for z in B), Q(0)) - Q(int(includes_last))
            constraints.append((tuple(coeffs), bound))
    best = None
    for active in combinations(constraints, n):
        point = _rational_linear_solve([a for a, b in active], [b for a, b in active])
        if point is None or any(sum(a*b for a, b in zip(coef, point)) < bound
                                for coef, bound in constraints):
            continue
        if best is None or point[-1] < best[-1]:
            best = point
    if best is None:
        raise AssertionError("No feasible TV center LP vertex")
    center = {alphabet[i]: best[i] for i in range(n-1)}
    center[alphabet[-1]] = Q(1) - sum(best[:-1])
    assert max(tv(p, center) for p in distributions) == best[-1]
    return best[-1], center

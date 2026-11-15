"""Exact finite graph-switch and permutation-sweep coupling gym.

Four slots and a binary coordinate sweep: a Thorp-inspired toy,
not the full asymptotic Thorp shuffle theorem. All TVs are rational.
"""
from fractions import Fraction
from itertools import permutations, product
import json

ZERO = Fraction(0)
ONE = Fraction(1)


def perfect_matchings(vs):
    if not vs:
        yield ()
        return
    a = vs[0]
    for j in range(1, len(vs)):
        for rest in perfect_matchings(vs[1:j] + vs[j + 1:]):
            yield tuple(sorted(((a, vs[j]),) + rest))


def act(perm, matching):
    return tuple(sorted(tuple(sorted((perm[a], perm[b]))) for a, b in matching))


def tv(p, q):
    return sum((abs(p.get(x, ZERO) - q.get(x, ZERO)) for x in p.keys() | q.keys()), ZERO) / 2


def push(mu, kernel):
    result = {}
    for state, mass in mu.items():
        for target, prob in kernel[state].items():
            result[target] = result.get(target, ZERO) + mass * prob
    return {x: v for x, v in result.items() if v}


def iterate(mu, kernel, t):
    for _ in range(t):
        mu = push(mu, kernel)
    return mu


def switch_kernel(states):
    """Lazy edge switch restricted to perfect matchings of four vertices."""
    assert len(states) == 3 and all(len(m) == 2 for m in states)
    kernel = {}
    for g in states:
        (a, b), (c, d) = g
        first = tuple(sorted(((min(a, c), max(a, c)), (min(b, d), max(b, d)))))
        second = tuple(sorted(((min(a, d), max(a, d)), (min(b, c), max(b, c)))))
        assert first in states and second in states and first != second
        kernel[g] = {g: Fraction(1, 2), first: Fraction(1, 4), second: Fraction(1, 4)}
    return kernel


def position_swap(perm, edge):
    u, v = edge
    return tuple(v if x == u else (u if x == v else x) for x in perm)


def sweep_kernel(perms):
    """Independent swaps along the two coordinate directions of F_2^2."""
    directions = (((0, 1), (2, 3)), ((0, 2), (1, 3)))
    kernel = {}
    for p in perms:
        outcomes = {}
        for bits in product((0, 1), repeat=4):
            q = p
            for edge, flag in zip(directions[0] + directions[1], bits):
                if flag:
                    q = position_swap(q, edge)
            outcomes[q] = outcomes.get(q, ZERO) + Fraction(1, 16)
        kernel[p] = outcomes
    return kernel


def tensor(a, b):
    return {(x, y): p * q for x, p in a.items() for y, q in b.items()}


def product_kernel(a, b):
    return {(x, y): tensor(row_x, b[y]) for x, row_x in a.items() for y in b}


def shared_quantile(a, b):
    """Exact shared-uniform-quantile coupling of two finite laws."""
    sa, sb = sorted(a.items()), sorted(b.items())
    i = j = 0
    left_a = sa[0][1]
    left_b = sb[0][1]
    result = {}
    while i < len(sa) and j < len(sb):
        mass = min(left_a, left_b)
        key = (sa[i][0], sb[j][0])
        result[key] = result.get(key, ZERO) + mass
        left_a -= mass
        left_b -= mass
        if left_a == 0:
            i += 1
            if i < len(sa):
                left_a = sa[i][1]
        if left_b == 0:
            j += 1
            if j < len(sb):
                left_b = sb[j][1]
    assert sum(result.values(), ZERO) == ONE
    return result


def coupled_kernel(a, b, share):
    """Preserve both conditional marginals; mix product and shared coupling."""
    assert 0 <= share <= 1
    result = {}
    for x in a:
        for y in b:
            independent = tensor(a[x], b[y])
            synchronized = shared_quantile(a[x], b[y])
            result[x, y] = {z: (1 - share) * independent.get(z, ZERO) + share * synchronized.get(z, ZERO)
                            for z in independent.keys() | synchronized.keys()}
    return result


def dob_coefficient(kernel):
    rows = list(kernel.values())
    return max((tv(a, b) for i, a in enumerate(rows) for b in rows[:i]), default=ZERO)


def evaluate(share=Fraction(1, 4), horizon=5, block=3):
    states = tuple(perfect_matchings(tuple(range(4))))
    perms = tuple(permutations(range(4)))
    P, Q = switch_kernel(states), sweep_kernel(perms)
    stationary_g = {g: Fraction(1, len(states)) for g in states}
    stationary_p = {p: Fraction(1, len(perms)) for p in perms}
    stationary_joint = tensor(stationary_g, stationary_p)
    joint_product = product_kernel(P, Q)
    joint_coupled = coupled_kernel(P, Q, share)
    defect = max(tv(joint_product[z], joint_coupled[z]) for z in joint_product)
    assert all(sum(row.values(), ZERO) == ONE for row in joint_coupled.values())
    for (g, perm), row in joint_coupled.items():
        for g1, mass in P[g].items():
            assert sum((v for (u, _), v in row.items() if u == g1), ZERO) == mass
        for p1, mass in Q[perm].items():
            assert sum((v for (_, u), v in row.items() if u == p1), ZERO) == mass
    state_g, state_p = states[0], perms[0]
    mu_g, mu_p = {state_g: ONE}, {state_p: ONE}
    mu_coupled = {(state_g, state_p): ONE}
    records = []
    for t in range(horizon + 1):
        a = tv(mu_g, stationary_g)
        b = tv(mu_p, stationary_p)
        independent = tensor(mu_g, mu_p)
        measured = tv(mu_coupled, stationary_joint)
        independent_tv = tv(independent, stationary_joint)
        bound = min(ONE, a + b - a * b + t * defect)
        assert tv(mu_coupled, independent) <= min(ONE, t * defect)
        assert measured <= bound
        records.append({'t': t, 'graph_tv': str(a), 'permutation_tv': str(b),
                        'independent_tv': str(independent_tv), 'coupled_tv': str(measured),
                        'telescoping_bound': str(bound),
                        'joint_dependence_tv': str(tv(mu_coupled, independent))})
        mu_g, mu_p = push(mu_g, P), push(mu_p, Q)
        mu_coupled = push(mu_coupled, joint_coupled)
    pb = {x: iterate({x: ONE}, P, block) for x in states}
    qb = {x: iterate({x: ONE}, Q, block) for x in perms}
    rho = dob_coefficient(product_kernel(pb, qb))
    d0 = tv({(state_g, state_p): ONE}, stationary_joint)
    assert 0 <= rho < 1
    for row in records:
        k, r = divmod(row['t'], block)
        rhs = min(ONE, (rho ** k) * d0 + block * defect * sum((rho ** j for j in range(k)), ZERO) + r * defect)
        assert Fraction(row['coupled_tv']) <= rhs
        row['block_bound'] = str(rhs)
    m = len(states)
    return {'n_vertices': 4, 'graph_states': m, 'deck_permutations': len(perms),
            'share': str(share), 'conditional_independence_defect': str(defect),
            'block_length': block, 'block_contraction': str(rho),
            'orbit_lock_tv': str(Fraction(m - 1, m)),
            'orbit_lock_event_probability': '1',
            'independent_event_probability': str(Fraction(1, m)),
            'rows': records}


if __name__ == '__main__':
    print(json.dumps(evaluate(), indent=2))

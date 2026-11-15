"""SMC-02: exact orbit-locked Markov lift of a graph-switch chain.

Each step right-multiplies the current permutation by an independently
sampled cross-edge transposition of a fixed base perfect matching. Its
projection to matchings is precisely the lazy two-edge switch chain.
All finite probabilities use Fraction; no random sampling.
"""
from fractions import Fraction
from itertools import permutations

ONE = Fraction(1)
ZERO = Fraction(0)


def base_matching(k):
    if k < 2:
        raise ValueError("k >= 2 required")
    return tuple((2 * i, 2 * i + 1) for i in range(k))


def cross_pairs(base):
    return tuple((a, b) for i, edge in enumerate(base)
                 for other in base[i + 1:] for a in edge for b in other)


def relabel(perm, matching):
    return tuple(sorted(tuple(sorted((perm[a], perm[b]))) for a, b in matching))


def transposition_right(perm, i, j):
    nxt = list(perm)
    nxt[i], nxt[j] = nxt[j], nxt[i]
    return tuple(nxt)


def all_matchings(vertices):
    if not vertices:
        yield ()
        return
    a, *rest = vertices
    for pos, b in enumerate(rest):
        for tail in all_matchings(tuple(rest[:pos] + rest[pos + 1:])):
            yield tuple(sorted(((a, b),) + tail))


def tv(a, b):
    return sum((abs(a.get(x, ZERO) - b.get(x, ZERO))
                for x in a.keys() | b.keys()), ZERO) / 2


def push(dist, kernel):
    new = {}
    for a, weight in dist.items():
        for b, rate in kernel[a].items():
            new[b] = new.get(b, ZERO) + weight * rate
    return {x: y for x, y in new.items() if y}


def kernel_lift(k, hold=Fraction(1, 2)):
    """Return symmetric permutation random walk and its induced switch chain.

    k=2 has 24 permutation states and 3 perfect matching states.
    For k>3, enumeration is factorial and intentionally refused.
    """
    if k not in (2, 3):
        raise ValueError("finite enumeration supports 2 or 3 pairs")
    if not (0 < hold < 1):
        raise ValueError("0 < hold < 1 required")
    base = base_matching(k)
    pairs = cross_pairs(base)
    perms = tuple(permutations(range(2 * k)))
    ms = tuple(all_matchings(tuple(range(2 * k))))
    row_p = {}
    for p in perms:
        row = {p: hold}
        for i, j in pairs:
            t = transposition_right(p, i, j)
            row[t] = row.get(t, ZERO) + (1 - hold) / len(pairs)
        row_p[p] = row
    row_g = {}
    for g in ms:
        row = {g: hold}
        for (a, b) in cross_pairs(g):
            t = list(range(2 * k))
            t[a], t[b] = t[b], t[a]
            ng = relabel(t, g)
            row[ng] = row.get(ng, ZERO) + (1 - hold) / len(pairs)
        row_g[g] = row
    # Exact Markov lumpability on all permutation representatives.
    for p, row in row_p.items():
        g = relabel(p, base)
        for ng, w in row_g[g].items():
            projected = sum((weight for p1, weight in row.items()
                             if relabel(p1, base) == ng), ZERO)
            assert projected == w
    return base, perms, ms, row_p, row_g


def collision(dist):
    return sum((p * p for p in dist.values()), ZERO)


def locked_measure(dist_p, base):
    return {(relabel(p, base), p): w for p, w in dist_p.items()}


def independent_measure(dist_g, dist_p):
    return {(g, p): a * b for g, a in dist_g.items()
            for p, b in dist_p.items()}


def projection(dist_p, base):
    dist_g = {}
    for p, w in dist_p.items():
        g = relabel(p, base)
        dist_g[g] = dist_g.get(g, ZERO) + w
    return dist_g


def exact_sweep(k=2, horizon=12, hold=Fraction(1, 2)):
    base, perms, ms, Q, P = kernel_lift(k, hold)
    start_p = tuple(range(2 * k))
    current = {start_p: ONE}
    u_p = {x: Fraction(1, len(perms)) for x in perms}
    u_g = {x: Fraction(1, len(ms)) for x in ms}
    u_joint = independent_measure(u_g, u_p)
    records = []
    for t in range(horizon + 1):
        mg = projection(current, base)
        collision_value = collision(mg)
        locked = locked_measure(current, base)
        independent = independent_measure(mg, current)
        exact_gap = ONE - collision_value
        assert tv(locked, independent) == exact_gap
        assert tv(mg, u_g) <= tv(current, u_p)
        locked_tv = tv(locked, u_joint)
        assert locked_tv >= ONE - Fraction(1, len(ms))
        result = {
            't': t,
            'graph_tv': str(tv(mg, u_g)),
            'perm_tv': str(tv(current, u_p)),
            'joint_tv_from_uniform_product': str(locked_tv),
            'tv_from_time_matched_product': str(exact_gap),
            'graph_collision_probability': str(collision_value),
            'orbit_lock_event_probability': '1',
            'matched_null_event_probability': str(collision_value),
        }
        if k == 2 and hold == Fraction(1, 2):
            spectral_gap = Fraction(2, 3) * (ONE - Fraction(1, 16) ** t)
            assert exact_gap == spectral_gap
            assert tv(mg, u_g) == Fraction(2, 3) * Fraction(1, 4) ** t
            result['closed_form_relational_tv'] = str(spectral_gap)
        records.append(result)
        current = push(current, Q)
    return {'n_vertices': 2 * k, 'n_graph_states': len(ms),
            'n_permutation_states': len(perms), 'hold': str(hold),
            'cross_transpositions': len(cross_pairs(base)),
            'asymptotic_relational_tv': str(ONE - Fraction(1, len(ms))),
            'rows': records}


def _apply_operator(vec, kernel):
    return push(vec, kernel)


def check_four_slot_spectrum():
    """Exact polynomial annihilator and trace moments of 24 x 24 kernel.

    Spectrum is {0:1, 1/4:8, 1/2:6, 3/4:8, 1:1}.
    No floating eigensolver or SymPy required.
    """
    _, perms, _, K, _ = kernel_lift(2)
    roots = (ZERO, Fraction(1, 4), Fraction(1, 2), Fraction(3, 4), ONE)

    def operator_minus_scalar(v, scale):
        next_vec = _apply_operator(v, K)
        return {p: next_vec.get(p, ZERO) - scale * v.get(p, ZERO) for p in perms}

    traces = [Fraction(len(perms))]
    for p in perms:
        v = {p: ONE}
        for root in roots:
            v = operator_minus_scalar(v, root)
        assert all(w == 0 for w in v.values())
    for n in range(1, 5):
        trace = ZERO
        for p in perms:
            v = {p: ONE}
            for _ in range(n):
                v = _apply_operator(v, K)
            trace += v.get(p, ZERO)
        traces.append(trace)
    expected = {ZERO: 1, Fraction(1, 4): 8,
                Fraction(1, 2): 6, Fraction(3, 4): 8, ONE: 1}
    for n in range(5):
        assert traces[n] == sum((r**n * multiplicity for r, multiplicity in expected.items()), ZERO)
    return {'roots_with_multiplicity': {str(r): m for r, m in expected.items()},
            'trace_moments_0_through_4': list(map(str, traces)),
            'annihilator_checked_all_24_basis_rows': True}


if __name__ == '__main__':
    import json
    report = exact_sweep(2, 16)
    report['spectral_certificate'] = check_four_slot_spectrum()
    print(json.dumps(report, indent=2))

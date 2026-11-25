"""Problem families for DCF-14. Each family fixes how factor scopes are drawn;
noise, factor kinds, query scope and model misspecification are set separately
so that structure and difficulty vary independently."""
import numpy as np
from fg import Problem, sensor_table, sensor_value

FAMILIES = ('chain', 'tree', 'star', 'grid', 'sparse', 'dense', 'hyper', 'ktree', 'xor_noiseless', 'weak_dense')
NOISE = {'low': (0.02, 0.08), 'high': (0.15, 0.30)}


def scopes_for(family, n, rng, k=4):
    if family == 'chain':
        return [(i, i + 1) for i in range(n - 1)]
    if family == 'tree':
        return [(int(rng.integers(i)), i) for i in range(1, n)]
    if family == 'star':
        return [(0, i) for i in range(1, n)]
    if family == 'grid':
        r = int(np.floor(np.sqrt(n)))
        c = int(np.ceil(n / r))
        out = []
        for v in range(n):
            a, b = divmod(v, c)
            if b + 1 < c and v + 1 < n:
                out.append((v, v + 1))
            if v + c < n:
                out.append((v, v + c))
        return out
    if family == 'sparse':
        m = int(round(1.5 * n))
        return [tuple(int(x) for x in rng.choice(n, 2, replace=False)) for _ in range(m)]
    if family in ('dense', 'weak_dense'):
        return [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < 0.5]
    if family == 'hyper':
        return [tuple(int(x) for x in rng.choice(n, int(rng.choice([3, 4])), replace=False)) for _ in range(n)]
    if family == 'xor_noiseless':
        return [tuple(int(x) for x in rng.choice(n, 3, replace=False)) for _ in range(n)]
    if family == 'ktree':
        # planted partial k-tree: treewidth <= k regardless of n
        k = min(k, n - 1)
        cliques = [list(range(k + 1))]
        edges = {(a, b) for a in range(k + 1) for b in range(a + 1, k + 1)}
        for v in range(k + 1, n):
            base = cliques[int(rng.integers(len(cliques)))]
            drop = int(rng.integers(len(base)))
            sub = [u for i, u in enumerate(base) if i != drop]
            for u in sub:
                edges.add((u, v))
            cliques.append(sub + [v])
        return [e for e in sorted(edges) if rng.random() < 0.7]
    raise ValueError(family)


def make_problem(family, n, seed, noise='low', query_size=2, misspec=False, unary_frac=0.5, k=4):
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 2, n)
    multi = scopes_for(family, n, rng, k)
    lo, hi = NOISE[noise]
    scopes, tables, true_tables, meta = [], [], [], []

    def add(scope, kind, e, e_model):
        bits = x[list(scope)][None, :]
        clean = int(sensor_value(bits, kind)[0])
        y = clean ^ int(rng.random() < e)
        scopes.append(tuple(scope))
        true_tables.append(sensor_table(len(scope), kind, y, e))
        tables.append(sensor_table(len(scope), kind, y, e_model))
        meta.append(dict(kind=kind, y=y, e=e_model, arity=len(scope)))

    if family == 'xor_noiseless':
        for v in range(n):
            if rng.random() < 0.25:
                add((v,), 'id', 0.0, 0.0)
        for s in multi:
            add(s, 'xor', 0.0, 0.0)
    else:
        for v in range(n):
            if rng.random() < unary_frac:
                add((v,), 'id', 0.15, 0.15)
        for s in multi:
            kinds = ['xor', 'and', 'agree'] if len(s) == 2 else ['xor', 'and', 'maj']
            kind = str(rng.choice(kinds))
            e = 0.45 if family == 'weak_dense' else float(rng.uniform(lo, hi))
            e_model = max(0.005, e * 0.4) if misspec else e
            add(s, kind, e, e_model)
    # query: a random variable and up to query_size-1 of its graph neighbours or random others
    q = [int(rng.integers(n))]
    nb = sorted({u for s in multi if q[0] in s for u in s} - {q[0]})
    while len(q) < query_size:
        pool = [u for u in nb if u not in q] if (nb and rng.random() < 0.75) else [u for u in range(n) if u not in q]
        if not pool:
            pool = [u for u in range(n) if u not in q]
        q.append(int(rng.choice(pool)))
    info = dict(family=family, n=n, seed=seed, noise=noise, query_size=query_size, misspec=misspec)
    return Problem(n, scopes, tables, q, meta, true_tables if misspec else None, x, info)

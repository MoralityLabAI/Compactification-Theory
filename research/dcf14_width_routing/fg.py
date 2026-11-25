"""Binary factor graphs: elimination orderings, exact and approximate inference.

Shared library for DCF-13D through DCF-16. Variables are binary with a uniform
independent prior; every factor is a nonnegative likelihood table over its scope.
The query is the parity of a set of variables.

Every inference routine returns a dict with at least
    p      probability that the query parity is 1 (None when the method declines)
    ops    counted table-entry products/lookups (method-specific, see THEORY.md)
Exact routines are written along three independent code paths (einsum variable
elimination, chunked log-space enumeration, batched cutset conditioning with
forest peeling) so that each can verify the others.
"""
import math
import numpy as np

KINDS = ('id', 'xor', 'and', 'maj', 'agree', 'amb')


def sensor_value(bits, kind, mix=0.0):
    """Noise-free sensor output for assignment rows `bits` (m x arity)."""
    arity = bits.shape[1]
    s = bits.sum(1)
    if kind in ('id', 'xor'):
        v = s % 2
    elif kind == 'and':
        v = s == arity
    elif kind == 'maj':
        v = 2 * s > arity
    elif kind == 'agree':
        v = (s == 0) | (s == arity)
    elif kind == 'amb':
        v = (1 - mix) * (s % 2) + mix * (2 * s > arity)
    else:
        raise ValueError(kind)
    return np.asarray(v, dtype=float)


def sensor_table(arity, kind, y, e, mix=0.0):
    """Likelihood table P(y | x_scope) with bit-flip probability e; axis i <-> scope[i]."""
    bits = np.indices((2,) * arity).reshape(arity, -1).T
    p = e + (1 - 2 * e) * sensor_value(bits, kind, mix)
    return (p if y else 1 - p).reshape((2,) * arity)


class Problem:
    """A query over a binary factor graph.

    scopes/tables: the model the solver sees. true_tables: the generating model
    (differs only under deliberate misspecification). meta: per-factor sensor
    descriptions (kind, y, e) used by specialised operators such as GF(2).
    """

    def __init__(self, n, scopes, tables, query, meta=None, true_tables=None, truth=None, info=None):
        self.n = n
        self.scopes = [tuple(int(v) for v in s) for s in scopes]
        self.tables = [np.asarray(t, dtype=float) for t in tables]
        self.query = tuple(sorted(int(v) for v in query))
        self.meta = meta or [None] * len(self.scopes)
        self.true_tables = true_tables
        self.truth = truth
        self.info = info or {}

    def with_tables(self, tables):
        return Problem(self.n, self.scopes, tables, self.query, self.meta, None, self.truth, self.info)

    def truth_parity(self):
        return int(sum(int(self.truth[v]) for v in self.query) % 2)


# ---------------------------------------------------------------- structure

def query_component(prob):
    """Restrict to factors connected to the query. Exact for the query marginal
    because the prior factorises and disconnected likelihoods cancel."""
    parent = list(range(prob.n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for s in prob.scopes:
        for v in s[1:]:
            ra, rb = find(s[0]), find(v)
            if ra != rb:
                parent[ra] = rb
    roots = {find(v) for v in prob.query}
    keep = [i for i, s in enumerate(prob.scopes) if s and find(s[0]) in roots]
    variables = sorted({v for v in range(prob.n) if find(v) in roots})
    sub = Problem(prob.n, [prob.scopes[i] for i in keep], [prob.tables[i] for i in keep], prob.query,
                  [prob.meta[i] for i in keep],
                  None if prob.true_tables is None else [prob.true_tables[i] for i in keep],
                  prob.truth, prob.info)
    return sub, variables


def adjacency(variables, scopes):
    adj = {v: set() for v in variables}
    for s in scopes:
        for a in s:
            adj[a].update(b for b in s if b != a)
    return adj


def elimination_order(variables, scopes, keep, heuristic='min_fill'):
    """Greedy elimination order over `variables` minus `keep`.

    min_degree  fewest current neighbours (equals DCF-13B's minimum union scope)
    min_fill    fewest fill edges
    weighted_min_fill  fill edges weighted by the product of the endpoints'
                factor counts. With uniform binary domains Koller-Friedman's
                domain-size weighting is identical to min_fill, so this
                variant weights by local factor load instead.
    Returns (order, induced_width) where induced width is the largest neighbour
    set at elimination time (keep variables are never eliminated).
    """
    adj = adjacency(variables, scopes)
    load = {v: 0 for v in variables}
    for s in scopes:
        for a in s:
            load[a] += 1
    todo = set(variables) - set(keep)
    order, width = [], 0
    while todo:
        best, best_score = None, None
        for v in sorted(todo):
            nb = adj[v]
            if heuristic == 'min_degree':
                score = len(nb)
            else:
                nbl = sorted(nb)
                fill = 0
                for i, a in enumerate(nbl):
                    for b in nbl[i + 1:]:
                        if b not in adj[a]:
                            fill += 1 if heuristic == 'min_fill' else (load[a] + 1) * (load[b] + 1)
                score = (fill, len(nb))
            if best_score is None or score < best_score:
                best, best_score = v, score
        nb = adj[best]
        width = max(width, len(nb))
        for a in nb:
            adj[a].update(b for b in nb if b != a)
            adj[a].discard(best)
            load[a] += 1
        del adj[best]
        todo.discard(best)
        order.append(best)
    return order, width


def symbolic_ve(scopes, order, keep):
    """Exact operation count and largest union scope of einsum VE for this order,
    computed without touching any table."""
    live = [frozenset(s) for s in scopes] + [frozenset([q]) for q in keep]
    ops, max_union, peak = 0, 0, sum(2 ** len(s) for s in live)
    cur = peak
    for v in order:
        bucket = [s for s in live if v in s]
        if not bucket:
            continue
        union = frozenset().union(*bucket)
        ops += 2 ** len(union) * len(bucket)
        max_union = max(max_union, len(union))
        live = [s for s in live if v not in s]
        out = union - {v}
        cur += 2 ** len(out)
        peak = max(peak, cur)
        cur -= sum(2 ** len(s) for s in bucket)
        live.append(out)
    ops += 2 ** len(keep) * max(1, len(live))
    return dict(ops=ops, max_union=max_union, peak_entries=peak)


def treewidth_lower_bound(variables, scopes):
    """Minor-min-width style lower bound (degeneracy of the interaction graph)."""
    adj = adjacency(variables, scopes)
    lb = 0
    while adj:
        v = min(adj, key=lambda a: (len(adj[a]), a))
        lb = max(lb, len(adj[v]))
        for a in adj[v]:
            adj[a].discard(v)
        del adj[v]
    return lb


def loop_cutset(variables, scopes, query):
    """Greedy cutset (always containing the query) that leaves a factor forest."""
    cut = set(query)
    var_list = sorted(variables)
    while True:
        free = [v for v in var_list if v not in cut]
        multi = [tuple(v for v in s if v not in cut) for s in scopes]
        multi = [m for m in multi if len(m) >= 2]
        nodes = set(v for m in multi for v in m)
        edges = sum(len(m) for m in multi)
        parent = {('v', v): ('v', v) for v in nodes}
        parent.update({('f', i): ('f', i) for i in range(len(multi))})

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        comps = len(parent)
        for i, m in enumerate(multi):
            for v in m:
                ra, rb = find(('f', i)), find(('v', v))
                if ra != rb:
                    parent[ra] = rb
                    comps -= 1
        if edges == len(parent) - comps:
            return sorted(cut)
        count = {v: 0 for v in free}
        for m in multi:
            for v in m:
                count[v] += 1
        cut.add(max(free, key=lambda v: (count[v], -v)))


def features(prob, budget_ops=None):
    """Structural routing features. Cost is part of the measurement (routing overhead)."""
    comp, variables = query_component(prob)
    order, w_fill = elimination_order(variables, comp.scopes, comp.query, 'min_fill')
    _, w_deg = elimination_order(variables, comp.scopes, comp.query, 'min_degree')
    sym = symbolic_ve(comp.scopes, order, comp.query)
    cut = loop_cutset(variables, comp.scopes, comp.query)
    arities = [len(s) for s in comp.scopes] or [0]
    noises = [m['e'] for m in comp.meta if m and m.get('arity', 1) > 1] or [0.5]
    return dict(n=prob.n, comp_vars=len(variables), comp_factors=len(comp.scopes),
                width_min_fill=w_fill, width_min_degree=w_deg,
                width_lower=treewidth_lower_bound(variables, comp.scopes),
                max_arity=max(arities), mean_arity=float(np.mean(arities)),
                query_size=len(prob.query), log2_ve_ops=math.log2(sym['ops']),
                log2_enum_ops=len(variables) + math.log2(max(1, len(comp.scopes))),
                cutset=len(cut), log2_cutset_ops=len(cut) + math.log2(max(1, len(comp.scopes) * 8)),
                mean_noise=float(np.mean(noises)), gf2_applicable=int(gf2_applicable(comp)),
                _order=order, _cut=cut)


def _ein_labels(ops, out, max_ops=30):
    """np.einsum over (table, labels) operands; more than max_ops operands are first
    multiplied pairwise onto their running label union (NumPy caps operands at 32)."""
    ops = list(ops)
    if len(ops) > max_ops:
        acc_t, acc_l = ops[0]
        for t, l in ops[1:]:
            u = list(dict.fromkeys(list(acc_l) + list(l)))
            acc_t = np.einsum(acc_t, list(acc_l), t, list(l), u)
            acc_l = u
        ops = [(acc_t, acc_l)]
    args = []
    for t, l in ops:
        args += [t, list(l)]
    return np.einsum(*args, list(out))


def _einsum(pairs, out):
    """einsum over (scope, table) pairs with variable ids remapped below 52."""
    lab = {}
    ops = [(t, [lab.setdefault(v, len(lab)) for v in s]) for s, t in pairs]
    return _ein_labels(ops, [lab.setdefault(v, len(lab)) for v in out])


# ---------------------------------------------------------------- exact: VE

def parity_prob(joint, query_order, query):
    """P(parity of `query` = 1) from a normalised joint table over `query_order`."""
    idx = np.indices(joint.shape).reshape(len(query_order), -1)
    sel = [query_order.index(q) for q in query]
    par = idx[sel].sum(0) % 2 if sel else np.zeros(idx.shape[1], int)
    flat = joint.reshape(-1)
    return float(flat[par == 1].sum() / flat.sum())


def variable_elimination(prob, order=None, heuristic='min_fill', budget_entries=None):
    """Exact einsum VE on the query component. With budget_entries, declines (p=None)
    before computing anything if the symbolic largest table exceeds the budget."""
    comp, variables = query_component(prob)
    if order is None:
        order, _ = elimination_order(variables, comp.scopes, comp.query, heuristic)
    sym = symbolic_ve(comp.scopes, order, comp.query)
    if budget_entries is not None and 2 ** sym['max_union'] > budget_entries:
        return dict(p=None, ops=0, declined=True, sym_ops=sym['ops'], max_union=sym['max_union'], peak_entries=sym['peak_entries'])
    factors = [(list(s), t) for s, t in zip(comp.scopes, comp.tables)] + [([q], np.ones(2)) for q in comp.query]
    ops = 0
    for v in order:
        bucket = [f for f in factors if v in f[0]]
        if not bucket:
            continue
        factors = [f for f in factors if v not in f[0]]
        union = sorted(set().union(*[set(s) for s, _ in bucket]))
        out = [u for u in union if u != v]
        new = _einsum(bucket, out)
        m = new.max()
        if m > 0:
            new = new / m
        ops += 2 ** len(union) * len(bucket)
        factors.append((out, new))
    q = list(comp.query)
    joint = _einsum(factors, q) if factors else np.ones((2,) * len(q))
    joint = np.broadcast_to(joint, (2,) * len(q)) if joint.ndim < len(q) else joint
    ops += 2 ** len(q) * max(1, len(factors))
    return dict(p=parity_prob(np.asarray(joint, float), q, comp.query), ops=ops, declined=False, sym_ops=sym['ops'], max_union=sym['max_union'], peak_entries=sym['peak_entries'])


def ve_broadcast(prob, heuristic='min_degree', max_union=24):
    """Second exact elimination path: explicit axis alignment and broadcasting
    (no einsum), reverse tie-breaking order, log-scaled. Used only as a verifier."""
    comp, variables = query_component(prob)
    adj = adjacency(variables, comp.scopes)
    keep = set(comp.query)
    order = []
    todo = set(variables) - keep
    while todo:
        v = max(sorted(todo), key=lambda a: (-len(adj[a]), a))
        nb = adj.pop(v)
        for a in nb:
            adj[a].update(b for b in nb if b != a)
            adj[a].discard(v)
        todo.discard(v)
        order.append(v)
    facs = []
    for s, t in zip(comp.scopes, comp.tables):
        vs = sorted(set(s))
        tt = np.transpose(t, [s.index(v) for v in vs])
        facs.append((tuple(vs), tt))
    facs += [((q,), np.ones(2)) for q in comp.query]

    def align(vs, t, union):
        shape = [2 if u in vs else 1 for u in union]
        return t.reshape(shape)

    for v in order:
        bucket = [f for f in facs if v in f[0]]
        if not bucket:
            continue
        facs = [f for f in facs if v not in f[0]]
        union = tuple(sorted(set().union(*[set(f[0]) for f in bucket])))
        if len(union) > max_union:
            return dict(p=None, declined=True)
        prod = np.ones([1] * len(union))
        for vs, t in bucket:
            prod = prod * align(vs, t, union)
        new = prod.sum(axis=union.index(v))
        m = new.max()
        facs.append((tuple(u for u in union if u != v), new / m if m > 0 else new))
    q = tuple(sorted(comp.query))
    joint = np.ones([2] * len(q))
    for vs, t in facs:
        joint = joint * align(vs, t, q)
    return dict(p=parity_prob(joint, list(q), comp.query), declined=False)


# --------------------------------------------------------- exact: enumeration

def enumerate_exact(prob, max_vars=24, chunk=1 << 16):
    """Log-space brute force over the query component (independent code path)."""
    comp, variables = query_component(prob)
    c = len(variables)
    if c > max_vars:
        return dict(p=None, ops=0, declined=True)
    pos = {v: i for i, v in enumerate(variables)}
    acc = [-np.inf, -np.inf]
    shifts = np.arange(c)
    for start in range(0, 2 ** c, chunk):
        idx = np.arange(start, min(2 ** c, start + chunk), dtype=np.int64)
        bits = (idx[:, None] >> shifts) & 1
        logw = np.zeros(len(idx))
        with np.errstate(divide='ignore'):
            for s, t in zip(comp.scopes, comp.tables):
                flat = np.zeros(len(idx), dtype=np.int64)
                for v in s:
                    flat = flat * 2 + bits[:, pos[v]]
                logw += np.log(t.reshape(-1)[flat])
        par = bits[:, [pos[q] for q in comp.query]].sum(1) % 2 if comp.query else np.zeros(len(idx), int)
        for k in (0, 1):
            sel = logw[par == k]
            if sel.size and np.isfinite(sel).any():
                m = sel[np.isfinite(sel)].max()
                acc[k] = np.logaddexp(acc[k], m + math.log(np.exp(sel - m).sum()))
    m = max(acc)
    w = np.exp(np.array(acc) - m)
    return dict(p=float(w[1] / w.sum()), ops=2 ** c * max(1, len(comp.scopes)), declined=False)


# ------------------------------------------------------ exact: cutset + peeling

def cutset_exact(prob, max_cutset=16, chunk=4096):
    """Condition on a loop cutset (including the query); peel the remaining factor
    forest leaf-first with batched messages. Independent of the einsum VE path's
    ordering and table layout."""
    comp, variables = query_component(prob)
    cut = loop_cutset(variables, comp.scopes, comp.query)
    if len(cut) > max_cutset:
        return dict(p=None, ops=0, declined=True, cutset=len(cut))
    cpos = {v: i for i, v in enumerate(cut)}
    M = 2 ** len(cut)
    logz = np.empty(M)
    ops = 0
    for start in range(0, M, chunk):
        rows = np.arange(start, min(M, start + chunk), dtype=np.int64)
        B = len(rows)
        cbits = (rows[:, None] >> np.arange(len(cut))) & 1
        facs = []  # (free vars list, batched table (B, 2,...,2))
        for s, t in zip(comp.scopes, comp.tables):
            free = [v for v in s if v not in cpos]
            index = []
            for v in s:
                index.append(cbits[:, cpos[v]] if v in cpos else slice(None))
            # advanced indexing over cut axes; batch axis first
            tb = np.empty((B,) + (2,) * len(free))
            fixed = [k for k, v in enumerate(s) if v in cpos]
            if not fixed:
                tb[...] = t
            else:
                perm = fixed + [k for k in range(len(s)) if k not in fixed]
                tt = np.transpose(t, perm).reshape((2 ** len(fixed),) + (2,) * len(free))
                flat = np.zeros(B, dtype=np.int64)
                for k in fixed:
                    flat = flat * 2 + cbits[:, cpos[s[k]]]
                tb[...] = tt[flat]
            facs.append((free, tb))
        scale = np.zeros(B)
        remaining = sorted(set(v for f, _ in facs for v in f))
        while remaining:
            pick = None
            for v in remaining:
                if sum(1 for f, _ in facs if v in f and len(f) >= 2) <= 1:
                    pick = v
                    break
            if pick is None:
                raise RuntimeError('cutset did not leave a forest')
            bucket = [x for x in facs if pick in x[0]]
            facs = [x for x in facs if pick not in x[0]]
            union = sorted(set().union(*[set(f) for f, _ in bucket]))
            out = [u for u in union if u != pick]
            lab = {v: i + 1 for i, v in enumerate(union)}
            new = _ein_labels([(tb, [0] + [lab[v] for v in f]) for f, tb in bucket], [0] + [lab[v] for v in out])
            m = new.reshape(B, -1).max(1)
            m = np.where(m > 0, m, 1.0)
            new = new / m.reshape((B,) + (1,) * len(out))
            with np.errstate(divide='ignore'):
                scale += np.log(m)
            ops += B * 2 ** len(union) * len(bucket)
            facs.append((out, new))
            remaining.remove(pick)
        tot = np.ones(B)
        for f, tb in facs:
            tot = tot * tb.reshape(B)
        with np.errstate(divide='ignore'):
            logz[start:start + B] = np.log(tot) + scale
    allbits = (np.arange(M)[:, None] >> np.arange(len(cut))) & 1
    par = allbits[:, [cpos[q] for q in comp.query]].sum(1) % 2
    acc = []
    for k in (0, 1):
        sel = logz[par == k]
        fin = sel[np.isfinite(sel)]
        acc.append(-np.inf if fin.size == 0 else fin.max() + math.log(np.exp(fin - fin.max()).sum()))
    m = max(acc)
    w = np.exp(np.array(acc) - m)
    return dict(p=float(w[1] / w.sum()), ops=ops, declined=False, cutset=len(cut))


# ------------------------------------------------------------- exact: GF(2)

def gf2_applicable(prob):
    return bool(prob.meta) and all(m is not None and m['e'] == 0 and m['kind'] in ('id', 'xor') for m in prob.meta)


def gf2_exact(prob):
    """Noise-free parity systems: the posterior is uniform on an affine subspace, so
    the query parity is determined iff the query vector lies in the row space."""
    if not gf2_applicable(prob):
        return dict(p=None, ops=0, declined=True)
    rows = {}
    ops = 0
    for s, m in zip(prob.scopes, prob.meta):
        vec = 0
        for v in s:
            vec ^= 1 << v
        rhs = m['y']
        while vec:
            ops += 1
            piv = vec.bit_length() - 1
            if piv not in rows:
                rows[piv] = (vec, rhs)
                break
            vec ^= rows[piv][0]
            rhs ^= rows[piv][1]
        else:
            if rhs:
                return dict(p=None, ops=ops, declined=True, inconsistent=True)
    q = 0
    for v in prob.query:
        q ^= 1 << v
    rhs = 0
    while q:
        ops += 1
        piv = q.bit_length() - 1
        if piv not in rows:
            return dict(p=0.5, ops=ops, declined=False)
        q ^= rows[piv][0]
        rhs ^= rows[piv][1]
    return dict(p=float(rhs), ops=ops, declined=False)


# ------------------------------------------------------------ approximations

def loopy_bp(prob, max_iters=100, damping=0.5, tol=1e-7, op_budget=None):
    """Sum-product on the factor graph of the query component. The query parity is
    read from a factor belief when one factor covers the query, otherwise from
    singleton beliefs assuming independence."""
    comp, variables = query_component(prob)
    F = len(comp.scopes)
    per_iter = sum(2 ** len(s) * len(s) for s in comp.scopes) + 1
    iters = max_iters if op_budget is None else min(max_iters, int(op_budget // per_iter))
    if iters < 1:
        return dict(p=None, ops=0, declined=True, iters=0, converged=False)
    f2v = {(i, v): np.full(2, .5) for i, s in enumerate(comp.scopes) for v in s}
    v2f = dict(f2v)
    nbrs = {v: [i for i, s in enumerate(comp.scopes) if v in s] for v in variables}
    ops, converged, it = 0, False, 0
    for it in range(1, iters + 1):
        for v in variables:
            for i in nbrs[v]:
                msg = np.ones(2)
                for j in nbrs[v]:
                    if j != i:
                        msg = msg * f2v[(j, v)]
                v2f[(i, v)] = msg / msg.sum()
        delta = 0.0
        for i, (s, t) in enumerate(zip(comp.scopes, comp.tables)):
            for a, v in enumerate(s):
                b = t
                for c, u in enumerate(s):
                    if u != v:
                        shape = [1] * len(s)
                        shape[c] = 2
                        b = b * v2f[(i, u)].reshape(shape)
                msg = b.sum(axis=tuple(c for c in range(len(s)) if c != a)) if len(s) > 1 else b.copy()
                tot = msg.sum()
                msg = msg / tot if tot > 0 else np.full(2, .5)
                new = damping * f2v[(i, v)] + (1 - damping) * msg
                delta = max(delta, float(np.abs(new - f2v[(i, v)]).max()))
                f2v[(i, v)] = new
        ops += per_iter
        if delta < tol:
            converged = True
            break
    q = list(comp.query)
    for i, s in enumerate(comp.scopes):
        if set(q) <= set(s):
            b = comp.tables[i]
            for c, u in enumerate(s):
                shape = [1] * len(s)
                shape[c] = 2
                b = b * v2f[(i, u)].reshape(shape)
            keep_axes = [s.index(u) for u in q]
            marg = b.sum(axis=tuple(c for c in range(len(s)) if c not in keep_axes))
            order = [s[c] for c in sorted(keep_axes)]
            return dict(p=parity_prob(marg, order, q), ops=ops, declined=False, iters=it, converged=converged)
    sign = 1.0
    for v in q:
        bel = np.ones(2)
        for j in nbrs[v]:
            bel = bel * f2v[(j, v)]
        bel = bel / bel.sum() if bel.sum() > 0 else np.full(2, .5)
        sign *= 1 - 2 * bel[1]
    return dict(p=float((1 - sign) / 2), ops=ops, declined=False, iters=it, converged=converged)


def mini_bucket(prob, ibound, order=None):
    """Mini-bucket elimination (sum in every mini-bucket: an upper bound on each
    query-configuration mass); returns the ratio of bounds as an approximation."""
    comp, variables = query_component(prob)
    if order is None:
        order, _ = elimination_order(variables, comp.scopes, comp.query, 'min_fill')
    factors = [(list(s), t) for s, t in zip(comp.scopes, comp.tables)] + [([q], np.ones(2)) for q in comp.query]
    ops = 0
    for v in order:
        bucket = [f for f in factors if v in f[0]]
        if not bucket:
            continue
        factors = [f for f in factors if v not in f[0]]
        bucket.sort(key=lambda f: -len(f[0]))
        minis = []
        for f in bucket:
            for mb in minis:
                if len(mb['scope'] | set(f[0])) <= ibound + 1:
                    mb['scope'] |= set(f[0])
                    mb['fs'].append(f)
                    break
            else:
                minis.append(dict(scope=set(f[0]), fs=[f]))
        for mb in minis:
            union = sorted(mb['scope'])
            out = [u for u in union if u != v]
            new = _einsum(mb['fs'], out)
            m = new.max()
            if m > 0:
                new = new / m
            ops += 2 ** len(union) * len(mb['fs'])
            factors.append((out, new))
    q = list(comp.query)
    joint = _einsum(factors, q) if factors else np.ones((2,) * len(q))
    ops += 2 ** len(q) * max(1, len(factors))
    return dict(p=parity_prob(np.asarray(joint, float), q, comp.query), ops=ops, declined=False)


def mini_bucket_ops(prob, ibound, order=None):
    """Symbolic op count for mini_bucket (same greedy partition, no tables)."""
    comp, variables = query_component(prob)
    if order is None:
        order, _ = elimination_order(variables, comp.scopes, comp.query, 'min_fill')
    live = [frozenset(s) for s in comp.scopes] + [frozenset([q]) for q in comp.query]
    ops = 0
    for v in order:
        bucket = sorted([s for s in live if v in s], key=lambda s: -len(s))
        if not bucket:
            continue
        live = [s for s in live if v not in s]
        minis = []
        for s in bucket:
            for mb in minis:
                if len(mb[0] | s) <= ibound + 1:
                    mb[0] |= s
                    mb[1] += 1
                    break
            else:
                minis.append([set(s), 1])
        for sc, cnt in minis:
            ops += 2 ** len(sc) * cnt
            live.append(frozenset(sc - {v}))
    return ops + 2 ** len(comp.query) * max(1, len(live))


def likelihood_weighting(prob, n_particles=4096, seed=0, z=2.0):
    """Importance sampling from the uniform prior. Returns p, an ESS-based interval
    half-width and the effective sample size."""
    comp, variables = query_component(prob)
    rng = np.random.default_rng(seed)
    pos = {v: i for i, v in enumerate(variables)}
    bits = rng.integers(0, 2, size=(n_particles, len(variables)))
    logw = np.zeros(n_particles)
    with np.errstate(divide='ignore'):
        for s, t in zip(comp.scopes, comp.tables):
            flat = np.zeros(n_particles, dtype=np.int64)
            for v in s:
                flat = flat * 2 + bits[:, pos[v]]
            logw += np.log(t.reshape(-1)[flat])
    ops = n_particles * max(1, sum(len(s) for s in comp.scopes))
    if not np.isfinite(logw).any():
        return dict(p=None, ops=ops, declined=True, ess=0.0, half_width=0.5)
    w = np.exp(logw - logw[np.isfinite(logw)].max())
    w = w / w.sum()
    par = bits[:, [pos[q] for q in comp.query]].sum(1) % 2
    p = float(w[par == 1].sum())
    ess = float(1.0 / (w ** 2).sum())
    hw = float(z * math.sqrt(max(p * (1 - p), 1e-12) / ess))
    return dict(p=p, ops=ops, declined=False, ess=ess, half_width=hw)


# ----------------------------------------------------------------- decisions

def decide(p, penalty=4.0, lo=None, hi=None):
    """Bayes action for reward +1 / -penalty / 0. With an interval [lo, hi] act
    only when the action is optimal at every point of the interval."""
    tau, low = penalty / (1 + penalty), 1 / (1 + penalty)
    if p is None:
        return -1
    if lo is not None:
        return 1 if lo >= tau else 0 if hi <= low else -1
    return 1 if p >= tau else 0 if p <= low else -1


def expected_utility(action, p_ref, penalty=4.0):
    if action < 0:
        return 0.0
    q = p_ref if action == 1 else 1 - p_ref
    return q - penalty * (1 - q)


def oracle_utility(p_ref, penalty=4.0):
    return max(0.0, expected_utility(1, p_ref, penalty), expected_utility(0, p_ref, penalty))

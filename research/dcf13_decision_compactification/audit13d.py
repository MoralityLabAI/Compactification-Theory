"""DCF-13D: completes the DCF-13 checklist on fresh seeds.

A  cost-estimated hybrid: choose component enumeration or min-scope elimination per
   episode from exact symbolic work counts (bound first, refine only when needed).
B  elimination orderings (min-degree, min-fill, weighted min-fill), budgeted
   elimination with pre-execution abstention, fixed-cluster approximation, with
   calibration and selective risk.
C  unknown sensor law: exact Bayesian model averaging over a shared mixture weight
   w ~ U[0,1] using Bernstein-basis polynomials, and credal decisions certified by
   Bernstein coefficient signs, against the DCF-13B pointwise bounds.

Usage: python audit13d.py calibrate   (dev seeds; writes calibration13d.json)
       python audit13d.py run         (fresh seeds; requires the frozen calibration)
"""
import sys
import math
import tracemalloc
from math import comb
from hybrid import *  # noqa: F401,F403  (gym, eliminate, hybrid: fixture, predict, ve, eliminate, components, ...)

sys.path.insert(0, str(ROOT.parent / 'dcf14_width_routing'))
import fg  # noqa: E402

P = json.loads((ROOT / 'protocol13d.json').read_text())
KIND = {0: 'xor', 1: 'and', 2: 'maj', 3: 'amb'}


# ------------------------------------------------------------------ part A

@njit(cache=True)
def ve_symbolic(k, query, masks, cap):
    """Replays eliminate()'s greedy order on scopes only; returns its exact work count."""
    F = len(masks)
    scopes = np.zeros(F + k, np.int64)
    active = np.zeros(F + k, np.bool_)
    work = 0
    width = 0
    for j in range(F):
        scopes[j] = masks[j]
        active[j] = True
        work += 1 << pop(masks[j])
        width = max(width, pop(masks[j]))
    remaining = ((1 << k) - 1) & ~query
    idx = F
    while remaining:
        best = -1
        bestwidth = k + 1
        union = 0
        for b in range(k):
            if not ((remaining >> b) & 1):
                continue
            m = 0
            for j in range(idx):
                if active[j] and ((scopes[j] >> b) & 1):
                    m |= scopes[j]
            d = pop(m)
            if d < bestwidth:
                bestwidth = d
                best = b
                union = m
        width = max(width, bestwidth)
        if bestwidth > cap:
            return work, width, 1
        remaining &= ~(1 << best)
        if union == 0:
            continue
        nw = 0
        for j in range(idx):
            if active[j] and ((scopes[j] >> best) & 1):
                nw += 1
                active[j] = False
        work += (1 << bestwidth) * nw
        scopes[idx] = union & ~(1 << best)
        active[idx] = True
        idx += 1
    nact = 0
    for j in range(idx):
        if active[j]:
            nact += 1
    work += (1 << pop(query)) * nact
    return work, width, 0


@njit(cache=True)
def enum_work(k, q, masks):
    comp = components(k, masks)
    todo = q
    w = 0
    while todo:
        b = 0
        while not ((todo >> b) & 1):
            b += 1
        m = comp[b]
        todo &= ~m
        c = 0
        for j in range(len(masks)):
            if masks[j] & m == masks[j]:
                c += 1
        w += (1 << pop(m)) * c
    return w, comp


@njit(cache=True)
def hybrid_cost(k, q, mm, kk, yy, ee, alpha):
    n = len(q)
    p = np.empty(n)
    route = np.zeros(n, np.int64)
    work = np.zeros(n, np.int64)
    for i in range(n):
        ew, comp = enum_work(k, q[i], mm[i])
        lb = 0
        for j in range(len(mm[i])):
            lb += 1 << pop(mm[i][j])
        use_enum = alpha * ew <= lb
        if not use_enum:
            vw, _, _ = ve_symbolic(k, q[i], mm[i], k)
            use_enum = alpha * ew <= vw
            route[i] = 1
        if use_enum:
            todo = q[i]
            sign = 1.
            while todo:
                b = 0
                while not ((todo >> b) & 1):
                    b += 1
                m = comp[b]
                todo &= ~m
                a, w, _ = enumerate_factor(m, q[i], mm[i], kk[i], yy[i], ee[i], .5)
                sign *= 1 - 2 * a
                work[i] += w
            p[i] = (1 - sign) / 2
        else:
            a, b, w, mem, d, s = eliminate(k, q[i], mm[i], kk[i], yy[i], ee[i], k, .5, 0)
            p[i] = b / (a + b)
            work[i] = w
            route[i] = 2
    return p, route, work


def timed_arms(arms, reps, rng):
    ret = {a: f() for a, f in arms.items()}
    times = {a: [] for a in arms}
    for _ in range(reps):
        for a in rng.permutation(list(arms)):
            t = time.perf_counter()
            r = arms[a]()
            times[a].append(time.perf_counter() - t)
            assert all(np.array_equal(x, y) for x, y in zip(r, ret[a]))
    return ret, times


def calibrate():
    rows = []
    rng = np.random.default_rng(P['calibration']['timing_rng'])
    for k, seed, sc in itertools.product(P['calibration']['widths'], P['calibration']['seeds'], S['scenarios']):
        d = fixture(k, seed + 300000 + 100 * k + 10000 * S['scenarios'].index(sc), P['n'], sc)
        q, mm, kk, yy, ee, truth = d
        args = (k, q, mm, kk, yy, ee)
        arms = {'component_full': lambda: predict(*args, 2, k, .5), 've_full': lambda: ve(*args, k, .5, False)}
        ret, times = timed_arms(arms, 3, rng)
        te, we = np.median(times['component_full']), ret['component_full'][1].sum()
        tv, wv = np.median(times['ve_full']), ret['ve_full'][3].sum()
        rows.append(dict(k=k, seed=seed, scenario=sc, enum_s_per_work=te / we, ve_s_per_work=tv / wv, ratio=(te / we) / (tv / wv)))
    alpha = float(np.median([r['ratio'] for r in rows]))
    out = dict(alpha=alpha, rows=rows, environment=env())
    (ROOT / 'calibration13d.json').write_text(json.dumps(out, indent=1) + '\n')
    print('alpha', alpha)


def env():
    return dict(python=platform.python_version(), numpy=np.__version__, numba=__import__('numba').__version__,
                machine=platform.machine(), processor=platform.processor())


def part_a(fixtures, alpha):
    rows, audit = [], dict(symbolic_mismatch=0, episodes=0)
    rng = np.random.default_rng(P['part_a']['timing_rng'])
    for (k, seed, sc), d in fixtures.items():
        if sc == 'uncertain_w':
            continue
        q, mm, kk, yy, ee, truth = d
        args = (k, q, mm, kk, yy, ee)
        arms = {'full': lambda: predict(*args, 0, k, .5),
                'component_full': lambda: predict(*args, 2, k, .5),
                've_full': lambda: ve(*args, k, .5, False),
                've6': lambda: ve(*args, 6, .5, False),
                'hybrid13c': lambda: hybrid(*args, 4, 6),
                'hybrid_cost': lambda: hybrid_cost(*args, alpha)}
        ret, times = timed_arms(arms, P['part_a']['reps'], rng)
        ref = ret['full'][0]
        # exact symbolic cost prediction audit
        for i in range(len(q)):
            sw, _, _ = ve_symbolic(k, q[i], mm[i], k)
            w = eliminate(k, q[i], mm[i], kk[i], yy[i], ee[i], k, .5, 0)[2]
            audit['symbolic_mismatch'] += int(sw != w)
            audit['episodes'] += 1
        for a, v in ret.items():
            p = v[0]
            skip = v[6] if a.startswith('ve') else v[2] if a == 'hybrid13c' else v[3] if a == 'component_full' or a == 'full' else np.zeros(len(q), np.int64)
            ok = skip == 0
            err = float(np.max(np.abs(p[ok] - ref[ok]))) if ok.any() else 0.
            row = dict(k=k, seed=seed, scenario=sc, arm=a, n=len(q), **decide(np.where(ok, p, .5), truth, ref),
                       seconds=float(np.median(times[a])), timing_samples=times[a], max_delta=err, skip=float((~ok).mean()))
            if a == 'hybrid_cost':
                row['route_enum_bound'] = float((v[1] == 0).mean())
                row['route_enum_refined'] = float(((v[1] == 1)).mean())
                row['route_ve'] = float((v[1] == 2).mean())
            rows.append(row)
        print('A', k, seed, sc, [(r['arm'], round(r['seconds'] * 1000, 3)) for r in rows[-6:]], flush=True)
    return rows, audit


# ------------------------------------------------------------------ part B

def to_problem(k, q, masks, kinds, ys, errs, mix=.5):
    scopes, tables = [], []
    for m, t, y, e in zip(masks, kinds, ys, errs):
        s = [b for b in range(k) if (int(m) >> b) & 1]
        scopes.append(s)
        tables.append(fg.sensor_table(len(s), KIND[int(t)], int(y), float(e), mix))
    return fg.Problem(k, scopes, tables, [b for b in range(k) if (int(q) >> b) & 1])


def calibration_stats(p, truth):
    p = np.asarray(p, float)
    bins = np.minimum((p * 10).astype(int), 9)
    ece = 0.
    for b in range(10):
        sel = bins == b
        if sel.any():
            ece += sel.mean() * abs(p[sel].mean() - truth[sel].mean())
    return dict(ece=float(ece), brier=float(np.mean((p - truth) ** 2)))


def score(p, truth, ref, covered=None):
    d = decide(p, truth, ref)
    a = np.array(d['actions'])
    cov = (a >= 0).sum()
    d['selective_risk'] = float(d['wrong'] / cov) if cov else 0.
    d.pop('actions')
    d.pop('word_utility')
    return d


def part_b(fixtures):
    rows, widths = [], []
    for (k, seed, sc), d in fixtures.items():
        if sc == 'uncertain_w':
            continue
        q, mm, kk, yy, ee, truth = d
        ref = predict(k, q, mm, kk, yy, ee, 0, k, .5)[0]
        probs = [to_problem(k, q[i], mm[i], kk[i], yy[i], ee[i]) for i in range(len(q))]
        arms = {}
        for h in ('min_degree', 'min_fill', 'weighted_min_fill'):
            arms['ve_' + h] = dict(heuristic=h, budget=None)
        for b in P['part_b']['budgets']:
            arms[f've_min_fill_budget{b}'] = dict(heuristic='min_fill', budget=b)
        for a, cfg in arms.items():
            ps, ops, unions, dec, ts, peaks = [], [], [], [], [], []
            tracemalloc.start()
            for i, pr in enumerate(probs):
                t = time.perf_counter()
                r = fg.variable_elimination(pr, heuristic=cfg['heuristic'], budget_entries=cfg['budget'])
                ts.append(time.perf_counter() - t)
                ps.append(.5 if r['p'] is None else r['p'])
                dec.append(r['declined'])
                ops.append(r['ops'])
                unions.append(r['max_union'])
                peaks.append(r['peak_entries'])
                if not r['declined']:
                    assert abs(r['p'] - ref[i]) < 1e-9, (a, i, r['p'], ref[i])
            _, peak_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            ps = np.array(ps)
            rows.append(dict(k=k, seed=seed, scenario=sc, arm=a, n=len(q), **score(ps, truth, ref), skip=float(np.mean(dec)),
                             mean_ops=float(np.mean(ops)), max_union=int(max(unions)), mean_union=float(np.mean(unions)),
                             max_peak_entries=int(max(peaks)), tracemalloc_peak_bytes=int(peak_bytes),
                             median_episode_seconds=float(np.median(ts)), **calibration_stats(ps, truth)))
        fx = predict(k, q, mm, kk, yy, ee, 1, 4, .5)
        rows.append(dict(k=k, seed=seed, scenario=sc, arm='fixed4', n=len(q), **score(fx[0], truth, ref), skip=0.,
                         **calibration_stats(fx[0], truth)))
        rows.append(dict(k=k, seed=seed, scenario=sc, arm='exact_reference', n=len(q), **score(ref, truth, ref), skip=0.,
                         **calibration_stats(ref, truth)))
        for pr in probs:
            comp, variables = fg.query_component(pr)
            widths.append(dict(k=k, scenario=sc, comp_vars=len(variables),
                               **{h: fg.elimination_order(variables, comp.scopes, comp.query, h)[1]
                                  for h in ('min_degree', 'min_fill', 'weighted_min_fill')}))
        print('B', k, seed, sc, flush=True)
    return rows, widths


# ------------------------------------------------------------------ part C

def fixture_w(k, seed, n):
    """Ambiguous triple sensors are majority with probability w_true ~ U[0,1] (shared
    within an episode), otherwise XOR."""
    rng = np.random.default_rng(seed)
    F = S['factors']
    masks = np.empty((n, F), np.int64)
    kind = masks.copy()
    ys = masks.copy()
    errs = np.empty((n, F))
    queries = np.empty(n, np.int64)
    truth = np.empty(n, np.int64)
    wtrue = np.empty(n)
    for i in range(n):
        w = float(rng.random())
        wtrue[i] = w
        order = rng.permutation(k)
        groups = [order[a:a + 4] for a in range(0, k, 4)]
        x = int(rng.integers(1 << k))
        g = groups[int(rng.integers(len(groups)))]
        qbits = rng.choice(g, size=2, replace=False)
        if rng.random() < .25:
            qbits = rng.choice(k, size=2, replace=False)
        qq = sum(1 << int(b) for b in qbits)
        queries[i] = qq
        truth[i] = parity(x & qq)
        for j in range(F):
            if j < k:
                bb = [j]
                t = 0
                e = .12
            else:
                g = groups[(j - k) % len(groups)]
                a = 2 if rng.random() < .5 else 3
                bb = rng.choice(g, size=a, replace=False)
                t = 3 if a == 3 else int(rng.choice([0, 1]))
                e = float(rng.uniform(.04, .16))
            m = sum(1 << int(b) for b in bb)
            actual = (2 if rng.random() < w else 0) if t == 3 else t
            y = value(x, m, actual) ^ int(rng.random() < e)
            masks[i, j] = m
            kind[i, j] = t
            ys[i, j] = y
            errs[i, j] = e
    return queries, masks, kind, ys, errs, truth, wtrue


def _lik(val, y, e):
    p = e + (1 - 2 * e) * val
    return p if y else 1 - p


def poly_masses(k, q, masks, kinds, ys, errs, region):
    """Scaled-Bernstein coefficients (basis w^i (1-w)^(m-i)) of the unnormalised
    masses of query parity 0 and 1, enumerating only the variables in `region`.
    A positive common factor is divided out (it cancels in every ratio and sign)."""
    pos = [b for b in range(k) if (region >> b) & 1]
    u = np.arange(1 << len(pos), dtype=np.int64)
    x = np.zeros_like(u)
    for t, b in enumerate(pos):
        x |= ((u >> t) & 1) << b
    coef = np.ones((len(u), 1))
    for m, t, y, e in zip(masks, kinds, ys, errs):
        m = int(m)
        if m & region != m:
            continue
        cnt = np.bitwise_count((x & m).astype(np.uint64)).astype(np.int64)
        if t == 3:
            l0 = _lik((cnt % 2).astype(float), y, e)
            l1 = _lik((cnt >= 2).astype(float), y, e)
            new = np.zeros((len(u), coef.shape[1] + 1))
            new[:, :-1] += coef * l0[:, None]
            new[:, 1:] += coef * l1[:, None]
            coef = new
        else:
            v = cnt % 2 if t == 0 else (cnt == pop_py(m)) if t == 1 else cnt >= 2
            coef = coef * _lik(np.asarray(v, float), y, e)[:, None]
        coef /= coef.max()
    par = np.bitwise_count((x & int(q)).astype(np.uint64)) % 2
    return coef[par == 0].sum(0), coef[par == 1].sum(0)


def pop_py(m):
    return bin(int(m)).count('1')


def combine(a, b):
    """XOR-combine two (N0, N1) parity mass pairs of independent regions."""
    a0, a1 = a
    b0, b1 = b
    c0 = np.convolve(a0, b0) + np.convolve(a1, b1)
    c1 = np.convolve(a0, b1) + np.convolve(a1, b0)
    s = max(c0.max(), c1.max())
    return c0 / s, c1 / s


def masses_by_component(k, q, masks, kinds, ys, errs):
    comp = components(k, masks)
    regions = sorted(set(int(c) for c in comp))
    relevant = [r for r in regions if (r & int(q)) or any(int(m) & r == int(m) and t == 3 for m, t in zip(masks, kinds))]
    acc = (np.ones(1), np.zeros(1))
    for r in relevant:
        acc = combine(acc, poly_masses(k, int(q) & r, masks, kinds, ys, errs, r))
    qcomp = sum(r for r in regions if r & int(q))
    return acc, sum(pop_py(r) for r in relevant), pop_py(qcomp)


def bern_eval(c, w):
    m = len(c) - 1
    i = np.arange(m + 1)
    return float(np.sum(c * w ** i * (1 - w) ** (m - i)))


def bern_integral(c):
    m = len(c) - 1
    return float(sum(ci / ((m + 1) * comb(m, i)) for i, ci in enumerate(c)))


def certify_nonneg(c, depth):
    """Sufficient test that sum c_i w^i (1-w)^(m-i) >= 0 on [0,1]: Bernstein
    coefficients nonnegative on every dyadic subinterval up to `depth`. Returns
    True (certified), False (a negative point value found) or None (undecided)."""
    m = len(c) - 1
    b = np.array([ci / comb(m, i) for i, ci in enumerate(c)])
    stack = [(b, 0)]
    undecided = False
    while stack:
        b, d = stack.pop()
        if b[0] < 0 or b[-1] < 0:
            return False
        if (b >= 0).all():
            continue
        if d >= depth:
            undecided = True
            continue
        left, right = [b[0]], [b[-1]]
        cur = b.copy()
        for _ in range(m):
            cur = (cur[:-1] + cur[1:]) / 2
            left.append(cur[0])
            right.append(cur[-1])
        stack += [(np.array(left), d + 1), (np.array(right[::-1]), d + 1)]
    return None if undecided else True


def credal_action(n0, n1, depth, penalty=4.):
    if certify_nonneg(n1 - penalty * n0, depth) is True:
        return 1
    if certify_nonneg(n0 - penalty * n1, depth) is True:
        return 0
    return -1


def brute_mass(k, q, masks, kinds, ys, errs, mix):
    """Independent full-joint masses (no polynomials, no components)."""
    xs = np.arange(1 << k)
    bits = (xs[:, None] >> np.arange(k)) & 1
    w = np.ones(len(xs))
    for m, t, y, e in zip(masks, kinds, ys, errs):
        sc = [a for a in range(k) if (int(m) >> a) & 1]
        c = bits[:, sc].sum(1)
        a = (c % 2) if t == 0 else (c == len(sc)) if t == 1 else (c >= 2) if t == 2 else (1 - mix) * (c % 2) + mix * (c >= 2)
        a = e + (1 - 2 * e) * np.asarray(a, float)
        w *= a if y else 1 - a
    par = bits[:, [a for a in range(k) if (int(q) >> a) & 1]].sum(1) % 2
    return float(w[par == 0].sum()), float(w[par == 1].sum())


def part_c(seeds, widths, n):
    rows, checks, rel = [], [], []
    gl_x, gl_w = np.polynomial.legendre.leggauss(16)
    gl_x, gl_w = (gl_x + 1) / 2, gl_w / 2
    grid = np.array(P['part_c']['bma_grid'])
    for k, seed in itertools.product(widths, seeds):
        q, mm, kk, yy, ee, truth, wtrue = fixture_w(k, seed + 900000 + 100 * k, n)
        out = {a: np.full(n, .5) for a in ('nominal_xor', 'plugin_half', 'bma_grid5', 'bma_exact', 'oracle_w')}
        acts = {a: np.full(n, -1) for a in ('credal_shared_d0', 'credal_shared_d6', 'pointwise_robust')}
        t_full = t_comp = 0.
        tracemalloc.start()
        for i in range(n):
            t = time.perf_counter()
            (n0, n1), relvars, qvars = masses_by_component(k, q[i], mm[i], kk[i], yy[i], ee[i])
            t_comp += time.perf_counter() - t
            rel.append(dict(k=k, relevant_vars=relvars, query_component_vars=qvars))
            if i < P['part_c']['full_joint_checks']:
                t = time.perf_counter()
                f0, f1 = poly_masses(k, q[i], mm[i], kk[i], yy[i], ee[i], (1 << k) - 1)
                t_full += time.perf_counter() - t
                s = max(f0.max(), f1.max())
                for w in (0., .3, .7, 1.):
                    pa = bern_eval(n1, w) / (bern_eval(n0, w) + bern_eval(n1, w))
                    pb = bern_eval(f1 / s, w) / (bern_eval(f0 / s, w) + bern_eval(f1 / s, w))
                    assert abs(pa - pb) < 1e-10
            tot = n0 + n1
            pw = lambda w: bern_eval(n1, w) / bern_eval(tot, w)
            out['nominal_xor'][i] = pw(0.)
            out['plugin_half'][i] = pw(.5)
            out['oracle_w'][i] = pw(wtrue[i])
            out['bma_grid5'][i] = sum(bern_eval(n1, g) for g in grid) / sum(bern_eval(tot, g) for g in grid)
            out['bma_exact'][i] = bern_integral(n1) / bern_integral(tot)
            acts['credal_shared_d0'][i] = credal_action(n0, n1, 0)
            acts['credal_shared_d6'][i] = credal_action(n0, n1, P['part_c']['subdivision_depth'])
            if i < P['part_c']['independent_checks']:
                z = [brute_mass(k, q[i], mm[i], kk[i], yy[i], ee[i], w) for w in gl_x]
                bma = sum(wt * zz[1] for wt, zz in zip(gl_w, z)) / sum(wt * (zz[0] + zz[1]) for wt, zz in zip(gl_w, z))
                o0, o1 = brute_mass(k, q[i], mm[i], kk[i], yy[i], ee[i], wtrue[i])
                worst = []
                for a in ('credal_shared_d0', 'credal_shared_d6'):
                    if acts[a][i] >= 0:
                        for w in np.linspace(0, 1, 101):
                            z0, z1 = brute_mass(k, q[i], mm[i], kk[i], yy[i], ee[i], w)
                            pp = z1 / (z0 + z1)
                            worst.append(pp if acts[a][i] == 1 else 1 - pp)
                checks.append(dict(k=k, seed=seed, episode=i, bma_delta=abs(bma - out['bma_exact'][i]),
                                   oracle_delta=abs(o1 / (o0 + o1) - out['oracle_w'][i]),
                                   credal_min_grid_confidence=min(worst) if worst else None))
                assert checks[-1]['bma_delta'] < 1e-9 and checks[-1]['oracle_delta'] < 1e-9
                assert not worst or min(worst) >= .8 - 1e-9
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        t = time.perf_counter()
        rob = ve(k, q, mm, kk, yy, ee, k, .5, True)
        t_rob = time.perf_counter() - t
        acts['pointwise_robust'] = np.where(rob[1] >= .8, 1, np.where(rob[2] <= .2, 0, -1))
        ref = out['oracle_w']
        for a, p in out.items():
            rows.append(dict(k=k, seed=seed, arm=a, n=n, **score(p, truth, ref), **calibration_stats(p, truth)))
        for a, act in acts.items():
            p = np.where(act == 1, 1., np.where(act == 0, 0., .5))
            r = dict(k=k, seed=seed, arm=a, n=n, **score(p, truth, ref))
            exp = np.where(act < 0, 0, np.where(act == 1, 5 * ref - 4, 1 - 5 * ref))
            r['negative_expected_utility_actions'] = int((exp < -1e-12).sum())
            rows.append(r)
        rows.append(dict(k=k, seed=seed, arm='_timing', component_poly_seconds=t_comp,
                         full_poly_seconds_first_checks=t_full, pointwise_robust_numba_seconds=t_rob,
                         tracemalloc_peak_bytes=int(peak_bytes)))
        print('C', k, seed, [(r['arm'], round(r.get('expected_utility', 0), 3), round(r.get('coverage', 0), 3)) for r in rows[-9:-1]], flush=True)
    return rows, checks, rel


def run():
    cal = json.loads((ROOT / 'calibration13d.json').read_text())
    alpha = cal['alpha']
    fixtures = {}
    for k, seed, sc in itertools.product(P['widths'], P['seeds'], S['scenarios']):
        fixtures[(k, seed, sc)] = fixture(k, seed + 700000 + 100 * k + 10000 * S['scenarios'].index(sc), P['n'], sc)
    a_rows, audit = part_a(fixtures, alpha)
    (ROOT / 'receipt13d_a.json').write_text(json.dumps(dict(protocol=P, alpha=alpha, audit=audit, environment=env(), rows=a_rows), separators=(',', ':')) + '\n')
    b_rows, widths = part_b(fixtures)
    (ROOT / 'receipt13d_b.json').write_text(json.dumps(dict(protocol=P, rows=b_rows, widths=widths), separators=(',', ':')) + '\n')
    c_rows, checks, rel = part_c(P['seeds'], P['part_c']['widths'], P['n'])
    (ROOT / 'receipt13d_c.json').write_text(json.dumps(dict(protocol=P, rows=c_rows, checks=checks, relevance=rel), separators=(',', ':')) + '\n')


if __name__ == '__main__':
    {'calibrate': calibrate, 'run': run}[sys.argv[1]]()

"""DCF-20: routing in a harder mesh world.

DCF-19B found that on the DCF-15 mesh exact inference changes BP's decision on only
1.6% of queries, so routing had about .002 utility to win. This generator keeps the
DCF-15 mesh (units, operators, slow merges, calibrated batch cost model) and changes
only the world, along three axes that stress BP:

  xor_frac   probability that a cross-unit sensor is XOR (parity) rather than AND/majority
  triangle   active unit pairs per epoch form a triangle (a loop at unit level) instead of
             two random pairs
  aligned    with probability 0.6 a query is the parity of one bit on each side of the
             most recent cross-unit sensor, which BP's cross-block independence read-out
             cannot represent
  noise      'tight' adds a near-noiseless sensor band

Stage `world` scores a grid of worlds on development seeds and selects the one with the
largest per-epoch routing headroom (epoch oracle minus the better fixed operator). The
routing stages then tune policies on development seeds of that world and test them on
fresh seeds. Policies:

  fixed_bp, fixed_exact      one operator always
  regime (tau)               DCF-19 rule: exact for the epoch if mean pending KL_hat >= tau
  probe (delta, phi)         DCF-19B: BP probe at epoch start; exact for the epoch if the
                             near-threshold fraction >= phi (probe charged on switch)
  escalate (delta)           BP for every query, then exact only for queries whose BP
                             posterior is within delta of 0.8 or 0.2 (second launch)
  epoch_oracle, query_oracle privileged upper bounds (per batch-epoch, per query)

Usage: python world20.py world | dev | test | report
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'dcf19_regime_routing'))
import regime19 as R19  # noqa: E402
import regime19b as R19B  # noqa: E402

M, CM = R19.M, R19.CM
P = json.loads((ROOT / 'PROTOCOL.json').read_text())


def episode(seed, rho, noise, xor_frac, triangle, aligned):
    rng = np.random.default_rng(seed)
    lo, hi = P['noise'][noise]
    x = int(rng.integers(1 << M.K))
    facs, queries = [], []
    pairs = [(a, b) for a in range(M.U) for b in range(a + 1, M.U)]
    last_cross = None
    for t in range(M.T):
        if t % M.M == 0:
            if triangle:
                tri = sorted(int(u) for u in rng.choice(M.U, 3, replace=False))
                act = [(tri[0], tri[1]), (tri[0], tri[2]), (tri[1], tri[2])]
            else:
                act = [pairs[i] for i in rng.choice(len(pairs), size=2, replace=False)]
            last_cross = None
        if rng.random() < rho:
            a, b = act[int(rng.integers(len(act)))]
            ar = int(rng.choice([2, 3]))
            na = int(rng.integers(1, ar))
            ba = list(rng.choice(4, na, replace=False) + 4 * a)
            bb = list(rng.choice(4, ar - na, replace=False) + 4 * b)
            bits = ba + bb
            kind = 0 if rng.random() < xor_frac else int(rng.choice([1, 2]))
            last_cross = (int(ba[0]), int(bb[0]))
        else:
            u = int(rng.integers(M.U))
            ar = int(rng.choice([1, 1, 2, 3]))
            bits = list(rng.choice(4, ar, replace=False) + 4 * u)
            kind = 0 if len(bits) == 1 else int(rng.integers(3))
        mask = sum(1 << int(v) for v in bits)
        e = 0.12 if len(bits) == 1 else float(rng.uniform(lo, hi))
        y = int(M.sensor(x, mask, kind)) ^ int(rng.random() < e)
        facs.append((mask, kind, y, e))
        r = rng.random()
        if aligned and last_cross is not None and r < 0.6:
            qb = list(last_cross)
        elif rng.random() < 0.6:
            u = int(rng.integers(M.U))
            qb = list(rng.choice(4, 2, replace=False) + 4 * u)
        else:
            a, b = act[int(rng.integers(len(act)))]
            qb = [int(rng.integers(4)) + 4 * a, int(rng.integers(4)) + 4 * b]
        queries.append(sum(1 << int(v) for v in qb))
    return dict(x=x, facs=facs, queries=queries)


def trace(seeds, rho, world, B):
    """Both operators on every query (regime19b trace format) for a world dict."""
    out = []
    for start in range(0, len(seeds), B):
        batch = seeds[start:start + B]
        eps = [episode(s, rho, world['noise'], world['xor_frac'], world['triangle'], world['aligned']) for s in batch]
        meshes = [M.Mesh() for _ in batch]
        steps = []
        for t in range(M.T):
            upd = 0
            for e, mesh in zip(eps, meshes):
                mesh.absorb(e['facs'][t])
                if t % M.M == M.M - 1:
                    mesh.cost_update += M.slow_merge(mesh, 'kl', M.P['merge_tau'])
                upd += mesh.cost_update
                mesh.cost_update = 0
            stat, stat_s = None, 0.0
            if t % M.M == 0:
                t0 = time.perf_counter()
                stat = float(np.mean([R19.pending_kl(m) for m in meshes]))
                stat_s = time.perf_counter() - t0
            qs = []
            for e, mesh in zip(eps, meshes):
                q = e['queries'][t]
                p_ref = M.parity_p(mesh.full, M.STATES, q)
                truth = bin(e['x'] & q).count('1') % 2
                res = {}
                for op in ('bp', 'exact'):
                    p, w = mesh.bp(q) if op == 'bp' else mesh.exact(q)
                    a = R19.decide(p)
                    eu = 0.0 if a < 0 else (5 * p_ref - 4 if a == 1 else 1 - 5 * p_ref)
                    res[op] = dict(p=float(p), eu=eu, work=int(w), act=int(a >= 0), wrong=int(a >= 0 and a != truth),
                                   realized=0.0 if a < 0 else (1.0 if a == truth else -4.0))
                qs.append(dict(oracle=max(0.0, 5 * p_ref - 4, 1 - 5 * p_ref), **res))
            steps.append(dict(t=t, update_work=upd, stat=stat, stat_seconds=stat_s, queries=qs))
        out.append(dict(seeds=batch, steps=steps))
    return out


def escalate_rows(tr, delta=None, oracle=False):
    """BP for all queries; exact for the selected subset (second kernel launch)."""
    rows = []
    for b in tr:
        nb = len(b['seeds'])
        for s in b['steps']:
            t0 = time.perf_counter()
            if oracle:
                sel = [q['exact']['eu'] > q['bp']['eu'] for q in s['queries']]
            else:
                sel = [R19B.near(q['bp']['p'], delta) for q in s['queries']]
            route = 0.0 if oracle else (time.perf_counter() - t0) / nb
            bp_work = sum(q['bp']['work'] for q in s['queries'])
            ex_work = sum(q['exact']['work'] for q, k in zip(s['queries'], sel) if k)
            step_s = M.op_seconds(CM, 'bp', bp_work) + M.op_seconds(CM, 'update', s['update_work'])
            if any(sel):
                step_s += M.op_seconds(CM, 'exact', ex_work)
            for q, k in zip(s['queries'], sel):
                r = q['exact'] if k else q['bp']
                rows.append(dict(op='exact' if k else 'bp', eu=r['eu'], regret=q['oracle'] - r['eu'], realized=r['realized'],
                                 act=r['act'], wrong=r['wrong'], work=r['work'], step_seconds_share=step_s / nb,
                                 route_seconds=route, launches=1 + int(any(sel))))
    return rows


def evaluate(tr, policy, params=None):
    if policy in ('fixed_bp', 'fixed_exact', 'epoch_oracle'):
        return R19.evaluate(tr, policy)
    if policy == 'regime':
        return R19.evaluate(tr, 'regime', params['tau'])
    if policy == 'probe':
        return R19B.evaluate(tr, 'probe', params)
    if policy == 'escalate':
        return escalate_rows(tr, params['delta'])
    if policy == 'query_oracle':
        return escalate_rows(tr, oracle=True)
    raise ValueError(policy)


def obj(tr, policy, params=None):
    return R19.summary(evaluate(tr, policy, params))['objective_total']


def decision_change_rate(tr):
    n = ch = 0
    for b in tr:
        for s in b['steps']:
            for q in s['queries']:
                n += 1
                ch += R19.decide(q['bp']['p']) != R19.decide(q['exact']['p'])
    return ch / n


def worlds():
    out = []
    for noise in P['world_grid']['noise']:
        for xf in P['world_grid']['xor_frac']:
            for tri in P['world_grid']['triangle']:
                for al in P['world_grid']['aligned']:
                    out.append(dict(noise=noise, xor_frac=xf, triangle=tri, aligned=al))
    return out


def save_trace(name, obj):
    """Traces live in this directory (DCF-20 first used regime19's helper, which writes to dcf19_regime_routing)."""
    import gzip
    with gzip.GzipFile(ROOT / name, 'wb', mtime=0) as f:
        f.write(json.dumps(obj, separators=(',', ':')).encode())


def load_trace(name):
    import gzip
    return json.loads(gzip.open(ROOT / name, 'rt').read())


def seeds(base, ci, n):
    return [base + 10000 * ci + i for i in range(n)]


def stage_world():
    rows = []
    for wi, w in enumerate(worlds()):
        tr = trace(seeds(P['seed_base']['world'], wi, P['world_episodes']), P['world_rho'], w, P['batch'])
        fb, fe, eo = obj(tr, 'fixed_bp'), obj(tr, 'fixed_exact'), obj(tr, 'epoch_oracle')
        rows.append(dict(world=w, fixed_bp=fb, fixed_exact=fe, epoch_oracle=eo, headroom=eo - max(fb, fe),
                         decision_change_rate=decision_change_rate(tr)))
        print('world', w, round(rows[-1]['headroom'], 5), round(rows[-1]['decision_change_rate'], 4), flush=True)
    best = max(rows, key=lambda r: r['headroom'])
    (ROOT / 'world20.json').write_bytes((json.dumps(dict(selected=best['world'], rows=rows), indent=1) + '\n').encode())
    print('selected', best)


def selected_world():
    return json.loads((ROOT / 'world20.json').read_text())['selected']


def conditions(split):
    return P['rho'][split]


def candidates():
    out = [('regime', dict(tau=t)) for t in P['tau_grid']]
    out += [('probe', dict(delta=d, phi=f)) for d in P['delta_grid'] for f in P['phi_grid']]
    out += [('escalate', dict(delta=d)) for d in P['delta_grid']]
    return out


def stage_dev():
    w = selected_world()
    traces = []
    for ci, rho in enumerate(conditions('dev')):
        traces.append(dict(rho=rho, trace=trace(seeds(P['seed_base']['dev'], ci, P['episodes']), rho, w, P['batch'])))
        print('dev', rho, flush=True)
    save_trace('trace20_dev.json.gz', traces)
    scores = [dict(policy=k, params=p, mean_objective_total=float(np.mean([obj(c['trace'], k, p) for c in traces])))
              for k, p in candidates()]
    fixed = {k: float(np.mean([obj(c['trace'], k) for c in traces])) for k in ('fixed_bp', 'fixed_exact', 'epoch_oracle', 'query_oracle')}
    best = max(scores, key=lambda s: s['mean_objective_total'])
    strongest_fixed = max(('fixed_bp', 'fixed_exact'), key=lambda k: fixed[k])
    (ROOT / 'selection20.json').write_bytes((json.dumps(dict(world=w, selected=best, strongest_fixed=strongest_fixed,
                                                             reference=fixed, scores=scores), indent=1) + '\n').encode())
    print('selected', best, fixed)


def stage_test():
    w = selected_world()
    traces = []
    for ci, rho in enumerate(conditions('test')):
        traces.append(dict(rho=rho, trace=trace(seeds(P['seed_base']['test'], ci, P['episodes']), rho, w, P['batch'])))
        print('test', rho, flush=True)
    save_trace('trace20_test.json.gz', traces)


def stage_report():
    sel = json.loads((ROOT / 'selection20.json').read_text())
    test = load_trace('trace20_test.json.gz')
    names = ['fixed_bp', 'fixed_exact', 'selected', 'epoch_oracle', 'query_oracle']
    table, dcr = {}, {}
    for c in test:
        for n in names:
            rows = evaluate(c['trace'], sel['selected']['policy'], sel['selected']['params']) if n == 'selected' else evaluate(c['trace'], n)
            table[f"{c['rho']}/{n}"] = R19.summary(rows)
        dcr[str(c['rho'])] = decision_change_rate(c['trace'])
    conds = [str(c['rho']) for c in test]
    o = lambda c, n: table[f'{c}/{n}']['objective_total']
    sf = sel['strongest_fixed']
    w1 = sum(o(c, 'selected') > o(c, sf) + 1e-12 for c in conds)
    l1 = sum(o(c, 'selected') < o(c, sf) - 1e-12 for c in conds)
    wb = sum(o(c, 'selected') > max(o(c, 'fixed_bp'), o(c, 'fixed_exact')) + 1e-12 for c in conds)
    head = {c: o(c, 'epoch_oracle') - max(o(c, 'fixed_bp'), o(c, 'fixed_exact')) for c in conds}
    gates = dict(
        W1=dict(rule=P['gates']['W1'], mean_decision_change_rate=float(np.mean(list(dcr.values()))),
                mean_epoch_headroom=float(np.mean(list(head.values()))),
                passed=float(np.mean(list(dcr.values()))) >= .05 and float(np.mean(list(head.values()))) >= .01),
        R1=dict(rule=P['gates']['R1'], strongest_fixed=sf, wins=w1, losses=l1, passed=w1 > l1),
        R2=dict(rule=P['gates']['R2'], conditions_beating_better_fixed=wb, conditions=len(conds), passed=2 * wb >= len(conds)))
    res = dict(world=sel['world'], selected=sel['selected'], gates=gates, decision_change_rate=dcr, headroom=head, table=table,
               mean={n: {k: float(np.mean([table[f'{c}/{n}'][k] for c in conds])) for k in ('eu', 'objective_total', 'kernel_ms', 'route_ms', 'exact_fraction', 'regret', 'coverage', 'selective_risk')} for n in names})
    (ROOT / 'results20.json').write_bytes((json.dumps(res, indent=1) + '\n').encode())
    for c in conds:
        print(c, {n: (round(o(c, n), 4), round(table[f'{c}/{n}']['exact_fraction'], 2)) for n in names}, 'dcr', round(dcr[c], 3))
    print({n: round(v['objective_total'], 5) for n, v in res['mean'].items()})
    print(gates)


if __name__ == '__main__':
    {'world': stage_world, 'dev': stage_dev, 'test': stage_test, 'report': stage_report}[sys.argv[1]]()

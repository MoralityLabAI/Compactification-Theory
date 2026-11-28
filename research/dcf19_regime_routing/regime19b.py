"""DCF-19B: decision-margin signals for regime-level routing.

DCF-19's statistic (pending discarded correlation) did not separate the batch-epochs
where exact inference pays. Exact only changes utility when it changes a decision, and
that can only happen for queries whose approximate posterior is near a threshold.
Two signals measure that directly:

  probe:  at each epoch start, run BP on the batch's current queries and take the
          fraction with |p_bp - 0.8| < delta or |p_bp - 0.2| < delta. If it is >= phi
          the epoch (including this step) uses exact; the probe's BP kernel is then an
          extra charge. If BP is kept, the probe answers the step at no extra cost.
  lagged: the same fraction over the posteriors actually returned during the previous
          epoch (free; uses only answered queries).

Baselines: fixed BP, fixed exact, DCF-19's regime rule (tau = 2, frozen), and the
privileged epoch oracle. Accounting is DCF-19's (one kernel launch per step, DCF-15
calibrated batch model).

Usage: python regime19b.py dev | test | report
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import regime19 as R19  # noqa: E402

M, CM = R19.M, R19.CM
P = json.loads((ROOT / 'PROTOCOL19B.json').read_text())


def near(p, delta):
    return abs(p - .8) < delta or abs(p - .2) < delta


def trace(seeds, rho, noise, B):
    """Both operators on every query, recording posteriors, utilities and work."""
    out = []
    for start in range(0, len(seeds), B):
        batch = seeds[start:start + B]
        eps = [M.episode(s, rho, noise) for s in batch]
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


def evaluate(tr, policy, params=None):
    """Rows for a policy. params: tau (regime19), delta/phi (probe, lagged)."""
    if policy in ('fixed_bp', 'fixed_exact', 'epoch_oracle'):
        return R19.evaluate(tr, policy)
    if policy == 'regime19':
        return R19.evaluate(tr, 'regime', params['tau'])
    rows = []
    for b in tr:
        nb = len(b['seeds'])
        op, prev_ps = 'bp', []
        probe_extra_work, route_share = 0, 0.0
        for s in b['steps']:
            t = s['t']
            probe_extra_work = 0
            if t % M.M == 0:
                t0 = time.perf_counter()
                if policy == 'probe':
                    ps = [q['bp']['p'] for q in s['queries']]
                    frac = float(np.mean([near(p, params['delta']) for p in ps]))
                    op = 'exact' if frac >= params['phi'] else 'bp'
                    if op == 'exact':
                        probe_extra_work = sum(q['bp']['work'] for q in s['queries'])
                elif policy == 'lagged':
                    if t == 0:
                        op = 'bp'
                    else:
                        frac = float(np.mean([near(p, params['delta']) for p in prev_ps]))
                        op = 'exact' if frac >= params['phi'] else 'bp'
                    prev_ps = []
                else:
                    raise ValueError(policy)
                route_share = (time.perf_counter() - t0) / (M.M * nb)
            work = sum(q[op]['work'] for q in s['queries'])
            step_s = M.op_seconds(CM, op, work) + M.op_seconds(CM, 'update', s['update_work'])
            if probe_extra_work:
                step_s += M.op_seconds(CM, 'bp', probe_extra_work)
            for q in s['queries']:
                r = q[op]
                prev_ps.append(r['p'])
                rows.append(dict(op=op, eu=r['eu'], regret=q['oracle'] - r['eu'], realized=r['realized'], act=r['act'],
                                 wrong=r['wrong'], work=r['work'], step_seconds_share=step_s / nb, route_seconds=route_share,
                                 launches=1 + int(probe_extra_work > 0)))
    return rows


def grid():
    out = [('regime19', dict(tau=P['regime19_tau']))]
    for kind in ('probe', 'lagged'):
        for d in P['delta_grid']:
            for f in P['phi_grid']:
                out.append((kind, dict(delta=d, phi=f)))
    return out


def conditions(split):
    return [(rho, noise) for rho in P['rho'][split] for noise in ('low', 'high')]


def stage_dev():
    traces = []
    for ci, (rho, noise) in enumerate(conditions('dev')):
        traces.append(dict(rho=rho, noise=noise, trace=trace(R19.seeds_for(P['seed_base']['dev'], ci, P['episodes']), rho, noise, P['batch'])))
        print('dev', rho, noise, flush=True)
    R19.save_trace('trace19b_dev.json.gz', traces)
    scores = []
    for kind, params in grid():
        v = float(np.mean([R19.summary(evaluate(c['trace'], kind, params))['objective_total'] for c in traces]))
        scores.append(dict(policy=kind, params=params, mean_objective_total=v))
    best = max(scores, key=lambda s: s['mean_objective_total'])
    fixed = {p: float(np.mean([R19.summary(evaluate(c['trace'], p))['objective_total'] for c in traces])) for p in ('fixed_bp', 'fixed_exact', 'epoch_oracle')}
    (ROOT / 'selection19b.json').write_text(json.dumps(dict(selected=best, scores=scores, reference=fixed), indent=1) + '\n')
    print('selected', best, fixed)


def stage_test():
    traces = []
    for ci, (rho, noise) in enumerate(conditions('test')):
        traces.append(dict(rho=rho, noise=noise, trace=trace(R19.seeds_for(P['seed_base']['test'], ci, P['episodes']), rho, noise, P['batch'])))
        print('test', rho, noise, flush=True)
    R19.save_trace('trace19b_test.json.gz', traces)


def stage_report():
    sel = json.loads((ROOT / 'selection19b.json').read_text())['selected']
    test = R19.load_trace('trace19b_test.json.gz')
    pols = [('fixed_bp', None), ('fixed_exact', None), ('regime19', dict(tau=P['regime19_tau'])), ('selected', None), ('epoch_oracle', None)]
    table = {}
    for c in test:
        key = f"{c['rho']}/{c['noise']}"
        for name, params in pols:
            if name == 'selected':
                rows = evaluate(c['trace'], sel['policy'], sel['params'])
            else:
                rows = evaluate(c['trace'], name, params)
            table[f'{key}/{name}'] = R19.summary(rows)
    conds = [f"{c['rho']}/{c['noise']}" for c in test]
    obj = lambda c, p: table[f'{c}/{p}']['objective_total']
    g = {}
    for gname, other in (('R1', 'fixed_bp'), ('R3', 'regime19')):
        w = sum(obj(c, 'selected') > obj(c, other) + 1e-12 for c in conds)
        lo = sum(obj(c, 'selected') < obj(c, other) - 1e-12 for c in conds)
        g[gname] = dict(rule=P['gates'][gname], wins=w, losses=lo, passed=w > lo)
    r2 = sum(obj(c, 'selected') >= max(obj(c, 'fixed_bp'), obj(c, 'fixed_exact')) - P['r2_tolerance'] for c in conds)
    g['R2'] = dict(rule=P['gates']['R2'], conditions_within=r2, passed=r2 >= 4)
    names = [n for n, _ in pols]
    res = dict(selected=sel, gates=g, table=table,
               mean={n: {k: float(np.mean([table[f'{c}/{n}'][k] for c in conds])) for k in ('eu', 'objective_total', 'kernel_ms', 'route_ms', 'exact_fraction', 'regret', 'coverage', 'selective_risk')} for n in names})
    (ROOT / 'results19b.json').write_text(json.dumps(res, indent=1) + '\n')
    for c in conds:
        print(c, {n: (round(obj(c, n), 4), round(table[f'{c}/{n}']['exact_fraction'], 2)) for n in names})
    print({n: round(v['objective_total'], 5) for n, v in res['mean'].items()})
    print(g)


if __name__ == '__main__':
    {'dev': stage_dev, 'test': stage_test, 'report': stage_report}[sys.argv[1]]()

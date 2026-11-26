"""DCF-19: regime-level routing for the DCF-15 mesh.

One operator (cluster BP or exact-over-relevant-component) is chosen per batch per
slow epoch from a single batch statistic, the mean pending discarded-correlation
estimate S = sum over pending cross-unit sensors of (ln 2 - H(e)). Every query in
the batch-epoch uses that operator, so each step launches exactly one inference
kernel and routing costs one pass over the pending factors per episode per epoch.

Operator choice never changes mesh state (absorption, slow merges and block tables
are identical under BP and exact), so a trace that executes both operators on every
query determines the outcome of every policy exactly. Kernel time is the DCF-15
calibrated batch model, charged once per step: a_op + b_op * (summed work). Routing
time is the measured wall time of computing S.

Usage: python regime19.py dev | test | sweep | report
"""
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'dcf15_ldt_mesh_routing'))
import mesh15 as M  # noqa: E402

P = json.loads((ROOT / 'PROTOCOL.json').read_text())
CM = M.cost_fit()


def pending_kl(mesh):
    return sum(M.kl_bound(f[3]) for f, _ in mesh.cross())


def decide(p):
    return -1 if p is None else 1 if p >= .8 else 0 if p <= .2 else -1


def trace(seeds, rho, noise, B):
    """Execute both operators on every query; record per batch-epoch statistics."""
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
                stat = float(np.mean([pending_kl(m) for m in meshes]))
                stat_s = time.perf_counter() - t0
            qs = []
            for e, mesh in zip(eps, meshes):
                q = e['queries'][t]
                p_ref = M.parity_p(mesh.full, M.STATES, q)
                truth = bin(e['x'] & q).count('1') % 2
                res = {}
                for op in ('bp', 'exact'):
                    p, w = mesh.bp(q) if op == 'bp' else mesh.exact(q)
                    a = decide(p)
                    eu = 0.0 if a < 0 else (5 * p_ref - 4 if a == 1 else 1 - 5 * p_ref)
                    res[op] = dict(eu=eu, work=int(w), act=int(a >= 0), wrong=int(a >= 0 and a != truth),
                                   realized=0.0 if a < 0 else (1.0 if a == truth else -4.0))
                qs.append(dict(oracle=max(0.0, 5 * p_ref - 4, 1 - 5 * p_ref), **res))
            steps.append(dict(t=t, update_work=upd, stat=stat, stat_seconds=stat_s, queries=qs))
        out.append(dict(seeds=batch, steps=steps))
    return out


def evaluate(tr, policy, tau=None):
    """Per-query rows for a policy applied to a trace."""
    rows, lam = [], P['lambda']
    for b in tr:
        nb = len(b['seeds'])
        op, route_share = 'bp', 0.0
        epoch_ops = {}
        if policy == 'epoch_oracle':
            for t0 in range(0, M.T, M.M):
                vals = {}
                for o in ('bp', 'exact'):
                    v = 0.0
                    for s in b['steps'][t0:t0 + M.M]:
                        v += sum(q[o]['eu'] for q in s['queries'])
                        v -= lam * 1000 * M.op_seconds(CM, o, sum(q[o]['work'] for q in s['queries']))
                    vals[o] = v
                epoch_ops[t0] = max(vals, key=vals.get)
        for s in b['steps']:
            t = s['t']
            if t % M.M == 0:
                if policy == 'regime':
                    op = 'bp' if t == 0 else ('exact' if s['stat'] >= tau else 'bp')
                    route_share = s['stat_seconds'] / (M.M * nb)
                elif policy == 'epoch_oracle':
                    op = epoch_ops[t]
                    route_share = 0.0
                else:
                    op = policy[len('fixed_'):]
                    route_share = 0.0
            work = sum(q[op]['work'] for q in s['queries'])
            step_s = M.op_seconds(CM, op, work) + M.op_seconds(CM, 'update', s['update_work'])
            for q in s['queries']:
                r = q[op]
                rows.append(dict(op=op, eu=r['eu'], regret=q['oracle'] - r['eu'], realized=r['realized'], act=r['act'],
                                 wrong=r['wrong'], work=r['work'], step_seconds_share=step_s / nb, route_seconds=route_share,
                                 launches=1))
    return rows


def summary(rows):
    s = M.summarize(rows)
    s['exact_fraction'] = float(np.mean([r['op'] == 'exact' for r in rows]))
    return s


def conditions(split):
    return [(rho, noise) for rho in P['rho'][split] for noise in ('low', 'high')]


def seeds_for(base, ci, n):
    return [base + 10000 * ci + i for i in range(n)]


def save_trace(name, obj):
    import gzip
    raw = json.dumps(obj, separators=(',', ':')).encode()
    with gzip.GzipFile(ROOT / name, 'wb', mtime=0) as f:
        f.write(raw)


def load_trace(name):
    import gzip
    return json.loads(gzip.open(ROOT / name, 'rt').read())


def stage_dev():
    traces = []
    for ci, (rho, noise) in enumerate(conditions('dev')):
        tr = trace(seeds_for(P['seed_base']['dev'], ci, P['episodes']), rho, noise, P['batch'])
        traces.append(dict(rho=rho, noise=noise, trace=tr))
        print('dev', rho, noise, flush=True)
    save_trace('trace19_dev.json.gz', traces)
    grid = {}
    for tau in P['tau_grid']:
        vals = [summary(evaluate(c['trace'], 'regime', tau))['objective_total'] for c in traces]
        grid[str(tau)] = dict(mean_objective_total=float(np.mean(vals)), per_condition=vals)
    best = max(P['tau_grid'], key=lambda t: (grid[str(t)]['mean_objective_total'], -t))
    fixed = {p: float(np.mean([summary(evaluate(c['trace'], p))['objective_total'] for c in traces])) for p in ('fixed_bp', 'fixed_exact')}
    sel = dict(tau=best, grid=grid, fixed_dev_objective=fixed, environment=dict(python=platform.python_version(), numpy=np.__version__))
    (ROOT / 'selection19.json').write_text(json.dumps(sel, indent=1) + '\n')
    print('tau*', best, {k: round(v['mean_objective_total'], 5) for k, v in grid.items()}, fixed)


def stage_test():
    traces = []
    for ci, (rho, noise) in enumerate(conditions('test')):
        tr = trace(seeds_for(P['seed_base']['test'], ci, P['episodes']), rho, noise, P['batch'])
        traces.append(dict(rho=rho, noise=noise, trace=tr))
        print('test', rho, noise, flush=True)
    save_trace('trace19_test.json.gz', traces)


def stage_sweep():
    rho, noise = P['sweep']['condition']
    out = []
    for B in P['batch_sizes']:
        tr = trace(seeds_for(P['seed_base']['sweep'], 0, P['sweep']['episodes']), rho, noise, B)
        out.append(dict(B=B, trace=tr))
        print('sweep', B, flush=True)
    save_trace('trace19_sweep.json.gz', out)


def stage_report():
    tau = json.loads((ROOT / 'selection19.json').read_text())['tau']
    pols = ['fixed_bp', 'fixed_exact', 'regime', 'epoch_oracle']
    test = load_trace('trace19_test.json.gz')
    table = {}
    for c in test:
        for p in pols:
            table[f"{c['rho']}/{c['noise']}/{p}"] = summary(evaluate(c['trace'], p, tau))
    conds = [f"{c['rho']}/{c['noise']}" for c in test]
    obj = lambda c, p: table[f'{c}/{p}']['objective_total']
    r1w = sum(obj(c, 'regime') > obj(c, 'fixed_bp') + 1e-12 for c in conds)
    r1l = sum(obj(c, 'regime') < obj(c, 'fixed_bp') - 1e-12 for c in conds)
    r2 = sum(obj(c, 'regime') >= max(obj(c, 'fixed_bp'), obj(c, 'fixed_exact')) - P['r2_tolerance'] for c in conds)
    sweep = []
    for s in load_trace('trace19_sweep.json.gz'):
        sweep.append(dict(B=s['B'], **{p: summary(evaluate(s['trace'], p, tau)) for p in pols}))
    res = dict(tau=tau, table=table,
               gates=dict(R1=dict(rule=P['gates']['R1'], wins=r1w, losses=r1l, passed=r1w > r1l),
                          R2=dict(rule=P['gates']['R2'], conditions_within=r2, required=4, passed=r2 >= 4)),
               mean={p: {k: float(np.mean([table[f'{c}/{p}'][k] for c in conds])) for k in ('eu', 'objective_total', 'kernel_ms', 'route_ms', 'exact_fraction', 'regret', 'coverage', 'selective_risk')} for p in pols},
               sweep=sweep)
    (ROOT / 'results19.json').write_text(json.dumps(res, indent=1) + '\n')
    for c in conds:
        print(c, {p: (round(table[f'{c}/{p}']['eu'], 4), round(obj(c, p), 4), round(table[f'{c}/{p}']['exact_fraction'], 2)) for p in pols})
    print(res['gates'])


if __name__ == '__main__':
    {'dev': stage_dev, 'test': stage_test, 'sweep': stage_sweep, 'report': stage_report}[sys.argv[1]]()

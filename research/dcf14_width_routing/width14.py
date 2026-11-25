"""DCF-14: computational width as a routing signal.

Stages (run in order; PROTOCOL.json fixes seeds and gates):
  python width14.py dev      run every method on development families
  python width14.py fit      fit cost models, the width-rule threshold and learned selectors on dev only
  python width14.py test     run every method on held-out families/sizes (after the fit is committed)
  python width14.py report   evaluate hypothesis H1 and all routing policies; writes results14.json
"""
import json
import math
import os
import pickle
import platform
import sys
import time
from multiprocessing import Pool
from pathlib import Path

os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import fg  # noqa: E402
from gen import make_problem  # noqa: E402

P = json.loads((ROOT / 'PROTOCOL.json').read_text())
EXACT_CAP = P['exact_max_union']
FEATS = ['log2_n', 'comp_vars', 'comp_factors', 'width_min_fill', 'width_min_degree', 'width_lower', 'max_arity',
         'mean_arity', 'query_size', 'log2_ve_ops', 'log2_enum_ops', 'cutset', 'mean_noise', 'gf2_applicable']


def specs(split):
    out = []
    for block in P['splits'][split]:
        for fam in block['families']:
            for n in block['sizes']:
                for s in range(block['seeds']):
                    seed = block['seed_base'] + 1000 * s + n
                    out.append(dict(family=fam, n=n, seed=seed, noise=('low', 'high')[s % 2], query_size=1 + s % 3,
                                    misspec=(s % 3 == 2) and fam != 'xor_noiseless'))
    return out


def timed(f, *a, **k):
    t = time.perf_counter()
    r = f(*a, **k)
    r['seconds'] = time.perf_counter() - t
    return r


def run_instance(spec):
    prob = make_problem(spec['family'], spec['n'], spec['seed'], spec['noise'], spec['query_size'], spec['misspec'])
    t = time.perf_counter()
    feat = fg.features(prob)
    feat_seconds = time.perf_counter() - t
    order = feat.pop('_order')
    feat.pop('_cut')
    feat['log2_n'] = math.log2(prob.n)
    comp, variables = fg.query_component(prob)
    sym = fg.symbolic_ve(comp.scopes, order, comp.query)
    feat['max_union_min_fill'] = sym['max_union']
    rec = dict(spec=spec, features=feat, feature_seconds=feat_seconds, truth=prob.truth_parity(), methods={})
    # reference posterior under the generating model, with an independent second exact path
    ref_prob = prob.with_tables(prob.true_tables) if prob.true_tables is not None else prob
    ref = None
    if sym['max_union'] <= EXACT_CAP:
        r = fg.variable_elimination(ref_prob, order=order)
        ref = r['p']
        checks = []
        if feat['comp_vars'] <= P['enum_max_vars']:
            checks.append(('enumeration', fg.enumerate_exact(ref_prob, max_vars=P['enum_max_vars'])['p']))
        cs = fg.cutset_exact(ref_prob, max_cutset=P['cutset_max'])
        if cs['p'] is not None:
            checks.append(('cutset', cs['p']))
        vb = fg.ve_broadcast(ref_prob, max_union=EXACT_CAP + 2)
        if vb['p'] is not None:
            checks.append(('ve_broadcast', vb['p']))
        rec['reference_checks'] = [(name, abs(p - ref)) for name, p in checks]
        assert all(d < 1e-8 for _, d in rec['reference_checks']), rec['reference_checks']
    rec['reference'] = ref
    rec['verified'] = ref is not None and bool(rec.get('reference_checks'))
    m = rec['methods']
    if sym['max_union'] <= EXACT_CAP:
        for h in ('min_fill', 'min_degree', 'weighted_min_fill'):
            o, w = fg.elimination_order(variables, comp.scopes, comp.query, h)
            s2 = fg.symbolic_ve(comp.scopes, o, comp.query)
            if s2['max_union'] <= EXACT_CAP:
                m['ve_' + h] = timed(fg.variable_elimination, prob, order=o)
                m['ve_' + h]['width'] = w
    if feat['comp_vars'] <= P['enum_max_vars']:
        m['enumeration'] = timed(fg.enumerate_exact, prob, max_vars=P['enum_max_vars'])
    cs = timed(fg.cutset_exact, prob, max_cutset=P['cutset_max'])
    if cs['p'] is not None:
        m['cutset'] = cs
    if feat['gf2_applicable']:
        m['gf2'] = timed(fg.gf2_exact, prob)
    for B in P['budgets']:
        m[f'bp_B{B}'] = timed(fg.loopy_bp, prob, op_budget=2 ** B)
        per = max(1, sum(len(s) for s in comp.scopes))
        npart = min(P['max_particles'], 2 ** B // per)
        if npart >= 16:
            m[f'particle_B{B}'] = timed(fg.likelihood_weighting, prob, n_particles=npart, seed=spec['seed'])
        ib = [i for i in range(1, 21) if fg.mini_bucket_ops(prob, i, order) <= 2 ** B]
        if ib:
            m[f'mbe_B{B}'] = timed(fg.mini_bucket, prob, max(ib), order)
            m[f'mbe_B{B}']['ibound'] = max(ib)
    for v in m.values():
        for key in [k for k in v if isinstance(v[k], (np.floating, np.integer))]:
            v[key] = v[key].item()
    return rec


def stage_run(split):
    sp = specs(split)
    with Pool(P['workers']) as pool:
        recs = pool.map(run_instance, sp, chunksize=1)
    out = dict(protocol=P, split=split, environment=dict(python=platform.python_version(), numpy=np.__version__,
               workers=P['workers'], note='wall times measured with parallel worker processes'), records=recs)
    (ROOT / f'receipt14_{split}.json').write_text(json.dumps(out, separators=(',', ':')) + '\n')
    print(split, len(recs), 'instances;', sum(r['verified'] for r in recs), 'verified')


# ------------------------------------------------------------------ routing

EXACT_METHODS = ('ve_min_fill', 'enumeration', 'cutset', 'gf2')


def options(rec, B, penalty):
    """Every routing option available at budget 2^B: (name, p, ops, interval)."""
    m = rec['methods']
    ops_cap = 2 ** B
    out = {'abstain': (None, 0, None)}
    exact = [(m[k]['ops'], k) for k in EXACT_METHODS if k in m and m[k]['ops'] <= ops_cap]
    if exact:
        ops, k = min(exact)
        out['exact'] = (m[k]['p'], ops, None)
        out['exact_method'] = k
    for kind in ('bp', 'mbe', 'particle'):
        k = f'{kind}_B{B}'
        if k in m and m[k]['p'] is not None:
            out[kind] = (m[k]['p'], m[k]['ops'], None)
            if kind == 'particle':
                hw = m[k]['half_width']
                out['particle_ci'] = (m[k]['p'], m[k]['ops'], (m[k]['p'] - hw, m[k]['p'] + hw))
    return out


def option_value(rec, opt, penalty, use_truth=False):
    p, ops, iv = opt
    a = fg.decide(p, penalty) if iv is None else fg.decide(p, penalty, lo=iv[0], hi=iv[1])
    if use_truth:
        return (0.0 if a < 0 else 1.0 if a == rec['truth'] else -penalty), a
    return fg.expected_utility(a, rec['reference'], penalty), a


OPTION_NAMES = ['exact', 'bp', 'mbe', 'particle', 'particle_ci', 'abstain']


def feature_vector(rec, B, penalty):
    f = rec['features']
    return [f[k] for k in FEATS] + [B, penalty]


def label_best(rec, B, penalty, lam):
    opts = options(rec, B, penalty)
    best, bestv = 'abstain', 0.0
    for name in OPTION_NAMES:
        if name in opts:
            u, _ = option_value(rec, opts[name], penalty)
            v = u - lam * opts[name][1] / 2 ** B
            if v > bestv + 1e-12:
                best, bestv = name, v
    return best


def stage_fit():
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier
    dev = json.loads((ROOT / 'receipt14_dev.json').read_text())['records']
    dev = [r for r in dev if r['verified']]
    lam = P['lambda']
    X, y = [], []
    for r in dev:
        for B in P['budgets']:
            for pen in P['penalties']:
                X.append(feature_vector(r, B, pen))
                y.append(label_best(r, B, pen, lam))
    X = np.array(X, float)
    t = time.perf_counter()
    tree = DecisionTreeClassifier(max_depth=P['tree_depth'], min_samples_leaf=5, random_state=0).fit(X, y)
    tree_s = time.perf_counter() - t
    scaler = StandardScaler().fit(X)
    t = time.perf_counter()
    mlp = MLPClassifier(hidden_layer_sizes=tuple(P['mlp_hidden']), max_iter=2000, random_state=0).fit(scaler.transform(X), y)
    mlp_s = time.perf_counter() - t
    # width-rule threshold per budget, tuned on dev mean utility
    wstar = {}
    for B in P['budgets']:
        best = None
        for W in range(1, 25):
            u = np.mean([policy_value(r, B, pen, 'width_rule', dict(W={str(B): W}))[0] for r in dev for pen in P['penalties']])
            if best is None or u > best[1] + 1e-12:
                best = (W, u)
        wstar[str(B)] = best[0]
    # single-feature log-cost models for H1
    cost = {}
    for target in ('best_exact_ops', 've_ops'):
        ys, xs = [], {k: [] for k in P['h1_features']}
        for r in dev:
            c = cost_target(r, target)
            if c is None:
                continue
            ys.append(math.log2(c))
            for k in P['h1_features']:
                xs[k].append(r['features'][k])
        cost[target] = {k: list(np.polyfit(xs[k], ys, 1)) for k in P['h1_features']}
    model = dict(tree=tree, mlp=mlp, scaler=scaler, W=wstar, cost=cost, train_seconds=dict(tree=tree_s, mlp=mlp_s),
                 n_train=len(y), label_counts={k: y.count(k) for k in set(y)}, mlp_params=int(sum(a.size for a in mlp.coefs_) + sum(b.size for b in mlp.intercepts_)))
    (ROOT / 'selector14.pkl').write_bytes(pickle.dumps(model))
    (ROOT / 'selector14_summary.json').write_text(json.dumps(dict(W=wstar, cost=cost, train_seconds=model['train_seconds'],
                                                                 n_train=len(y), label_counts=model['label_counts'],
                                                                 mlp_params=model['mlp_params'], tree_nodes=int(tree.tree_.node_count)), indent=1) + '\n')
    print('fit', model['label_counts'], wstar)


def cost_target(r, target):
    m = r['methods']
    if target == 've_ops':
        return max(1, m['ve_min_fill']['ops']) if 've_min_fill' in m else None
    c = [m[k]['ops'] for k in EXACT_METHODS if k in m]
    return max(1, min(c)) if c else None


def policy_value(rec, B, penalty, policy, model, rng=None, use_truth=False):
    """Returns (utility, ops, chosen option, routing seconds)."""
    opts = options(rec, B, penalty)
    route_s = rec['feature_seconds']
    if policy == 'exact_or_abstain':
        choice = 'exact' if 'exact' in opts else 'abstain'
    elif policy.startswith('exact_else_'):
        alt = policy[len('exact_else_'):]
        choice = 'exact' if 'exact' in opts else alt if alt in opts else 'abstain'
    elif policy == 'width_rule':
        W = model['W'][str(B)]
        if rec['features']['width_min_fill'] <= W:
            choice = 'exact' if 'exact' in opts else 'abstain'
        else:
            choice = 'bp' if 'bp' in opts else 'abstain'
    elif policy.startswith('always_'):
        choice = policy[len('always_'):]
        choice = choice if choice in opts else 'abstain'
        route_s = 0.0
    elif policy in ('learned_tree', 'learned_mlp'):
        x = np.array([feature_vector(rec, B, penalty)], float)
        t = time.perf_counter()
        if policy == 'learned_tree':
            pr = model['tree'].predict_proba(x)[0]
            classes = model['tree'].classes_
        else:
            pr = model['mlp'].predict_proba(model['scaler'].transform(x))[0]
            classes = model['mlp'].classes_
        route_s += time.perf_counter() - t
        ranked = [classes[i] for i in np.argsort(-pr)]
        choice = next((c for c in ranked if c in opts), 'abstain')
    elif policy == 'random':
        choice = rng.choice([c for c in OPTION_NAMES if c in opts])
    elif policy == 'oracle':
        choice = label_best(rec, B, penalty, P['lambda'])
    else:
        raise ValueError(policy)
    u, a = option_value(rec, opts[choice], penalty, use_truth)
    return u, opts[choice][1], choice, route_s


POLICIES = ['exact_or_abstain', 'exact_else_bp', 'exact_else_mbe', 'exact_else_particle_ci', 'width_rule',
            'always_bp', 'always_particle_ci', 'learned_tree', 'learned_mlp', 'random', 'oracle']


def r2(y, yhat):
    y, yhat = np.asarray(y), np.asarray(yhat)
    return float(1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def spearman(a, b):
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def stage_report():
    model = pickle.loads((ROOT / 'selector14.pkl').read_bytes())
    res = {}
    for split in ('dev', 'test'):
        recs = json.loads((ROOT / f'receipt14_{split}.json').read_text())['records']
        ver = [r for r in recs if r['verified']]
        # H1: single-feature cost prediction with dev-fitted coefficients
        h1 = {}
        for target in ('best_exact_ops', 've_ops'):
            rows = [(r, cost_target(r, target)) for r in ver]
            rows = [(r, c) for r, c in rows if c is not None]
            y = [math.log2(c) for _, c in rows]
            h1[target] = {k: dict(r2=r2(y, np.polyval(model['cost'][target][k], [r['features'][k] for r, _ in rows])),
                                  spearman=spearman([r['features'][k] for r, _ in rows], y)) for k in P['h1_features']}
            h1[target]['n'] = len(rows)
            fams = sorted({r['spec']['family'] for r, _ in rows})
            h1[target]['by_family_spearman'] = {f: {k: spearman([r['features'][k] for r, _ in rows if r['spec']['family'] == f],
                                                                [math.log2(c) for r, c in rows if r['spec']['family'] == f])
                                                    for k in P['h1_features']} for f in fams if sum(r['spec']['family'] == f for r, _ in rows) > 3}
        # policies, paired by condition (family, n, budget, penalty)
        rng = np.random.default_rng(P['random_policy_seed'])
        cond = {}
        for r in ver:
            for B in P['budgets']:
                for pen in P['penalties']:
                    key = (r['spec']['family'], r['spec']['n'], B, pen)
                    for pol in POLICIES:
                        u, ops, ch, rs = policy_value(r, B, pen, pol, model, rng)
                        cond.setdefault(key, {}).setdefault(pol, []).append((u, ops, ch, rs))
        summary = {}
        for pol in POLICIES:
            allv = [x for c in cond.values() for x in c[pol]]
            summary[pol] = dict(mean_eu=float(np.mean([x[0] for x in allv])), mean_ops=float(np.mean([x[1] for x in allv])),
                                coverage_proxy=float(np.mean([x[2] != 'abstain' for x in allv])),
                                mean_route_seconds=float(np.mean([x[3] for x in allv])),
                                choices={c: sum(x[2] == c for x in allv) for c in OPTION_NAMES})
        # paired: each learned policy vs the strongest hand-designed policy (chosen on dev)
        res[split] = dict(h1=h1, policies=summary, n_instances=len(recs), n_verified=len(ver), conditions=len(cond),
                          cond=[dict(family=k[0], n=k[1], B=k[2], penalty=k[3], **{p: float(np.mean([x[0] for x in v[p]])) for p in POLICIES},
                                     **{p + '_ops': float(np.mean([x[1] for x in v[p]])) for p in POLICIES}) for k, v in cond.items()])
        # unverifiable instances: realized utility only
        unv = [r for r in recs if not r['verified']]
        res[split]['unverified'] = dict(n=len(unv), realized={pol: float(np.mean([policy_value(r, B, pen, pol, model, rng, use_truth=True)[0]
                                                                              for r in unv for B in P['budgets'] for pen in P['penalties']]))
                                                              for pol in POLICIES if pol != 'oracle'} if unv else {})
        # approximation error vs width
        err = []
        for r in ver:
            for k, v in r['methods'].items():
                if v.get('p') is not None and not k.startswith(('ve_', 'enumeration', 'cutset', 'gf2')):
                    err.append(dict(method=k, family=r['spec']['family'], width=r['features']['width_min_fill'],
                                    abs_err=abs(v['p'] - r['reference']), misspec=r['spec']['misspec']))
        res[split]['approx_error'] = err
    dev_best = max([p for p in POLICIES if not p.startswith(('learned', 'oracle', 'random'))], key=lambda p: res['dev']['policies'][p]['mean_eu'])
    res['strongest_hand_policy_on_dev'] = dev_best
    for split in ('dev', 'test'):
        cs = res[split]['cond']
        for pol in ('learned_tree', 'learned_mlp'):
            wins = sum(c[pol] > c[dev_best] + 1e-9 for c in cs)
            losses = sum(c[pol] < c[dev_best] - 1e-9 for c in cs)
            cheaper_eq = sum(abs(c[pol] - c[dev_best]) <= 1e-9 and c[pol + '_ops'] < c[dev_best + '_ops'] for c in cs)
            res[split][f'{pol}_vs_{dev_best}'] = dict(wins=wins, losses=losses, ties=len(cs) - wins - losses, cheaper_at_equal_utility=cheaper_eq)
    res['train_seconds'] = model['train_seconds']
    res['mlp_params'] = model['mlp_params']
    (ROOT / 'results14.json').write_text(json.dumps(res, indent=1, default=float) + '\n')
    for split in ('dev', 'test'):
        print(split, {k: (round(v['mean_eu'], 4), round(v['mean_ops'])) for k, v in res[split]['policies'].items()})
        print(split, 'H1', {t: {k: round(v['r2'], 3) for k, v in d.items() if isinstance(v, dict) and 'r2' in v} for t, d in res[split]['h1'].items()})


if __name__ == '__main__':
    stage = sys.argv[1]
    if stage in ('dev', 'test'):
        stage_run(stage)
    elif stage == 'fit':
        stage_fit()
    elif stage == 'report':
        stage_report()

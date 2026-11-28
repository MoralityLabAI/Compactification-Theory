"""DCF-15: an experimental mesh of inference units with an explicit routing objective

    J(a) = E[R_hat(a)] + lambda C(a) + mu KL_hat(a) + gamma F(a)

World: K=16 latent bits in four 4-bit units. Each episode streams T noisy factor
observations (intra-unit or cross-unit; the active cross-unit pairs change every
slow epoch) and asks one parity query per step. Each unit exposes its local state,
dependencies, uncertainty, operators, cost estimates, verification status and the
estimated information lost by keeping it separate (MeshUnit.expose()).

Timescales: fast (every step: absorb evidence, answer the query with local / cluster
BP / exact / abstain), slow (every M steps: merge or split units into blocks), and
occasional global exact inference whenever the router requests it.

Stages: python mesh15.py calibrate | dev | fit | test | report
"""
import json
import math
import os
import pickle
import platform
import sys
import time
from pathlib import Path

os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent
P = json.loads((ROOT / 'PROTOCOL.json').read_text())
K, U, T, M = 16, 4, P['steps'], P['slow_period']
STATES = np.arange(1 << K, dtype=np.int64)
UNIT_MASK = [0xF << (4 * u) for u in range(U)]
OPS = ('local', 'bp', 'exact', 'abstain')


def popcount(a):
    return np.bitwise_count(a.astype(np.uint64)).astype(np.int64)


def unit_of(bit):
    return bit // 4


# ------------------------------------------------------------------ world

def sensor(x, mask, kind):
    c = popcount(np.asarray(x) & mask)
    a = popcount(np.asarray(mask))
    return np.where(kind == 0, c % 2, np.where(kind == 1, c == a, 2 * c > a)).astype(float)


def episode(seed, rho, noise):
    """Factors and queries for one episode. Returns dict of arrays."""
    rng = np.random.default_rng(seed)
    lo, hi = P['noise'][noise]
    x = int(rng.integers(1 << K))
    facs, queries, epochs = [], [], []
    pairs = [(a, b) for a in range(U) for b in range(a + 1, U)]
    for t in range(T):
        if t % M == 0:
            act = [pairs[i] for i in rng.choice(len(pairs), size=2, replace=False)]
            epochs.append(act)
        if rng.random() < rho:
            a, b = act[int(rng.integers(2))]
            ar = int(rng.choice([2, 3]))
            na = int(rng.integers(1, ar))
            bits = list(rng.choice(4, na, replace=False) + 4 * a) + list(rng.choice(4, ar - na, replace=False) + 4 * b)
        else:
            u = int(rng.integers(U))
            ar = int(rng.choice([1, 1, 2, 3]))
            bits = list(rng.choice(4, ar, replace=False) + 4 * u)
        mask = sum(1 << int(v) for v in bits)
        kind = 0 if len(bits) == 1 else int(rng.integers(3))
        e = 0.12 if len(bits) == 1 else float(rng.uniform(lo, hi))
        y = int(sensor(x, mask, kind)) ^ int(rng.random() < e)
        facs.append((mask, kind, y, e))
        if rng.random() < 0.6:
            u = int(rng.integers(U))
            qb = rng.choice(4, 2, replace=False) + 4 * u
        else:
            a, b = act[int(rng.integers(2))]
            qb = [int(rng.integers(4)) + 4 * a, int(rng.integers(4)) + 4 * b]
        queries.append(sum(1 << int(v) for v in qb))
    return dict(x=x, facs=facs, queries=queries, epochs=epochs)


def loglik(states, fac):
    mask, kind, y, e = fac
    v = sensor(states, mask, kind)
    p = e + (1 - 2 * e) * v
    return np.log(p if y else 1 - p)


def parity_p(logp, states, q):
    w = np.exp(logp - logp.max())
    par = popcount(states & q) % 2
    return float(w[par == 1].sum() / w.sum())


# ------------------------------------------------------------- mesh units

class Mesh:
    """Blocks of units with exact block tables; cross-block factors are pending
    dependencies. `full` is the evaluation joint, never read by routers."""

    def __init__(self):
        self.blocks = [(u,) for u in range(U)]
        self.facs = []
        self.full = np.zeros(1 << K)
        self.tables = {b: np.zeros(16 ** len(b)) for b in self.blocks}
        self.cost_update = 0
        self.last_exact = {}

    def block_of(self, u):
        return next(b for b in self.blocks if u in b)

    def block_states(self, b):
        """Global-bit encodings of all states of block b."""
        idx = np.arange(16 ** len(b), dtype=np.int64)
        s = np.zeros_like(idx)
        for i, u in enumerate(b):
            s |= ((idx >> (4 * i)) & 0xF) << (4 * u)
        return s

    def touched_units(self, mask):
        return sorted({u for u in range(U) if mask & UNIT_MASK[u]})

    def absorb(self, fac):
        self.facs.append(fac)
        self.full += loglik(STATES, fac)
        units = self.touched_units(fac[0])
        blocks = {self.block_of(u) for u in units}
        if len(blocks) == 1:
            b = blocks.pop()
            self.tables[b] += loglik(self.block_states(b), fac)
            self.cost_update += 16 ** len(b)

    def cross(self):
        out = []
        for f in self.facs:
            bl = sorted({self.block_of(u) for u in self.touched_units(f[0])})
            if len(bl) > 1:
                out.append((f, bl))
        return out

    def set_blocks(self, blocks):
        blocks = [tuple(sorted(b)) for b in blocks]
        cost = 0
        for b in blocks:
            if b not in self.tables:
                st = self.block_states(b)
                t = np.zeros(len(st))
                bm = sum(UNIT_MASK[u] for u in b)
                for f in self.facs:
                    if f[0] & bm == f[0]:
                        t += loglik(st, f)
                        cost += len(st)
                self.tables[b] = t
        self.blocks = blocks
        self.tables = {b: self.tables[b] for b in blocks}
        return cost

    # -- exposed unit interface (what a router may read)
    def expose(self, b):
        cr = [f for f, bl in self.cross() if b in bl]
        t = self.tables[b]
        w = np.exp(t - t.max())
        w /= w.sum()
        ent = float(-(w * np.log(np.maximum(w, 1e-300))).sum())
        return dict(units=b, state_size=16 ** len(b), dependencies=sorted({x for f, bl in self.cross() if b in bl for x in bl if x != b}),
                    entropy=ent, operators=list(OPS), cost_local=16 ** len(b), n_cross=len(cr),
                    kl_hat=sum(kl_bound(f[3]) for f in cr), verified=self.last_exact.get(b, -1))

    # -- operators
    def local(self, q):
        sign, ops = 1.0, 0
        for b in self.blocks:
            bm = sum(UNIT_MASK[u] for u in b)
            if q & bm:
                p = parity_p(self.tables[b], self.block_states(b), q & bm)
                sign *= 1 - 2 * p
                ops += 16 ** len(b)
        return (1 - sign) / 2, ops

    def bp(self, q, iters=5, damp=0.5):
        cr = self.cross()
        beliefs = {b: self.tables[b].copy() for b in self.blocks}
        msgs = {}
        st = {b: self.block_states(b) for b in self.blocks}
        ops = 0
        for _ in range(iters):
            new = {}
            for i, (f, bl) in enumerate(cr):
                b1, b2 = bl[0], bl[1]
                # pairwise table over (b1 states, b2 states)
                joint = st[b1][:, None] | st[b2][None, :]
                lf = loglik(joint.reshape(-1), f).reshape(joint.shape)
                for src, dst, ax in ((b2, b1, 1), (b1, b2, 0)):
                    cav = beliefs[src] - msgs.get((i, src), 0.0)
                    cav = cav - cav.max()
                    m = np.log(np.maximum((np.exp(lf) * np.exp(cav)[None, :] if ax == 1 else np.exp(lf) * np.exp(cav)[:, None]).sum(axis=ax), 1e-300))
                    m = m - m.max()
                    old = msgs.get((i, dst), np.zeros_like(m))
                    new[(i, dst)] = damp * old + (1 - damp) * m
                    ops += joint.size
            for b in self.blocks:
                beliefs[b] = self.tables[b] + sum((v for (i, d), v in new.items() if d == b), np.zeros(16 ** len(b)))
            msgs = new
        sign = 1.0
        for b in self.blocks:
            bm = sum(UNIT_MASK[u] for u in b)
            if q & bm:
                sign *= 1 - 2 * parity_p(beliefs[b], st[b], q & bm)
        return (1 - sign) / 2, ops + 1

    def relevant(self, q):
        """Blocks connected to the query's blocks through pending cross factors."""
        cr = self.cross()
        todo = [self.block_of(unit_of(v)) for v in range(K) if (q >> v) & 1]
        comp = set(todo)
        while todo:
            b = todo.pop()
            for f, bl in cr:
                if b in bl:
                    for c in bl:
                        if c not in comp:
                            comp.add(c)
                            todo.append(c)
        return sorted(comp)

    def exact_cost(self, q):
        comp = self.relevant(q)
        units = sum(len(b) for b in comp)
        cm = sum(UNIT_MASK[u] for b in comp for u in b)
        nf = sum(1 for f in self.facs if f[0] & cm == f[0])
        return 16 ** units * max(1, nf), comp

    def exact(self, q):
        """Exact over the relevant component. By the component factorisation theorem
        this equals the global posterior, which is what is returned; the cost charged
        is that of enumerating the component (verified in tests/test_mesh15.py)."""
        cost, comp = self.exact_cost(q)
        for b in comp:
            self.last_exact[b] = len(self.facs)
        return parity_p(self.full, STATES, q), cost

    def true_kl(self):
        """KL(p || product of block marginals of p) under the evaluation joint."""
        p = np.exp(self.full - self.full.max())
        p /= p.sum()
        logq = np.zeros(1 << K)
        for b in self.blocks:
            bm = sum(UNIT_MASK[u] for u in b)
            key = STATES & bm
            marg = np.bincount(key, weights=p, minlength=1 << K)
            logq += np.log(np.maximum(marg[key], 1e-300))
        return float((p * (np.log(np.maximum(p, 1e-300)) - logq)).sum())


def kl_bound(e):
    """Information a binary sensor with flip rate e can carry about its scope:
    ln 2 - H(e) nats (an upper bound on the mutual information it induces)."""
    e = min(max(e, 1e-9), 1 - 1e-9)
    return math.log(2) + e * math.log(e) + (1 - e) * math.log(1 - e)


# --------------------------------------------------------------- cost model

def calibrate():
    """Time batched NumPy kernels at batch sizes 1..256 and fit t = a_op + b_op * work."""
    rng = np.random.default_rng(P['calibration_seed'])
    rows = []
    for B in P['batch_sizes']:
        for s in (1, 2):
            tab = rng.normal(size=(B, 16 ** s))
            st = np.arange(16 ** s)
            rows.append(('local', B, B * 16 ** s, _time(lambda: (np.exp(tab - tab.max(1, keepdims=True)) * (popcount(st & 3) % 2)).sum(1))))
            rows.append(('update', B, B * 16 ** s, _time(lambda: tab + np.log(0.3 + 0.4 * (popcount(st & 5) % 2))[None, :])))
        for s in (1, 2):
            A = rng.random((B, 16 ** s, 16))
            v = rng.random((B, 16))
            rows.append(('bp', B, B * 16 ** s * 16, _time(lambda: np.einsum('bij,bj->bi', A, v))))
        for u in (1, 2, 3, 4):
            if B * 16 ** u > 2 ** 22:
                continue
            nf = 8 * u
            st = np.arange(16 ** u, dtype=np.int64)
            masks = rng.integers(1, 16 ** u, size=nf)

            def kern():
                acc = np.zeros((B, 16 ** u))
                for m in masks:
                    acc += np.log(0.2 + 0.6 * (popcount(st & m) % 2))[None, :]
                return acc
            rows.append(('exact', B, B * 16 ** u * nf, _time(kern)))
    fit = {}
    for op in ('local', 'update', 'bp', 'exact'):
        r = [x for x in rows if x[0] == op]
        X = np.array([[1.0, x[2]] for x in r])
        y = np.array([x[3] for x in r])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        pred = X @ coef
        fit[op] = dict(a=float(max(coef[0], 0.0)), b=float(max(coef[1], 0.0)),
                       r2=float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()))
    (ROOT / 'cost_model15.json').write_text(json.dumps(dict(fit=fit, rows=rows, environment=env()), indent=1) + '\n')
    print(fit)


def _time(f, reps=7):
    f()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts))


def env():
    return dict(python=platform.python_version(), numpy=np.__version__, processor=platform.processor())


# ------------------------------------------------------------------ routers

def query_features(mesh, q, t):
    """Routing features built only from what the units expose (MeshUnit interface)."""
    cost, comp = mesh.exact_cost(q)
    p_loc, _ = mesh.local(q)
    qb = sorted({mesh.block_of(unit_of(v)) for v in range(K) if (q >> v) & 1})
    ex = [mesh.expose(b) for b in qb]
    exc = [mesh.expose(b) for b in comp]
    nf = len(mesh.facs)
    return dict(p_local=p_loc, margin=min(abs(p_loc - .8), abs(p_loc - .2)), conf=max(p_loc, 1 - p_loc),
                n_query_blocks=len(qb), comp_units=sum(len(b) for b in comp), n_cross_comp=sum(e['n_cross'] for e in exc),
                kl_hat_query=sum(e['kl_hat'] for e in ex), kl_hat_comp=sum(e['kl_hat'] for e in exc),
                entropy_query=sum(e['entropy'] for e in ex), staleness=min(nf - max(e['verified'], 0) for e in ex),
                log2_exact_cost=math.log2(cost), step=t, merged=sum(len(b) > 1 for b in mesh.blocks))


FEAT_KEYS = ['p_local', 'margin', 'conf', 'n_query_blocks', 'comp_units', 'n_cross_comp', 'kl_hat_query', 'kl_hat_comp',
             'entropy_query', 'staleness', 'log2_exact_cost', 'step', 'merged']


def op_seconds(cm, op, work):
    f = cm[op]
    return f['a'] + f['b'] * work


def slow_merge(mesh, rule, tau):
    """Slow timescale: merge the unit pair with the largest estimated discarded
    correlation if it exceeds tau; blocks are capped at two units."""
    if rule == 'none':
        return 0
    score = {}
    for f, bl in mesh.cross():
        units = mesh.touched_units(f[0])
        if len(units) == 2:
            score[tuple(units)] = score.get(tuple(units), 0) + kl_bound(f[3])
    blocks = [(u,) for u in range(U)]
    used = set()
    for pair, s in sorted(score.items(), key=lambda kv: -kv[1]):
        if s > tau and not (set(pair) & used):
            blocks = [b for b in blocks if b[0] not in pair] + [pair]
            used |= set(pair)
    return mesh.set_blocks(blocks)


def run_condition(seeds, rho, noise, router, model, cm, B, lam, gamma, mu, record=False):
    """Run len(seeds) episodes in lockstep batches of B. Returns per-query rows."""
    rows, labels, kls = [], [], []
    for start in range(0, len(seeds), B):
        batch = seeds[start:start + B]
        eps = [episode(s, rho, noise) for s in batch]
        meshes = [Mesh() for _ in batch]
        rng = np.random.default_rng(batch[0] + 17)
        for t in range(T):
            launched = set()
            step_work = {}
            for e, mesh in zip(eps, meshes):
                mesh.absorb(e['facs'][t])
                if t % M == M - 1:
                    mesh.cost_update += slow_merge(mesh, model.get('merge_rule', 'kl'), model.get('merge_tau', P['merge_tau']))
            for i, (e, mesh) in enumerate(zip(eps, meshes)):
                q = e['queries'][t]
                p_ref = parity_p(mesh.full, STATES, q)  # evaluation only; outside the timed window
                t0 = time.perf_counter()
                if router.startswith('fixed_') or router == 'random':
                    feats = None
                else:
                    feats = query_features(mesh, q, t)
                choice = choose(router, feats, model, mesh, q, p_ref, cm, launched, lam, gamma, mu, rng)
                route_s = time.perf_counter() - t0
                if record and feats is None:
                    feats = query_features(mesh, q, t)
                if choice == 'local':
                    p, w = mesh.local(q)
                elif choice == 'bp':
                    p, w = mesh.bp(q)
                elif choice == 'exact':
                    p, w = mesh.exact(q)
                else:
                    p, w = None, 0
                if choice != 'abstain':
                    launched.add(choice)
                    step_work[choice] = step_work.get(choice, 0) + w
                a = -1 if p is None else 1 if p >= .8 else 0 if p <= .2 else -1
                eu = 0.0 if a < 0 else (5 * p_ref - 4 if a == 1 else 1 - 5 * p_ref)
                oracle = max(0.0, 5 * p_ref - 4, 1 - 5 * p_ref)
                truth = bin(e['x'] & q).count('1') % 2
                ru = 0.0 if a < 0 else (1.0 if a == truth else -4.0)
                rows.append(dict(seed=batch[i], t=t, op=choice, eu=eu, regret=oracle - eu, realized=ru, work=w,
                                 route_seconds=route_s, wrong=int(a >= 0 and a != truth), act=int(a >= 0)))
                if record:
                    regs = {}
                    for op in ('local', 'bp'):
                        pp, _ = mesh.local(q) if op == 'local' else mesh.bp(q)
                        aa = 1 if pp >= .8 else 0 if pp <= .2 else -1
                        regs[op] = oracle - (0.0 if aa < 0 else (5 * p_ref - 4 if aa == 1 else 1 - 5 * p_ref))
                    regs['exact'] = 0.0
                    regs['abstain'] = oracle
                    labels.append(dict(features=feats, regret=regs))
            # batch-level kernel accounting: one launch per distinct operator per step
            step_s = sum(op_seconds(cm, op, wk) for op, wk in step_work.items())
            upd = sum(m.cost_update for m in meshes)
            for m in meshes:
                m.cost_update = 0
            step_s += op_seconds(cm, 'update', upd)
            for r in rows[-len(batch):]:
                r['step_seconds_share'] = step_s / len(batch)
                r['launches'] = len(step_work)
        for e, mesh in zip(batch, meshes):
            kls.append(dict(seed=e, final_true_kl=mesh.true_kl(), blocks=[list(b) for b in mesh.blocks]))
    return rows, labels, kls


def choose(router, feats, model, mesh, q, p_ref, cm, launched, lam, gamma, mu, rng):
    if router.startswith('fixed_'):
        return router[len('fixed_'):]
    if router == 'random':
        return str(rng.choice(['local', 'bp', 'exact']))
    cost_ex = 2 ** feats['log2_exact_cost']
    work = {'local': 16 * feats['n_query_blocks'], 'bp': 512 * max(1, feats['n_cross_comp']) * 5, 'exact': cost_ex, 'abstain': 0}
    if cost_ex > 2 ** P['exact_budget_log2']:
        work.pop('exact')
    if router == 'oracle':
        regs = {}
        for op in work:
            if op == 'exact':
                regs[op] = 0.0
                continue
            oracle = max(0.0, 5 * p_ref - 4, 1 - 5 * p_ref)
            if op == 'abstain':
                regs[op] = oracle
                continue
            pp, _ = mesh.local(q) if op == 'local' else mesh.bp(q)
            aa = 1 if pp >= .8 else 0 if pp <= .2 else -1
            regs[op] = oracle - (0.0 if aa < 0 else (5 * p_ref - 4 if aa == 1 else 1 - 5 * p_ref))
    elif router == 'symbolic':
        # hand-designed: exact when the local answer is near a decision threshold or the
        # discarded correlation estimate is large; BP when moderate; otherwise local
        th = model['symbolic']
        if 'exact' in work and (feats['margin'] < th['margin'] or feats['kl_hat_query'] > th['kl_exact']):
            return 'exact'
        if feats['kl_hat_query'] > th['kl_bp']:
            return 'bp'
        return 'local'
    elif router == 'symbolic_j':
        # structural J: regret estimate = kappa * KL_hat for approximations
        kap = model['kappa']
        regs = {'local': kap['local'] * feats['kl_hat_query'] + kap['local0'] * (feats['margin'] < .1),
                'bp': kap['bp'] * feats['kl_hat_query'] + kap['bp0'] * (feats['margin'] < .1),
                'exact': 0.0, 'abstain': max(0.0, 5 * feats['conf'] - 4) if feats['conf'] >= .8 else 0.0}
    elif router == 'learned':
        x = np.array([[feats[k] for k in FEAT_KEYS]])
        pr = model['mlp'].predict(model['scaler'].transform(x))[0]
        regs = dict(zip(['local', 'bp', 'abstain'], pr))
        regs['exact'] = 0.0
    else:
        raise ValueError(router)
    best, bj = None, None
    for op in work:
        j = regs.get(op, 0.0) + lam * 1000 * op_seconds(cm, op if op != 'abstain' else 'local', work[op]) * (op != 'abstain') \
            + gamma * (op not in launched and op != 'abstain')
        if op in ('local', 'bp'):
            j += mu * feats['kl_hat_query'] * (op == 'local')
        if bj is None or j < bj:
            best, bj = op, j
    return best


# ------------------------------------------------------------------- stages

ROUTERS = ['fixed_local', 'fixed_local_nomerge', 'fixed_bp', 'fixed_exact', 'random', 'symbolic', 'symbolic_j', 'learned', 'oracle']


def conditions(split):
    return [(rho, noise) for rho in P['rho'][split] for noise in ('low', 'high')]


def seeds_for(split, ci, n):
    return [P['seed_base'][split] + 10000 * ci + i for i in range(n)]


def cost_fit():
    return json.loads((ROOT / 'cost_model15.json').read_text())['fit']


def op_work(f):
    w = {'local': 16 * f['n_query_blocks'], 'bp': 512 * max(1, f['n_cross_comp']) * 5, 'abstain': 0}
    if f['log2_exact_cost'] <= P['exact_budget_log2']:
        w['exact'] = 2 ** f['log2_exact_cost']
    return w


def offline_choice(policy, f, th=None):
    w = op_work(f)
    if policy.startswith('fixed_'):
        op = policy[len('fixed_'):]
        return op if op in w else 'abstain'
    if 'exact' in w and (f['margin'] < th['margin'] or f['kl_hat_query'] > th['kl_exact']):
        return 'exact'
    return 'bp' if f['kl_hat_query'] > th['kl_bp'] else 'local'


def offline_objective(labels, policy, cm, lam, th=None):
    vals = []
    for lab in labels:
        f = lab['features']
        op = offline_choice(policy, f, th)
        cost = 0.0 if op == 'abstain' else op_seconds(cm, op, op_work(f)[op])
        vals.append(-lab['regret'][op] - lam * 1000 * cost)
    return float(np.mean(vals))


def stage_dev():
    cm = cost_fit()
    labels = []
    for ci, (rho, noise) in enumerate(conditions('dev')):
        seeds = seeds_for('dev', ci, P['episodes']['dev'])
        _, lab, _ = run_condition(seeds, rho, noise, 'fixed_exact', {}, cm, P['batch'], P['lambda'], P['gamma'], P['mu'], record=True)
        for x in lab:
            x.update(rho=rho, noise=noise)
        labels += lab
        print('dev', rho, noise, len(labels), flush=True)
    (ROOT / 'receipt15_dev.json').write_text(json.dumps(dict(protocol=P, labels=labels, environment=env()), separators=(',', ':')) + '\n')


def stage_fit():
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    cm = cost_fit()
    labels = json.loads((ROOT / 'receipt15_dev.json').read_text())['labels']
    lam = P['lambda']
    best = None
    for mg in (0.0, 0.03, 0.06, 0.1, 0.15):
        for ke in (0.05, 0.1, 0.2, 0.4, 0.8, 99):
            for kb in (0.02, 0.05, 0.1, 0.2, 0.4, 99):
                th = dict(margin=mg, kl_exact=ke, kl_bp=kb)
                v = offline_objective(labels, 'symbolic', cm, lam, th)
                if best is None or v > best[1] + 1e-12:
                    best = (th, v)
    X = np.array([[lab['features'][k] for k in FEAT_KEYS] for lab in labels])
    kap = {}
    for op in ('local', 'bp'):
        A = np.array([[lab['features']['kl_hat_query'], float(lab['features']['margin'] < .1)] for lab in labels])
        y = np.array([lab['regret'][op] for lab in labels])
        c, *_ = np.linalg.lstsq(A, y, rcond=None)
        kap[op], kap[op + '0'] = float(c[0]), float(c[1])
    Y = np.array([[lab['regret'][op] for op in ('local', 'bp', 'abstain')] for lab in labels])
    scaler = StandardScaler().fit(X)
    t = time.perf_counter()
    mlp = MLPRegressor(hidden_layer_sizes=tuple(P['mlp_hidden']), max_iter=3000, random_state=0).fit(scaler.transform(X), Y)
    train_s = time.perf_counter() - t
    fixed = {p: offline_objective(labels, p, cm, lam) for p in ('fixed_local', 'fixed_bp', 'fixed_exact')}
    model = dict(symbolic=best[0], kappa=kap, mlp=mlp, scaler=scaler, train_seconds=train_s, n_train=len(labels),
                 mlp_params=int(sum(a.size for a in mlp.coefs_) + sum(b.size for b in mlp.intercepts_)))
    (ROOT / 'model15.pkl').write_bytes(pickle.dumps(model))
    summary = dict(symbolic=best[0], symbolic_dev_objective=best[1], kappa=kap, fixed_dev_objective=fixed,
                   train_seconds=train_s, n_train=len(labels), mlp_params=model['mlp_params'])
    (ROOT / 'model15_summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    print(summary)


def summarize(rows, train_amortized=0.0):
    eu = np.array([r['eu'] for r in rows])
    kern = np.array([r['step_seconds_share'] for r in rows])
    route = np.array([r['route_seconds'] for r in rows])
    act = np.array([r['act'] for r in rows])
    wrong = np.array([r['wrong'] for r in rows])
    lam = P['lambda']
    return dict(n=len(rows), eu=float(eu.mean()), regret=float(np.mean([r['regret'] for r in rows])),
                realized=float(np.mean([r['realized'] for r in rows])), coverage=float(act.mean()),
                selective_risk=float(wrong.sum() / max(1, act.sum())), kernel_ms=float(kern.mean() * 1000),
                route_ms=float(route.mean() * 1000), launches_per_step=float(np.mean([r['launches'] for r in rows])),
                objective_kernel=float((eu - lam * 1000 * kern).mean()),
                objective_total=float((eu - lam * 1000 * (kern + route)).mean() - lam * 1000 * train_amortized),
                ops={o: float(np.mean([r['op'] == o for r in rows])) for o in OPS})


def stage_test():
    cm = cost_fit()
    model = pickle.loads((ROOT / 'model15.pkl').read_bytes())
    out = dict(protocol=P, environment=env(), conditions=[], batch_sweep=[])
    for ci, (rho, noise) in enumerate(conditions('test')):
        seeds = seeds_for('test', ci, P['episodes']['test'])
        for router in ROUTERS:
            m = dict(model)
            if router == 'fixed_local_nomerge':
                m['merge_rule'] = 'none'
            name = 'fixed_local' if router == 'fixed_local_nomerge' else router
            rows, _, kls = run_condition(seeds, rho, noise, name, m, cm, P['batch'], P['lambda'], P['gamma'], P['mu'])
            out['conditions'].append(dict(rho=rho, noise=noise, router=router, rows=rows, kls=kls))
            print('test', rho, noise, router, round(float(np.mean([r['eu'] for r in rows])), 4), flush=True)
        (ROOT / 'receipt15_test.json').write_text(json.dumps(out, separators=(',', ':')) + '\n')
    rho, noise = P['batch_sweep']['condition']
    seeds = seeds_for('sweep', 0, P['batch_sweep']['episodes'])
    for router in P['batch_sweep']['routers']:
        for B in P['batch_sizes']:
            t = time.perf_counter()
            rows, _, _ = run_condition(seeds, rho, noise, router, dict(model), cm, B, P['lambda'], P['gamma'], P['mu'])
            out['batch_sweep'].append(dict(router=router, B=B, summary=summarize(rows), simulator_wall_seconds=time.perf_counter() - t))
            print('sweep', router, B, flush=True)
    (ROOT / 'receipt15_test.json').write_text(json.dumps(out, separators=(',', ':')) + '\n')


def stage_report():
    model = json.loads((ROOT / 'model15_summary.json').read_text())
    rec = json.loads((ROOT / "receipt15_test.json").read_text()) if (ROOT / "receipt15_test.json").exists() else json.loads(__import__("gzip").open(ROOT / "receipt15_test.json.gz", "rt").read())
    n_dec = sum(len(c['rows']) for c in rec['conditions'] if c['router'] == 'learned')
    amort = model['train_seconds'] / max(1, n_dec)
    table = {}
    for c in rec['conditions']:
        table[(c['rho'], c['noise'], c['router'])] = dict(summarize(c['rows'], amort if c['router'] == 'learned' else 0.0),
                                                          mean_final_kl=float(np.mean([k['final_true_kl'] for k in c['kls']])))
    conds = sorted({(k[0], k[1]) for k in table})
    sym = ['fixed_local', 'fixed_bp', 'fixed_exact', 'symbolic', 'symbolic_j']
    devbest = max(['fixed_local', 'fixed_bp', 'fixed_exact'], key=lambda p: model['fixed_dev_objective'][p])
    strongest = 'symbolic' if model['symbolic_dev_objective'] >= model['fixed_dev_objective'][devbest] else devbest
    paired = {}
    for metric in ('objective_total', 'objective_kernel', 'eu'):
        for other in sym:
            w = sum(table[c + ('learned',)][metric] > table[c + (other,)][metric] + 1e-12 for c in conds)
            lo = sum(table[c + ('learned',)][metric] < table[c + (other,)][metric] - 1e-12 for c in conds)
            paired[f'learned_vs_{other}_{metric}'] = dict(wins=w, losses=lo, conditions=len(conds))
    res = dict(strongest_symbolic_on_dev=strongest, train_seconds=model['train_seconds'], amortized_train_ms_per_decision=amort * 1000,
               mlp_params=model['mlp_params'], table={f'{k[0]}/{k[1]}/{k[2]}': v for k, v in table.items()}, paired=paired,
               promotion=dict(rule='learned wins more held-out conditions than it loses against the strongest dev-selected symbolic router on objective_total',
                              wins=paired[f'learned_vs_{strongest}_objective_total']['wins'],
                              losses=paired[f'learned_vs_{strongest}_objective_total']['losses']),
               batch_sweep=rec['batch_sweep'], cost_model=json.loads((ROOT / 'cost_model15.json').read_text())['fit'])
    res['promotion']['promoted'] = res['promotion']['wins'] > res['promotion']['losses']
    (ROOT / 'results15.json').write_text(json.dumps(res, indent=1) + '\n')
    for c in conds:
        print(c, {r: (round(table[c + (r,)]['eu'], 3), round(table[c + (r,)]['objective_total'], 3)) for r in ROUTERS})
    print(res['promotion'])


if __name__ == '__main__':
    {'calibrate': calibrate, 'dev': stage_dev, 'fit': stage_fit, 'test': stage_test, 'report': stage_report}[sys.argv[1]]()

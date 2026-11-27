"""DCF-23: a learned message-passing model on the factor graph.

DCF-22's token models summarised factors and learned almost nothing transferable. This
model runs on the query component's bipartite factor graph:

  variable nodes  [is query, BP belief logit, degree]
  factor nodes    [arity one-hot, kind one-hot, observation, noise, evidence strength,
                   the factor's log-likelihood table padded to 16 entries]
  rounds          4 x (variables -> factors MLP, factors -> variables MLP + GRU cell)
  readout         sum over query variables, plus BP's parity logit; the output is a
                  correction added to BP's parity logit (variant 'mp_bp'), or a posterior
                  logit from the graph alone (variant 'mp_pure', no BP inputs)

Training data and splits are DCF-22's: DCF-16 train plus train_large (3,600 records of
the four training families), DCF-16 val for epoch selection, DCF-16 held-out once.
BP is fg.loopy_bp with a fixed 50-iteration cap and damping 0.5 (independent of the
record's budget), and is both an input and the baseline.

Usage: python mp23.py features | train | predict | report
"""
import gzip
import json
import math
import random
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
CUR = ROOT.parent / 'dcf16_decision_curriculum'
D22 = ROOT.parent / 'dcf22_curriculum_model'
sys.path.insert(0, str(CUR))
sys.path.insert(0, str(ROOT.parent / 'dcf14_width_routing'))
import curriculum16 as C  # noqa: E402
import fg  # noqa: E402

P = json.loads((ROOT / 'PROTOCOL.json').read_text())
KINDS = ['id', 'xor', 'and', 'maj', 'agree']
FDIM = 4 + len(KINDS) + 3 + 16
VDIM = 3


def load_split(name):
    path = (D22 / 'data' / 'train_large.jsonl.gz') if name == 'train_large' else (CUR / 'data' / f'{name}.jsonl.gz')
    return [json.loads(line) for line in gzip.open(path, 'rt')]


def logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def bp_beliefs(comp, variables, iters, damping):
    """Same sum-product schedule as fg.loopy_bp, returning per-variable beliefs."""
    f2v = {(i, v): np.full(2, .5) for i, s in enumerate(comp.scopes) for v in s}
    v2f = dict(f2v)
    nbrs = {v: [i for i, s in enumerate(comp.scopes) if v in s] for v in variables}
    for _ in range(iters):
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
        if delta < 1e-7:
            break
    out = {}
    for v in variables:
        bel = np.ones(2)
        for j in nbrs[v]:
            bel = bel * f2v[(j, v)]
        out[v] = float(bel[1] / bel.sum()) if bel.sum() > 0 else 0.5
    return out


def featurize(e):
    prob = C.to_problem(e)
    comp, variables = fg.query_component(prob)
    r = fg.loopy_bp(prob, max_iters=P['bp_iters'], damping=P['bp_damping'])
    p_bp = r['p'] if r['p'] is not None else 0.5
    bel = bp_beliefs(comp, variables, P['bp_iters'], P['bp_damping'])
    vid = {v: i for i, v in enumerate(variables)}
    deg = {v: 0 for v in variables}
    for s in comp.scopes:
        for v in s:
            deg[v] += 1
    qs = set(comp.query)
    vfeat = [[float(v in qs), max(-8.0, min(8.0, logit(bel[v]))) / 8, min(deg[v], 20) / 20] for v in variables]
    ffeat, edges = [], []
    for fi, (s, t, m) in enumerate(zip(comp.scopes, comp.tables, comp.meta)):
        f = [0.0] * FDIM
        f[min(len(s), 4) - 1] = 1.0
        f[4 + KINDS.index(m['kind'])] = 1.0
        e_ = min(max(m['e'], 1e-3), 0.5)
        f[9], f[10], f[11] = float(m['y']), m['e'], math.log((1 - e_) / e_) / 7
        lt = np.log(np.maximum(t.reshape(-1), 1e-9))
        lt = (lt - lt.max()) / 7
        f[12:12 + min(16, lt.size)] = lt[:16].tolist()
        ffeat.append(f)
        for v in s:
            edges.append((fi, vid[v]))
    return dict(id=e['id'], vfeat=vfeat, ffeat=ffeat or [[0.0] * FDIM], edges=edges, query=[vid[q] for q in comp.query],
                p_bp=p_bp, bp_ops=int(r['ops']), p_ref=e['verification']['reference_p'],
                penalty=-e['decision_rule']['reward']['wrong'], truth=e['verification']['truth_parity'], family=e['family'])


def features(name):
    cache = ROOT / 'cache' / f'{name}.json.gz'
    if cache.exists():
        return json.loads(gzip.open(cache, 'rt').read())
    with Pool(P['workers']) as pool:
        out = pool.map(featurize, load_split(name), chunksize=4)
    cache.parent.mkdir(exist_ok=True)
    with gzip.GzipFile(cache, 'wb', mtime=0) as f:
        f.write(json.dumps(out).encode())
    return out


def stage_features():
    for name in ('train', 'train_large', 'val', 'heldout'):
        print(name, len(features(name)), flush=True)


# ----------------------------------------------------------------------- model

def build(variant, seed):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    H = P['hidden']
    use_bp = variant == 'mp_bp'

    class MP(nn.Module):
        def __init__(self):
            super().__init__()
            self.venc = nn.Linear(VDIM if use_bp else VDIM - 1, H)
            self.fenc = nn.Linear(FDIM, H)
            self.v2f = nn.Sequential(nn.Linear(2 * H, H), nn.ReLU())
            self.f2v = nn.Sequential(nn.Linear(2 * H, H), nn.ReLU())
            self.cell = nn.GRUCell(H, H)
            self.read = nn.Sequential(nn.Linear(H + 2, H), nn.ReLU(), nn.Linear(H, 1))

        def forward(self, g):
            vf = g['vfeat'] if use_bp else g['vfeat'][:, [0, 2]]
            hv = torch.relu(self.venc(vf))
            hf = torch.relu(self.fenc(g['ffeat']))
            fe, ve = g['edge_f'], g['edge_v']
            for _ in range(P['rounds']):
                agg_f = torch.zeros_like(hf).index_add_(0, fe, hv[ve])
                hf = self.v2f(torch.cat([hf, agg_f], 1))
                msg = self.f2v(torch.cat([hf[fe], hv[ve]], 1))
                agg_v = torch.zeros_like(hv).index_add_(0, ve, msg)
                hv = self.cell(agg_v, hv)
            q = torch.zeros(g['n_graphs'], hv.shape[1]).index_add_(0, g['q_graph'], hv[g['q_node']])
            bp_l = g['bp_logit'] if use_bp else torch.zeros_like(g['bp_logit'])
            out = self.read(torch.cat([q, bp_l[:, None] / 8, g['q_size'][:, None] / 3], 1)).squeeze(1)
            return out + (bp_l if use_bp else 0)
    return MP()


def collate(items):
    import torch
    vf, ff, ef, ev, qn, qg = [], [], [], [], [], []
    vo = fo = 0
    for gi, it in enumerate(items):
        vf += it['vfeat']
        ff += it['ffeat']
        for f, v in it['edges']:
            ef.append(f + fo)
            ev.append(v + vo)
        for q in it['query']:
            qn.append(q + vo)
            qg.append(gi)
        vo += len(it['vfeat'])
        fo += len(it['ffeat'])
    return dict(vfeat=torch.tensor(vf), ffeat=torch.tensor(ff), edge_f=torch.tensor(ef, dtype=torch.long),
                edge_v=torch.tensor(ev, dtype=torch.long), q_node=torch.tensor(qn, dtype=torch.long),
                q_graph=torch.tensor(qg, dtype=torch.long), n_graphs=len(items),
                bp_logit=torch.tensor([logit(it['p_bp']) for it in items], dtype=torch.float32),
                q_size=torch.tensor([len(it['query']) for it in items], dtype=torch.float32),
                p_ref=torch.tensor([it['p_ref'] for it in items], dtype=torch.float32))


def macs(variant, it):
    H, R = P['hidden'], P['rounds']
    nv, nf, ne = len(it['vfeat']), len(it['ffeat']), len(it['edges'])
    enc = nv * VDIM * H + nf * FDIM * H
    per_round = nf * 2 * H * H + ne * 2 * H * H + nv * 3 * (2 * H * H)
    return int(enc + R * per_round + (H + 2) * H + H)


def stage_train():
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(P['threads'])
    train = features('train') + features('train_large')
    val = features('val')
    vb = collate(val)
    log = {}
    (ROOT / 'models').mkdir(exist_ok=True)
    for variant in ('mp_bp', 'mp_pure'):
        for seed in P['seeds']:
            net = build(variant, seed)
            opt = torch.optim.Adam(net.parameters(), lr=P['lr'])
            rng = random.Random(seed)
            best, best_ep, bad, hist = None, -1, 0, []
            for ep in range(P['max_epochs']):
                net.train()
                idx = list(range(len(train)))
                rng.shuffle(idx)
                for s in range(0, len(idx), P['batch']):
                    b = collate([train[i] for i in idx[s:s + P['batch']]])
                    opt.zero_grad()
                    loss = F.binary_cross_entropy_with_logits(net(b), b['p_ref'])
                    loss.backward()
                    opt.step()
                net.eval()
                with torch.no_grad():
                    v = float(F.binary_cross_entropy_with_logits(net(vb), vb['p_ref']))
                hist.append(v)
                if best is None or v < best - 1e-5:
                    best, best_ep, bad = v, ep, 0
                    torch.save(net.state_dict(), ROOT / 'models' / f'{variant}_s{seed}.pt')
                else:
                    bad += 1
                    if bad >= P['patience']:
                        break
            log[f'{variant}_s{seed}'] = dict(best_val_bce=best, best_epoch=best_ep, epochs_run=len(hist), val_history=hist,
                                             params=int(sum(p.numel() for p in net.parameters())), n_train=len(train))
            print(variant, seed, round(best, 4), best_ep, log[f'{variant}_s{seed}']['params'], flush=True)
    (ROOT / 'train23.json').write_bytes((json.dumps(log, indent=1) + '\n').encode())


def export_weights():
    import torch
    out = {}
    for f in sorted((ROOT / 'models').glob('*.pt')):
        out[f.name] = {k: v.tolist() for k, v in torch.load(f, weights_only=True).items()}
    (ROOT / 'models23_weights.json').write_bytes((json.dumps(out) + '\n').encode())


def stage_predict():
    import torch
    held = features('heldout')
    out = {}
    for variant in ('mp_bp', 'mp_pure'):
        for seed in P['seeds']:
            net = build(variant, seed)
            net.load_state_dict(torch.load(ROOT / 'models' / f'{variant}_s{seed}.pt', weights_only=True))
            net.eval()
            with torch.no_grad():
                p = torch.sigmoid(net(collate(held))).tolist()
            out[f'{variant}_s{seed}'] = [dict(id=it['id'], p=float(q), macs=macs(variant, it)) for it, q in zip(held, p)]
    (ROOT / 'predictions23.json').write_bytes((json.dumps(out) + '\n').encode())
    print('predictions', list(out))


def bootstrap_lower(d, seed, reps=2000):
    rng = np.random.default_rng(seed)
    d = np.asarray(d)
    return float(np.percentile([d[rng.integers(0, len(d), len(d))].mean() for _ in range(reps)], 2.5)), float(d.mean())


def stage_report():
    held = features('heldout')
    preds = json.loads((ROOT / 'predictions23.json').read_text())
    ref = np.array([it['p_ref'] for it in held])
    pbp = np.array([it['p_bp'] for it in held])

    def eu(p, it):
        return fg.expected_utility(fg.decide(float(p), it['penalty']), it['p_ref'], it['penalty'])
    bp_eu = np.array([eu(it['p_bp'], it) for it in held])
    ex_eu = np.array([fg.oracle_utility(it['p_ref'], it['penalty']) for it in held])
    res = dict(n=len(held), bp=dict(mse=float(np.mean((pbp - ref) ** 2)), eu=float(bp_eu.mean()), mean_ops=float(np.mean([it['bp_ops'] for it in held]))),
               constant=dict(mse=float(np.mean((ref - .5) ** 2))), exact_posterior_eu=float(ex_eu.mean()), models={})
    fams = sorted({it['family'] for it in held})
    for k, v in preds.items():
        p = np.array([x['p'] for x in v])
        m_eu = np.array([eu(q, it) for q, it in zip(p, held)])
        lo, d = bootstrap_lower(m_eu - bp_eu, P['bootstrap_seed'])
        res['models'][k] = dict(
            mse=float(np.mean((p - ref) ** 2)), eu=float(m_eu.mean()), eu_minus_bp=d, eu_minus_bp_lower95=lo,
            coverage=float(np.mean([fg.decide(float(q), it['penalty']) >= 0 for q, it in zip(p, held)])),
            mean_macs=float(np.mean([x['macs'] for x in v])),
            per_family_mse={f: float(np.mean([(q - it['p_ref']) ** 2 for q, it in zip(p, held) if it['family'] == f])) for f in fams},
            per_family_eu_minus_bp={f: float(np.mean([a - b for a, b, it in zip(m_eu, bp_eu, held) if it['family'] == f])) for f in fams})
    res['bp_per_family_mse'] = {f: float(np.mean([(it['p_bp'] - it['p_ref']) ** 2 for it in held if it['family'] == f])) for f in fams}
    S = P['seeds']
    mb = [res['models'][f'mp_bp_s{s}'] for s in S]
    mp = [res['models'][f'mp_pure_s{s}'] for s in S]
    res['gates'] = dict(
        M1=dict(rule=P['gates']['M1'], mse=[m['mse'] for m in mb], bp_mse=res['bp']['mse'], passed=sum(m['mse'] < res['bp']['mse'] for m in mb) >= 2),
        M2=dict(rule=P['gates']['M2'], lower95=[m['eu_minus_bp_lower95'] for m in mb], passed=sum(m['eu_minus_bp_lower95'] > 0 for m in mb) >= 2),
        M3=dict(rule=P['gates']['M3'], mse=[m['mse'] for m in mp], constant_mse=res['constant']['mse'],
                passed=sum(m['mse'] < res['constant']['mse'] for m in mp) >= 2))
    (ROOT / 'results23.json').write_bytes((json.dumps(res, indent=1) + '\n').encode())
    print('bp', res['bp'], 'constant', res['constant'], 'exact EU', round(res['exact_posterior_eu'], 4))
    for k, m in res['models'].items():
        print(k, {x: round(y, 4) for x, y in m.items() if isinstance(y, float)})
    print({k: v['passed'] for k, v in res['gates'].items()})


if __name__ == '__main__':
    {'features': stage_features, 'train': stage_train, 'predict': stage_predict, 'report': stage_report,
     'export': export_weights}[sys.argv[1]]()

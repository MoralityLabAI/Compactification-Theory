"""DCF-22: a compact recurrent model trained on the DCF-16 curriculum.

Inputs per record (no operator posterior is visible, so choosing an operator never
leaks its answer):
  tokens  factors of the query's component, ordered far-to-near by hop distance from
          the query (the order elimination would consume them), nearest K kept:
          arity one-hot, kind one-hot, observation, noise, evidence strength,
          query variables in scope, hop distance
  global  size, component size, budget, penalty, query size, and for each operator
          its counted ops (log2) and whether it fits the budget

Heads (multi-task, the DCF-16 targets):
  posterior   P(query parity = 1) against the verified reference (calibration target)
  choice      selected option in {exact, mbe, bp, particle, abstain}, masked to
              operators that fit the budget
  cost        log2 exact cost (complexity estimation)

Architectures with matched parameter counts: 'gru' (recurrent over tokens) and 'mlp'
(mean and max pooling of the same tokens, no recurrence). Three training seeds each.
Training uses DCF-16 train plus fresh records from the same four training families
(train_large); validation selects the epoch; held-out is scored once.

Usage: python model22.py data | train | predict | report
"""
import gzip
import hashlib
import json
import math
import random
import sys
from collections import deque
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
CUR = ROOT.parent / 'dcf16_decision_curriculum'
sys.path.insert(0, str(CUR))
sys.path.insert(0, str(ROOT.parent / 'dcf14_width_routing'))
import curriculum16 as C  # noqa: E402
import fg  # noqa: E402

P = json.loads((ROOT / 'PROTOCOL.json').read_text())
KINDS = ['id', 'xor', 'and', 'maj', 'agree']
CLASSES = ['exact', 'mbe', 'bp', 'particle', 'abstain']
OPS = ['ve_min_fill', 'enumeration', 'cutset', 'gf2', 'bp', 'mbe', 'particle']
TOK_DIM = 4 + len(KINDS) + 5
GLOB_DIM = 6 + 3 * len(OPS)


def load_split(name):
    path = (ROOT / 'data' / f'{name}.jsonl.gz') if name == 'train_large' else (CUR / 'data' / f'{name}.jsonl.gz')
    return [json.loads(line) for line in gzip.open(path, 'rt')]


# ----------------------------------------------------------------------- data

def stage_data():
    specs = C._block(P['train_large'], 'train_large')
    with Pool(P['workers']) as pool:
        exs = [e for e in pool.map(C.build, specs, chunksize=1) if e is not None]
    held = {e['id'] for e in load_split('heldout')} | {e['id'] for e in load_split('val')}
    overlap = sum(e['id'] in held for e in exs)
    assert overlap == 0 and {e['family'] for e in exs} <= set(P['train_large']['families'])
    raw = ''.join(json.dumps(e, separators=(',', ':'), default=float) + '\n' for e in exs).encode()
    (ROOT / 'data').mkdir(exist_ok=True)
    with gzip.GzipFile(ROOT / 'data' / 'train_large.jsonl.gz', 'wb', mtime=0) as f:
        f.write(raw)
    man = dict(examples=len(exs), sha256_uncompressed=hashlib.sha256(raw).hexdigest(), overlap_with_val_or_heldout=overlap,
               families=sorted({e['family'] for e in exs}))
    (ROOT / 'data' / 'MANIFEST.json').write_bytes((json.dumps(man, indent=1) + '\n').encode())
    print(man)


# ------------------------------------------------------------------- encoding

def encode(e):
    prob = C.to_problem(e)
    comp, variables = fg.query_component(prob)
    adj = fg.adjacency(variables, comp.scopes)
    dist = {q: 0 for q in comp.query}
    dq = deque(comp.query)
    while dq:
        v = dq.popleft()
        for u in adj[v]:
            if u not in dist:
                dist[u] = dist[v] + 1
                dq.append(u)
    qs = set(comp.query)
    toks = []
    for s, m in zip(comp.scopes, comp.meta):
        d = min(dist.get(v, 99) for v in s)
        ar = min(len(s), 4)
        e_ = min(max(m['e'], 1e-3), 0.5)
        f = [0.0] * TOK_DIM
        f[ar - 1] = 1.0
        f[4 + KINDS.index(m['kind'])] = 1.0
        f[9] = float(m['y'])
        f[10] = m['e']
        f[11] = math.log((1 - e_) / e_) / 7.0
        f[12] = sum(v in qs for v in s) / 3.0
        f[13] = min(d, 12) / 12.0
        toks.append((d, f))
    toks.sort(key=lambda t: -t[0])
    toks = [f for _, f in toks][-P['max_tokens']:]
    ops = {o['name']: o for o in e['operators']}
    g = [math.log2(e['n']) / 8, len(variables) / 64, len(comp.scopes) / 200, len(e['problem']['query']) / 3,
         e['budget_log2_ops'] / 18, -e['decision_rule']['reward']['wrong'] / 9]
    for name in OPS:
        o = ops.get(name)
        g += [0.0, 0.0, 0.0] if o is None else [1.0, float(o['within_budget']), math.log2(max(1, o['ops'])) / 20]
    mask = [any(o['exact'] and o['within_budget'] and o['p'] is not None for o in e['operators'])]
    for name in ('mbe', 'bp', 'particle'):
        o = ops.get(name)
        mask.append(o is not None and o['within_budget'] and o['p'] is not None)
    mask.append(True)
    opt = e['selected']['option']
    label = CLASSES.index('particle' if opt.startswith('particle') else opt)
    return dict(tokens=toks or [[0.0] * TOK_DIM], glob=g, mask=mask, p_ref=e['verification']['reference_p'],
                choice=label, cost=e['targets']['log2_exact_cost'] / 20, n_tok=max(1, len(toks)))


# ---------------------------------------------------------------------- model

def build(arch, seed):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    H = P['hidden'][arch]

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            if arch == 'gru':
                self.enc = nn.GRU(TOK_DIM, H, batch_first=True)
                feat = H
            else:
                self.enc = nn.Sequential(nn.Linear(TOK_DIM, H), nn.ReLU())
                feat = 2 * H
            self.trunk = nn.Sequential(nn.Linear(feat + GLOB_DIM, P['trunk']), nn.ReLU())
            self.p = nn.Linear(P['trunk'], 1)
            self.choice = nn.Linear(P['trunk'], len(CLASSES))
            self.cost = nn.Linear(P['trunk'], 1)

        def forward(self, tok, lens, glob):
            if arch == 'gru':
                packed = nn.utils.rnn.pack_padded_sequence(tok, lens.cpu(), batch_first=True, enforce_sorted=False)
                _, h = self.enc(packed)
                z = h[-1]
            else:
                x = self.enc(tok)
                m = (torch.arange(tok.shape[1])[None, :] < lens[:, None]).float().unsqueeze(-1)
                z = torch.cat([(x * m).sum(1) / lens[:, None].float(), (x * m - (1 - m) * 1e9).max(1).values], 1)
            t = self.trunk(torch.cat([z, glob], 1))
            return self.p(t).squeeze(1), self.choice(t), self.cost(t).squeeze(1)
    return Net()


def n_params(net):
    return int(sum(p.numel() for p in net.parameters()))


def macs(arch, n_tok):
    H, T = P['hidden'][arch], P['trunk']
    enc = n_tok * (3 * (TOK_DIM * H + H * H) if arch == 'gru' else TOK_DIM * H)
    feat = H if arch == 'gru' else 2 * H
    return int(enc + (feat + GLOB_DIM) * T + T * (2 + len(CLASSES)))


def batchify(items):
    import torch
    L = max(len(it['tokens']) for it in items)
    tok = torch.zeros(len(items), L, TOK_DIM)
    for i, it in enumerate(items):
        tok[i, :len(it['tokens'])] = torch.tensor(it['tokens'])
    return (tok, torch.tensor([len(it['tokens']) for it in items]), torch.tensor([it['glob'] for it in items]),
            torch.tensor([it['mask'] for it in items]), torch.tensor([it['p_ref'] for it in items], dtype=torch.float32),
            torch.tensor([it['choice'] for it in items]), torch.tensor([it['cost'] for it in items], dtype=torch.float32))


def loss_fn(net, b):
    import torch
    import torch.nn.functional as F
    tok, lens, glob, mask, p_ref, choice, cost = b
    lp, lc, cc = net(tok, lens, glob)
    lc = lc.masked_fill(~mask, -1e9)
    return (F.binary_cross_entropy_with_logits(lp, p_ref) + F.cross_entropy(lc, choice)
            + P['cost_weight'] * F.mse_loss(cc, cost))


def encoded(name):
    cache = ROOT / 'cache' / f'{name}.json.gz'
    if cache.exists():
        return json.loads(gzip.open(cache, 'rt').read())
    with Pool(P['workers']) as pool:
        out = pool.map(encode, load_split(name), chunksize=8)
    cache.parent.mkdir(exist_ok=True)
    with gzip.GzipFile(cache, 'wb', mtime=0) as f:
        f.write(json.dumps(out).encode())
    return out


def stage_train():
    import torch
    torch.set_num_threads(P['threads'])
    train = encoded('train') + encoded('train_large')
    val = encoded('val')
    log = {}
    for arch in ('gru', 'mlp'):
        for seed in P['seeds']:
            net = build(arch, seed)
            opt = torch.optim.Adam(net.parameters(), lr=P['lr'])
            rng = random.Random(seed)
            best, best_ep, bad, hist = None, -1, 0, []
            vb = batchify(val)
            for ep in range(P['max_epochs']):
                net.train()
                idx = list(range(len(train)))
                rng.shuffle(idx)
                for s in range(0, len(idx), P['batch']):
                    b = batchify([train[i] for i in idx[s:s + P['batch']]])
                    opt.zero_grad()
                    loss = loss_fn(net, b)
                    loss.backward()
                    opt.step()
                net.eval()
                with torch.no_grad():
                    v = float(loss_fn(net, vb))
                hist.append(v)
                if best is None or v < best - 1e-5:
                    best, best_ep, bad = v, ep, 0
                    torch.save(net.state_dict(), ROOT / 'models' / f'{arch}_s{seed}.pt') if (ROOT / 'models').exists() else None
                else:
                    bad += 1
                    if bad >= P['patience']:
                        break
            log[f'{arch}_s{seed}'] = dict(best_val_loss=best, best_epoch=best_ep, epochs_run=len(hist), val_history=hist,
                                         params=n_params(net), n_train=len(train))
            print(arch, seed, round(best, 4), best_ep, n_params(net), flush=True)
    (ROOT / 'train22.json').write_bytes((json.dumps(log, indent=1) + '\n').encode())


def stage_predict():
    import torch
    rows = load_split('heldout')
    enc = encoded('heldout')
    out = {}
    for arch in ('gru', 'mlp'):
        for seed in P['seeds']:
            net = build(arch, seed)
            net.load_state_dict(torch.load(ROOT / 'models' / f'{arch}_s{seed}.pt'))
            net.eval()
            with torch.no_grad():
                b = batchify(enc)
                lp, lc, cc = net(b[0], b[1], b[2])
                lc = lc.masked_fill(~b[3], -1e9)
            preds = []
            for e, it, p, k, c in zip(rows, enc, torch.sigmoid(lp).tolist(), lc.argmax(1).tolist(), cc.tolist()):
                cls = CLASSES[k]
                ops = {o['name']: o for o in e['operators']}
                if cls == 'exact':
                    ex = [o for o in e['operators'] if o['exact'] and o['within_budget'] and o['p'] is not None]
                    pick = min(ex, key=lambda o: o['ops'])
                elif cls == 'abstain':
                    pick = None
                else:
                    pick = ops[cls]
                sel = dict(id=e['id'], algorithm='abstain' if pick is None else pick['name'],
                           action=-1 if pick is None else int(pick['action']), claimed_ops=0 if pick is None else int(pick['ops']))
                if pick is not None and pick['p'] is not None:
                    sel['p'] = float(pick['p'])
                pen = -e['decision_rule']['reward']['wrong']
                post = dict(id=e['id'], algorithm='learned_posterior', action=int(fg.decide(min(max(p, 0.0), 1.0), pen)),
                            p=float(min(max(p, 0.0), 1.0)))
                preds.append(dict(selector=sel, posterior=post, cost_pred=c * 20, macs=macs(arch, it['n_tok'])))
            out[f'{arch}_s{seed}'] = preds
    (ROOT / 'predictions22.json').write_bytes((json.dumps(out) + '\n').encode())
    for k, v in out.items():
        with open(ROOT / 'preds' / f'{k}_selector.jsonl', 'w', newline='\n') as f:
            for pr in v:
                f.write(json.dumps(pr['selector']) + '\n')
        with open(ROOT / 'preds' / f'{k}_posterior.jsonl', 'w', newline='\n') as f:
            for pr in v:
                f.write(json.dumps(pr['posterior']) + '\n')
    print('predictions written', list(out))


def bootstrap_lower(diffs, seed, reps=2000):
    rng = np.random.default_rng(seed)
    d = np.asarray(diffs)
    means = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(reps)]
    return float(np.percentile(means, 2.5)), float(d.mean())


def stage_report():
    rows = load_split('heldout')
    preds = json.loads((ROOT / 'predictions22.json').read_text())
    pens = [-e['decision_rule']['reward']['wrong'] for e in rows]
    refs = [e['verification']['reference_p'] for e in rows]
    truths = [e['verification']['truth_parity'] for e in rows]

    def eu(a, i):
        return fg.expected_utility(a, refs[i], pens[i])
    sym, bp, sym_ops = [], [], []
    for i, e in enumerate(rows):
        cap = 2 ** e['budget_log2_ops']
        ex = [o for o in e['operators'] if o['exact'] and o['ops'] <= cap and o['p'] is not None]
        mb = [o for o in e['operators'] if o['name'] == 'mbe' and o['ops'] <= cap and o['p'] is not None]
        pick = min(ex, key=lambda o: o['ops']) if ex else (mb[0] if mb else None)
        sym.append(0.0 if pick is None else eu(int(pick['action']), i))
        sym_ops.append(0 if pick is None else pick['ops'])
        b = [o for o in e['operators'] if o['name'] == 'bp' and o['within_budget'] and o['p'] is not None]
        bp.append(eu(fg.decide(b[0]['p'], pens[i]), i) if b else 0.0)
    res = dict(n=len(rows), symbolic=dict(eu=float(np.mean(sym)), mean_ops=float(np.mean(sym_ops))),
               always_bp=dict(eu=float(np.mean(bp))), models={})
    for k, v in preds.items():
        s_eu = [eu(pr['selector']['action'], i) for i, pr in enumerate(v)]
        p_eu = [eu(pr['posterior']['action'], i) for i, pr in enumerate(v)]
        ph = np.array([pr['posterior']['p'] for pr in v])
        lo1, d1 = bootstrap_lower(np.array(s_eu) - np.array(sym), P['bootstrap_seed'])
        lo3, d3 = bootstrap_lower(np.array(p_eu) - np.array(bp), P['bootstrap_seed'])
        cp = np.array([pr['cost_pred'] for pr in v])
        ct = np.array([e['targets']['log2_exact_cost'] for e in rows])
        res['models'][k] = dict(
            selector_eu=float(np.mean(s_eu)), selector_minus_symbolic=d1, selector_minus_symbolic_lower95=lo1,
            selector_mean_ops=float(np.mean([pr['selector']['claimed_ops'] for pr in v])),
            selector_alg_match=float(np.mean([pr['selector']['algorithm'] == e['selected']['operator'] for pr, e in zip(v, rows)])),
            posterior_eu=float(np.mean(p_eu)), posterior_minus_bp=d3, posterior_minus_bp_lower95=lo3,
            posterior_mse_vs_reference=float(np.mean((ph - np.array(refs)) ** 2)),
            posterior_brier_vs_truth=float(np.mean((ph - np.array(truths)) ** 2)),
            posterior_coverage=float(np.mean([pr['posterior']['action'] >= 0 for pr in v])),
            posterior_selective_risk=float(sum(pr['posterior']['action'] >= 0 and pr['posterior']['action'] != t for pr, t in zip(v, truths))
                                           / max(1, sum(pr['posterior']['action'] >= 0 for pr in v))),
            cost_r2=float(1 - ((cp - ct) ** 2).sum() / ((ct - ct.mean()) ** 2).sum()),
            mean_macs=float(np.mean([pr['macs'] for pr in v])),
            per_family_posterior_minus_bp={f: float(np.mean([p_eu[i] - bp[i] for i, e in enumerate(rows) if e['family'] == f]))
                                           for f in sorted({e['family'] for e in rows})})
    seeds = P['seeds']
    g = lambda arch, key: [res['models'][f'{arch}_s{s}'][key] for s in seeds]
    res['gates'] = dict(
        G1=dict(rule=P['gates']['G1'], per_seed_lower95=g('gru', 'selector_minus_symbolic_lower95'),
                per_seed_mean_ops=g('gru', 'selector_mean_ops'), symbolic_mean_ops=res['symbolic']['mean_ops'],
                passed=sum(lo > -0.01 and o < res['symbolic']['mean_ops'] for lo, o in
                           zip(g('gru', 'selector_minus_symbolic_lower95'), g('gru', 'selector_mean_ops'))) >= 2),
        G2=dict(rule=P['gates']['G2'], gru=g('gru', 'posterior_mse_vs_reference'), mlp=g('mlp', 'posterior_mse_vs_reference'),
                passed=float(np.mean(g('gru', 'posterior_mse_vs_reference'))) < float(np.mean(g('mlp', 'posterior_mse_vs_reference')))
                and sum(a < b for a, b in zip(g('gru', 'posterior_mse_vs_reference'), g('mlp', 'posterior_mse_vs_reference'))) >= 2),
        G3=dict(rule=P['gates']['G3'], per_seed_lower95=g('gru', 'posterior_minus_bp_lower95'),
                passed=sum(lo > 0 for lo in g('gru', 'posterior_minus_bp_lower95')) >= 2))
    (ROOT / 'results22.json').write_bytes((json.dumps(res, indent=1) + '\n').encode())
    print('symbolic', res['symbolic'], 'always_bp', res['always_bp'])
    for k, m in res['models'].items():
        print(k, {x: round(y, 4) for x, y in m.items() if isinstance(y, float)})
    print({k: v['passed'] for k, v in res['gates'].items()})


if __name__ == '__main__':
    for d in ('models', 'preds'):
        (ROOT / d).mkdir(exist_ok=True)
    {'data': stage_data, 'train': stage_train, 'predict': stage_predict, 'report': stage_report}[sys.argv[1]]()

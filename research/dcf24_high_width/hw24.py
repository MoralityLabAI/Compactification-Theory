"""DCF-24: the frozen DCF-23 message-passing correction in the high-width regime.

DCF-23's model improved on BP but cost far more than exact inference on DCF-16's
records, whose graphs are mostly narrow. Here the instances are fresh and wide
(dense n 16-28, grids n 64-144, sparse random n 48-128) but still verifiable offline:
each reference posterior is accepted only when two independent exact paths agree
(DCF-14 run_instance). No model is retrained; the three mp_bp seeds from DCF-23 are
loaded as frozen.

Per instance: exact cost = counted ops of the cheapest exact operator (symbolic, known
before running); BP path = 50-iteration loopy BP (counted ops); model path = BP ops +
model multiply-accumulates. Policies at a compute cap C: exact if its ops <= C, else
the fallback (BP or the learned correction).

Usage: python hw24.py run | report
"""
import gzip
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'dcf16_decision_curriculum'))
sys.path.insert(0, str(ROOT.parent / 'dcf23_message_passing'))
sys.path.insert(0, str(ROOT.parent / 'dcf14_width_routing'))
import curriculum16 as C  # noqa: E402
import fg  # noqa: E402
import mp23  # noqa: E402

P = json.loads((ROOT / 'PROTOCOL.json').read_text())


def specs():
    out = []
    for block in P['blocks']:
        out += C._block(block, 'high_width')
    for s in out:
        s['budget'] = P['record_budget']
    return out


def build(spec):
    return C.build(spec)


def run():
    with Pool(P['workers']) as pool:
        recs = [e for e in pool.map(build, specs(), chunksize=1) if e is not None]
    raw = ''.join(json.dumps(e, separators=(',', ':'), default=float) + '\n' for e in recs).encode()
    with gzip.GzipFile(ROOT / 'records24.jsonl.gz', 'wb', mtime=0) as f:
        f.write(raw)
    with Pool(P['workers']) as pool:
        feats = pool.map(mp23.featurize, recs, chunksize=1)
    with gzip.GzipFile(ROOT / 'features24.json.gz', 'wb', mtime=0) as f:
        f.write(json.dumps(feats).encode())
    print(len(specs()), 'generated;', len(recs), 'verified records')


def predict(feats):
    import torch
    out = {}
    for seed in P['seeds']:
        net = mp23.build('mp_bp', seed)
        net.load_state_dict(torch.load(ROOT.parent / 'dcf23_message_passing' / 'models' / f'mp_bp_s{seed}.pt', weights_only=True))
        net.eval()
        with torch.no_grad():
            ps = []
            for s in range(0, len(feats), 16):
                ps += torch.sigmoid(net(mp23.collate(feats[s:s + 16]))).tolist()
        out[seed] = ps
    return out


def bootstrap_lower(d, seed, reps=2000):
    rng = np.random.default_rng(seed)
    d = np.asarray(d, float)
    if len(d) == 0:
        return None, None
    return float(np.percentile([d[rng.integers(0, len(d), len(d))].mean() for _ in range(reps)], 2.5)), float(d.mean())


def report():
    recs = [json.loads(line) for line in gzip.open(ROOT / 'records24.jsonl.gz', 'rt')]
    feats = json.loads(gzip.open(ROOT / 'features24.json.gz', 'rt').read())
    preds = predict(feats)
    pen = [it['penalty'] for it in feats]
    ref = [it['p_ref'] for it in feats]

    def eu(p, i):
        return fg.expected_utility(fg.decide(float(p), pen[i]), ref[i], pen[i])
    n = len(feats)
    exact_ops = []
    exact_eu = []
    for i, e in enumerate(recs):
        ex = [o for o in e['operators'] if o['exact'] and o['p'] is not None]
        o = min(ex, key=lambda o: o['ops'])
        exact_ops.append(o['ops'])
        exact_eu.append(eu(o['p'], i))
    bp_ops = [it['bp_ops'] for it in feats]
    macs = [mp23.macs('mp_bp', it) for it in feats]
    bp_eu = [eu(it['p_bp'], i) for i, it in enumerate(feats)]
    widths = [e['candidate_representations'][3]['induced_width'] for e in recs]
    cheaper = [exact_ops[i] > bp_ops[i] + macs[i] for i in range(n)]
    res = dict(n=n, families={f: sum(e['family'] == f for e in recs) for f in sorted({e['family'] for e in recs})},
               width=dict(min=int(min(widths)), median=float(np.median(widths)), max=int(max(widths))),
               ops=dict(exact_median=float(np.median(exact_ops)), bp_median=float(np.median(bp_ops)), model_macs_median=float(np.median(macs))),
               exact_eu=float(np.mean(exact_eu)), bp=dict(eu=float(np.mean(bp_eu)), mse=float(np.mean([(it['p_bp'] - it['p_ref']) ** 2 for it in feats]))),
               model_cheaper_than_exact=int(sum(cheaper)), seeds={}, policies={})
    for seed, ps in preds.items():
        m_eu = [eu(p, i) for i, p in enumerate(ps)]
        lo1, d1 = bootstrap_lower(np.array(m_eu) - np.array(bp_eu), P['bootstrap_seed'])
        idx = [i for i in range(n) if cheaper[i]]
        lo2, d2 = bootstrap_lower([m_eu[i] - bp_eu[i] for i in idx], P['bootstrap_seed'])
        res['seeds'][str(seed)] = dict(eu=float(np.mean(m_eu)), mse=float(np.mean([(p - r) ** 2 for p, r in zip(ps, ref)])),
                                       eu_minus_bp=d1, eu_minus_bp_lower95=lo1, cheaper_regime_eu_minus_bp=d2, cheaper_regime_lower95=lo2,
                                       per_family_eu_minus_bp={f: float(np.mean([m_eu[i] - bp_eu[i] for i in range(n) if recs[i]['family'] == f]))
                                                               for f in res['families']})
        for cap in P['caps']:
            C_ = 2 ** cap
            mp_pol, bp_pol, mp_cost, bp_cost = [], [], [], []
            for i in range(n):
                if exact_ops[i] <= C_:
                    mp_pol.append(exact_eu[i]); bp_pol.append(exact_eu[i])
                    mp_cost.append(exact_ops[i]); bp_cost.append(exact_ops[i])
                else:
                    mp_pol.append(m_eu[i]); bp_pol.append(bp_eu[i])
                    mp_cost.append(bp_ops[i] + macs[i]); bp_cost.append(bp_ops[i])
            lo3, d3 = bootstrap_lower(np.array(mp_pol) - np.array(bp_pol), P['bootstrap_seed'])
            res['policies'][f'cap{cap}_s{seed}'] = dict(exact_fraction=float(np.mean([exact_ops[i] <= C_ for i in range(n)])),
                                                        mp_policy_eu=float(np.mean(mp_pol)), bp_policy_eu=float(np.mean(bp_pol)),
                                                        diff=d3, lower95=lo3, mp_policy_mean_cost=float(np.mean(mp_cost)),
                                                        bp_policy_mean_cost=float(np.mean(bp_cost)))
    S = [str(s) for s in P['seeds']]
    res['gates'] = dict(
        H1=dict(rule=P['gates']['H1'], lower95=[res['seeds'][s]['eu_minus_bp_lower95'] for s in S],
                passed=sum((res['seeds'][s]['eu_minus_bp_lower95'] or -1) > 0 for s in S) >= 2),
        H2=dict(rule=P['gates']['H2'], instances=res['model_cheaper_than_exact'], lower95=[res['seeds'][s]['cheaper_regime_lower95'] for s in S],
                passed=res['model_cheaper_than_exact'] >= P['h2_min_instances'] and sum((res['seeds'][s]['cheaper_regime_lower95'] or -1) > 0 for s in S) >= 2),
        H3=dict(rule=P['gates']['H3'], lower95={c: [res['policies'][f'cap{c}_s{s}']['lower95'] for s in S] for c in P['caps']},
                passed=all(sum((res['policies'][f'cap{c}_s{s}']['lower95'] or -1) > 0 for s in S) >= 2 for c in P['caps'])))
    (ROOT / 'results24.json').write_bytes((json.dumps(res, indent=1) + '\n').encode())
    print({k: res[k] for k in ('n', 'families', 'width', 'ops', 'exact_eu', 'bp', 'model_cheaper_than_exact')})
    for s, v in res['seeds'].items():
        print('seed', s, {k: (round(x, 4) if isinstance(x, float) else x) for k, x in v.items() if not isinstance(x, dict)})
    for k, v in res['policies'].items():
        print(k, {x: round(y, 4) if isinstance(y, float) else y for x, y in v.items()})
    print({k: v['passed'] for k, v in res['gates'].items()})


if __name__ == '__main__':
    {'run': run, 'report': report}[sys.argv[1]]()

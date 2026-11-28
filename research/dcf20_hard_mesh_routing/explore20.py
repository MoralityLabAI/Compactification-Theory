"""DCF-20 exploratory (not gated): routing headroom where BP errs most, across compute prices.

World: the scan's highest decision-change world (low noise, every cross sensor XOR,
two active pairs, aligned queries). Fresh seeds. For each lambda in LAMBDAS, every
policy family is re-scored as EU - lambda * (kernel + routing ms), and the best
parameter of each family is chosen per lambda on this same data. That makes the
family rows optimistic upper bounds, labelled as such; the oracles are privileged
bounds by construction.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import world20 as W  # noqa: E402

WORLD = dict(noise='low', xor_frac=1.0, triangle=False, aligned=True)
RHOS = [0.3, 0.5, 0.7]
LAMBDAS = [0.0, 0.003, 0.01, 0.03, 0.1, 0.3]
BASE = 880000


def objective(summary, lam):
    return summary['eu'] - lam * (summary['kernel_ms'] + summary['route_ms'])


def main():
    traces = [dict(rho=r, trace=W.trace(W.seeds(BASE, i, 64), r, WORLD, 32)) for i, r in enumerate(RHOS)]
    dcr = {str(c['rho']): W.decision_change_rate(c['trace']) for c in traces}
    fams = {'fixed_bp': [None], 'fixed_exact': [None], 'epoch_oracle': [None], 'query_oracle': [None],
            'regime': [dict(tau=t) for t in W.P['tau_grid']],
            'probe': [dict(delta=d, phi=f) for d in W.P['delta_grid'] for f in W.P['phi_grid']],
            'escalate': [dict(delta=d) for d in W.P['delta_grid']]}
    summ = {}
    for fam, plist in fams.items():
        for p in plist:
            summ[(fam, json.dumps(p))] = [W.R19.summary(W.evaluate(c['trace'], fam, p)) for c in traces]
    out = dict(world=WORLD, rhos=RHOS, decision_change_rate=dcr, by_lambda={})
    for lam in LAMBDAS:
        row = {}
        for fam, plist in fams.items():
            best = max(plist, key=lambda p: np.mean([objective(s, lam) for s in summ[(fam, json.dumps(p))]]))
            ss = summ[(fam, json.dumps(best))]
            row[fam] = dict(params=best, mean_objective=float(np.mean([objective(s, lam) for s in ss])),
                            mean_eu=float(np.mean([s['eu'] for s in ss])), exact_fraction=float(np.mean([s['exact_fraction'] for s in ss])),
                            kernel_ms=float(np.mean([s['kernel_ms'] for s in ss])))
        bf = max(row['fixed_bp']['mean_objective'], row['fixed_exact']['mean_objective'])
        row['_headroom'] = dict(epoch=row['epoch_oracle']['mean_objective'] - bf, query=row['query_oracle']['mean_objective'] - bf,
                                best_family_gain={f: row[f]['mean_objective'] - bf for f in ('regime', 'probe', 'escalate')})
        out['by_lambda'][str(lam)] = row
        print(lam, {k: round(v['mean_objective'], 4) for k, v in row.items() if not k.startswith('_')}, row['_headroom'], flush=True)
    W.save_trace('trace20_explore.json.gz', traces)
    (ROOT / 'explore20.json').write_bytes((json.dumps(out, indent=1) + '\n').encode())
    print('decision change', dcr)


if __name__ == '__main__':
    main()

"""DCF-20B (gated): escalate-on-abstention in the hardest mesh world.

Hypothesis from explore20.json (post-hoc there, tested here on fresh seeds): when
BP's posterior falls inside the abstention band (0.2 < p < 0.8) the exact operator is
far more likely to change the decision (95% of decision changes in the exploratory
traces). Policy, with no tunable parameter: run BP for every query in the step (one
launch), then run exact for the queries BP abstains on (second launch, only if any).

Usage: python gated20b.py run | report
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import world20 as W  # noqa: E402

P = json.loads((ROOT / 'PROTOCOL20B.json').read_text())


def escalate_on_abstain(tr):
    rows = []
    for b in tr:
        nb = len(b['seeds'])
        for s in b['steps']:
            sel = [W.R19.decide(q['bp']['p']) < 0 for q in s['queries']]
            bpw = sum(q['bp']['work'] for q in s['queries'])
            exw = sum(q['exact']['work'] for q, k in zip(s['queries'], sel) if k)
            st = W.M.op_seconds(W.CM, 'bp', bpw) + W.M.op_seconds(W.CM, 'update', s['update_work'])
            if any(sel):
                st += W.M.op_seconds(W.CM, 'exact', exw)
            for q, k in zip(s['queries'], sel):
                r = q['exact'] if k else q['bp']
                rows.append(dict(op='exact' if k else 'bp', eu=r['eu'], regret=q['oracle'] - r['eu'], realized=r['realized'],
                                 act=r['act'], wrong=r['wrong'], work=r['work'], step_seconds_share=st / nb, route_seconds=0.0,
                                 launches=1 + int(any(sel))))
    return rows


def objective(s, lam):
    return s['eu'] - lam * (s['kernel_ms'] + s['route_ms'])


def run():
    traces = [dict(rho=r, trace=W.trace(W.seeds(P['seed_base'], i, P['episodes']), r, P['world'], P['batch']))
              for i, r in enumerate(P['rho'])]
    W.save_trace('trace20b.json.gz', traces)


def report():
    traces = W.load_trace('trace20b.json.gz')
    res = dict(world=P['world'], by_lambda={}, decision_change_rate={str(c['rho']): W.decision_change_rate(c['trace']) for c in traces})
    summ = []
    for c in traces:
        summ.append(dict(rho=c['rho'], esc=W.R19.summary(escalate_on_abstain(c['trace'])),
                         bp=W.R19.summary(W.evaluate(c['trace'], 'fixed_bp')), ex=W.R19.summary(W.evaluate(c['trace'], 'fixed_exact'))))
    for lam in P['lambdas']:
        per = []
        for s in summ:
            e, b, x = objective(s['esc'], lam), objective(s['bp'], lam), objective(s['ex'], lam)
            per.append(dict(rho=s['rho'], escalate=e, fixed_bp=b, fixed_exact=x, beats_better_fixed=bool(e > max(b, x) + 1e-12)))
        wins = sum(p['beats_better_fixed'] for p in per)
        res['by_lambda'][str(lam)] = dict(per_condition=per, wins=wins, conditions=len(per), passed=2 * wins > len(per),
                                          mean=dict(escalate=float(np.mean([p['escalate'] for p in per])),
                                                    fixed_bp=float(np.mean([p['fixed_bp'] for p in per])),
                                                    fixed_exact=float(np.mean([p['fixed_exact'] for p in per]))))
    res['escalated_fraction'] = float(np.mean([s['esc']['exact_fraction'] for s in summ]))
    res['eu'] = dict(escalate=float(np.mean([s['esc']['eu'] for s in summ])), fixed_bp=float(np.mean([s['bp']['eu'] for s in summ])),
                     fixed_exact=float(np.mean([s['ex']['eu'] for s in summ])))
    res['selective_risk'] = dict(escalate=float(np.mean([s['esc']['selective_risk'] for s in summ])),
                                 fixed_exact=float(np.mean([s['ex']['selective_risk'] for s in summ])))
    (ROOT / 'results20b.json').write_bytes((json.dumps(res, indent=1) + '\n').encode())
    for lam, v in res['by_lambda'].items():
        print(lam, v['wins'], '/', v['conditions'], 'passed' if v['passed'] else 'FAIL', {k: round(x, 4) for k, x in v['mean'].items()})
    print('dcr', res['decision_change_rate'], 'escalated', round(res['escalated_fraction'], 3), res['eu'])


if __name__ == '__main__':
    {'run': run, 'report': report}[sys.argv[1]]()

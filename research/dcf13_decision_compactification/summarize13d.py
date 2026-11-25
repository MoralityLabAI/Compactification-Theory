"""Summarise DCF-13D receipts and evaluate the predeclared gates (writes results13d.json)."""
import json, collections
import numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent
A = json.loads((ROOT / 'receipt13d_a.json').read_text()); B = json.loads((ROOT / 'receipt13d_b.json').read_text()); C = json.loads((ROOT / 'receipt13d_c.json').read_text())
out = {}
def agg(rows, keys, by=('arm',)):
    g = collections.defaultdict(list)
    for r in rows:
        g[tuple(r[b] for b in by)].append(r)
    return {'/'.join(map(str, k)): {kk: float(np.mean([r[kk] for r in v if r.get(kk) is not None])) for kk in keys if any(r.get(kk) is not None for r in v)} for k, v in g.items()}
ra = A['rows']; key = lambda r: (r['k'], r['seed'], r['scenario'])
by = collections.defaultdict(dict)
for r in ra: by[key(r)][r['arm']] = r
conds = sorted(by)
a1 = all(by[c]['hybrid_cost']['max_delta'] < 1e-10 and by[c]['hybrid_cost']['skip'] == 0 and by[c]['hybrid_cost']['actions'] if False else by[c]['hybrid_cost']['max_delta'] < 1e-10 and by[c]['hybrid_cost']['skip'] == 0 and abs(by[c]['hybrid_cost']['expected_utility'] - by[c]['full']['expected_utility']) < 1e-12 for c in conds)
a2 = sum(by[c]['hybrid_cost']['seconds'] <= min(by[c]['component_full']['seconds'], by[c]['ve_full']['seconds']) for c in conds)
h13 = sum(by[c]['hybrid13c']['seconds'] <= min(by[c]['component_full']['seconds'], by[c]['ve_full']['seconds']) for c in conds)
vs = sum(by[c]['hybrid_cost']['seconds'] < by[c]['hybrid13c']['seconds'] for c in conds)
out['A'] = dict(gate_A1_exact=a1, gate_A2_passes=a2, gate_A2_required=27, conditions=len(conds), gate_A3=A['audit'], alpha=A['alpha'],
    hybrid13c_same_latency_criterion=h13, hybrid_cost_faster_than_hybrid13c=vs,
    median_speedup_vs_full=float(np.median([by[c]['full']['seconds'] / by[c]['hybrid_cost']['seconds'] for c in conds])),
    arms=agg(ra, ['expected_utility', 'regret', 'coverage', 'seconds', 'skip']),
    by_k_scenario={f'{k}/{sc}': {arm: float(np.mean([by[(k, s, sc)][arm]['seconds'] * 1000 for s in (1301, 1303, 1307)])) for arm in by[conds[0]]} for k in (8, 12, 16) for sc in ('stable', 'rewired', 'bridge', 'ambiguous')},
    routes=agg([r for r in ra if r['arm'] == 'hybrid_cost'], ['route_enum_bound', 'route_enum_refined', 'route_ve'], ('k', 'scenario')),
    failing_conditions=[dict(k=c[0], seed=c[1], scenario=c[2], hybrid_ms=by[c]['hybrid_cost']['seconds'] * 1000, comp_ms=by[c]['component_full']['seconds'] * 1000, ve_ms=by[c]['ve_full']['seconds'] * 1000) for c in conds if by[c]['hybrid_cost']['seconds'] > min(by[c]['component_full']['seconds'], by[c]['ve_full']['seconds'])])
rb = B['rows']
out['B'] = dict(arms=agg(rb, ['expected_utility', 'regret', 'coverage', 'selective_risk', 'skip', 'mean_ops', 'max_union', 'mean_union', 'max_peak_entries', 'tracemalloc_peak_bytes', 'median_episode_seconds', 'ece', 'brier']),
    arms_by_k=agg(rb, ['expected_utility', 'regret', 'coverage', 'skip', 'mean_union', 'mean_ops'], ('k', 'arm')),
    widths=agg(B['widths'], ['min_degree', 'min_fill', 'weighted_min_fill', 'comp_vars'], ('k', 'scenario')),
    width_disagreements=dict(min_fill_lt_min_degree=sum(w['min_fill'] < w['min_degree'] for w in B['widths']), min_fill_gt_min_degree=sum(w['min_fill'] > w['min_degree'] for w in B['widths']), wmf_lt_mf=sum(w['weighted_min_fill'] < w['min_fill'] for w in B['widths']), wmf_gt_mf=sum(w['weighted_min_fill'] > w['min_fill'] for w in B['widths']), n=len(B['widths'])))
rc = [r for r in C['rows'] if r['arm'] != '_timing']
byc = collections.defaultdict(dict)
for r in rc: byc[(r['k'], r['seed'])][r['arm']] = r
cc = sorted(byc)
out['C'] = dict(arms=agg(rc, ['expected_utility', 'realized_utility' if False else 'utility', 'regret', 'coverage', 'wrong', 'selective_risk', 'ece', 'brier', 'negative_expected_utility_actions']),
    gate_C1=dict(max_bma_delta=max(c['bma_delta'] for c in C['checks']), max_oracle_delta=max(c['oracle_delta'] for c in C['checks']), audited=len(C['checks'])),
    gate_C2=dict(min_grid_confidence=min([c['credal_min_grid_confidence'] for c in C['checks'] if c['credal_min_grid_confidence'] is not None] or [None]),
                 negative_eu_actions=sum(r.get('negative_expected_utility_actions', 0) for r in rc)),
    gate_C3=dict(bma_mean_eu=float(np.mean([byc[c]['bma_exact']['expected_utility'] for c in cc])), plugin_mean_eu=float(np.mean([byc[c]['plugin_half']['expected_utility'] for c in cc])),
                 credal_cov_gt_pointwise=sum(byc[c]['credal_shared_d6']['coverage'] > byc[c]['pointwise_robust']['coverage'] for c in cc), conditions=len(cc),
                 bma_beats_plugin_conditions=sum(byc[c]['bma_exact']['expected_utility'] > byc[c]['plugin_half']['expected_utility'] for c in cc)),
    relevance=agg(C['relevance'], ['relevant_vars', 'query_component_vars'], ('k',)),
    timing=[r for r in C['rows'] if r['arm'] == '_timing'])
(ROOT / 'results13d.json').write_text(json.dumps(out, indent=1) + '\n')
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ('by_k_scenario', 'arms_by_k', 'widths', 'timing', 'failing_conditions')} for k, v in out.items()}, indent=1)[:9000])

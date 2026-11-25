"""DCF-16: verifier-grounded curriculum of computational decisions.

Every example is a query on a binary factor graph (DCF-14 generator), the candidate
dependency representations, the operators available under a compute budget with
their exact symbolic costs, the selected strategy, intermediate receipts, the final
decision or abstention, the exact verification outcome and the regret table.

Usage: python curriculum16.py generate   -> data/*.jsonl.gz, MANIFEST.json
       python curriculum16.py verify     -> re-derives every reference posterior from the
                                            serialised problem and checks split hygiene
"""
import gzip
import hashlib
import json
import math
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'dcf14_width_routing'))
import numpy as np  # noqa: E402
import fg  # noqa: E402
import width14 as W  # noqa: E402
from gen import make_problem  # noqa: E402

P = json.loads((ROOT / 'PROTOCOL.json').read_text())
SCHEMA = 'dcf16.decision_trace.v1'


def specs(split):
    blocks = P['splits'][split]
    out = []
    for cfg in (blocks if isinstance(blocks, list) else [blocks]):
        out += _block(cfg, split)
    return out


def _block(cfg, split):
    out = []
    for fam in cfg['families']:
        for n in cfg['sizes']:
            for s in range(cfg['seeds']):
                out.append(dict(family=fam, n=n, seed=cfg['seed_base'] + 1000 * s + n, noise=('low', 'high')[s % 2],
                                query_size=1 + s % 3, misspec=(s % 3 == 2) and fam != 'xor_noiseless',
                                budget=P['budgets'][s % len(P['budgets'])], penalty=P['penalties'][(s // 3) % len(P['penalties'])],
                                split=split))
    return out


def signature(prob):
    canon = dict(n=prob.n, f=[(list(s), m['kind'], m['y'], round(m['e'], 12)) for s, m in zip(prob.scopes, prob.meta)], q=list(prob.query))
    return hashlib.sha256(json.dumps(canon, sort_keys=True).encode()).hexdigest()


def elimination_trace(scopes, order, keep):
    live = [frozenset(s) for s in scopes] + [frozenset([q]) for q in keep]
    trace = []
    for v in order:
        bucket = [s for s in live if v in s]
        if not bucket:
            continue
        union = frozenset().union(*bucket)
        trace.append(dict(var=int(v), union=len(union), factors=len(bucket), ops=2 ** len(union) * len(bucket)))
        live = [s for s in live if v not in s] + [union - {v}]
    return trace


def subproblem_decision(prob, keep_factor, penalty):
    idx = [i for i, s in enumerate(prob.scopes) if keep_factor(s)]
    sub = fg.Problem(prob.n, [prob.scopes[i] for i in idx], [prob.tables[i] for i in idx], prob.query, [prob.meta[i] for i in idx])
    p = fg.variable_elimination(sub)['p']
    return fg.decide(p, penalty), p


def build(spec):
    prob = make_problem(spec['family'], spec['n'], spec['seed'], spec['noise'], spec['query_size'], spec['misspec'])
    rec = W.run_instance({k: spec[k] for k in ('family', 'n', 'seed', 'noise', 'query_size', 'misspec')})
    if not rec['verified']:
        return None
    B, pen = spec['budget'], spec['penalty']
    comp, variables = fg.query_component(prob)
    order, width = fg.elimination_order(variables, comp.scopes, comp.query, 'min_fill')
    cut = fg.loop_cutset(variables, comp.scopes, comp.query)
    ref = rec['reference']
    exact_dec = fg.decide(ref, pen)
    ops_cap = 2 ** B
    operators = []
    for name, m in rec['methods'].items():
        if name.split('_B')[0] in ('bp', 'particle', 'mbe') and not name.endswith(f'_B{B}'):
            continue
        exact = name in W.EXACT_METHODS or name.startswith('ve_')
        a = fg.decide(m['p'], pen) if m.get('p') is not None else -1
        if name.startswith('particle') and m.get('p') is not None:
            a = fg.decide(m['p'], pen, lo=m['p'] - m['half_width'], hi=m['p'] + m['half_width'])
        operators.append(dict(name=name.split('_B')[0], exact=exact, ops=int(m['ops']), within_budget=bool(m['ops'] <= ops_cap),
                              p=m.get('p'), action=a, regret=fg.oracle_utility(ref, pen) - fg.expected_utility(a, ref, pen)))
    label = W.label_best(rec, B, pen, P['lambda'])
    opts = W.options(rec, B, pen)
    sel_op = opts.get('exact_method', 'exact') if label == 'exact' else label
    p_sel = opts[label][0] if label != 'abstain' else None
    a_sel = W.option_value(rec, opts[label], pen)[1]
    if label == 'exact':
        depth = len(elimination_trace(comp.scopes, order, comp.query)) if sel_op.startswith('ve') else (
            len(variables) if sel_op != 'gf2' else len(comp.scopes))
    elif label == 'bp':
        depth = rec['methods'][f'bp_B{B}'].get('iters', 0)
    else:
        depth = 0
    qset = set(prob.query)
    loc_a, loc_p = subproblem_decision(prob, lambda s: len(s) == 1 and s[0] in qset, pen)
    hop_a, hop_p = subproblem_decision(prob, lambda s: bool(set(s) & qset), pen)
    bp = rec['methods'].get(f'bp_B{B}')
    bp_a = fg.decide(bp['p'], pen) if bp and bp.get('p') is not None else None
    edges = sorted({(min(a, b), max(a, b)) for s in comp.scopes for a in s for b in s if a < b})
    ex = dict(
        schema=SCHEMA, id=signature(prob)[:20], split=spec['split'], family=spec['family'], n=spec['n'], seed=spec['seed'],
        conditions=dict(noise=spec['noise'], query_size=spec['query_size'], model_misspecified=spec['misspec']),
        problem=dict(n_vars=prob.n, prior='independent uniform bits', query=list(prob.query), query_type='parity',
                     factors=[dict(scope=list(s), kind=m['kind'], y=int(m['y']), noise=float(m['e'])) for s, m in zip(prob.scopes, prob.meta)]),
        observation_history=list(range(len(prob.scopes))),
        decision_rule=dict(reward=dict(correct=1, wrong=-pen, abstain=0), act_threshold=pen / (1 + pen)),
        budget_log2_ops=B,
        candidate_representations=[
            dict(name='unary_evidence_only', variables=sorted(qset), sufficient=loc_a == exact_dec, p=loc_p),
            dict(name='one_hop_neighbourhood', variables=sorted({v for s in prob.scopes if set(s) & qset for v in s}), sufficient=hop_a == exact_dec, p=hop_p),
            dict(name='query_component', variables=variables, sufficient=True, reason='posterior factorises over evidence components'),
            dict(name='min_fill_elimination_order', order=[int(v) for v in order], induced_width=width,
                 max_table_entries=2 ** rec['features']['max_union_min_fill'], sufficient=True),
            dict(name='loop_cutset', variables=[int(v) for v in cut], sufficient=True, enumerations=2 ** len(cut)),
        ],
        operators=operators,
        selected=dict(option=label, operator=sel_op, ops=int(opts[label][1]), recurrent_depth=int(depth)),
        receipts=dict(elimination_trace=elimination_trace(comp.scopes, order, comp.query),
                      bp=None if not bp else dict(iters=bp.get('iters'), converged=bp.get('converged'), p=bp.get('p'))),
        final=dict(action=int(a_sel), p_model=p_sel),
        verification=dict(reference_p=ref, verified_by=[c[0] for c in rec['reference_checks']],
                          max_disagreement=max(c[1] for c in rec['reference_checks']), exact_action=exact_dec,
                          truth_parity=rec['truth'], reference_model='generating model' if spec['misspec'] else 'solver model'),
        regret=fg.oracle_utility(ref, pen) - fg.expected_utility(a_sel, ref, pen),
        targets=dict(relevant_variables=variables, dependency_edges=[list(e) for e in edges],
                     log2_exact_cost=math.log2(max(1, min([o['ops'] for o in operators if o['exact']] or [1]))),
                     algorithm=sel_op, recurrent_depth=int(depth), p_reference=ref, abstain=bool(a_sel < 0)),
        hard_negative=dict(unary_shortcut_wrong=bool(loc_a != exact_dec), one_hop_wrong=bool(hop_a != exact_dec),
                           bp_wrong=bool(bp_a is not None and bp_a != exact_dec)),
    )
    return ex


def generate():
    out_dir = ROOT / 'data'
    out_dir.mkdir(exist_ok=True)
    manifest = dict(schema=SCHEMA, protocol=P, files={})
    sigs = {}
    for split in ('train', 'val', 'heldout'):
        with Pool(P['workers']) as pool:
            exs = [e for e in pool.map(build, specs(split), chunksize=1) if e is not None]
        sigs[split] = {e['id'] for e in exs}
        path = out_dir / f'{split}.jsonl.gz'
        raw = ''.join(json.dumps(e, separators=(',', ':'), default=float) + '\n' for e in exs).encode()
        with gzip.GzipFile(path, 'wb', mtime=0) as f:
            f.write(raw)
        manifest['files'][path.name] = dict(examples=len(exs), sha256_uncompressed=hashlib.sha256(raw).hexdigest(),
                                            families=sorted({e['family'] for e in exs}),
                                            labels={k: sum(e['selected']['option'] == k for e in exs) for k in W.OPTION_NAMES},
                                            hard_negatives=sum(any(e['hard_negative'].values()) for e in exs))
        print(split, len(exs), flush=True)
    manifest['overlap'] = {f'{a}&{b}': len(sigs[a] & sigs[b]) for a in sigs for b in sigs if a < b}
    (ROOT / 'MANIFEST.json').write_text(json.dumps(manifest, indent=1) + '\n')


def load(split):
    with gzip.open(ROOT / 'data' / f'{split}.jsonl.gz', 'rt') as f:
        return [json.loads(line) for line in f]


def to_problem(ex):
    """Rebuild the solver-model Problem from the serialised example."""
    scopes = [f['scope'] for f in ex['problem']['factors']]
    tables = [fg.sensor_table(len(f['scope']), f['kind'], f['y'], f['noise']) for f in ex['problem']['factors']]
    return fg.Problem(ex['problem']['n_vars'], scopes, tables, ex['problem']['query'],
                      [dict(kind=f['kind'], y=f['y'], e=f['noise'], arity=len(f['scope'])) for f in ex['problem']['factors']])


def verify():
    bad = 0
    total = 0
    fams = {}
    for split in ('train', 'val', 'heldout'):
        exs = load(split)
        fams[split] = {e['family'] for e in exs}
        for e in exs:
            total += 1
            prob = to_problem(e)
            p = fg.variable_elimination(prob)['p']
            if not e['conditions']['model_misspecified']:
                bad += abs(p - e['verification']['reference_p']) > 1e-8
            ve = [o for o in e['operators'] if o['name'] == 've_min_fill']
            if ve:
                bad += abs(ve[0]['p'] - p) > 1e-8
            bad += abs(e['regret'] - (fg.oracle_utility(e['verification']['reference_p'], -e['decision_rule']['reward']['wrong'])
                                      - fg.expected_utility(e['final']['action'], e['verification']['reference_p'], -e['decision_rule']['reward']['wrong']))) > 1e-9
    man = json.loads((ROOT / 'MANIFEST.json').read_text())
    held_only = fams['heldout'] - fams['train'] - fams['val']
    res = dict(examples=total, mismatches=int(bad), overlap=man['overlap'], heldout_only_families=sorted(held_only),
               train_families=sorted(fams['train']))
    assert bad == 0 and all(v == 0 for v in man['overlap'].values())
    (ROOT / 'verify16.json').write_text(json.dumps(res, indent=1) + '\n')
    print(res)


if __name__ == '__main__':
    {'generate': generate, 'verify': verify}[sys.argv[1]]()

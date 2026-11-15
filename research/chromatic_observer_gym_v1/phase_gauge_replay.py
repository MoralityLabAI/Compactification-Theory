"""Evaluate whether already-selected frozen one-bit heads survive lost tick-zero.

This is a pure post-hoc diagnostic on immutable COE-05 fit-only heads. It does
not refit, inspect labels to select a model, or touch holdout data while
constructing the observer. Original COE-05 metric remains the primary one.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

try:
    from .gauge_orbit_codes import necklace
except ImportError:
    from gauge_orbit_codes import necklace

# Reuse COE-05 parser from the same package; deliberately no torch or numpy.
try:
    from .phase_order_replay import code_by_label, load_json, verify_receipt, EDGES, HELDOUT, FIT
except ImportError:
    from phase_order_replay import code_by_label, load_json, verify_receipt, EDGES, HELDOUT, FIT


def evaluate(item: dict, head: dict, contexts=HELDOUT) -> dict:
    n = len(EDGES) * len(contexts)
    counts = dict(ordered=0, count=0, cyclic=0, order_only=0,
                  cyclic_beyond_count=0, phase_sensitive_only=0, denominator=n)
    breakdown = []
    for ctx in contexts:
        words = code_by_label(item, head, ctx)
        ordered = len({w for w in words})
        detail = dict(context=ctx, distinct_codes=ordered,
                      distinct_necklaces=len({necklace(w, 4) for w in words}),
                      distinct_counts=len({w.bit_count() for w in words}))
        for a, b in EDGES:
            aw, bw = words[a], words[b]
            x = aw != bw
            y = aw.bit_count() != bw.bit_count()
            z = necklace(aw, 4) != necklace(bw, 4)
            assert (not y or z) and (not z or x)
            counts['ordered'] += x
            counts['count'] += y
            counts['cyclic'] += z
            counts['order_only'] += x and not y
            counts['cyclic_beyond_count'] += z and not y
            counts['phase_sensitive_only'] += x and not z
        breakdown.append(detail)
    counts['context_details'] = breakdown
    return counts


def evidence(activations: dict, receipt: dict) -> dict:
    verify_receipt(activations, receipt)
    items={i['model_seed']:i for i in activations['items']}
    result=[]
    for row in receipt['records']:
        head=row['primary_stable_order_head']
        m=evaluate(items[row['model_seed']], head)
        assert m['ordered']==head['heldout_total_edge_hits']
        assert m['count']==head['heldout_count_only_hits']
        assert m['order_only']==head['heldout_order_only_hits']
        result.append(dict(model_seed=row['model_seed'],model_checkpoint_sha256=row['source_checkpoint_sha256'],**m))
    sums={k:sum(r[k] for r in result) for k in
          ('ordered','count','cyclic','order_only','cyclic_beyond_count',
           'phase_sensitive_only','denominator')}
    return dict(schema='COE-06-posthoc-frozen-phase-gauge-v1',
                source_schema=receipt['schema'],
                cohort_results=result, pooled=sums,
                interpretation='post-hoc on fit-only selected COE-05 heads; no new fitting')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--activation',type=Path,required=True)
    p.add_argument('--receipt',type=Path,required=True)
    p.add_argument('--out',type=Path)
    opts=p.parse_args()
    result=evidence(load_json(opts.activation),load_json(opts.receipt))
    content=json.dumps(result,indent=2,sort_keys=True)+'\n'
    if opts.out:opts.out.write_text(content,encoding='utf-8')
    print(content)

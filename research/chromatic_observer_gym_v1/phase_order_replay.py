"""COE-05: independently replay selected heads, model/split gates and exact counts.

Pure standard-library validator; full fit-only hyperplane search is a separate
optional NumPy runner. This module never performs model inference or fits heads.
"""
from __future__ import annotations
import gzip, hashlib, json, math
from pathlib import Path

FIT = (0, 1, 2, 3)
HELDOUT = (4, 5, 6, 7)
N = 8
T = 4
D = 24
EDGES = tuple((a,b) for a in range(N) for b in range(a+1,N))


def load_json(path: Path) -> dict:
    if str(path).endswith('.gz'):
        with gzip.open(path,'rt',encoding='utf8') as file:
            return json.load(file)
    return json.loads(path.read_text(encoding='utf8'))


def digest_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_by_label(item, head, context):
    weights=head['weight']; theta=head['threshold']
    if len(weights) != D or not all(math.isfinite(x) for x in weights+[theta]):
        raise ValueError('invalid readout')
    result=[]
    for a in range(N):
        history=item['activations'][f'{a}:{context}']
        if len(history)!=T or any(len(v)!=D for v in history):
            raise ValueError('invalid trace shape')
        word=0
        for t in range(T):
            value=sum(weights[j]*history[t][j] for j in range(D))
            if value>theta: word |= (1 << t)
        result.append(word)
    return tuple(result)


def score_head(item, head):
    counts={name:0 for name in ('fit_order_only_hits','fit_total_edge_hits',
           'heldout_order_only_hits','heldout_total_edge_hits','heldout_count_only_hits')}
    stable_edges={edge:0 for edge in EDGES}
    by_heldout_context=[]
    for context in FIT+HELDOUT:
        words=code_by_label(item,head,context)
        novel=total=count=0
        for a,b in EDGES:
            differs=words[a]!=words[b]
            count_diff=words[a].bit_count()!=words[b].bit_count()
            if differs: total+=1
            if count_diff: count+=1
            if differs and not count_diff:
                novel+=1
                if context in FIT:stable_edges[(a,b)]+=1
        if context in FIT:
            counts['fit_order_only_hits']+=novel
            counts['fit_total_edge_hits']+=total
        else:
            counts['heldout_order_only_hits']+=novel
            counts['heldout_total_edge_hits']+=total
            counts['heldout_count_only_hits']+=count
            by_heldout_context.append(novel)
    counts['fit_stable_novel_edges']=sum(v>=2 for v in stable_edges.values())
    counts['heldout_order_only_by_context']=by_heldout_context
    return counts


def verify_receipt(bundle: dict, receipt: dict):
    if bundle['schema']!='COE-04-native-tied-loop-frozen-v1':raise ValueError('model source schema')
    if receipt['schema']!='COE-05-fit-only-phase-probe-exploratory-v1':raise ValueError('receipt schema')
    if tuple(bundle['probe_fit_contexts'])!=FIT or tuple(bundle['probe_heldout_contexts'])!=HELDOUT:
        raise ValueError('source split changed')
    if receipt['split']!={'fit':list(FIT),'heldout':list(HELDOUT)}:
        raise ValueError('receipt split changed')
    items={item['model_seed']:item for item in bundle['items']}
    if len(items)!=len(bundle['items']) or len(items)!=len(receipt['records']):
        raise ValueError('duplicate or missing seeds')
    for r in receipt['records']:
        item=items[r['model_seed']]
        if r['source_checkpoint_sha256']!=item['model_checkpoint_sha256']:
            raise ValueError('unmatched checkpoint SHA')
        if item['train_steps'] !=160 or item['train_examples']!=64:
            raise ValueError('changed base model')
        if len(item['model_predictions'])!=64 or any(item['model_predictions'][8*a+b]!=a
                     for a in range(N) for b in range(N)):
            raise ValueError('base model has not passed admission gate')
        if set(item['activations'])!={f'{a}:{b}' for a in range(8) for b in range(8)}:
            raise ValueError('missing activation cells')
        for key in ('primary_stable_order_head','comparative_best_total_head'):
            head=r[key]
            actual=score_head(item,head)
            for field,value in actual.items():
                if head[field]!=value:
                    raise ValueError(f'head scoreboard mismatch {r["model_seed"]}:{key}:{field}')
        if r['candidate_count']<=0:raise ValueError('empty search pool')
    return True


def pooled(receipt):
    rows=[r['primary_stable_order_head'] for r in receipt['records']]
    return {'model_seeds':[r['model_seed'] for r in receipt['records']],
            'heldout_total':sum(r['heldout_total_edge_hits'] for r in rows),
            'heldout_count_only':sum(r['heldout_count_only_hits'] for r in rows),
            'heldout_order_only':sum(r['heldout_order_only_hits'] for r in rows),
            'denominator':len(rows)*4*len(EDGES)}


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('--activation',type=Path,required=True)
    p.add_argument('--receipt',type=Path,required=True)
    args=p.parse_args()
    r=load_json(args.receipt)
    verify_receipt(load_json(args.activation),r)
    print(json.dumps(pooled(r),indent=2,sort_keys=True))

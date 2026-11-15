"""COE-05 exploratory fit-only phase-head search on frozen 4-visit traces.

Requires numpy only for batched speed; the standalone replay verifier is stdlib.
No held-out activation or held-out label is used when generating or selecting heads.
"""
import argparse, json, gzip, sys, hashlib, time
from pathlib import Path
import numpy as np
try:
    from .temporal_observers import candidate_projections, EDGES
except ImportError:
    from temporal_observers import candidate_projections, EDGES

A,B=np.array(EDGES).T
POWERS=np.array([1,2,4,8])

def load(path):
    with gzip.open(path,'rt') as f: return json.load(f)

def screen_model(item):
    trace=np.array([[[item['activations'][f'{label}:{ctx}'][t]
                  for t in range(4)] for label in range(8)]
                  for ctx in range(8)],dtype=np.float64)
    candidates=candidate_projections(item,seed=item['model_seed']+601)
    weights=np.asarray([w for w,th in candidates],dtype=np.float64)
    thresholds=np.asarray([th for w,th in candidates],dtype=np.float64)
    bits=np.einsum('hd,catd->hcat',weights,trace,optimize=True)>thresholds[:,None,None,None]
    words=(bits.astype(np.int16)*POWERS).sum(-1)
    counts=bits.sum(-1)
    separated=words[:,:,A]!=words[:,:,B]
    novelty=separated & (counts[:,:,A]==counts[:,:,B])
    fit=novelty[:,:4]
    stable=(fit.sum(1)>=2).sum(1)
    train_novel=fit.sum((1,2))
    train_total=separated[:,:4].sum((1,2))
    # Fixed fit-only selection. `np.lexsort` last criterion has priority.
    idx=int(np.lexsort((-np.arange(len(candidates)),train_total,train_novel,stable))[-1])
    idx_full=int(np.lexsort((-np.arange(len(candidates)),train_novel,train_total))[-1])
    def result(i):
        return {'candidate_index':int(i),'weight':[float(v) for v in candidates[i][0]],
                'threshold':float(candidates[i][1]),
                'fit_stable_novel_edges':int(stable[i]),
                'fit_order_only_hits':int(train_novel[i]),
                'fit_total_edge_hits':int(train_total[i]),
                'heldout_order_only_hits':int(novelty[i,4:].sum()),
                'heldout_total_edge_hits':int(separated[i,4:].sum()),
                'heldout_count_only_hits':int((counts[i,4:,A]!=counts[i,4:,B]).sum()),
                'heldout_order_only_by_context':[int(v) for v in novelty[i,4:].sum(1)]}
    return {'model_seed':item['model_seed'],
            'source_checkpoint_sha256':item['model_checkpoint_sha256'],
            'candidate_count':len(candidates),
            'primary_stable_order_head':result(idx),
            'comparative_best_total_head':result(idx_full)}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    data=load(args.input)
    t0=time.monotonic()
    rows=[]
    for item in data['items']:
        row=screen_model(item)
        rows.append(row)
        print(row['model_seed'], 'stable order', {k:v for k,v in row['primary_stable_order_head'].items() if k not in ('weight','threshold')}, 'full',{k:v for k,v in row['comparative_best_total_head'].items() if k not in ('weight','threshold')},flush=True)
    out={'schema':'COE-05-fit-only-phase-probe-exploratory-v1','split':{'fit':[0,1,2,3],'heldout':[4,5,6,7]},
         'selection':'lexmax (fit_novel_edges_seen_in_at_least_two_contexts, fit_novel_edge_hits, fit_all_edge_hits, -candidate_index)',
         'context_warning':'Heldout contexts appeared in original base-model training, and earlier seeds were examined in COE-04',
         'records':rows}
    args.out.write_text(json.dumps(out,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print('time',time.monotonic()-t0)
if __name__=='__main__': main()

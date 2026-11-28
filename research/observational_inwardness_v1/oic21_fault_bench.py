"""Run exactly all shifts, phases, and fault positions using cached public histories."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
from oic21_bench import load_model
from oic21_faults import fast_fault_panel


def run(ns=(4,6,8,12),splits=('full','heldout'),seeds=tuple(range(5))):
    start=time.perf_counter();records=[]
    for n in ns:
        for split in splits:
            for seed in seeds:
                _,r=load_model(split,seed)
                panels,cache=fast_fault_panel(n,r['weights'])
                records.append({'n':n,'split':split,'seed':seed,'panels':panels,'cache_states':cache['currsize']})
                print('n',n,split,seed,{x['kind']:round(x['accuracy'],4) for x in panels},flush=True)
    agg=[]
    for n in ns:
        for split in splits:
            rs=[x for x in records if x['n']==n and x['split']==split]
            bykind=[]
            for kind in ('clean','flip','erase'):
                x=[next(p for p in r['panels'] if p['kind']==kind) for r in rs]
                bykind.append({'kind':kind,'mean_accuracy':sum(a['accuracy'] for a in x)/len(x),
                               'worst_seed_accuracy':min(a['accuracy'] for a in x),
                               'episodes':sum(a['episodes'] for a in x),'correct':sum(a['correct'] for a in x)})
            agg.append({'n':n,'split':split,'kinds':bykind})
    return {'status':'exact_cyclic_posterior_noisy_online_adaptive_action_evaluation',
            'n':list(ns),'splits':list(splits),'seeds':list(seeds),
            'likelihood_noise':0.04,'corruption':'all individual flip and erase positions, one per history',
            'results':records,'aggregates':agg,'elapsed_seconds':round(time.perf_counter()-start,3)}

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--ns',type=int,nargs='+',default=[4,6,8,12]);p.add_argument('--splits',nargs='+',default=['full','heldout']);p.add_argument('--seeds',type=int,nargs='+',default=list(range(5)));p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 x=run(tuple(a.ns),tuple(a.splits),tuple(a.seeds))
 a.out.parent.mkdir(exist_ok=True)
 a.out.write_text(json.dumps(x,sort_keys=True,indent=2)+'\n')
 print('saved',a.out,'seconds',x['elapsed_seconds'])

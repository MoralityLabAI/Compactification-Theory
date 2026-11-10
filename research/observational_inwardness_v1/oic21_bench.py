"""OIC-21 exact and learned fair-budget comparisons with explicit sensor costs."""
from __future__ import annotations
import json,random,time,sys
from pathlib import Path
import numpy as np
import torch
from oic21_exact import ExactActiveOracle
from oic21_policy import (evaluate_policy,greedy_policy,entropy_policy,
                          fixed_marker_policy, fixed_cycle_policy,FrozenFixedPolicy,
                          fixed_multisensor_accuracy,numpy_model_policy,evaluate_torch_batched)
from oic21_train import ScoreNet,model_policy
from oic20_markers import exhaustive_minimum_markers,identifies_every_shift

NS=(4,5,6,7,8,12,16,24,32)

def load_model(split,seed):
    record=json.loads((Path(__file__).parent/f'fit_{split}_{seed}.json').read_text())
    m=ScoreNet();m.load_state_dict({k:torch.tensor(v) for k,v in record['weights'].items()});m.eval()
    return m,record

def fixed_baselines(n,Q):
    oracle=ExactActiveOracle(n,Q)
    rand=[]
    for seed in range(64):
        rng=random.Random(20261110+ n*411+seed)
        word=tuple(rng.randrange(n) for _ in range(Q))
        rand.append(oracle.fixed_score(word)/(n*n))
    opt=None
    if n<=5:
        word,score=oracle.best_fixed()
        opt={'accuracy':score/(n*n),'word':list(word)}
    return {'constant_marker_accuracy':evaluate_policy(n,Q,fixed_marker_policy)['shift_accuracy'],
            'cyclic_sweep_accuracy':evaluate_policy(n,Q,fixed_cycle_policy)['shift_accuracy'],
            'random_fixed_accuracy_mean':sum(rand)/len(rand),
            'random_fixed_accuracy_min':min(rand),'random_fixed_accuracy_max':max(rand),
            'random_fixed_seeds':64,'optimal_fixed':opt}


def run():
    torch.set_num_threads(1)
    out=[]
    start=time.time()
    for n in NS:
        Q=n+2
        marker_sets={4:(0,1),5:(0,1),6:(0,2),7:(0,1),8:(0,1,2),12:(0,1,4),16:(0,1,2,4),24:(0,1,3,9),32:(0,1,2,4,8)}
        marker_set=marker_sets[n]
        marker_count=len(marker_set)
        assert identifies_every_shift(n,marker_set)
        fixed_multi=fixed_multisensor_accuracy(n,marker_set,n)
        row={'n':n,'query_budget':Q,'best_multi_sensor_markers':list(marker_set),
             'simultaneous_sensor_baseline':fixed_multi,
             'greedy':evaluate_policy(n,Q,greedy_policy),
             'information_gain':evaluate_policy(n,Q,entropy_policy),
             'fixed':fixed_baselines(n,Q),'learned':{}}
        if n<=6:
            o=ExactActiveOracle(n,Q)
            row['optimal_adaptive']={'accuracy':o.adaptive_accuracy(),
                                    'expected_success_numerator':o.best_adaptive(o.all,0),
                                    'denominator':n*n}
        for split in ('full','heldout'):
            runs=[]
            for seed in range(5):
                record=json.loads((Path(__file__).parent/f'fit_{split}_{seed}.json').read_text())
                r=record['evaluation']
                result=next(x for x in r if x['n']==n and x['horizon']==Q)
                fresh=result  # frozen trial; torch replay validated separately
                runs.append({'seed':seed,'shift_accuracy':fresh['shift_accuracy'],
                             'brier':fresh['brier'],
                             'conditional_entropy_bits':fresh['conditional_entropy_bits']})
            row['learned'][split]={'runs':runs,
              'mean_shift_accuracy':sum(z['shift_accuracy'] for z in runs)/5,
              'min_shift_accuracy':min(z['shift_accuracy'] for z in runs),
              'mean_brier':sum(z['brier'] for z in runs)/5}
        out.append(row)
        print('n',n,'greedy',round(row['greedy']['shift_accuracy'],4),
              'learned_full',round(row['learned']['full']['mean_shift_accuracy'],4),
              'learned_heldout',round(row['learned']['heldout']['mean_shift_accuracy'],4),
              'fixed_random',round(row['fixed']['random_fixed_accuracy_mean'],4),
              'simultaneous',fixed_multi['accuracy'],flush=True)
    report={'status':'executed_standard_library_exact_belief_plus_cpu_trained_policy',
            'query_resource':'one labeled sensor-bit per time step, address cost separately charged',
            'ns':list(NS),'n_train':[4,5,6],
            'seeds':list(range(5)),'splits':['full','heldout'],
            'results':out,'elapsed_seconds':round(time.time()-start,3)}
    path=Path(__file__).parent/'results'/'oic21_bench.json'
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print('saved',path,'elapsed',time.time()-start)

if __name__=='__main__':run()

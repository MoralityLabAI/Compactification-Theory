"""OIC-15 receipt analysis and exhaustive bounded raw-sensor fault tests.

Optional PyTorch numerical verification reconstructs the stored 20 fitted models.
The finite nearest-codeword decoding bound is checked independently on every
initial phase and every single fault position for n=4,6,8.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from oic15_pilot import (GenericGRU, CyclicFilter, RawFilter, TRAIN_NS, OOD_NS, actions_for,
    balanced_panel, exact_posterior, exact_min_distance_classifier, oracle_batch,
    score_panel, validate_control)


def restore_model(run: dict):
    if run['architecture'] in ('cyclic','raw'):
        m=CyclicFilter() if run['architecture']=='cyclic' else RawFilter()
        with torch.no_grad():m.log_beta.copy_(torch.tensor(run['weights']['log_beta']))
    else:
        m=GenericGRU()
        m.load_state_dict({k:torch.tensor(v,dtype=torch.float32) for k,v in run['weights'].items()})
    return m.eval()


def rate_from_prob(p, fam):
    pred=np.asarray(p)>.5
    return float(np.mean(pred==np.asarray(fam)))


def oracle_robust_test(n:int, word:tuple[int,...], kind:str):
    fam,phase,u,x=balanced_panel(n,word)
    decoded=[]
    for t in range(len(word)+1):
        altered=x.clone()
        if kind=='flip':altered[:,t]^=1
        elif kind=='erasure':altered[:,t]=-1
        else:raise ValueError(kind)
        decoded += [exact_min_distance_classifier(n,word,tuple(row.tolist()))==int(fam[k].item())
                    for k,row in enumerate(altered)]
    if not all(decoded):raise AssertionError(f'raw-codeword bound violated: n={n}, kind={kind}')
    return {'n':n,'kind':kind,'tested_episodes':len(decoded),'accuracy':sum(decoded)/len(decoded)}


def fault_panel(model,n,word,kind):
    fam,phase,u,clean=balanced_panel(n,word)
    true=fam.numpy()
    correctness=[]; per_phase=[True]*(2*n)
    deviations=[]
    for t in range(len(word)+1):
        obs=clean.clone()
        if kind=='flip': obs[:,t]^=1
        elif kind=='erasure': obs[:,t]=-1
        with torch.no_grad(): p=torch.sigmoid(model(n,u,obs)[:,-1]).numpy()
        current=(p>.5).astype(int)==true
        correctness.extend(map(bool,current))
        per_phase=[old and bool(now) for old,now in zip(per_phase,current)]
        deviations.extend(abs(p-true))
    return {'mean_accuracy':sum(correctness)/len(correctness),
            'worst_case_phase_success_rate':sum(per_phase)/len(per_phase),
            'all_phase_all_fault_positions_correct':all(correctness),
            'total_tested_episodes':len(correctness),
            'mean_absolute_probability_error':float(np.mean(deviations))}


def phase_relabel_test(n:int,word:tuple[int,...]):
    fam,phase,u,zero=balanced_panel(n,word,marker=0)
    source=Counter(tuple(z) for z in zero.tolist())
    for m in range(n):
        _,_,_,shifted=balanced_panel(n,word,marker=m)
        if Counter(tuple(z) for z in shifted.tolist())!=source:
            raise AssertionError('cyclic relabel changed observable trace multiset')
    return True


def analytic_full_probe_error(n,t):
    """Equal model priors, each with uniform latent phase; OIC-T04."""
    if t<2:return 0.5
    return max(n-t,0)/(2*n)


def summarize(data:dict):
    torch.set_num_threads(1)
    runs=data['runs']
    if len(runs)!=30:raise ValueError('expected 30 fitted observers')
    metrics=[]
    active_trajectory=[]
    faults=[]
    for split in ('full','heldout'):
        for arc in ('gru','cyclic','raw'):
            rs=sorted((r for r in runs if r['split']==split and r['architecture']==arc),key=lambda r:r['seed'])
            if [r['seed'] for r in rs]!=list(range(5)):
                raise ValueError('seed group not five matched fits')
            models={r['seed']:restore_model(r) for r in rs}
            for n in TRAIN_NS+OOD_NS:
                for schedule in ('passive','full_active','latch','random_budget'):
                    records=[r['panels'][str(n)][schedule] for r in rs]
                    # Re-evaluate stored weights against reported probability arrays.
                    for r in rs:
                        panel=score_panel(models[r['seed']],n,actions_for(n,schedule))
                        saved=r['panels'][str(n)][schedule]['predicted_probabilities']
                        if not np.allclose(panel['predicted_probabilities'],saved,rtol=1e-5,atol=3e-6):
                            raise AssertionError(f'weight replay mismatch {(arc,split,r["seed"],n,schedule)}')
                    metrics.append({'split':split,'architecture':arc,'n':n,'schedule':schedule,
                        'mean_accuracy':float(np.mean([p['accuracy'] for p in records])),
                        'mean_brier':float(np.mean([p['brier'] for p in records])),
                        'mean_passive_deviation':float(np.mean([p['passive_deviation'] for p in records])),
                        'parameters':rs[0]['training']['parameters'],
                        'per_seed_accuracies':[p['accuracy'] for p in records]})
                for t in range(1,n+1):
                    acc=[score_panel(models[r['seed']],n,(1,)*t)['accuracy'] for r in rs]
                    active_trajectory.append({'split':split,'architecture':arc,'n':n,'steps':t,
                                              'mean_accuracy':float(np.mean(acc)),
                                              'optimal_bayes_accuracy':1-analytic_full_probe_error(n,t)})
                if n in (4,6,8):
                    for kind,horizon in [('flip',n+2),('erasure',n+1)]:
                        word=(1,)*n+(0,)*(horizon-n)
                        tests=[fault_panel(models[r['seed']],n,word,kind) for r in rs]
                        faults.append({'split':split,'architecture':arc,'n':n,'fault':kind,
                                       'horizon':horizon,'active_probes':n,
                                       'tested_episodes':sum(r['total_tested_episodes'] for r in tests),
                                       'mean_accuracy':float(np.mean([r['mean_accuracy'] for r in tests])),
                                       'mean_worst_case_phase_success':float(np.mean([r['worst_case_phase_success_rate'] for r in tests])),
                                       'successful_seeds':sum(int(r['all_phase_all_fault_positions_correct']) for r in tests),
                                       'per_seed_accuracy':[r['mean_accuracy'] for r in tests]})
    controls={'unseen_marker_relabelings':{},'exact_raw_oracle':[]}
    for n in (4,5,6,7,8):
        controls['unseen_marker_relabelings'][str(n)]=phase_relabel_test(n,actions_for(n,'latch'))
    for n in (4,6,8):
        for kind,t in [('flip',n+2),('erasure',n+1)]:
            controls['exact_raw_oracle'].append(oracle_robust_test(n,(1,)*n+(0,)*(t-n),kind))
    return {'experiment':'OIC-15-CPU-01','observability_controls':validate_control(),
            'matched_training_sample_seeds':True,
            'metrics':metrics,'prefix_trajectories':active_trajectory,
            'raw_fault_experiments':faults,'structural_controls':controls}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args()
    d=json.loads(a.input.read_text())
    s=summarize(d)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(s,indent=2,sort_keys=True)+'\n')
    print('summarized',len(d['runs']),'fits,',len(s['metrics']),'panels,',len(s['raw_fault_experiments']),'fault groups')
    for rec in s['raw_fault_experiments']:
        if rec['split']=='heldout':
            print('fault',rec['architecture'],rec['n'],rec['fault'],round(rec['mean_accuracy'],3),'complete seeds',rec['successful_seeds'])

if __name__=='__main__':main()

"""H-M0 v2 post-hoc fairness follow-up: fresh 500M worlds and greedy MAP stop.

This study was designed after the 400M final test but before inspecting 500M outcomes.
The exact already-trained terminal model stays frozen. The new baseline can stop
and maximizes one-step expected *terminal MAP decision utility* per probe.
"""
from __future__ import annotations
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from experiments.h_m0_pruner import N_H, N_PROBES, BUDGET, make_episode, posterior_update
from experiments.h_m0_terminal import (
    COST_PENALTY, Selector, TerminalPlanner, paired_interval, results_summary,
    run as run_existing, legal_actions, posterior_from_outcomes,
)


def greedy_map(ep):
    mask, outcomes, spent, probes = 0, 0, 0., []
    while True:
        p=posterior_from_outcomes(ep,mask,outcomes)
        legal=legal_actions(ep,mask,spent)
        if not legal:break
        gains=[]
        for i in legal:
            pi=float(np.dot(p,ep.likelihoods[i]))
            pu=posterior_update(p,ep.likelihoods[i],1)
            pn=posterior_update(p,ep.likelihoods[i],0)
            accuracy_gain=pi*pu.max()+(1-pi)*pn.max()-p.max()
            gains.append(float(accuracy_gain-COST_PENALTY*ep.costs[i]))
        index=int(np.argmax(gains))
        if gains[index]<=1e-10:break
        a=legal[index]
        spent+=float(ep.costs[a]);mask|=1<<a;outcomes|=int(ep.observations[a])<<a;probes.append(a)
    p=posterior_from_outcomes(ep,mask,outcomes)
    return {'seed':int(ep.seed),'arm':'greedy_MAP_with_stop','truth':ep.truth,
        'predicted':int(p.argmax()),'correct':int(p.argmax()==ep.truth),
        'expected_correct':float(p.max()),'terminal_utility':float(p.max()-COST_PENALTY*spent),
        'brier':float(np.sum((p-np.eye(N_H)[ep.truth])**2)),
        'log_loss':float(-np.log(max(1e-12,p[ep.truth]))),
        'cost':spent,'probes':probes,'decisions':len(probes)+1,'posterior':np.round(p,8).tolist()}


def main():
    base=Path(__file__).resolve().parents[1]
    out=base/'results/h_m0_02'
    study='fair_stop_followup'
    model=Selector();model.load_state_dict(torch.load(out/'terminal_selector.pt',weights_only=True,map_location='cpu'));model.eval()
    armlist=['random','bayes_ig','terminal_policy','greedy_MAP_with_stop']
    records=[]
    start=time.perf_counter()
    # This 500M seed range is separate from the original 300M, v2 test 400M,
    # and expert-data 100M / 350M generator streams.
    for j in range(1000):
        ep=make_episode(500000000+j)
        for arm in armlist:
            r=greedy_map(ep) if arm=='greedy_MAP_with_stop' else run_existing(arm,ep,model)
            records.append(r)
    means={a:{key:float(np.mean([r[key] for r in records if r['arm']==a])) for key in
                 ['correct','expected_correct','terminal_utility','brier','log_loss','cost','decisions']}
           for a in armlist}
    comparisons={a:{key:paired_interval(records,'terminal_policy',a,key) for key in ['correct','terminal_utility','cost']}
                 for a in armlist if a != 'terminal_policy'}
    result={'status':'posthoc_comparator_preregistered_on_new_seed_range',
        'n_fresh_worlds':1000,'seed_range':[500000000,500000999],
        'mean':means,'comparisons':comparisons,
        'wall_seconds':time.perf_counter()-start,
        'interpretation':'myopic MAP baseline sees same known likelihoods and budget, and may stop at any time',
        'note':'exact dynamic-programming expert omitted from this 1000-case comparator; it was measured on a separate 100-case subset in first evaluation'}
    (out/(study+'_metrics.json')).write_text(json.dumps(result,indent=2)+'\n')
    with (out/(study+'_records.jsonl')).open('w') as f:
        for rec in records:f.write(json.dumps(rec)+'\n')
    print(json.dumps({'accuracy':{a:round(means[a]['correct'],4) for a in armlist},
        'utility':{a:round(means[a]['terminal_utility'],4) for a in armlist},
        'cost':{a:round(means[a]['cost'],3) for a in armlist},
        'learned_minus_greedy_MAP':comparisons['greedy_MAP_with_stop']},indent=2))

if __name__=='__main__':main()

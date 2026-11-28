"""SHAFT-COMP-11: on-manifold future alias test in trained toy GRUs.

Each pair consists of two naturally reachable frozen-model states, produced
by distinct random prefixes for the SAME synthetic task. A shared held-out
continuation tests whether the probe's bit code suffices for future behavior.
No direct activation editing is used.
"""
import argparse
import json
from pathlib import Path
import torch
from research.shaft_compactification.experiments.causal_probe import (
    fit, train_probe, capture, counterfactual, TOKENS, TASKS, PER_TASK
)

SEEDS=(101,202)
DEPTHS=(12,40,64)
HORIZONS=(0,1,4,12,32)

def run():
    torch.set_num_threads(3)
    torch.use_deterministic_algorithms(True)
    results={}
    for seed in SEEDS:
        model,_=fit(seed)
        probe,_=train_probe(model,seed)
        result={}
        for depth in DEPTHS:
            ha,goals=capture(model,(depth,),PER_TASK,seed+300000+depth)[depth]
            hb,goals_b=capture(model,(depth,),PER_TASK,seed+400000+depth)[depth]
            assert torch.equal(goals.sort().values,goals_b.sort().values)
            # Pair contexts by the same goal, with independent random prefixes.
            indices_a=torch.argsort(goals,stable=True)
            indices_b=torch.argsort(goals_b,stable=True)
            ha, hb = ha[indices_a], hb[indices_b]
            goals=goals[indices_a]
            gen=torch.Generator().manual_seed(seed+500000+depth)
            code_a=(probe(ha)>=0)
            code_b=(probe(hb)>=0)
            equal_code=(code_a==code_b).all(dim=1)
            vals={}
            for k in HORIZONS:
                suffix=torch.randint(TOKENS,(len(goals),k),generator=gen)
                qa=counterfactual(model,ha,suffix)>=0
                qb=counterfactual(model,hb,suffix)>=0
                different=(qa!=qb).any(dim=1)
                vals[str(k)]={
                    'future_disagreement_overall':round(float(different.float().mean()),6),
                    'same_code_future_disagreement_overall':round(float((equal_code & different).float().mean()),6),
                    'conditional_disagreement_given_same_code':round(float((equal_code & different).sum())/max(1,int(equal_code.sum())),6),
                    'same_code_pairs':int(equal_code.sum()),
                    'pairs':len(goals),
                }
            result[str(depth)]={'same_goal':True,'probe_code_agreement_rate':round(float(equal_code.float().mean()),6),'future':vals}
        results[str(seed)]=result
    return {'status':'natural_reachable_state_alias_test','model_seeds':list(SEEDS),'state_depths':list(DEPTHS),'continuations':list(HORIZONS),'pairs_per_depth_per_model':TASKS*PER_TASK,'results':results}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out')
    opts=ap.parse_args()
    s=json.dumps(run(),sort_keys=True,indent=2)+'\n'
    if opts.out:Path(opts.out).write_text(s)
    else:print(s,end='')
if __name__=='__main__':main()

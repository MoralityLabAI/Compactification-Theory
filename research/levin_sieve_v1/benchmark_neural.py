"""Frozen exact-rational heldout damage/stale-repair evaluation for LSV-04.

These are development results from a tiny task-conditioned learned proxy. All
cases are synthetic and disjoint from the Torch training PRNG. A direct
algebraic solver is an input-equivalent upper bound with different compute.
"""
from __future__ import annotations
import argparse, hashlib, json, random
from pathlib import Path
from fractions import Fraction as Q
from neural_bridge import (TARGET, CLEAN_LIMIT, DAMAGE_LIMIT, WEIGHTS,
    audited_rollout, ex_ante_bound, forcing, guaranteed_absolute_goal,
    load_checkpoint, make_certificate, model_matrix, rollout)

TEST_SEED = 20261105 + 7102
CASES = 128
DEPTHS = (0,1,2,4,8,12,16,24)
GOAL_TOLERANCE = Q(1,5)
FROZEN_UNTRAINED = ((Q(0), Q('1.1000')),(Q('0.1510'), Q(0)))


def roster(seed=TEST_SEED,n=CASES):
    if n<1: raise ValueError('empty test roster')
    rng = random.Random(seed)
    cases=[]
    for k in range(n):
        goal=tuple(Q(rng.randint(-900,900),1000) for _ in range(2))
        damage=tuple(Q(rng.randint(-1500,1500),1000) for _ in range(2))
        cases.append((goal,damage))
    return cases


def _metrics(predictions, targets):
    n=len(targets)
    abs_errors=[max(abs(x-y) for x,y in zip(pred,goal))
                for pred,goal in zip(predictions,targets)]
    mae=sum((sum((abs(x-y) for x,y in zip(pred,goal)),Q(0))
            for pred,goal in zip(predictions,targets)),Q(0))/(2*n)
    mse=sum((sum(((x-y)**2 for x,y in zip(pred,goal)),Q(0))
            for pred,goal in zip(predictions,targets)),Q(0))/(2*n)
    return {'mean_absolute_error':round(float(mae),6),
            'mean_squared_error':round(float(mse),6),
            'goal_le_0p2':sum(e<=GOAL_TOLERANCE for e in abs_errors),
            'goal_le_0p1':sum(e<=Q(1,10) for e in abs_errors),
            'n':n}


def evaluate(ck,seed=TEST_SEED,n=CASES,verify_all=True):
    W=model_matrix(ck)
    cert,eta=make_certificate(W)
    cases=roster(seed,n)
    roster_hash=hashlib.sha256(json.dumps(
        [[[str(x) for x in goal],[str(x) for x in damage]]
         for goal,damage in cases],separators=(',',':')).encode()).hexdigest()
    goals=[goal for goal,_ in cases]
    outputs={k:[] for k in ('damage','forcing','oracle')}
    time_series={name:{k:[] for k in DEPTHS} for name in
                 ('learned_fresh','learned_stale_D2','untrained_fresh','swapped_fresh','no_coupling_fresh')}
    methods={
        'learned_fresh':(W,0),
        'learned_stale_D2':(W,2),
        'untrained_fresh':(FROZEN_UNTRAINED,0),
        'swapped_fresh':(((Q(0), W[1][0]),(W[0][1],Q(0))),0),
        'no_coupling_fresh':(((Q(0),Q(0)),(Q(0),Q(0))),0),
    }
    for goal,damage in cases:
        outputs['damage'].append(tuple(x+y for x,y in zip(goal,damage)))
        outputs['forcing'].append(forcing(goal))
        outputs['oracle'].append(goal)
        for method,(gain,delay) in methods.items():
            if method.startswith('learned') and verify_all:
                states,_,_,_=audited_rollout(ck,goal,damage,cycles=max(DEPTHS),delay=delay)
            else:
                _,states,_,_=rollout(gain,goal,damage,cycles=max(DEPTHS),delay=delay)
            for k in DEPTHS:
                time_series[method][k].append(states[2*k])
    result={
        'schema':'LSV-04/exploratory-benchmark-v1',
        'source':'cpu_exact_rational_evaluation_of_torch_trained_quantized_checkpoint',
        'seed':seed,'cases':n,'roster_sha256':roster_hash,
        'weights_sha256':ck['weights_sha256'],
        'contract':{'target_bounds':'[-9/10,9/10] per coordinate',
                    'damage_bounds':'[-3/2,3/2] per coordinate',
                    'goal_tolerance_absolute':str(GOAL_TOLERANCE),
                    'schedule':'0,1 repeated; stale mode reads 2 steps back',
                    'exact_arithmetic':True,
                    'same_computation_controls':'untrained; swapped; zero coupling at each depth',
                    'oracle':'exact inversion of known task generator',
                    'trained_checkpoint_is_pretrained_TRM':False},
        'certificate':{'weights':[str(v) for v in cert.weights],
                       'rho':str(cert.rho),
                       'eta':[str(e) for e in eta],
                       'radius_weighted':str(cert.radius(eta)),
                       'initial_error_bound_weighted':'3/2',
                       'max_weight':str(max(cert.weights)),
                       'certified_first_cycles_for_abs_goal':{
                           str(delay):next((k for k in range(1,201)
                                            if guaranteed_absolute_goal(cert,eta,k,delay,GOAL_TOLERANCE)),None)
                           for delay in (0,2)}},
        'baselines':{name:_metrics(pred,goals) for name,pred in outputs.items()},
        'methods':{},
    }
    for method,depths in time_series.items():
        result['methods'][method]={str(k):_metrics(pred,goals) for k,pred in depths.items()}
    # Each exact-rational full trace was audited under the declared neural
    # input/update/output contract before output was used for any metric.
    result['certificate']['trajectory_audits']=2*n if verify_all else 0
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,default=Path(__file__).parent/'checkpoints/repair_two_node_20261104.json')
    p.add_argument('--out',type=Path,default=None)
    p.add_argument('--check',type=Path,default=None)
    a=p.parse_args()
    if a.out and a.check: p.error('choose --out or --check')
    result=evaluate(load_checkpoint(a.checkpoint))
    content=json.dumps(result,indent=2,sort_keys=True)+'\n'
    if a.check:
        if a.check.read_text()!=content:raise ValueError('receipt mismatch')
        print('receipt matches regenerated exact-rational results')
    elif a.out:
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(content)
        print('wrote',a.out)
    else:
        print(content)

if __name__=='__main__': main()

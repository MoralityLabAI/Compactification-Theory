"""SMC-10C: robust endpoint effect ordering through VERIFIED local Lipschitz tubes.

This complements SMC-10's direct rectangular interval propagation. For each
clean and perturbed frozen exact-real state, it validates the local L2
Lipschitz upper bound of the entire 3-step recurrent map on a finite
Euclidean ball, using the existing LSV-17 Arb-verified source spectra,
LayerNorm radii and recurrent transport.

No numerical Jacobian, finite Hessian sampling or empirical maximum is
accepted as a mathematical Lipschitz bound. Source SAE training and roster
are identical to SMC-09; source roundoff remains outside proof scope.
"""
import argparse
import copy
import json
from fractions import Fraction as Q
from pathlib import Path
import sys

import torch
from flint import arb,ctx

HERE=Path(__file__).resolve().parent
PRE=HERE.parent/'levin_sieve_v1'/'pretrained_trm'
ARB_DIR=HERE.parent/'levin_sieve_v1'/'goal_taylor_v1'
for directory in (HERE,PRE,ARB_DIR):
    sys.path.insert(0,str(directory))

from arb_engine import TaylorNet,aq,up,verified_spectral,exact_matrix
from run_checkpoint import locked_bytes,load_pretrained,prepare_roster
from smc07_real_sae import fit_sae,split_ids,EXPECTED_ROSTER
from smc08_curvature import goal_jacobian,sae_direction_set
from smc09_endpoint_arb import (
    CHECKPOINT_SHA,DIRECTIONS,CONTROLS,shifted_anchor,
    normalized_arb_direction,squared_logit_effect,
)


def exact_effect_norm_interval(squared_interval):
    """Return Arb bounds for sqrt of a nominal squared-effect enclosure."""
    if not squared_interval.is_finite() or squared_interval.upper()<0:
        raise ArithmeticError('invalid nominal squared effect interval')
    # Physically squared output norm is nonnegative even if naive interval
    # self-products have a very small negative lower bound.
    norm=squared_interval.nonnegative_part().sqrt()
    if not norm.is_finite():
        raise ArithmeticError('nonfinite nominal effect norm')
    lo=norm.lower() if norm.lower()>0 else arb(0)
    hi=norm.upper()
    return lo,hi


def validated_lipschitz_on_ball(net,tape,head_norm_upper,radius):
    """Certified global-on-the-ball Lipschitz coefficient for goal readout.

    Transport returns verified per-layer Jacobian op-norm bounds on
    propagated Euclidean balls around this exact center, and the radius
    progression R_i is also a verified outer bound on all trajectories.
    """
    r=Q(radius)
    if r<0:
        raise ValueError('negative local ball radius')
    transported=net.transport(tape,r,dimensions=1)
    total=arb(1)
    for item in transported:
        if item['a']<0:
            raise ArithmeticError('negative layer derivative bound')
        total=up(total*item['a'])
    full=up(total*head_norm_upper)
    if not full.is_finite() or full<0:
        raise ArithmeticError('nonfinite verified output Lipschitz upper')
    return full, {
        'lipschitz_upper_arb':str(full),
        'layernorm_epsilon_floor_fallbacks':sum(
            bool(item['fallback']) for item in transported),
        'last_propagated_radius_upper':str(transported[-1]['R']),
        'layers_validated':len(transported),
    }


def certified_effect_interval(nominal_norm,clean_lipschitz,
                              perturbed_lipschitz,radius,
                              anchor_l2_budget,direction_l2_budget):
    """Uniform interval for ||F(h'+r*u')-F(h')|| under Euclidean budgets."""
    r,da,du=map(Q,(radius,anchor_l2_budget,direction_l2_budget))
    if r<=0 or min(da,du)<0:
        raise ValueError('invalid robust budget')
    eps=up(clean_lipschitz*aq(da)+
           perturbed_lipschitz*aq(da+r*du))
    lo=(nominal_norm[0]-eps).lower()
    hi=(nominal_norm[1]+eps).upper()
    lower=lo if lo>0 else arb(0)
    if not (lower.is_finite() and hi.is_finite() and hi>=lower):
        raise ArithmeticError('invalid validated effect interval')
    return (lower,hi),str(eps)


def rank_interval_effects(active,control):
    """Return strict interval ordering of nonnegative effect norms."""
    if active[0]>control[1]:
        return 'sae_active_greater'
    if active[1]<control[0]:
        return 'control_greater'
    return 'inconclusive'


def source_locked_run(checkpoint=None,allow_fetch=False,max_test_puzzles=2,
                      seeds=(71,),steps=300,radii=('0.02','0.2'),
                      budgets=('0','1e-12','1e-10','1e-8','1e-6'),
                      precision=160):
    if not 1<=max_test_puzzles<=16 or not seeds or steps<1 or precision<100:
        raise ValueError('invalid experiment dimensions')
    radii=tuple(str(Q(x)) for x in radii)
    budgets=tuple(str(Q(x)) for x in budgets)
    if not radii or any(Q(x)<=0 for x in radii) or not budgets or any(Q(x)<0 for x in budgets):
        raise ValueError('positive radii and nonnegative uncertainty budgets required')
    ctx.prec=precision
    torch.set_num_threads(1)
    lock=json.loads((PRE/'source_lock.json').read_text())
    raw,digest=locked_bytes(lock,checkpoint_path=checkpoint,allow_fetch=allow_fetch)
    if digest!=CHECKPOINT_SHA:
        raise ValueError('source checkpoint hash mismatch')
    teacher=load_pretrained(lock,raw).eval()
    (inputs,_,_),roster=prepare_roster(lock)
    if roster!=EXPECTED_ROSTER:
        raise ValueError('source roster hash mismatch')
    split=split_ids(64)
    test=split['test'][:max_test_puzzles]
    with torch.no_grad():
        stages=teacher.reason_states(inputs)[1].detach()
    numteacher=copy.deepcopy(teacher).double().eval()
    validated=TaylorNet(teacher)
    head_norm,head_LDL=verified_spectral(teacher.head.weight)
    assert head_norm>0

    frozen={}
    for pid in test:
        goal=int(torch.where(inputs[pid]==0)[0][0])
        anchor=exact_matrix(stages[pid])
        final,tape=validated.forward(anchor,steps=3)
        clean=validated.logits(final,goal)
        frozen[pid]={'goal':goal,'anchor':anchor,'clean':clean,'tape':tape}
    cases=[]
    for seed in seeds:
        sae,mu,std,fit=fit_sae(stages[split['train']],
                              stages[split['val']],seed=seed,steps=steps)
        for pid in test:
            item=frozen[pid]
            goal,anchor,clean=item['goal'],item['anchor'],item['clean']
            _,_,J,top,_,_=goal_jacobian(numteacher,stages[pid:pid+1],goal)
            dirs,_=sae_direction_set(sae,stages[pid,goal],mu,std,seed,pid,top)
            dirs={name:normalized_arb_direction(dirs[name])[0] for name in DIRECTIONS}
            # Pointwise center output and the verified derivative transport
            # are reused at every chosen uncertainty budget.
            endpoints={}
            for radius in radii:
                per_dir={}
                for name in DIRECTIONS:
                    shifted=shifted_anchor(anchor,goal,dirs[name],radius)
                    final,tape=validated.forward(shifted,steps=3)
                    out=validated.logits(final,goal)
                    square=squared_logit_effect(out,clean)
                    per_dir[name]={'tape':tape,
                                   'nominal_norm':exact_effect_norm_interval(square)}
                endpoints[radius]=per_dir
            for eta in budgets:
                clean_lipschitz,clean_metadata=validated_lipschitz_on_ball(
                    validated,item['tape'],head_norm,Q(eta))
                for radius in radii:
                    comparisons={}; effects={}; errors={}
                    for name in DIRECTIONS:
                        try:
                            r=Q(radius)
                            e=Q(eta)*(1+r)
                            lip,meta=validated_lipschitz_on_ball(
                                validated,endpoints[radius][name]['tape'],head_norm,e)
                            rng,err=certified_effect_interval(
                                endpoints[radius][name]['nominal_norm'],
                                clean_lipschitz,lip,r,Q(eta),Q(eta))
                            effects[name]={'interval':rng,
                                           'verified_error_upper':err,
                                           'endpoint_local_L_upper':str(lip),
                                           'endpoint_layers':meta}
                        except (ArithmeticError,ValueError,OverflowError,ZeroDivisionError) as exc:
                            errors[name]=type(exc).__name__+': '+str(exc)[:240]
                    for control in CONTROLS:
                        if 'sae_active' in effects and control in effects:
                            comparisons[control]=rank_interval_effects(
                                effects['sae_active']['interval'],effects[control]['interval'])
                        else:
                            comparisons[control]='inconclusive'
                    cases.append({
                        'seed':seed,'puzzle_id':pid,'goal_token':goal,
                        'radius':radius,'common_l2_uncertainty':eta,
                        'clean_endpoint_lipschitz_upper':str(clean_lipschitz),
                        'clean_endpoint_metadata':clean_metadata,
                        'effects':{name:{**{k:v for k,v in v.items() if k!='interval'},
                                         'rigorous_effect_norm_interval':list(map(str,v['interval']))}
                                   for name,v in effects.items()},
                        'failed_enclosures':errors,'comparisons':comparisons,
                    })
                    print('LIPSCHITZ_ROBUST',seed,pid,radius,eta,comparisons,flush=True)
    summary={}
    for radius in radii:
        for eta in budgets:
            for control in CONTROLS:
                selected=[case['comparisons'][control] for case in cases
                          if case['radius']==radius and case['common_l2_uncertainty']==eta]
                summary[f'{radius}|{eta}|{control}']={
                    'sae_active_greater':selected.count('sae_active_greater'),
                    'control_greater':selected.count('control_greater'),
                    'inconclusive':selected.count('inconclusive'),
                    'total':len(selected),
                }
    return {
        'schema':'SMC-10C/verified-endpoint-Lipschitz-neighborhood-v1',
        'checkpoint_sha256':digest,'roster_sha256':roster,
        'source_commit':lock['revision'],'test_puzzle_ids':test,
        'seeds':list(seeds),'sae_steps':steps,'radii':list(radii),
        'common_l2_budget_grid':list(budgets),'arb_precision_bits':precision,
        'verified_head_spectral_upper':str(head_norm),
        'head_LDL_certificate':head_LDL,
        'mathematical_object':'same 48d goal-token anchor/direction L2 uncertainty contract as SMC-10, with validated local endpoint Lipschitz transport and nominal Arb interval effects',
        'boundary':'exact-real source model; no Torch hardware rounding, no uncertainty at remote tokens, no decoder retraining or semantic feature guarantee',
        'cases':cases,'summary':summary,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--fetch',action='store_true')
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--max-test-puzzles',type=int,default=2)
    p.add_argument('--seeds',default='71')
    p.add_argument('--steps',type=int,default=300)
    p.add_argument('--radii',default='0.02,0.2')
    p.add_argument('--budgets',default='0,1e-12,1e-10,1e-8,1e-6')
    p.add_argument('--precision',type=int,default=160)
    p.add_argument('--out',type=Path,default=Path('runs/smc10/lipschitz-pilot.json'))
    a=p.parse_args()
    result=source_locked_run(a.checkpoint,a.fetch,a.max_test_puzzles,
                             tuple(int(s) for s in a.seeds.split(',')),a.steps,
                             tuple(s for s in a.radii.split(',')),
                             tuple(s for s in a.budgets.split(',')),a.precision)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({
        'out':str(a.out),'checkpoint_sha256':result['checkpoint_sha256'],
        'n_cases':len(result['cases']),
        'summary':result['summary'],
    },indent=2))


if __name__=='__main__':
    main()

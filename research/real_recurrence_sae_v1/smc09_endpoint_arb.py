"""SMC-09: Arb-validated finite endpoint ordering for learned SAE interventions.

All interval-forward calls evaluate the *exact-real* frozen recurrent function
at exact binary stage-one anchors and intervention directions specified as
normalized exact dyadic vectors. The intervention radius is an exact rational.

This is an endpoint certificate, not a curvature tube, an FP64 rounding
certificate, or a claim that the directions have semantic interpretation.
"""
import argparse
import copy
import json
import math
import sys
from fractions import Fraction as Q
from pathlib import Path

import torch
from flint import arb, arb_mat, ctx

HERE=Path(__file__).resolve().parent
PRE=HERE.parent/'levin_sieve_v1'/'pretrained_trm'
ARB_DIR=HERE.parent/'levin_sieve_v1'/'goal_taylor_v1'
for module_dir in (HERE,PRE,ARB_DIR):
    sys.path.insert(0,str(module_dir))

from arb_engine import TaylorNet, aq, exact_matrix
from run_checkpoint import locked_bytes, load_pretrained, prepare_roster
from smc07_real_sae import fit_sae,split_ids,EXPECTED_ROSTER
from smc08_curvature import goal_jacobian,sae_direction_set

CHECKPOINT_SHA='3bab6189d4f25fe1e7a326a6b29ddc3141d21325f835ed1cd6409c0ac31447e1'
DIRECTIONS=('sae_active','sae_inactive','isotropic','jacobian_optimal')
CONTROLS=('sae_inactive','isotropic','jacobian_optimal')


def normalized_arb_direction(torch_vector):
    """Exact-real L2 normalization of the 48 frozen FP64 dyadic components.

    The source vector components are interpreted as *exact binary rationals*.
    Division by the exact real Euclidean norm is enclosed by Arb.
    """
    if torch_vector.ndim!=1 or len(torch_vector)!=48:
        raise ValueError('expected one 48-dimensional direction')
    raw=[Q.from_float(float(v)) for v in torch_vector.detach().cpu().tolist()]
    if not any(raw):
        raise ValueError('zero direction cannot be normalized')
    norm_sq=sum((v*v for v in raw),Q(0))
    exact=[aq(v) for v in raw]
    radius=aq(norm_sq).sqrt()
    if not radius.lower()>0:
        raise ArithmeticError('uncertain normalization denominator')
    unit=[v/radius for v in exact]
    check=sum((v*v for v in unit),arb(0))
    if not check.lower() <= 1 <= check.upper():
        raise ArithmeticError('normalized direction interval excludes L2 unit norm')
    return unit,{'dyadic_input_norm_sq':str(norm_sq),'arb_unit_norm2':str(check)}


def continue_arb(net,anchor):
    """Existing validated mixing/LN operations, from an arb_mat anchor."""
    if anchor.nrows()!=81 or anchor.ncols()!=48:
        raise ValueError('81x48 full recurrent anchor required')
    z=anchor
    for step in range(3):
        z,_=net.mix(z,'t',True,f'{step}.token')
        z,_=net.ln(z,net.norms[0],f'{step}.ln1')
        z,_=net.mix(z,'c',False,f'{step}.channel')
        z,_=net.ln(z,net.norms[1],f'{step}.ln2')
    return z


def endpoint_logits(net,anchor,goal):
    if not 0<=goal<81:
        raise ValueError('goal token index out of range')
    return net.logits(continue_arb(net,anchor),goal)


def shifted_anchor(anchor,goal,direction,radius):
    """Construct the exact-real input h + r*u with interval inclusions."""
    r=Q(radius)
    if not r>0 or len(direction)!=48:
        raise ValueError('positive rational radius and 48-vector required')
    coords=[[anchor[i,j] for j in range(48)] for i in range(81)]
    for j in range(48):
        coords[goal][j] += aq(r)*direction[j]
    return arb_mat(coords)


def squared_logit_effect(perturbed_logits,clean_logits):
    if len(perturbed_logits)!=10 or len(clean_logits)!=10:
        raise ValueError('ten frozen logit coordinates required')
    return sum(((x-y)*(x-y) for x,y in zip(perturbed_logits,clean_logits)),arb(0))


def classify_ordering(effect_a2,effect_b2):
    """Only a strictly separated interval is labeled a theorem."""
    gap=effect_a2-effect_b2
    if gap.lower()>0:
        proof='sae_active_greater'
    elif gap.upper()<0:
        proof='control_greater'
    else:
        proof='inconclusive'
    return {
        'proof':proof,
        'certified_squared_effect_gap_interval':str(gap),
        'squared_effect_gap_midpoint_float':float(gap.mid()),
        'signed_gap_contains_zero':bool(gap.lower()<=0<=gap.upper()),
        'interval_width_float':float(gap.rad())*2,
    }


def source_model_run(checkpoint=None,allow_fetch=False,max_test_puzzles=2,
                     seeds=(71,),steps=300,radii=('0.02','0.2'),
                     precision=160):
    if not 1<=max_test_puzzles<=16 or precision<100 or steps<1:
        raise ValueError('test size, precision or training steps invalid')
    radii=tuple(str(Q(r)) for r in radii)
    if not radii or any(Q(r)<=0 for r in radii):
        raise ValueError('positive radii required')
    ctx.prec=precision
    torch.set_num_threads(1)
    lock=json.loads((PRE/'source_lock.json').read_text())
    raw,sha=locked_bytes(lock,checkpoint,allow_fetch)
    if sha!=CHECKPOINT_SHA:
        raise ValueError('checkpoint provenance mismatch')
    model=load_pretrained(lock,raw).eval()
    (inputs,targets,_),roster=prepare_roster(lock)
    if roster!=EXPECTED_ROSTER:
        raise ValueError('heldout roster provenance mismatch')
    split=split_ids(64)
    test=split['test'][:max_test_puzzles]
    with torch.no_grad():
        states=model.reason_states(inputs)[1].detach()
    net=TaylorNet(model)
    teacher_double=copy.deepcopy(model).double().eval()

    frozen={}
    for pid in test:
        goal=int(torch.where(inputs[pid]==0)[0][0])
        anchor=exact_matrix(states[pid])
        clean=endpoint_logits(net,anchor,goal)
        frozen[pid]=(goal,anchor,clean)
    cases=[]
    for seed in seeds:
        sae,mu,std,fit=fit_sae(states[split['train']],
                               states[split['val']],seed=seed,steps=steps)
        for pid in test:
            goal,anchor,clean=frozen[pid]
            _,_,J,top,sv,_=goal_jacobian(
                teacher_double,states[pid:pid+1],goal)
            directions,selection=sae_direction_set(
                sae,states[pid,goal],mu,std,seed,pid,top)
            normed={}
            for name in DIRECTIONS:
                normed[name]=normalized_arb_direction(directions[name])
            for radius in radii:
                effects={}
                for name in DIRECTIONS:
                    perturbed=shifted_anchor(anchor,goal,normed[name][0],radius)
                    response=endpoint_logits(net,perturbed,goal)
                    effect=squared_logit_effect(response,clean)
                    if effect.upper()<0:
                        raise ArithmeticError('squared norm interval strictly negative')
                    effects[name]={
                        'interval':effect,
                        'squared_effect_interval':str(effect),
                        'squared_effect_midpoint':float(effect.mid()),
                        'effect_norm_midpoint':math.sqrt(max(0,float(effect.mid()))),
                        'exact_unit_direction_proven':True,
                    }
                compare={control:classify_ordering(
                    effects['sae_active']['interval'],effects[control]['interval'])
                    for control in CONTROLS}
                cases.append({
                    'puzzle_id':pid,'seed':seed,'radius':radius,
                    'goal_token':goal,'selected_feature_ids':selection,
                    'squared_effects':{
                        name:{k:v for k,v in row.items() if k!='interval'}
                        for name,row in effects.items()},
                    'comparisons':compare,
                })
                print('CERTIFIED_ENDPOINT',seed,pid,radius,
                      {k:v['proof'] for k,v in compare.items()},flush=True)

    summary={}
    for r in radii:
        for control in CONTROLS:
            selected=[row['comparisons'][control]['proof']
                      for row in cases if row['radius']==r]
            summary[f'{r}_active_minus_{control}']={
                'sae_active_greater':selected.count('sae_active_greater'),
                'control_greater':selected.count('control_greater'),
                'inconclusive':selected.count('inconclusive'),
                'total':len(selected),
            }
    return {
        'schema':'SMC-09/validated-exact-real-endpoint-ordering-v1',
        'source_commit':lock['revision'],
        'checkpoint_sha256':sha,'roster_sha256':roster,
        'source_model':'44,001-parameter externally pretrained TRM-inspired Sudoku',
        'puzzle_train_validation_test_ids':split,
        'test_puzzle_ids':test,'seeds':list(seeds),
        'sae_steps':steps,'radii':list(radii),'arb_precision_bits':precision,
        'mathematical_object':'exact-real three-step frozen recurrence from an FP32 source anchor; radius rational; each frozen FP64 intervention direction interpreted as a dyadic vector and renormalized exactly to unit length',
        'certificate_interpretation':'strict Arb interval separation of squared ten-goal-logit effect norms at two fixed intervention endpoints, with same norm and goal token',
        'limits':'does not validate Torch FP64 roundoff, all radii along a path, semantic feature identity, or a uniform Hessian enclosure',
        'cases':cases,'summary':summary,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--fetch',action='store_true')
    p.add_argument('--max-test-puzzles',type=int,default=2)
    p.add_argument('--sae-seeds',default='71')
    p.add_argument('--sae-steps',type=int,default=300)
    p.add_argument('--radii',default='0.02,0.2')
    p.add_argument('--precision',type=int,default=160)
    p.add_argument('--out',type=Path,default=Path('runs/smc09/pilot.json'))
    a=p.parse_args()
    result=source_model_run(a.checkpoint,a.fetch,a.max_test_puzzles,
                            tuple(int(x) for x in a.sae_seeds.split(',')),
                            a.sae_steps,tuple(s.strip() for s in a.radii.split(',')),
                            a.precision)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'out':str(a.out),'checkpoint_sha256':result['checkpoint_sha256'],
                      'cases':len(result['cases']),'summary':result['summary']},indent=2))


if __name__=='__main__':
    main()

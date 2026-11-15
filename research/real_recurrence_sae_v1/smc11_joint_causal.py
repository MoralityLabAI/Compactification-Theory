"""SMC-11C: robust endpoint ordering AND exact same-fit SAE selection proof.

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
import hashlib
import struct
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
from smc11_selection import goal_selection_cases
from smc09_endpoint_arb import (
    CHECKPOINT_SHA,DIRECTIONS,CONTROLS,shifted_anchor,
    normalized_arb_direction,squared_logit_effect,
)




def forward_arb_with_tape(net,stage):
    """The existing TaylorNet forward recurrence on an existing Arb matrix.

    TaylorNet.forward first converts a Torch tensor with exact_matrix.
    Here the input is already an Arb matrix (possibly a normalized exact-real
    intervention endpoint), so this variant preserves the Arb enclosures.
    """
    if stage.nrows()!=81 or stage.ncols()!=48:
        raise ValueError('expected complete 81x48 source anchor')
    z=stage
    tape=[]
    for k in range(3):
        z,op=net.mix(z,'t',True,f'{k}.token')
        tape.append(op)
        z,op=net.ln(z,net.norms[0],f'{k}.ln1')
        tape.append(op)
        z,op=net.mix(z,'c',False,f'{k}.channel')
        tape.append(op)
        z,op=net.ln(z,net.norms[1],f'{k}.ln2')
        tape.append(op)
    return z,tape

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




def fp32_bytes(t):
    data=t.detach().cpu().float().contiguous().reshape(-1).tolist()
    return struct.pack('<'+'f'*len(data),*data)


def frozen_sae_fingerprint(sae,mean,scale):
    h=hashlib.sha256()
    tensors={**{'sae.'+name:t for name,t in sae.state_dict().items()},
             'train_mean':mean,'train_scale':scale}
    for name in sorted(tensors):
        t=tensors[name]
        h.update(name.encode('utf-8')+b'\x00')
        h.update(str(tuple(t.shape)).encode('ascii')+b'\x00')
        h.update(fp32_bytes(t))
    return h.hexdigest()


def serialize_snapshot(sae,mean,scale,teacher_sha,seed,target):
    """Store fitted SAE and train-only norm in a replayable weights-only file."""
    target=Path(target)
    target.parent.mkdir(parents=True,exist_ok=True)
    state={name:t.detach().cpu().clone() for name,t in sae.state_dict().items()}
    payload={'seed':seed,'teacher_sha256':teacher_sha,
             'state_dict':state,
             'train_mean':mean.detach().cpu().clone(),
             'train_scale':scale.detach().cpu().clone()}
    torch.save(payload,target)
    restored=torch.load(target,weights_only=True,map_location='cpu')
    assert restored['seed']==seed and restored['teacher_sha256']==teacher_sha
    assert all(torch.equal(restored['state_dict'][name],t)
               for name,t in state.items())
    assert torch.equal(restored['train_mean'],payload['train_mean'])
    assert torch.equal(restored['train_scale'],payload['train_scale'])
    return hashlib.sha256(target.read_bytes()).hexdigest()



def same_fit_policy(proof,identity,control,eta):
    """Causal label plus SAME-fit exact top1/top4 policy selection guard."""
    if control not in CONTROLS:
        raise ValueError('unknown causal reference control')
    guard='top4' if control=='sae_inactive' else 'top1'
    if identity['status']!='certified' or not identity['budget_guards'][eta][guard]:
        return 'abstain_selection'
    return proof

def source_locked_run(checkpoint=None,allow_fetch=False,max_test_puzzles=2,
                      seeds=(71,),steps=300,radii=('0.02','0.2'),
                      budgets=('0','1e-12','1e-10','1e-8','1e-6'),
                      precision=160,weights_dir=None):
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
        final,tape=forward_arb_with_tape(validated,anchor)
        clean=validated.logits(final,goal)
        frozen[pid]={'goal':goal,'anchor':anchor,'clean':clean,'tape':tape}
    cases=[]
    exact_selection_cases=[]
    frozen_sae_snapshots=[]
    for seed in seeds:
        sae,mu,std,fit=fit_sae(stages[split['train']],
                              stages[split['val']],seed=seed,steps=steps)
        selection=goal_selection_cases(stages,inputs,sae,mu,std,test,seed)
        by_pid={r['puzzle_id']:r for r in selection}
        if len(by_pid)!=len(test):
            raise ValueError('duplicate source identity cases')
        exact_selection_cases.extend(selection)
        sha=frozen_sae_fingerprint(sae,mu,std)
        snapshot_file_sha=None
        if weights_dir is not None:
            snapshot_file_sha=serialize_snapshot(sae,mu,std,digest,seed,
                Path(weights_dir)/('sae_seed'+str(seed)+'.pt'))
        frozen_sae_snapshots.append({
            'seed':seed,'trained_sae_sha256':sha,
            'snapshot_file_sha256':snapshot_file_sha,
            'selected_best_step':fit['best_step'],
            'validation_mse':fit['best_val_mse']
        })
        for pid in test:
            item=frozen[pid]
            goal,anchor,clean=item['goal'],item['anchor'],item['clean']
            _,_,J,top,_,_=goal_jacobian(numteacher,stages[pid:pid+1],goal)
            dirs,source_selected=sae_direction_set(sae,stages[pid,goal],mu,std,seed,pid,top)
            identity=by_pid[pid]
            if identity['status']=='certified' and source_selected['active_feature_id']!=identity['top1_feature']:
                raise AssertionError('causal decoder was not same exact fit/selected SAE atom')
            dirs={name:normalized_arb_direction(dirs[name])[0] for name in DIRECTIONS}
            # Pointwise center output and the verified derivative transport
            # are reused at every chosen uncertainty budget.
            endpoints={}
            for radius in radii:
                per_dir={}
                for name in DIRECTIONS:
                    shifted=shifted_anchor(anchor,goal,dirs[name],radius)
                    final,tape=forward_arb_with_tape(validated,shifted)
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
                    policy_comparisons={
                        control:same_fit_policy(comparisons[control],identity,control,eta)
                        for control in CONTROLS}
                    cases.append({
                        'seed':seed,'puzzle_id':pid,'goal_token':goal,
                        'radius':radius,'common_l2_uncertainty':eta,
                        'clean_endpoint_lipschitz_upper':str(clean_lipschitz),
                        'clean_endpoint_metadata':clean_metadata,
                        'effects':{name:{**{k:v for k,v in v.items() if k!='interval'},
                                         'rigorous_effect_norm_interval':list(map(str,v['interval']))}
                                   for name,v in effects.items()},
                        'failed_enclosures':errors,'comparisons':comparisons,
                        'composed_policy_comparisons':policy_comparisons,
                        'sae_snapshot_sha256':sha,
                        'exact_top1_guard':identity['budget_guards'][eta]['top1'],
                        'exact_top4_guard':identity['budget_guards'][eta]['top4'],
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
    joint_summary={}
    for radius in radii:
        for eta in budgets:
            for control in CONTROLS:
                labels=[case['composed_policy_comparisons'][control]
                        for case in cases if case['radius']==radius
                        and case['common_l2_uncertainty']==eta]
                joint_summary[f'{radius}|{eta}|{control}']={
                    'sae_active_greater':labels.count('sae_active_greater'),
                    'control_greater':labels.count('control_greater'),
                    'inconclusive_original':labels.count('inconclusive'),
                    'abstain_selection':labels.count('abstain_selection'),
                    'total':len(labels)}
    return {
        'schema':'SMC-11C/same-fit-SAE-identity-and-Arb-local-causal-v1',
        'checkpoint_sha256':digest,'roster_sha256':roster,
        'source_commit':lock['revision'],'test_puzzle_ids':test,
        'seeds':list(seeds),'sae_steps':steps,'radii':list(radii),
        'common_l2_budget_grid':list(budgets),'arb_precision_bits':precision,
        'verified_head_spectral_upper':str(head_norm),
        'head_LDL_certificate':head_LDL,
        'mathematical_object':'same 48d goal-token anchor/direction L2 uncertainty contract as SMC-10, with validated local endpoint Lipschitz transport and nominal Arb interval effects',
        'boundary':'exact-real source model; no Torch hardware rounding, no uncertainty at remote tokens, no decoder retraining or semantic feature guarantee',
        'cases':cases,'summary':summary,
        'same_fit_composed_summary':joint_summary,
        'exact_goal_selection_cases':exact_selection_cases,
        'frozen_sae_snapshots':frozen_sae_snapshots,
        'source_exactness':'The same fitted SAE FP32 bytes and train-only mean/std appear in the exact rational selection audit and the 160-bit Arb causal local-Lipschitz evaluation in one Python process; saved weights-only snapshots retain those exact bytes.',
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
    p.add_argument('--weights-dir',type=Path,default=Path('runs/smc11/sae-weights'))
    p.add_argument('--out',type=Path,default=Path('runs/smc11/joint-pilot.json'))
    a=p.parse_args()
    result=source_locked_run(a.checkpoint,a.fetch,a.max_test_puzzles,
                             tuple(int(s) for s in a.seeds.split(',')),a.steps,
                             tuple(s for s in a.radii.split(',')),
                             tuple(s for s in a.budgets.split(',')),a.precision,a.weights_dir)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({
        'out':str(a.out),'checkpoint_sha256':result['checkpoint_sha256'],
        'n_cases':len(result['cases']),
        'frozen_sae_snapshots':result['frozen_sae_snapshots'],
        'same_fit_composed_summary':result['same_fit_composed_summary'],
    },indent=2))


if __name__=='__main__':
    main()

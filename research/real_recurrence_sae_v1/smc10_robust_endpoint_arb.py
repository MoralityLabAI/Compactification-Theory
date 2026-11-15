"""SMC-10: Arb-validated robust causal ordering on uncertain SAE intervention boxes.

A complete 48-channel stage-one goal-token anchor h' may vary in an L2
ball of radius δ_h, and the *exact unit* active/control vectors may each
vary in L2 by δ_u, δ_v. The other 80 tokens are fixed. Componentwise Arb
boxes enclose the entire three-factor uncertainty set, even though they
discard correlations. Exact-real recurrent endpoint evaluation then yields
a valid *uniform* squared-effect ranking whenever output intervals separate.

The frozen source model, data split and SAE fitting are identical to SMC-09.
The experiment does not certify hardware FP64 roundoff, decoder retraining,
uncertainty at remote tokens or an all-radius theorem.
"""
import argparse
import copy
import json
import math
from fractions import Fraction as Q
from pathlib import Path
import sys

import torch
from flint import arb,arb_mat,ctx

HERE=Path(__file__).resolve().parent
PRE=HERE.parent/'levin_sieve_v1'/'pretrained_trm'
ARB_DIR=HERE.parent/'levin_sieve_v1'/'goal_taylor_v1'
for module_dir in (HERE,PRE,ARB_DIR):
    sys.path.insert(0,str(module_dir))

from arb_engine import TaylorNet,LN,aq,exact_matrix
from run_checkpoint import locked_bytes,load_pretrained,prepare_roster
from smc07_real_sae import fit_sae,split_ids,EXPECTED_ROSTER
from smc08_curvature import goal_jacobian,sae_direction_set
from smc09_endpoint_arb import (
    CHECKPOINT_SHA,DIRECTIONS,CONTROLS,
    endpoint_logits,normalized_arb_direction,
    squared_logit_effect,classify_ordering
)




class RobustTaylorNet(TaylorNet):
    """Source-equivalent LayerNorm with mathematically nonnegative squares.

    In interval arithmetic x*x for x crossing zero can have negative lower
    values even though the true x^2 is nonnegative. Intersecting the product
    enclosure with [0,∞) is sound and prevents artificial negative variance.
    This override leaves the exact real pointwise function unchanged.
    """
    def ln(self,X,norm,name):
        gamma=[arb(float(z)) for z in norm.weight.detach().cpu().tolist()]
        beta=[arb(float(z)) for z in norm.bias.detach().cpu().tolist()]
        eps=arb(float(norm.eps))
        d=len(gamma)
        us=[];ss=[];out=[]
        for row in X.tolist():
            mean=sum(row,arb(0))/d
            u=[v-mean for v in row]
            # Correct nonnegative range of every exact square.
            nonneg_squares=[(v*v).nonnegative_part() for v in u]
            avg_variance=(sum(nonneg_squares,arb(0))/d).nonnegative_part()
            squared_denom=eps+avg_variance
            if not squared_denom.lower()>0:
                raise ArithmeticError('uncertain denominator after nonnegative-square refinement')
            den=squared_denom.sqrt()
            if not den.lower()>0:
                raise ArithmeticError('uncertain LayerNorm square-root denominator')
            us.append(u);ss.append(den)
            out.append([g*v/den+b for g,v,b in zip(gamma,u,beta)])
        return arb_mat(out),LN(us,ss,gamma,eps,name)

def arb_sym_ball(radius):
    """Validated symmetric ball around zero, with an upward-rounded radius.

    python-flint's Arb radius constructor admits a rational-string radius.
    """
    r=Q(radius)
    if r<0:
        raise ValueError('uncertainty radius must be nonnegative')
    return arb(0,str(r)) if r else arb(0)


def uncertain_anchor(anchor,goal,direction,radius,
                     anchor_l2_budget,direction_l2_budget=Q(0)):
    """Full 81x48 Arb input box enclosing an uncertain stage-one goal patch.

    Clean endpoint: direction=None, radius=0, box width δ_h.
    Intervention endpoint: nominal center h+r*u, box width δ_h+r*δ_u.
    For any admissible h' and unit u' within those L2 distances, their
    state h'+r*u' is in this coordinatewise hull.
    """
    if anchor.nrows()!=81 or anchor.ncols()!=48 or not 0<=goal<81:
        raise ValueError('expected 81x48 recurrent state and goal id')
    da,du,r=map(Q,(anchor_l2_budget,direction_l2_budget,radius))
    if min(da,du,r)<0:
        raise ValueError('uncertainty and radius must be nonnegative')
    if direction is None:
        if r!=0 or du!=0:
            raise ValueError('clean endpoint needs zero radius/direction uncertainty')
    elif len(direction)!=48 or r<=0:
        raise ValueError('intervened endpoint needs unit 48-vector and positive radius')
    width=da+r*du
    uncertainty=arb_sym_ball(width)
    coords=[[anchor[i,j] for j in range(48)] for i in range(81)]
    for j in range(48):
        shifted=(aq(r)*direction[j] if direction is not None else arb(0))
        coords[goal][j]=coords[goal][j]+shifted+uncertainty
    return arb_mat(coords)


def robust_effects(net,anchor,goal,directions,radius,
                   anchor_budget,direction_budgets=None):
    """Strict ranking applies to every admitted common anchor and direction.

    Direction budgets are L2 distances from each exact unit nominal vector.
    Every comparison uses one same uncertain clean endpoint. Independence
    in the interval extension widens enclosures but cannot fake a proof.
    """
    r,da=Q(radius),Q(anchor_budget)
    if r<=0 or da<0:
        raise ValueError('positive intervention radius and nonnegative anchor uncertainty')
    budgets={name:Q(0) for name in DIRECTIONS}
    if direction_budgets:
        budgets.update({name:Q(v) for name,v in direction_budgets.items()})
    if any(name not in DIRECTIONS for name in budgets) or any(v<0 for v in budgets.values()):
        raise ValueError('invalid direction budgets')
    if set(directions)!=set(DIRECTIONS):
        raise ValueError('every declared intervention direction required')

    baseline=endpoint_logits(net,uncertain_anchor(anchor,goal,None,0,da),goal)
    outputs={}
    errors={}
    for name in DIRECTIONS:
        try:
            perturbed=uncertain_anchor(anchor,goal,directions[name],r,da,budgets[name])
            logits=endpoint_logits(net,perturbed,goal)
            interval=squared_logit_effect(logits,baseline)
            if not interval.is_finite() or interval.upper()<0:
                raise ArithmeticError('nonfinite or impossible squared-effect enclosure')
            outputs[name]=interval
        except (ArithmeticError,ValueError,ZeroDivisionError,OverflowError) as exc:
            errors[name]=type(exc).__name__+': '+str(exc)[:240]
    comparisons={}
    for control in CONTROLS:
        if 'sae_active' in outputs and control in outputs:
            comparisons[control]=classify_ordering(outputs['sae_active'],outputs[control])
        else:
            comparisons[control]={'proof':'inconclusive','interval_error':errors}
    return {
        'radius':str(r),
        'anchor_l2_budget':str(da),
        'direction_l2_budgets':{k:str(v) for k,v in budgets.items()},
        'clean_box_half_width_linf':str(da),
        'perturbed_box_half_width_linf':{k:str(da+r*v) for k,v in budgets.items()},
        'squared_effects':{name:str(iv) for name,iv in outputs.items()},
        'failures':errors,
        'comparisons':comparisons,
        'certificate_domain':'all shared goal-token anchors within L2 budget and exact unit vectors within respective L2 distances; remote tokens and weights fixed',
    }


def summary_table(cases,radii,budgets):
    result={}
    for radius in radii:
        for eta in budgets:
            for control in CONTROLS:
                proofs=[c['comparisons'][control]['proof'] for c in cases
                        if c['radius']==str(Q(radius)) and
                           c['anchor_l2_budget']==str(Q(eta))]
                result[f'{Q(radius)}|{Q(eta)}|{control}']={
                    'active_greater':proofs.count('sae_active_greater'),
                    'control_greater':proofs.count('control_greater'),
                    'inconclusive':proofs.count('inconclusive'),
                    'total':len(proofs),
                }
    return result


def run(checkpoint=None,allow_fetch=False,max_test_puzzles=2,
        seeds=(71,),steps=300,radii=('0.02','0.2'),
        budgets=('0','1e-12','1e-10','1e-8'),precision=160):
    if not 1<=max_test_puzzles<=16 or precision<100 or steps<1 or not seeds:
        raise ValueError('invalid run size, seeds or precision')
    radii=tuple(str(Q(x)) for x in radii)
    budgets=tuple(str(Q(x)) for x in budgets)
    if not radii or any(Q(r)<=0 for r in radii) or not budgets or any(Q(b)<0 for b in budgets):
        raise ValueError('radii must be positive, uncertainty budgets nonnegative')
    if len(set(budgets))!=len(budgets):
        raise ValueError('repeated uncertainty budget')
    ctx.prec=precision
    torch.set_num_threads(1)
    lock=json.loads((PRE/'source_lock.json').read_text())
    raw,digest=locked_bytes(lock,checkpoint_path=checkpoint,allow_fetch=allow_fetch)
    if digest!=CHECKPOINT_SHA:
        raise ValueError('wrong source checkpoint digest')
    teacher=load_pretrained(lock,raw).eval()
    (inputs,_,_),roster=prepare_roster(lock)
    if roster!=EXPECTED_ROSTER:
        raise ValueError('wrong puzzle roster')
    split=split_ids(64)
    chosen=split['test'][:max_test_puzzles]
    with torch.no_grad():
        stages=teacher.reason_states(inputs)[1].detach()
    model64=copy.deepcopy(teacher).double().eval()
    validated=RobustTaylorNet(teacher)
    frozen={}
    for pid in chosen:
        goal=int(torch.where(inputs[pid]==0)[0][0])
        frozen[pid]={'goal':goal,'arb_anchor':exact_matrix(stages[pid])}
    cases=[]
    for seed in seeds:
        sae,mu,std,fit=fit_sae(stages[split['train']],
                              stages[split['val']],seed=seed,steps=steps)
        for pid in chosen:
            anchor=frozen[pid]['arb_anchor']
            goal=frozen[pid]['goal']
            _,_,J,top,sv,_=goal_jacobian(model64,stages[pid:pid+1],goal)
            torchdirs,selection=sae_direction_set(sae,stages[pid,goal],mu,std,seed,pid,top)
            dirs={name:normalized_arb_direction(torchdirs[name])[0]
                  for name in DIRECTIONS}
            # Cache each uncertain clean Arb forward state per budget.
            clean_outputs={}
            clean_errors={}
            for eta in budgets:
                try:
                    clean_outputs[eta]=endpoint_logits(
                        validated,uncertain_anchor(anchor,goal,None,0,Q(eta)),goal)
                except (ArithmeticError,ValueError,OverflowError,ZeroDivisionError) as exc:
                    clean_outputs[eta]=None
                    clean_errors[eta]=type(exc).__name__+': '+str(exc)[:240]
            for radius in radii:
                for eta in budgets:
                    budgets_by_direction={k:Q(eta) for k in DIRECTIONS}
                    if clean_outputs[eta] is None:
                        errors={'clean_endpoint':clean_errors[eta]}
                        result={
                            'radius':str(Q(radius)),
                            'anchor_l2_budget':str(Q(eta)),
                            'direction_l2_budgets':{k:str(v) for k,v in budgets_by_direction.items()},
                            'clean_box_half_width_linf':str(Q(eta)),
                            'perturbed_box_half_width_linf':{k:str(Q(eta)+Q(radius)*v)
                                                              for k,v in budgets_by_direction.items()},
                            'squared_effects':{},
                            'comparisons':{control:{'proof':'inconclusive',
                                                    'interval_error':errors}
                                           for control in CONTROLS},
                            'failures':errors,
                        }
                    else:
                        result=robust_effects_with_clean(
                            validated,anchor,goal,dirs,Q(radius),Q(eta),
                            budgets_by_direction,clean_outputs[eta])
                    cases.append({
                        'seed':seed,'puzzle_id':pid,'goal_token':goal,
                        'selected_sae_features':selection,**result})
                    print('ROBUST_ENDPOINT',seed,pid,radius,eta,
                          {k:v['proof'] for k,v in result['comparisons'].items()},flush=True)
    summary=summary_table(cases,radii,budgets)
    # Frozen SMC-09 full cohort at zero budget provides a strong self-check.
    if max_test_puzzles==16 and set(seeds)=={71,89,113} and '0' in budgets:
        expected={
            '1/50':{'isotropic':44,'sae_inactive':32,'jacobian_optimal':0},
            '1/5':{'isotropic':43,'sae_inactive':32,'jacobian_optimal':0},
            '4/5':{'isotropic':42,'sae_inactive':30,'jacobian_optimal':0}
        }
        for radius,by_control in expected.items():
            if radius not in radii:
                continue
            for control,n in by_control.items():
                key=f'{radius}|0|{control}'
                if summary[key]['active_greater']!=n or summary[key]['inconclusive']:
                    raise AssertionError('SMC-09 zero-budget baseline mismatch: '+key)
    return {
        'schema':'SMC-10/robust-exact-real-sae-endpoint-neighborhoods-v1',
        'source_commit':lock['revision'],'checkpoint_sha256':digest,
        'roster_sha256':roster,'sae_seeds':list(seeds),'sae_steps':steps,
        'test_puzzle_ids':chosen,'arb_precision_bits':precision,
        'radii':list(radii),'l2_budget_grid':list(budgets),
        'uncertainty_model':'common L2-uncertain 48d goal token anchor and independently L2-uncertain exact unit directions; all other 80 tokens frozen',
        'interval_outer_box':'clean ±δ; each perturbed nominal h+r*u ±(δ+rδ) componentwise for common budget δ',
        'proof_scope':'strict squared-output-norm ordering for every admitted exact-real state/direction; no Torch machine rounding or retrained-SAE selection guarantee',
        'cases':cases,'summary':summary,
    }


def robust_effects_with_clean(net,anchor,goal,directions,r,da,budgets,baseline):
    """Performance variant, reusing a single clean endpoint for all radii."""
    outputs={}; errors={}
    for name in DIRECTIONS:
        try:
            perturbed=uncertain_anchor(anchor,goal,directions[name],r,da,budgets[name])
            response=endpoint_logits(net,perturbed,goal)
            interval=squared_logit_effect(response,baseline)
            if not interval.is_finite() or interval.upper()<0:
                raise ArithmeticError('nonfinite/impossible squared effect')
            outputs[name]=interval
        except (ArithmeticError,ValueError,OverflowError,ZeroDivisionError) as exc:
            errors[name]=type(exc).__name__+': '+str(exc)[:240]
    comparisons={}
    for control in CONTROLS:
        if 'sae_active' in outputs and control in outputs:
            comparisons[control]=classify_ordering(outputs['sae_active'],outputs[control])
        else:
            comparisons[control]={'proof':'inconclusive','interval_error':errors}
    return {
        'radius':str(r),'anchor_l2_budget':str(da),
        'direction_l2_budgets':{k:str(v) for k,v in budgets.items()},
        'clean_box_half_width_linf':str(da),
        'perturbed_box_half_width_linf':{k:str(da+r*v) for k,v in budgets.items()},
        'squared_effects':{name:str(iv) for name,iv in outputs.items()},
        'comparisons':comparisons,'failures':errors,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--fetch',action='store_true')
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--max-test-puzzles',type=int,default=2)
    p.add_argument('--sae-seeds',default='71')
    p.add_argument('--steps',type=int,default=300)
    p.add_argument('--radii',default='0.02,0.2')
    p.add_argument('--budgets',default='0,1e-12,1e-10,1e-8')
    p.add_argument('--precision',type=int,default=160)
    p.add_argument('--out',type=Path,default=Path('runs/smc10/pilot.json'))
    args=p.parse_args()
    report=run(args.checkpoint,args.fetch,args.max_test_puzzles,
               tuple(int(s) for s in args.sae_seeds.split(',')),args.steps,
               tuple(s.strip() for s in args.radii.split(',')),
               tuple(s.strip() for s in args.budgets.split(',')),args.precision)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({
        'out':str(args.out),'source_commit':report['source_commit'],
        'checkpoint_sha256':report['checkpoint_sha256'],
        'cases':len(report['cases']),'summary':report['summary'],
    },indent=2))


if __name__=='__main__':
    main()

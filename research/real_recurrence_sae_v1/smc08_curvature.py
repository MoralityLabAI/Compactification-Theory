"""SMC-08: actual pretrained recurrent Jacobians and exact-real curvature bounds.

Runs on the already source-locked 44K TRM-inspired Sudoku checkpoint. It
reuses SMC-07's frozen SAE training and puzzle split. The mathematical
global Hessian bound comes from exact rational Frobenius arithmetic and is
rigorous for the exact-real frozen-weight model, often vacuous numerically.

Autograd Jacobians, SVD, finite differences, and displayed curvature ratios
are FP64 numerical measurements, not certified interval enclosures.
"""
import argparse
import copy
import json
import math
from fractions import Fraction
from pathlib import Path
import sys
import torch

HERE=Path(__file__).resolve().parent
PRE=HERE.parent/'levin_sieve_v1'/'pretrained_trm'
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(PRE))
from smc07_real_sae import split_ids,fit_sae,EXPECTED_ROSTER
from validated_causal_tube import source_model_global_certificate,sqrt_upper

CHECKPOINT_SHA='3bab6189d4f25fe1e7a326a6b29ddc3141d21325f835ed1cd6409c0ac31447e1'


def frozen_weight_certificate(model):
    needed=(
        'block.tok_mix.0.weight','block.tok_mix.2.weight',
        'block.chan_mix.0.weight','block.chan_mix.2.weight',
        'block.ln1.weight','block.ln2.weight','head.weight'
    )
    sd=model.state_dict()
    if any(k not in sd for k in needed):
        raise ValueError('locked teacher has unexpected weight layout')
    frozen={k:sd[k].detach().cpu().tolist() for k in needed}
    result=source_model_global_certificate(frozen,bits=32)
    for field in ('global_jacobian_upper','global_hessian_upper'):
        value=result[field]
        assert isinstance(value,Fraction) and value>0
        result[field]=str(value)
        result[field+'_log10_approx']=math.log10(value.numerator)-math.log10(value.denominator)
    return result


def make_goal_function(model64,anchor,goal):
    """One 48-vector token substituted into an 81x48 stage-one anchor."""
    base=anchor.detach().to(dtype=torch.float64).clone()
    if base.shape!=(1,81,48) or not (0<=goal<81):
        raise ValueError('incorrect recurrent state or goal')
    def F_goal(v):
        z=base.clone()
        z[0,goal,:]=v
        for _ in range(3):
            z=model64.block(z)
        return model64.head(z)[0,goal,:]
    return F_goal,base[0,goal,:].detach().clone()


def goal_jacobian(model64,anchor,goal):
    F,x=make_goal_function(model64,anchor,goal)
    with torch.enable_grad():
        J=torch.autograd.functional.jacobian(F,x,vectorize=False,create_graph=False)
    if J.shape!=(10,48) or not bool(torch.isfinite(J).all()):
        raise ArithmeticError('invalid numerical goal Jacobian')
    singular=torch.linalg.svdvals(J)
    _,_,vh=torch.linalg.svd(J,full_matrices=False)
    top=vh[0].clone()
    if top[int(top.abs().argmax())]<0:
        top=-top
    with torch.no_grad():
        original=F(x)
    return F,x,J,top,singular,original


def sae_direction_set(sae,stage1_vector,mean,std,seed,puzzle_id,top_singular):
    with torch.no_grad():
        encoded=sae.encode((((stage1_vector-mean)/std).float())[None,:])[0]
        top=int(encoded.argmax())
        available=torch.where(encoded==0)[0]
        rng=torch.Generator().manual_seed(101031+seed*1009+puzzle_id)
        inactive=int(available[torch.randint(len(available),(1,),generator=rng)])
        dactive=(sae.decoder.weight[:,top]*std).detach().to(dtype=torch.float64)
        dinactive=(sae.decoder.weight[:,inactive]*std).detach().to(dtype=torch.float64)
        diso=torch.randn((48,),generator=rng,dtype=torch.float64)
        directions={'sae_active':dactive,'sae_inactive':dinactive,
                    'isotropic':diso,'jacobian_optimal':top_singular}
        for name,v in directions.items():
            norm=v.norm()
            if float(norm)<=1e-12:
                raise ValueError(f'zero decoder direction {name}')
            directions[name]=v/norm
        return directions,{'active_feature_id':top,'inactive_feature_id':inactive,
                           'active_value':float(encoded[top])}


def empirical_tube(F,x,J,reference,directions,global_M,radii=('0.02','0.2','0.8')):
    """Measure finite-radius residuals; global_M is the proven outer bound."""
    result={}
    with torch.no_grad():
        smax=float(torch.linalg.svdvals(J)[0])
        for name,u in directions.items():
            ju=J@u
            slope=float(ju.norm())
            exact_dyadic_norm2=sum((Fraction.from_float(float(z))**2
                                    for z in u.tolist()),Fraction(0))
            unit_upper=sqrt_upper(exact_dyadic_norm2,32)
            record={'jacobian_slope_l2':slope,
                    'fraction_of_optimal_jacobian_slope':slope/smax if smax>0 else None,
                    'dyadic_direction_norm_upper':str(unit_upper),
                    'radii':{}}
            for radius in radii:
                r=float(radius)
                moved=F(x+r*u)
                change=moved-reference
                linear=r*ju
                remainder=change-linear
                rem_l2=float(remainder.norm())
                eff=float(change.norm())
                finite_curvature=2*rem_l2/(r*r)
                # The global bound is evaluated with an exact upper for the
                # actual dyadic-normalized direction, which can exceed 1.
                rational_radius=Fraction(radius)
                true_upper=global_M*(rational_radius*unit_upper)**2/2
                if Fraction.from_float(rem_l2)>true_upper:
                    raise ArithmeticError('numerical residual exceeds exact-real global envelope')
                record['radii'][radius]={
                    'goal_logit_effect_l2_fp64':eff,
                    'first_order_effect_l2_fp64':r*slope,
                    'linear_residual_l2_fp64':rem_l2,
                    'observed_secant_curvature_2rem_over_r2':finite_curvature,
                    'relative_residual_to_observed_effect':rem_l2/max(1e-30,eff),
                    'teacher_label_flip':int(moved.argmax()!=reference.argmax()),
                    'global_rigorous_remainder_log10_approx':
                        math.log10(true_upper.numerator)-math.log10(true_upper.denominator),
                    'certified_global_remainder_bound_is_vacuous':
                        true_upper>Fraction(1000000),
                }
            result[name]=record
    return result


def synthetic_measure(model,states,inputs,sae,mu,std,split,seed=71,
                      radii=('0.02','0.2','0.8')):
    """Evaluate preloaded source or toy model. Source weights remain frozen."""
    if len(states)!=5 or len(inputs)<len(split['test']):
        raise ValueError('invalid recurrent source state roster')
    model64=copy.deepcopy(model).eval().double()
    bound=frozen_weight_certificate(model)
    M=Fraction(bound['global_hessian_upper'])
    output=[]
    for puzzle_id in split['test']:
        goal=int(torch.where(inputs[puzzle_id]==0)[0][0])
        anchor=states[1][puzzle_id:puzzle_id+1]
        F,x,J,u_opt,sv,ref=goal_jacobian(model64,anchor,goal)
        dirs,chosen=sae_direction_set(sae,states[1][puzzle_id,goal],mu,std,
                                      seed,puzzle_id,u_opt)
        empirical=empirical_tube(F,x,J,ref,dirs,M,radii)
        output.append({
            'puzzle_id':int(puzzle_id),'goal_token':goal,
            'selected_sae_features':chosen,
            'goal_jacobian_singular_values_fp64':list(map(float,sv.tolist())),
            'numerical_rank_rel1e-8':int((sv>sv[0]*1e-8).sum()),
            'directions':empirical,
        })
    return {'source':bound,'seeds':[seed],'radii':list(radii),
            'n_independent_test_puzzles':len(output),'per_puzzle':output}


def aggregate_reports(reports,radii=('0.02','0.2','0.8')):
    if not reports:
        raise ValueError('no reports')
    grouped={}
    for report in reports:
        for case in report['per_puzzle']:
            grouped.setdefault(case['puzzle_id'],[]).append(case)
    out={}
    for radius in radii:
        for control in ('sae_inactive','isotropic','jacobian_optimal'):
            causal=[];linear=[]
            for pid,rows in sorted(grouped.items()):
                diff=[];ldiff=[]
                for row in rows:
                    dirs=row['directions']
                    a=dirs['sae_active']['radii'][radius]
                    b=dirs[control]['radii'][radius]
                    diff.append(a['goal_logit_effect_l2_fp64']-b['goal_logit_effect_l2_fp64'])
                    ldiff.append(a['first_order_effect_l2_fp64']-b['first_order_effect_l2_fp64'])
                causal.append(sum(diff)/len(diff))
                linear.append(sum(ldiff)/len(ldiff))
            key=f'{radius}_active_minus_{control}'
            out[key]={'mean_finite_radius_effect_difference':sum(causal)/len(causal),
                      'mean_jacobian_predicted_difference':sum(linear)/len(linear),
                      'n_puzzle_clusters':len(causal),
                      'positive_effect_differences':sum(x>0 for x in causal),
                      'per_puzzle_effect_differences':dict(zip(map(str,sorted(grouped)),causal))}
    return out


def run(checkpoint=None,allow_fetch=False,max_cases=64,seeds=(71,89,113),
        sae_steps=300,radii=('0.02','0.2','0.8')):
    from run_checkpoint import locked_bytes,load_pretrained,prepare_roster
    lock=json.loads((PRE/'source_lock.json').read_text())
    raw,checksum=locked_bytes(lock,checkpoint_path=checkpoint,allow_fetch=allow_fetch)
    if checksum!=CHECKPOINT_SHA:
        raise ValueError('unexpected checkpoint SHA256')
    teacher=load_pretrained(lock,raw).eval()
    torch.set_num_threads(2)
    (x,target,holes),roster_hash=prepare_roster(lock)
    if roster_hash!=EXPECTED_ROSTER:
        raise ValueError('roster provenance mismatch')
    x=x[:max_cases]; target=target[:max_cases]
    split=split_ids(max_cases)
    with torch.no_grad():
        states=[z.detach() for z in teacher.reason_states(x)]
    fitted=[]
    for seed in seeds:
        sae,mu,std,fit=fit_sae(states[1][split['train']],
                             states[1][split['val']],seed=seed,steps=sae_steps)
        report=synthetic_measure(teacher,states,x,sae,mu,std,split,seed,radii)
        report['sae_training']={'seed':seed,'steps':sae_steps,
                                 'selected_best_step':fit['best_step'],
                                 'best_val_mse':fit['best_val_mse']}
        fitted.append(report)
    return {'schema':'SMC-08/source-locked-directional-curvature-v1',
            'source_commit':lock['revision'],
            'checkpoint_sha256':checksum,
            'roster_sha256':roster_hash,'n_puzzles':max_cases,
            'test_puzzle_ids':list(split['test']),
            'sae_seed_list':list(seeds),
            'radii':list(radii),
            'sae_steps':sae_steps,
            'numerical_site':'48d stage1 goal token, three FP64 recurrence steps, ten goal logits',
            'rigorous_bounds_scope':'exact-real stored binary weights, global Frobenius norm bounds; excludes FP64 roundoff',
            'numerical_derivative_scope':'FP64 autograd and finite-radius replay; not interval-certified',
            'results':fitted,
            'aggregate':aggregate_reports(fitted,radii)}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--fetch',action='store_true')
    p.add_argument('--max-cases',type=int,choices=(16,64),default=64)
    p.add_argument('--sae-steps',type=int,default=300)
    p.add_argument('--seeds',default='71,89,113')
    p.add_argument('--radii',default='0.02,0.2,0.8')
    p.add_argument('--out',type=Path,default=Path('runs/smc08/full.json'))
    args=p.parse_args()
    seeds=tuple(int(x) for x in args.seeds.split(','))
    radii=tuple(x.strip() for x in args.radii.split(','))
    if not all(Fraction(r)>0 for r in radii):
        raise ValueError('positive radii required')
    report=run(args.checkpoint,args.fetch,args.max_cases,seeds,args.sae_steps,radii)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({
        'file':str(args.out),'checkpoint_sha256':report['checkpoint_sha256'],
        'test_puzzles':report['test_puzzle_ids'],
        'global_hessian_log10':report['results'][0]['source']['global_hessian_upper_log10_approx'],
        'aggregate':{k:{f:v[f] for f in ('mean_finite_radius_effect_difference',
                                         'mean_jacobian_predicted_difference',
                                         'positive_effect_differences')}
                     for k,v in report['aggregate'].items()},
    },indent=2))


if __name__=='__main__':
    main()

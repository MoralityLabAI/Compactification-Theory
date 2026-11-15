"""SMC-08B: validated Arb *local* SAE-direction Taylor residuals.

Reuses LSV-17's rigorously bounded weight spectral norms, exact-binary
stored stage-one anchors, interval forward/reverse calculus, and per-layer
reachable-tube curvature. An Arb enclosure bounds the exact-real
winner-minus-runner margin displacement for a *specific* SAE direction.

Torch FP64 forward measurements are diagnostics, not interval proofs.
This is a directional finite-difference remainder bound, not a new global
operator-norm Hessian theorem.
"""
import argparse
import copy
import json
import math
from fractions import Fraction as Q
from pathlib import Path
import sys
import torch

HERE=Path(__file__).resolve().parent
PRE=HERE.parent/'levin_sieve_v1'/'pretrained_trm'
ARB_DIR=HERE.parent/'levin_sieve_v1'/'goal_taylor_v1'
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(PRE))
sys.path.insert(0,str(ARB_DIR))
from smc07_real_sae import split_ids,fit_sae,EXPECTED_ROSTER
from validated_causal_tube import sqrt_upper
from run_checkpoint import locked_bytes,load_pretrained,prepare_roster
from flint import arb,ctx
from arb_engine import TaylorNet,up,aq,endpoint


def log10_upper(interval_upper):
    q=endpoint(interval_upper.upper())
    if q<=0:
        return None
    return math.log10(q.numerator)-math.log10(q.denominator)


def direction_from_sae(sae,state,mu,std):
    with torch.no_grad():
        activation=sae.encode(((state-mu)/std)[None,:])[0]
        atom=int(activation.argmax())
        v=(sae.decoder.weight[:,atom]*std).detach().double()
        if float(v.norm())<=1e-12:
            raise ValueError('zero learned active decoder atom')
        direction=v/v.norm()
    exact=sum((Q.from_float(float(z))**2 for z in direction.tolist()),Q(0))
    norm_hi=sqrt_upper(exact,32)
    return direction,atom,norm_hi


def certified_radius(net,tape,gradient,adj,goal,direction,dir_norm_upper,radius):
    r=Q(radius)
    if r<=0:raise ValueError('radius must be positive')
    rows=gradient.tolist()
    slope=sum((rows[goal][j]*arb(float(direction[j]))
               for j in range(48)),arb(0))
    linear=aq(r)*slope
    tube=net.transport(tape,r*dir_norm_upper,dimensions=1)
    remainder=up(sum((coef*t['defect'] for coef,t in zip(adj['q'],tube)),arb(0)))
    lower=(linear-remainder).lower()
    upper=(linear+remainder).upper()
    if not lower.is_finite() or not upper.is_finite():
        raise ArithmeticError('nonfinite Arb residual interval')
    sign='positive' if lower>0 else 'negative' if upper<0 else 'inconclusive'
    return {
        'radius':str(r),
        'direction_l2_upper_exact':str(dir_norm_upper),
        'linear_signed_margin_mid_fp64':float(linear.mid()),
        'linear_signed_margin_arb_interval':str(linear),
        'rigorous_remainder_upper_arb':str(remainder),
        'rigorous_remainder_log10':log10_upper(remainder),
        'rigorous_margin_displacement_interval':[str(lower),str(upper)],
        'certified_displacement_sign':sign,
        'ln_epsilon_floor_used':any(t['fallback'] for t in tube),
        'largest_intermediate_radius_log10':max(
            (log10_upper(t['R']) for t in tube if t['R']>0),default=None),
        'certificate_scope':'exact-real, exact stored checkpoint FP32 binary weights and stage-one anchor; excludes Torch arithmetic roundoff'
    }


def actual_margin_fp64(model64,stage,goal,winner,runner,direction,r):
    with torch.no_grad():
        z=stage.double().clone()
        z[0,goal,:]+=float(r)*direction.double()
        for _ in range(3):
            z=model64.block(z)
        return float((model64.head(z)[0,goal,winner]-
                      model64.head(z)[0,goal,runner]))


def run(checkpoint=None,allow_fetch=False,max_test_puzzles=2,seed=71,
        steps=300,radii=('1e-10','1e-8','1e-6','1e-4','0.2'),
        precision=160):
    if not 1<=max_test_puzzles<=16 or precision<80:
        raise ValueError('bounded test puzzle count and Arb precision>=80 required')
    ctx.prec=precision
    torch.set_num_threads(1)
    lock=json.loads((PRE/'source_lock.json').read_text())
    raw,digest=locked_bytes(lock,checkpoint,allow_fetch)
    model=load_pretrained(lock,raw).eval()
    (inputs,targets,_),roster=prepare_roster(lock)
    if roster!=EXPECTED_ROSTER:
        raise ValueError('roster mismatch')
    split=split_ids(64)
    with torch.no_grad():
        stage_one=model.reason_states(inputs)[1].detach()
    sae,mu,std,fit=fit_sae(stage_one[split['train']],
                            stage_one[split['val']],seed=seed,steps=steps)
    model64=copy.deepcopy(model).to(torch.float64).eval()
    net=TaylorNet(model)
    cases=[]
    for i in split['test'][:max_test_puzzles]:
        goal=int(torch.where(inputs[i]==0)[0][0])
        stage=stage_one[i]
        final,tape=net.forward(stage,steps=3)
        logits=net.logits(final,goal)
        ranks=sorted(range(10),key=lambda j:float(logits[j].mid()),reverse=True)
        winner,runner=ranks[:2]
        if not logits[winner].lower()>logits[runner].upper():
            raise ArithmeticError('winner-runner argmax cannot be verified')
        adj=net.margin_adjoint(tape,goal,winner,runner)
        direction,feature,nu=direction_from_sae(sae,stage[goal],mu,std)
        base=actual_margin_fp64(model64,stage[None,:],goal,winner,runner,direction,0)
        records=[]
        for r in radii:
            cert=certified_radius(net,tape,adj['gradient'],adj,goal,direction,nu,r)
            measured=actual_margin_fp64(model64,stage[None,:],goal,winner,runner,direction,Q(r))
            cert['numeric_fp64_signed_margin_delta']=measured-base
            cert['numeric_fp64_clean_margin']=base
            records.append(cert)
        cases.append({
            'puzzle_id':i,'goal_token':goal,'winner':winner,'runner':runner,
            'sae_active_feature':feature,
            'clean_margin_arb_lower':str((logits[winner]-logits[runner]).lower()),
            'directional_certificates':records,
        })
        print('CERTIFIED_CASE',i,
              [(c['radius'],c['certified_displacement_sign'],c['rigorous_remainder_log10'])
               for c in records],flush=True)
    return {
        'schema':'SMC-08B/arb-directional-taylor-remainder-v1',
        'source_commit':lock['revision'],'checkpoint_sha256':digest,
        'roster_sha256':roster,'sae_seed':seed,'sae_steps':steps,
        'arb_precision_bits':precision,'test_puzzle_ids':[c['puzzle_id'] for c in cases],
        'radii':list(radii),'readout':'fixed clean winner vs runner at first blank goal token',
        'spectral_ldl_certificates':str(net.spectral),
        'cases':cases,
        'summary':{r:{'count':len(cases),
                     'sign_certified':sum(
                         c['certified_displacement_sign']!='inconclusive'
                         for case in cases for c in case['directional_certificates'] if c['radius']==str(Q(r))),
                     'eps_floor_used':sum(
                         c['ln_epsilon_floor_used']
                         for case in cases for c in case['directional_certificates'] if c['radius']==str(Q(r)))}
                  for r in radii},
        'claim_scope':'Arb enclosure for exact-real frozen-weight local directional margin; no certified neural semantic interpretation'
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--fetch',action='store_true')
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--max-test-puzzles',type=int,default=2)
    p.add_argument('--seed',type=int,default=71)
    p.add_argument('--steps',type=int,default=300)
    p.add_argument('--precision',type=int,default=160)
    p.add_argument('--radii',default='1e-10,1e-8,1e-6,1e-4,0.2')
    p.add_argument('--out',type=Path,default=Path('runs/smc08/local-arb.json'))
    a=p.parse_args()
    result=run(a.checkpoint,a.fetch,a.max_test_puzzles,a.seed,a.steps,
               tuple(x.strip() for x in a.radii.split(',')),a.precision)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'out':str(a.out),'checkpoint_sha256':result['checkpoint_sha256'],
                      'summary':result['summary']},indent=2))


if __name__=='__main__':
    main()

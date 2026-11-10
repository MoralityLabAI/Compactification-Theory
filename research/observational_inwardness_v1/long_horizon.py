"""OIC-14 out-of-window recurrence test from saved trained parameters.

Uses the archived weights only; no retraining. Computes first incorrect
always-probe event across all initial phases, then tests an analytic
angular-drift sufficient condition. Does not certify floating-point arithmetic.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
from statistics import median

HERE=Path(__file__).resolve().parent
DATA=HERE/'results'/'oic14_full.json'


def sign_event(n,architecture,fit,phase,step):
    w=fit['learned_values']['gain'][0]
    b=fit['learned_values']['bias'][0]
    theta0=2*math.pi*phase/n
    if architecture.startswith('orthogonal'):
        omega=fit['learned_values']['omega'][0]
        hcos=math.cos(theta0 + step*omega)
    else:
        matrix=fit['transition_matrix']
        v=(math.cos(theta0),math.sin(theta0))
        for _ in range(step):
            v=(matrix[0][0]*v[0]+matrix[0][1]*v[1],
               matrix[1][0]*v[0]+matrix[1][1]*v[1])
        hcos=v[0]
    return (w*hcos+b)>=0


def first_wrong(n,architecture,fit,family,limit=600):
    for t in range(limit):
        for phase in range(n):
            actual=(phase+(t if family=='rotating' else 0))%n==0
            if sign_event(n,architecture,fit,phase,t)!=actual:
                return t,phase
    return None


def angular_certificate(n,fit,family):
    """Conservative continuous-angle drift criterion, conditional on readout lattice correctness."""
    true_omega=0.0 if family=='fixed' else 2*math.pi/n
    omega=fit['learned_values']['omega'][0]
    w=fit['learned_values']['gain'][0]
    b=fit['learned_values']['bias'][0]
    signed=[w*math.cos(2*math.pi*k/n)+b for k in range(n)]
    right=all((v>=0)==(k==0) for k,v in enumerate(signed))
    abs_margin=min(abs(v) for v in signed)
    delta=abs(omega-true_omega)
    if delta==0:
        return {'valid_lattice':right,'phase_drift_per_step':0,'sufficient_time':None,'margin':abs_margin}
    t_bound=abs_margin/(abs(w)*delta) if right and w!=0 else 0
    return {'valid_lattice':right,'phase_drift_per_step':delta,'sufficient_time':t_bound,'margin':abs_margin}


def run():
    data=json.loads(DATA.read_text())
    rows=[]
    for item in data['runs']:
        n=item['n'];arch=item['architecture'];split=item['training_split'];seed=item['seed']
        for family in ('fixed','rotating'):
            fit=item['training'][family]
            first=first_wrong(n,arch,fit,family)
            cert=angular_certificate(n,fit,family) if arch.startswith('orthogonal') else None
            rows.append({'n':n,'architecture':arch,'training_split':split,'seed':seed,
                         'family':family,'first_wrong_event':None if first is None else first[0],
                         'first_wrong_phase':None if first is None else first[1],
                         'tested_transitions':600,'analytic_angular_drift_certificate':cert})
    return {'status':'empirical_long_horizon_replay_from_saved_parameters',
            'rounding_scope':'float64 readout reconstruction from 8-decimal stored weights',
            'rows':rows}


if __name__=='__main__':
    out=run()
    path=HERE/'results'/'oic14_long_horizon.json'
    path.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print('rows',len(out['rows']))
    for arch in ('orthogonal_local','orthogonal_grid','unconstrained'):
      for split in ('full','heldout'):
        for n in (4,5,6):
          vals=[x['first_wrong_event'] if x['first_wrong_event'] is not None else 600
                for x in out['rows'] if x['n']==n and x['architecture']==arch and x['training_split']==split]
          print(arch,split,'n=',n,'min=',min(vals),'median=',median(vals),'max=',max(vals))

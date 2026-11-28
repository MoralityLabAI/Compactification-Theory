"""OIC-14: finite-group projection of a learned SO(2) transition.

The nearest Z_n shift is determined from learned weights and known n,
without consulting the true family in the projection. Exact infinite-horizon
fidelity is checked *after* projection against the held-out oracle labels.
"""
from __future__ import annotations
import json
import math
from pathlib import Path

HERE=Path(__file__).resolve().parent


def nearest_shift(n:int, omega:float) -> tuple[int,float]:
    if n<2:raise ValueError('n>=2 required')
    step=2*math.pi/n
    q=round(omega/step)
    projected=q*step
    angular_error=abs(((omega-projected+math.pi)%(2*math.pi))-math.pi)
    return q%n,angular_error


def detector_lattice(n:int,gain:float,bias:float) -> tuple[bool,...]:
    return tuple(gain*math.cos(2*math.pi*z/n)+bias>=0 for z in range(n))


def gate_correct(n:int,gain:float,bias:float) -> bool:
    return detector_lattice(n,gain,bias)==(True,)+(False,)*(n-1)


def projected_law(n:int, omega:float, gain:float, bias:float, family:str) -> dict:
    shift,delta=nearest_shift(n,omega)
    actual=int(family=='rotating')
    return {
      'selected_shift':shift,
      'target_shift':actual,
      'angle_distance_to_group_step':delta,
      'shift_correct':shift==actual,
      'readout_lattice_correct':gate_correct(n,gain,bias),
      'all_horizon_oracle_exact':shift==actual and gate_correct(n,gain,bias),
    }


def annotate(payload:dict) -> dict:
    rows=[]
    for item in payload['runs']:
      if not item['architecture'].startswith('orthogonal'):continue
      for family in ('fixed','rotating'):
        fit=item['training'][family]
        vals=fit['learned_values']
        claim=projected_law(item['n'],vals['omega'][0],vals['gain'][0],vals['bias'][0],family)
        rows.append({'n':item['n'],'split':item['training_split'],'architecture':item['architecture'],
                     'seed':item['seed'],'family':family,**claim})
    return {'scope':'posttrain_nearest_Zn_step_projection_on_known_phase_group',
            'rows':rows}


if __name__=='__main__':
    p=json.loads((HERE/'results'/'oic14_full.json').read_text())
    result=annotate(p)
    (HERE/'results'/'oic14_projection.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    for arch in ('orthogonal_grid','orthogonal_local'):
      for split in ('full','heldout'):
        rows=[x for x in result['rows'] if x['architecture']==arch and x['split']==split]
        print(arch,split,'n rows',len(rows),'all_horizon_exact',sum(x['all_horizon_oracle_exact'] for x in rows),
              'shift_correct',sum(x['shift_correct'] for x in rows),'lattice_correct',sum(x['readout_lattice_correct'] for x in rows))

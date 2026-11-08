import json
from fractions import Fraction as F
from critic import values,norm,residual_bound,sweeps,transport
p0=[[F(4,5),F(1,10)],[F(1,5),F(3,5)]];p1=[[F(3,5),F(1,4)],[F(1,20),F(1,2)]]
r0=[F(1,10),F(2,5)];r1=[F(1,5),F(7,20)];g=F(9,10)
v0=values(p0,r0,g);rows=[]
for policy_shift,p in [(False,p0),(True,p1)]:
    for reward_shift,r in [(False,r0),(True,r1)]:
        target=values(p,r,g);reset=[F(0),F(0)]
        error=lambda v:float(norm([a-b for a,b in zip(v,target)]))
        rows.append({'policy_shift':policy_shift,'reward_shift':reward_shift,
                     'reset_initial_error':error(reset),'inherited_initial_error':error(v0),
                     'reset_after_four_sweeps_error':error(sweeps(p,r,g,reset,4)),
                     'inherited_after_four_sweeps_error':error(sweeps(p,r,g,v0,4)),
                     'inherited_residual_error_upper_bound':float(residual_bound(p,r,g,v0))})
a,b=transport(p0,r0,p1,r1,g)
print(json.dumps({'status':'exact_finite_state_reference_only','model_training':False,
                  'histogram_progress_example':{'true_expected_progress':.25,'progress_of_mean_count':1/3},'exact_transport_identity':a==b,'matched_bellman_sweep_budget':4,'factorial_cells':rows,
                  'negative_control':{'old_reward':1,'new_reward':0,'inherited_value':10,'reset_value':0,'new_true_value':0}},indent=2))

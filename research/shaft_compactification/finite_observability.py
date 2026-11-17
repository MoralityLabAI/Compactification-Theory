"""SHAFT-COMP-12: exact finite shift-register observability reference.

Six binary state coordinates are shifted toward a one-bit observation.
All horizon ranks and Gramian entries are integer-exact, with an exhaustive
64-state check of observation classes and indistinguishable partners.
"""
import json
from itertools import product

DIM=6
STATES=tuple(product((0,1),repeat=DIM))
HORIZONS=tuple(range(9))

def shift(s):
    assert len(s)==DIM
    return tuple(s[1:])+(0,)

def observe(s):
    return s[0]

def trace(s,horizon):
    result=[]
    for _ in range(horizon+1):
        result.append(observe(s))
        s=shift(s)
    return tuple(result)

def observability_matrix(horizon):
    return tuple(tuple(int(k==j) for j in range(DIM)) if k<DIM
                 else (0,)*DIM for k in range(horizon+1))

def gramian(horizon):
    matrix=observability_matrix(horizon)
    return tuple(tuple(sum(row[i]*row[j] for row in matrix)
                       for j in range(DIM)) for i in range(DIM))

def reference():
    records=[]
    for horizon in HORIZONS:
        M=observability_matrix(horizon)
        G=gramian(horizon)
        rank=sum(G[i][i]>0 for i in range(DIM))
        classes={trace(s,horizon) for s in STATES}
        partner_counts={sum(trace(s,horizon)==label for s in STATES)
                        for label in classes}
        assert partner_counts=={2**(DIM-rank)}
        assert all(G[i][j]==0 for i in range(DIM)
                   for j in range(DIM) if j!=i)
        assert rank==min(horizon+1,DIM)
        assert len(classes)==2**rank
        for s in STATES:
            energy=sum(s[i]*G[i][j]*s[j] for i in range(DIM) for j in range(DIM))
            observations=sum(bit*bit for bit in trace(s,horizon))
            assert energy==observations
        records.append({
            'horizon':horizon,
            'exact_observability_rank':rank,
            'kernel_dimension':DIM-rank,
            'gramian_diagonal':[G[i][i] for i in range(DIM)],
            'behavior_classes':len(classes),
            'compatible_states_per_signature':next(iter(partner_counts)),
        })
    start=(0,0,0,0,0,0)
    delayed=(0,0,0,0,0,1)
    assert all(trace(start,L)==trace(delayed,L) for L in range(5))
    assert trace(start,5)!=trace(delayed,5)
    return {
        'evidence':'exhaustive_integer_six_bit_shift_register',
        'states':len(STATES),
        'state_dimensions':DIM,
        'measurement_dimension':1,
        'results':records,
        'five_step_delay_witness':{
            'initial_state':list(start),
            'other_state':list(delayed),
            'indistinguishable_through_horizon':4,
            'distinguished_at_horizon':5,
        },
        'check':'exact integer Gramian quadratic form equals sum of squared observations across all 64 states and nine horizons',
    }

if __name__=='__main__':
    print(json.dumps(reference(),sort_keys=True,indent=2))

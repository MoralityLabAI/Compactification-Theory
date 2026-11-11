"""Exact 0-1 integer formulation for nonadaptive n=6 single-flip correction.

Every time has exactly one addressed marker; each pair of distinct shifts must
have Hamming distance >=3 for every pair of unknown initial phases.
A certified INFEASIBLE result proves all fixed schedules fail at that horizon.
"""
from __future__ import annotations
import argparse
import json
import time
import numpy as np
from scipy.optimize import Bounds,LinearConstraint,milp
from scipy.sparse import lil_matrix
N=6


def distance(word, d, z, d2, z2):
    return sum(int((z+t*d)%N==m)^int((z2+t*d2)%N==m)
               for t,m in enumerate(word))


def minimum_cross(word):
    return min(distance(word,d,z,d2,z2) for d in range(N) for d2 in range(d+1,N)
               for z in range(N) for z2 in range(N))


def model(horizon:int,target:int=3,seconds:float=45):
    constraints=[]; lower=[];upper=[]
    for t in range(horizon):
        row={t*N+m:1 for m in range(N)}
        constraints.append(row);lower.append(1);upper.append(1)
    for d in range(N):
        for e in range(d+1,N):
            for z in range(N):
                for z2 in range(N):
                    row={}
                    for t in range(horizon):
                        a=(z+t*d)%N;b=(z2+t*e)%N
                        if a!=b:
                            row[t*N+a]=row.get(t*N+a,0)+1
                            row[t*N+b]=row.get(t*N+b,0)+1
                    constraints.append(row);lower.append(target);upper.append(np.inf)
    A=lil_matrix((len(constraints),horizon*N),dtype=np.float64)
    for i,row in enumerate(constraints):
        for col,v in row.items():A[i,col]=v
    lb=np.zeros(horizon*N);ub=np.ones(horizon*N)
    lb[0]=1;ub[0]=1 # phase-translation symmetry: first sensor at marker 0
    result=milp(c=np.zeros(horizon*N), integrality=np.ones(horizon*N),
                bounds=Bounds(lb,ub),
                constraints=LinearConstraint(A.tocsc(),np.array(lower),np.array(upper)),
                options={'time_limit':seconds,'presolve':True,'mip_rel_gap':0})
    out={'n':N,'horizon':horizon,'target_cross_shift_distance':target,
         'fixed_schedule_first_address':0,'status':int(result.status),
         'solver_message':str(result.message), 'node_count':getattr(result,'mip_node_count',None),
         'dual_bound':getattr(result,'mip_dual_bound',None)}
    if result.x is not None:
        word=[int(np.argmax(result.x[t*N:(t+1)*N])) for t in range(horizon)]
        out.update(word=word,exact_min_distance=minimum_cross(word),
                   valid=minimum_cross(word)>=target)
    return out

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('horizon',type=int);parser.add_argument('--seconds',type=int,default=45)
    opts=parser.parse_args();started=time.monotonic();o=model(opts.horizon,seconds=opts.seconds)
    o['wall_seconds']=round(time.monotonic()-started,3)
    print(json.dumps(o,sort_keys=True,indent=2))

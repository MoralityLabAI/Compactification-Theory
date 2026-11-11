"""OIC-30 exact count-per-residue ILP for fixed schedules over cyclic n=7.

For each t mod n, choose counts of sensor addresses summing to the number
of such time positions. Any integer assignment can be expanded into a
specific schedule, and its cross-family code distances are exactly linear
in the occupancy counts, because the model is periodic in t mod n.
"""
from __future__ import annotations
import argparse, json, time
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import lil_matrix

N=7

def reduced_model(Q:int=21,seconds:float=60.0,fix_gauge:bool=True, markers=None):
  n=N
  row_counts=[len(range(r,Q,n)) for r in range(n)]
  neq=n*n
  rows=[]; lo=[]; hi=[]
  for t in range(n):
    v={t*n+m:1 for m in range(n)}
    rows.append(v);lo.append(row_counts[t]);hi.append(row_counts[t])
  for d in range(n):
    for d2 in range(d+1,n):
      for z in range(n):
        for z2 in range(n):
          row={}
          for t in range(n):
            # line L(d,z), line L(d2,z2)
            a=(z+t*d)%n; b=(z2+t*d2)%n
            if a!=b:
              row[t*n+a]=row.get(t*n+a,0)+1
              row[t*n+b]=row.get(t*n+b,0)+1
          rows.append(row);lo.append(3);hi.append(np.inf)
  A=lil_matrix((len(rows),neq),dtype=np.float64)
  for i,row in enumerate(rows):
    for j,v in row.items():A[i,j]=v
  lb=np.zeros(neq,dtype=np.float64)
  ub=np.array([row_counts[t] for t in range(n) for m in range(n)],dtype=np.float64)
  if markers is not None:
    allowed=set(markers)
    if not allowed or any(m not in range(n) for m in allowed):raise ValueError('invalid marker subset')
    for t in range(n):
      for m in range(n):
        if m not in allowed:ub[t*n+m]=0
  if fix_gauge and (markers is None or 0 in markers):
    # translation gauge fixing: some t=0 address can be moved to zero
    lb[0]=1
  start=time.monotonic()
  result=milp(np.zeros(neq),integrality=np.ones(neq),bounds=Bounds(lb,ub),
      constraints=LinearConstraint(A.tocsc(),np.array(lo),np.array(hi)),
      options={'time_limit':seconds,'presolve':True,'mip_rel_gap':0.0})
  elapsed=time.monotonic()-start
  out={'n':n,'Q':Q,'row_counts':row_counts,'status':int(result.status),
       'message':str(result.message),'elapsed':round(elapsed,3),
       'nodes':getattr(result,'mip_node_count',None),'gap':getattr(result,'mip_gap',None)}
  if result.x is not None:
    counts=np.rint(result.x).astype(int).reshape(n,n)
    word=[None]*Q
    for t in range(n):
      addresses=[m for m in range(n) for _ in range(counts[t,m])]
      if len(addresses)!=row_counts[t]:raise AssertionError('invalid row sum')
      for i,m in zip(range(t,Q,n),addresses):word[i]=m
    out['word']=word
    out['counts']=counts.tolist()
    out['minimum_distance']=min(
      sum(((z+t*d)%n==m)!=((z2+t*d2)%n==m) for t,m in enumerate(word))
      for d in range(n) for d2 in range(d+1,n) for z in range(n) for z2 in range(n))
    if out['minimum_distance']<3:raise AssertionError('integral solution violates code distance')
  return out

if __name__=='__main__':
  p=argparse.ArgumentParser();p.add_argument('--q',type=int,default=21);p.add_argument('--seconds',type=float,default=60.0);p.add_argument('--no-gauge',action='store_true');p.add_argument('--markers',type=str,default='0123456');args=p.parse_args()
  print(json.dumps(reduced_model(args.q,args.seconds,not args.no_gauge,[int(x) for x in args.markers]),indent=2))

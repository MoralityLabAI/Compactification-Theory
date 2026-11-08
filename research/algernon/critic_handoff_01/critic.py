"""Exact rational finite-state references for critic transport; no learned model."""
from fractions import Fraction as F
from typing import Sequence

def _validate(p,r,gamma):
    n=len(r)
    if n==0 or len(p)!=n or any(len(row)!=n for row in p):raise ValueError('Square nonempty kernel required')
    if not F(0)<=gamma<F(1):raise ValueError('Require 0 <= gamma < 1')
    if any(x<0 for row in p for x in row) or any(sum(row)>1 for row in p):
        raise ValueError('Require a nonnegative substochastic kernel')

def mv(p,v):return [sum(a*b for a,b in zip(row,v)) for row in p]

def solve(a,b):
    n=len(b)
    if n==0 or len(a)!=n or any(len(row)!=n for row in a):raise ValueError('Dimensions')
    aug=[[F(x) for x in row]+[F(y)] for row,y in zip(a,b)]
    for j in range(n):
        k=next((k for k in range(j,n) if aug[k][j]),None)
        if k is None:raise ValueError('Singular matrix')
        aug[j],aug[k]=aug[k],aug[j];z=aug[j][j];aug[j]=[x/z for x in aug[j]]
        for k in range(n):
            if k!=j:
                z=aug[k][j];aug[k]=[x-z*y for x,y in zip(aug[k],aug[j])]
    return [row[-1] for row in aug]

def values(p,r,gamma):
    _validate(p,r,gamma);n=len(r)
    return solve([[F(i==j)-gamma*p[i][j] for j in range(n)] for i in range(n)],r)

def bellman(p,r,gamma,v):
    _validate(p,r,gamma)
    if len(v)!=len(r):raise ValueError('Value dimensions')
    return [a+gamma*b for a,b in zip(r,mv(p,v))]

def norm(v):return max(map(abs,v),default=F(0))

def residual_bound(p,r,gamma,v):
    return norm([a-b for a,b in zip(bellman(p,r,gamma,v),v)])/(1-gamma)

def transport(old_p,old_r,new_p,new_r,gamma):
    """Return the exact drift and resolvent-predicted drift separately."""
    vo=values(old_p,old_r,gamma);vn=values(new_p,new_r,gamma);n=len(vo)
    if len(vn)!=n:raise ValueError('State spaces must match')
    pd=[[b-a for a,b in zip(ra,rb)] for ra,rb in zip(old_p,new_p)]
    rhs=[b-a+gamma*d for a,b,d in zip(old_r,new_r,mv(pd,vo))]
    predicted=solve([[F(i==j)-gamma*new_p[i][j] for j in range(n)] for i in range(n)],rhs)
    return [b-a for a,b in zip(vo,vn)],predicted

def sweeps(p,r,gamma,v,n):
    if type(n) is not int or n<0:raise ValueError('Nonnegative sweep count required')
    for _ in range(n):v=bellman(p,r,gamma,v)
    return v

def expected_terminal_spr(success_probability, passed_pmf, divisor):
    """Exact reward readout from a terminal histogram; gamma=1, bounded episode."""
    s=F(success_probability);d=F(divisor)
    if not 0<=s<=1 or d<=0 or not passed_pmf:raise ValueError('Invalid probability/divisor')
    if any(type(k) is not int or k<0 or F(p)<0 for k,p in passed_pmf.items()):
        raise ValueError('Invalid histogram')
    if sum(F(p) for p in passed_pmf.values())!=1:raise ValueError('Histogram must sum to one')
    return s+sum(F(p)*F(k)/(k+d) for k,p in passed_pmf.items())

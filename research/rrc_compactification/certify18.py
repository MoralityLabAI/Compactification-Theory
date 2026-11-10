"""RRC-18: exact symbolic finite-degree semantics of the four graph wirings.

All coefficients are polynomials over Z[x_1,...,x_T]. A monomial is encoded as
base-32 digits containing exponents (up to 24 at T=12). Python integers are
exact, so identical serialized polynomial vectors give deterministic
proof-by-normalization of equality on *all* integer unit series through T.

This module does not prove equality at unbounded degree.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json

try:
    from . import graph11 as g, topology16 as t, quotient17 as q
except ImportError:
    import graph11 as g
    import topology16 as t
    import quotient17 as q

T = g.DEGREE
BASE_BITS = 5
MAX_WEIGHT = 2*T
TOPOLOGIES = t.TOPOLOGIES

# Sparse polynomials are dict[encoded_monomial, integer_coefficient].
def poly_add(a: dict[int,int], b: dict[int,int], sign=1) -> dict[int,int]:
    z=a.copy()
    for k,v in b.items():
        n=z.get(k,0)+sign*v
        if n: z[k]=n
        elif k in z: del z[k]
    return z


def poly_mul(a:dict[int,int], b:dict[int,int]) -> dict[int,int]:
    z={}
    for k,x in a.items():
        for h,y in b.items():
            key=k+h
            z[key]=z.get(key,0)+x*y
    return {k:v for k,v in z.items() if v}


def poly_inv(a:list[dict[int,int]]) -> list[dict[int,int]]:
    if a[0]!={0:1}:raise ValueError('constant unit coefficient required')
    r=[{0:1}]
    for n in range(1,len(a)):
        cur={}
        for j in range(1,n+1):
            cur=poly_add(cur,poly_mul(a[j],r[n-j]),-1)
        r.append(cur)
    return r


def unary(a:list[dict[int,int]], op:str) -> list[dict[int,int]]:
    if op=='I':return a
    if op=='N':return [({k:-v for k,v in p.items()} if n%2 else p) for n,p in enumerate(a)]
    if op=='S':return [a[n//2] if n%2==0 else {} for n in range(len(a))]
    if op=='R':return poly_inv(a)
    raise ValueError(op)


def convolve(a:list[dict[int,int]],b:list[dict[int,int]])->list[dict[int,int]]:
    return [sum_poly(poly_mul(a[k],b[n-k]) for k in range(n+1)) for n in range(len(a))]


def sum_poly(items):
    out={}
    for p in items:out=poly_add(out,p)
    return out


def execute(base:list[dict[int,int]], graph:tuple[str,str,str], topology:str):
    if graph not in g.GRAPHS or topology not in TOPOLOGIES:raise ValueError('invalid program')
    a,b,c=graph
    left=unary(base,a)
    right=unary(left if topology=='cascade' else base,b)
    if topology=='additive':
        mixed=[{0:1}]+[poly_add(x,y) for x,y in zip(left[1:],right[1:])]
    elif topology=='hadamard':
        mixed=[{0:1}]+[poly_mul(x,y) for x,y in zip(left[1:],right[1:])]
    else:mixed=convolve(left,right)
    return unary(mixed,c)


def variables(degree=T):
    if not 1<=degree<=12:raise ValueError('degree 1..12')
    return [{0:1}]+[{1<<(BASE_BITS*(n-1)):1} for n in range(1,degree+1)]


def polynomial_signature(series):
    # Serialized canonical polynomial coefficients in each truncated q degree.
    return tuple(tuple(sorted(p.items())) for p in series)


def build_classes(topology,degree=T):
    x=variables(degree)
    groups={}
    total_terms=0
    for graph in g.GRAPHS:
        outcome=execute(x,graph,topology)
        total_terms+=sum(len(v) for v in outcome)
        groups.setdefault(polynomial_signature(outcome),[]).append(graph)
    return tuple(tuple(items) for items in groups.values()),total_terms


def certificate():
    report={'schema':'rrc18.symbolic-class-certificate.v1','degree':T,
            'polynomial_ring':'Z[x_1,...,x_12]; degree-T formal-series truncation',
            'monomial_encoding':{'bits_per_exponent':BASE_BITS,'max_exponent':MAX_WEIGHT},
            'topologies':{},'claim_scope':'Exact equality of integer coefficient polynomials through q^12 for every input in 1+q Z[[q]]. No infinite-degree equivalence claim.'}
    for name in TOPOLOGIES:
        classes,term_ops=build_classes(name)
        # Each class is a list of graph code triples, sorted by canonical graph order.
        group_list=[[list(p) for p in cls] for cls in classes]
        probe=q.build_quotient(name)
        probe_groups={frozenset(cls) for cls in probe.classes}
        exact_groups={frozenset(cls) for cls in classes}
        report['topologies'][name]={
            'class_count':len(classes),
            'class_sizes':[len(c) for c in classes],
            'exact_classes':group_list,
            'total_normal_form_monomial_entries':term_ops,
            'matches_rrc17_probe_partition':probe_groups==exact_groups,
        }
        if probe_groups!=exact_groups:
            raise AssertionError(f'RRC-17 probe grouping differs from exact finite semantics for {name}')
    return report


if __name__=='__main__':
    result=certificate()
    print([(k,v['class_count'],v['total_normal_form_monomial_entries']) for k,v in result['topologies'].items()])

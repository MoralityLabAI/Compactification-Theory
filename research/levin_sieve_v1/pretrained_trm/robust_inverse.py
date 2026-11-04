"""LSV-14: exact rational inverse perturbation certificate for interval minors.

Given a frozen 4x4 rational reference C and rigorous real interval matrix B,
prove invertibility of every matrix M inside B by the Neumann series:
M = C(I + C^-1(M-C)), ||C^-1(M-C)||_inf < 1.
The reference C may be chosen heuristically; its numerical provenance is not
needed for this algebraic proof, only exact invertibility and interval inclusion.
"""
from __future__ import annotations
from fractions import Fraction
from verify_serialized_minors import parse_ball


def inverse4(a):
    if len(a)!=4 or any(len(row)!=4 for row in a):
        raise ValueError('require 4x4 rational matrix')
    z=[[Fraction(x) for x in row]+[Fraction(int(i==j)) for j in range(4)]
       for i,row in enumerate(a)]
    for k in range(4):
        pivot=next((j for j in range(k,4) if z[j][k]),None)
        if pivot is None:
            raise ValueError('singular rational reference')
        z[k],z[pivot]=z[pivot],z[k]
        p=z[k][k]
        z[k]=[v/p for v in z[k]]
        for i in range(4):
            if i!=k:
                factor=z[i][k]
                z[i]=[x-factor*y for x,y in zip(z[i],z[k])]
    return [row[4:] for row in z]


def inverse_perturbation_witness(entry_strings,ref_hex):
    """Exact Neumann norm upper bound for all Jacobians in interval matrix."""
    if len(entry_strings)!=4 or any(len(x)!=4 for x in entry_strings):
        raise ValueError('wrong interval matrix shape')
    if len(ref_hex)!=4 or any(len(x)!=4 for x in ref_hex):
        raise ValueError('wrong rational reference shape')
    a=[[Fraction.from_float(float.fromhex(s)) for s in row] for row in ref_hex]
    inv=inverse4(a)
    box=[[parse_ball(s) for s in row] for row in entry_strings]
    deviations=[sum(max(abs(lo-c),abs(hi-c)) for (lo,hi),c in zip(row,old))
                for row,old in zip(box,a)]
    q=max(sum(abs(inv[i][j])*deviations[j] for j in range(4)) for i in range(4))
    return {
        'rank_four_by_neumann':q<1,
        'strict_neumann_norm_upper_rational':str(q),
        'one_minus_neumann_norm_lower_rational':str(1-q),
    }

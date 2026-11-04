"""Hodge certificate for additive evidence updates on a filled Boolean cube.

Requires NumPy. Oriented boundaries B=boundary_1 (vertices by edges)
and C=boundary_2 (edges by squares) satisfy B@C=0.
"""
from __future__ import annotations
from itertools import combinations
from math import comb
import numpy as np


def cube(n: int):
    if not 2 <= n <= 7:
        raise ValueError('2 <= n <= 7 required')
    edges=[(s,i) for s in range(1<<n) for i in range(n) if not (s>>i)&1]
    edge_index={x:j for j,x in enumerate(edges)}
    B=np.zeros((1<<n,len(edges)),dtype=float)
    for k,(s,i) in enumerate(edges):
        B[s,k]=-1.
        B[s|(1<<i),k]=1.
    faces=[(s,i,j) for s in range(1<<n)
           for i,j in combinations(range(n),2)
           if not ((s>>i)&1 or (s>>j)&1)]
    C=np.zeros((len(edges),len(faces)),dtype=float)
    for k,(s,i,j) in enumerate(faces):
        C[edge_index[(s,i)],k]=1.
        C[edge_index[(s|(1<<i),j)],k]=1.
        C[edge_index[(s,j)],k]=-1.
        C[edge_index[(s|(1<<j),i)],k]=-1.
    return edges,faces,B,C


def theoretical_spectrum(n):
    """Eigenvalues 2k, multiplicities n*binomial(n-1,k-1) for 1-Laplacian."""
    return [(2*k,n*comb(n-1,k-1)) for k in range(1,n+1)]


def certificate(n, edge_values):
    """Compute global nonintegrability by projecting the observed edge field
    onto gradients. On a filled cube, residual <= 0.5 * face_curl_norm.
    """
    edges,faces,B,C=cube(n)
    g=np.asarray(edge_values,dtype=float)
    if g.shape != (len(edges),) or not np.isfinite(g).all():
        raise ValueError('one finite real increment per oriented edge')
    phi=np.linalg.lstsq(B.T,g,rcond=None)[0]
    residual=g-B.T@phi
    curl=C.T@g
    rnorm=float(np.linalg.norm(residual))
    cnorm=float(np.linalg.norm(curl))
    return {
        'dimension':n,'vertex_count':int(B.shape[0]),'edge_count':len(edges),
        'face_count':len(faces),'gradient_residual_l2':rnorm,
        'face_curl_l2':cnorm,'sharp_upper_bound':0.5*cnorm,
        'holds':bool(rnorm <= 0.5*cnorm + 1e-9),
        'boundary_squared_zero':bool(np.array_equal(B@C,np.zeros((B.shape[0],C.shape[1])))),
    }


def sharp_example(n):
    """Repeat the oriented circulation on the (0,1) square across other bits.
    This is divergence free and achieves residual_norm = curl_norm/2.
    """
    edges,_,B,C=cube(n)
    g=np.zeros(len(edges))
    for k,(s,i) in enumerate(edges):
        if i==0: g[k] = -1 if (s>>1)&1 else 1
        elif i==1: g[k] = 1 if (s>>0)&1 else -1
    return g

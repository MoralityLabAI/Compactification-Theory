"""LSV-10: exact-rational minimal linear goal-visible state under action words.

Classical finite-dimensional observability / switched-system quotient theory.
No claim of a new general theorem or nonlinear neural robustness certificate.
"""
from __future__ import annotations
from fractions import Fraction as Q
from collections.abc import Sequence


def exact(v):
    if isinstance(v, bool) or not isinstance(v, (Q, int, str)):
        raise TypeError('exact rational inputs only: int, Fraction, rational string')
    return Q(v)


def matrix(data, ncols=None):
    try:
        rows=tuple(tuple(exact(x) for x in row) for row in data)
    except TypeError as exc:
        raise ValueError('invalid matrix') from exc
    if not rows:
        raise ValueError('nonempty matrix required')
    width=len(rows[0]) if ncols is None else ncols
    if width<1 or any(len(row)!=width for row in rows):
        raise ValueError('inconsistent matrix dimensions')
    return rows


def system(modes, readout):
    a=tuple(matrix(mode) for mode in modes)
    if not a:
        raise ValueError('require at least one mode')
    n=len(a[0])
    if any(len(m)!=n or any(len(row)!=n for row in m) for m in a):
        raise ValueError('all action matrices must be n by n')
    c=matrix(readout,n)
    return a,c


def row_times(row, a):
    n=len(a)
    if len(row)!=n or any(len(r)!=n for r in a):
        raise ValueError('dimension mismatch')
    return tuple(sum((row[i]*a[i][j] for i in range(n)),Q(0)) for j in range(n))


def rref(rows,ncols=None):
    """Return a canonical rational basis for the input row span."""
    if not rows:
        return ()
    a=[list(row) for row in matrix(rows,ncols)]
    m=len(a)
    n=len(a[0])
    rank=0
    for col in range(n):
        pivot=next((i for i in range(rank,m) if a[i][col]),None)
        if pivot is None:
            continue
        a[rank],a[pivot]=a[pivot],a[rank]
        inv=1/a[rank][col]
        a[rank]=[v*inv for v in a[rank]]
        for i in range(m):
            if i!=rank and a[i][col]:
                t=a[i][col]
                a[i]=[u-t*v for u,v in zip(a[i],a[rank])]
        rank+=1
        if rank==m:
            break
    return tuple(tuple(row) for row in a[:rank])


def coefficient_row(target,basis):
    """Express target in the full-row-rank basis; reject if outside span."""
    target=tuple(exact(v) for v in target)
    if not basis:
        if all(v==0 for v in target):
            return ()
        raise ValueError('vector outside empty row span')
    p=matrix(basis)
    n=len(p[0]);r=len(p)
    if len(target)!=n or r>n:
        raise ValueError('basis dimensions wrong')
    # RREF to find pivot columns; use exact independent r x r minor.
    rank=0;pivots=[];a=[list(v) for v in p]
    for col in range(n):
        pivot=next((i for i in range(rank,r) if a[i][col]),None)
        if pivot is None:continue
        a[rank],a[pivot]=a[pivot],a[rank]
        inv=1/a[rank][col]
        a[rank]=[v*inv for v in a[rank]]
        for i in range(r):
            if i!=rank:
                factor=a[i][col]
                a[i]=[u-factor*v for u,v in zip(a[i],a[rank])]
        pivots.append(col);rank+=1
        if rank==r:break
    if rank!=r:
        raise ValueError('basis is linearly dependent')
    # P_basis may be non-RREF. Solve using exact square minor at pivot columns.
    mat=[[p[j][pivots[i]] for j in range(r)]+[target[pivots[i]]] for i in range(r)]
    for col in range(r):
        pivot=next((i for i in range(col,r) if mat[i][col]),None)
        if pivot is None:
            raise ValueError('singular basis minor')
        mat[col],mat[pivot]=mat[pivot],mat[col]
        inv=1/mat[col][col]
        mat[col]=[v*inv for v in mat[col]]
        for i in range(r):
            if i!=col:
                t=mat[i][col]
                mat[i]=[u-t*v for u,v in zip(mat[i],mat[col])]
    coeff=tuple(row[-1] for row in mat)
    if tuple(sum((v*p[j][k] for j,v in enumerate(coeff)),Q(0)) for k in range(n))!=target:
        raise ValueError('target outside basis row span')
    return coeff


def horizon_bases(modes,readout,horizon):
    """V_h = span {rows C A_w for every action word |w| <= h}."""
    A,C=system(modes,readout)
    if type(horizon) is not int or horizon<0:
        raise ValueError('nonnegative integer horizon required')
    bases=[rref(C)]
    for _ in range(horizon):
        newrows=list(bases[-1])
        for row in bases[-1]:
            for a in A:
                newrows.append(row_times(row,a))
        bases.append(rref(newrows))
    return tuple(bases)


def exact_goal_quotient(modes,readout):
    """Minimum-dimensional linear autonomous quotient for all mode words.

    Returns P, induced mode matrices B_a, and goal decoder D with
    P A_a = B_a P, C = D P. A finite-dimensional observability construction.
    """
    A,C=system(modes,readout)
    n=len(A[0])
    P=horizon_bases(A,C,n-1)[-1]
    B=tuple(tuple(coefficient_row(row_times(row,a),P) for row in P) for a in A)
    D=tuple(coefficient_row(row,P) for row in C)
    for j,a in enumerate(A):
        for i,row in enumerate(P):
            recovered=tuple(sum((B[j][i][t]*P[t][k] for t in range(len(P))),Q(0)) for k in range(n))
            if recovered!=row_times(row,a):
                raise AssertionError('induced transition failed')
    return {'projection':P, 'actions':B, 'readout':D, 'micro_dim':n,
            'goal_dim':len(C),'minimal_dim':len(P)}


def step(a, x, bias=None):
    if len(a)!=len(x):raise ValueError('bad state length')
    b=tuple(Q(0) for _ in x) if bias is None else tuple(map(exact,bias))
    if len(b)!=len(x):raise ValueError('bad bias length')
    return tuple(sum((v*w for v,w in zip(row,x)),Q(0))+v for row,v in zip(a,b))


def project(P,x):
    return tuple(sum((v*w for v,w in zip(row,x)),Q(0)) for row in P)


def defect_transfer_bound(initial, rho, defect, steps):
    """Finite-step bound under ||pi F(x)-G pi(x)||<=defect, Lip(G)<=rho."""
    err,rate,eps=exact(initial),exact(rho),exact(defect)
    if min(err,rate,eps)<0 or rate>=1 or type(steps) is not int or steps<0:
        raise ValueError('contractive macro and nonnegative error budget required')
    return rate**steps*err+eps*(1-rate**steps)/(1-rate)


def margin_preserved(nominal_margin, goal_readout_gain, projected_error):
    """Sufficient strict winner condition under a projected error radius."""
    m,L,e=exact(nominal_margin),exact(goal_readout_gain),exact(projected_error)
    if L<0 or e<0:
        raise ValueError('invalid nonnegative gain/error')
    return m>L*e

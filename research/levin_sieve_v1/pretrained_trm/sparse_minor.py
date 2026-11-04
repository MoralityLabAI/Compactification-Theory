"""LSV-12: sparse square-minor witnesses for full row rank.

Mathematical rank certs require a validated *entrywise* bound on the true
Jacobian. Here the greedy minor finder uses point derivatives and reports the
required error budget; it does not assert that the budget has been proved.
"""
from __future__ import annotations
import torch


def greedy_minor(rows:torch.Tensor) -> dict:
    """Select r columns of an r-by-n matrix by residual-volume greedy selection.

    Returns an observed min singular value and max permissible entrywise error
    for a validated singular-value rank certificate (strict inequality).
    """
    if rows.ndim != 2 or rows.shape[0]<1 or rows.shape[1]<rows.shape[0]:
        raise ValueError('expect r rows and n >= r columns')
    if not rows.is_floating_point() or not torch.isfinite(rows).all():
        raise ValueError('finite floating-point matrix required')
    J=rows.double().detach()
    r,n=J.shape
    residual=J.clone()
    picks=[]
    for _ in range(r):
        scores=residual.square().sum(0)
        if picks:scores[picks]=-1
        j=int(scores.argmax().item())
        if scores[j]<=0:
            raise ValueError('observed matrix is rank deficient')
        q=residual[:,j]/torch.sqrt(scores[j])
        picks.append(j)
        residual=residual - q[:,None]*(q@residual)[None,:]
    M=J[:,picks]
    singular=torch.linalg.svdvals(M)
    sigma_min=float(singular[-1])
    if not sigma_min>0:
        raise ValueError('selected minor is singular')
    # If every entry of true M differs from observed M by at most epsilon,
    # then ||Mtrue-Mobserved||_2 <= ||.||_F <= r*epsilon.
    return {'columns':picks,'rows':r,'columns_total':n,
            'singular_values':singular.tolist(),
            'observed_sigma_min':sigma_min,
            'required_verified_abs_entry_error_lt':sigma_min/r,
            'observed_max_abs_entry':float(M.abs().amax())}


def exact_det(M):
    """Fraction-preserving determinant for independent small-matrix tests."""
    from fractions import Fraction as Q
    a=[[Q(x) for x in row] for row in M]
    n=len(a)
    if not n or any(len(row)!=n for row in a):
        raise ValueError('square nonempty matrix required')
    det=Q(1)
    for k in range(n):
        pivot=next((i for i in range(k,n) if a[i][k]),None)
        if pivot is None:return Q(0)
        if pivot!=k:
            a[k],a[pivot]=a[pivot],a[k]
            det=-det
        p=a[k][k]
        det*=p
        for i in range(k+1,n):
            factor=a[i][k]/p
            for j in range(k+1,n):a[i][j]-=factor*a[k][j]
            a[i][k]=Q(0)
    return det

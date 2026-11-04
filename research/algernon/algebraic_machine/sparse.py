"""First-order intertwining defects and support-restricted linear repairs.

Real-valued diagnostic only; independent of Qwen weights. Assumes matrices
represent local Jacobians of a fixed neural-state decoder and transition.
"""
from __future__ import annotations

import numpy as np


def intertwining_defect(P, M, A):
    P, M, A = (np.asarray(t, dtype=np.float64) for t in (P, M, A))
    if P.ndim != 2 or M.shape != (P.shape[1], P.shape[1]) or A.shape != (P.shape[0], P.shape[0]):
        raise ValueError("expected decoder [s,h], neural [h,h], target [s,s]")
    return P @ M - A @ P


def sparse_repair(P, M, A, U, atol=1e-9):
    """Minimize ||(PM-AP) + (P U)B||_F over B with chosen columns U.

    Returns the optimal B, ΔM=U B, residual and relative defect reduction.
    The best possible residual is (I-K K†)D where K=P U and D=PM-AP.
    """
    P=np.asarray(P,dtype=np.float64)
    U=np.asarray(U,dtype=np.float64)
    D=intertwining_defect(P,M,A)
    if U.ndim!=2 or U.shape[0]!=P.shape[1]:
        raise ValueError("U must be [hidden,rank]")
    K=P@U
    if U.shape[1]:
        B=np.linalg.lstsq(K,-D,rcond=None)[0]
        projection=K@np.linalg.pinv(K)@D
    else:
        B=np.zeros((0,P.shape[1]),dtype=np.float64)
        projection=np.zeros_like(D)
    delta=U@B
    repaired=intertwining_defect(P,np.asarray(M,dtype=np.float64)+delta,A)
    theoretical=D-projection
    if not np.allclose(repaired,theoretical,atol=atol,rtol=1e-7):
        raise ArithmeticError("projected defect differs from direct repair")
    original_norm=float(np.linalg.norm(D))
    optimal_norm=float(np.linalg.norm(theoretical))
    return {"B":B,"delta_M":delta,"defect":D,"repaired_defect":repaired,
            "optimal_residual_norm":optimal_norm,
            "actual_residual_norm":float(np.linalg.norm(repaired)),
            "exactly_repairable":bool(optimal_norm<=atol),
            "explained_energy_fraction":1-(optimal_norm**2/original_norm**2) if original_norm>0 else 1.0,
            "rank_PU":int(np.linalg.matrix_rank(K)) if K.size else 0,
            "rank_defect":int(np.linalg.matrix_rank(D))}


def step_error_bound(A, qs, initial):
    """Numerical certificate for ||q_n-A^n q_0||_1 <= sum ||q_(k+1)-A q_k||_1.

    The nonexpansiveness claim requires a column-stochastic A. Returns the
    evaluated bound and terminal deviation; initial truth error is separate.
    """
    A=np.asarray(A,dtype=np.float64)
    arr=[np.asarray(q,dtype=np.float64) for q in qs]
    init=np.asarray(initial,dtype=np.float64)
    n=A.shape[0]
    if A.shape!=(n,n) or not arr or any(x.shape!=(n,) for x in arr) or init.shape!=(n,):
        raise ValueError("bad transition dimensions")
    if np.min(A)<-1e-12 or not np.allclose(A.sum(axis=0),1,atol=1e-12):
        raise ValueError("bound requires a column-stochastic map")
    if any(np.min(x)<-1e-12 or not np.isclose(x.sum(),1) for x in arr):
        raise ValueError("states must be probability vectors")
    eps=[float(np.linalg.norm(arr[i+1]-A@arr[i],ord=1)) for i in range(len(arr)-1)]
    exact=arr[0]
    for _ in eps:
        exact=A@exact
    deviation=float(np.linalg.norm(arr[-1]-exact,ord=1))
    truth=init
    for _ in eps:
        truth=A@truth
    initial_error=float(np.linalg.norm(arr[0]-init,ord=1))
    truth_error=float(np.linalg.norm(arr[-1]-truth,ord=1))
    bound=initial_error+sum(eps)
    if truth_error>bound+1e-9 or deviation>sum(eps)+1e-9:
        raise ArithmeticError("telescoping error bound violated")
    return {"n":len(eps),"one_step_errors":eps,"accumulated_step_bound":sum(eps),
            "initial_error":initial_error,"truth_error":truth_error,
            "certified_bound":bound,"correct_argmax_guaranteed":bound<1.0}

"""CGC-01 mathematical reference for task-protected recurrent compactification.

Source-only finite-dimensional linear algebra; no neural model, pretrained
checkpoint, trained BPD result, privileged heldout datum, or security claim.
The Euclidean physical-state metric and output weights must be declared.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import ceil
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ProtectedSpectrum:
    protected_basis: np.ndarray
    available_dimension: int
    task_singular_values: np.ndarray
    task_rank: int
    task_gramian: np.ndarray


def numerical_rank(singular_values: np.ndarray, *, tol: float = 1e-10) -> int:
    """Thresholded numerical rank; tol is an absolute, stated threshold."""
    return int(np.count_nonzero(np.asarray(singular_values) > tol))


def null_basis(jacobian: np.ndarray, *, tol: float = 1e-10) -> np.ndarray:
    """Orthonormal Euclidean columns spanning the null space of a 2-D map."""
    b = np.asarray(jacobian, dtype=float)
    if b.ndim != 2:
        raise ValueError("expected 2D protected Jacobian")
    _, s, vt = np.linalg.svd(b, full_matrices=True)
    rank = numerical_rank(s, tol=tol)
    return vt[rank:].T.copy()


def protected_task_spectrum(protected: np.ndarray, task: np.ndarray,
                            *, tol: float = 1e-10) -> ProtectedSpectrum:
    """Task-observability Gramian restricted to first-order protected kernel.

    `task` may stack weighted future readouts from multiple horizons/tasks.
    """
    b = np.asarray(protected, dtype=float)
    a = np.asarray(task, dtype=float)
    if b.ndim != 2 or a.ndim != 2 or b.shape[1] != a.shape[1]:
        raise ValueError("protected/task matrices must share physical state dimension")
    q = null_basis(b, tol=tol)
    restricted = a @ q
    s = np.linalg.svd(restricted, compute_uv=False)
    gramian = restricted.T @ restricted
    return ProtectedSpectrum(q, q.shape[1], s, numerical_rank(s, tol=tol), gramian)


def task_optimal_block(protected: np.ndarray, task: np.ndarray, rank: int,
                       *, tol: float = 1e-10) -> tuple[np.ndarray, float, float]:
    """Rank-r physical-state block; exact first-order spectral/Frobenius tails.

    Returns physical orthonormal basis E, operator residual and squared
    Frobenius residual on the protected kernel. These are local linear facts.
    """
    spec = protected_task_spectrum(protected, task, tol=tol)
    k = spec.available_dimension
    if not (0 <= rank <= k):
        raise ValueError("rank must be between 0 and dimension of protected kernel")
    a = np.asarray(task, dtype=float) @ spec.protected_basis
    _, s, vt = np.linalg.svd(a, full_matrices=True)
    e = spec.protected_basis @ vt[:rank, :].T
    tail = s[rank:]
    operator_error = float(tail[0]) if tail.size else 0.0
    frob_sq_error = float(tail @ tail)
    return e, operator_error, frob_sq_error


def physical_svd_write(feature_delta: np.ndarray, physical_weight: np.ndarray,
                       rank: int) -> tuple[np.ndarray, float | None]:
    """Physical output SVD truncation, invariant to V,U factor re-gauging.

    For row-state convention: feature_delta [N, d_in], W [d_in, d_out].
    """
    x = np.asarray(feature_delta, dtype=float)
    w = np.asarray(physical_weight, dtype=float)
    if x.ndim != 2 or w.ndim != 2 or x.shape[1] != w.shape[0]:
        raise ValueError("incompatible physical write shapes")
    u, s, vt = np.linalg.svd(w, full_matrices=False)
    if not (1 <= rank <= len(s)):
        raise ValueError("rank outside allowed physical SVD range")
    basis = vt[:rank, :]
    write = x @ w
    result = (write @ basis.T) @ basis
    gap = float(s[rank-1] - s[rank]) if rank < len(s) else None
    return result, gap


def least_norm_public_repair(protected_kernel_basis: np.ndarray,
                             public_jacobian: np.ndarray,
                             public_residual: np.ndarray,
                             *, tol: float = 1e-10) -> np.ndarray:
    """Minimum-L2 tangent correction inside the existing protected kernel.

    Requires full row rank of public_jacobian restricted to K so every
    first-order public-output residual can be cancelled. No finite safety
    assertion is made without a second-order remainder certificate.
    """
    q = np.asarray(protected_kernel_basis, dtype=float)
    b = np.asarray(public_jacobian, dtype=float)
    residual = np.asarray(public_residual, dtype=float).reshape(-1)
    if q.ndim != 2 or b.ndim != 2 or b.shape[1] != q.shape[0] or b.shape[0] != residual.size:
        raise ValueError("incompatible kernel, public Jacobian, or residual")
    if not np.allclose(q.T @ q, np.eye(q.shape[1]), atol=1e-8):
        raise ValueError("kernel columns must be orthonormal")
    restricted = b @ q
    if np.linalg.matrix_rank(restricted, tol=tol) != b.shape[0]:
        raise ValueError("public Jacobian is not surjective on the protected kernel")
    return q @ (-np.linalg.pinv(restricted, rcond=tol) @ residual)


def local_finite_compactification_bound(sigma_next: float, radius: float,
                                          hessian_operator_bound: float) -> float:
    """||G(h+delta)-G(h+P_E delta)|| <= rho*sigma_(r+1) + M*rho^2.

    Requires both edits to lie in an r-radius convex neighborhood on which
    ||D^2 G||op <= M; block E must be first-order task-optimal.
    """
    if min(sigma_next, radius, hessian_operator_bound) < 0:
        raise ValueError("nonnegative bound arguments required")
    return float(radius*sigma_next + hessian_operator_bound*radius*radius)


def finite_newton_remainder_bound(hessian_operator_bound: float,
                                   step: np.ndarray) -> float:
    """||public residual after exact tangent cancellation|| <= M||s||^2/2."""
    if hessian_operator_bound < 0:
        raise ValueError("Hessian bound cannot be negative")
    return float(0.5*hessian_operator_bound * np.linalg.norm(step)**2)


def rowspace_basis(rows: np.ndarray, *, tol: float = 1e-10) -> np.ndarray:
    b = np.asarray(rows, dtype=float)
    if b.ndim != 2:
        raise ValueError("expected 2D row matrix")
    _, s, vt = np.linalg.svd(b, full_matrices=False)
    return vt[:numerical_rank(s, tol=tol), :].copy()


@dataclass(frozen=True)
class AffineQuotient:
    encoder: np.ndarray
    transitions: tuple[np.ndarray, ...]
    readout: np.ndarray
    input_offsets: tuple[np.ndarray, ...]
    dimension: int


def minimal_real_affine_quotient(transition_matrices: Sequence[np.ndarray],
                                 output_matrix: np.ndarray,
                                 translations: Sequence[np.ndarray] | None = None,
                                 *, tol: float = 1e-10) -> AffineQuotient:
    """Construct minimal exact real-*linear* future-observable affine quotient.

    Rowspace closure is the real-field counterpart of the RC-01 F2 theorem.
    It proves minimality among linear sufficient encoders for this full state
    domain and supplied action alphabet, not a nonlinear-model certificate.
    """
    c = np.asarray(output_matrix, dtype=float)
    if c.ndim != 2:
        raise ValueError("output must be 2D")
    d = c.shape[1]
    aa = tuple(np.asarray(a, dtype=float) for a in transition_matrices)
    if not aa or any(a.shape != (d,d) for a in aa):
        raise ValueError("nonempty alphabet of square, shared-dimension matrices required")
    if translations is None:
        bb = tuple(np.zeros(d) for _ in aa)
    else:
        bb = tuple(np.asarray(b,dtype=float).reshape(-1) for b in translations)
        if len(bb) != len(aa) or any(b.shape != (d,) for b in bb):
            raise ValueError("translations must match action alphabet")
    basis = rowspace_basis(c,tol=tol)
    for _ in range(d+1):
        grown = rowspace_basis(np.vstack((basis,)*(1) + tuple(basis @ a for a in aa)),tol=tol)
        if grown.shape[0] == basis.shape[0]:
            basis = grown
            break
        basis = grown
    else:
        raise AssertionError("row-span closure exceeded state dimension")
    reduced = tuple(basis @ a @ basis.T for a in aa)
    out = c @ basis.T
    off = tuple(basis @ b for b in bb)
    for a,hat in zip(aa,reduced):
        if not np.allclose(basis @ a,hat @ basis,atol=1e-7):
            raise AssertionError("row space did not close")
    if not np.allclose(out @ basis,c,atol=1e-7):
        raise AssertionError("readout factorization failed")
    return AffineQuotient(basis,reduced,out,off,basis.shape[0])


def affine_observability_matrix(transition: np.ndarray, readout: np.ndarray,
                                horizon: int) -> np.ndarray:
    """Stack C, C A, ..., C A^(horizon-1) for one fixed affine transition."""
    a = np.asarray(transition,dtype=float)
    c = np.asarray(readout,dtype=float)
    if horizon < 1 or a.ndim != 2 or a.shape[0] != a.shape[1] or c.ndim != 2 or c.shape[1] != a.shape[0]:
        raise ValueError("invalid observability input")
    acc = np.eye(len(a))
    rows = []
    for _ in range(horizon):
        rows.append(c @ acc)
        acc = acc @ a
    return np.vstack(rows)


def observability_gramian(transition: np.ndarray, readout: np.ndarray,
                           horizon: int, protected_basis: np.ndarray | None = None) -> np.ndarray:
    stacked = affine_observability_matrix(transition,readout,horizon)
    if protected_basis is not None:
        stacked = stacked @ np.asarray(protected_basis,dtype=float)
    return stacked.T @ stacked


def minimum_code_bits(separated_futures: int, bits_per_coordinate: int) -> tuple[int,int]:
    """If decoder error < half the target separation, N states need ceil(log2 N) bits.

    The dimension bound assumes each stored coordinate has at most b bits;
    inputs, decoder, selector and any side information must be fixed/charged.
    """
    if separated_futures < 1 or bits_per_coordinate < 1:
        raise ValueError("positive N and precision required")
    bits = (separated_futures-1).bit_length()
    return bits,ceil(bits/bits_per_coordinate)


def exact_rational_rank(matrix: Sequence[Sequence[int | Fraction]]) -> int:
    """Small exact rational rank oracle independent of floating-point SVD."""
    if not matrix:
        return 0
    a = [[Fraction(x) for x in row] for row in matrix]
    width = len(a[0]); nrows = len(a)
    if any(len(row) != width for row in a):
        raise ValueError("ragged matrix")
    pos=0
    for col in range(width):
        pivot = next((row for row in range(pos,nrows) if a[row][col]),None)
        if pivot is None:
            continue
        a[pos],a[pivot]=a[pivot],a[pos]
        k=a[pos][col]
        a[pos]=[v/k for v in a[pos]]
        for i in range(nrows):
            if i!=pos:
                coef=a[i][col]
                a[i]=[v-coef*p for v,p in zip(a[i],a[pos])]
        pos+=1
        if pos==nrows:
            break
    return pos


def exact_affine_krylov_rank(a: Sequence[Sequence[int]], c: Sequence[Sequence[int]]) -> int:
    """Exact rank of [C;CA;...;CA^(d-1)] over rational numbers for integer data."""
    aa=[[Fraction(x) for x in row] for row in a]
    cc=[[Fraction(x) for x in row] for row in c]
    d=len(aa)
    if not aa or any(len(row)!=d for row in aa) or any(len(row)!=d for row in cc):
        raise ValueError("invalid rational affine system")
    block = cc
    rows=[]
    for _ in range(d):
        rows.extend(block)
        block=[[sum(row[k]*aa[k][j] for k in range(d)) for j in range(d)] for row in block]
    return exact_rational_rank(rows)


def synthetic_receipt() -> dict[str,object]:
    """All claims below concern exact/simple linear-algebra constructions only."""
    a = np.array([[1.,1.,0.],[0.,1.,0.],[0.,0.,.5]])
    c = np.array([[1.,0.,0.]])
    exact = exact_affine_krylov_rank([[1,1,0],[0,1,0],[0,0,1]],[[1,0,0]])
    q = minimal_real_affine_quotient([a],c)
    p=np.array([[1.,0.,0.]])
    target=np.array([[0.,0.,1.]])
    basis, op_tail, frob = task_optimal_block(p,target,1)
    physical=np.diag([10.,1.])
    feature=np.array([[1.,1.]])
    physical_top,_=physical_svd_write(feature,physical,1)
    return {
        "source_only":True,
        "full_real_affine_quotient_rank":q.dimension,
        "integer_rational_krylov_rank":exact,
        "protected_kernel_dimension":int(null_basis(p).shape[1]),
        "protected_task_rank":protected_task_spectrum(p,target).task_rank,
        "protected_task_one_rank_operator_tail":op_tail,
        "physical_weight_top_singular_output":physical_top[0].tolist(),
        "task_gradient_orthogonal_to_top_weight_output":bool(abs(physical_top[0] @ np.array([0.,1.]))<1e-12),
        "trained_BPD_evidence":False,
        "real_model_privacy_evidence":False,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(synthetic_receipt(),indent=2,sort_keys=True))

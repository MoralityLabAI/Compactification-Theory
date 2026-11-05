"""PSS-01: CPU reference for protected-subspace, low-rank weight surgery.

Columns of D/S/H are failure/output-protection/input-protection directions.
W has shape [out, in]. Algebraic guarantees concern one linear receiver.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import numpy as np


def matrix(x, name: str) -> np.ndarray:
    if np.iscomplexobj(x):
        raise ValueError(f"{name} must be real")
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] == 0 or not np.isfinite(a).all():
        raise ValueError(f"{name} must be a finite 2D matrix with positive row count")
    return a


def orth(x, *, rtol=1e-10, atol=1e-12, reference_scale=None):
    """Rank-revealing basis; original-scale cutoff rejects projection roundoff."""
    a = matrix(x, "directions")
    if not np.isfinite([rtol, atol]).all() or min(rtol, atol) < 0:
        raise ValueError("rank tolerances must be finite and nonnegative")
    if a.shape[1] == 0:
        return a.copy()
    u, singular, _ = np.linalg.svd(a, full_matrices=False)
    scale = singular[0] if reference_scale is None else float(reference_scale)
    if not np.isfinite(scale) or scale < 0:
        raise ValueError("reference_scale must be finite and nonnegative")
    q = u[:, singular > max(atol, rtol * scale)]
    for j in range(q.shape[1]):
        if q[np.argmax(np.abs(q[:, j])), j] < 0:
            q[:, j] *= -1
    return q


def fingerprint(x) -> str:
    a = np.ascontiguousarray(matrix(x, "weights"), dtype="<f8")
    return hashlib.sha256(str(a.shape).encode() + a.tobytes()).hexdigest()


def complement_right(a, h):
    return a - (a @ h) @ h.T


@dataclass(frozen=True)
class SurgeryPlan:
    q: np.ndarray
    s: np.ndarray
    h: np.ndarray
    b: np.ndarray
    a: np.ndarray
    strength: float
    source_sha256: str
    retained_failure_energy: float

    def delta(self) -> np.ndarray:
        return self.b @ self.a

    def merge(self, w) -> np.ndarray:
        w = matrix(w, "weights")
        if fingerprint(w) != self.source_sha256:
            raise ValueError("plan belongs to a different source tensor")
        return w + self.delta()


def plan_surgery(w, d, s=None, h=None, *, rank=None, strength=1.0,
                 rtol=1e-10, atol=1e-12) -> SurgeryPlan:
    """Return Delta = -strength Q Q.T W (I-H H.T), factored as B @ A.

    Q is an orthonormal basis of the retained part of (I-S S.T) D.
    S and H are orthonormalized internally. A truncated rank keeps the
    leading singular directions of the protected failure matrix.
    """
    w, d = matrix(w, "weights"), matrix(d, "failure directions")
    if w.shape[1] == 0 or d.shape[0] != w.shape[0]:
        raise ValueError("weights/failure dimensions do not agree")
    if not np.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("strength must be in [0, 1]")
    if rank is not None and (isinstance(rank, bool) or
                            not isinstance(rank, (int, np.integer)) or rank < 0):
        raise ValueError("rank must be a nonnegative integer or None")
    s = matrix(np.empty((w.shape[0], 0)) if s is None else s, "protected outputs")
    h = matrix(np.empty((w.shape[1], 0)) if h is None else h, "protected inputs")
    if s.shape[0] != w.shape[0] or h.shape[0] != w.shape[1]:
        raise ValueError("protection dimensions do not agree with weights")
    s, h = orth(s, rtol=rtol, atol=atol), orth(h, rtol=rtol, atol=atol)
    residual = d - s @ (s.T @ d)
    residual -= s @ (s.T @ residual)
    scale = np.linalg.norm(d, 2) if d.shape[1] else 0.0
    q = orth(residual, rtol=rtol, atol=atol, reference_scale=scale)
    if rank is not None:
        q = q[:, :rank]
    # Reproject after the SVD to keep near-parallel directions orthogonal.
    q = orth(q - s @ (s.T @ q), rtol=rtol, atol=atol, reference_scale=1.0)
    energy = float(np.linalg.norm(q.T @ d) ** 2 / np.linalg.norm(d) ** 2) if np.any(d) else 0.0
    a = complement_right(q.T @ w, h)
    b = -float(strength) * q
    # Frozen dataclass plus read-only arrays prevents accidental plan mutation.
    for arr in (q, s, h, a, b):
        arr.setflags(write=False)
    return SurgeryPlan(q, s, h, b, a, float(strength), fingerprint(w), energy)


def audit(w, candidate, plan: SurgeryPlan, *, rtol=1e-10, atol=1e-12) -> dict:
    """Measure constraints on the supplied realized weights, including casts."""
    w, candidate = matrix(w, "weights"), matrix(candidate, "candidate")
    if w.shape != candidate.shape or fingerprint(w) != plan.source_sha256:
        raise ValueError("candidate/source mismatch")
    if not np.isfinite([rtol, atol]).all() or min(rtol, atol) < 0:
        raise ValueError("audit tolerances must be finite and nonnegative")
    delta = candidate - w
    restriction = complement_right(plan.q.T @ delta, plan.h) + plan.strength * plan.a
    errors = {
        "protected_output_error_fro": float(np.linalg.norm(plan.s.T @ delta)),
        "protected_input_error_fro": float(np.linalg.norm(delta @ plan.h)),
        "target_constraint_error_fro": float(np.linalg.norm(restriction)),
    }
    tolerance = float(atol + rtol * np.linalg.norm(w))
    return {
        **errors,
        "within_tolerance": all(e <= tolerance for e in errors.values()),
        "tolerance": tolerance,
        "delta_fro": float(np.linalg.norm(delta)),
        "merge_deviation_fro": float(np.linalg.norm(delta - plan.delta())),
        "remaining_q_output_fro": float(np.linalg.norm(plan.q.T @ candidate)),
        "protected_input_residual_floor_fro": float(np.linalg.norm(plan.q.T @ w @ plan.h)),
        "retained_failure_energy": plan.retained_failure_energy,
        "factor_rank_bound": int(plan.q.shape[1]),
        "changed_entries": int(np.count_nonzero(delta)),
        "total_entries": int(w.size),
        "source_sha256": plan.source_sha256,
        "candidate_sha256": fingerprint(candidate),
    }


def bf16_round(x) -> np.ndarray:
    """Finite FP32 -> BF16 round-to-nearest-even, returned as FP32 values.

    This audits storage rounding, not device matmul/accumulation behavior.
    """
    source = matrix(x, "BF16 input")
    with np.errstate(over="ignore", invalid="ignore"):
        f = source.astype(np.float32)
    if not np.isfinite(f).all():
        raise ValueError("input overflows float32")
    bits = np.ascontiguousarray(f).view(np.uint32).astype(np.uint64)
    rounded = ((bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000)
    result = rounded.astype(np.uint32).view(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("input overflows BF16")
    return result


def assert_disjoint_ids(**splits):
    """Use dataset-qualified task-family IDs so paraphrases share one split."""
    owner = {}
    for name, ids in splits.items():
        seen = set()
        for key in ids:
            if not isinstance(key, str) or not key:
                raise ValueError("split IDs must be nonempty strings")
            if key in seen or key in owner:
                raise ValueError(f"duplicate or overlapping split ID: {key}")
            seen.add(key)
            owner[key] = name

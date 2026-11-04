"""MONO-HODGE-01: observational evidence-order diagnostics and CPU rank-one toy editor.

This module DOES NOT infer hidden-state transition operators from completed-prompt
activations. The intervention API requires an executable transition function.
All output of toy_demo() is synthetic; no 27B Qwen model is loaded.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

import numpy as np


def _matrix(x, name: str, cols: int | None = None) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 2 or not len(a) or (cols is not None and a.shape[1] != cols):
        raise ValueError(f"{name}: expected nonempty rank-two matrix")
    if not np.isfinite(a).all():
        raise ValueError(f"{name}: nonfinite data")
    return a


def bootstrap_cluster_ci(values, groups, *, seed=20261104, repeats=2000):
    """95% percentile interval by resampling unique families, not rows.

    Returns descriptive uncertainty for the supplied sample; no causal claim.
    """
    v = np.asarray(values, dtype=np.float64)
    g = np.asarray(groups, dtype=str)
    if v.ndim != 1 or len(v) == 0 or g.shape != v.shape or not np.isfinite(v).all():
        raise ValueError("values and groups must be aligned and finite")
    uniq, inv = np.unique(g, return_inverse=True)
    if len(uniq) < 2:
        return [None, None]  # one family cannot support a family-bootstrap interval
    members = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(repeats):
        picked = rng.integers(len(members), size=len(members))
        rows = np.concatenate([members[i] for i in picked])
        means.append(float(v[rows].mean()))
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def order_gap_report(log_ab, log_ba, true_log_odds, groups, *, seed=20261104):
    """Paired observational readout diagnostics; log-odds truth is order-invariant.

    Input must contain one matched final-prompt pair per row, with identical
    evidence multiset and an order-invariant *task-defined* posterior target.
    """
    a, b, t = map(lambda x: np.asarray(x, dtype=np.float64), (log_ab, log_ba, true_log_odds))
    g = np.asarray(groups, dtype=str)
    if any(x.ndim != 1 or len(x) == 0 or x.shape != a.shape for x in (a, b, t)) or g.shape != a.shape:
        raise ValueError("paired arrays must have identical nonempty shape")
    if not np.isfinite(np.column_stack([a, b, t])).all():
        raise ValueError("nonfinite paired readouts")
    gap = np.abs(a - b)
    err_a = np.abs(a-t)
    err_b = np.abs(b-t)
    mean_abs_pair_error = (err_a+err_b)/2
    return {
        "evidence": "observational_paired_final_readouts",
        "n_pairs": int(len(a)), "n_families": int(len(np.unique(g))),
        "mean_abs_order_gap_nats": float(np.mean(gap)),
        "mean_abs_order_gap_cluster_ci95": bootstrap_cluster_ci(gap,g,seed=seed),
        "mean_abs_posterior_error_nats": float(np.mean(mean_abs_pair_error)),
        "mean_abs_posterior_error_cluster_ci95": bootstrap_cluster_ci(mean_abs_pair_error,g,seed=seed+1),
        "signed_mean_ab_minus_ba_nats": float(np.mean(a-b)),
        "has_operator_commutator_measurement": False,
        "has_causal_intervention": False,
    }


def read_normalized_pairs(path: str | Path):
    """Read strictly normalized schema, never guess from original archive keys.

    JSONL columns: family_id, log_odds_ab, log_odds_ba, true_log_odds,
    optionally state_ab/state_ba vectors. Group by generator-root family_id.
    """
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            required = {"family_id", "log_odds_ab", "log_odds_ba", "true_log_odds"}
            if not required.issubset(row):
                raise ValueError(f"line {line_no}: missing normalized fields: {required - set(row)}")
            if not isinstance(row["family_id"], str) or not row["family_id"].strip():
                raise ValueError("family_id must be a nonempty group identifier")
            rows.append(row)
    if not rows:
        raise ValueError("empty pair dataset")
    keys = ["log_odds_ab", "log_odds_ba", "true_log_odds"]
    cols = [np.asarray([r[k] for r in rows], dtype=np.float64) for k in keys]
    report = order_gap_report(*cols, [r["family_id"] for r in rows])
    if all("state_ab" in r and "state_ba" in r for r in rows):
        hab = _matrix([r["state_ab"] for r in rows], "state_ab")
        hba = _matrix([r["state_ba"] for r in rows], "state_ba", hab.shape[1])
        report["state_differences"] = hab - hba
    elif any("state_ab" in r or "state_ba" in r for r in rows):
        raise ValueError("states must be provided in complete ab/ba pairs")
    return rows, report


def woodbury_solve(reference, B, ridge=1e-3):
    """(lambda I + X^T X/N)^-1 B using only an N-by-N inverse."""
    X = _matrix(reference, "reference")
    B = _matrix(B, "B")
    if B.shape[0] != X.shape[1]:
        raise ValueError("B: wrong number of rows")
    if not np.isfinite(ridge) or ridge <= 0:
        raise ValueError("ridge must be positive")
    N = len(X)
    return (B - X.T @ np.linalg.solve(X @ X.T + N*ridge*np.eye(N), X @ B))/ridge


def antisymmetric_modes(differences, reference, *, ridge=1e-3, topk=3):
    """Generalized eigenvectors: D^T D/n v = eigval * C v.

    These are directions of *order sensitivity* relative to ordinary-state
    second moment. A large eigval is neither causal evidence nor a repair.
    """
    D = _matrix(differences, "differences")
    X = _matrix(reference, "reference", D.shape[1])
    if not 1 <= topk <= min(len(D), D.shape[1]):
        raise ValueError("invalid topk")
    Z = woodbury_solve(X, D.T, ridge)
    kernel = D @ Z / len(D)
    kernel = (kernel+kernel.T)/2
    eigs, U = np.linalg.eigh(kernel)
    ix = np.argsort(eigs)[::-1]
    eigs, U = eigs[ix], U[:,ix]
    dirs, vals = [], []
    for idx in range(len(eigs)):
        if eigs[idx] <= 1e-12:
            continue
        v = Z @ U[:,idx] / np.sqrt(len(D)*eigs[idx])
        dirs.append(v)
        vals.append(float(eigs[idx]))
        if len(vals) == topk:
            break
    V = np.column_stack(dirs) if dirs else np.zeros((D.shape[1], 0))
    return np.array(vals), V


def deterministic_group_folds(groups, folds=5, seed=20261104):
    groups = np.asarray(groups, dtype=str)
    unique = np.unique(groups)
    if folds < 2 or len(unique) < folds:
        raise ValueError("fewer independent groups than folds")
    # Deterministic independently of row order or Python's salted hash.
    ordered = sorted(unique, key=lambda g: hashlib.sha256(f"{seed}:{g}".encode()).digest())
    assignment = {g: i % folds for i, g in enumerate(ordered)}
    return np.asarray([assignment[g] for g in groups], dtype=int)


def _ridge_predict(X_train, y_train, X_test, ridge):
    """Train-only centering/scaling; intercept stays unpenalized."""
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0)
    sd[sd < 1e-8] = 1.0
    A = (X_train-mu)/sd
    T = (X_test-mu)/sd
    center_y = float(y_train.mean())
    centered = y_train-center_y
    if A.shape[1] > A.shape[0]:
        # Dual ridge handles thousands of latent coordinates with <200 families.
        alpha = np.linalg.solve(A@A.T + ridge*np.eye(A.shape[0]), centered)
        return (T@A.T)@alpha + center_y
    beta = np.linalg.solve(A.T@A + ridge*np.eye(A.shape[1]), A.T@centered)
    return T @ beta + center_y


def incremental_probe_value(state, latent, advantage, groups, *, folds=5, ridge=10.0):
    """Group-held-out predictive gain from Z beyond S; NOT a mutual-information estimate."""
    S = _matrix(state, "state")
    Z = _matrix(latent, "latent")
    y = np.asarray(advantage, dtype=np.float64)
    if Z.shape[0] != S.shape[0] or y.shape != (len(S),) or not np.isfinite(y).all():
        raise ValueError("incompatible outcome dimensions")
    part = deterministic_group_folds(groups, folds)
    p0, p1 = np.empty_like(y), np.empty_like(y)
    for f in range(folds):
        tr, te = part != f, part == f
        p0[te] = _ridge_predict(S[tr], y[tr], S[te], ridge)
        p1[te] = _ridge_predict(np.concatenate((S[tr], Z[tr]), axis=1), y[tr],
                                np.concatenate((S[te], Z[te]), axis=1), ridge)
    err0, err1 = (y-p0)**2, (y-p1)**2
    gain = err0-err1
    return {"evidence":"group_disjoint_predictive_probe_not_mutual_information",
            "n":int(len(y)), "n_groups":int(len(np.unique(groups))),
            "state_only_mse":float(err0.mean()),"state_plus_latent_mse":float(err1.mean()),
            "incremental_mse_gain":float(gain.mean()),
            "paired_cluster_ci95":bootstrap_cluster_ci(gain,groups),
            "folds":folds,"ridge":ridge}


def commutator(transition: Callable, W, states, evidence_a, evidence_b):
    """T_b(T_a(h))-T_a(T_b(h)) for executable same-gauge state transitions."""
    h = _matrix(states, "states")
    a = _matrix(evidence_a, "a", h.shape[1])
    b = _matrix(evidence_b, "b", h.shape[1])
    if a.shape != h.shape or b.shape != h.shape:
        raise ValueError("evidence matrices must match states")
    W = _matrix(W, "W", h.shape[1])
    if W.shape[0] != h.shape[1]:
        raise ValueError("W must be square for this toy interface")
    return transition(transition(h,a,W),b,W)-transition(transition(h,b,W),a,W)


def finite_diff_rankone_edit(transition, W, states, evidence_a, evidence_b, u, reference,
                             *, support=None, diff_eps=1e-5, preservation=1e-6):
    """One Gauss-Newton rank-one update: W' = W + u z^T on chosen columns.

    J derived from finite-difference *operator replay*. This cannot be applied
    to frozen final-state shards without replaying the target checkpoint.
    """
    W = _matrix(W, "W")
    if W.shape[0] != W.shape[1]:
        raise ValueError("square toy W required")
    d = W.shape[0]
    u = np.asarray(u,dtype=np.float64)
    if u.shape != (d,) or np.linalg.norm(u) < 1e-10:
        raise ValueError("bad output direction")
    u = u/np.linalg.norm(u)
    S = list(range(d)) if support is None else list(support)
    if not S or len(set(S)) != len(S) or any(j<0 or j>=d for j in S):
        raise ValueError("bad support")
    X = _matrix(reference,"reference",d)
    if diff_eps <= 0 or preservation < 0:
        raise ValueError("invalid parameters")
    r = commutator(transition,W,states,evidence_a,evidence_b).ravel()
    J = np.zeros((len(r),len(S)))
    for i,j in enumerate(S):
        e=np.zeros(d);e[j]=diff_eps
        plus=commutator(transition,W+np.outer(u,e),states,evidence_a,evidence_b).ravel()
        minus=commutator(transition,W-np.outer(u,e),states,evidence_a,evidence_b).ravel()
        J[:,i]=(plus-minus)/(2*diff_eps)
    if np.linalg.norm(J) < 1e-10:
        raise ValueError("unidentifiable edit: zero operator sensitivity")
    C = X[:,S].T@X[:,S]/len(X) + 1e-6*np.eye(len(S))
    z = -np.linalg.solve(J.T@J + preservation*C, J.T@r)
    full_z = np.zeros(d); full_z[S]=z
    D = np.outer(u,full_z)
    post=commutator(transition,W+D,states,evidence_a,evidence_b).ravel()
    return D,{"evidence":"finite_difference_operator_replay_synthetic_or_real_requires_model",
              "support":S, "rank":int(np.linalg.matrix_rank(D)),
              "pre_l2":float(np.linalg.norm(r)), "post_l2":float(np.linalg.norm(post)),
              "edit_fro":float(np.linalg.norm(D)),
              "reference_quadratic_cost":float(np.mean(np.sum((X@D.T)**2,axis=1)))}


def affine_evidence_transition(h,e,W):
    """T_e(h)=(I+W)h+e, batched. Linear test case, not a Qwen operator."""
    return h @ (np.eye(W.shape[0])+W).T + e


def toy_demo(seed=20261104):
    rng=np.random.default_rng(seed)
    d=4; ncal=48; ntest=128
    U=np.array([0.,1.,0.,0.])
    W_target=np.diag([0.,0.03,-0.01,0.08])
    W_base=W_target+0.22*np.outer(U,[1.,0.,0.,0.])
    def samples(n):
        h=rng.normal(size=(n,d))
        a=np.zeros((n,d));b=np.zeros((n,d))
        a[:,0]=rng.normal(size=n)+1.
        b[:,0]=rng.normal(size=n)-1.
        return h,a,b
    cal=samples(ncal);test=samples(ntest)
    X=rng.normal(size=(192,d));X[:,0]*=0.2
    D,fit=finite_diff_rankone_edit(affine_evidence_transition,W_base,*cal,U,X,
                                  support=[0,1],preservation=1e-8)
    pre=commutator(affine_evidence_transition,W_base,*test)
    post=commutator(affine_evidence_transition,W_base+D,*test)
    # Matched Frobenius norm random directions as a negative control, no tuning.
    random_gaps=[]
    for _ in range(100):
        R=rng.normal(size=W_base.shape);R*=np.linalg.norm(D)/np.linalg.norm(R)
        rand=commutator(affine_evidence_transition,W_base+R,*test)
        random_gaps.append(float(np.mean(np.linalg.norm(rand,axis=1))))
    gap_pre=float(np.mean(np.linalg.norm(pre,axis=1)))
    gap_post=float(np.mean(np.linalg.norm(post,axis=1)))
    return {"evidence":"synthetic_affine_evidence_operator_only", "seed":seed,
            "calibration_cases":ncal,"untouched_test_cases":ntest,
            "pre_test_commutator_l2_mean":gap_pre,"post_test_commutator_l2_mean":gap_post,
            "reduction_fraction":(gap_pre-gap_post)/gap_pre,
            "random_matched_fro_median_test_gap":float(np.median(random_gaps)),
            "fraction_random_worse_than_edit":float(np.mean(np.array(random_gaps)>gap_post)),
            "edit_rank":int(np.linalg.matrix_rank(D)),"edit_fro":float(np.linalg.norm(D)),
            "distance_to_planted_commuting_target_before":float(np.linalg.norm(W_base-W_target)),
            "distance_to_planted_commuting_target_after":float(np.linalg.norm(W_base+D-W_target)),
            "calibration_diagnostics":fit,"real_qwen_or_hf_data_loaded":False}


if __name__=="__main__":
    print(json.dumps(toy_demo(),indent=2))
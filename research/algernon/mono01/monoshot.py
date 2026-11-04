"""MONO-01 reference: closed-form minimum-energy rank-limited weight edits.

Independent NumPy implementation of the workorder's mathematics (not the ChatGPT zip, which is not on disk).
Setting: linear receiver y = W x, keys K [n, m] (columns are fault activations), desired output corrections
V [d, m], reference second moment C = lam*I + X^T X / N (uncentered, positive definite).

  min_D tr(D C D^T)  s.t.  D K = V      =>  D* = V (K^T C^-1 K)^-1 K^T C^-1,   cost tr(V (K^T C^-1 K)^-1 V^T).
"""
from __future__ import annotations
from itertools import combinations
import numpy as np


def second_moment(X: np.ndarray, lam: float) -> np.ndarray:
    return lam * np.eye(X.shape[1]) + X.T @ X / X.shape[0]


def woodbury_cinv(X: np.ndarray, lam: float, B: np.ndarray) -> np.ndarray:
    """C^-1 B for C = lam I + X^T X / N without forming C: lam^-1 [B - X^T (N lam I + X X^T)^-1 X B]."""
    N = X.shape[0]
    inner = np.linalg.solve(N * lam * np.eye(N) + X @ X.T, X @ B)
    return (B - X.T @ inner) / lam


def min_energy_edit(K: np.ndarray, V: np.ndarray, C: np.ndarray) -> np.ndarray:
    CK = np.linalg.solve(C, K)                                   # C^-1 K
    return V @ np.linalg.solve(K.T @ CK, CK.T)                   # V (K^T C^-1 K)^-1 K^T C^-1


def edit_cost(D: np.ndarray, C: np.ndarray) -> float:
    return float(np.trace(D @ C @ D.T))


def optimal_cost(K, V, C) -> float:
    CK = np.linalg.solve(C, K)
    return float(np.trace(V @ np.linalg.solve(K.T @ CK, V.T)))


def q_support(k: np.ndarray, C: np.ndarray, S) -> float:
    """q(S) = k_S^T C_SS^-1 k_S for a single key; the optimal disturbance with support S is ||v||^2 / q(S)."""
    S = list(S)
    if not S:
        return 0.0
    return float(k[S] @ np.linalg.solve(C[np.ix_(S, S)], k[S]))


def gram_support(K: np.ndarray, C: np.ndarray, S) -> np.ndarray:
    S = list(S)
    return K[S].T @ np.linalg.solve(C[np.ix_(S, S)], K[S])


def support_cost(K, V, C, S) -> float:
    """Optimal cost when the edit may only touch input columns S: tr(V (K_S^T C_SS^-1 K_S)^-1 V^T)."""
    S = list(S)
    if len(S) < K.shape[1]:
        return float("inf")
    return float(np.trace(V @ np.linalg.solve(gram_support(K, C, S), V.T)))


def support_edit(K, V, C, S) -> np.ndarray:
    S = list(S)
    Dfull = np.zeros((V.shape[0], K.shape[0]))
    CSK = np.linalg.solve(C[np.ix_(S, S)], K[S])
    Dfull[:, S] = V @ np.linalg.solve(K[S].T @ CSK, CSK.T)
    return Dfull


def schur_gain(k: np.ndarray, C: np.ndarray, S, j: int) -> float:
    """Exact q(S u {j}) - q(S) = (k_j - C_jS C_SS^-1 k_S)^2 / (C_jj - C_jS C_SS^-1 C_Sj)."""
    S = list(S)
    if not S:
        return float(k[j] ** 2 / C[j, j])
    CSS_inv_kS = np.linalg.solve(C[np.ix_(S, S)], k[S])
    num = (k[j] - C[j, S] @ CSS_inv_kS) ** 2
    den = C[j, j] - C[j, S] @ np.linalg.solve(C[np.ix_(S, S)], C[S, j])
    return float(num / den)


def select_greedy(k, C, budget):
    S = []
    for _ in range(budget):
        rest = [j for j in range(len(k)) if j not in S]
        S.append(max(rest, key=lambda j: schur_gain(k, C, S, j)))
    return S, q_support(k, C, S)


def select_beam(k, C, budget, width=8):
    beam = [((), 0.0)]
    for _ in range(budget):
        cand = {}
        for S, _q in beam:
            for j in range(len(k)):
                if j in S:
                    continue
                T = tuple(sorted(S + (j,)))
                if T not in cand:
                    cand[T] = q_support(k, C, T)
        beam = sorted(cand.items(), key=lambda t: -t[1])[:width]
    S, q = beam[0]
    return list(S), q


def select_exhaustive(k, C, budget):
    best = max(combinations(range(len(k)), budget), key=lambda S: q_support(k, C, S))
    return list(best), q_support(k, C, best)


def to_bf16(x: np.ndarray) -> np.ndarray:
    """Round float32 to bfloat16 (round-to-nearest-even), returned as float32."""
    u = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    rnd = ((u >> 16) & 1) + 0x7FFF
    return ((u + rnd) & 0xFFFF0000).astype(np.uint32).view(np.float32)


def eigen_shift_edit(A: np.ndarray, r: np.ndarray, l: np.ndarray, mu_target: float, mu: float) -> np.ndarray:
    """Rank-one dA = (mu_target - mu) r l^T / (l^T r) moves a simple eigenvalue mu of A (A r = mu r, l^T A = mu l^T)."""
    return (mu_target - mu) * np.outer(r, l) / float(l @ r)


def robust_mixture(K, V, Cs, iters=200):
    """min_D max_e tr(D C_e D^T) s.t. D K = V equals max over simplex weights w of g(w)=tr(V (K^T C_w^-1 K)^-1 V^T)
    (minimax: cost is convex in D and linear in the mixture). Exponentiated-gradient ascent; envelope gradient is
    tr(D*(w) C_e D*(w)^T)."""
    E = len(Cs)
    w = np.full(E, 1.0 / E)
    best = (-np.inf, w.copy())
    for t in range(iters):
        C = sum(wi * Ci for wi, Ci in zip(w, Cs))
        D = min_energy_edit(K, V, C)
        costs = np.array([edit_cost(D, Ci) for Ci in Cs])
        g = float(np.sum(w * costs))
        if g > best[0]:
            best = (g, w.copy())
        w = w * np.exp(0.5 / np.sqrt(t + 1) * (costs / costs.max() - 1.0))
        w /= w.sum()
    w = best[1]
    D = min_energy_edit(K, V, sum(wi * Ci for wi, Ci in zip(w, Cs)))
    return D, w, max(edit_cost(D, Ci) for Ci in Cs)

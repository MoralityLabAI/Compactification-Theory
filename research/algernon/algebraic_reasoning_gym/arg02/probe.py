"""Linear probes on recurrent states, numpy only.

Standardise, project to k principal components (fitted on the training
states), multinomial logistic regression with L2 penalty lam, fitted by
deterministic full-batch gradient descent. (k, lam) is chosen on a separate
selection split; the test split is scored once per step.
"""
from __future__ import annotations

import numpy as np

GRID_K = (8, 16, 32)
GRID_LAM = (1e-3, 1e-2, 1e-1, 1.0)
ITERS = 400
LR = 0.5
PERMUTATIONS = 200
PERM_SEED = 2026111106


class Probe:
    def __init__(self, k, lam, classes):
        self.k, self.lam, self.classes = k, lam, list(classes)

    def fit(self, x, y):
        x = np.asarray(x, dtype=np.float64)
        self.mu = x.mean(0)
        self.sd = x.std(0) + 1e-6
        z = (x - self.mu) / self.sd
        _, _, vt = np.linalg.svd(z, full_matrices=False)
        k = min(self.k, vt.shape[0])
        self.proj = vt[:k].T
        f = z @ self.proj
        self.fs = f.std(0) + 1e-6
        f = f / self.fs
        n, c = len(y), len(self.classes)
        yi = np.array([self.classes.index(v) for v in y])
        onehot = np.zeros((n, c))
        onehot[np.arange(n), yi] = 1
        self.w = np.zeros((f.shape[1], c))
        self.b = np.zeros(c)
        for _ in range(ITERS):
            logits = f @ self.w + self.b
            logits -= logits.max(1, keepdims=True)
            p = np.exp(logits)
            p /= p.sum(1, keepdims=True)
            g = p - onehot
            self.w -= LR * (f.T @ g / n + self.lam * self.w)
            self.b -= LR * g.mean(0)
        return self

    def predict(self, x):
        f = ((np.asarray(x, dtype=np.float64) - self.mu) / self.sd) @ self.proj / self.fs
        return [self.classes[i] for i in (f @ self.w + self.b).argmax(1)]


def accuracy(pred, y):
    return float(np.mean([a == b for a, b in zip(pred, y)])) if len(y) else 0.0


def select(train_x, train_y, sel_x, sel_y, classes):
    best = None
    for k in GRID_K:
        for lam in GRID_LAM:
            acc = accuracy(Probe(k, lam, classes).fit(train_x, train_y).predict(sel_x), sel_y)
            if best is None or acc > best[0] + 1e-12:
                best = (acc, k, lam)
    return {"selection_accuracy": best[0], "k": best[1], "lam": best[2]}


def evaluate(train_x, train_y, sel_x, sel_y, tests, classes, *, permutations=PERMUTATIONS, seed=PERM_SEED):
    """tests: {name: (x, y)}. Returns selected settings, test accuracies,
    majority baseline and a label-permutation null for each test split."""
    choice = select(train_x, train_y, sel_x, sel_y, classes)
    probe = Probe(choice["k"], choice["lam"], classes).fit(train_x, train_y)
    values, counts = np.unique(np.asarray(train_y), return_counts=True)
    majority = values[counts.argmax()]
    rng = np.random.default_rng(seed)
    null = {name: [] for name in tests}
    for _ in range(permutations):
        perm = list(rng.permutation(np.asarray(train_y)))
        pp = Probe(choice["k"], choice["lam"], classes).fit(train_x, perm)
        for name, (x, y) in tests.items():
            null[name].append(accuracy(pp.predict(x), y))
    out = {"settings": choice, "majority_label": str(majority), "tests": {}}
    for name, (x, y) in tests.items():
        acc = accuracy(probe.predict(x), y)
        nd = np.array(null[name])
        out["tests"][name] = {"n": len(y), "accuracy": acc,
                              "majority_accuracy": accuracy([majority] * len(y), y),
                              "permutation_mean": float(nd.mean()) if len(nd) else None,
                              "permutation_p": float((1 + (nd >= acc).sum()) / (1 + len(nd))) if len(nd) else None}
    return out

"""Recursive learning gym: select -> execute -> verify -> record -> update -> confirm -> select.

The cycle is model-agnostic: the trainer, the specialists and the evaluators are
callables, so the same loop drives the 27B adapter runs and cheap tests. What
this module owns is the bookkeeping that keeps the loop honest:

- training feedback (dev) and confirmatory evaluation are separate objects;
  confirmatory sets live in a `Vault` with a fixed access budget and a log of
  every read, and nothing in the loop's reward path can read them;
- dev reuse goes through `Thresholdout` (Dwork et al. 2015), so repeated
  selection against the same dev items does not overfit them silently;
- a gain counts only if a paired bootstrap CI over fresh dev items excludes 0
  (`paired_gain`) and, when dev items are reused, also exceeds the phantom-gain
  bound for G generations of K candidates on n items, sqrt(ln(GK+1)/(2n))
  (EvalMath L5, `phantom_gain_bound`), which guards against spurious gains;
- confirmatory reads report a mean rounded to `levels` values, so each read
  leaks at most log2(levels) bits (EvalMath L1) and the ledger totals them;
- `contamination` flags long n-gram overlap between training rows and sealed items;
- `RegressionGuard` rejects an update that drops any anchor task beyond tolerance;
- quantizer artifacts: an update can be required to replicate on a second
  runtime format (PTQ1_0 and PQ2_0 hold the same trits) before it is kept.

The controller baseline is LinUCB over task features (`LinUCB`); it learns which
task to train next from verified dev gains only.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np


# ---------------------------------------------------------------- ledger
class Ledger:
    """Append-only JSONL record of every cycle event."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, kind: str, **rec) -> dict:
        rec = {"kind": kind, "t": time.time(), **rec}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
        return rec

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(l) for l in self.path.read_text(encoding="utf-8").split("\n") if l.strip()]


# ---------------------------------------------------------------- confirmatory vault
class VaultBudgetExceeded(RuntimeError):
    pass


class Vault:
    """Sealed confirmatory items: hash-pinned, read only through `confirm`, budgeted."""

    def __init__(self, name: str, items: list[dict], budget: int, ledger: Ledger, levels: int = 21):
        self.name, self._items, self.budget, self.ledger = name, items, budget, ledger
        self.levels = levels
        self.sha256 = hashlib.sha256(json.dumps(items, sort_keys=True).encode()).hexdigest()
        self.reads = sum(1 for r in ledger.read() if r["kind"] == "vault_read" and r["vault"] == self.name)

    def texts(self) -> list[str]:
        """Item text for contamination checks only (no labels leave the vault)."""
        return [json.dumps({k: v for k, v in it.items() if k not in ("answer", "target")}, sort_keys=True)
                for it in self._items]

    def confirm(self, evaluate: Callable[[list[dict]], list[float]], purpose: str) -> dict:
        """Score the sealed items once; only a mean rounded to `levels` values leaves the vault."""
        if self.reads >= self.budget:
            raise VaultBudgetExceeded(f"{self.name}: confirmatory budget {self.budget} used")
        self.reads += 1
        scores = evaluate(self._items)
        step = 1.0 / (self.levels - 1)
        reported = round(float(np.mean(scores)) / step) * step
        bits = math.log2(self.levels)
        rec = self.ledger.write("vault_read", vault=self.name, sha256=self.sha256, read=self.reads,
                                budget=self.budget, purpose=purpose, reported_mean=reported, n=len(scores),
                                bits_this_read=bits, bits_total=bits * self.reads)
        return {"reported_mean": reported, "n": len(scores), "read": self.reads, "bits_total": rec["bits_total"]}


# ---------------------------------------------------------------- dev reuse
class Thresholdout:
    """Reusable holdout: answer a dev query from train unless train and holdout disagree."""

    def __init__(self, threshold: float = 0.04, sigma: float = 0.01, budget: int = 50, seed: int = 0):
        self.t, self.sigma, self.budget = threshold, sigma, budget
        self.rng = np.random.default_rng(seed)
        self.overfit_signals = 0

    def query(self, train_mean: float, holdout_mean: float) -> float:
        if self.budget <= 0:
            raise RuntimeError("thresholdout budget exhausted")
        if abs(train_mean - holdout_mean) > self.t + self.rng.laplace(0, 2 * self.sigma):
            self.budget -= 1
            self.overfit_signals += 1
            return holdout_mean + self.rng.laplace(0, self.sigma)
        return train_mean


def phantom_gain_bound(n: int, generations: int, candidates: int) -> float:
    """EvalMath L5: largest expected phantom gain from keeping the best of K candidates for G
    generations on one reused set of n items scored in [0, 1], with no true differences."""
    return math.sqrt(math.log(generations * candidates + 1) / (2 * n))


def paired_gain(before: list[float], after: list[float], n_boot: int = 5000, seed: int = 0,
                reuse: tuple[int, int] | None = None) -> dict:
    """Mean paired difference with a bootstrap 95% CI; `verified` if the CI lower end is above 0
    and, for items reused over (G generations, K candidates), above the phantom-gain bound."""
    d = np.asarray(after, dtype=float) - np.asarray(before, dtype=float)
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(1)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    floor = phantom_gain_bound(len(d), *reuse) if reuse else 0.0
    return {"n": int(len(d)), "delta": float(d.mean()), "ci95": [float(lo), float(hi)],
            "phantom_floor": floor, "verified": bool(lo > floor)}


# ---------------------------------------------------------------- contamination
def ngrams(text: str, n: int) -> set[str]:
    toks = text.split()
    return {" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def contamination(train_texts: list[str], sealed_texts: list[str], n: int = 13) -> dict:
    """Sealed items sharing any n-gram (whitespace tokens) or an exact hash with training rows."""
    grams = set()
    hashes = set()
    for t in train_texts:
        grams |= ngrams(t, n)
        hashes.add(hashlib.sha256(t.strip().encode()).hexdigest())
    hit_idx = [i for i, t in enumerate(sealed_texts)
               if hashlib.sha256(t.strip().encode()).hexdigest() in hashes or (ngrams(t, n) & grams)]
    return {"n": n, "sealed": len(sealed_texts), "contaminated": len(hit_idx), "indices": hit_idx[:50]}


# ---------------------------------------------------------------- regression
@dataclass
class RegressionGuard:
    tolerance: float = 0.02          # absolute drop allowed per anchor task
    baseline: dict[str, float] = field(default_factory=dict)

    def check(self, scores: dict[str, float]) -> dict:
        drops = {k: self.baseline[k] - v for k, v in scores.items() if k in self.baseline}
        bad = {k: d for k, d in drops.items() if d > self.tolerance}
        return {"ok": not bad, "drops": drops, "violations": bad}


# ---------------------------------------------------------------- controller
class LinUCB:
    """Disjoint LinUCB over arms (task families) with per-arm context features."""

    def __init__(self, arms: list[str], dim: int, alpha: float = 1.0):
        self.arms, self.alpha = arms, alpha
        self.A = {a: np.eye(dim) for a in arms}
        self.b = {a: np.zeros(dim) for a in arms}

    def scores(self, ctx: dict[str, np.ndarray]) -> dict[str, float]:
        out = {}
        for a in self.arms:
            ainv = np.linalg.inv(self.A[a])
            theta = ainv @ self.b[a]
            x = ctx[a]
            out[a] = float(theta @ x + self.alpha * math.sqrt(x @ ainv @ x))
        return out

    def select(self, ctx: dict[str, np.ndarray]) -> str:
        s = self.scores(ctx)
        return max(self.arms, key=lambda a: (s[a], -self.arms.index(a)))

    def update(self, arm: str, x: np.ndarray, reward: float) -> None:
        self.A[arm] += np.outer(x, x)
        self.b[arm] += reward * x

    def state(self) -> dict:
        return {"A": {a: m.tolist() for a, m in self.A.items()}, "b": {a: v.tolist() for a, v in self.b.items()}}


# ---------------------------------------------------------------- the cycle
@dataclass
class Task:
    name: str
    features: Callable[[], np.ndarray]              # controller context (from the ledger/probes)
    train: Callable[[int], dict]                    # executes a specialist/adapter update, returns info
    dev_eval: Callable[[int], list[float]]          # fresh dev sample for round r (training feedback)
    revert: Callable[[], None]                      # undo the last update
    replicate: Callable[[], bool] | None = None     # second-format check (quantizer artifacts)


def run_cycle(tasks: dict[str, Task], controller: LinUCB, anchors: Callable[[], dict[str, float]],
              guard: RegressionGuard, ledger: Ledger, rounds: int, confirm: Callable[[], dict] | None = None,
              confirm_every: int = 0, reject_cost: float = 0.02) -> list[dict]:
    """Select a task, train, verify on fresh dev items, keep or revert, record, update the controller.

    `confirm` (a Vault read) runs only every `confirm_every` rounds and its result never
    reaches the controller: it is reported, not optimized. A reverted update earns
    -reject_cost (the compute it burned), so a harmful arm ranks below a neutral one.
    """
    out = []
    for r in range(rounds):
        ctx = {k: t.features() for k, t in tasks.items()}
        arm = controller.select(ctx)
        t = tasks[arm]
        before = t.dev_eval(r)
        info = t.train(r)
        after = t.dev_eval(r)
        gain = paired_gain(before, after, seed=r)
        reg = guard.check(anchors())
        rep = t.replicate() if (t.replicate and gain["verified"]) else None
        keep = gain["verified"] and reg["ok"] and rep is not False
        if not keep:
            t.revert()
        reward = gain["delta"] if keep else -reject_cost
        controller.update(arm, ctx[arm], reward)
        rec = ledger.write("cycle", round=r, arm=arm, info=info, gain=gain, regression=reg,
                           replicated=rep, kept=keep, reward=reward,
                           scores=controller.scores(ctx))
        if keep:
            guard.baseline.update(anchors())
        if confirm and confirm_every and (r + 1) % confirm_every == 0:
            rec["confirm"] = ledger.write("confirm", round=r, **confirm())
        out.append(rec)
    return out

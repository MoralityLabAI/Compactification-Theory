"""Representation probes and causal interventions on the packed model.

- `capture`: residual-stream vectors at chosen layers and token positions, for
  base and adapted weights on identical token ids (LoRA can be toggled in place).
- `probe_cv`: ridge/logistic probe with group-held-out cross-validation, so a
  probe trained on some task families is scored on families it never saw
  (transfer), against a label-permutation baseline. This is correlational.
- `steer`: adds alpha * direction to one layer's output and measures the change
  in target NLL, against matched random directions. This is the causal test;
  a probe direction counts as causally relevant only if its effect exceeds the
  random-direction distribution.
"""
from __future__ import annotations

import contextlib

import numpy as np
import torch
from torch.nn import functional as F


@contextlib.contextmanager
def lora_disabled(model):
    saved = []
    for m in model.packed_linears():
        if m.lora_A is not None:
            saved.append((m, m.lora_scale))
            m.lora_scale = 0.0
    try:
        yield
    finally:
        for m, s in saved:
            m.lora_scale = s


@torch.no_grad()
def capture(model, examples, layers: tuple[int, ...], position: str = "last_prompt") -> dict[int, np.ndarray]:
    """[n_examples, n_embd] per layer at the last prompt token (where the answer is decided)."""
    out = {il: [] for il in layers}
    for e in examples:
        ids = torch.tensor(e.prompt_ids if position == "last_prompt" else e.prompt_ids + e.target_ids, device="cuda")
        _, caps = model.hidden(ids, capture=layers)
        for il in layers:
            out[il].append(caps[il][-1].float().cpu().numpy())
    return {il: np.stack(v) for il, v in out.items()}


def _ridge_fit(x, y, lam):
    xm, ym = x.mean(0), y.mean()
    xc = x - xm
    w = np.linalg.solve(xc.T @ xc + lam * np.eye(x.shape[1]), xc.T @ (y - ym))
    return w, ym - xm @ w


def probe_cv(x: np.ndarray, y: np.ndarray, groups: np.ndarray, lam: float = 10.0, n_perm: int = 50,
             seed: int = 0) -> dict:
    """Leave-one-group-out ridge probe: R^2 (continuous y) on held-out groups vs permuted labels."""
    def score(yv):
        preds = np.zeros_like(yv, dtype=float)
        for g in np.unique(groups):
            tr, te = groups != g, groups == g
            w, b = _ridge_fit(x[tr], yv[tr], lam)
            preds[te] = x[te] @ w + b
        ss = ((yv - yv.mean()) ** 2).sum()
        return 1 - ((yv - preds) ** 2).sum() / ss if ss > 0 else 0.0

    rng = np.random.default_rng(seed)
    real = score(y.astype(float))
    null = [score(rng.permutation(y).astype(float)) for _ in range(n_perm)]
    w, _ = _ridge_fit(x, y.astype(float), lam)
    return {"r2_heldout_groups": float(real), "null_mean": float(np.mean(null)),
            "null_q95": float(np.quantile(null, 0.95)), "beats_null": bool(real > np.quantile(null, 0.95)),
            "direction": w / (np.linalg.norm(w) + 1e-12), "groups": int(len(np.unique(groups)))}


@torch.no_grad()
def steer(model, examples, layer: int, direction: np.ndarray, alpha: float, n_random: int = 8, seed: int = 0) -> dict:
    """Mean change in target NLL when alpha * unit direction is added after `layer`, vs random unit directions."""
    from .data import collate
    target = model.layers[layer]

    def nll_with(vec):
        handle = None
        if vec is not None:
            v = torch.tensor(vec, device="cuda", dtype=model.dtype)
            handle = target.register_forward_hook(lambda mod, inp, out: out + alpha * v)
        try:
            tot, n = 0.0, 0
            for e in examples:
                ids, lab = collate([e], 0, "cuda")
                h, _ = model.hidden(ids)
                sel = lab != -100
                tot += float(F.cross_entropy(model.logits(h[sel]), lab[sel], reduction="sum"))
                n += int(sel.sum())
            return tot / n
        finally:
            if handle:
                handle.remove()

    base = nll_with(None)
    d = direction / np.linalg.norm(direction)
    eff = nll_with(d) - base
    rng = np.random.default_rng(seed)
    rand = []
    for _ in range(n_random):
        r = rng.standard_normal(d.shape)
        rand.append(nll_with(r / np.linalg.norm(r)) - base)
    return {"layer": layer, "alpha": alpha, "base_nll": base, "effect": eff, "random_effects": rand,
            "exceeds_random": bool(abs(eff) > max(abs(x) for x in rand))}

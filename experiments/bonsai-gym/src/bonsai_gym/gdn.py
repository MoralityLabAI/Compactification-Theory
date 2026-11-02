"""Gated delta rule (Qwen3-Next / Qwen3.5 linear attention), pure PyTorch.

`chunk_gated_delta_rule` follows `torch_chunk_gated_delta_rule` in
transformers 5.17.0 (models/qwen3_5/modeling_qwen3_5.py, Apache-2.0), reduced
to the no-cache training case: q, k and v already have the same head count, and
q/k are already L2-normalized. `recurrent_gated_delta_rule` is the per-token
reference the tests compare it against.

Shapes: q, k [B, T, H, Dk]; v [B, T, H, Dv]; g, beta [B, T, H]. State decays
by exp(g) per token, so g <= 0. q is scaled by Dk**-0.5 inside, as in the fork
(delta-net-base.cpp) and in transformers.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F


def recurrent_gated_delta_rule(q, k, v, g, beta):
    dtype = q.dtype
    q, k, v, g, beta = (t.transpose(1, 2).float() for t in (q, k, v, g, beta))
    b, h, t, dk = k.shape
    q = q * dk ** -0.5
    state = q.new_zeros(b, h, dk, v.shape[-1])
    out = torch.empty_like(v)
    for i in range(t):
        state = state * g[:, :, i].exp()[..., None, None]
        mem = (state * k[:, :, i, :, None]).sum(-2)
        delta = (v[:, :, i] - mem) * beta[:, :, i, None]
        state = state + k[:, :, i, :, None] * delta[..., None, :]
        out[:, :, i] = (state * q[:, :, i, :, None]).sum(-2)
    return out.transpose(1, 2).to(dtype)


def chunk_gated_delta_rule(q, k, v, g, beta, chunk_size: int = 64):
    dtype = q.dtype
    q, k, v, beta, g = (t.transpose(1, 2).float().contiguous() for t in (q, k, v, beta, g))
    b, h, t, dk = k.shape
    dv = v.shape[-1]
    q = q * dk ** -0.5
    pad = (chunk_size - t % chunk_size) % chunk_size
    q, k, v = (F.pad(x, (0, 0, 0, pad)) for x in (q, k, v))
    beta, g = (F.pad(x, (0, pad)) for x in (beta, g))
    n = (t + pad) // chunk_size

    v_beta = v * beta.unsqueeze(-1)
    k_beta = k * beta.unsqueeze(-1)
    q, k, k_beta, v_beta = (x.reshape(b, h, n, chunk_size, x.shape[-1]) for x in (q, k, k_beta, v_beta))
    g = g.reshape(b, h, n, chunk_size)
    upper = torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device).triu(1)

    cum = g.cumsum(dim=-1)
    pair = (cum.unsqueeze(-1) - cum.unsqueeze(-2)).masked_fill(upper, float("-inf")).exp()
    ut = (k_beta @ k.transpose(-1, -2)) * pair
    intra = (q @ k.transpose(-1, -2)) * pair
    decayed_k_beta = k_beta * cum.exp().unsqueeze(-1)
    new_v = torch.linalg.solve_triangular(ut, v_beta, upper=False, unitriangular=True)
    k_cum = torch.linalg.solve_triangular(ut, decayed_k_beta, upper=False, unitriangular=True)

    state = q.new_zeros(b, h, dk, dv)
    q = q * cum.exp().unsqueeze(-1)
    k = k * (cum[..., -1:] - cum).exp().unsqueeze(-1)
    chunk_decay = cum[..., -1].exp()[..., None, None]
    outs = []
    for i in range(n):
        v_new = new_v[:, :, i] - k_cum[:, :, i] @ state
        outs.append(q[:, :, i] @ state + intra[:, :, i] @ v_new)
        state = state * chunk_decay[:, :, i] + k[:, :, i].transpose(-1, -2) @ v_new
    out = torch.stack(outs, dim=2).reshape(b, h, -1, dv)[:, :, :t]
    return out.transpose(1, 2).contiguous().to(dtype)

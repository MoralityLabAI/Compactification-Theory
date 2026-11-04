"""Minimal architecture mirror for an external 44,001-parameter Sudoku recurrence.

Copyright (c) 2026 Naga Satya Phanindra Vallabhaneni. MIT License.
Source: NagaSatyaPhanindraVallabhaneni/tiny-recursive-model, commit f0406195.
Model equations and state_dict field names are reproduced verbatim in structure.
"""
import torch
from torch import nn


class ReasoningBlock(nn.Module):
    def __init__(self, d_model=48, n_tokens=81, d_tok=162, d_ff=128):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.tok_mix = nn.Sequential(nn.Linear(n_tokens, d_tok), nn.GELU(), nn.Linear(d_tok, n_tokens))
        self.ln2 = nn.LayerNorm(d_model)
        self.chan_mix = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, z):
        z = self.ln1(z + self.tok_mix(z.transpose(1, 2)).transpose(1, 2))
        return self.ln2(z + self.chan_mix(z))


class TinyRecursiveModel(nn.Module):
    def __init__(self, d_model=48, n_steps=4):
        super().__init__()
        self.n_steps = n_steps
        self.token_emb = nn.Embedding(10, d_model)
        self.pos_emb = nn.Embedding(81, d_model)
        self.block = ReasoningBlock(d_model)
        self.head = nn.Linear(d_model, 10)

    def initial_state(self, x):
        b = x.shape[0]
        pos = torch.arange(81, device=x.device).unsqueeze(0).expand(b, -1)
        return self.token_emb(x) + self.pos_emb(pos)

    def reason_states(self, x):
        z = self.initial_state(x)
        states = [z]
        for _ in range(self.n_steps):
            z = self.block(z)
            states.append(z)
        return states

    def forward(self, x):
        return self.head(self.reason_states(x)[-1])

    def predict(self, x):
        with torch.no_grad():
            return self.forward(x).argmax(-1)

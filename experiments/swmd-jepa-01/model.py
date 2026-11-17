"""Shared recurrent encoders and VICReg-style JEPA objectives."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class Encoder(nn.Module):
    def __init__(self, vocab_size, dim=48):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.rnn = nn.GRU(dim, dim, batch_first=True)

    def forward(self, ids, lengths, sequence=False):
        out, _ = self.rnn(self.embedding(ids))
        if sequence:
            return out
        return out[torch.arange(len(lengths), device=ids.device), lengths - 1]


class Experiment(nn.Module):
    def __init__(self, vocab_size, n_actions, dim=48):
        super().__init__()
        self.encoder = Encoder(vocab_size, dim)
        self.lm_head = nn.Linear(dim, vocab_size)
        self.action_embedding = nn.Embedding(n_actions, dim)
        self.transition = nn.GRUCell(dim, dim)

    def next_latent(self, z, actions):
        return self.transition(self.action_embedding(actions), z)


def vicreg(a, b):
    # VICReg is a practical non-collapse proxy, not the paper's LeJEPA loss.
    invariance = F.mse_loss(a, b)
    variance = sum(torch.relu(1.0 - torch.sqrt(z.var(dim=0, unbiased=False) + 1e-4)).mean()
                   for z in (a, b)) / 2
    covariance = a.new_zeros(())
    for z in (a, b):
        z = z - z.mean(0)
        c = z.T @ z / max(len(z) - 1, 1)
        covariance = covariance + (c.square().sum() - c.diag().square().sum()) / z.shape[1]
    return 25 * invariance + 25 * variance + covariance / 2


def pad(batch, device):
    lengths = torch.tensor([len(x) for x in batch], device=device)
    out = torch.zeros((len(batch), int(lengths.max())), dtype=torch.long, device=device)
    for i, x in enumerate(batch):
        out[i, :len(x)] = torch.tensor(x, device=device)
    return out, lengths

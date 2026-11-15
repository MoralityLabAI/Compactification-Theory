"""COE-04: a small trained tied-attention looper, frozen before probe fitting.

This is a CPU-only reproduction cell using the visit-state equations of
LoopedTransformers/research_gym/neural/looped_decoder.py (one physical block,
four unrolled visits, causal two-token inputs). It trains a deliberately small
synthetic copy-first-token task, then writes literal frozen hidden states.

The native Qwen/Nanbeige/TRM pretrained checkpoint weights are not included.
The probes therefore test a locally trained and independently frozen small
transformer, not those external pretrained checkpoints.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

VOCAB = 8
DIM = 24
VISITS = 4
MODEL_SEEDS = (2406, 2407, 2408)
TRAIN_STEPS = 160
LR = 0.010


class TiedCausalBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(DIM)
        self.attention = nn.MultiheadAttention(DIM, 3, bias=False, batch_first=True)
        self.norm2 = nn.LayerNorm(DIM)
        self.mlp = nn.Sequential(
            nn.Linear(DIM, 4 * DIM, bias=False),
            nn.GELU(),
            nn.Linear(4 * DIM, DIM, bias=False),
        )

    def forward(self, state):
        mask = torch.triu(torch.ones(2, 2, device=state.device, dtype=torch.bool), diagonal=1)
        h = self.norm1(state)
        att = self.attention(h, h, h, attn_mask=mask, need_weights=False)[0]
        state = state + 0.7 * att
        return state + 0.7 * self.mlp(self.norm2(state))


class FrozenTiedLooper(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB, DIM)
        self.input_norm = nn.LayerNorm(DIM)
        self.block = TiedCausalBlock()
        self.final_norm = nn.LayerNorm(DIM)

    def forward(self, tokens, *, trace=False):
        state = self.input_norm(self.embedding(tokens))
        histories = []
        for _ in range(VISITS):
            state = self.block(state)
            histories.append(state[:, 1, :])
        logits = F.linear(self.final_norm(state[:, 1, :]), self.embedding.weight)
        return (logits, torch.stack(histories, dim=1)) if trace else logits


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def train_one(seed: int, steps: int = TRAIN_STEPS):
    torch.manual_seed(seed)
    m = FrozenTiedLooper().float()
    optimizer = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=0.0)
    examples = torch.tensor([(a, b) for a in range(VOCAB) for b in range(VOCAB)], dtype=torch.long)
    target = examples[:, 0]
    for _ in range(steps):
        output = m(examples)
        loss = F.cross_entropy(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    m.eval()
    with torch.no_grad():
        logits, trajectories = m(examples, trace=True)
        predicted = logits.argmax(dim=1)
    return m, examples, predicted, trajectories, float(loss.item())


def freeze(seed: int):
    m, examples, predicted, trajectories, loss = train_one(seed)
    if any(a != y for (a, _), y in zip(examples.tolist(), predicted.tolist())):
        raise RuntimeError("model failed pre-probe 64-cell admission gate")
    # Model parameters are permanently frozen before any head search.
    for p in m.parameters():
        p.requires_grad_(False)
    model_weights = {name: weight.detach().cpu().tolist() for name, weight in m.state_dict().items()}
    sha = hashlib.sha256(canonical(model_weights).encode()).hexdigest()
    raw = trajectories.detach().cpu().tolist()
    cells = {f"{a}:{b}": raw[8 * a + b] for a in range(8) for b in range(8)}
    return {
        "model_seed": seed,
        "model_checkpoint_sha256": sha,
        "architecture": "tied_causal_mha_1block_4visits_2tokens_d24_h3",
        "train_examples": 64,
        "train_steps": TRAIN_STEPS,
        "train_loss": loss,
        "model_predictions": predicted.tolist(),
        "activations": cells,
    }, model_weights


def freeze_all(out: Path):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows, weights = [], {}
    for seed in MODEL_SEEDS:
        row, w = freeze(seed)
        rows.append(row)
        weights[str(seed)] = w
    doc = {
        "schema": "COE-04-native-tied-loop-frozen-v1",
        "task": "copy first token, with second token as a distracting query position",
        "model_fit_pairs": "all 8 x 8 symbolic cases before probe fit",
        "probe_fit_contexts": [0, 1, 2, 3],
        "probe_heldout_contexts": [4, 5, 6, 7],
        "evaluator_graph": "K8 on source-token labels 0..7, independently from probe bits",
        "model_training_framework": {"torch": torch.__version__, "python": __import__("platform").python_version()},
        "items": rows,
    }
    out.write_text(json.dumps(doc, indent=2, sort_keys=True, allow_nan=False) + "\n")
    weights_path = out.with_name("frozen_loop_weights.json")
    weights_path.write_text(json.dumps(weights, separators=(",", ":"), sort_keys=True, allow_nan=False) + "\n")
    print("frozen",len(rows), "seeds",out.stat().st_size,weights_path.stat().st_size,[(r["model_seed"],r["model_checkpoint_sha256"][:12],round(r["train_loss"],4)) for r in rows])

if __name__ == "__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--out",type=Path,default=Path(__file__).with_name("frozen_loop_activations.json"))
    args=ap.parse_args();freeze_all(args.out)

"""Load the first N layers of the packed model, run one LoRA forward/backward, report memory.

A truncated stack has no meaningful loss; this only checks shapes, finiteness,
gradient flow and the packed-resident memory footprint per layer.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bonsai_gym.model import attach_lora, load_packed_qwen35  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--gguf", required=True)
p.add_argument("--layers", type=int, default=4)
p.add_argument("--tokens", type=int, default=64)
p.add_argument("--rank", type=int, default=8)
args = p.parse_args()

torch.manual_seed(0)
t0 = time.time()
model = load_packed_qwen35(args.gguf, n_layers=args.layers)
load_s = time.time() - t0
n = attach_lora(model, args.rank, 2 * args.rank, ["attn_qkv", "attn_gate", "ssm_out", "attn_q", "attn_k", "attn_v",
                                                   "attn_output", "ffn_gate", "ffn_up", "ffn_down"], seed=0)
for m in model.packed_linears():
    if m.lora_B is not None:
        torch.nn.init.normal_(m.lora_B, std=1e-3)
ids = torch.randint(0, 200000, (args.tokens,), device="cuda")
torch.cuda.reset_peak_memory_stats()
t1 = time.time()
h, _ = model.hidden(ids)
logits = model.logits(h[-8:])
loss = torch.nn.functional.cross_entropy(logits[:-1], ids[-7:])
loss.backward()
torch.cuda.synchronize()
grads = [p.grad for p in model.lora_parameters()]
print(json.dumps({
    "layers": args.layers, "tokens": args.tokens, "load_s": round(load_s, 1),
    "step_s": round(time.time() - t1, 2), "loss": float(loss), "lora_params": n,
    "grads_finite": all(bool(torch.isfinite(g).all()) for g in grads),
    "grad_nonzero_tensors": sum(int(g.abs().sum() > 0) for g in grads), "grad_tensors": len(grads),
    "hidden_rms": float(h.float().pow(2).mean().sqrt()),
    "allocated_mib": torch.cuda.memory_allocated() / 2**20,
    "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
    "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
}, indent=1))

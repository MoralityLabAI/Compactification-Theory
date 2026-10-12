"""Depth axis for adapter_bits.py: the same parameter budget spread over more layers of Qwen3-0.6B.

The adapter is LoRA on gate_proj, applied only at the answer position (the prompt's last token), in a
chosen set of layers. Earlier positions are then never changed by the adapter, so their keys and
values at every layer from `--first` up are cached once (in RAM, fp16 by default), together with the
residual entering layer `--first` at the last position. Each training step recomputes only the last
token through layers first..27, which is the original model's computation for that token.

Same task, items and measurements as adapter_bits.py (it reuses that module's `measure`). The
question it answers: at a fixed number of adapter parameters, does spreading them over more layers
let a trained adapter store more label bits per parameter? The theory allows it (capacity up to
W * min(b, depth)); the note's conjecture is that trained adapters do not use it.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import rotate_half

import config
from adapter_bits import ANSWERS, TEMPLATE, make_items, measure, rms, wait_for_commit
from qwen_suffix import load_model


@torch.no_grad()
def cache_suffix(model, ids: torch.Tensor, first: int, batch: int, kv_dtype: torch.dtype) -> dict:
    """Residual entering layer `first` at the last position, and keys/values of earlier positions per layer."""
    layers = range(first, len(model.model.layers))
    resid, keys, values = [], {i: [] for i in layers}, {i: [] for i in layers}
    start = time.time()
    for index, chunk in enumerate(ids.split(batch)):
        out = model(chunk, use_cache=True, output_hidden_states=True)
        resid.append(out.hidden_states[first][:, -1].clone())
        for i in layers:
            layer = out.past_key_values.layers[i]
            keys[i].append(layer.keys[:, :, :-1].to(kv_dtype))
            values[i].append(layer.values[:, :, :-1].to(kv_dtype))
        if index % 20 == 0:
            print(json.dumps({"event": "cache", "done": (index + 1) * batch, "of": len(ids),
                              "seconds": round(time.time() - start, 1)}), flush=True)
    length = ids.shape[1]
    dummy = torch.zeros(1, 1, model.config.hidden_size)
    cos, sin = model.model.rotary_emb(dummy, torch.tensor([[length - 1]]))
    return {"resid": torch.cat(resid), "keys": {i: torch.cat(v) for i, v in keys.items()},
            "values": {i: torch.cat(v) for i, v in values.items()}, "cos": cos[0, 0], "sin": sin[0, 0]}


class DeepHead(nn.Module):
    """Last-token path through layers first..27 with LoRA on gate_proj in the listed layers.

    forward takes item indices (a LongTensor), not hidden states, and looks up that item's caches.
    """

    def __init__(self, model, cache: dict, first: int, ranks: dict[int, int], readout: torch.Tensor, seed: int):
        super().__init__()
        self.cache, self.first, self.ranks = cache, first, ranks
        self.blocks = [(i, model.model.layers[i]) for i in range(first, len(model.model.layers))]  # frozen, not registered
        self.final_norm = model.model.norm.weight
        self.eps = model.config.rms_norm_eps
        self.readout = readout
        gen = torch.Generator().manual_seed(seed)
        self.factors = nn.ParameterDict()
        for layer, rank in ranks.items():
            d_out, d_in = model.model.layers[layer].mlp.gate_proj.weight.shape
            self.factors[f"L{layer}_U"] = nn.Parameter(torch.zeros(d_out, rank))
            self.factors[f"L{layer}_V"] = nn.Parameter(torch.randn(rank, d_in, generator=gen) / math.sqrt(d_in))

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        cos, sin = self.cache["cos"], self.cache["sin"]
        x = self.cache["resid"][idx]
        batch = x.shape[0]
        for i, layer in self.blocks:
            attn = layer.self_attn
            a = rms(x, layer.input_layernorm.weight, self.eps)
            q = attn.q_norm(attn.q_proj(a).view(batch, -1, attn.head_dim))
            k = attn.k_norm(attn.k_proj(a).view(batch, -1, attn.head_dim))
            v = attn.v_proj(a).view(batch, -1, attn.head_dim)
            q = q * cos + rotate_half(q) * sin
            k = k * cos + rotate_half(k) * sin
            keys = torch.cat([self.cache["keys"][i][idx].float(), k[:, :, None]], 2)
            values = torch.cat([self.cache["values"][i][idx].float(), v[:, :, None]], 2)
            heads = keys.shape[1]
            qg = q.view(batch, heads, -1, attn.head_dim)  # query heads grouped by their shared key/value head
            weights = (torch.einsum("bhgd,bhtd->bhgt", qg, keys) * attn.scaling).softmax(-1)
            mixed = torch.einsum("bhgt,bhtd->bhgd", weights, values).reshape(batch, -1)
            x = x + attn.o_proj(mixed)
            m = rms(x, layer.post_attention_layernorm.weight, self.eps)
            gate = layer.mlp.gate_proj(m)
            if i in self.ranks:
                gate = gate + (m @ self.factors[f"L{i}_V"].T) @ self.factors[f"L{i}_U"].T
            x = x + layer.mlp.down_proj(F.silu(gate) * layer.mlp.up_proj(m))
        return rms(x, self.final_norm, self.eps) @ self.readout


def parse_ranks(spec: str) -> dict[int, int]:
    """'26:1,27:1' -> {26: 1, 27: 1}"""
    return {int(layer): int(rank) for layer, rank in (part.split(":") for part in spec.split(","))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--first", type=int, default=24, help="lowest layer any spec adapts; the cache starts here")
    parser.add_argument("--specs", default="27:4;24:1,25:1,26:1,27:1",
                        help="';'-separated adapter specs, each 'layer:rank,...'")
    parser.add_argument("--items", default="8192", help="comma list of item counts")
    parser.add_argument("--pool", type=int, default=8192)
    parser.add_argument("--lr", default="0.01", help="comma list")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--optimizer", default="adam", choices=["adam", "gd"])
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--sharpness-every", type=int, default=0)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--grad-items", type=int, default=128)
    parser.add_argument("--hessian-items", type=int, default=1024)
    parser.add_argument("--hessian-iters", type=int, default=10)
    parser.add_argument("--cache-batch", type=int, default=32)
    parser.add_argument("--kv-dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--wait-commit-gb", type=float, default=12.0,
                        help="wait for this much free commit before loading the model (0 = don't wait)")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=config.OUT)
    parser.add_argument("--name", default=None)
    args = parser.parse_args()
    config.lower_priority()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    specs = [parse_ranks(spec) for spec in args.specs.split(";")]
    if min(min(spec) for spec in specs) < args.first:
        raise SystemExit("a spec adapts a layer below --first")
    item_counts = [int(x) for x in args.items.split(",")]
    if max(item_counts) > args.pool:
        raise SystemExit("--items exceeds --pool")

    out_dir = args.output / (args.name or time.strftime("deep-%Y%m%d-%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "results.jsonl"

    def log(record: dict) -> None:
        line = json.dumps(record)
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    log({"event": "config", **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
         "template": TEMPLATE, "answers": ANSWERS})
    tok = AutoTokenizer.from_pretrained(config.BASE_MODEL, revision=config.BASE_REVISION, cache_dir=config.HF_CACHE)
    ids, sigma = make_items(tok, args.pool, args.seed)
    wait_for_commit(args.wait_commit_gb)
    model = load_model("cpu")
    rows = [tok(answer, add_special_tokens=False)["input_ids"][0] for answer in ANSWERS]
    readout = (model.lm_head.weight[rows[0]] - model.lm_head.weight[rows[1]]).detach().clone()
    with torch.no_grad():
        full = model(ids[:4]).logits[:, -1]
        full_margin = full[:, rows[0]] - full[:, rows[1]]
    cache = cache_suffix(model, ids, args.first, args.cache_batch, getattr(torch, args.kv_dtype))
    # Free everything below --first and the tied embedding; only the suffix layers and readout stay in RAM.
    for index in range(args.first):
        model.model.layers[index] = nn.Identity()
    model.model.embed_tokens = None
    model.lm_head = None
    gc.collect()

    index_all = torch.arange(args.pool)
    with torch.no_grad():
        check = DeepHead(model, cache, args.first, {}, readout, args.seed)(index_all[:4])
    gap = (check - full_margin).abs().max().item()
    log({"event": "split_check", "max_abs_margin_gap": gap, "kv_dtype": args.kv_dtype, "tokens_per_prompt": ids.shape[1]})
    if gap > 2e-2:
        raise SystemExit(f"last-token recomputation disagrees with the full model by {gap}")

    measure_args = argparse.Namespace(steps=args.steps, batch=args.batch, seed=args.seed, grad_items=args.grad_items,
                                      optimizer=args.optimizer, eval_every=args.eval_every,
                                      sharpness_every=args.sharpness_every,
                                      hessian_items=args.hessian_items, hessian_iters=args.hessian_iters)
    for spec in specs:
        for items in item_counts:
            idx, ss = index_all[:items], sigma[:items]
            with torch.no_grad():
                base = DeepHead(model, cache, args.first, {}, readout, args.seed)
                base_err = (torch.cat([base(chunk) for chunk in idx.split(1024)]) * ss <= 0).float().mean().item()
            for lr in args.lr.split(","):
                module = DeepHead(model, cache, args.first, spec, readout, args.seed)
                header = {"adapter": "deep-gate", "size": ",".join(f"{k}:{v}" for k, v in spec.items()),
                          "layers": len(spec), "items": items, "lr": float(lr), "optimizer": args.optimizer}
                measure(module, header, idx, ss, base_err, measure_args, log)
    log({"event": "done"})


if __name__ == "__main__":
    main()

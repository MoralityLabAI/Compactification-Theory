"""Nanbeige4.2-3B: a natively looped decoder, run through the ARG-02 trajectory API.

Nanbeige4.2-3B was trained with `num_loops = 2`: all 22 decoder layers run, the
final RMSNorm is applied, and the result is the next loop's input; the LM head
reads the state after the last loop. `loop_idx` only selects KV-cache slots,
so without a cache the loop count can be set at inference. Here

    x   = embed(tokens)
    h_1 = norm(layers(x))                  loop 1, the initial pass
    h_t = norm(layers(h_{t-1}))            loop t; t = 2 is the trained depth
    logits = lm_head(h_t)                  readout after any loop

which is `NanbeigeModel.forward` with `num_loops = t` (checked in tests). The
open-loop, shuffle, patch and perturbation modes of `recurrence.BlockLoop`
apply unchanged with the whole loop body as the block.

The checkpoint's remote code targets transformers 4.42-4.46, so this module
runs in a 4.46 environment. Weights are streamed from the safetensors shards
straight onto the device: `from_pretrained` stages an extra CPU copy that the
shared PC's commit limit cannot hold (pattern from LoopedTransformers
`rmp_nanbeige_vpd_heatmap.py`).
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from .recurrence import BlockLoop


class BosTokenizer:
    """Wraps a fast tokenizer: optional BOS first, no other special tokens."""

    def __init__(self, tok, bos_id=None):
        self.tok, self.bos_id = tok, bos_id

    def encode(self, text, add_special_tokens=False):
        return self.tok.encode(text, add_special_tokens=False)

    def encode_prompt(self, text):
        ids = self.tok.encode(text, add_special_tokens=False)
        return ([self.bos_id] if self.bos_id is not None else []) + ids


def load_tokenizer(root, bos_id):
    from transformers import PreTrainedTokenizerFast
    return BosTokenizer(PreTrainedTokenizerFast(tokenizer_file=str(Path(root) / "tokenizer.json")), bos_id)


def build(root, device, dtype=torch.bfloat16, attn="sdpa", config_overrides=None, random_init=False):
    """Model from the snapshot's remote code; weights streamed shard by shard onto `device`."""
    from transformers import AutoConfig, AutoModelForCausalLM
    config = AutoConfig.from_pretrained(str(root), trust_remote_code=True, local_files_only=True)
    for k, v in (config_overrides or {}).items():
        setattr(config, k, v)
    config._attn_implementation = attn
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device(device):
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=True, attn_implementation=attn)
    finally:
        torch.set_default_dtype(previous)
    if random_init:
        return model.eval()
    root = Path(root)
    index = json.loads((root / "model.safetensors.index.json").read_text())
    params = model.state_dict()
    dtypes = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
    loaded = set()
    for shard in sorted(set(index["weight_map"].values())):
        with open(root / shard, "rb") as f:
            size = int.from_bytes(f.read(8), "little")
            header = json.loads(f.read(size))
            base = 8 + size
            for key, spec in header.items():
                if key == "__metadata__":
                    continue
                start, end = spec["data_offsets"]
                f.seek(base + start)
                data = bytearray(f.read(end - start))
                tensor = torch.frombuffer(data, dtype=dtypes[spec["dtype"]]).reshape(spec["shape"])
                with torch.no_grad():
                    params[key].copy_(tensor.to(device))
                loaded.add(key)
    missing = sorted(k for k in set(params) - loaded if "rotary" not in k)
    if missing:
        raise AssertionError(f"weights not loaded: {missing[:5]}")
    return model.eval()


def check_supported(config):
    """The adapter covers the plain full-stack loop only."""
    for flag in ("enable_double_loop_split", "enable_hyper_connection", "enable_mhc", "enable_depth_attention",
                 "loop_share_kv", "skip_loop_final_norm"):
        if getattr(config, flag, False):
            raise ValueError(f"Nanbeige option {flag} is not supported by the ARG-02 adapter")
    if getattr(config, "emb_neighbor_num", None) is not None:
        raise ValueError("Nanbeige n-gram embeddings are not supported by the ARG-02 adapter")
    if getattr(config, "pretraining_tp", 1) != 1:
        raise ValueError("pretraining_tp != 1 is not supported")


class NanbeigeLoop(BlockLoop):
    """Block = all decoder layers + final norm; prelude and post-block layers are empty."""

    def __init__(self, model, block=None):
        self.model = model
        self.base = model.model
        self.layers = self.base.layers
        self.cfg = model.config
        if getattr(self.cfg, "model_type", None) != "nanbeige":
            raise ValueError("NanbeigeLoop supports Nanbeige models only")
        check_supported(self.cfg)
        self.n = len(self.layers)
        if block is not None and tuple(block) != (0, self.n):
            raise ValueError("Nanbeige loops the whole stack; block must be (0, num_hidden_layers)")
        self.block = (0, self.n)

    @torch.no_grad()
    def prepare(self, ids):
        ids = ids.to(self.device)
        emb = self.base.embed_tokens(ids)
        rows, length = ids.shape
        cache_position = torch.arange(length, device=self.device)
        pos = cache_position.unsqueeze(0).expand(rows, -1)
        mask = self.base._update_causal_mask(None, emb, cache_position, None, False)
        return emb, {"mask": mask, "pos": pos, "cache_position": cache_position}

    @torch.no_grad()
    def run(self, h, lo, hi, ctx):
        if lo == hi:
            return h
        if (lo, hi) != (0, self.n):
            raise ValueError("Nanbeige runs whole loops only")
        for layer in self.layers:
            h = layer(h, attention_mask=ctx["mask"], position_ids=ctx["pos"], past_key_value=None,
                      output_attentions=False, use_cache=False, cache_position=ctx["cache_position"],
                      loop_idx=0)[0]
        return self.base.norm(h)

    @torch.no_grad()
    def readout(self, h, ctx):
        return torch.log_softmax(self.model.lm_head(h[:, -3:, :]).float(), dim=-1)

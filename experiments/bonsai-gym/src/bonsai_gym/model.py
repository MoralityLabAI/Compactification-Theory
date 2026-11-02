"""Frozen Bonsai 2 (GGUF arch `qwen35`) forward pass over packed PTQ1_0 weights.

Mirrors PrismML-Eng/llama.cpp c96baf4 `src/models/qwen35.cpp` for one
sequence without cache: the tensor layouts are the GGUF's own (tiled GDN value
heads, +1 already folded into the RMSNorm weights, ssm_a = -exp(A_log)), so no
conversion back to the Hugging Face layout is needed. Only LoRA parameters are
trainable; everything else is a buffer.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .gdn import chunk_gated_delta_rule
from .gguf_reader import GGUFFile
from .packed import PackedLinear, hadamard_matrix, inverse_rotate, rotate, split_blocks, decode

PTQ1, F32, BF16 = 143, 0, 30


@dataclass
class Qwen35Config:
    n_layer: int
    n_embd: int
    n_ff: int
    n_head: int
    n_head_kv: int
    head_dim: int
    n_rot: int
    rope_base: float
    eps: float
    full_attn_interval: int
    ssm_conv: int
    ssm_d_state: int
    ssm_n_group: int
    ssm_dt_rank: int
    ssm_d_inner: int
    vocab: int

    @classmethod
    def from_gguf(cls, md: dict) -> "Qwen35Config":
        p = "qwen35."
        return cls(
            n_layer=md[p + "block_count"], n_embd=md[p + "embedding_length"],
            n_ff=md[p + "feed_forward_length"], n_head=md[p + "attention.head_count"],
            n_head_kv=md[p + "attention.head_count_kv"], head_dim=md[p + "attention.key_length"],
            n_rot=md[p + "rope.dimension_count"], rope_base=md[p + "rope.freq_base"],
            eps=md[p + "attention.layer_norm_rms_epsilon"],
            full_attn_interval=md[p + "full_attention_interval"], ssm_conv=md[p + "ssm.conv_kernel"],
            ssm_d_state=md[p + "ssm.state_size"], ssm_n_group=md[p + "ssm.group_count"],
            ssm_dt_rank=md[p + "ssm.time_step_rank"], ssm_d_inner=md[p + "ssm.inner_size"],
            vocab=len(md["tokenizer.ggml.tokens"]),
        )

    def is_recurrent(self, il: int) -> bool:
        return (il + 1) % self.full_attn_interval != 0


def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (y * w).to(x.dtype)


def l2_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    return (xf / xf.pow(2).sum(-1, keepdim=True).sqrt().clamp_min(eps)).to(x.dtype)


def rope_neox(x: torch.Tensor, pos: torch.Tensor, n_rot: int, base: float) -> torch.Tensor:
    """Rotate the first n_rot dims of each head, pairs (i, i + n_rot/2). x [..., T, H, D]."""
    half = n_rot // 2
    inv = base ** (-torch.arange(0, half, device=x.device, dtype=torch.float32) * 2 / n_rot)
    ang = pos.float()[:, None] * inv[None, :]
    cos, sin = ang.cos()[:, None, :], ang.sin()[:, None, :]
    xf = x.float()
    x1, x2, rest = xf[..., :half], xf[..., half:n_rot], xf[..., n_rot:]
    out = torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin, rest), dim=-1)
    return out.to(x.dtype)


class Layer(nn.Module):
    def __init__(self, il: int, cfg: Qwen35Config, lin: dict[str, PackedLinear], small: dict[str, torch.Tensor]):
        super().__init__()
        self.il, self.cfg = il, cfg
        self.recurrent = cfg.is_recurrent(il)
        self.lin = nn.ModuleDict({k.replace(".", "_"): v for k, v in lin.items()})
        for k, v in small.items():
            self.register_buffer(k.replace(".", "_"), v, persistent=False)

    def _p(self, name: str) -> PackedLinear:
        return self.lin[name]

    def attn_linear(self, x: torch.Tensor) -> torch.Tensor:
        c = self.cfg
        b, t = x.shape[:2]
        nk, nv, dk = c.ssm_n_group, c.ssm_dt_rank, c.ssm_d_state
        dv = c.ssm_d_inner // nv
        z = self._p("attn_qkv").rotated_input(x)  # shared by every 5120-wide folded weight
        qkv = self._p("attn_qkv")(x, z)
        gate = self._p("attn_gate")(x, z)
        beta = torch.sigmoid(F.linear(x, self.ssm_beta_weight.to(x.dtype)).float())
        alpha = F.linear(x, self.ssm_alpha_weight.to(x.dtype)).float()
        g = F.softplus(alpha + self.ssm_dt_bias) * self.ssm_a              # log decay, <= 0
        conv = F.conv1d(qkv.transpose(1, 2).float(), self.ssm_conv1d_weight.unsqueeze(1),
                        padding=c.ssm_conv - 1, groups=qkv.shape[-1])[..., :t]
        conv = F.silu(conv).transpose(1, 2).to(x.dtype)
        q = conv[..., : nk * dk].reshape(b, t, nk, dk)
        k = conv[..., nk * dk: 2 * nk * dk].reshape(b, t, nk, dk)
        v = conv[..., 2 * nk * dk:].reshape(b, t, nv, dv)
        q, k = l2_norm(q, c.eps), l2_norm(k, c.eps)
        rep = nv // nk
        q, k = q.repeat(1, 1, rep, 1), k.repeat(1, 1, rep, 1)               # tiled: head h uses k-head h % nk
        o = chunk_gated_delta_rule(q, k, v, g, beta)
        o = rms_norm(o, self.ssm_norm_weight, c.eps) * F.silu(gate.reshape(b, t, nv, dv).float()).to(o.dtype)
        return self._p("ssm_out")(o.reshape(b, t, nv * dv))

    def attn_full(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        c = self.cfg
        b, t, hd = x.shape[0], x.shape[1], c.head_dim
        z = self._p("attn_q").rotated_input(x)
        qg = self._p("attn_q")(x, z).reshape(b, t, c.n_head, 2 * hd)
        q, gate = qg[..., :hd], qg[..., hd:]
        k = self._p("attn_k")(x, z).reshape(b, t, c.n_head_kv, hd)
        v = self._p("attn_v")(x, z).reshape(b, t, c.n_head_kv, hd)
        q = rope_neox(rms_norm(q, self.attn_q_norm_weight, c.eps), pos, c.n_rot, c.rope_base)
        k = rope_neox(rms_norm(k, self.attn_k_norm_weight, c.eps), pos, c.n_rot, c.rope_base)
        o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                           is_causal=True, scale=1.0 / math.sqrt(hd), enable_gqa=True)
        o = o.transpose(1, 2).reshape(b, t, c.n_head * hd)
        o = o * torch.sigmoid(gate.reshape(b, t, c.n_head * hd).float()).to(o.dtype)
        return self._p("attn_output")(o)

    def ffn(self, x: torch.Tensor) -> torch.Tensor:
        z = self._p("ffn_gate").rotated_input(x)
        h = F.silu(self._p("ffn_gate")(x, z).float()).to(x.dtype) * self._p("ffn_up")(x, z)
        return self._p("ffn_down")(h)

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        h = rms_norm(x, self.attn_norm_weight, self.cfg.eps)
        h = self.attn_linear(h) if self.recurrent else self.attn_full(h, pos)
        x = x + h
        return x + self.ffn(rms_norm(x, self.post_attention_norm_weight, self.cfg.eps))


class PackedQwen35(nn.Module):
    def __init__(self, cfg: Qwen35Config, layers: list[Layer], embd_codes, embd_scales, embd_signs,
                 output: PackedLinear, output_norm: torch.Tensor, hmat: torch.Tensor, dtype):
        super().__init__()
        self.cfg, self.dtype = cfg, dtype
        self.layers = nn.ModuleList(layers)
        self.register_buffer("embd_codes", embd_codes, persistent=False)
        self.register_buffer("embd_scales", embd_scales, persistent=False)
        self.register_buffer("embd_signs", embd_signs, persistent=False)
        self.register_buffer("output_norm", output_norm, persistent=False)
        self.register_buffer("hmat", hmat, persistent=False)
        self.output = output
        self.gradient_checkpointing = True

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        flat = ids.reshape(-1)
        z = decode(self.embd_codes[flat], self.embd_scales[flat], self.dtype)
        return inverse_rotate(z, self.embd_signs, self.hmat).reshape(*ids.shape, -1)

    def hidden(self, ids: torch.Tensor, capture: tuple[int, ...] = ()) -> tuple[torch.Tensor, dict]:
        """ids [T] or [B, T] (right-padded) -> final normalized hidden; `capture` layer outputs too."""
        single = ids.dim() == 1
        if single:
            ids = ids[None]
        pos = torch.arange(ids.shape[1], device=ids.device)
        x = self.embed(ids)
        caps = {}
        for layer in self.layers:
            if self.gradient_checkpointing and torch.is_grad_enabled():
                x = checkpoint(layer, x, pos, use_reentrant=False)
            else:
                x = layer(x, pos)
            if layer.il in capture:
                caps[layer.il] = x.detach()[0] if single else x.detach()
        h = rms_norm(x, self.output_norm, self.cfg.eps)
        return (h[0] if single else h), caps

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        return self.output(h).float()

    def lora_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def packed_linears(self):
        for m in self.modules():
            if isinstance(m, PackedLinear):
                yield m


def _np(g: GGUFFile, name: str, dtype) -> np.ndarray:
    return np.frombuffer(g.payload(name), dtype=dtype)


def _small(g: GGUFFile, name: str, device) -> torch.Tensor:
    t = g.tensors[name]
    if t.ggml_type == F32:
        arr = torch.from_numpy(_np(g, name, np.float32).copy())
    elif t.ggml_type == BF16:
        raw = torch.from_numpy(_np(g, name, np.uint16).astype(np.int32) << 16)
        arr = raw.view(torch.float32)
    else:
        raise ValueError(f"{name}: unexpected type {t.ggml_type}")
    return arr.reshape(tuple(reversed(t.dims))).to(device)


def load_packed_qwen35(path: str | Path, device="cuda", dtype=torch.bfloat16, n_layers: int | None = None,
                       chunk_elems: int = 1 << 26, log=print) -> PackedQwen35:
    """Copy the packed PTQ1_0 payloads to `device` tensor by tensor (never dense)."""
    g = GGUFFile(path)
    md = g.metadata
    if md.get("general.architecture") != "qwen35":
        raise ValueError("expected a qwen35 GGUF")
    if md.get("prism.hadamard.transform") != "normalized-sylvester-walsh-hadamard" or \
            md.get("prism.hadamard.axis") != "input-last-dimension":
        raise ValueError("unsupported prism.hadamard metadata")
    cfg = Qwen35Config.from_gguf(md)
    hmat = hadamard_matrix(int(md["prism.hadamard.block_size"]), device=device)
    signs, off = {}, 0
    values = np.asarray(md["prism.hadamard.sign_values"], dtype=np.float32)
    for w in md["prism.hadamard.sign_widths"]:
        signs[int(w)] = torch.from_numpy(values[off:off + w].copy()).to(device)
        off += w
    folded = set(md["prism.hadamard.weight_names"])
    inverse = set(md["prism.hadamard.inverse_weight_names"])
    grouped = bool(md.get("prism.hadamard.gdn_v_grouped", False))

    def packed(name: str) -> tuple[torch.Tensor, torch.Tensor, int]:
        t = g.tensors[name]
        if t.ggml_type != PTQ1:
            raise ValueError(f"{name}: expected PTQ1_0")
        raw = torch.from_numpy(_np(g, name, np.uint8).copy())
        codes, scales = split_blocks(raw, t.rows, t.row_length)
        return codes.to(device), scales.to(device), t.row_length

    def linear(name: str) -> PackedLinear:
        codes, scales, cols = packed(name)
        perm = None
        if grouped and ".ssm_out." in name:
            perm = (cols // cfg.ssm_dt_rank, cfg.ssm_n_group, cfg.ssm_dt_rank // cfg.ssm_n_group)
        sg = signs[cols] if name in folded else None
        return PackedLinear(name, codes, scales, cols, sg, hmat if sg is not None else None, perm, chunk_elems)

    n = cfg.n_layer if n_layers is None else n_layers
    layers = []
    for il in range(n):
        pre = f"blk.{il}."
        names = [k[len(pre):] for k in g.tensors if k.startswith(pre)]
        lin = {k[:-len(".weight")]: linear(pre + k) for k in names if g.tensors[pre + k].ggml_type == PTQ1}
        small = {k: _small(g, pre + k, device) for k in names if g.tensors[pre + k].ggml_type != PTQ1}
        if "ssm_conv1d.weight" in small:
            small["ssm_conv1d.weight"] = small["ssm_conv1d.weight"].reshape(-1, cfg.ssm_conv)
        layers.append(Layer(il, cfg, lin, small))
        if il % 8 == 0:
            log(f"loaded layer {il}, cuda allocated {torch.cuda.memory_allocated() / 2**20:.0f} MiB"
                if str(device).startswith("cuda") else f"loaded layer {il}")
    ec, es, _ = packed("token_embd.weight")
    if "token_embd.weight" not in inverse:
        raise ValueError("token_embd is expected to be a Hadamard-latent table")
    out = linear("output.weight")
    out.chunk = max(1, (1 << 25) // out.in_features)
    model = PackedQwen35(cfg, layers, ec, es, signs[cfg.n_embd], out,
                         _small(g, "output_norm.weight", device), hmat, dtype)
    model.source = {"path": str(path), "size": Path(path).stat().st_size, "n_layers_loaded": n}
    for p in model.parameters():
        p.requires_grad_(False)
    g.close()
    return model


def attach_lora(model: PackedQwen35, rank: int, alpha: float, targets: list[str], seed: int,
                quant: str = "none") -> int:
    """Add LoRA to every packed projection whose short name is in `targets`."""
    gen = torch.Generator().manual_seed(seed)
    total = 0
    for m in model.packed_linears():
        if m.name == "output.weight":
            continue
        short = m.name.split(".")[2]
        if short in targets:
            total += m.add_lora(rank, alpha, gen, quant)
    return total


def lora_state(model: PackedQwen35) -> dict[str, torch.Tensor]:
    out = {}
    for m in model.packed_linears():
        if m.lora_A is not None:
            out[m.name + ".lora_a"] = m.lora_A.detach().cpu().contiguous()
            out[m.name + ".lora_b"] = m.lora_B.detach().cpu().contiguous()
    return out


def load_lora_state(model: PackedQwen35, state: dict[str, torch.Tensor]) -> int:
    n = 0
    for m in model.packed_linears():
        if m.lora_A is not None:
            m.lora_A.data.copy_(state[m.name + ".lora_a"].to(m.lora_A.device))
            m.lora_B.data.copy_(state[m.name + ".lora_b"].to(m.lora_B.device))
            n += 1
    return n

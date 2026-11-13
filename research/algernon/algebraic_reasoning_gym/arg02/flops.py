"""Analytic multiply-accumulate counts for Qwen3.5 text decoders.

An estimate from the config: dense projections, the gated delta-rule state
update (about 4 dk*dv MACs per value head per token), causal softmax attention
(QK and AV) and the LM head on the scored positions only. Norms, activations
and elementwise residual arithmetic are omitted. FLOPs = 2 * MACs. Measured
GPU seconds are recorded separately and are the primary cost; this count makes
arms comparable across machines.
"""
from __future__ import annotations


def text_config(cfg):
    return cfg.get("text_config", cfg)


def _dense_linear_attention(t):
    h = t["hidden_size"]
    nk, nv = t["linear_num_key_heads"], t["linear_num_value_heads"]
    dk, dv = t["linear_key_head_dim"], t["linear_value_head_dim"]
    key_dim, value_dim = nk * dk, nv * dv
    conv_dim = 2 * key_dim + value_dim
    proj = h * conv_dim + h * value_dim + 2 * h * nv + value_dim * h
    conv = conv_dim * t["linear_conv_kernel_dim"]
    state = 4 * nv * dk * dv
    return proj + conv + state


def _dense_full_attention(t):
    h, nh, nkv, hd = t["hidden_size"], t["num_attention_heads"], t["num_key_value_heads"], t["head_dim"]
    q = h * nh * hd * (2 if t.get("attn_output_gate") else 1)
    return q + 2 * h * nkv * hd + nh * hd * h


def _mlp(t):
    return 3 * t["hidden_size"] * t["intermediate_size"]


def _dense_softmax_attention(t):
    h, nh, nkv, hd = t["hidden_size"], t["num_attention_heads"], t["num_key_value_heads"], t["head_dim"]
    return h * nh * hd + 2 * h * nkv * hd + nh * hd * h


def layer_macs(cfg, layer, length):
    """MACs for one row of `length` tokens through decoder layer `layer`."""
    t = text_config(cfg)
    if t.get("model_type") == "nanbeige":
        # Every layer is softmax attention + SwiGLU MLP; the loop re-runs all of them.
        attn = t["num_attention_heads"] * t["head_dim"] * length * (length + 1)
        return length * (_dense_softmax_attention(t) + _mlp(t)) + attn
    kind = t["layer_types"][layer]
    if kind == "linear_attention":
        return length * (_dense_linear_attention(t) + _mlp(t))
    if kind == "full_attention":
        attn = t["num_attention_heads"] * t["head_dim"] * length * (length + 1)
        return length * (_dense_full_attention(t) + _mlp(t)) + attn
    raise ValueError(f"unknown layer type {kind}")


def head_macs(cfg, positions):
    t = text_config(cfg)
    return positions * t["hidden_size"] * t["vocab_size"]


def path(cfg, block=None, depth=1):
    """Layer indices evaluated by a block-looped forward."""
    n = text_config(cfg)["num_hidden_layers"]
    if block is None or depth == 1:
        return list(range(n))
    a, b = block
    if not (0 <= a < b <= n) or depth < 1:
        raise ValueError("bad block or depth")
    return list(range(a)) + list(range(a, b)) * depth + list(range(b, n))


def forward_macs(cfg, length, rows, layers, scored_positions=3):
    per_row = sum(layer_macs(cfg, i, length) for i in layers)
    return rows * (per_row + head_macs(cfg, scored_positions))


def layer_evaluations(rows, length, layers):
    return rows * length * len(layers)


def ensemble_sizes(n_layers, block, depths, max_variants):
    """Repeated-forward count matched to a depth-d block loop's layer evaluations."""
    a, b = block
    return {d: max(1, min(max_variants, round((n_layers + (d - 1) * (b - a)) / n_layers))) for d in depths}

"""Qwen3 plumbing: load the pinned model in fp32, and run a range of decoder layers on cached hidden states."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM

import config


def load_model(device: str = "cpu"):
    model = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL, revision=config.BASE_REVISION, cache_dir=config.HF_CACHE, dtype=torch.float32,
    )
    return model.eval().requires_grad_(False).to(device)


def run_layers(model, hidden: torch.Tensor, start: int, stop: int | None) -> torch.Tensor:
    """Decoder layers start..stop-1 with a causal mask, on hidden states of shape [batch, length, d]."""
    batch, length = hidden.shape[:2]
    positions = torch.arange(length, device=hidden.device).unsqueeze(0).expand(batch, -1)
    rotary = model.model.rotary_emb(hidden, positions)
    causal = torch.full((length, length), float("-inf"), device=hidden.device, dtype=hidden.dtype).triu(1)[None, None]
    for layer in model.model.layers[start:stop]:
        hidden = layer(hidden, attention_mask=causal, position_embeddings=rotary)
        hidden = hidden[0] if isinstance(hidden, tuple) else hidden
    return hidden

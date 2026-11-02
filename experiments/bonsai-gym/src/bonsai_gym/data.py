"""Chat rendering, tokenization and batching for SFT on Bonsai 2.

The prompt string is the GGUF chat template rendered with
add_generation_prompt=True and enable_thinking=False (checked against the
template in the PTQ1_0 file by tests/test_data.py), so the trained prefix is
exactly what llama-server sends in non-thinking mode. Loss is on the answer and
the closing <|im_end|> only.
"""
from __future__ import annotations

import difflib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

import torch

IM_END = "<|im_end|>"


def render_prompt(messages: list[dict]) -> str:
    out = []
    for m in messages:
        if m["role"] not in ("system", "user"):
            raise ValueError("prompt messages must be system/user")
        out.append(f"<|im_start|>{m['role']}\n{m['content']}{IM_END}\n")
    return "".join(out) + "<|im_start|>assistant\n<think>\n\n</think>\n\n"


@dataclass
class Example:
    id: str
    prompt_ids: list[int]
    target_ids: list[int]
    meta: dict


def load_examples(path: str | Path, tok, max_len: int) -> tuple[list[Example], int]:
    """JSONL rows with `messages` (system/user) and `target` (assistant text)."""
    out, skipped = [], 0
    for line in Path(path).read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        r = json.loads(line)
        p = tok.encode(render_prompt(r["messages"]), add_special_tokens=False).ids
        t = tok.encode(r["target"] + IM_END, add_special_tokens=False).ids
        if len(p) + len(t) > max_len:
            skipped += 1
            continue
        out.append(Example(r["id"], p, t, {k: v for k, v in r.items() if k not in ("messages", "target")}))
    return out, skipped


def collate(batch: list[Example], pad_id: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-padded ids [B, T] and labels [B, T] (-100 off-target); label at t is token t+1."""
    n = max(len(e.prompt_ids) + len(e.target_ids) for e in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), n), -100, dtype=torch.long)
    for i, e in enumerate(batch):
        seq = e.prompt_ids + e.target_ids
        ids[i, :len(seq)] = torch.tensor(seq)
        p = len(e.prompt_ids)
        lab[i, p - 1:len(seq) - 1] = torch.tensor(e.target_ids)
    return ids.to(device), lab.to(device)


class Sampler:
    """Deterministic epoch shuffles from one seed."""

    def __init__(self, n: int, seed: int):
        self.n, self.rng, self.order = n, random.Random(seed), []

    def next(self, k: int) -> list[int]:
        out = []
        while len(out) < k:
            if not self.order:
                self.order = list(range(self.n))
                self.rng.shuffle(self.order)
            out.append(self.order.pop())
        return out

    def state(self) -> dict:
        return {"order": list(self.order), "rng": self.rng.getstate()}

    def load(self, st: dict) -> None:
        self.order = list(st["order"])
        rs = st["rng"]
        self.rng.setstate((rs[0], tuple(rs[1]), rs[2]))


_REV = re.compile(r"<reversed_text>(.*?)</reversed_text>", re.S)


def lcs_reward(text: str, answer: str) -> float:
    """reverse-text env score: SequenceMatcher ratio of the parsed <reversed_text> body (0 if unclosed)."""
    m = _REV.search(text)
    if not m:
        return 0.0
    parsed = m.group(1).strip()
    return difflib.SequenceMatcher(None, parsed, answer).ratio()

"""Compare prompt templates for adapter_bits.py by how much the answer-position features vary across keys.

For each template, 768 random keys run through the frozen model. The script reports, for the residual
after the last layer's attention (h) and for the final hidden state, the centered share of the norm,
the effective rank (participation ratio) and two condition numbers. It also reports the spread of the
base A-vs-B margin. More varied features make random labels easier to store.
"""

from __future__ import annotations

import json

import torch
from transformers import AutoTokenizer

import adapter_bits as ab
import config
from qwen_suffix import load_model

TEMPLATES = [
    "Item {key} belongs to group",
    "Group of item {key}:",
    "The secret label for code {key} is",
    "{key} ->",
    "Q: Which group is item {key} in?\nA:",
]


def spread(x: torch.Tensor) -> dict:
    centered = x - x.mean(0)
    s = torch.linalg.svdvals(centered)
    return {"centered_share": round((centered.norm() / x.norm()).item(), 4),
            "eff_rank": round((s.pow(2).sum() ** 2 / s.pow(4).sum()).item(), 1),
            "cond_128": round((s[0] / s[127]).item(), 1), "cond_512": round((s[0] / s[511]).item(), 1)}


def main() -> None:
    config.lower_priority()
    torch.set_num_threads(8)
    tok = AutoTokenizer.from_pretrained(config.BASE_MODEL, revision=config.BASE_REVISION, cache_dir=config.HF_CACHE)
    model = load_model("cpu")
    frozen = ab.Frozen(model, tok)
    for template in TEMPLATES:
        ab.TEMPLATE = template
        try:
            ids, _ = ab.make_items(tok, 768, 0)
        except SystemExit as error:
            print(json.dumps({"template": template, "skipped": str(error)}), flush=True)
            continue
        h = ab.cache_residual(model, ids, 64)
        final = frozen.final_hidden(h)
        print(json.dumps({"template": template, "tokens": ids.shape[1], "h": spread(h), "final": spread(final),
                          "base_margin_std": round((final @ frozen.readout).std().item(), 4)}), flush=True)


if __name__ == "__main__":
    main()

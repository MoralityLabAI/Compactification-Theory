"""Write plans/bdb_expanded.json and one bonsai-gym config per model: a 1-bit rank-8 BDB LoRA on every
Ternary Bonsai 2 series model, trained on the expanded Bitcoin DeFi Bench SFT rows (every family with a
teacher skill, bitcoin-defi-bench scripts/make_sft_data.ts), with each model's base and adapter scored on the
eval split by score_sft_replies.

  python plans/make_bdb_expanded.py [--epochs 1.0] [--max-steps 200]

Steps per model = ceil(epochs * train rows / 8), capped at --max-steps (batch 8 = microbatch 1 x grad_accum 8).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
GYM = HERE.parents[1] / "bonsai-gym"
WORK = "C:/Users/patri/Documents/Codex/bdb-expanded"
M = "C:/Users/patri/models"
# id, gguf for training, gguf for serving, spec args, eval vram, training max_len (tokens; 27B went out of
# memory above ~3,400 at 15 GB, smaller models carry less activation memory per token)
MODELS = [
    ("b2-2b", f"{M}/ternary-bonsai-2-2b/Ternary-Bonsai-2-2B-PTQ1_0.gguf", None, [], "5G", 10240),
    ("b2-4b", f"{M}/ternary-bonsai-2-4b/Ternary-Bonsai-2-4B-PTQ1_0.gguf", None, [], "6G", 8192),
    ("b2-9b", f"{M}/ternary-bonsai-2-9b/Ternary-Bonsai-2-9B-PTQ1_0.gguf", None, [], "8G", 5120),
    ("b2-27b", f"{M}/ternary-bonsai-2-27b/Ternary-Bonsai-2-27B-PTQ1_0.gguf", None, [], "12G", 3072),
    ("b25-27b", f"{M}/ternary-bonsai-2.5-27b/Ternary-Bonsai-2.5-27B-PTQ1_0.gguf",
     f"{M}/ternary-bonsai-2.5-27b/Ternary-Bonsai-2.5-27B-PTQ1_0-mtp.gguf",
     ["--spec-type", "draft-mtp", "--spec-draft-n-max", "3", "--spec-draft-p-min", "0.1"], "12G", 3072),
]
# eval prompts run to ~17k tokens (later turns of bitvm3, nostr, cashu): 2 slots x 18,432
EVAL = {"slots": 2, "ctx": 36864, "max_tokens": 1024}


def rows(p: Path) -> int:
    return sum(1 for l in p.read_text(encoding="utf-8").splitlines() if l.strip())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=150)
    a = ap.parse_args()
    data = Path(WORK) / "data" / "bdb"
    n_train, n_val = rows(data / "train.jsonl"), rows(data / "val.jsonl")
    steps = min(a.max_steps, math.ceil(a.epochs * n_train / 8))
    base_cfg = json.loads((GYM / "configs" / "b25_onebit_bdb.json").read_text(encoding="utf-8"))
    plan_steps, backends = [], {}
    for mid, train_gguf, serve_gguf, spec, vram, max_len in MODELS:
        name = f"bdbx_onebit_{mid.replace('-', '_')}"
        cfg = {**base_cfg, "name": name,
               "description": f"1-bit (binary STE) rank-8 LoRA on {mid}: expanded Bitcoin DeFi Bench SFT rows "
                              f"(every family with a teacher skill; make_sft_data.ts train 5000:50, val 6000:4).",
               "gguf": train_gguf, "train": str(data / "train.jsonl").replace("\\", "/"),
               "val": str(data / "val.jsonl").replace("\\", "/"), "n_val": n_val, "max_len": max_len,
               "out_dir": f"{WORK}/runs/{name}", "steps": steps, "eval_every": max(20, steps // 4)}
        (GYM / "configs" / f"{name}.json").write_text(json.dumps(cfg, indent=1) + "\n", encoding="utf-8")
        backends[mid] = {"kind": "llama", "name": mid, "bin": "C:/Users/patri/tools/llama-prism-b12178/bin",
                         "model": serve_gguf or train_gguf,
                         "runtime": f"llama-prism-b12178 PTQ1_0 pure-GPU q8 KV{', MTP draft n-max 3' if spec else ''}"}
        if spec:
            backends[mid]["spec_args"] = spec
        t = [f"bdbx-{mid}"]
        plan_steps += [
            {"id": f"{mid}-base-eval", "kind": "generate", "backend": mid, "arm": f"{mid}-base", "families": ["bdb"],
             **EVAL, "vram": vram, "tags": t + ["bdbx-eval"]},
            {"id": f"{mid}-base-score", "kind": "score", "arm": f"{mid}-base", "after": [f"{mid}-base-eval"], "tags": t + ["bdbx-eval"]},
            {"id": f"{mid}-train", "kind": "train_bonsai", "config": name, "vram": "15G", "tags": t + ["bdbx-train"]},
            {"id": f"{mid}-adapter-eval", "kind": "generate", "backend": mid, "arm": f"{mid}-onebit", "families": ["bdb"],
             "adapter": f"{WORK}/runs/{name}/adapter.gguf", **EVAL, "after": [f"{mid}-train"], "vram": vram,
             "tags": t + ["bdbx-eval"]},
            {"id": f"{mid}-adapter-score", "kind": "score", "arm": f"{mid}-onebit", "after": [f"{mid}-adapter-eval"], "tags": t + ["bdbx-eval"]},
        ]
    src = json.loads((HERE / "bdb_compare.json").read_text(encoding="utf-8-sig"))
    plan = {"name": "bdb-expanded",
            "about": f"1-bit rank-8 BDB LoRA on every Ternary Bonsai 2 series model (2B, 4B, 9B, 27B, 2.5 27B) from the "
                     f"expanded Bitcoin DeFi Bench SFT rows ({n_train} train, {n_val} val), {steps} steps each; base and "
                     f"adapter scored on the eval split. Tags: bdbx-<model>, bdbx-train, bdbx-eval.",
            "work": WORK, "port": 18290, "paths": src["paths"], "backends": backends, "steps": plan_steps}
    (HERE / "bdb_expanded.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"train": n_train, "val": n_val, "steps": steps, "models": [m[0] for m in MODELS]}))


if __name__ == "__main__":
    main()

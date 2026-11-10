"""Pre-run checkpoint check on CPU: file hashes, tokenizer contract, prompt
lengths, and depth-1 parity of the block-loop runner against the HF forward
on the real weights. Uses no GPU and grades nothing.

    python -m research.algernon.algebraic_reasoning_gym.arg02.checkpoint_check \\
        --run-dir D --out results/arg02_checkpoint_check.json [--parity 2]

`--run-dir` must hold the `analyze export` output.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from . import runner
from .prompts import VARIANTS
from .settings import load_config, load_lock


def lengths(tok, rows, shots, variants=VARIANTS):
    eng = type("E", (), {})()
    eng.tok, eng.shots = tok, shots
    out = {}
    for v in variants:
        ls = [len(runner.Engine.prompt_ids(eng, r, v)) for r in rows]
        out[v] = {"min": min(ls), "max": max(ls), "mean": sum(ls) / len(ls)}
    return out


def parity(model, tok, rows, block, op_ids, n):
    import torch
    from .recurrence import BlockLoop, candidate_rows
    bl = BlockLoop(model, block)
    eng = type("E", (), {})()
    eng.tok, eng.shots = tok, []
    out = []
    for row in rows[:n]:
        ids = candidate_rows(runner.Engine.prompt_ids(eng, row), op_ids)[:2]
        t0 = time.perf_counter()
        with torch.inference_mode():
            ref = torch.log_softmax(model(input_ids=ids).logits[:, -3:].float(), -1)
            emb, ctx = bl.prepare(ids)
            h = bl.run(emb, 0, bl.n, ctx)
            mine = torch.log_softmax(model.lm_head(bl.base.norm(h[:, -3:])).float(), -1)
            x = bl.run(emb, 0, block[0], ctx)
            h1 = bl.run(x, block[0], block[1], ctx)
            h2 = bl.run(h1, block[0], block[1], ctx)
            rel = float((h2 - h1).float().norm() / h1.float().norm())
        out.append({"case_id": row["case_id"], "tokens": ids.shape[1],
                    "max_abs_logprob_diff": float((ref - mine).abs().max()),
                    "top_token_equal": bool((ref.argmax(-1) == mine.argmax(-1)).all()),
                    "depth2_relative_residual": rel, "seconds": time.perf_counter() - t0})
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--parity", type=int, default=0, help="rows to run through the real model on CPU")
    args = p.parse_args()
    lock, cfg = load_lock(), load_config()
    from transformers import AutoTokenizer
    report = {"schema": "algernon.arg02.checkpoint_check.v1", "device": "cpu"}
    for key in ("model", "ff_control"):
        t0 = time.perf_counter()
        root = runner.verify_snapshot(lock[key])
        report[key] = {"repo": lock[key]["repo"], "revision": lock[key]["revision"], "files_verified": sorted(lock[key]["files_sha256"]),
                       "hash_seconds": time.perf_counter() - t0}
    tok = AutoTokenizer.from_pretrained(runner.snapshot_dir(lock["model"]), local_files_only=True)
    op_ids = runner.op_token_ids(tok)
    run = Path(args.run_dir)
    test = runner.read_public(run / "public" / "test.jsonl")
    dev = runner.read_public(run / "public" / "development.jsonl")
    shots = runner.read_shots(run / "public" / "shots.json", max(cfg["shot_options"]))
    report["tokenizer"] = {"class": type(tok).__name__, "op_token_ids": op_ids,
                           "tail_ids": tok.encode("\nProgram:", add_special_tokens=False)}
    report["prompt_tokens"] = {f"test_shots{k}": lengths(tok, test, shots[:k]) for k in cfg["shot_options"]}
    report["prompt_tokens"].update({f"development_shots{k}": lengths(tok, dev, shots[:k], ("canonical",))
                                    for k in cfg["shot_options"]})
    if args.parity:
        import torch
        from transformers import AutoModelForCausalLM
        root = runner.snapshot_dir(lock["model"])
        t0 = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(root, dtype=torch.bfloat16, local_files_only=True,
                                                     attn_implementation="sdpa").eval()
        report["load_seconds"] = time.perf_counter() - t0
        report["model_class"] = type(model).__name__
        report["text_parameters"] = sum(p.numel() for p in model.model.parameters())
        report["parity"] = parity(model, tok, dev, tuple(cfg["block_candidates"][0]), op_ids, args.parity)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

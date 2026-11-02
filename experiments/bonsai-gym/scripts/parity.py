"""Frozen-base parity: the torch packed forward against the PrismML runtime.

Reads reference.json from llama_reference.py and checks, on the same token ids:
- tokenizer: HF tokenizer ids == llama-server /tokenize ids for every prompt;
- greedy agreement: teacher-forced argmax of the torch model over the llama
  greedy continuation (if every position agrees, greedy decoding reproduces it);
- perplexity on the parity text with llama-perplexity's protocol (chunks of
  `ctx` tokens, score the second half of each chunk).
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bonsai_gym.model import load_packed_qwen35  # noqa: E402
from bonsai_gym.runinfo import run_info  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gguf", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--text", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--lora", default=None, help="safetensors adapter to load before scoring")
    args = p.parse_args()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(args.tokenizer)
    ref = json.loads(Path(args.reference).read_text(encoding="utf-8"))

    t0 = time.time()
    model = load_packed_qwen35(args.gguf)
    if args.lora:
        from bonsai_gym.adapter_io import load_adapter
        load_adapter(model, args.lora)
    model.eval()
    load_s = time.time() - t0
    res = {"load_s": round(load_s, 1), "prompts": []}
    agree_all, total_all = 0, 0
    with torch.no_grad():
        for c in ref["completions"]:
            hf_ids = tok.encode(c["prompt"], add_special_tokens=False).ids
            ids = c["prompt_ids"] + c["completion_ids"]
            x = torch.tensor(ids, device="cuda")
            t1 = time.time()
            h, _ = model.hidden(x)
            n0 = len(c["prompt_ids"])
            logits = model.logits(h[n0 - 1:-1])
            pred = logits.argmax(-1).tolist()
            agree = [int(a == b) for a, b in zip(pred, c["completion_ids"])]
            first_miss = agree.index(0) if 0 in agree else None
            agree_all += sum(agree)
            total_all += len(agree)
            res["prompts"].append({
                "id": c["id"], "tokenizer_match": hf_ids == c["prompt_ids"], "n_prompt": n0,
                "n_completion": len(c["completion_ids"]), "top1_agree": sum(agree) / max(1, len(agree)),
                "first_disagreement": first_miss, "forward_s": round(time.time() - t1, 2),
                "torch_greedy_prefix": tok.decode(pred[: (first_miss or len(pred)) + 1]),
            })
            print(res["prompts"][-1], flush=True)
        ids = tok.encode(Path(args.text).read_text(encoding="utf-8"), add_special_tokens=False).ids
        ctx, chunks = ref["ctx"], ref["chunks"]
        nll, count = 0.0, 0
        t1 = time.time()
        for ci in range(min(chunks, len(ids) // ctx)):
            x = torch.tensor(ids[ci * ctx:(ci + 1) * ctx], device="cuda")
            h, _ = model.hidden(x)
            first = ctx // 2
            lg = model.logits(h[first:-1])
            nll += float(F.cross_entropy(lg, x[first + 1:], reduction="sum"))
            count += ctx - 1 - first
        res["ppl"] = math.exp(nll / count)
        res["ppl_tokens"] = count
        res["ppl_s"] = round(time.time() - t1, 1)
        res["ppl_forward_tok_s"] = round(min(chunks, len(ids) // ctx) * ctx / (time.time() - t1), 1)
    res["llama_ppl"] = ref["ppl"]
    res["llama_ppl_err"] = ref["ppl_err"]
    res["ppl_rel_diff"] = (res["ppl"] - ref["ppl"]) / ref["ppl"] if ref["ppl"] else None
    res["top1_agree_all"] = agree_all / max(1, total_all)
    res["tokenizer_match_all"] = all(r["tokenizer_match"] for r in res["prompts"])
    res["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
    res["peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20
    res["run"] = run_info(args, gguf=args.gguf)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k not in ("prompts", "run")}, indent=1))


if __name__ == "__main__":
    main()

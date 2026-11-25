"""Greedy completions from a squid5090 server (/v1/completions) for rendered prompt text, resumable.

  python3 squid_gen.py --prompts eval_prompt_text.jsonl --out samples.jsonl [--port 18330] [--max-tokens 1536]

Prompts are the Tinker renderer's tokens decoded to text (they re-encode to the same ids with the checkpoint's
tokenizer.json; squid parses special tokens). squid's "text" is the raw completion with special tokens and thinking;
reply is the model message's text content (between <|content_text|> and <|end_message|>), the same field the Tinker
sampler's parse_response gives, or "" when the budget ran out before it. Writes {"id","reply","raw","tokens","tok_s"}.
"""
import argparse, json, re, time, urllib.request
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--prompts", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--port", type=int, default=18330); ap.add_argument("--max-tokens", type=int, default=1536)
a = ap.parse_args()


def content_of(raw: str) -> str:
    m = re.search(r"<\|content_text\|>(.*?)(<\|end_message\|>|<\|content_model_end_sampling\|>|$)", raw, re.S)
    return m.group(1) if m else ""


rows = [json.loads(l) for l in Path(a.prompts).read_text(encoding="utf-8").split("\n") if l.strip()]
done = set()
if Path(a.out).exists():
    done = {json.loads(l)["id"] for l in Path(a.out).read_text(encoding="utf-8").split("\n") if l.strip()}
with open(a.out, "a", encoding="utf-8") as f:
    for i, r in enumerate(rows):
        if r["id"] in done:
            continue
        body = {"model": "inkling-2-mini", "prompt": r["prompt"], "max_tokens": a.max_tokens, "temperature": 0,
                "top_p": 1, "top_k": 0, "seed": 523, "mtp": 0, "stream": False}
        t0 = time.time()
        req = urllib.request.Request(f"http://127.0.0.1:{a.port}/v1/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        d = json.loads(urllib.request.urlopen(req, timeout=3600).read())
        c = d["choices"][0]
        rec = {"id": r["id"], "reply": content_of(c["text"]), "raw": c["text"], "finish": c.get("finish_reason"),
               "tokens": d.get("usage", {}).get("completion_tokens"), "tok_s": d.get("timings", {}).get("tokens_per_second"),
               "prefill_ms": d.get("timings", {}).get("prefill_ms"), "s": time.time() - t0}
        f.write(json.dumps(rec) + "\n"); f.flush()
        print(i + 1, r["id"], rec["tokens"], rec["tok_s"], flush=True)

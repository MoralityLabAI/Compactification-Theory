"""Greedy completions from a local llama-server for pre-tokenized prompts (the exact Tinker renderer tokens).

  python local_gen.py --prompts eval_prompt_tokens.jsonl --out samples_local.jsonl [--port 18320] [--max-tokens 1536]

Writes {"id", "reply", "raw", "tokens"}: reply is the model message's text content (between <|content_text|> and
<|end_message|> when the special tokens are rendered), so it matches the Tinker sampler's parsed content.
"""
import argparse, json, re, time, urllib.request
from pathlib import Path


def post(url, body, timeout=3600):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def content_of(raw: str) -> str:
    m = re.search(r"<\|content_text\|>(.*?)(<\|end_message\|>|<\|content_model_end_sampling\|>|$)", raw, re.S)
    return m.group(1) if m else raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=18320)
    ap.add_argument("--max-tokens", type=int, default=1536)
    a = ap.parse_args()
    rows = [json.loads(l) for l in Path(a.prompts).read_text().split("\n") if l.strip()]
    done = set()
    if Path(a.out).exists():
        done = {json.loads(l)["id"] for l in Path(a.out).read_text(encoding="utf-8").split("\n") if l.strip()}
    with open(a.out, "a", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            if r["id"] in done:
                continue
            t0 = time.time()
            res = post(f"http://127.0.0.1:{a.port}/completion", {
                "prompt": r["prompt_tokens"], "n_predict": a.max_tokens, "temperature": 0, "top_k": 1, "top_p": 1,
                "min_p": 0, "seed": 523, "cache_prompt": True, "return_tokens": True,
                "stop": ["<|content_model_end_sampling|>", "<|end_message|>"]})
            raw = res.get("content", "")
            f.write(json.dumps({"id": r["id"], "reply": content_of(raw), "raw": raw,
                                "tokens": res.get("tokens_predicted"), "s": time.time() - t0,
                                "tok_s": res.get("timings", {}).get("predicted_per_second")}) + "\n")
            f.flush()
            print(i + 1, r["id"], res.get("tokens_predicted"), round(res.get("timings", {}).get("predicted_per_second", 0), 1), flush=True)


if __name__ == "__main__":
    main()

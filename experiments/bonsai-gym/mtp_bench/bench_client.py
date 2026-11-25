"""Send the first N MIT-13 prompts to a running llama-server exactly as mit13_dispatch.py does (same system prompt,
greedy, seed, cache_prompt false, row max_tokens) and record speed, draft acceptance and the full reply.

  python bench_client.py --prompts prompts.jsonl --gym <MathematicalInstrumentTransfer dir> --port P --arm NAME --out out.jsonl
"""
import argparse, json, sys, time
from pathlib import Path
from urllib import request

ap = argparse.ArgumentParser()
ap.add_argument("--prompts", required=True); ap.add_argument("--gym", required=True)
ap.add_argument("--port", type=int, required=True); ap.add_argument("--arm", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=12); ap.add_argument("--seed", type=int, default=13)
a = ap.parse_args()
sys.path.insert(0, a.gym)
from mit13_dispatch import SYSTEM  # noqa: E402

rows = [json.loads(l) for l in Path(a.prompts).read_text(encoding="utf-8").split("\n") if l.strip()][: a.n + 1]
with open(a.out, "w", encoding="utf-8") as f:
    for i, row in enumerate(rows):
        body = dict(model="local", messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": row["prompt"]}],
                    max_tokens=row["max_tokens"], temperature=0.0, seed=a.seed, cache_prompt=False, stream=False)
        t0 = time.time()
        req = request.Request(f"http://127.0.0.1:{a.port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
        d = json.loads(request.urlopen(req, timeout=1800).read())
        t = d.get("timings", {}); msg = d["choices"][0]["message"]
        rec = {"arm": a.arm, "id": row["id"], "warmup": i == 0, "wall_s": time.time() - t0,
               "predicted_n": t.get("predicted_n"), "tok_s": t.get("predicted_per_second"), "prompt_tok_s": t.get("prompt_per_second"),
               "draft_n": t.get("draft_n"), "draft_accepted": t.get("draft_n_accepted"),
               "content": msg.get("content"), "reasoning": msg.get("reasoning_content")}
        f.write(json.dumps(rec) + "\n"); f.flush()
        print(a.arm, i, row["id"], rec["predicted_n"], rec["tok_s"] and round(rec["tok_s"], 1), rec["draft_n"], rec["draft_accepted"], flush=True)

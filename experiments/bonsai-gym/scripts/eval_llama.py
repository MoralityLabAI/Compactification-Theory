"""Score a base or base + GGUF LoRA on the reverse-text eval items through the
PrismML llama-server, with q1-adapters' sampling (T 0.7, top-p 0.8, top-k 20,
per-example seed sha256(id/k)[:8], 384 new tokens, non-thinking), so the
numbers are paired with each other and comparable with the Bonsai 8B results.

Run inside a vramorama lease. Writes <out>/<label>.jsonl and <label>.summary.json.
"""
import argparse
import hashlib
import json
import math
import random
import statistics
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bonsai_gym.data import lcs_reward  # noqa: E402


def seed_of(ex_id, k):
    return int(hashlib.sha256(f"{ex_id}/{k}".encode()).hexdigest()[:8], 16)


def chat(port, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--lora", action="append", default=[])
    ap.add_argument("--port", type=int, default=18086)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=384)
    ap.add_argument("--n", type=int, default=None)
    args = ap.parse_args()
    items = [json.loads(l) for l in Path(args.items).read_text(encoding="utf-8").split("\n") if l.strip()]
    items = items[: args.n] if args.n else items
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cmd = [str(Path(args.bin) / "llama-server.exe"), "-m", args.gguf, "-ngl", "99", "-c", str(2048 * args.parallel),
           "-np", str(args.parallel), "--port", str(args.port), "--no-webui", "--jinja"]
    for l in args.lora:
        cmd += ["--lora", l]
    log = open(out / f"{args.label}.server.log", "w", encoding="utf-8")
    srv = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    rows = []
    try:
        for _ in range(900):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/health", timeout=2) as r:
                    if r.status == 200:
                        break
            except Exception:
                time.sleep(1)
        else:
            raise RuntimeError("llama-server did not become healthy")

        def one(it):
            t0 = time.time()
            r = chat(args.port, {"messages": it["messages"], "temperature": 0.7, "top_p": 0.8, "top_k": 20,
                                 "seed": seed_of(it["id"], 0), "max_tokens": args.max_tokens,
                                 "chat_template_kwargs": {"enable_thinking": False}})
            ch = r["choices"][0]
            text = ch["message"].get("content") or ""
            return {"id": it["id"], "k": 0, "answer": it["answer"], "text": text,
                    "finish_reason": ch.get("finish_reason"), "completion_tokens": r["usage"]["completion_tokens"],
                    "latency_s": time.time() - t0, "score": lcs_reward(text, it["answer"]), "label": args.label}

        t0 = time.time()
        with ThreadPoolExecutor(args.parallel) as ex:
            rows = list(ex.map(one, items))
        wall = time.time() - t0
    finally:
        srv.terminate()
        srv.wait(timeout=60)
        log.close()
    with open(out / f"{args.label}.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    sc = [r["score"] for r in rows]
    summ = {"label": args.label, "lora": args.lora, "n": len(sc), "score_mean": sum(sc) / len(sc),
            "score_ci95": 1.96 * statistics.stdev(sc) / math.sqrt(len(sc)) if len(sc) > 1 else 0.0,
            "truncated": sum(r["finish_reason"] == "length" for r in rows), "wall_s": wall,
            "completion_tok_s": sum(r["completion_tokens"] for r in rows) / wall,
            "sampling": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "max_tokens": args.max_tokens,
                         "enable_thinking": False, "parallel": args.parallel}}
    (out / f"{args.label}.summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print(json.dumps(summ))


def paired(a_path, b_path, n_boot=10000, seed=0):
    """Paired bootstrap of mean(b - a) over shared ids."""
    load = lambda p: {json.loads(l)["id"]: json.loads(l)["score"] for l in Path(p).read_text(encoding="utf-8").split("\n") if l.strip()}
    a, b = load(a_path), load(b_path)
    ids = sorted(set(a) & set(b))
    d = [b[i] - a[i] for i in ids]
    rng = random.Random(seed)
    boots = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(n_boot))
    return {"n": len(d), "delta": sum(d) / len(d), "ci95": [boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot)]]}


if __name__ == "__main__":
    main()

"""Reference outputs from the PrismML llama.cpp runtime for the parity check.

1. llama-server: tokenize each parity prompt and greedily continue it
   (temperature 0, raw completion, no chat template), keeping token ids.
2. llama-perplexity on the parity text at a fixed context.
Run inside a vramorama lease. Writes reference.json in --out.
"""
import argparse
import json
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def post(port, path, body, timeout=600):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bin", required=True)
    p.add_argument("--gguf", required=True)
    p.add_argument("--prompts", required=True)
    p.add_argument("--text", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--port", type=int, default=18085)
    p.add_argument("--n-predict", type=int, default=48)
    p.add_argument("--ctx", type=int, default=256)
    p.add_argument("--chunks", type=int, default=6)
    p.add_argument("--lora", default=None, help="GGUF LoRA to apply (scale 1)")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    binp = Path(args.bin)
    prompts = [json.loads(l) for l in Path(args.prompts).read_text(encoding="utf-8").split("\n") if l.strip()]

    cmd = [str(binp / "llama-server.exe"), "-m", args.gguf, "-ngl", "99", "-c", "4096", "--port", str(args.port),
           "-np", "1", "--no-webui"]
    if args.lora:
        cmd += ["--lora", args.lora]
    log = open(out / "server.log", "w", encoding="utf-8")
    srv = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    results = []
    try:
        for _ in range(600):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/health", timeout=2) as r:
                    if r.status == 200:
                        break
            except Exception:
                time.sleep(1)
        else:
            raise RuntimeError("llama-server did not become healthy")
        for pr in prompts:
            ids = post(args.port, "/tokenize", {"content": pr["prompt"], "add_special": False})["tokens"]
            r = post(args.port, "/completion", {"prompt": pr["prompt"], "n_predict": args.n_predict,
                                                "temperature": 0, "top_k": 1, "cache_prompt": False,
                                                "return_tokens": True, "special": True})
            results.append({"id": pr["id"], "prompt": pr["prompt"], "prompt_ids": ids,
                            "completion": r["content"], "completion_ids": r["tokens"],
                            "stop_type": r.get("stop_type"), "timings": r.get("timings")})
            print(pr["id"], ascii(r["content"][:80]), flush=True)
    finally:
        srv.terminate()
        srv.wait(timeout=60)
        log.close()

    ppl_cmd = [str(binp / "llama-perplexity.exe"), "-m", args.gguf, "-ngl", "99", "-f", args.text,
               "-c", str(args.ctx), "--chunks", str(args.chunks), "-b", str(args.ctx)]
    if args.lora:
        ppl_cmd += ["--lora", args.lora]
    pr = subprocess.run(ppl_cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    (out / "perplexity.log").write_text(pr.stdout + pr.stderr, encoding="utf-8")
    m = re.search(r"Final estimate: PPL = ([0-9.]+) \+/- ([0-9.]+)", pr.stdout + pr.stderr)
    ref = {"llama_bin": str(binp), "gguf": args.gguf, "lora": args.lora, "n_predict": args.n_predict,
           "ctx": args.ctx, "chunks": args.chunks, "completions": results,
           "ppl": float(m.group(1)) if m else None, "ppl_err": float(m.group(2)) if m else None,
           "perplexity_exit": pr.returncode}
    (out / "reference.json").write_text(json.dumps(ref, indent=1), encoding="utf-8")
    print("ppl", ref["ppl"], "exit", pr.returncode)
    sys.exit(0 if m else 1)


if __name__ == "__main__":
    main()

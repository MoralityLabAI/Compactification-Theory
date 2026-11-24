"""Generate replies for suite rows from a local llama-server, optionally under one of its loaded LoRA adapters.

  python generate.py --rows eval.jsonl --out replies.jsonl --base-url http://127.0.0.1:18299/v1 \
      [--adapter math --adapters math,logic,srb,bdb] [--temperature 0] [--samples 1] [--max-tokens 768]

Stdlib only. The server is started with every adapter loaded and none applied
(`--lora a.gguf,b.gguf,... --lora-init-without-apply`); each request names the adapter to apply through
the per-request `lora` field (id = position in --adapters, scale 1; all others scale 0), so one server
serves the base model and every adapter. Thinking is off (`enable_thinking: false`), as in training.

Resumable: (id, sample) pairs already in --out are skipped, so a run stopped by the power guard continues.
Rows are the suite schema: {"id", "family", "messages": [...], ...}. Output lines:
{"id", "sample", "adapter", "reply", "finish_reason", "completion_tokens", "seed"}.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


_cap_lock = threading.Lock()
_cap_state = {"t": 0.0, "capped": False}


def enforced_power_w() -> float | None:
    import subprocess
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=enforced.power.limit", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout.strip().splitlines()
        return float(out[0])
    except Exception:
        return None


def wait_while_capped(threshold: float) -> None:
    """Hold new requests while the laptop card sits at its 15 W cap: in-flight requests finish, the card idles,
    and the cap clears after a few idle minutes (work under the cap runs ~5x slower)."""
    if threshold <= 0:
        return
    with _cap_lock:   # one thread polls; the others wait on the lock
        while True:
            if time.time() - _cap_state["t"] > 10:
                w = enforced_power_w()
                was = _cap_state["capped"]
                _cap_state.update(t=time.time(), capped=w is not None and w < threshold)
                if _cap_state["capped"] != was:
                    print(f"power cap {'on' if _cap_state['capped'] else 'off'} ({w} W) {time.strftime('%H:%M:%S')}", flush=True)
            if not _cap_state["capped"]:
                return
            time.sleep(10)


def post(url: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:18299/v1")
    ap.add_argument("--model", default="bonsai-2-27b")
    ap.add_argument("--adapter", default="none", help="adapter name from --adapters, or none for the base model")
    ap.add_argument("--adapters", default="", help="comma list, in the order the server's --lora loaded them")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--seed", type=int, default=523)
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--cap-watch-w", type=float, default=60, help="hold new requests while the enforced GPU limit is below this (0 = off)")
    a = ap.parse_args()

    names = [n for n in a.adapters.split(",") if n]
    if a.adapter != "none" and a.adapter not in names:
        sys.exit(f"adapter {a.adapter} not in --adapters {names}")
    lora = [{"id": i, "scale": 1.0 if n == a.adapter else 0.0} for i, n in enumerate(names)]

    rows = [json.loads(l) for l in Path(a.rows).read_text(encoding="utf-8").split("\n") if l.strip()]
    if a.limit:
        rows = rows[: a.limit]
    out = Path(a.out)
    done = set()
    if out.exists():
        for l in out.read_text(encoding="utf-8").split("\n"):
            if l.strip():
                r = json.loads(l)
                done.add((r["id"], r["sample"]))
    todo = [(r, s) for r in rows for s in range(a.samples) if (r["id"], s) not in done]
    print(f"{len(rows)} rows x {a.samples} samples: {len(done)} done, {len(todo)} to go (adapter {a.adapter})", flush=True)
    lock = threading.Lock()
    f = out.open("a", encoding="utf-8")
    t0, n_done, n_tok = time.time(), 0, 0

    def one(item):
        nonlocal n_done, n_tok
        row, s = item
        body = {"model": a.model, "messages": row["messages"], "temperature": a.temperature, "top_p": a.top_p,
                "max_tokens": a.max_tokens, "seed": a.seed + s, "chat_template_kwargs": {"enable_thinking": False}}
        if names:
            body["lora"] = lora
        for attempt in range(1, 1000):
            wait_while_capped(a.cap_watch_w)
            try:
                res = post(a.base_url.rstrip("/") + "/chat/completions", body, a.timeout)
                break
            except Exception as e:  # server restarting under the power guard: wait and retry the same request
                print(f"retry {attempt} {row['id']}#{s}: {e}", flush=True)
                time.sleep(min(60, 5 * attempt))
        ch = res["choices"][0]
        rec = {"id": row["id"], "sample": s, "adapter": a.adapter, "reply": ch["message"].get("content") or "",
               "finish_reason": ch.get("finish_reason"), "completion_tokens": res.get("usage", {}).get("completion_tokens"),
               "seed": a.seed + s}
        with lock:
            f.write(json.dumps(rec) + "\n")
            f.flush()
            n_done += 1
            n_tok += rec["completion_tokens"] or 0
            if n_done % 10 == 0 or n_done == len(todo):
                dt = time.time() - t0
                print(f"{n_done}/{len(todo)}  {n_tok / dt:.1f} tok/s aggregate", flush=True)

    with ThreadPoolExecutor(a.concurrency) as ex:
        list(ex.map(one, todo))
    f.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

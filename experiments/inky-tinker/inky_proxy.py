"""OpenAI-style /v1/chat/completions in front of a squid5090 server, rendering each conversation with the Tinker
cookbook's tml_v0 renderer (the format the Inkling 2 Mini LoRA was trained on) instead of squid's own chat
template, which adds a "Thinking effort level" system message the training data never had.

  python inky_proxy.py --squid-port 18330 --port 18331 [--effort 0.0]   (WSL tinker venv: tml-renderers is Linux-only)

Per request: messages -> renderer.build_generation_prompt(effort) -> token ids -> text (re-encodes to the same ids
with the checkpoint's tokenizer.json; squid parses special tokens) -> squid /v1/completions (sampling settings
passed through, MTP off) -> the raw completion split into reasoning (content_thinking) and content (content_text).
"""
import argparse
import json
import re
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from tinker_cookbook import renderers, tokenizer_utils

ap = argparse.ArgumentParser()
ap.add_argument("--squid-port", type=int, default=18330)
ap.add_argument("--port", type=int, default=18331)
ap.add_argument("--effort", type=float, default=0.0)
a = ap.parse_args()
MODEL = "thinkingmachines/Inkling-2-Mini"
tok = tokenizer_utils.get_tokenizer(MODEL)
renderer = renderers.get_renderer("tml_v0", tok, model_name=MODEL)
END = r"(<\|content_[a-z_]+\|>|<\|end_message\|>|<\|content_model_end_sampling\|>|$)"


def part(raw: str, tag: str) -> str:
    m = re.search(r"<\|" + tag + r"\|>(.*?)" + END, raw, re.S)
    return m.group(1) if m else ""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send(self, code: int, obj: dict):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        self.send(200, {"status": "ok"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        msgs = [{"role": m["role"], "content": m["content"]} for m in body["messages"]]
        text = tok.decode(renderer.build_generation_prompt(msgs, effort=a.effort).to_ints())
        req = {"model": "inkling-2-mini", "prompt": text, "max_tokens": body.get("max_tokens", 1024),
               "temperature": body.get("temperature", 0), "top_p": body.get("top_p", 1),
               "top_k": body.get("top_k", 0) or 0, "seed": body.get("seed", 523), "mtp": 0, "stream": False}
        try:
            r = urllib.request.Request(f"http://127.0.0.1:{a.squid_port}/v1/completions", json.dumps(req).encode(),
                                       {"Content-Type": "application/json"})
            d = json.loads(urllib.request.urlopen(r, timeout=3600).read())
        except Exception as e:  # squid errors reach the client as HTTP 502
            self.send(502, {"error": f"{type(e).__name__}: {e}"})
            return
        raw = d["choices"][0]["text"]
        msg = {"role": "assistant", "content": part(raw, "content_text"), "reasoning_content": part(raw, "content_thinking")}
        self.send(200, {"object": "chat.completion", "model": body.get("model"), "usage": d.get("usage"),
                        "choices": [{"index": 0, "message": msg, "finish_reason": d["choices"][0].get("finish_reason")}],
                        "timings": d.get("timings")})


ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()

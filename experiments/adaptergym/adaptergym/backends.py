"""Model servers the gym can bring up for one step. Each exposes an OpenAI-style chat base URL.

  llama  PrismML/upstream llama-server on Windows (Bonsai GGUFs): optional MTP draft flags, LoRA GGUFs either applied
         (one adapter) or loaded unapplied and picked per request (generate.py --adapter/--adapters).
  squid  squid5090 in WSL (Inkling 2 Mini safetensors) with SQUID_OVERLAY files (a folded LoRA), behind
         inky_proxy.py, which renders conversations with the Tinker tml_v0 renderer the adapters were trained on.
"""
from __future__ import annotations

import json
import subprocess
import time
import urllib.request
from pathlib import Path


def _get(url: str, timeout: float = 5) -> str:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.read().decode("utf-8", "replace")
    except Exception:
        return ""


def _healthy(url: str) -> bool:
    return "ok" in _get(url)


class Llama:
    def __init__(self, cfg: dict, port: int, log: Path, slots: int = 1, ctx: int = 16384,
                 lora: str | None = None, loras: dict[str, str] | None = None):
        self.cfg, self.port, self.log, self.slots, self.ctx = cfg, port, log, slots, ctx
        self.lora, self.loras = lora, loras or {}
        self.proc: subprocess.Popen | None = None
        self.runtime = cfg["runtime"]
        self.spec_off = False

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def _cmd(self, spec: bool) -> list[str]:
        c = self.cfg
        cmd = [str(Path(c["bin"]) / "llama-server.exe"), "-m", c["model"], "--alias", c["name"], "--host", "127.0.0.1",
               "--port", str(self.port), "-ngl", "99", "--load-mode", "none", "--ctx-size", str(self.ctx),
               "-np", str(self.slots), "-ctk", "q8_0", "-ctv", "q8_0", "-fa", "on", "-b", "256", "-ub", "128", "--jinja",
               "--reasoning", "off", "--reasoning-budget", "0", "--no-reasoning-preserve", "--min-p", "0",
               "--seed", "523", "--temp", "0", "--metrics"]
        if spec:
            cmd += c.get("spec_args", [])
        if self.lora:
            cmd += ["--lora", self.lora]
        if self.loras:
            cmd += ["--lora", ",".join(self.loras.values()), "--lora-init-without-apply"]
        return cmd

    def __enter__(self):
        self.log.parent.mkdir(parents=True, exist_ok=True)
        for spec in ([True, False] if self.cfg.get("spec_args") else [False]):
            with open(self.log, "w", encoding="utf-8") as lf:
                self.proc = subprocess.Popen(self._cmd(spec), stdout=lf, stderr=subprocess.STDOUT)
            for _ in range(900):
                if _healthy(f"http://127.0.0.1:{self.port}/health"):
                    print(f"llama ready on {self.port} (spec {'on' if spec else 'off'})", flush=True)
                    self._warm()
                    return self
                if self.proc.poll() is not None:
                    break
                time.sleep(2)
            self.stop()
            if spec:
                self.spec_off = True
                self.runtime += " [MTP off: server refused it]"
                print("llama refused MTP flags, retrying without", flush=True)
        raise RuntimeError(f"llama-server did not start; see {self.log}")

    def _warm(self):
        body = {"model": "x", "max_tokens": 8, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "user", "content": "ok"}]}
        req = urllib.request.Request(self.url + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(req, timeout=600).read()
        except Exception:
            pass

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None

    def __exit__(self, *exc):
        self.stop()


class Squid:
    WSL = ["wsl.exe", "-d", "Ubuntu-24.04", "-u", "root", "--"]

    def __init__(self, cfg: dict, port: int, log: Path, overlays: list[str] | None = None, ctx: int = 16384):
        self.cfg, self.port, self.log, self.overlays, self.ctx = cfg, port, log, overlays or [], ctx
        self.proxy_port = port + 1
        self.proc: subprocess.Popen | None = None
        self.runtime = cfg["runtime"] + (f" + overlays {','.join(Path(o).name for o in self.overlays)}" if self.overlays else "")

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.proxy_port}/v1"

    def __enter__(self):
        self.log.parent.mkdir(parents=True, exist_ok=True)
        c = self.cfg
        args = [c["up_script"], c["bin"], c["model_dir"], str(self.port), str(self.proxy_port), str(self.ctx),
                ",".join(self.overlays) or "-", c["proxy"], c["proxy_python"], str(c.get("effort", 0.0))]
        with open(self.log, "w", encoding="utf-8") as lf:
            self.proc = subprocess.Popen(self.WSL + ["bash"] + args, stdout=lf, stderr=subprocess.STDOUT)
        for _ in range(240):
            if _healthy(f"http://127.0.0.1:{self.port}/health") and _healthy(f"http://127.0.0.1:{self.proxy_port}/health"):
                print(f"squid ready on {self.port}, proxy {self.proxy_port}", flush=True)
                return self
            if self.proc.poll() is not None:
                break
            time.sleep(5)
        self.stop()
        raise RuntimeError(f"squid or its proxy did not start; see {self.log}")

    def stop(self):
        for pat in ("squid_up.sh", Path(self.cfg["bin"]).name + " serve", "inky_proxy.py"):
            subprocess.run(self.WSL + ["pkill", "-f", pat], capture_output=True)
        if self.proc:
            try:
                self.proc.wait(30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None

    def __exit__(self, *exc):
        self.stop()


def make(backends: dict, name: str, port: int, log: Path, **kw):
    cfg = backends[name]
    if cfg["kind"] == "llama":
        return Llama(cfg, port, log, **{k: v for k, v in kw.items() if k in ("slots", "ctx", "lora", "loras")})
    if cfg["kind"] == "squid":
        return Squid(cfg, port, log, **{k: v for k, v in kw.items() if k in ("overlays", "ctx")})
    raise ValueError(f"unknown backend kind {cfg['kind']}")

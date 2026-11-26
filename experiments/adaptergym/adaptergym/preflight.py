"""Check everything a plan names before a detached launch, so a run does not die hours later on a missing file.

Catches the failure that killed the 2026-11-25 overnight run: a venv whose base interpreter lives in the Claude
desktop app's virtualized storage (AppData\\Local\\Packages\\Claude_*\\LocalCache) works for tool calls but not for
processes started outside the app.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from . import steps
from .backends import Squid


def _venv_problem(python: str) -> str | None:
    p = Path(python)
    if not p.exists():
        return "missing"
    cfg = p.parent.parent / "pyvenv.cfg"
    if cfg.exists():
        txt = cfg.read_text(encoding="utf-8", errors="replace")
        if "\\Packages\\" in txt and "LocalCache" in txt:
            return "base interpreter is inside the Claude app's virtualized LocalCache (fails outside the app)"
        home = next((l.split("=", 1)[1].strip() for l in txt.splitlines() if l.startswith("home")), "")
        if home and not (Path(home) / "python.exe").exists():
            return f"base interpreter missing: {home}"
    return None


def _wsl_exists(path: str) -> bool:
    return subprocess.run(Squid.WSL + ["test", "-e", path], capture_output=True).returncode == 0


def check(plan: dict) -> int:
    errs, P = [], plan["paths"]
    for key in ("gym_python", "suite_python", "train_python"):
        if key in P and (why := _venv_problem(P[key])):
            errs.append(f"paths.{key} {P[key]}: {why}")
    for key in ("node", "bench", "suite", "bonsai_gym"):
        if key in P and not Path(P[key]).exists():
            errs.append(f"paths.{key} missing: {P[key]}")
    for name, b in plan["backends"].items():
        if b["kind"] == "llama":
            for f in (Path(b["bin"]) / "llama-server.exe", Path(b["model"])):
                if not f.exists():
                    errs.append(f"backend {name}: missing {f}")
        elif b["kind"] == "squid":
            for f in (b["bin"], b["model_dir"], b["up_script"], b["proxy"], b["proxy_python"]):
                if not _wsl_exists(f):
                    errs.append(f"backend {name}: missing in WSL {f}")
    produced = set()
    for s in plan["steps"]:
        if s["kind"] == "train_bonsai":
            cfg = json.loads((Path(P["bonsai_gym"]) / "configs" / f"{s['config']}.json").read_text())
            produced.add(str(Path(cfg["out_dir"]) / "adapter.gguf").lower())
    for s in plan["steps"]:
        if s.get("backend") and s["backend"] not in plan["backends"]:
            errs.append(f"step {s['id']}: unknown backend {s['backend']}")
        for f in [s.get("adapter")] + list(s.get("loras", {}).values()):
            if f and not Path(f).exists() and str(Path(f)).lower() not in produced:
                errs.append(f"step {s['id']}: adapter {f} missing and no train step makes it")
        for f in s.get("overlays", []):
            if not _wsl_exists(f):
                errs.append(f"step {s['id']}: overlay missing in WSL {f}")
        for d in s.get("after", []):
            if d not in {x["id"] for x in plan["steps"]}:
                errs.append(f"step {s['id']}: after {d} is not a step")
        if s["kind"] == "generate" and "rows" not in s:
            for r, _ in steps._gen_jobs(plan, s):
                if not r.exists():
                    errs.append(f"step {s['id']}: rows missing {r}")
    for e in errs:
        print("PREFLIGHT", e)
    print(f"preflight: {len(errs)} problem(s)")
    return len(errs)

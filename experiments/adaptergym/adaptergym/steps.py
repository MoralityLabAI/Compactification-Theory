"""Step kinds. Each has done(plan, step) -> bool and run(plan, step) (the GPU part runs inside a vramorama lease).

  generate      eval splits or sample pools through a backend with generate.py (resumable, power-cap aware):
                {"backend", "arm", "families": [...]} writes <work>/gen/<arm>/<family>.jsonl from data/<family>/eval.jsonl,
                or {"rows", "out"} for one file; optional "adapter" (LoRA GGUF applied), "loras" + "use" (unapplied,
                picked per request), "overlays" (squid), "temperature", "top_p", "samples", "max_tokens", "slots", "ctx"
  score         score_all.py score --arm <arm> (bdb by bitcoin-defi-bench score_sft_replies.ts; srb/math/logic by the
                suite scorers), no GPU
  bench         bitcoin-defi-bench agent scripts (bonding, dlc, netting; development seeds) against a backend:
                {"backend", "name", optional "adapter"/"overlays"}
  train_bonsai  bonsai-gym scripts/canary.py --config <config> (resumes checkpoint.pt; done when result.json exists)
  tinker_sft    inky-tinker/tinker_sft.py in WSL (Tinker API; no local GPU)
  rft_rows      score_all.py rft: keep each pool prompt's shortest correct finished sample as a training row, then
                assemble data/<family>/train.jsonl ({"family", "val": N held out, "gold": prepend gold_train.jsonl})
  router        onebit_suite/router.py (TRM router over the per-arm scores; CPU torch from paths.train_python)
  hillclimb     one evaluation move per arm in a DataGoblin hill-climb lane, driven by the model under test through
                the data-goblin GDK app: {"backend", "lane", "arms": [{"label", optional "adapter", "alias" (served
                model id), "args" (text after /datagoblin-hillclimb), "hypothesis"}], optional "conversations" (app
                runs per arm, default 2), "fallback": "script" (datagoblin.hillclimb.drive records the move when no
                app run reached a decision), "bench" {"seed", "count", "limit"} for that script, "ctx", "max_turns",
                "approve"}; paths datagoblin, datagoblin_app, datagoblin_python, datagoblin_state (the ledger). Done
                when every arm has <work>/results/<step>-<label>.json, written once the lane's ledger has a decision
                with a measured value made while that arm was served.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

from . import backends


def lines(p: Path) -> int:
    try:
        return sum(1 for l in p.read_text(encoding="utf-8").split("\n") if l.strip())
    except FileNotFoundError:
        return 0


def _gen_jobs(plan: dict, s: dict) -> list[tuple[Path, Path]]:
    work = Path(plan["work"])
    if "rows" in s:
        return [(Path(s["rows"]), Path(s["out"]))]
    return [(work / "data" / f / "eval.jsonl", work / "gen" / s["arm"] / f"{f}.jsonl") for f in s["families"]]


def _server(plan: dict, s: dict, tag: str):
    cfg = plan["backends"][s["backend"]]
    kw = {"slots": s.get("slots", 4 if s["kind"] == "generate" else 1), "ctx": s.get("ctx", 18432 if s["kind"] == "generate" else 16384)}
    if s.get("adapter"):
        kw["lora"] = s["adapter"]
    if s.get("loras"):
        kw["loras"] = s["loras"]
    if s.get("overlays"):
        kw["overlays"] = s["overlays"]
    if cfg["kind"] == "squid":
        kw["ctx"] = s.get("ctx", 16384)
    named = {**plan["backends"], s["backend"]: {**cfg, "name": s["alias"]}} if s.get("alias") else plan["backends"]
    return backends.make(named, s["backend"], plan.get("port", 18290), Path(plan["work"]) / "logs" / f"gym-{tag}.log", **kw)


def _hc_marker(plan: dict, s: dict, arm: dict) -> Path:
    return Path(plan["work"]) / "results" / f"{s['id']}-{arm['label']}.json"


def _hc_env(P: dict) -> dict:
    root = Path(P["datagoblin"])
    extra = [os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else []
    return {**os.environ, "PYTHONPATH": os.pathsep.join([str(root / "src"), str(root), *extra]), "PYTHONIOENCODING": "utf-8"}


def _hc_ledger(plan: dict, s: dict) -> dict:
    """The lane's ledger status from DataGoblin's own reader, which also checks the hash chain."""
    P = plan["paths"]
    out = subprocess.run([P["datagoblin_python"], "-m", "datagoblin.hillclimb", "--root", P["datagoblin"], "--state-dir",
                          P["datagoblin_state"], "--lane", s["lane"], "--last", "50"],
                         capture_output=True, text=True, encoding="utf-8", cwd=P["datagoblin"], env=_hc_env(P))
    if out.returncode != 0:
        raise RuntimeError(f"ledger status failed ({out.returncode}): {out.stdout[-400:]} {out.stderr[-400:]}")
    return json.loads(out.stdout)


def _hc_closed(status: dict, lane: str, after: int) -> dict | None:
    """The first decision after record `after` that closed a move of `lane` with a measured value."""
    for r in status["recent"]:
        if r["seq"] > after and r["kind"] == "decision" and r.get("lane") == lane and r.get("verdict") != "replicate" \
                and r.get("value") is not None:
            return r
    return None


def _hc_arm(plan: dict, s: dict, arm: dict) -> None:
    P, cfg = plan["paths"], plan["backends"][s["backend"]]
    model = arm.get("alias") or cfg["name"]
    tag = f"{s['id']}-{arm['label']}"
    with _server(plan, {**s, "adapter": arm.get("adapter"), "alias": arm.get("alias")}, tag) as srv:
        start = _hc_ledger(plan, s)["records"]
        closed, how = None, None
        for n in range(1, s.get("conversations", 2) + 1):
            cmd = [P["datagoblin_app"], "run", "--root", P["datagoblin"], "--python", P["datagoblin_python"],
                   "--skill", "datagoblin-hillclimb", "--args", arm.get("args", s["lane"]),
                   "--approve", s.get("approve", "local"), "--base-url", srv.url, "--model", model,
                   "--state-dir", P["datagoblin_state"], "--context", str(s.get("ctx", 32768)),
                   "--max-turns", str(s.get("max_turns", 40))]
            with open(Path(plan["work"]) / "logs" / f"{tag}-app{n}.log", "w", encoding="utf-8") as lf:
                rc = subprocess.run(cmd, cwd=P["datagoblin"], stdout=lf, stderr=subprocess.STDOUT, env=_hc_env(P)).returncode
            closed = _hc_closed(_hc_ledger(plan, s), s["lane"], start)
            print(f"{tag}: app run {n} exit {rc}, decision {closed}", flush=True)
            if closed:
                how = f"app run {n}"
                break
        if closed is None and s.get("fallback") == "script":
            b = s.get("bench", {})
            cmd = [P["datagoblin_python"], "-m", "datagoblin.hillclimb.drive", "--root", P["datagoblin"],
                   "--state-dir", P["datagoblin_state"], "--base-url", srv.url, "--model", model,
                   "--seed", str(b.get("seed", 0)), "--count", str(b.get("count", 120)), "--limit", str(b.get("limit", 120)),
                   "--note", f"adaptergym {plan['name']} {tag}: scripted, no app run reached a decision"]
            if arm.get("hypothesis"):
                cmd += ["--hypothesis", arm["hypothesis"]]
            with open(Path(plan["work"]) / "logs" / f"{tag}-script.log", "w", encoding="utf-8") as lf:
                rc = subprocess.run(cmd, cwd=P["datagoblin"], stdout=lf, stderr=subprocess.STDOUT, env=_hc_env(P)).returncode
            closed = _hc_closed(_hc_ledger(plan, s), s["lane"], start)
            print(f"{tag}: scripted move exit {rc}, decision {closed}", flush=True)
            how = "script" if closed else None
        if closed is None:
            raise RuntimeError(f"{tag}: no decision with a value in the {s['lane']} ledger; see {plan['work']}/logs/{tag}-*.log")
        status = _hc_ledger(plan, s)
        marker = _hc_marker(plan, s, arm)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({
            "step": s["id"], "arm": arm["label"], "lane": s["lane"], "model": model, "adapter": arm.get("adapter"),
            "runtime": srv.runtime, "backend_model": cfg["model"], "backend_sha256": cfg.get("sha256"), "recorded_by": how,
            "decision": closed, "records": [r for r in status["recent"] if r["seq"] > start], "lane_state": status["lanes"][0],
            "ledger": str(Path(P["datagoblin_state"]) / "ledger.jsonl"), "ledger_records": status["records"],
            "t": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1), encoding="utf-8")
        print(f"{tag}: {closed['verdict']} {closed['value']} ({how}) -> {marker}", flush=True)


def done(plan: dict, s: dict) -> bool:
    k = s["kind"]
    if k == "generate":
        n = s.get("samples", 1)
        return all(lines(o) >= lines(r) * n for r, o in _gen_jobs(plan, s))
    if k == "score":
        sc = Path(plan["work"]) / "scores" / s["arm"]
        gen = Path(plan["work"]) / "gen" / s["arm"]
        fams = [p.stem for p in gen.glob("*.jsonl")]
        return bool(fams) and all((sc / f"{f}.jsonl").exists() and (sc / f"{f}.jsonl").stat().st_mtime >= (gen / f"{f}.jsonl").stat().st_mtime for f in fams)
    if k == "bench":
        return (Path(plan["paths"]["bench"]) / "results" / f"v0-netting-{s['name']}-development.md").exists()
    if k == "train_bonsai":
        cfg = json.loads((Path(plan["paths"]["bonsai_gym"]) / "configs" / f"{s['config']}.json").read_text())
        return (Path(cfg["out_dir"]) / "result.json").exists()
    if k == "tinker_sft":
        return (Path(s["out_win"]) / "run.json").exists()
    if k == "rft_rows":
        d = Path(plan["work"]) / "data" / s["family"]
        src = Path(plan["work"]) / "gen" / "rft" / f"{s['family']}.jsonl"
        return (d / "train.jsonl").exists() and src.exists() and (d / "train.jsonl").stat().st_mtime >= src.stat().st_mtime
    if k == "router":
        return (Path(plan["work"]) / "results" / "router" / "router.md").exists()
    if k == "hillclimb":
        return all(_hc_marker(plan, s, arm).exists() for arm in s["arms"])
    raise ValueError(k)


def run(plan: dict, s: dict) -> None:
    k, P = s["kind"], plan["paths"]
    if k == "hillclimb":
        for arm in s["arms"]:  # arms run in plan order, one server each, inside this step's lease
            if _hc_marker(plan, s, arm).exists():
                print("skip", s["id"], arm["label"], flush=True)
                continue
            _hc_arm(plan, s, arm)
    elif k == "generate":
        with _server(plan, s, s["id"]) as srv:
            names = list(s.get("loras", {}))
            for rows, out in _gen_jobs(plan, s):
                out.parent.mkdir(parents=True, exist_ok=True)
                cmd = [P["suite_python"], "-u", str(Path(P["suite"]) / "generate.py"), "--rows", str(rows), "--out", str(out),
                       "--base-url", srv.url, "--model", plan["backends"][s["backend"]]["name"],
                       "--adapter", s.get("use", "none"), "--adapters", ",".join(names),
                       "--max-tokens", str(s.get("max_tokens", 1024)), "--concurrency", str(s.get("slots", 4)),
                       "--temperature", str(s.get("temperature", 0.0)), "--top-p", str(s.get("top_p", 1.0)),
                       "--samples", str(s.get("samples", 1))]
                print("generate", out, flush=True)
                subprocess.run(cmd, check=True)
    elif k == "score":
        subprocess.run([P["suite_python"], str(Path(P["suite"]) / "score_all.py"), "score", "--work", plan["work"], "--arm", s["arm"]],
                       check=True, cwd=P["suite"])
    elif k == "bench":
        cfg = plan["backends"][s["backend"]]
        with _server(plan, s, s["id"]) as srv:
            runtime = srv.runtime + (f" + {s['adapter_label']}" if s.get("adapter_label") else "")
            for fam, script, mt in (("bonding", "run_bonding_agent.ts", 768), ("dlc", "run_dlc_agent.ts", 768),
                                    ("netting", "run_netting_agent.ts", 1024)):
                out = f"v0-{fam}-{s['name']}-development"
                if (Path(P["bench"]) / "results" / f"{out}.md").exists():
                    print("skip", out, flush=True)
                    continue
                print("bench", out, flush=True)
                log = Path(plan["work"]) / "logs" / f"{out}.log"
                with open(log, "w", encoding="utf-8") as lf:
                    subprocess.run([P["node"], "--experimental-strip-types", f"scripts/{script}", "--block", "development",
                                    "--name", s["name"], "--out", out, "--base-url", srv.url, "--model", cfg["name"],
                                    "--model-sha256", cfg["sha256"], "--runtime", runtime, "--temperature", "0", "--top-p", "1",
                                    "--top-k", "1", "--seed", "523", "--thinking", "false", "--max-tokens", str(mt)],
                                   cwd=P["bench"], stdout=lf, stderr=subprocess.STDOUT, check=True)
    elif k == "train_bonsai":
        subprocess.run([P["train_python"], "-u", "scripts/canary.py", "--config", f"configs/{s['config']}.json"],
                       cwd=P["bonsai_gym"], check=True, env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
    elif k == "tinker_sft":
        subprocess.run(backends.Squid.WSL + ["bash", P["tinker_wpy"], "-u", P["tinker_sft"]] + s["args"], check=True)
    elif k == "rft_rows":
        d = Path(plan["work"]) / "data" / s["family"]
        subprocess.run([P["suite_python"], str(Path(P["suite"]) / "score_all.py"), "rft", "--work", plan["work"],
                        "--family", s["family"], "--val", str(s.get("val", 0))], check=True, cwd=P["suite"])
        parts = ([d / "gold_train.jsonl"] if s.get("gold") else []) + [d / "rft_train.jsonl"]
        (d / "train.jsonl").write_text("".join(x.read_text(encoding="utf-8") for x in parts), encoding="utf-8")
        if s.get("val"):
            (d / "val.jsonl").write_text((d / "rft_val.jsonl").read_text(encoding="utf-8"), encoding="utf-8")
        print("rows", s["family"], {x.name: lines(x) for x in (d / "train.jsonl", d / "val.jsonl")}, flush=True)
    elif k == "router":
        w = Path(plan["work"])
        subprocess.run([P["train_python"], str(Path(P["suite"]) / "router.py"), "--data", str(w / "data"),
                        "--scores", str(w / "scores"), "--out", str(w / "results" / "router")], check=True, cwd=P["suite"])
    else:
        raise ValueError(k)

"""adaptergym: one plan file drives adapter training and evals across model backends, one vramorama lease per GPU step.

  python -m adaptergym preflight <plan.json>                 check every path, venv and backend the plan names
  python -m adaptergym status    <plan.json> [--tags a,b]    done / pending per step
  python -m adaptergym run       <plan.json> [--tags a,b]    run pending steps in order (GPU steps inside leases)
  python -m adaptergym launch    <plan.json> [--tags a,b] [--after-pid N]
                                                            preflight, then `run` detached (survives the session);
                                                            --after-pid waits for that process (e.g. an earlier run) to exit
  python -m adaptergym step      <plan.json> <step id>       one step in this process (what a lease runs)

Run with a stdlib-only interpreter that exists outside the Claude app container (plan paths.gym_python).
Steps are idempotent: a step whose outputs exist is skipped, and every generator resumes from its output file.
Events go to <work>/logs/gym-events.jsonl; each launch writes <work>/logs/gym-run-<time>.log.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from . import preflight as pf
from . import steps

HERE = Path(__file__).resolve().parent


def load(p: str) -> dict:
    plan = json.loads(Path(p).read_text(encoding="utf-8-sig"))
    plan["_path"] = str(Path(p).resolve())
    return plan


def selected(plan: dict, tags: str) -> list[dict]:
    want = {t for t in tags.split(",") if t}
    return [s for s in plan["steps"] if not want or want & set(s.get("tags", [])) or s["id"] in want]


def event(plan: dict, **rec):
    rec["t"] = time.strftime("%Y-%m-%d %H:%M:%S")
    log = Path(plan["work"]) / "logs" / "gym-events.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print("===", json.dumps(rec), flush=True)


def wait_pid(pid: int) -> None:
    """Block until process `pid` exits (Windows), so a queued run never overlaps an earlier one."""
    import ctypes
    k32 = ctypes.windll.kernel32
    h = k32.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE
    if h:
        k32.WaitForSingleObject(h, 0xFFFFFFFF)
        k32.CloseHandle(h)


def cmd_run(plan: dict, tags: str, after_pid: int = 0) -> int:
    if after_pid:
        event(plan, state="waiting-for-pid", pid=after_pid)
        wait_pid(after_pid)
    bad = 0
    for s in selected(plan, tags):
        if steps.done(plan, s):
            event(plan, step=s["id"], state="done-before")
            continue
        missing = [d for d in s.get("after", []) if not steps.done(plan, next(x for x in plan["steps"] if x["id"] == d))]
        if missing:
            event(plan, step=s["id"], state="blocked", needs=missing)
            bad += 1
            continue
        for attempt in range(1, s.get("retries", 3) + 1):
            event(plan, step=s["id"], state="start", attempt=attempt)
            if s.get("vram"):
                cmd = ["vramorama", "run", "--vram", s["vram"], "--owner", f"gym-{s['id']}", "--guard-power", "1800",
                       "--restart", "20", "--", plan["paths"]["gym_python"], "-m", "adaptergym", "step", plan["_path"], s["id"]]
                rc = subprocess.run(cmd, cwd=HERE.parent).returncode
            else:
                rc = subprocess.run([sys.executable, "-m", "adaptergym", "step", plan["_path"], s["id"]], cwd=HERE.parent).returncode
            ok = steps.done(plan, s)
            event(plan, step=s["id"], state="done" if ok else "failed", rc=rc, attempt=attempt)
            if ok:
                break
        else:
            bad += 1
    event(plan, state="run-complete", tags=tags, failed_or_blocked=bad)
    return 1 if bad else 0


def cmd_launch(plan: dict, tags: str, after_pid: int = 0) -> int:
    if pf.check(plan):
        print("preflight failed; not launching")
        return 2
    # one log per launch: a running launch's cmd holds its log open, so a second `>>` to it would fail silently
    log = Path(plan["work"]) / "logs" / f"gym-run-{time.strftime('%Y%m%d-%H%M%S')}.log"
    inner = f'"{plan["paths"]["gym_python"]}" -u -m adaptergym run "{plan["_path"]}"' + (f" --tags {tags}" if tags else "") + (f" --after-pid {after_pid}" if after_pid else "")
    ps = ("$s = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ EnvironmentVariables = "
          "[string[]](Get-ChildItem env: | ForEach-Object { \"$($_.Name)=$($_.Value)\" }) }; "
          f"$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{{ CommandLine = 'cmd /c \"{inner} >> \"{log}\" 2>&1\"'; "
          f"CurrentDirectory = '{HERE.parent}'; ProcessStartupInformation = $s }}; \"rc=$($r.ReturnValue) pid=$($r.ProcessId)\"")
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True)
    print(out.stdout.strip() or out.stderr.strip(), "log:", log)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="adaptergym")
    ap.add_argument("cmd", choices=["preflight", "status", "run", "launch", "step"])
    ap.add_argument("plan")
    ap.add_argument("step_id", nargs="?")
    ap.add_argument("--tags", default="")
    ap.add_argument("--after-pid", type=int, default=0)
    a = ap.parse_args()
    plan = load(a.plan)
    if a.cmd == "preflight":
        return 1 if pf.check(plan) else 0
    if a.cmd == "status":
        for s in selected(plan, a.tags):
            print(f"{'done   ' if steps.done(plan, s) else 'pending'}  {s['id']:<28} {s['kind']:<13} {','.join(s.get('tags', []))}")
        return 0
    if a.cmd == "run":
        return cmd_run(plan, a.tags, a.after_pid)
    if a.cmd == "launch":
        return cmd_launch(plan, a.tags, a.after_pid)
    s = next(x for x in plan["steps"] if x["id"] == a.step_id)
    steps.run(plan, s)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    raise SystemExit(main())

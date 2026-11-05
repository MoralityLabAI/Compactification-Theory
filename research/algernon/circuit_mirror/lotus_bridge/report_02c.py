"""Render the H-A2c run directory into results/algernon-02c: progress.md plus compact receipts.

Reads only what the run wrote: the data manifest, each phase's train_manifest.json, depth_curve.json
summaries and the 02B-instrument manifests and metrics when they exist. Copies the small JSON receipts
(no checkpoints, captures or per-case parts). Run it as often as wanted; it reflects the state on disk.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

PHASES = (("cot", "Stage 1: explicit CoT"), ("armA", "Arm A: published (after-loop supervision)"),
          ("armB", "Arm B: loop-tied (supervision at loop k)"))
DEPTHS = (("depth_cot", "stage-1 CoT"), ("depth_armA", "arm A"), ("depth_armB", "arm B"))
CAPS = (("cap_armA", "arm A"), ("cap_armB", "arm B"))


def ci(v):
    return f"[{v[0]:+.3f}, {v[1]:+.3f}]"


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def phase_section(name, title, m):
    if m is None:
        return [f"### {title}", "", "Not started.", ""]
    lines = [f"### {title}", "",
             f"Status `{m['status']}`; process starts {m['attempts']}; power-cap pauses {len(m['pauses'])}; "
             f"out-of-memory pauses {len(m.get('faults', []))}."]
    if m.get("final_sha256"):
        lines[-1] += f" Final weights `{m['final_sha256'][:12]}` after {m['epochs_trained']} epochs."
    lines += ["", "| epoch | stage | train loss | val n | val acc | depth 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | minutes (last process) |",
              "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for e in sorted(m["epochs_done"], key=lambda e: e["epoch"]):
        by = e["val_accuracy_by_depth"]
        lines.append(f"| {e['epoch']} | {e['stage']} | {e['train_loss_mean']:.4f} | {e['val_n']} | {e['val_accuracy']:.3f} | "
                     + " | ".join(f"{by[str(d)]:.2f}" if str(d) in by else "" for d in range(1, 9))
                     + f" | {e['seconds'] / 60:.1f} |")
    if m.get("faults"):
        lines += ["", "Out-of-memory pauses (resumed from the last checkpoint): "
                  + "; ".join(f"{f['at'][11:16]} epoch {f['epoch']} step {f['step']}" for f in m["faults"]) + "."]
    return lines + [""]


def depth_section(curves):
    rows = [(label, c) for label, c in curves if c]
    if not rows:
        return ["### Behavioural depth curve", "", "Not run yet.", ""]
    lines = ["### Behavioural depth curve (frozen roster, held-out families, d steps asked)", "",
             "| model | " + " | ".join(str(d) for d in range(1, 17)) + " | held-out 9-16 [95% CI] | long_cycle 9-16 |",
             "| --- |" + " ---: |" * 18]
    for label, c in rows:
        s = c["summary"]
        h = s["accuracy_by_depth"]["heldout"]
        lines.append(f"| {label} | " + " | ".join(f"{h[str(d)]:.2f}" for d in range(1, 17))
                     + f" | {s['heldout_9_16']['accuracy']:.3f} [{s['heldout_9_16']['ci95'][0]:.2f}, "
                       f"{s['heldout_9_16']['ci95'][1]:.2f}] | {s['long_cycle_9_16']['accuracy']:.3f} |")
    return lines + ["", "Latent models run max(8, d) blocks and loops; the CoT model decodes freely.", ""]


def cap_section(caps):
    lines = ["### 02B instrument on the trained arms", ""]
    any_ = False
    for label, run, metrics in caps:
        if run is None:
            continue
        any_ = True
        lines.append(f"- {label}: capture `{run['status']}`"
                     + (f", parity {'passed' if run['stages'].get('parity', {}).get('passed') else 'FAILED'}"
                        if run["stages"].get("parity") else ""))
        if metrics:
            g = metrics["gate"]
            lines.append(f"  gate `{g['classification']}`: primary `{g['primary_view']}` {g['primary_acc_9_16']:.3f} vs "
                         f"`{g['strongest_no_recurrence_control']}` {g['control_acc_9_16']:.3f} "
                         f"(diff {g['vs_strongest_control']['diff']:+.3f} {ci(g['vs_strongest_control']['ci95'])}); "
                         f"long_cycle vs pass0 {g['long_cycle_vs_pass0']['diff']:+.3f} {ci(g['long_cycle_vs_pass0']['ci95'])}")
    if not any_:
        lines.append("Not run yet.")
    return lines + [""]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    root = a.run_root
    data = load(root / "data" / "pointer_data.json")
    status = (root / "chain.status").read_text(encoding="utf-8-sig").strip() if (root / "chain.status").exists() else "?"
    lines = ["# Algernon-02C progress: LOTUS's own recipe on the pointer task", "",
             f"Rendered {time.strftime('%Y-%m-%d %H:%M %z')} from the run directory by `report_02c.py`. Protocol: "
             "[EXPERIMENT.md, H-A2c](../../../EXPERIMENT.md#h-a2c-lotuss-own-recipe-on-the-pointer-task-loop-to-step-tying-registered-2026-11-04-before-any-training). "
             f"Chain status: `{status}`.", ""]
    if data:
        p = data["parameters"]
        lines += [f"Data: {p['n_train']} train / {p['n_val']} validation `{p['family']}` cases, depth 1-{p['max_depth']}, "
                  f"seed {p['seed']}; train `{data['train_sha256'][:12]}`, validation `{data['val_sha256'][:12]}`. "
                  "Validation accuracy is the model's final answer with its trained blocks and loops.", ""]
    for name, title in PHASES:
        m = load(root / name / "train_manifest.json")
        lines += phase_section(name, title, m)
        if m:
            shutil.copyfile(root / name / "train_manifest.json", a.out / f"train_{name}.json")
    curves = [(label, load(root / d / "depth_curve.json")) for d, label in DEPTHS]
    lines += depth_section(curves)
    for (d, _), (_, c) in zip(DEPTHS, curves):
        if c:
            (a.out / f"{d}.json").write_text(json.dumps({k: v for k, v in c.items() if k != "records"}, indent=1,
                                                         sort_keys=True) + "\n", encoding="utf-8")
    caps = [(label, load(root / d / "run_manifest.json"), load(root / d / "analysis" / "metrics.json")) for d, label in CAPS]
    lines += cap_section(caps)
    cmp_ = load(root / "compare.json")
    if cmp_ and "loop_step_tying" in cmp_:
        t = cmp_["loop_step_tying"]
        lines += ["### Registered decision: does loop-to-step tying matter?", "",
                  f"Arm B minus arm A, held-out behavioural accuracy over d = 9..16: {t['b_minus_a_heldout_9_16']:+.3f} "
                  f"{ci(t['ci95'])} (required >= 0.10 with the lower bound > 0): "
                  f"**{'matters' if t['matters'] else 'not shown'}**.", ""]
        rows = [(label, cmp_[d]["long_cycle_9_16_vs_best_trivial"]) for d, label in DEPTHS
                if "long_cycle_9_16_vs_best_trivial" in cmp_.get(d, {})]
        if rows:
            lines += ["`long_cycle` behavioural accuracy over d = 9..16 minus the best trivial 02B baseline "
                      "(train majority by depth), per case: "
                      + "; ".join(f"{label} {v['diff']:+.3f} {ci(v['ci95'])}" for label, v in rows) + ".", ""]
        shutil.copyfile(root / "compare.json", a.out / "compare.json")
    (a.out / "progress.md").write_text("\n".join(lines), encoding="utf-8")
    print(a.out / "progress.md")


if __name__ == "__main__":
    main()

"""Render the ARG-02 receipt and profile as Markdown tables.

    python -m research.algernon.algebraic_reasoning_gym.arg02.tables --results DIR

Writes DIR/arg02_tables.md from DIR/arg02_receipt.json and DIR/arg02_profile.json.
Numbers only; interpretation lives in REPORT.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _f(x, nd=3):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def _ci(r):
    lo, hi = r["cluster_ci95"]
    return f"[{lo:+.3f}, {hi:+.3f}]"


def arm_order(receipt):
    d = receipt["selection"]["depth"]
    summary = receipt["summary"]
    depths = sorted({int(k.split("@")[1]) for k in summary if k.startswith("loop@")})
    main = [f"loop@{t}" for t in depths if t in (1, 2, 4, 8, 16) or t == d]
    main += [a for a in (f"open@{d}", f"shuffle@{d}", f"shuffle_a1@{d}", f"ensemble@{d}", "ff2b@1", "controller")
             if a in summary]
    main += sorted(a for a in summary if a.startswith("symbolic:"))
    return main


def render(receipt, profile):
    sel = receipt["selection"]
    d = sel["depth"]
    s = receipt["summary"]
    comp = profile["compute"]
    eff = receipt["decisions"]["efficiency_future12_per_e2e_second"]
    out = []
    out.append(f"Selection (development data): shots {sel['shots']}, block {sel['block']}, d* = {d}, "
               f"ensemble size m(d*) = {receipt['ensemble_sizes'][str(d)]}. "
               f"Development future-correct@12: {sel.get('dev_future12_depth1')} at depth 1, "
               f"{sel.get('dev_future12_selected')} at the selection.\n")
    n = next(iter(s.values()))["positives"]
    out.append(f"## Arms on the {n} test positives\n")
    out.append("| Arm | future@12 | accepted@12 | future@1 | greedy valid | greedy accepted | novel future@12 | "
               "negatives accepted | collision pairs same trace | GPU s | end-to-end s | p50 / p95 ms | PFLOP | future@12 per e2e s |")
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for arm in arm_order(receipt):
        r = s[arm]
        c = comp.get(arm, {})
        e = r["future12"] / c["end_to_end_seconds"] if c.get("end_to_end_seconds") else None
        out.append(f"| {arm} | {r['future12']} | {r['accepted12']} | {_f(r.get('future1'))} | {_f(r.get('greedy_valid'))} | "
                   f"{_f(r.get('greedy_accepted'))} | {r['novel_future12']} | {r['negative_accepted']} | "
                   f"{r['collision_pairs_identical_trace']} | {_f(c.get('gpu_seconds'), 1)} | {_f(c.get('end_to_end_seconds'), 1)} | "
                   f"{_f(c.get('latency_ms_p50'), 0)} / {_f(c.get('latency_ms_p95'), 0)} | {_f(c.get('total_pflop'), 3)} | {_f(e, 4)} |")
    out.append("\n## Paired comparisons, future-correct at 12 checks\n")
    out.append("| A vs B | A | B | A only | B only | A - B | cluster 95% CI (per case) | McNemar p |")
    out.append("|---|---:|---:|---:|---:|---:|---|---:|")
    for k, r in receipt["paired_future12"].items():
        out.append(f"| {k.replace('_vs_', ' vs ')} | {r['a_total']} | {r['b_total']} | {r['a_only']} | {r['b_only']} | "
                   f"{r['difference']:+d} | {_ci(r)} | {r['mcnemar_p']:.4f} |")
    out.append("\n## Decision rules\n")
    dec = receipt["decisions"]
    for k in ("H1_depth", "H2a_feedback", "H2b_state", "H2b_state_original_shuffle", "H3a_repeat", "H3b_ff",
              "H3c_efficiency", "H4_symbolic", "mechanism_claim"):
        if k in dec:
            out.append(f"- {k}: **{'met' if dec[k] else 'not met'}**")
    dep = receipt["depth"]
    out.append("\n## Depth curve (loop, test positives)\n")
    out.append("| step | future@12 | future@1 | accepted@1 | greedy valid | greedy accepted | median relative residual | median reduction |")
    out.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for t, r in dep["curve"].items():
        res = dep["residual_relative_norm"].get(t, {}).get("median")
        red = dep["residual_reduction_median"].get(t)
        out.append(f"| {t} | {r['future12']} | {r['future1']} | {r['accepted1']} | {r['greedy_valid']} | "
                   f"{r['greedy_accepted']} | {_f(res)} | {_f(red)} |")
    out.append(f"\nFirst step with a future-correct top-1 answer: {json.dumps(dep['first_success_step_hist'], sort_keys=True)}. "
               f"Found only after step 1: {dep['first_success_after_step1']}; still correct at the last step: "
               f"{dep['first_success_after_step1_kept_at_T']}.\n")
    out.append("| depth vs step 1 | lost | gained |")
    out.append("|---|---:|---:|")
    for k, r in dep["damage"].items():
        out.append(f"| {k} | {r['lost']} | {r['gained']} |")
    pt = receipt["perturbation"]
    out.append(f"\n## Perturbation (kick {pt['sigma']} x RMS after step {pt['step']})\n")
    out.append("| readout step | future@12 | clean future@12 | top-1 agrees with clean | median distance / kick |")
    out.append("|---:|---:|---:|---:|---:|")
    for t, r in pt["readouts"].items():
        out.append(f"| {t} | {r['future12']} | {r['clean_future12']} | {r['top1_agrees_with_clean']} | "
                   f"{_f(pt['distance_over_kick_median'].get(t))} |")
    pa = receipt["patching"]
    out.append(f"\n## Answer-site patching (donor site states, loop to {pa['depth']})\n")
    out.append("| patch step | top-1 kept | top-1 moved to donor's | top-1 changed | future@12 | median KL to clean | p95 KL |")
    out.append("|---:|---:|---:|---:|---:|---:|---:|")
    for t, r in pa["steps"].items():
        out.append(f"| {t} | {r['top1_kept']} | {r['top1_to_donor']} | {r['top1_changed']} | {r['future12']} | "
                   f"{_f(r['kl_median'], 4)} | {_f(r['kl_p95'], 4)} |")
    pr = receipt["probes"]
    out.append(f"\n## Linear probes ({pr['site']}; train {pr['train']}, select {pr['select']}, "
               f"test seen {pr['test_seen']}, novel {pr['test_novel']})\n")
    for name, per in pr["labels"].items():
        out.append(f"**{name}**\n")
        out.append("| step | novel acc | novel majority | novel perm. p | seen acc | seen majority | k | lam |")
        out.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
        for t, r in per.items():
            nv, sn = r["tests"]["novel"], r["tests"]["seen"]
            out.append(f"| {t} | {nv['accuracy']:.3f} | {nv['majority_accuracy']:.3f} | {_f(nv['permutation_p'])} | "
                       f"{sn['accuracy']:.3f} | {sn['majority_accuracy']:.3f} | {r['settings']['k']} | {r['settings']['lam']} |")
        out.append("")
    fam_arms = [a for a in ("loop@1", f"loop@{d}", "ff2b@1", "symbolic:certified_mass_full", "symbolic:frozen_gru") if a in s]
    out.append("## Future-correct@12 by source family\n")
    fams = list(next(iter(s.values()))["by_family"])
    out.append("| family | " + " | ".join(fam_arms) + " |")
    out.append("|---|" + "---:|" * len(fam_arms))
    for f in fams:
        out.append(f"| {f} | " + " | ".join(str(s[a]["by_family"][f]) for a in fam_arms) + " |")
    out.append("\n## ARG-01 external.py cross-check (top-1 loop answers)\n")
    for t, r in receipt["external_crosscheck"]["by_depth"].items():
        out.append(f"- depth {t}: ARG-01 exact accepted {r['arg01_exact_accepted']} = ARG-02 {r['arg02_at1_accepted']}; "
                   f"future {r['arg01_future_correct']} = {r['arg02_at1_future']}")
    peaks = profile.get("peak_allocated_bytes", {})
    if any(v for v in peaks.values()):
        out.append("\n## Peak allocated VRAM per stage (GiB)\n")
        out.append(", ".join(f"{k} {v / 2**30:.2f}" for k, v in peaks.items() if v))
    return "\n".join(out) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", required=True)
    args = p.parse_args()
    root = Path(args.results)
    receipt = json.loads((root / "arg02_receipt.json").read_text(encoding="utf-8"))
    profile = json.loads((root / "arg02_profile.json").read_text(encoding="utf-8"))
    (root / "arg02_tables.md").write_text(render(receipt, profile), encoding="utf-8", newline="\n")
    print("wrote", root / "arg02_tables.md")


if __name__ == "__main__":
    main()

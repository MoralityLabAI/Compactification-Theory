"""NPU gym: can a 4B Qwen on the Ryzen AI NPU turn controller-gathered evidence into predictions?

Reuses H-M1 worlds, controllers and costs (experiments.h_m1_revision). For each fresh-seed world
and controller arm, the evidence the controller chose to buy is rendered as text and the NPU model
predicts the 4-entry truth table. We compare the model with the exact Bayes prediction on the same
evidence, so the gap is the model's reasoning loss rather than the controller's evidence quality.

Usage: python -m npu_gym.npu_gym --model Qwen3-4B-Hybrid --worlds 40 --out DIR
"""
from __future__ import annotations
import argparse, json, re, time, urllib.request
from pathlib import Path
import numpy as np
from experiments import h_m1_revision as H

URL = "http://localhost:13305/api/v1/chat/completions"
START = 700000000          # fresh range, disjoint from train/val/test/confirm (61e7..64e7)
ARMS = ("simple_only", "myopic_expand", "always_expand")


def render(w: H.World, mask: int, outcomes: int, expanded: bool, tool: str = "") -> str:
    names = list(H.SIMPLE) + (list(H.COMPOSED) if expanded else [])
    n = len(names)
    lines = ["A hidden Boolean rule f(x0,x1) is one of these candidates, with prior probabilities:"]
    for i in range(n):
        tt = "".join(str(int(v)) for v in H.TABLE[i])
        lines.append(f"- {names[i]} (truth table for inputs 00,01,10,11 = {tt}): prior {w.prior[i]:.3f}")
    lines.append("Each probe reveals f at one input, flipped with the given noise probability.")
    obs = []
    for i in range(H.NQ):
        if mask & (1 << i):
            obs.append(f"- input {format(i, '02b')}: observed {(outcomes >> i) & 1} (noise {w.noise[i]:.3f})")
    lines.append("Observations:\n" + ("\n".join(obs) if obs else "- none"))
    if tool in ("posterior", "marginals"):
        post = H.posterior(w, mask, outcomes, expanded)
        lines.append("Computed posterior over candidates (exact): " + ", ".join(
            f"{names[i]} {post[i]:.3f}" for i in range(n)))
    if tool == "marginals":
        marg = post @ H.TABLE[:n, :]
        lines.append("Computed P(f=1) at inputs 00,01,10,11 (exact): " + ", ".join(f"{m:.3f}" for m in marg))
    lines.append("Predict f at inputs 00,01,10,11 as the Bayes-optimal MAP labels (each entry 1 if the posterior "
                 "probability that f=1 exceeds 0.5). Reply with only one line, no explanation: ANSWER: <four digits>")
    return "\n".join(lines)


def ask(model: str, prompt: str, max_tokens: int) -> tuple[str, float, int]:
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": "/no_think\n" + prompt}],
                       "temperature": 0, "max_tokens": max_tokens}).encode()
    t = time.time()
    for attempt in range(3):
        try:
            req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
            r = json.load(urllib.request.urlopen(req, timeout=90))
            break
        except OSError:
            if attempt == 2:
                raise
    return r["choices"][0]["message"]["content"], time.time() - t, r.get("usage", {}).get("completion_tokens", 0)


def episode(w: H.World, arm: str):
    rec = H.evaluate_episode(w, arm)
    mask, out, expanded = rec["evidence_mask"], rec["observations"], bool(rec["expanded"])
    return rec, mask, out, expanded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen3-4B-Hybrid")
    ap.add_argument("--worlds", type=int, default=40)
    ap.add_argument("--max-tokens", type=int, default=24)
    ap.add_argument("--tool", default="", choices=["", "posterior", "marginals"], help="append the exact posterior (deterministic-computation action)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    recs = []
    for s in range(START, START + a.worlds):
        w = H.make_world(s)
        for arm in ARMS:
            rec, mask, out, expanded = episode(w, arm)
            text, dt, ntok = ask(a.model, render(w, mask, out, expanded, a.tool), a.max_tokens)
            m = re.findall(r"ANSWER:\s*([01]{4})", text)
            pred = [int(c) for c in m[-1]] if m else None
            truth = H.TABLE[w.truth]
            llm_acc = float(((np.array(pred) == truth) * w.weights).sum()) if pred else 0.0
            recs.append({"seed": s, "arm": arm, "model": a.model, "parsed": pred is not None, "llm_pred": pred,
                         "bayes_pred": rec["predictions"], "bayes_acc": rec["weighted_accuracy"], "llm_acc": llm_acc,
                         "cost": rec["cost"], "llm_utility": llm_acc - H.PENALTY * rec["cost"],
                         "bayes_utility": rec["utility"], "seconds": dt, "completion_tokens": ntok,
                         "agree_with_bayes": pred == rec["predictions"] if pred else False})
        print(s, flush=True)
    (a.out / "records.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    summ = {}
    for arm in ARMS:
        r = [x for x in recs if x["arm"] == arm]
        summ[arm] = {k: float(np.mean([x[k] for x in r])) for k in
                     ("parsed", "llm_acc", "bayes_acc", "llm_utility", "bayes_utility", "cost",
                      "agree_with_bayes", "seconds", "completion_tokens")}
        summ[arm]["n"] = len(r)
    (a.out / "metrics.json").write_text(json.dumps({"model": a.model, "start_seed": START, "worlds": a.worlds, "tool": a.tool,
                                                    "summary": summ}, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()

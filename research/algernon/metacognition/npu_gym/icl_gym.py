"""In-context learning curve on the NPU: does k worked examples (evidence -> Bayes MAP labels) teach a 4B Qwen?

Per-input queries (one 0/1 answer per call) avoid the four-digit emission failure of npu_gym direct answers.
The first version asked for a bare character with a 3-token cap and parsed 0%: the model emits a blank line then prose, so the reply format is an ANSWER line.
Shots come from worlds seeded SHOT_START.., disjoint from the evaluation worlds (EVAL_START..) and from all
H-M1 train/val/test/confirm ranges. Controller arm: myopic_expand. Weights are never updated (NPU is inference-only);
"learning" here is in-context.

Usage: python -m npu_gym.icl_gym --worlds 30 --shots 0 3 6 --out DIR
"""
from __future__ import annotations
import argparse, json, re, time
from pathlib import Path
import numpy as np
from experiments import h_m1_revision as H
from npu_gym.npu_gym import render, ask

SHOT_START, EVAL_START, ARM = 710000000, 700000000, "myopic_expand"


def body(w, ep):
    return render(w, ep["evidence_mask"], ep["observations"], bool(ep["expanded"])).split("\nPredict f at inputs")[0]


def shot_text(i, s):
    w = H.make_world(s)
    ep = H.evaluate_episode(w, ARM)
    lab = "".join(str(int(v)) for v in ep["predictions"])
    return f"Example {i + 1}:\n{body(w, ep)}\nBayes-optimal labels at inputs 00,01,10,11: {lab}\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen3-4B-Hybrid")
    ap.add_argument("--worlds", type=int, default=30)
    ap.add_argument("--shots", type=int, nargs="+", default=[0, 3, 6])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    recs = []
    for k in a.shots:
        pre = "\n".join(shot_text(i, SHOT_START + i) for i in range(k))
        for s in range(EVAL_START, EVAL_START + a.worlds):
            w = H.make_world(s)
            ep = H.evaluate_episode(w, ARM)
            truth = H.TABLE[w.truth]
            preds = []
            tsec = 0.0
            for q in range(4):
                q_text = (f"{pre}\nNow this task:\n{body(w, ep)}\nIs the Bayes-optimal label at input {format(q, '02b')} "
                          f"equal to 1? Reply with only one line, no explanation: ANSWER: 1 or ANSWER: 0")
                text, dt, _ = ask(a.model, q_text, 10)
                tsec += dt
                m = re.search(r"ANSWER:\s*([01])", text)
                preds.append(int(m.group(1)) if m else -1)
            ok = all(p >= 0 for p in preds)
            acc = float(((np.array(preds) == truth) * w.weights).sum()) if ok else 0.0
            recs.append({"shots": k, "seed": s, "parsed": ok, "pred": preds, "bayes_pred": ep["predictions"],
                         "llm_acc": acc, "bayes_acc": ep["weighted_accuracy"], "cost": ep["cost"],
                         "agree": preds == ep["predictions"], "seconds": tsec})
        print("shots", k, "done", flush=True)
    (a.out / "records.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    summ = {}
    for k in a.shots:
        r = [x for x in recs if x["shots"] == k]
        summ[k] = {m: float(np.mean([x[m] for x in r])) for m in ("parsed", "llm_acc", "bayes_acc", "agree", "seconds")}
    (a.out / "metrics.json").write_text(json.dumps({"model": a.model, "arm": ARM, "worlds": a.worlds, "summary": summ}, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()

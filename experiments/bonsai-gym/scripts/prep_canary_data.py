"""Canary data: the reverse-text task from q1-adapters, so Bonsai 2 27B and the
Bonsai 8B QLoRA baseline see the same training rows and the same eval items.

train/val: q1-adapters runs/sft/reverse-text-gold.jsonl (800 env gold rows, the
exact rows the 8B QLoRA adapter was trained on), split by row order.
eval: the 200 env eval items of results/reverse-text/q1-base.jsonl (prompt +
gold answer only), disjoint from the gold training rows.
Also checks our LCS scorer against the scores stored in the q1 result files.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bonsai_gym.data import lcs_reward  # noqa: E402


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q1", default=r"C:\Users\patri\OneDrive\Documents\GitHub\q1-adapters")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "datasets" / "canary" / "reverse_text"))
    ap.add_argument("--n-val", type=int, default=64)
    args = ap.parse_args()
    q1 = Path(args.q1)
    gold_p = q1 / "runs" / "sft" / "reverse-text-gold.jsonl"
    eval_p = q1 / "results" / "reverse-text" / "q1-base.jsonl"
    gold = [json.loads(l) for l in gold_p.read_text(encoding="utf-8").split("\n") if l.strip()]
    ev = [json.loads(l) for l in eval_p.read_text(encoding="utf-8").split("\n") if l.strip()]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def row(r, target):
        return {"id": r["id"], "messages": r["messages"], "target": target, "answer": r["answer"]}

    train = [row(r, r["text"]) for r in gold[:-args.n_val]]
    val = [row(r, r["text"]) for r in gold[-args.n_val:]]
    seen = {r["messages"][1]["content"] for r in gold}
    evrows = sorted((r for r in ev if r.get("k", 0) == 0), key=lambda r: r["id"])
    evrows = [row(r, "<reversed_text>\n" + r["answer"] + "\n</reversed_text>") for r in evrows]
    overlap = sum(r["messages"][1]["content"] in seen for r in evrows)
    files = {}
    for name, rows in (("train", train), ("val", val), ("eval", evrows)):
        p = out / f"{name}.jsonl"
        p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8", newline="\n")
        files[name] = {"rows": len(rows), "sha256": sha(p)}

    checks = {}
    for f in ("q1-base.jsonl", "q1+ours.jsonl"):
        rs = [json.loads(l) for l in (q1 / "results" / "reverse-text" / f).read_text(encoding="utf-8").split("\n") if l.strip()]
        diffs = [abs(lcs_reward(r["text"], r["answer"]) - r["score"]) for r in rs]
        checks[f] = {"rows": len(rs), "exact": sum(d < 1e-9 for d in diffs), "max_abs_diff": max(diffs),
                     "stored_mean": sum(r["score"] for r in rs) / len(rs)}
    commit = None
    head = q1 / ".git" / "HEAD"
    try:
        import subprocess
        commit = subprocess.run(["git", "-C", str(q1), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        pass
    manifest = {"task": "reverse-text (primeintellect/reverse-text, via q1-adapters)", "q1_adapters_commit": commit,
                "sources": {"gold": {"path": str(gold_p), "sha256": sha(gold_p)},
                            "eval": {"path": str(eval_p), "sha256": sha(eval_p)}},
                "files": files, "eval_prompt_overlap_with_gold": overlap, "scorer_check": checks}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8", newline="\n")
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()

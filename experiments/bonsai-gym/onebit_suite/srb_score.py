"""Score replies to SRB-REV-02 rows using the bench's own validation.

score(row, reply_text) -> {"score","action_correct","valid","brier"}.
"""
import argparse, json, os, sys

DEFAULT_ROOT = r"C:\Users\patri\Documents\Codex\2026-11-16\star\outputs\SystemicRiskBench"
_B = {}


def _bench(srb_root=None):
    if not _B:
        root = os.path.abspath(srb_root or os.environ.get("SRB_ROOT", DEFAULT_ROOT))
        if root not in sys.path:
            sys.path.insert(0, root)
        from systemic_risk_bench import agent_revision as ar, finance_revision as fr
        from systemic_risk_bench.scoring import loads_json
        _B.update(ar=ar, fr=fr, loads_json=loads_json)
    return _B


def score(row, reply_text):
    b = _bench()
    ar, fr = b["ar"], b["fr"]
    g = row["gold"]
    out = {"score": 0, "action_correct": False, "valid": False, "brier": None}
    try:
        resp = b["loads_json"](reply_text)
        world = ar.fixture(g["root"])
        forecast, action, _ = fr.parse_response(resp, g["phase"], world, g["mode"])
    except (ValueError, TypeError, KeyError, AttributeError):
        return out
    out["valid"] = True
    out["brier"] = ar.squared(forecast, g["reference_posterior"])
    out["action_correct"] = action == g["optimal_action"]
    out["score"] = int(out["action_correct"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--replies", required=True)
    ap.add_argument("--out")
    ap.add_argument("--srb-root", default=DEFAULT_ROOT)
    a = ap.parse_args()
    _bench(a.srb_root)
    with open(a.rows, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    replies = {}
    with open(a.replies, encoding="utf-8") as f:
        for l in f:
            if l.strip():
                d = json.loads(l)
                replies[d["id"]] = d["reply"]
    items = []
    for r in rows:
        s = score(r, replies.get(r["id"], ""))
        items.append({"id": r["id"], "scenario": r["gold"]["scenario"], "phase": r["gold"]["phase"],
                      "mode": r["gold"]["mode"], "missing_reply": r["id"] not in replies, **s})

    def mean(xs):
        return sum(xs) / len(xs) if xs else None
    print(f"n={len(items)} mean_score={mean([i['score'] for i in items]):.4f} "
          f"valid={mean([int(i['valid']) for i in items]):.4f}")
    for key in ("scenario", "phase", "mode"):
        for v in sorted({i[key] for i in items}):
            sub = [i for i in items if i[key] == v]
            print(f"  {key}={v}: n={len(sub)} score={mean([i['score'] for i in sub]):.4f}")
    if a.out:
        with open(a.out, "w", encoding="utf-8", newline="\n") as f:
            for i in items:
                f.write(json.dumps(i) + "\n")


if __name__ == "__main__":
    main()

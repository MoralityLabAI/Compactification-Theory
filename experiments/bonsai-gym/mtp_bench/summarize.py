"""Summarize run_bench.sh output: per-arm decode tok/s (warm-up excluded), draft acceptance, and whether each reply
matches the plain arm's byte for byte. Writes summary.json beside the inputs."""
import json, statistics, sys
from pathlib import Path
d = Path(sys.argv[1])
arms = {}
for p in sorted(d.glob("*.jsonl")):
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").split("\n") if l.strip()]
    arms[p.stem] = [r for r in rows if not r["warmup"]]
out = {}
plain = {r["id"]: r for r in arms.get("plain", [])}
for arm, rows in arms.items():
    toks = sum(r["predicted_n"] or 0 for r in rows); secs = sum((r["predicted_n"] or 0) / r["tok_s"] for r in rows if r["tok_s"])
    dn = sum(r["draft_n"] or 0 for r in rows); da = sum(r["draft_accepted"] or 0 for r in rows)
    same = sum(1 for r in rows if r["id"] in plain and (r["content"], r["reasoning"]) == (plain[r["id"]]["content"], plain[r["id"]]["reasoning"]))
    out[arm] = {"n": len(rows), "tokens": toks, "tok_s_pooled": toks / secs if secs else None,
                "tok_s_median": statistics.median(r["tok_s"] for r in rows if r["tok_s"]) if rows else None,
                "draft_n": dn, "draft_accepted": da, "acceptance": da / dn if dn else None, "identical_to_plain": same}
if "plain" in out and "mtp25graft" in out and out["plain"]["tok_s_pooled"]:
    out["speedup"] = out["mtp25graft"]["tok_s_pooled"] / out["plain"]["tok_s_pooled"]
(d / "summary.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))

"""Read exact DISTILL-11 archived rows; finite cap observations only."""
import argparse
import json
from fractions import Fraction as F
from pathlib import Path

def profile(rows, q):
    if q <= 1:
        raise ValueError("q must exceed 1")
    groups = {}
    for row in rows:
        root, cap = row["root_id"], row["cap"]
        for mode in ("flags", "marginal"):
            for arm in ("tail", "broad"):
                p = row["policies"]
                a, b = p[mode+"_"+arm], p["literal_"+arm]
                key = (row["split"], root, mode, arm)
                g = groups.setdefault(key, {})
                if cap in g:
                    raise ValueError("duplicate root/cap")
                aw,bw=F(a["work"]),F(b["work"])
                comparable = F(a["utility"]) >= F(b["utility"]) and F(a["false_recommendations"]) <= F(b["false_recommendations"])
                g[cap] = {"candidate_work":str(aw),"literal_work":str(bw),
                          "utility_risk_dominance":comparable,
                          "hit":comparable and bw>0 and q*aw<=bw}
    out=[]
    for key,g in sorted(groups.items()):
        hits=[c for c in sorted(g) if g[c]["hit"]]
        out.append(dict(zip(("split","root","mode","arm"),key)) | {
            "observed_first_hit": hits[0] if hits else None,
            "no_hit_on_grid": not hits, "observations":g,
            "stable_tail_certified":False})
    return out

def main():
    p=argparse.ArgumentParser()
    p.add_argument("run",type=Path)
    p.add_argument("--q",type=F,default=F(2))
    a=p.parse_args()
    paths=sorted((a.run/"rows").glob("*.json"))
    if not paths:
        p.error("no archived rows")
    rows=[json.loads(x.read_text()) for x in paths]
    print(json.dumps({"source":"distill.quotient_campaign.v1","profiles":profile(rows,a.q)},indent=2))

if __name__=="__main__":
    main()

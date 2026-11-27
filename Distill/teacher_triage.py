"""Matched teacher triage for operator-owned research exams.

No model calls, paid authorization or training. Exact outcomes are recomputed
after both rosters finish; independent review flags remain reviewer judgments.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from .core import digest, read_json, render_prompt, require, write_json
from .teacher_qualification import frozen_exam, load_exam, legacy_rows, bind
from .research_oracle import grade_research
from .research_cases import DOCKET
from .review import validate_review

SCHEMA="distill.teacher_triage.v1"

def index(rows):
    result={}
    for p,r in rows:
        require("research" in p and p["packet_id"] not in result,"nonresearch or duplicate packet")
        result[p["packet_id"]]=(p,r)
    require(bool(result),"empty roster")
    return result

def holdout_binding(dev,held):
    require(all("research" in p for p in held),"nonresearch holdout")
    seen={p["research"]["content_sha256"] for p in dev}
    eligible=[p for p in held if p["research"]["family"]!=DOCKET]
    require(bool(eligible),"empty holdout")
    require(not seen.intersection(p["research"]["content_sha256"] for p in eligible),"development/holdout case overlap")
    return {"eligible_packet_ids":sorted(p["packet_id"] for p in eligible),
            "excluded_fixed_docket":sorted(p["packet_id"] for p in held if p["research"]["family"]==DOCKET),
            "content_hashes":sorted({p["research"]["content_sha256"] for p in eligible}),
            "scope":"fresh generated instances; same families, not structural or private confirmation"}

def packet_outcome(p,r,review_dir=None):
    decision=any(q["id"]=="case.solve" for q in p["questions"])
    out={"format_valid":r is not None,"answered_slots":0,"reviewed_slots":0,
         "planned_slots":len(p["questions"]),"exact":"invalid" if decision else "not_applicable",
         "metrics":None,"material_review_flags":[]}
    if r is None:
        return out
    grade=grade_research(p,r)
    if decision:
        check=next(q["exact_check"] for q in grade["checks"] if q["question_id"]=="case.solve")
        require(check!="checker_error","grader failed; stop triage")
        out["exact"]=check
        out["metrics"]=(grade["research_decision"] or {}).get("metrics")
    out["answered_slots"]=sum(a["status"]=="answered" for a in r["answers"])
    if review_dir is not None:
        path=Path(review_dir)/(p["packet_id"]+".json")
        if path.exists():
            record=read_json(path)
            receipt=validate_review(p,r,record)
            out["reviewed_slots"]=receipt["reviewed_questions"]
            out["material_review_flags"]=[row["question_id"]+":"+k for row in record["rows"]
                for k,v in row["scores"].items() if v==0]
    return out

def compare_rows(left,right,*,left_review=None,right_review=None):
    a,b=index(left),index(right)
    require(a.keys()==b.keys() and all(a[k][0]==b[k][0] for k in a),"teacher rosters differ")
    cases=defaultdict(list)
    aggregate={x:Counter() for x in ("neon","inky")}
    by_family={}
    for rid,(p,r) in a.items():
        family=p["research"]["family"]
        group=by_family.setdefault(family,{x:Counter() for x in ("neon","inky")})
        result={"packet_id":rid}
        for arm,response,review in (("neon",r,left_review),("inky",b[rid][1],right_review)):
            q=packet_outcome(p,response,review)
            result[arm]=q
            for c in (aggregate[arm],group[arm]):
                c["packets"]+=1
                c["format_valid"]+=int(q["format_valid"])
                c["planned_slots"]+=q["planned_slots"]
                c["answered_slots"]+=q["answered_slots"]
                c["reviewed_slots"]+=q["reviewed_slots"]
                c["decisions"]+=int(q["exact"]!="not_applicable")
                c["exact_pass"]+=int(q["exact"]=="pass")
                c["material_review_flags"]+=len(q["material_review_flags"])
        cases[(family,p["difficulty"],p["research"]["content_sha256"])].append(result)
    rows=[]
    for (family,difficulty,h),sections in cases.items():
        require(len(sections)==2,"expected complete two-section case")
        decisions=[s for s in sections if s["neon"]["exact"]!="not_applicable"]
        require(len(decisions)==1,"expected one decision section")
        arms=("neon","inky")
        failed=2-sum(decisions[0][arm]["exact"]=="pass" for arm in arms)
        incomplete=sum(s[arm]["answered_slots"]<s[arm]["planned_slots"] for s in sections for arm in arms)
        flags=sum(len(s[arm]["material_review_flags"]) for s in sections for arm in arms)
        priority=3 if failed==2 else 2 if failed==1 else 1 if incomplete or flags else 0
        rows.append({"case_sha256":h,"family":family,"difficulty":difficulty,
                     "priority":priority,"failed_decisions":failed,"incomplete_sections":incomplete,
                     "review_flags":flags,"sections":sections})
    rows.sort(key=lambda r:(-r["priority"],r["difficulty"]!="frontier",r["family"],r["case_sha256"]))
    return {"aggregate":{k:dict(v) for k,v in aggregate.items()},
            "by_family":{f:{k:dict(v) for k,v in arms.items()} for f,arms in by_family.items()},
            "cases":rows,"scope":"paired public development; reviewer-supplied prose flags; no cross-family gap aggregation",
            "training_admission":False}

def shortlist(report,max_cases):
    require(type(max_cases) is int and 1<=max_cases<=31,"invalid shortlist cap")
    result=[]
    for tier in (3,2,1):
        queues=defaultdict(list)
        for row in report["cases"]:
            if row["priority"]==tier:
                queues[row["family"]].append(row)
        while any(queues.values()):
            for f in sorted(queues):
                if queues[f]:
                    result.append(queues[f].pop(0))
                    if len(result)==max_cases:
                        return result
    return result

def prepare_triage(out,*,neon,inky,holdout,max_cases=12,neon_review=None,inky_review=None):
    m,f,left=frozen_exam(neon)
    im,right=legacy_rows(inky)
    require(im["journal_mode"]=="live","live Inky journal required for actual triage")
    hm,hp=load_exam(holdout)
    hold=holdout_binding([p for p,r in left],hp)
    report=compare_rows(left,right,left_review=neon_review,right_review=inky_review)
    selected=shortlist(report,max_cases)
    ids={section["packet_id"] for row in selected for section in row["sections"]}
    root=Path(out); root.mkdir(parents=True,exist_ok=False)
    public=root/"public"; public.mkdir()
    for p,r in left:
        if p["packet_id"] in ids:
            (public/(p["packet_id"]+".txt")).write_text(render_prompt(p),encoding="utf-8")
    manifest=bind({"schema_version":SCHEMA,"neon_exam_id":m["exam_id"],"neon_freeze_id":f["freeze_id"],
          "inky_plan_id":im["plan_id"],"holdout_exam_id":hm["exam_id"],"holdout":hold,
          "report_sha256":digest(report),"selected_cases":[r["case_sha256"] for r in selected],
          "selected_packet_ids":sorted(ids),"paid_authorization":False,"training_admission":False},"triage_id")
    write_json(root/"comparison.json",report)
    write_json(root/"manifest.json",manifest)
    return manifest

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("out","neon","inky","holdout"):
        p.add_argument("--"+name,required=True)
    p.add_argument("--max-cases",type=int,default=12)
    p.add_argument("--neon-review"); p.add_argument("--inky-review")
    a=p.parse_args()
    print(json.dumps(prepare_triage(a.out,neon=a.neon,inky=a.inky,holdout=a.holdout,
          max_cases=a.max_cases,neon_review=a.neon_review,inky_review=a.inky_review),indent=2,sort_keys=True))

if __name__=="__main__":
    main()

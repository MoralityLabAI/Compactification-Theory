"""Operator-side teacher qualification. No provider calls or training."""
from __future__ import annotations
import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import secrets
from .core import (MAX_BYTES, digest, read_json, render_prompt, require,
                   validate_response, write_json, _unique, _reject_constant)
from .inky_free import config, make_plan, load_plan, Journal as ExamJournal
from .research_oracle import grade_research
from .review import template, validate_review
from .program_runner import prepare, Journal as ProgramJournal
from .research_program import Program
from .program_oracle import Oracle, Baseline, frac

EXAM_SEED=2026112805
PAIR_SEED=2026112909
SCHEMA="distill.teacher_qualification.v1"

def bind(x,key):
    return {**x,key:digest(x)}

def verify(x,key):
    require(x[key]==digest({k:v for k,v in x.items() if k!=key}),"manifest binding mismatch")
    require(x["schema_version"]==SCHEMA and x["training_admission"] is False,"invalid manifest")

def sha(b):
    return hashlib.sha256(b).hexdigest()

def prepare_exam(out,teacher,seed=EXAM_SEED):
    require(isinstance(teacher,str) and 0<len(teacher.strip())<=256,"teacher label required")
    root=Path(out)
    root.mkdir(parents=True,exist_ok=False)
    cfg=config("fixture",seed,0,False,research_per_family=6,max_generation=8192,effort=.6)
    p=make_plan(root/"roster",cfg)
    _,packets=load_plan(root/"roster")
    public=root/"public"
    public.mkdir()
    for packet in packets:
        (public/(packet["packet_id"]+".txt")).write_text(render_prompt(packet),encoding="utf-8")
    m=bind({"schema_version":SCHEMA,"kind":"external_exam","teacher_claimed":teacher,
            "plan_id":p["plan_id"],"requests":len(packets),
            "question_slots":sum(len(q["questions"]) for q in packets),
            "training_admission":False},"exam_id")
    write_json(root/"exam.json",m)
    return m

def load_exam(root):
    root=Path(root)
    m=read_json(root/"exam.json")
    verify(m,"exam_id")
    p,packets=load_plan(root/"roster")
    require(m["kind"]=="external_exam" and m["plan_id"]==p["plan_id"]
            and m["requests"]==len(packets)
            and m["question_slots"]==sum(len(q["questions"]) for q in packets),"exam roster mismatch")
    for packet in packets:
        require((root/"public"/(packet["packet_id"]+".txt")).read_text(encoding="utf-8")==render_prompt(packet),
                "public prompt changed")
    return m,packets

def freeze_exam(root,responses):
    root,responses=Path(root),Path(responses)
    m,packets=load_exam(root)
    ids=[p["packet_id"] for p in packets]
    require({p.name for p in responses.iterdir()}=={i+".json" for i in ids},"one raw file per packet required")
    data={}
    for i in ids:
        b=(responses/(i+".json")).read_bytes()
        require(len(b)<MAX_BYTES//2,"oversized external response")
        data[i]=b
    target=root/"frozen"
    target.mkdir(exist_ok=False)
    for i,b in data.items():
        (target/(i+".json")).write_bytes(b)
    f=bind({"schema_version":SCHEMA,"exam_id":m["exam_id"],
            "raw_sha256":{i:sha(b) for i,b in data.items()},
            "training_admission":False},"freeze_id")
    write_json(target/"manifest.json",f)
    return f

def frozen_exam(root):
    root=Path(root)
    m,packets=load_exam(root)
    f=read_json(root/"frozen"/"manifest.json")
    verify(f,"freeze_id")
    require(f["exam_id"]==m["exam_id"] and set(f["raw_sha256"])=={p["packet_id"] for p in packets},"freeze roster mismatch")
    rows=[]
    for p in packets:
        i=p["packet_id"]
        b=(root/"frozen"/(i+".json")).read_bytes()
        require(sha(b)==f["raw_sha256"][i] and len(b)<MAX_BYTES//2,"frozen response changed")
        response=None
        try:
            r=json.loads(b.decode("utf-8"),object_pairs_hook=_unique,parse_constant=_reject_constant)
            validate_response(p,r)
            response=r
        except (ValueError,TypeError,KeyError,RecursionError):
            pass
        rows.append((p,response))
    return m,f,rows

def grade_exam(root):
    m,f,rows=frozen_exam(root)
    counts=Counter()
    groups={}
    receipts=[]
    for p,r in rows:
        decision=any(q["id"]=="case.solve" for q in p["questions"])
        g=groups.setdefault(p["research"]["family"]+"/"+p["difficulty"],Counter())
        for c in (counts,g):
            c["packets"]+=1
            c["decisions"]+=int(decision)
            c["format_valid"]+=int(r is not None)
        if r is None:
            receipts.append({"packet_id":p["packet_id"],"format":"invalid","grade":None})
            continue
        grade=grade_research(p,r)
        metrics=(grade.get("research_decision") or {}).get("metrics") or {}
        check=next((q["exact_check"] for q in grade["checks"] if q["question_id"]=="case.solve"),None)
        for c in (counts,g):
            c["exact_pass"]+=int(check=="pass")
            c["feasible"]+=int(bool(metrics.get("feasible")))
            c["certificate_consistent"]+=int(bool(metrics.get("certificate_consistent")))
            c["answered_slots"]+=sum(a["status"]=="answered" for a in r["answers"])
        receipts.append({"packet_id":p["packet_id"],"format":"valid","grade":grade})
    return {"exam_id":m["exam_id"],"freeze_id":f["freeze_id"],"teacher_claimed":m["teacher_claimed"],
            "planned_slots":m["question_slots"],"counts":dict(counts),
            "groups":{k:dict(v) for k,v in groups.items()},"receipts":receipts,
            "provider_identity":"operator claimed, unverified","semantic_review":"pending",
            "training_admission":False}

def legacy_rows(root):
    j=ExamJournal(root)
    s=j.summary(replay=True)
    require(s["counts"].get("finished",0)==len(j.packets) and not s["counts"].get("pending",0),"legacy journal incomplete")
    rows=[]
    for p in j.packets:
        packet,raw,receipt=j.result(p["packet_id"])
        r=json.loads(raw["text"],object_pairs_hook=_unique,parse_constant=_reject_constant) if receipt["format"]=="valid" else None
        rows.append((packet,r))
    return {"plan_id":j.plan["plan_id"],"journal_mode":j.mode,
            "provenance":"journal replay; model identity requires operator evidence"},rows

def review_rows(rows,directory,*,create=False):
    directory=Path(directory)
    counts=Counter()
    totals=Counter()
    maxima=Counter()
    ids={p["packet_id"]+".json" for p,r in rows if r is not None}
    if create:
        directory.mkdir(parents=True,exist_ok=False)
    else:
        require({x.name for x in directory.iterdir()}<=ids,"unexpected review file")
    for p,r in rows:
        counts["planned_slots"]+=len(p["questions"])
        if r is None:
            continue
        t=template(p,r)
        counts["answered_slots"]+=len(t["rows"])
        path=directory/(p["packet_id"]+".json")
        if create:
            write_json(path,t)
        elif path.exists():
            result=validate_review(p,r,read_json(path))
            counts["reviewed_slots"]+=result["reviewed_questions"]
            totals.update(result["dimension_totals"])
            maxima.update(result.get("dimension_maxima") or
                          {k:result["max_per_dimension"] for k in result["dimension_totals"]})
    return {"counts":dict(counts),"dimension_totals":dict(totals),"dimension_maxima":dict(maxima),
            "all_planned_slots_reviewed":counts["planned_slots"]==counts["reviewed_slots"],
            "scope":"claimed reviewer scores; independent identity and judgments unverified",
            "training_admission":False}

def prepare_pairs(out,*,seed=PAIR_SEED,pairs_per_family=4,fixture=False):
    require(type(pairs_per_family) is int and 1<=pairs_per_family<=16,"invalid pair count")
    root=Path(out)
    root.mkdir(parents=True,exist_ok=False)
    pairs=[]
    for family in ("shared_control","shifted_control"):
        for i in range(pairs_per_family):
            pid=family+"-"+str(i)
            w=secrets.randbits(63)
            runs={}
            for arm,mode in (("neon","external"),("inky","live")):
                mode="fixture" if fixture else mode
                r=prepare(root/pid/arm,seed=seed,ordinal=i,family=family,
                          difficulty="frontier",mode=mode,world_seed=w)
                runs[arm]=r["run_id"]
            pairs.append({"pair_id":pid,"family":family,"ordinal":i,"runs":runs})
    m=bind({"schema_version":SCHEMA,"kind":"paired_programs","seed":seed,"fixture":fixture,
            "pairs":pairs,"training_admission":False},"campaign_id")
    write_json(root/"pairs.json",m)
    return m

def trajectory(j):
    with j.tx() as db:
        env,rows=j.reconstruct(db)
    require(env.state.stage==4 and all(r[1]=="accepted" for r in rows),"incomplete trajectory")
    p=Program(j.manifest["spec"],j.index)
    oracle=Oracle(p.model)
    regret=Fraction(0)
    try:
        for row in rows:
            a=json.loads(row[5])["action"]
            gap=oracle.expected(p.state)-oracle.action_value(p.state,a)
            require(gap>=0,"negative oracle regret")
            regret+=gap
            p.step(a)
        require(p.result()==env.result(),"trajectory replay disagreement")
    finally:
        oracle.close()
    b=Program(j.manifest["spec"],j.index)
    policy=Baseline(b.model,"reuse_first")
    while b.state.stage<4:
        b.step(policy.choose(b.state))
    return {**p.result(),"sum_conditional_action_regret":frac(regret),
            "reuse_first_realized_utility":b.result()["net_utility"],
            "utility_minus_reuse_first":p.result()["net_utility"]-b.result()["net_utility"]}

def compare_pairs(root):
    root=Path(root)
    m=read_json(root/"pairs.json")
    verify(m,"campaign_id")
    require(m["kind"]=="paired_programs","invalid pair manifest")
    ready=[]
    incomplete=[]
    for pair in m["pairs"]:
        a,b=(ProgramJournal(root/pair["pair_id"]/arm) for arm in ("neon","inky"))
        require(a.manifest["run_id"]==pair["runs"]["neon"] and b.manifest["run_id"]==pair["runs"]["inky"]
                and a.manifest["spec"]==b.manifest["spec"] and a.index==b.index
                and a.manifest["spec"]["family"]==pair["family"],"unmatched programs")
        require((a.manifest["mode"],b.manifest["mode"])==
                (("fixture","fixture") if m["fixture"] else ("external","live")),"wrong arm modes")
        sa,sb=a.status(),b.status()
        if sa["complete"] and sb["complete"]:
            ready.append((pair,a,b))
        else:
            incomplete.append({"pair_id":pair["pair_id"],"neon_complete":sa["complete"],
                               "inky_complete":sb["complete"]})
    result={"campaign_id":m["campaign_id"],"fixture":m["fixture"],"planned_pairs":len(m["pairs"]),
            "complete_pairs":len(ready),"incomplete":incomplete,"training_admission":False}
    if incomplete:
        return result
    rows=[]
    for pair,a,b in ready:
        x,y=trajectory(a),trajectory(b)
        rows.append({"pair_id":pair["pair_id"],"family":pair["family"],"neon":x,"inky":y,
                     "paired_utility_delta":x["net_utility"]-y["net_utility"]})
    result["rows"]=rows
    result["family_mean_utility_delta"]={f:frac(Fraction(sum(r["paired_utility_delta"] for r in rows if r["family"]==f),
                                      sum(r["family"]==f for r in rows))) for f in ("shared_control","shifted_control")}
    result["scope"]="paired realized synthetic worlds; regret uses optimal continuations; no matched-runtime or expected-policy-value claim"
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    a=sub.add_parser("prepare-exam"); a.add_argument("--out",required=True); a.add_argument("--teacher",required=True); a.add_argument("--seed",type=int,default=EXAM_SEED)
    a=sub.add_parser("freeze-exam"); a.add_argument("--campaign",required=True); a.add_argument("--responses",required=True)
    a=sub.add_parser("grade-exam"); a.add_argument("--campaign",required=True)
    a=sub.add_parser("review"); a.add_argument("--campaign",required=True); a.add_argument("--directory",required=True); a.add_argument("--legacy",action="store_true"); a.add_argument("--create",action="store_true")
    a=sub.add_parser("prepare-pairs"); a.add_argument("--out",required=True); a.add_argument("--seed",type=int,default=PAIR_SEED)
    a=sub.add_parser("compare-pairs"); a.add_argument("--campaign",required=True)
    a=p.parse_args()
    if a.command=="prepare-exam":
        result=prepare_exam(a.out,a.teacher,a.seed)
    elif a.command=="freeze-exam":
        result=freeze_exam(a.campaign,a.responses)
    elif a.command=="grade-exam":
        result=grade_exam(a.campaign)
    elif a.command=="review":
        if a.legacy:
            m,rows=legacy_rows(a.campaign)
        else:
            m,f,rows=frozen_exam(a.campaign)
            m={"exam_id":m["exam_id"],"freeze_id":f["freeze_id"]}
        result={**m,**review_rows(rows,a.directory,create=a.create)}
    elif a.command=="prepare-pairs":
        result=prepare_pairs(a.out,seed=a.seed)
    else:
        result=compare_pairs(a.campaign)
    print(json.dumps(result,sort_keys=True,indent=2))

if __name__=="__main__":
    main()

"""Budgeted Fable shortlist transport: local planning, reservation, and review.

Never calls an API. Rates, model identity and usage are operator-supplied.
A plan is a local spending envelope, not external billing authorization.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from .batch import cost, dollars, integer, rates, _job
from .core import (MAX_BYTES, RESEARCH_VERSION, digest, read_json, render_prompt,
                   require, split_quiz, string, validate_response, write_json,
                   _unique, _reject_constant)
from .research_cases import compile_research, DOCKET
from .research_oracle import grade_research
from .teacher_qualification import frozen_exam, load_exam, review_rows
from .teacher_triage import SCHEMA as TRIAGE_SCHEMA, holdout_binding, shortlist

SCHEMA="distill.research_paid_plan.v1"
TOTAL=dollars("200")
MIN_RESERVE=dollars("10")
DEFAULT_ACTIVE=dollars("80")

def verify_group(group):
    require(isinstance(group,list) and len(group)==2,"complete two-section research case required")
    p=group[0]
    ins=p["research"]
    parent=compile_research(ins["family"],seed=ins["seed"],ordinal=ins["ordinal"],
                            difficulty=p["difficulty"])
    require(group==split_quiz(parent,max_questions=2),"research section reconstruction mismatch")
    return ins["content_sha256"]

def create_plan(out,*,groups,source,prices,model,rate_evidence,terms_evidence,
                active=DEFAULT_ACTIVE,input_cap=8192,output_cap=8192):
    for value,name in ((model,"model"),(rate_evidence,"rate evidence"),(terms_evidence,"output-use review")):
        string(value,name,1024)
    integer(active,"active budget",1,TOTAL-MIN_RESERVE)
    integer(input_cap,"input token cap",1,10**7)
    integer(output_cap,"output token cap",1,10**7)
    require(isinstance(groups,list) and 1<=len(groups)<=31,"empty or oversized shortlist")
    hashes=[verify_group(group) for group in groups]
    require(len(set(hashes))==len(hashes),"duplicate research case")
    per_job=cost(prices,input_cap,output_cap)
    require(per_job>0,"paid plan requires positive modeled cost")
    require(prices["output_microusd_per_million"]>0,"paid output rate required")
    count=min(len(groups),active//(2*per_job))
    require(count>0,"budget cannot reserve both sections of a case")
    chosen=groups[:count]
    jobs=[_job(p,input_cap,output_cap,prices) for group in chosen for p in group]
    plan={"schema_version":SCHEMA,"library_version":RESEARCH_VERSION,
          "purpose":"reference_evaluation","training_admission":False,
          "paid_api_calls":0,"model_claimed":model,"rate_evidence":rate_evidence,
          "terms_evidence":terms_evidence,"source":source,"rates":prices,
          "budget_microusd":TOTAL,"reserve_microusd":TOTAL-active,
          "authorized_max_microusd":sum(j["max_cost_microusd"] for j in jobs),
          "input_cap":input_cap,"output_cap":output_cap,
          "candidate_cases":hashes,"selected_cases":hashes[:count],"jobs":jobs}
    plan["plan_id"]=digest(plan)
    root=Path(out); root.mkdir(parents=True,exist_ok=False)
    (root/"packets").mkdir(); (root/"public").mkdir()
    for group in chosen:
        for p in group:
            rid=p["packet_id"]
            write_json(root/"packets"/(rid+".json"),p)
            (root/"public"/(rid+".txt")).write_text(render_prompt(p),encoding="utf-8")
    write_json(root/"plan.json",plan)
    return plan

def plan_from_triage(out,*,triage,neon,holdout,**kwargs):
    t=Path(triage)
    m=read_json(t/"manifest.json")
    report=read_json(t/"comparison.json")
    require(m["schema_version"]==TRIAGE_SCHEMA and m["triage_id"]==digest({k:v for k,v in m.items() if k!="triage_id"})
            and m["report_sha256"]==digest(report)
            and m["paid_authorization"] is False and m["training_admission"] is False
            and report["training_admission"] is False,"triage binding mismatch")
    nm,nf,rows=frozen_exam(neon)
    hm,hp=load_exam(holdout)
    require(m["neon_exam_id"]==nm["exam_id"] and m["neon_freeze_id"]==nf["freeze_id"]
            and m["holdout_exam_id"]==hm["exam_id"]
            and m["holdout"]==holdout_binding([p for p,r in rows],hp),"exam/holdout binding mismatch")
    n=len(m["selected_cases"])
    require(n>0,"no selected hard cases; no paid request needed")
    selected=shortlist(report,n)
    require([r["case_sha256"] for r in selected]==m["selected_cases"],"shortlist order mismatch")
    packets={p["packet_id"]:p for p,r in rows}
    groups=[]
    for row in selected:
        g=sorted([packets[s["packet_id"]] for s in row["sections"]],key=lambda p:p["section"]["index"])
        require(verify_group(g)==row["case_sha256"],"case hash mismatch")
        for p in g:
            require((t/"public"/(p["packet_id"]+".txt")).read_text(encoding="utf-8")==render_prompt(p),"triage prompt changed")
        groups.append(g)
    ids=sorted(p["packet_id"] for g in groups for p in g)
    require(ids==m["selected_packet_ids"] and not set(ids).intersection(m["holdout"]["eligible_packet_ids"]),"selection/holdout mismatch")
    return create_plan(out,groups=groups,source={"triage_id":m["triage_id"],
        "neon_freeze_id":nf["freeze_id"],"holdout_exam_id":hm["exam_id"]},**kwargs)

def load_paid_plan(root,plan=None):
    root=Path(root)
    p=read_json(root/"plan.json") if plan is None else plan
    require(p["schema_version"]==SCHEMA and p["library_version"]==RESEARCH_VERSION
            and p["purpose"]=="reference_evaluation" and p["training_admission"] is False
            and p["paid_api_calls"]==0 and p["plan_id"]==digest({k:v for k,v in p.items() if k!="plan_id"}),"invalid research paid plan")
    require(p["budget_microusd"]==TOTAL,"global envelope changed")
    integer(p["reserve_microusd"],"retained budget",MIN_RESERVE,TOTAL-1)
    i=integer(p["input_cap"],"input cap",1,10**7)
    o=integer(p["output_cap"],"output cap",1,10**7)
    for key in ("model_claimed","rate_evidence","terms_evidence"):
        string(p[key],key,1024)
    cap=TOTAL-p["reserve_microusd"]
    per=cost(p["rates"],i,o)
    require(per>0 and p["rates"]["output_microusd_per_million"]>0,"invalid paid rate")
    candidates=p["candidate_cases"]
    require(isinstance(candidates,list) and 1<=len(candidates)<=31 and len(set(candidates))==len(candidates),"invalid case shortlist")
    count=min(len(candidates),cap//(2*per))
    require(count>0 and p["selected_cases"]==candidates[:count] and len(p["jobs"])==2*count,"whole-case selection mismatch")
    jobs={}
    hashes=[]
    for start in range(0,len(p["jobs"]),2):
        group=[]
        for j in p["jobs"][start:start+2]:
            rid=j["request_id"]
            require(isinstance(rid,str) and len(rid)==64 and all(c in "0123456789abcdef" for c in rid) and rid not in jobs,"invalid request ID")
            packet=read_json(root/"packets"/(rid+".json"))
            require(j==_job(packet,i,o,p["rates"]) and packet["packet_id"]==rid,"paid job binding mismatch")
            require((root/"public"/(rid+".txt")).read_text(encoding="utf-8")==render_prompt(packet),"paid prompt changed")
            jobs[rid]=j
            group.append(packet)
        hashes.append(verify_group(group))
    require(hashes==p["selected_cases"] and sum(j["max_cost_microusd"] for j in jobs.values())==p["authorized_max_microusd"]<=cap,"paid plan exceeds envelope")
    return p,jobs

def freeze(root,responses):
    from .ledger import Ledger
    root,responses=Path(root),Path(responses)
    plan,jobs=load_paid_plan(root)
    status=Ledger(root).status()
    require(not status["frozen"] and status["states"]=={"settled":len(jobs)},"all planned requests must be settled within caps")
    require({p.name for p in responses.iterdir()}=={rid+".json" for rid in jobs},"one raw response per paid request required")
    blobs={rid:(responses/(rid+".json")).read_bytes() for rid in jobs}
    require(all(len(b)<MAX_BYTES//2 for b in blobs.values()),"oversized response")
    out=root/"frozen";out.mkdir(exist_ok=False)
    for rid,b in blobs.items():
        (out/(rid+".json")).write_bytes(b)
    f={"plan_id":plan["plan_id"],"ledger_summary":status,
       "raw_sha256":{rid:hashlib.sha256(b).hexdigest() for rid,b in blobs.items()},
       "training_admission":False}
    f["freeze_id"]=digest(f)
    write_json(out/"manifest.json",f)
    return f

def frozen_rows(root):
    from .ledger import Ledger
    root=Path(root)
    plan,jobs=load_paid_plan(root)
    f=read_json(root/"frozen"/"manifest.json")
    require(f["freeze_id"]==digest({k:v for k,v in f.items() if k!="freeze_id"})
            and f["plan_id"]==plan["plan_id"] and f["training_admission"] is False
            and f["ledger_summary"]==Ledger(root).status()
            and set(f["raw_sha256"])==set(jobs),"paid freeze mismatch")
    rows=[]
    for rid in jobs:
        packet=read_json(root/"packets"/(rid+".json"))
        b=(root/"frozen"/(rid+".json")).read_bytes()
        require(len(b)<MAX_BYTES//2 and hashlib.sha256(b).hexdigest()==f["raw_sha256"][rid],"frozen raw response changed")
        r=None
        try:
            q=json.loads(b.decode("utf-8"),object_pairs_hook=_unique,parse_constant=_reject_constant)
            validate_response(packet,q)
            r=q
        except (ValueError,TypeError,KeyError,RecursionError):
            pass
        rows.append((packet,r))
    return plan,f,rows

def grade(root):
    p,f,rows=frozen_rows(root)
    receipts=[]
    for packet,r in rows:
        receipts.append({"packet_id":packet["packet_id"],"format":"invalid" if r is None else "valid",
                         "grade":None if r is None else grade_research(packet,r)})
    decisions=[x for (packet,r),x in zip(rows,receipts) if any(q["id"]=="case.solve" for q in packet["questions"])]
    return {"plan_id":p["plan_id"],"freeze_id":f["freeze_id"],"model_claimed":p["model_claimed"],
            "planned_packets":len(rows),"valid_packets":sum(r is not None for _,r in rows),
            "planned_decisions":len(decisions),"exact_pass":sum(any(q["question_id"]=="case.solve" and q["exact_check"]=="pass" for q in (x["grade"] or {}).get("checks",[])) for x in decisions),
            "receipts":receipts,"semantic_review":"pending","provider_identity":"operator claimed",
            "training_admission":False}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    a=sub.add_parser("plan")
    for k in ("out","triage","neon","holdout","model","rate-evidence","terms-evidence",
              "input-usd-per-million","output-usd-per-million"):
        a.add_argument("--"+k,required=True)
    a.add_argument("--fee-usd-per-request",default="0")
    a.add_argument("--active-usd",default="80")
    a.add_argument("--input-cap",type=int,default=8192);a.add_argument("--output-cap",type=int,default=8192)
    for name in ("freeze","grade","review"):
        a=sub.add_parser(name);a.add_argument("--campaign",required=True)
        if name=="freeze":a.add_argument("--responses",required=True)
        if name=="review":a.add_argument("--directory",required=True);a.add_argument("--create",action="store_true")
    a=p.parse_args()
    if a.command=="plan":
        result=plan_from_triage(a.out,triage=a.triage,neon=a.neon,holdout=a.holdout,
            prices=rates(dollars(a.input_usd_per_million),dollars(a.output_usd_per_million),dollars(a.fee_usd_per_request)),
            model=a.model,rate_evidence=a.rate_evidence,terms_evidence=a.terms_evidence,
            active=dollars(a.active_usd),input_cap=a.input_cap,output_cap=a.output_cap)
    elif a.command=="freeze":result=freeze(a.campaign,a.responses)
    elif a.command=="grade":result=grade(a.campaign)
    else:
        plan,f,rows=frozen_rows(a.campaign)
        result={"plan_id":plan["plan_id"],"freeze_id":f["freeze_id"],**review_rows(rows,a.directory,create=a.create)}
    print(json.dumps(result,indent=2,sort_keys=True))

if __name__=="__main__":
    main()

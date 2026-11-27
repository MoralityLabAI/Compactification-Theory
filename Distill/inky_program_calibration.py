"""Matched free-only Inky sampling calibration, with immutable run profiles.

This module prepares private host directories and reads replayed journals.
It makes no API calls. Original v1 runs are not overwritten or retried.
"""
from __future__ import annotations
import argparse
from fractions import Fraction
import json
from pathlib import Path
import secrets
from .core import digest, read_json, require, write_json
from .program_runner import CALIBRATION_PROFILES, Journal, prepare
from .research_program import Program
from .program_oracle import Oracle, frac

SCHEMA="distill.inky_program_calibration.v1"
SEED=2026113015

def arm_id(cap,effort):
    return f"g{cap}-e{round(effort*100):02d}"

def prepare_calibration(out,*,seed=SEED,roots_per_family=2,fixture=False):
    require(type(roots_per_family) is int and 1<=roots_per_family<=8,"bad root count")
    root=Path(out);root.mkdir(parents=True,exist_ok=False)
    rows=[]
    for family in ("shared_control","shifted_control"):
        for i in range(roots_per_family):
            pair=f"{family}-{i}"
            secret=secrets.randbits(63)
            arms=[]
            for cap,effort in CALIBRATION_PROFILES:
                arm=arm_id(cap,effort)
                result=prepare(root/pair/arm,seed=seed,ordinal=i,family=family,
                    difficulty="frontier",mode="fixture" if fixture else "live",
                    world_seed=secret,max_generation=cap,effort=effort)
                arms.append({"arm":arm,"run_id":result["run_id"],"max_generation":cap,"effort":effort})
            rows.append({"pair":pair,"family":family,"ordinal":i,"arms":arms})
    m={"schema_version":SCHEMA,"seed":seed,"fixture":fixture,"roots_per_family":roots_per_family,
       "roots":rows,"training_admission":False}
    m["campaign_id"]=digest(m)
    write_json(root/"calibration.json",m)
    return m

def load(root):
    root=Path(root)
    m=read_json(root/"calibration.json")
    require(m["schema_version"]==SCHEMA and m["training_admission"] is False
            and m["campaign_id"]==digest({k:v for k,v in m.items() if k!="campaign_id"}),"calibration binding mismatch")
    require(type(m["fixture"]) is bool and type(m["roots_per_family"]) is int
            and 1<=m["roots_per_family"]<=8,"invalid calibration roster")
    expected=[(f,i) for f in ("shared_control","shifted_control") for i in range(m["roots_per_family"])]
    require([(row["family"],row["ordinal"]) for row in m["roots"]]==expected,"root roster mismatch")
    out=[]
    for row in m["roots"]:
        require(row["pair"]==f'{row["family"]}-{row["ordinal"]}',"root ID mismatch")
        require([(a["max_generation"],a["effort"]) for a in row["arms"]]==list(CALIBRATION_PROFILES),"profile roster mismatch")
        matched=None
        for a in row["arms"]:
            require(a["arm"]==arm_id(a["max_generation"],a["effort"]),"bad profile name")
            j=Journal(root/row["pair"]/a["arm"])
            p=j.manifest
            require(p["run_id"]==a["run_id"] and p["max_generation"]==a["max_generation"]
                    and p["effort"]==a["effort"] and p["mode"]==("fixture" if m["fixture"] else "live"),"arm identity mismatch")
            require(p["spec"]["family"]==row["family"],"family mismatch")
            key=(p["spec"],j.index)
            require(matched is None or matched==key,"unpaired world or specification")
            matched=key
            out.append((row,a,j))
    return m,out

def report(root):
    m,arms=load(root)
    rows=[]
    ready=True
    for pair,a,j in arms:
        with j.tx() as db:
            env,attempts=j.reconstruct(db)
        accepted=sum(x[1]=="accepted" for x in attempts)
        invalid=sum(x[1]=="invalid" for x in attempts)
        pending=sum(x[1]=="pending" for x in attempts)
        complete=env.state.stage==4
        terminal=complete or bool(invalid)
        ready=ready and terminal
        row={"pair":pair["pair"],"family":pair["family"],"arm":a["arm"],
             "accepted":accepted,"invalid":invalid,"pending":pending,"complete":complete,
             "terminal":terminal}
        raw=[json.loads(x[3]) for x in attempts if x[3] is not None]
        if not m["fixture"]:
            for key in ("input_tokens","output_tokens","elapsed_ms"):
                row[key]=sum(x[key] for x in raw if type(x.get(key)) is int and x[key]>=0)
                row[key+"_missing"]=len(attempts)-sum(type(x.get(key)) is int and x[key]>=0 for x in raw)
            row["unclean"]=sum(x.get("clean") is not True for x in raw)
            row["at_output_cap"]=sum(x.get("output_tokens")==a["max_generation"] for x in raw)
            row["accepted_per_1000_output_tokens"]=frac(Fraction(1000*accepted,row["output_tokens"])) if row["output_tokens"] and not row["output_tokens_missing"] else None
        rows.append((row,j,env,attempts))
    result={"campaign_id":m["campaign_id"],"fixture":m["fixture"],"planned_arms":len(rows),
            "terminal_arms":sum(r["terminal"] for r,*_ in rows),"all_terminal":bool(ready),
            "training_admission":False}
    # Do not release even partial utility/oracle labels while another arm is
    # being solved. Pending calls need explicit operator reconciliation.
    if not ready:
        result["rows"]=[r for r,*_ in rows]
        return result
    for row,j,env,attempts in rows:
        p=Program(j.manifest["spec"],j.index)
        oracle=Oracle(p.model)
        gaps=[]
        try:
            for attempt in attempts:
                if attempt[1]!="accepted":
                    break
                a=json.loads(attempt[5])["action"]
                gap=oracle.expected(p.state)-oracle.action_value(p.state,a)
                require(gap>=0,"negative conditional gap")
                gaps.append(gap)
                p.step(a)
        finally:
            oracle.close()
        row["accepted_prefix_regret"]=frac(sum(gaps,Fraction(0)))
        row["complete_utility"]=p.result()["net_utility"] if row["complete"] else None
        if row["complete"]:
            require(p.result()==env.result(),"terminal replay disagreement")
    summary={}
    for cap,effort in CALIBRATION_PROFILES:
        arm=arm_id(cap,effort)
        subset=[r for r,*_ in rows if r["arm"]==arm]
        completed=[r for r in subset if r["complete"]]
        v={"planned":len(subset),"completed":len(completed),"accepted":sum(r["accepted"] for r in subset),
           "invalid":sum(r["invalid"] for r in subset),
           "completed_only_mean_utility":frac(Fraction(sum(r["complete_utility"] for r in completed),len(completed))) if completed else None}
        if not m["fixture"]:
            for key in ("input_tokens","output_tokens","elapsed_ms","input_tokens_missing","output_tokens_missing",
                        "elapsed_ms_missing","unclean","at_output_cap"):
                v[key]=sum(r[key] for r in subset)
            v["accepted_per_1000_output_tokens"]=frac(Fraction(1000*v["accepted"],v["output_tokens"])) if v["output_tokens"] and not v["output_tokens_missing"] else None
        summary[arm]=v
    result["rows"]=[r for r,*_ in rows]
    result["profiles"]=summary
    result["scope"]="paired realized worlds; incomplete utility censored; accepted-prefix gaps are not comparable to full-trajectory regret; billing requires provider reconciliation"
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    a=sub.add_parser("prepare");a.add_argument("--out",required=True)
    a.add_argument("--seed",type=int,default=SEED);a.add_argument("--roots-per-family",type=int,default=2)
    a.add_argument("--fixture",action="store_true")
    a=sub.add_parser("report");a.add_argument("--campaign",required=True)
    a=p.parse_args()
    result=prepare_calibration(a.out,seed=a.seed,roots_per_family=a.roots_per_family,fixture=a.fixture) if a.command=="prepare" else report(a.campaign)
    print(json.dumps(result,indent=2,sort_keys=True))

if __name__=="__main__":
    main()

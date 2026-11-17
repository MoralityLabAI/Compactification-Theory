"""SMC-25 independent standard-library audit of all-digit input-only predictions.

Recomputes source puzzle identities, native result arithmetic, prior-median
reductions and all preregistered puzzle-cluster bootstrap decisions.
No pretrained model, SAE, PyTorch, NumPy or target-label gradient is loaded.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import struct
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from smc13_family import (new_full_solution,dig_unique,rectangle_count,
                          count_solutions,fingerprint)

CAL_SCHEMA="SMC-25/teacher-temperature-calibration-before-test-20261117-v1"
PRED_SCHEMA="SMC-25/label-blind-all-class-native-causal-prediction-20261117-v1"
SCORE_SCHEMA="SMC-25/finite-native-all-candidate-scored-20261117-v1"
FAMILIES={"smc25_novel30":(202635030000,30),
          "smc25_novel34":(202635040000,34)}
B_COMPONENTS=("newB_501_SAE_71","newB_501_SAE_89",
              "newB_701_SAE_71","newB_701_SAE_89")
TEACHERS=("B501","B701")
TEMPS=(.5,1.,2.,4.,8.)
CONTROLS=("uniform_median","top_digit_effect","shifted_median")
PUZZLES=64
GOALS=8
BOOT=2000
LOCKS={
 "original_sae_receipt":"7caafcb54e36f508b49a082892b689789a751f86b52d6d206e955e3782431087",
 "independent_teacher_receipt":"1fc69e15c24087f631806a89d55beaf519c2f6ab58ac7b8193cf8fdc36d240e3",
 "conditioned_mapping_receipt":"3b4637699d5af236af2d3b8aa5d04e024025b16e0d59395535892c104d472d07",
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def seed(*parts):
    x="SMC24|"+"|".join(map(str,parts))
    return int.from_bytes(hashlib.sha256(x.encode()).digest()[:8],"big") & ((1<<63)-1)


def median(values,weights):
    pairs=sorted((float(v),i,float(w)) for i,(v,w) in enumerate(zip(values,weights)))
    if len(pairs)!=9 or abs(sum(w for _,_,w in pairs)-1)>1e-10 or any(w<0 for _,_,w in pairs):
        raise ValueError("bad source digit distribution or candidate effect count")
    total=0.
    for value,_,weight in pairs:
        total+=weight
        if total>=.5-1e-15:
            return value
    raise ArithmeticError("weighted median unavailable")


def softmax(logits,temperature):
    x=[float(z)/temperature for z in logits]
    m=max(x)
    e=[math.exp(t-m) for t in x]
    return [v/sum(e) for v in e]


def candidate_check(g,teacher_temperature):
    if len(g["classwise_effects"])!=9 or len(g["native_clean_logits"])!=10 or len(g["rival_for_digit"])!=9:
        raise ValueError("missing full native classwise output")
    clean=[float(v) for v in g["native_clean_logits"]]
    v=[float(x) for x in g["classwise_effects"]]
    if any(not math.isfinite(x) for x in clean+v):
        raise ValueError("nonfinite native source")
    rivals=[]
    for y in range(1,10):
        rival=max((z for z in range(10) if z!=y),key=lambda z:clean[z])
        rivals.append(rival)
    if rivals!=g["rival_for_digit"]:
        raise ValueError("candidate clean rival differs from archived native logits")
    q=softmax(clean[1:10],teacher_temperature)
    if abs(float(g["calibration_temperature"])-teacher_temperature)>1e-12:
        raise ValueError("prediction used new or unfrozen temperature")
    if any(abs(float(a)-float(b))>1e-11 for a,b in zip(q,g["calibrated_digit_distribution"])):
        raise ValueError("calibrated posterior changed after input-only prediction")
    top=max(range(1,10),key=lambda y:clean[y])
    if top!=g["model_top_digit"]:
        raise ValueError("model top digit changed")
    shifted=[q[(i-1)%9] for i in range(9)]
    pp={
        "calibrated_median":median(v,q),
        "uniform_median":median(v,[1./9]*9),
        "top_digit_effect":v[top-1],
        "shifted_median":median(v,shifted),
        "calibrated_mean":sum(a*b for a,b in zip(v,q)),
        "uniform_mean":sum(v)/9.,
        "zero_effect":0.,
    }
    if set(g["predictions"])!=set(pp) or any(abs(float(g["predictions"][k])-p)>1e-10
                                         for k,p in pp.items()):
        raise ValueError("frozen candidate effect aggregation does not replay")
    return pp


def verify(cal_raw,pred_raw,score_raw):
    cal=json.loads(cal_raw);pred=json.loads(pred_raw);score=json.loads(score_raw)
    protocol=sha((HERE/"SMC25_PROTOCOL.md").read_bytes())
    if (cal["schema"]!=CAL_SCHEMA or pred["schema"]!=PRED_SCHEMA or score["schema"]!=SCORE_SCHEMA
        or any(x["protocol_sha256"]!=protocol for x in (cal,pred,score))):
        raise ValueError("source registered schema/commit drift")
    if pred["calibration_sha256"]!=sha(cal_raw) or score["calibration_sha256"]!=sha(cal_raw):
        raise ValueError("frozen earlier calibration byte SHA mismatch")
    if score["prediction_sha256"]!=sha(pred_raw):
        raise ValueError("scored native input predictions were replaced")
    if not (cal["old_calibration_targets_only"] and cal["fresh_source_generated"] is False):
        raise ValueError("calibration source leaked future test data")
    if pred["contains_test_labels"] is not False or pred["uses_test_true_digit_gradient"] is not False:
        raise ValueError("target leaked into prediction computation")
    original_text=pred_raw.decode()
    for forbidden in ('"truth_digit"','"target":','"observed"','"true_D"','"goal_truth_digits"'):
        if forbidden in original_text:
            raise ValueError("label or task margin appeared in input-only artifact")
    if not(cal["source_snapshot"]==pred["source_snapshot"]==score["source_snapshot"]):
        raise ValueError("source checkpoint/SAE mapping changed across frozen stages")
    for k,v in LOCKS.items():
        if cal["source_snapshot"][k]!=v:
            raise ValueError("source model SHA changed")
    if set(cal["teacher_temperatures"])!=set(TEACHERS):
        raise ValueError("missing one independent calibrated teacher")
    for name,rec in cal["teacher_temperatures"].items():
        if rec["temperature"] not in TEMPS or rec["calibration_goals"]!=512:
            raise ValueError("wrong calibrated temperature or sample budget")
        values=rec["temperature_candidate_nll"]
        if set(values)!={str(t) for t in TEMPS}:
            raise ValueError("temperature calibration alternatives changed")
        best=min(TEMPS,key=lambda t:(values[str(t)],t))
        if float(best)!=float(rec["temperature"]):
            raise ValueError("calibration temperature chosen from future test")
    if pred["test_manifest"]!=score["test_manifest"] or pred["test_manifest"]["seed_bases"]!={k:list(v) for k,v in FAMILIES.items()}:
        raise ValueError("registered test puzzle seeds or roster hashes changed")
    if set(pred["test_prediction_data"])!=set(FAMILIES) or set(score["test_data"])!=set(FAMILIES):
        raise ValueError("source test split list changed")
    for family in FAMILIES:
        if set(pred["archived_source_stage1"][family])!=set(TEACHERS):
            raise ValueError("canonical source FP32 state incomplete")
        for teacher in TEACHERS:
            state=pred["archived_source_stage1"][family][teacher]
            if state["shape"]!=[64,81,48]:
                raise ValueError("wrong source stage shape")
            floats=[]
            for puzzle in state["tensor"]:
                if len(puzzle)!=81: raise ValueError("incomplete source token state")
                for token in puzzle:
                    if len(token)!=48: raise ValueError("wrong state hidden width")
                    floats.extend(token)
            if sha(struct.pack("<"+str(len(floats))+"f",*floats))!=state["tensor_sha256"]:
                raise ValueError("archived FP32 source byte identity mismatch")
            drift=score["replay_max_source_stage1_abs"][family][teacher]
            if not 0<=drift<=1e-4:
                raise ValueError("canonical crossrunner replay discrepancy exceeds admission")
    return cal,pred,score,protocol


def read_source_and_effects(cal,pred,scored):
    total=0
    seen_x=set();seen_y=set()
    for family,(base,holes) in FAMILIES.items():
        cases=[]
        for i in range(PUZZLES):
            sd=base+i
            y=new_full_solution(sd)
            x=dig_unique(y,sd+1000000,holes)
            if tuple(x) in seen_x or tuple(y) in seen_y or count_solutions(x,2)!=1 or rectangle_count(y)<=0:
                raise ValueError("fresh source duplicate, nonunique or canonical")
            seen_x.add(tuple(x));seen_y.add(tuple(y))
            cases.append({"seed":sd,"x":x,"target":y,"holes":holes,"rectangles":rectangle_count(y)})
        if fingerprint(cases)!=pred["test_manifest"]["cohorts"][family]["roster_sha256"]:
            raise ValueError("fresh source target/roster SHA differs")
        for component in B_COMPONENTS:
            tid="B501" if "_501_" in component else "B701"
            temperature=float(cal["teacher_temperatures"][tid]["temperature"])
            predicted=pred["test_prediction_data"][family][component]
            scored_rows=scored["test_data"][family][component]["per_puzzle"]
            if len(predicted)!=PUZZLES or len(scored_rows)!=PUZZLES:
                raise ValueError("source puzzle count incomplete")
            for idx,(x,y) in enumerate(zip(predicted,scored_rows)):
                if x["visible_input"]!=cases[idx]["x"] or x["seed"]!=cases[idx]["seed"] or y["seed"]!=cases[idx]["seed"]:
                    raise ValueError("source prediction is another puzzle")
                blanks=[j for j,a in enumerate(cases[idx]["x"]) if a==0]
                positions=[blanks[len(blanks)*j//GOALS] for j in range(GOALS)]
                if x["positions"]!=positions or y["goal_positions"]!=positions:
                    raise ValueError("goal locations are not input-only blanks")
                if len(x["goals"])!=GOALS or len(y["goals"])!=GOALS:
                    raise ValueError("unreported frozen goal")
                for j,(g,r) in enumerate(zip(x["goals"],y["goals"])):
                    if (g["status"]!=r["status"] or g["feature"]!=r["feature"] or
                        g["input_plan"]!=r["input_plan"] or g["token"]!=r["token"]):
                        raise ValueError("scoring changed an archived input feature or sham")
                    if g["status"]!="eligible":
                        if "observed" in r:raise ValueError("ineligible goal measured")
                        continue
                    total+=1
                    p=candidate_check(g,temperature)
                    if g["predictions"]!=r["predictions"]:
                        raise ValueError("native scoring altered frozen estimates")
                    observed=r["observed"]
                    truth=cases[idx]["target"][positions[j]]
                    if observed["truth_digit"]!=truth or observed["frozen_clean_rival"]!=g["rival_for_digit"][truth-1]:
                        raise ValueError("true digit or its source-clean rival changed")
                    margins=observed["margins"]
                    if margins["clean"]!=margins["zero_self"]:
                        raise ValueError("self patch not identical")
                    actual=margins["sham"]-margins["kill"]
                    if abs(actual-observed["true_D"])>1e-10:
                        raise ValueError("finite erasure task effect identity failed")
                    if abs(r["oracle_class_tangent"]-g["classwise_effects"][truth-1])>1e-10:
                        raise ValueError("posthoc true digit local tangent selected before label release")
                    if abs(r["oracle_finite_taylor_residual"]-(actual-g["classwise_effects"][truth-1]))>1e-10:
                        raise ValueError("finite Taylor residual not replayable")
                    d=[float(v) for v in g["edit_kill"]]
                    s=[float(v) for v in g["edit_sham"]]
                    if len(d)!=48 or len(s)!=48:
                        raise ValueError("wrong physical SAE edit width")
                    dose=math.sqrt(sum(x*x for x in d))
                    sham=math.sqrt(sum(x*x for x in s))
                    if abs(dose-sham)>1e-8 or abs(dose-g["input_plan"]["dose"])>1e-8:
                        raise ValueError("original source sham dose mismatch")
    return total


def independent_summary(rows,family,component):
    if len(rows)!=PUZZLES:
        raise ValueError("wrong puzzle-cluster count")
    n=0;ids=0
    stats=Counter()
    values=[]
    sign_a=sign_u=0
    for puzzle in rows:
        chosen=[]
        for g in puzzle["goals"]:
            stats[g["status"]]+=1
            if g["status"]!="eligible":
                continue
            true=float(g["observed"]["true_D"])
            pred=g["predictions"]
            chosen.append((true,pred))
            sign_a+=int(bool(true and pred["calibrated_median"] and ((true>0)==(pred["calibrated_median"]>0))))
            sign_u+=int(bool(true and pred["uniform_median"] and ((true>0)==(pred["uniform_median"]>0))))
        n+=len(chosen)
        ids+=int(bool(chosen))
        values.append(None if not chosen else {
            "gains":{c:statistics.mean(abs(d-p[c])-abs(d-p["calibrated_median"])
                                       for d,p in chosen) for c in CONTROLS},
            "errors":{c:statistics.mean(abs(d-p[c]) for d,p in chosen)
                      for c in ("calibrated_median",)+CONTROLS}})
    if n==0:
        return {"eligible_goals":0,"eligible_puzzles":0,"registered_pass":False,"status":"no_eligible_goals"}
    rng=random.Random(seed("bootstrap",20262517,family,component))
    boot={c:[] for c in CONTROLS}
    valid=0
    for _ in range(BOOT):
        index=[rng.randrange(PUZZLES) for i in range(PUZZLES)]
        picked=[values[i] for i in index if values[i] is not None]
        if not picked:continue
        valid+=1
        for c in CONTROLS:
            boot[c].append(sum(r["gains"][c] for r in picked)/len(picked))
    for c in CONTROLS:boot[c].sort()
    enough=n>=60 and ids>=30 and valid>=1990
    ci95={c:([boot[c][int(.025*valid)],boot[c][int(.975*valid)-1]]
             if valid>=1990 else None) for c in CONTROLS}
    ci99={c:([boot[c][int(.003125*valid)],boot[c][int(.996875*valid)-1]]
             if valid>=1990 else None) for c in CONTROLS}
    success=bool(enough and sign_a>sign_u and all(ci99[c][0]>0 for c in CONTROLS))
    return {
        "status":"measured" if enough else "underpowered_coverage",
        "eligible_goals":n,"eligible_puzzles":ids,
        "empty_puzzles":PUZZLES-ids,
        "preoutcome_status_counts":dict(stats),
        "mean_puzzle_MAE":{c:statistics.mean(v["errors"][c] for v in values if v is not None)
                           for c in ("calibrated_median",)+CONTROLS},
        "mean_puzzle_gain":{c:statistics.mean(v["gains"][c] for v in values if v is not None)
                            for c in CONTROLS},
        "bootstrap_valid":valid,
        "puzzle_bootstrap95_gain":ci95,
        "puzzle_bootstrap99_375_gain":ci99,
        "calibrated_median_sign_correct":sign_a,
        "uniform_median_sign_correct":sign_u,
        "registered_pass":success,
    }


def _close(a,b):
    if isinstance(a,dict) and isinstance(b,dict):
        if set(a)!=set(b):raise ValueError("independent source summary keys differ")
        for k in a:_close(a[k],b[k])
    elif isinstance(a,list) and isinstance(b,list):
        if len(a)!=len(b):raise ValueError("independent bootstrap vector length changed")
        for x,y in zip(a,b):_close(x,y)
    elif isinstance(a,(int,float)) and isinstance(b,(int,float)):
        if abs(a-b)>1e-9:raise ValueError("registered numerical result changed")
    elif a!=b:
        raise ValueError("independent statistic field changed")


def replay(cal_raw,pred_raw,scored_raw):
    cal,pred,scored,protocol=verify(cal_raw,pred_raw,scored_raw)
    total=read_source_and_effects(cal,pred,scored)
    cells=[]
    for family in FAMILIES:
        for component in B_COMPONENTS:
            r=scored["test_data"][family][component]
            stats=independent_summary(r["per_puzzle"],family,component)
            _close(stats,r["summary"])
            cells.append({"family":family,"component":component,
                          "eligible_goals":stats["eligible_goals"],
                          "calibrated_MAE":stats.get("mean_puzzle_MAE",{}).get("calibrated_median"),
                          "mean_gain":stats.get("mean_puzzle_gain"),
                          "adjusted_ci":stats.get("puzzle_bootstrap99_375_gain"),
                          "registered_pass":stats["registered_pass"]})
    pairs={name:all(next(c["registered_pass"] for c in cells
                         if c["component"]==name and c["family"]==family)
                    for family in FAMILIES) for name in B_COMPONENTS}
    success=any(pairs[n] for n in B_COMPONENTS[:2]) and any(pairs[n] for n in B_COMPONENTS[2:])
    if pairs!=scored["pair_success"] or success!=scored["program_success"]:
        raise ValueError("original registered scientific pair-level decision changed")
    return {
        "schema":"SMC25/source-independent-standard-library-receipt-audit-v1",
        "audit_status":"PASS","scientific_program_pass":success,
        "registered_protocol_sha256":protocol,
        "calibration_sha256":sha(cal_raw),
        "frozen_prediction_sha256":sha(pred_raw),
        "native_scored_sha256":sha(scored_raw),
        "n_unique_new_puzzles":128,
        "n_all_source_goal_slots":4096,
        "n_preoutcome_eligible_goals":total,
        "teacher_sae_pair_decisions":pairs,
        "results":cells,
        "limitations":"Checks source identities, preoutcome prediction arithmetic and stored model outcome receipts; does not independently rerun checkpoint native forward."
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--calibration",type=Path,required=True)
    p.add_argument("--prediction",type=Path,required=True)
    p.add_argument("--scored",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    report=replay(a.calibration.read_bytes(),a.prediction.read_bytes(),a.scored.read_bytes())
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    print("SMC25_INDEPENDENT_AUDIT_PASS",report["scientific_program_pass"],
          report["n_preoutcome_eligible_goals"],str(a.out),flush=True)


if __name__=="__main__":
    main()

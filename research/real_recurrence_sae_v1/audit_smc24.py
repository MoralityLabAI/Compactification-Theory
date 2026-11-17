"""SMC-24 independent Python stdlib replay of frozen prediction and native score.

Reads three immutable JSON receipts in chronological order; no torch,
checkpoint load, native gradient, or learned predictor refitting. Independently
recomputes new test full grids to verify hidden target-digit and source hashes.
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
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from smc13_family import new_full_solution, dig_unique, count_solutions, rectangle_count, fingerprint

COMPONENTS=("newB_501_SAE_71","newB_501_SAE_89",
            "newB_701_SAE_71","newB_701_SAE_89")
FAMILIES={"smc24_novel30":(202634030000,30),
          "smc24_novel34":(202634040000,34)}
BASELINES=("mean","shuffled","zero")
GOALS=8
PUZZLES=64
BOOT=2000
EXPECTED_SOURCES={
    "original_sae_receipt":"7caafcb54e36f508b49a082892b689789a751f86b52d6d206e955e3782431087",
    "independent_teacher_receipt":"1fc69e15c24087f631806a89d55beaf519c2f6ab58ac7b8193cf8fdc36d240e3",
    "conditioned_mapping_receipt":"3b4637699d5af236af2d3b8aa5d04e024025b16e0d59395535892c104d472d07",
}


def sha(raw):return hashlib.sha256(raw).hexdigest()


def source_seed(*items):
    t="SMC24|"+"|".join(map(str,items))
    return int.from_bytes(hashlib.sha256(t.encode()).digest()[:8],"big") & ((1<<63)-1)


def protocol_sha():
    return sha((HERE/"SMC24_PROTOCOL.md").read_bytes())


def verify_frozen(cal,raw_cal,pred,raw_pred,scored):
    h=protocol_sha()
    if any(x["protocol_sha256"]!=h for x in (cal,pred,scored)):
        raise ValueError("frozen SMC24 registration changed")
    if cal["schema"]!="SMC-24/labelled-calibration-gradient-ridge-20261117-v1":
        raise ValueError("wrong calibration schema")
    if pred["schema"]!="SMC-24/label-free-archived-native-predictions-20261117-v1":
        raise ValueError("wrong prediction schema")
    if scored["schema"]!="SMC-24/frozen-prediction-native-scored-20261117-v1":
        raise ValueError("wrong native score schema")
    if pred["calibration_artifact_sha256"]!=sha(raw_cal):
        raise ValueError("pretest calibration receipt was replaced")
    if scored["prediction_artifact_sha256"]!=sha(raw_pred):
        raise ValueError("native outcomes were scored from a different prediction")
    if not cal["new_test_rosters_generated"] is False:
        raise ValueError("calibration used future test cases")
    if pred["test_targets_present"] is not False or pred["test_gradient_from_test_labels"] is not False:
        raise ValueError("test labels entered frozen prediction")
    if cal["source_snapshot"]!=pred["source_snapshot"] or pred["source_snapshot"]!=scored["source_snapshot"]:
        raise ValueError("source checkpoint or correspondence drift")
    for k,v in EXPECTED_SOURCES.items():
        if cal["source_snapshot"][k]!=v:
            raise ValueError("source frozen original artifact differs: "+k)
    if set(cal["fits"])!={"B501","B701"}:
        raise ValueError("missing independent trained teacher")
    for fit in cal["fits"].values():
        if fit["n_calibration_goals"]!=512 or fit["lambda"]!=32.:
            raise ValueError("calibration sample count or hyperparameters changed")
        if len(fit["ridge_beta"])!=49 or len(fit["slot_shuffled_ridge_beta"])!=49:
            raise ValueError("wrong frozen ridge map rank")
        if any(len(row)!=48 for row in fit["ridge_beta"]+fit["slot_shuffled_ridge_beta"]):
            raise ValueError("wrong frozen target-gradient dimension")
        if len(fit["std"])!=48 or min(fit["std"])<=0:
            raise ValueError("invalid source-scale fit")
    if pred["test_manifest"]!=scored["test_manifest"]:
        raise ValueError("posthoc test manifest was changed")
    if pred["test_manifest"]["seed_bases"]!={k:list(v) for k,v in FAMILIES.items()}:
        raise ValueError("prospectively registered fresh seeds differ")
    if set(pred["test_prediction_data"])!=set(FAMILIES) or set(scored["test_data"])!=set(FAMILIES):
        raise ValueError("fresh cohort list differs")
    raw_text=raw_pred.decode()
    for term in ('"truth_digit"','"goal_truth_digits"','"target":','"true_D"','"margins"'):
        if term in raw_text:
            raise ValueError("test scoring or correct digit leaked into predictor artifact")
    return h


def roster_replay(pred,scored):
    seen_x=set();seen_y=set()
    source=pred["test_manifest"]
    if source["n_per_family"]!=64 or source["n_previous_excluded"]<768:
        raise ValueError("prior cohort exclusion missing from manifest")
    for family,(base,holes) in FAMILIES.items():
        grid=[]
        for i in range(PUZZLES):
            seed=base+i
            y=new_full_solution(seed)
            x=dig_unique(y,seed+1000000,holes)
            if count_solutions(x,2)!=1 or x.count(0)!=holes or rectangle_count(y)<=0:
                raise ValueError("new source puzzle invalid or canonical")
            if tuple(x) in seen_x or tuple(y) in seen_y:
                raise ValueError("new fresh cohort puzzle repeated")
            seen_x.add(tuple(x));seen_y.add(tuple(y))
            grid.append({"seed":seed,"x":x,"target":y,
                         "holes":holes,"rectangles":rectangle_count(y)})
        if source["cohorts"][family]["roster_sha256"]!=fingerprint(grid):
            raise ValueError("fresh complete hidden Sudoku target checksum differs")
        for comp in COMPONENTS:
            pred_rows=pred["test_prediction_data"][family][comp]
            scored_rows=scored["test_data"][family][comp]["per_puzzle"]
            if len(pred_rows)!=PUZZLES or len(scored_rows)!=PUZZLES:
                raise ValueError("puzzle roster incomplete")
            for idx,(p,q) in enumerate(zip(pred_rows,scored_rows)):
                if (p["puzzle_index"]!=idx or p["seed"]!=grid[idx]["seed"] or
                    p["visible_input"]!=grid[idx]["x"] or
                    q["puzzle_index"]!=idx or q["seed"]!=grid[idx]["seed"]):
                    raise ValueError("frozen input-only prediction does not match actual puzzle")
                blanks=[j for j,v in enumerate(grid[idx]["x"]) if v==0]
                positions=[blanks[j*len(blanks)//8] for j in range(8)]
                if p["positions"]!=positions or q["goal_positions"]!=positions:
                    raise ValueError("goal selection relied on more than visible blanks")
                if len(p["goals"])!=GOALS or len(q["goals"])!=GOALS:
                    raise ValueError("8 preselected goals absent")
                for j,(g,h) in enumerate(zip(p["goals"],q["goals"])):
                    if g["status"]!=h["status"] or g["feature"]!=h["feature"]:
                        raise ValueError("goal source feature or eligibility was replaced")
                    if g["status"]!="eligible":
                        if "observed" in h:raise ValueError("ineligible outcome counted")
                        continue
                    if not 0<=g["feature"]<96:
                        raise ValueError("SAE feature index out of range")
                    if h["observed"]["truth_digit"]!=grid[idx]["target"][positions[j]]:
                        raise ValueError("incorrect test true digit")
                    if h["observed"]["selected_feature"]!=g["feature"]:
                        raise ValueError("scored another source feature")
                    if g["input_plan"]!=h["input_plan"]:
                        raise ValueError("input-only edit plan was modified after scoring")
                    if g["predictions"]!=h["predictions"]:
                        raise ValueError("native score used different prediction vector")
                    if set(g["predictions"])!={"ridge","mean","shuffled","zero"}:
                        raise ValueError("unregistered predictor arm")
                    a,b=g["edit_kill"],g["edit_sham"]
                    if len(a)!=48 or len(b)!=48:
                        raise ValueError("wrong physical 48D edit")
                    ea=math.sqrt(sum(v*v for v in a))
                    eb=math.sqrt(sum(v*v for v in b))
                    if abs(ea-eb)>1e-8 or abs(ea-g["input_plan"]["dose"])>1e-8:
                        raise ValueError("physical-sham norm match failed")
                    if h["observed"]["margins"]["clean"]!=h["observed"]["margins"]["zero_self"]:
                        raise ValueError("self-patch baseline changed")
                    obs=h["observed"]["margins"]["sham"]-h["observed"]["margins"]["kill"]
                    if abs(obs-h["observed"]["true_D"])>1e-10:
                        raise ValueError("native margin effect identity differs")
                    for bname in BASELINES:
                        gain=abs(obs-g["predictions"][bname])-abs(obs-g["predictions"]["ridge"])
                        if abs(gain-h["registered_MAE_gains"][bname])>1e-10:
                            raise ValueError("registered MAE gain inconsistent")
    return {"new_puzzles":128,"preselected_goal_slots":128*GOALS*len(COMPONENTS),
            "checked_prediction_only_input_roster":True}


def sign_correct(a,b):
    return bool(a and b and ((a>0)==(b>0)))


def independent_summary(rows,component,family):
    if len(rows)!=64:
        raise ValueError("expected 64 puzzle clusters")
    puzzle=[]
    flipk=flips=sign_r=sign_m=0
    status=Counter()
    for p in rows:
        kept=[]
        for g in p["goals"]:
            status[g["status"]]+=1
            if g["status"]!="eligible":continue
            obs=g["observed"]
            d=obs["true_D"];pr=g["predictions"]
            kept.append((d,pr))
            flipk+=int(obs["kill_correct_to_wrong"])
            flips+=int(obs["sham_correct_to_wrong"])
            sign_r+=int(sign_correct(d,pr["ridge"]))
            sign_m+=int(sign_correct(d,pr["mean"]))
        if kept:
            puzzle.append({"count":len(kept),
                "gains":{b:statistics.mean(abs(d-pr[b])-abs(d-pr["ridge"])
                                    for d,pr in kept) for b in BASELINES},
                "mae":{b:statistics.mean(abs(d-pr[b]) for d,pr in kept)
                       for b in ("ridge",)+BASELINES}})
        else:puzzle.append(None)
    total=sum(p["count"] for p in puzzle if p)
    n_puzzle=sum(p is not None for p in puzzle)
    if not total:
        return {"eligible_goals":0,"registered_pass":False,"status":"no_eligible_goals"}
    root=random.Random(source_seed("bootstrap",20262417,family,component))
    samples={b:[] for b in BASELINES}
    valid=0
    for _ in range(BOOT):
        selected=[puzzle[root.randrange(PUZZLES)] for _ in range(PUZZLES)]
        nonempty=[p for p in selected if p]
        if not nonempty:continue
        valid+=1
        for b in BASELINES:
            samples[b].append(statistics.mean(p["gains"][b] for p in nonempty))
    for b in BASELINES:samples[b].sort()
    high=(total>=60 and n_puzzle>=30 and valid>=1990)
    ci95={b:[samples[b][int(.025*valid)],samples[b][int(.975*valid)-1]]
          if valid>=1990 else None for b in BASELINES}
    ci99={b:[samples[b][int(.003125*valid)],samples[b][int(.996875*valid)-1]]
          if valid>=1990 else None for b in BASELINES}
    good=bool(high and sign_r>sign_m and all(ci99[b][0]>0 for b in BASELINES))
    return {
        "status":"measured" if high else "underpowered_coverage",
        "eligible_goals":total,"eligible_puzzles":n_puzzle,
        "n_without_eligible":PUZZLES-n_puzzle,
        "observed_goal_status_counts":dict(status),
        "puzzle_mean_MAE":{b:statistics.mean(p["mae"][b] for p in puzzle if p)
                           for b in ("ridge",)+BASELINES},
        "puzzle_mean_ridge_gain":{b:statistics.mean(p["gains"][b] for p in puzzle if p)
                                   for b in BASELINES},
        "cluster_bootstrap_valid":valid,
        "puzzle_bootstrap95_gain":ci95,"puzzle_bootstrap99_375_gain":ci99,
        "ridge_sign_correct":sign_r,"mean_sign_correct":sign_m,
        "n_nonzero_truth_goal_signs":total,
        "kill_clean_correct_to_wrong":flipk,"sham_clean_correct_to_wrong":flips,
        "registered_pass":good,
    }


def replay(cal_raw,pred_raw,score_raw):
    cal=json.loads(cal_raw);pred=json.loads(pred_raw);scored=json.loads(score_raw)
    h=verify_frozen(cal,cal_raw,pred,pred_raw,scored)
    census=roster_replay(pred,scored)
    cells=[]
    for family in FAMILIES:
        for comp in COMPONENTS:
            published=scored["test_data"][family][comp]["summary"]
            replayed=independent_summary(scored["test_data"][family][comp]["per_puzzle"],comp,family)
            for key in ("status","eligible_goals","eligible_puzzles","n_without_eligible",
                        "observed_goal_status_counts","cluster_bootstrap_valid",
                        "ridge_sign_correct","mean_sign_correct",
                        "kill_clean_correct_to_wrong","sham_clean_correct_to_wrong",
                        "registered_pass"):
                if replayed[key]!=published[key]:
                    raise ValueError("independent registered metric mismatch: "+key)
            for key in ("puzzle_mean_MAE","puzzle_mean_ridge_gain",
                        "puzzle_bootstrap95_gain","puzzle_bootstrap99_375_gain"):
                for b in (("ridge",)+BASELINES if key=="puzzle_mean_MAE" else BASELINES):
                    a=replayed[key][b];z=published[key][b]
                    if a is None or z is None:
                        if a is not z:raise ValueError("wrong missing bootstrap data")
                    elif isinstance(a,list):
                        if any(abs(x-y)>1e-10 for x,y in zip(a,z)):
                            raise ValueError("bootstrap quantile changed")
                    elif abs(a-z)>1e-10:
                        raise ValueError("independent MAE or gain differs")
            cells.append({"family":family,"component":comp,
                          "eligible_goals":replayed["eligible_goals"],
                          "MAE":replayed["puzzle_mean_MAE"],
                          "gain":replayed["puzzle_mean_ridge_gain"],
                          "corrected_ci":replayed["puzzle_bootstrap99_375_gain"],
                          "registered_pass":replayed["registered_pass"]})
    pair={name:all(next(c["registered_pass"] for c in cells if c["component"]==name and c["family"]==fam)
                  for fam in FAMILIES) for name in COMPONENTS}
    both=any(pair[n] for n in COMPONENTS[:2]) and any(pair[n] for n in COMPONENTS[2:])
    if pair!=scored["pair_success"] or both!=scored["program_success"]:
        raise ValueError("combined source teacher-level decision changed")
    return {
        "schema":"SMC24/independent-stdlib-receipt-audit-v1",
        "status":"PASS", "scientific_program_pass":both,
        "source_protocol_sha256":h, "calibration_sha256":sha(cal_raw),
        "frozen_prediction_sha256":sha(pred_raw),"native_score_sha256":sha(score_raw),
        "puzzles":census,"results":cells,"pair_success":pair,
        "limitations":"Replays archived native outcomes and deterministic new generator, not an independent teacher forward or gradient training."
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
    a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("SMC24_INDEPENDENT_REPLAY_PASS",json.dumps({
        "program":report["scientific_program_pass"],
        "cells":len(report["results"]),
        "puzzles":report["puzzles"]["new_puzzles"]},sort_keys=True),flush=True)


if __name__=="__main__":
    main()

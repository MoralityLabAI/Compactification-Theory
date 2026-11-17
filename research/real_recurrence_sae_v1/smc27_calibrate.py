"""SMC-27 Stage A: train abstention threshold on OLD SMC-13 calibration only.

Four teachers and SAE71s are loaded from the ORIGINAL immutable SMC-26
training bundle (never retrained). No SMC-26 novel or SMC-27 fresh data is
available in the calibration source path.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys

import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from smc13_train_teachers import tensors
from smc14_causal_transfer import goal_positions
from smc23_native_first import plan_target_native,affine_physical
from smc24_native_score import native_score_case
from smc25_candidate_effects import label_blind_tangent,candidate_predictions,clean_native
from smc26_train import load_all_models,TEACHERS,digest
from smc26_score import decision_utility,target_class
from smc13_family import fingerprint

SCHEMA="SMC27/calibration-only-selected-abstention-v1"
OLD_BUNDLE_SHA="408348eeecc300e7518ee7e8ea14d71caac29821aff39aaaa5ee819c794231f1"
OLD_CAL_SHA="145312c04648847c409b506aecdf1a3464af855d4ec8aa638eb0d604185512c9"
GRID=(.15,.30,.50,.75,1.,1.50,2.,3.,5.)
GOALS=8
CAL_PUZZLES=64
ACTION_RATE_MIN=.05
ACTION_RATE_MAX=.90


def protocol_sha():
    return digest((HERE/"SMC27_PROTOCOL.md").read_bytes())


def policy(x,threshold):
    x=float(x)
    if not math.isfinite(x) or threshold<0:
        raise ValueError("nonfinite effect or invalid policy threshold")
    return 1 if x>threshold else -1 if x< -threshold else 0


def evaluate_old_threshold(rows,threshold):
    """Mean offline utility over all 512 source calibration slots."""
    if len(rows)!=CAL_PUZZLES or any(len(p["goals"])!=GOALS for p in rows):
        raise ValueError("calibration must contain original 64 eight-goal puzzles")
    eligible=active=0
    total=0.
    for puzzle in rows:
        for g in puzzle["goals"]:
            if g["status"]!="eligible":
                continue
            eligible+=1
            action=policy(g["calibrated_median"],threshold)
            active+=int(action!=0)
            total+=decision_utility(action,int(g["true_class"]))
    return {
        "threshold":float(threshold),
        "eligible_goals":eligible,
        "n_old_calibration_slots":CAL_PUZZLES*GOALS,
        "calibrated_action_rate":active/eligible if eligible else 0.,
        "mean_utility_all_original_slots":total/(CAL_PUZZLES*GOALS),
        "in_range":bool(eligible and ACTION_RATE_MIN<=active/eligible<=ACTION_RATE_MAX),
    }


def select_old_threshold(rows):
    scores=[evaluate_old_threshold(rows,threshold) for threshold in GRID]
    viable=[c for c in scores if c["in_range"]]
    if viable:
        choice=max(viable,key=lambda d:(d["mean_utility_all_original_slots"],d["threshold"]))
        status="CALIBRATED"
    else:
        choice=next(s for s in scores if s["threshold"]==5.)
        status="CALIBRATION_INADMISSIBLE"
    return {
        "source":"SMC13_original_feature_calibration_only",
        "threshold_grid":list(GRID),
        "threshold_grid_scores":scores,
        "selected_threshold":choice["threshold"],
        "selected_threshold_old_mean_utility":choice["mean_utility_all_original_slots"],
        "selected_threshold_old_action_rate":choice["calibrated_action_rate"],
        "admissible":status=="CALIBRATED",
        "status":status,
    }


def calibrate(original,independent,conditioned,bundle,old_calibration,allow_fetch):
    torch.set_num_threads(2)
    if digest(Path(bundle).read_bytes())!=OLD_BUNDLE_SHA:
        raise ValueError("source ORIGINAL SMC26 trained teacher bytes differ")
    old_bytes=Path(old_calibration).read_bytes()
    if digest(old_bytes)!=OLD_CAL_SHA:
        raise ValueError("source ORIGINAL SMC26 calibration bytes differ")
    models,comps,old,cal=load_all_models(
        original,independent,conditioned,bundle,old_calibration,allow_fetch)
    if not cal["all_four_capability_matched"] or tuple(cal["teacher_names"])!=TEACHERS:
        raise ValueError("original SMC26 four-teacher competence/source receipt changed")
    for teacher in TEACHERS:
        oldmeta=cal["all_teacher_temperatures"][teacher]
        if (oldmeta["temperature"]!=2. or
            not .40<=oldmeta["calibration_goal_accuracy"]<=.60):
            raise ValueError("old source teacher calibration is outside original frozen band")
    cases=old["feature_calibration"]
    if len(cases)!=CAL_PUZZLES:
        raise ValueError("old SMC13 calibration roster differs")
    x,y=tensors(cases)
    positions=goal_positions(x)
    data={}
    for teacher in TEACHERS:
        key=f"{teacher}_SAE71"
        tid,sae,mu,scale=comps[key]
        if tid!=teacher:
            raise ValueError("frozen original SAE belongs to another teacher")
        cov,dec=affine_physical(sae,mu,scale)
        with torch.no_grad():
            stage=models[teacher].reason_states(x)[1].detach()
        native=copy.deepcopy(models[teacher]).double().eval()
        clean=clean_native(native,stage)
        temp=cal["all_teacher_temperatures"][teacher]["temperature"]
        records=[]
        for pi,row in enumerate(cases):
            goals=[]
            for j,pos in enumerate(positions[pi].tolist()):
                info,kill,sham=plan_target_native(
                    stage[pi,pos],sae,mu,scale,cov,dec,
                    f"SMC27|old_calibration|{pi}|{key}|{j}")
                output={
                    "goal_index":j,"position":pos,
                    "feature":int(info["feature"]),"status":info["status"],
                }
                if info["status"]=="eligible":
                    clean_here,tangent=label_blind_tangent(
                        native,stage[pi],pos,kill,sham)
                    if float((clean_here-clean[pi,pos]).abs().max())>1e-8:
                        raise ArithmeticError("old source clean logit parity failed")
                    candidate=candidate_predictions(clean_here,tangent,temp)
                    true=int(y[pi,pos])
                    scored=native_score_case(native,stage[pi],clean[pi],
                                             true,pos,info["feature"],kill,sham)
                    d=float(scored["true_D"])
                    output.update({
                        "calibrated_median":float(candidate["predictions"]["calibrated_median"]),
                        "uniform_median":float(candidate["predictions"]["uniform_median"]),
                        "shifted_median":float(candidate["predictions"]["shifted_median"]),
                        "top_digit_effect":float(candidate["predictions"]["top_digit_effect"]),
                        "true_digit":true,
                        "native_finite_D":d,
                        "true_class":target_class(d),
                        "physical_dose":float(info["dose"]),
                    })
                goals.append(output)
            records.append({"puzzle_index":pi,"grid_seed":row["seed"],
                            "selected_positions":positions[pi].tolist(),"goals":goals})
        selected=select_old_threshold(records)
        data[teacher]={"threshold_selection":selected,"per_puzzle":records}
        print("SMC27_OLD_CALIBRATION",teacher,"eligible",
              selected["threshold_grid_scores"][0]["eligible_goals"],
              "threshold",selected["selected_threshold"],
              "mean_old_utility",selected["selected_threshold_old_mean_utility"],
              "rate",selected["selected_threshold_old_action_rate"],
              "status",selected["status"],flush=True)
    return {
        "schema":SCHEMA,"protocol_sha256":protocol_sha(),
        "old_smc26_training_bundle_sha256":OLD_BUNDLE_SHA,
        "old_smc26_calibration_sha256":OLD_CAL_SHA,
        "old_smc26_source_snapshot":cal["source_snapshot"],
        "original_teacher_names":list(TEACHERS),
        "old_feature_calibration_roster_sha256":fingerprint(cases),
        "has_generated_fresh_smc27_puzzles":False,
        "all_selected_thresholds_admissible":all(
            data[t]["threshold_selection"]["admissible"] for t in TEACHERS),
        "teacher_calibration":data,
        "original_old_temperature_by_teacher":{
            t:cal["all_teacher_temperatures"][t]["temperature"] for t in TEACHERS},
        "torch_version":torch.__version__,
    }


def main():
    p=argparse.ArgumentParser()
    for item in ("original-artifact","independent-artifact","conditioned-receipt",
                 "bundle","old-calibration","out"):
        p.add_argument("--"+item,type=Path,required=True)
    p.add_argument("--fetch-source",action="store_true")
    a=p.parse_args()
    report=calibrate(a.original_artifact,a.independent_artifact,
                     a.conditioned_receipt,a.bundle,a.old_calibration,a.fetch_source)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    print("SMC27_OLD_THRESHOLD_LOCK_COMPLETE",
          report["all_selected_thresholds_admissible"],str(a.out),flush=True)


if __name__=="__main__":
    main()

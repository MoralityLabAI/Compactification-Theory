"""SMC-27 Stage B: freeze input-only native causal effects and abstention policies.

Original four SMC26 checkpoints and one SAE71 per teacher remain immutable.
New complete Sudoku targets are used for source uniqueness only; target digits
never enter the model prediction or physical feature-selection API.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from smc13_family import (new_full_solution,dig_unique,valid_solution,
                          rectangle_count,count_solutions,fingerprint)
from smc14_causal_transfer import goal_positions
from smc23_native_first import plan_target_native,affine_physical
from smc25_candidate_effects import label_blind_tangent,candidate_predictions
from smc26_train import load_all_models,TEACHERS,digest
from smc26_audit import PRIOR_SOURCE_COHORTS
from smc26_predict import fresh_rosters as previous_smc26
from smc27_calibrate import (policy,protocol_sha,SCHEMA as CAL_SCHEMA,
                            OLD_BUNDLE_SHA,OLD_CAL_SHA)

FAMILIES={"smc27_novel30":(202637030000,30),
          "smc27_novel34":(202637040000,34)}
SIZE=128
GOALS=8
PRED_SCHEMA="SMC27/calibration-frozen-input-only-causal-sign-policies-v1"
POLICIES=("calibrated_selected","uniform_selected","calibrated_fixed015",
          "always_abstain","shifted_selected","top_digit_selected")
BASE_TEMPERATURE=2.0


def new_rosters(old):
    """Prove independence from original SMC13 and ALL registered fresh cohorts."""
    previous,_=previous_smc26(old)
    seen_x={tuple(r["x"]) for group in list(old.values())+list(previous.values())
            for r in group}
    seen_y={tuple(r["target"]) for group in list(old.values())+list(previous.values())
            for r in group}
    for base,holes,n in PRIOR_SOURCE_COHORTS:
        for i in range(n):
            y=new_full_solution(base+i)
            x=dig_unique(y,base+i+1000000,holes)
            seen_x.add(tuple(x));seen_y.add(tuple(y))
    old_count=len(seen_y)
    new={}
    for family,(base,holes) in FAMILIES.items():
        cases=[]
        for i in range(SIZE):
            sd=base+i
            solution=new_full_solution(sd)
            puzzle=dig_unique(solution,sd+1000000,holes)
            if (not valid_solution(solution) or rectangle_count(solution)<=0 or
                puzzle.count(0)!=holes or count_solutions(puzzle,2)!=1 or
                any(v and v!=solution[k] for k,v in enumerate(puzzle))):
                raise ValueError("new SMC27 noncanonical source uniqueness failure")
            if tuple(solution) in seen_y or tuple(puzzle) in seen_x:
                raise ValueError("SMC27 duplicated an earlier source solution or visible puzzle")
            seen_x.add(tuple(puzzle));seen_y.add(tuple(solution))
            cases.append({"seed":sd,"x":puzzle,"target":solution,
                          "holes":holes,"rectangles":rectangle_count(solution)})
        new[family]=cases
    manifest={
        "schema":"SMC27/256-fresh-disjoint-noncanonical-Sudoku-v1",
        "seed_bases":{k:list(v) for k,v in FAMILIES.items()},
        "n_prior_distinct_target_sources_excluded":old_count,
        "n_per_family":SIZE,
        "cohorts":{
            family:{
                "n":SIZE,
                "roster_sha256":fingerprint(cases),
                "input_sha256":digest(json.dumps([r["x"] for r in cases],
                   sort_keys=True,separators=(",",":")).encode()),
                "target_sha256":digest(json.dumps([r["target"] for r in cases],
                   sort_keys=True,separators=(",",":")).encode()),
                "min_rectangle":min(r["rectangles"] for r in cases),
            } for family,cases in new.items()
        }
    }
    return new,manifest


def load_frozen_thresholds(path,old_calibration):
    raw=Path(path).read_bytes()
    c=json.loads(raw)
    if (c["schema"]!=CAL_SCHEMA or c["protocol_sha256"]!=protocol_sha() or
        c["has_generated_fresh_smc27_puzzles"] is not False or
        c["old_smc26_calibration_sha256"]!=OLD_CAL_SHA or
        c["old_smc26_training_bundle_sha256"]!=OLD_BUNDLE_SHA):
        raise ValueError("SMC27 source calibration not frozen before test")
    if tuple(c["original_teacher_names"])!=TEACHERS or set(c["teacher_calibration"])!=set(TEACHERS):
        raise ValueError("calibration did not include all four original teacher seeds")
    for teacher in TEACHERS:
        item=c["teacher_calibration"][teacher]["threshold_selection"]
        if item["selected_threshold"] not in (.15,.30,.50,.75,1.,1.5,2.,3.,5.):
            raise ValueError("postcalibration threshold outside frozen grid")
    if digest(Path(old_calibration).read_bytes())!=OLD_CAL_SHA:
        raise ValueError("original SMC26 old calibration source changed")
    return c,raw


def actions_for_effects(pred,threshold):
    return {
        "calibrated_selected":policy(pred["calibrated_median"],threshold),
        "uniform_selected":policy(pred["uniform_median"],threshold),
        "calibrated_fixed015":policy(pred["calibrated_median"],.15),
        "always_abstain":0,
        "shifted_selected":policy(pred["shifted_median"],threshold),
        "top_digit_selected":policy(pred["top_digit_effect"],threshold),
    }


def predict(original,independent,conditioned,bundle,oldcal,threshold_cal,
            state_path,allow_fetch):
    torch.set_num_threads(2)
    calibrated,calbytes=load_frozen_thresholds(threshold_cal,oldcal)
    models,comps,old,original_metadata=load_all_models(
        original,independent,conditioned,bundle,oldcal,allow_fetch)
    if (calibrated["old_smc26_source_snapshot"]!=original_metadata["source_snapshot"] or
        digest(Path(bundle).read_bytes())!=OLD_BUNDLE_SHA):
        raise ValueError("training/checkpoint snapshot differs since old-only calibration")
    rosters,manifest=new_rosters(old)
    stage_archive={}
    out={}
    for family,cases in rosters.items():
        x=torch.tensor([r["x"] for r in cases],dtype=torch.long)
        positions=goal_positions(x)
        archive={}
        data={}
        for teacher in TEACHERS:
            component=f"{teacher}_SAE71"
            tid,sae,mu,scale=comps[component]
            if tid!=teacher:
                raise ValueError("frozen SAE not source-aligned with teacher")
            with torch.no_grad():
                stage=models[teacher].reason_states(x)[1].detach().float().cpu().contiguous()
            if tuple(stage.shape)!=(SIZE,81,48):
                raise ValueError("incorrect source FP32 physical hidden state")
            archive[teacher]=stage
            native=copy.deepcopy(models[teacher]).double().eval()
            cov,dec=affine_physical(sae,mu,scale)
            selected=calibrated["teacher_calibration"][teacher]["threshold_selection"]["selected_threshold"]
            per=[]
            for pi,row in enumerate(cases):
                slots=[]
                for j,pos in enumerate(positions[pi].tolist()):
                    info,kill,sham=plan_target_native(
                        stage[pi,pos],sae,mu,scale,cov,dec,
                        f"SMC27|{family}|{pi}|{component}|{j}")
                    record={"slot":j,"token":pos,"status":info["status"],
                            "feature":info["feature"],"input_plan":info}
                    if info["status"]=="eligible":
                        clean,tangent=label_blind_tangent(
                            native,stage[pi],pos,kill,sham)
                        cand=candidate_predictions(clean,tangent,BASE_TEMPERATURE)
                        record.update(cand)
                        record["native_clean_logits"]=clean.tolist()
                        record["edit_kill"]=kill.tolist()
                        record["edit_sham"]=sham.tolist()
                        record["actions"]=actions_for_effects(cand["predictions"],selected)
                        record["frozen_selected_threshold"]=selected
                    slots.append(record)
                per.append({"puzzle_index":pi,"seed":row["seed"],
                            "visible_input":row["x"],
                            "blank_positions":positions[pi].tolist(),
                            "goals":slots})
            data[component]=per
            admissible=sum(g["status"]=="eligible" for p in per for g in p["goals"])
            print("SMC27_INPUT_ONLY",family,teacher,
                  "selected_threshold",selected,"eligible",admissible,flush=True)
        stage_archive[family]=archive
        out[family]=data
    state_path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(stage_archive,state_path)
    return {
        "schema":PRED_SCHEMA,"protocol_sha256":protocol_sha(),
        "old_smc26_bundle_sha256":OLD_BUNDLE_SHA,
        "old_smc26_calibration_sha256":OLD_CAL_SHA,
        "old_only_selected_threshold_calibration_sha256":digest(calbytes),
        "original_teacher_source_snapshot":original_metadata["source_snapshot"],
        "selected_source_stage1_filename":state_path.name,
        "selected_source_stage1_sha256":digest(state_path.read_bytes()),
        "test_labels_present":False,
        "test_effects_present":False,
        "test_answer_conditioned_gradients_present":False,
        "policy_threshold_fixed_baseline":.15,
        "offline_action_cost":.10,
        "correct_effect_neutral_band":.15,
        "teacher_names":list(TEACHERS),
        "policy_names":list(POLICIES),
        "test_source_manifest":manifest,
        "test_prediction_data":out,
        "original_old_T":BASE_TEMPERATURE,
        "numeric_precision":"source FP32 state, FP64 original native 3-block +/-0.08 tangent",
    }


def main():
    p=argparse.ArgumentParser()
    for name in ("original-artifact","independent-artifact","conditioned-receipt",
                 "bundle","old-calibration","threshold-calibration","source-state","out"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--fetch-source",action="store_true")
    a=p.parse_args()
    report=predict(a.original_artifact,a.independent_artifact,
                   a.conditioned_receipt,a.bundle,a.old_calibration,
                   a.threshold_calibration,a.source_state,a.fetch_source)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    print("SMC27_PREDICTION_FROZEN",report["selected_source_stage1_sha256"],flush=True)


if __name__=="__main__":
    main()

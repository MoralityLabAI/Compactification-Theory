"""SMC-25 native finite-effect scoring after target-blind source predictions freeze."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random
import sys

import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from smc14_causal_transfer import source_group_context
from smc24_native_score import (
    native_score_case,admitted_frozen_stage1,check_archived_edit
)
from smc25_candidate_effects import (
    CAL_SCHEMA,PRED_SCHEMA,FAMILIES,B_COMPONENTS,TEACHERS,
    GOALS,PUZZLES,SMALL,sha,seed,protocol_digest,generate_fresh,
    load_source,check_frozen_calibration,candidate_predictions
)

SCORE_SCHEMA="SMC-25/finite-native-all-candidate-scored-20261117-v1"
CONTROLS=("uniform_median","top_digit_effect","shifted_median")
BOOT=2000
MIN_GOALS=60
MIN_PUZZLES=30


def sign_match(actual,pred):
    return bool(actual!=0. and pred!=0. and (actual>0)==(pred>0))


def summary(rows,family,component,draws=BOOT):
    if len(rows)!=PUZZLES:
        raise ValueError("64 puzzle clusters required")
    samples=[]
    statuses=Counter()
    sign_p=sign_uniform=0
    for r in rows:
        eligible=[]
        for g in r["goals"]:
            statuses[g["status"]]+=1
            if g["status"]!="eligible":
                continue
            observed=float(g["observed"]["true_D"])
            p=g["predictions"]
            chosen=p["calibrated_median"]
            eligible.append({
                "gain":{c:abs(observed-p[c])-abs(observed-chosen) for c in CONTROLS},
                "errors":{m:abs(observed-p[m]) for m in ("calibrated_median",)+CONTROLS},
                "true_d":observed,
            })
            sign_p+=int(sign_match(observed,p["calibrated_median"]))
            sign_uniform+=int(sign_match(observed,p["uniform_median"]))
        samples.append({
            "n":len(eligible),
            "gain":{c:sum(e["gain"][c] for e in eligible)/len(eligible)
                    for c in CONTROLS} if eligible else None,
            "MAE":{m:sum(e["errors"][m] for e in eligible)/len(eligible)
                   for m in ("calibrated_median",)+CONTROLS} if eligible else None,
        })
    n=sum(s["n"] for s in samples)
    pcount=sum(s["n"]>0 for s in samples)
    if n==0:
        return {"status":"no_eligible_goals","eligible_goals":0,
                "eligible_puzzles":0,"registered_pass":False}
    average={mode:sum(s["gain"][mode] for s in samples if s["n"])/pcount
             for mode in CONTROLS}
    mae={mode:sum(s["MAE"][mode] for s in samples if s["n"])/pcount
         for mode in ("calibrated_median",)+CONTROLS}
    rng=random.Random(seed("bootstrap",20262517,family,component))
    boot={control:[] for control in CONTROLS}
    valid=0
    for _ in range(draws):
        ids=[rng.randrange(len(samples)) for _ in samples]
        taken=[samples[i] for i in ids if samples[i]["n"]]
        if not taken:
            continue
        valid+=1
        for control in CONTROLS:
            boot[control].append(sum(x["gain"][control] for x in taken)/len(taken))
    for control in CONTROLS:boot[control].sort()
    ci95={c:[boot[c][int(.025*valid)],boot[c][int(.975*valid)-1]]
          if valid>=1990 else None for c in CONTROLS}
    ci99={c:[boot[c][int(.003125*valid)],boot[c][int(.996875*valid)-1]]
          if valid>=1990 else None for c in CONTROLS}
    admitted=(n>=MIN_GOALS and pcount>=MIN_PUZZLES and valid>=1990)
    passed=bool(admitted and sign_p>sign_uniform and
                all(ci99[c][0]>0 for c in CONTROLS))
    return {
        "status":"measured" if admitted else "underpowered_coverage",
        "eligible_goals":n,"eligible_puzzles":pcount,
        "empty_puzzles":PUZZLES-pcount,
        "preoutcome_status_counts":dict(statuses),
        "mean_puzzle_MAE":mae,
        "mean_puzzle_gain":average,
        "bootstrap_valid":valid,
        "puzzle_bootstrap95_gain":ci95,
        "puzzle_bootstrap99_375_gain":ci99,
        "calibrated_median_sign_correct":sign_p,
        "uniform_median_sign_correct":sign_uniform,
        "registered_pass":passed,
    }


def check_frozen_prediction(pred,cal_raw,cal,snapshot,manifest):
    if pred["schema"]!=PRED_SCHEMA or pred["protocol_sha256"]!=protocol_digest():
        raise ValueError("unregistered frozen prediction")
    if pred["calibration_sha256"]!=sha(cal_raw):
        raise ValueError("frozen earlier calibration artifact differs")
    check_frozen_calibration(cal,snapshot)
    if (pred["source_snapshot"]!=snapshot or pred["test_manifest"]!=manifest or
        pred["contains_test_labels"] is not False or
        pred["uses_test_true_digit_gradient"] is not False or
        pred["finite_tangent_radius"]!=SMALL):
        raise ValueError("prediction source, label gate or physical radius changed")
    if set(pred["test_prediction_data"])!=set(FAMILIES):
        raise ValueError("prediction missing fresh families")
    if set(pred["archived_source_stage1"])!=set(FAMILIES):
        raise ValueError("prediction missing canonical FP32 states")
    for family in FAMILIES:
        if set(pred["archived_source_stage1"][family])!=set(TEACHERS):
            raise ValueError("prediction missing one of two independent teachers")


def score_native(original,independent,conditioned,calibration,prediction,allow_fetch):
    torch.set_num_threads(2)
    cal_raw=Path(calibration).read_bytes()
    pred_raw=Path(prediction).read_bytes()
    cal=json.loads(cal_raw)
    frozen=json.loads(pred_raw)
    models,components,old,snapshot=load_source(original,independent,conditioned,allow_fetch)
    rosters,manifest=generate_fresh(old)
    check_frozen_prediction(frozen,cal_raw,cal,snapshot,manifest)
    ctx,_=source_group_context(models,components,rosters,tuple(FAMILIES))
    canonical,drift=admitted_frozen_stage1(frozen,ctx)
    result={}
    for family,cases in rosters.items():
        cells={}
        for component in B_COMPONENTS:
            tid,sae,mean,scale=components[component]
            cov,dec=__import__("smc23_native_first").affine_physical(sae,mean,scale)
            predictions=frozen["test_prediction_data"][family][component]
            if len(predictions)!=PUZZLES:
                raise ValueError("source prediction puzzle count changed")
            model=ctx[tid]["model64"]
            stage=canonical[family][tid]["stage"]
            clean=canonical[family][tid]["clean"]
            native=ctx[tid]["cohorts"][family]
            records=[]
            for pi in range(PUZZLES):
                p=predictions[pi]
                row=cases[pi]
                if p["seed"]!=row["seed"] or p["visible_input"]!=row["x"] or p["puzzle_index"]!=pi:
                    raise ValueError("prediction puzzle identity changed")
                pos=native["positions"][pi].tolist()
                if p["positions"]!=pos or len(p["goals"])!=GOALS:
                    raise ValueError("preselected goal count or positions changed")
                entries=[]
                for j,g in enumerate(p["goals"]):
                    tok=pos[j]
                    if g["goal_index"]!=j or g["token"]!=tok:
                        raise ValueError("frozen goal order changed")
                    item={"goal_index":j,"token":tok,"status":g["status"],
                          "feature":g["feature"],"input_plan":g["input_plan"]}
                    if g["status"]=="eligible":
                        physical={"archived_h":stage[pi,tok],
                                  "edit_kill":g["edit_kill"],
                                  "edit_sham":g["edit_sham"]}
                        kill,sham=check_archived_edit(
                            physical,g["input_plan"],sae,mean,scale,cov,dec)
                        if float((torch.tensor(g["native_clean_logits"],dtype=torch.float64) -
                                  clean[pi,tok]).abs().max())>1e-4:
                            raise ValueError("source clean logits mismatch after archived prediction")
                        truth=int(native["truth_at_goals"][pi,j])
                        observed=native_score_case(model,stage[pi],clean[pi],
                                                   truth,tok,g["feature"],kill,sham)
                        frozen_rival=int(g["rival_for_digit"][truth-1])
                        if observed["frozen_clean_rival"]!=frozen_rival:
                            raise ValueError("clean-rival mapping changed from frozen prediction")
                        # Recompute all probability-weighted reductions WITHOUT labels.
                        recompute=candidate_predictions(
                            torch.tensor(g["native_clean_logits"],dtype=torch.float64),
                            torch.zeros(10,dtype=torch.float64),
                            float(g["calibration_temperature"]))
                        # Weighted reductions use original classwise effects and q.
                        from smc25_candidate_effects import weighted_median
                        v=g["classwise_effects"];q=g["calibrated_digit_distribution"]
                        rotated=[q[(k-1)%9] for k in range(9)]
                        predictions_expected={
                            "calibrated_median":weighted_median(v,q),
                            "uniform_median":weighted_median(v,[1./9.]*9),
                            "top_digit_effect":v[int(g["model_top_digit"])-1],
                            "shifted_median":weighted_median(v,rotated),
                            "calibrated_mean":sum(x*y for x,y in zip(v,q)),
                            "uniform_mean":sum(v)/9,"zero_effect":0.}
                        if any(abs(float(predictions_expected[k])-float(g["predictions"][k]))>1e-9
                               for k in predictions_expected):
                            raise ValueError("frozen candidate-effect aggregator changed")
                        item["predictions"]=g["predictions"]
                        item["observed"]=observed
                        item["oracle_class_tangent"]=float(v[truth-1])
                        item["oracle_finite_taylor_residual"]=float(
                            observed["true_D"]-float(v[truth-1]))
                    entries.append(item)
                records.append({"puzzle_index":pi,"seed":row["seed"],
                                "goal_positions":pos,"goals":entries})
            stats=summary(records,family,component)
            cells[component]={"summary":stats,"per_puzzle":records}
            print("SMC25_SCORE",family,component,"n",stats["eligible_goals"],
                  "gains",stats.get("mean_puzzle_gain"),"pass",stats["registered_pass"],flush=True)
        result[family]=cells
    pairs={name:all(result[family][name]["summary"]["registered_pass"]
                    for family in FAMILIES) for name in B_COMPONENTS}
    return {
        "schema":SCORE_SCHEMA,"protocol_sha256":protocol_digest(),
        "calibration_sha256":sha(cal_raw),"prediction_sha256":sha(pred_raw),
        "source_snapshot":snapshot,"test_manifest":manifest,
        "replay_max_source_stage1_abs":drift,
        "test_data":result,"n_unique_puzzles":128,
        "n_goal_slots":PUZZLES*GOALS*len(B_COMPONENTS)*len(FAMILIES),
        "pair_success":pairs,
        "program_success":any(pairs[n] for n in B_COMPONENTS[:2]) and any(
            pairs[n] for n in B_COMPONENTS[2:]),
        "limitations":"Physical finite local directional response at radius 0.08 and source-frozen teacher-native finite erasure score."
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--original-artifact",type=Path,required=True)
    p.add_argument("--independent-artifact",type=Path,required=True)
    p.add_argument("--conditioned-receipt",type=Path,required=True)
    p.add_argument("--calibration",type=Path,required=True)
    p.add_argument("--prediction",type=Path,required=True)
    p.add_argument("--fetch-source",action="store_true")
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    result=score_native(a.original_artifact,a.independent_artifact,
                        a.conditioned_receipt,a.calibration,a.prediction,
                        a.fetch_source)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("SMC25_NATIVE_COMPLETE",result["program_success"],str(a.out),flush=True)


if __name__=="__main__":
    main()

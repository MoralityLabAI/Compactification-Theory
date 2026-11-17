"""SMC-24 native outcome scoring after frozen prediction-only artifact exists.

No model fit or gradient training occurs here. All four predictions are taken
from immutable input-only cases; true Sudoku digits enter ONLY this scorer.
"""
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
from smc15_tangent_conditioning import load_and_lock_inputs
from smc23_native_first import affine_physical, four_arm_logits, plan_target_native
from smc24_gradient_predict import (
    B_COMPONENTS, TEACHERS, FAMILIES, GOALS, PUZZLES,
    BOOT, MIN_GOALS, MIN_PUZZLES, PRED_SCHEMA,
    sha, proto_hash, seed, source_snapshot, fresh_rosters,
)

SCORE_SCHEMA="SMC-24/frozen-prediction-native-scored-20261117-v1"
BASELINES=("mean","shuffled","zero")


def check_input_predictions(frozen, original_bytes, snapshot,manifest):
    if (frozen["schema"]!=PRED_SCHEMA or frozen["protocol_sha256"]!=proto_hash() or
            frozen["source_snapshot"]!=snapshot or frozen["test_manifest"]!=manifest):
        raise ValueError("frozen prediction source, seed, or manifest changed")
    if frozen["test_targets_present"] is not False or frozen["test_gradient_from_test_labels"] is not False:
        raise ValueError("leaked test target or test-label gradient in prediction source")
    if set(frozen["test_prediction_data"])!=set(FAMILIES):
        raise ValueError("incorrect new-test family roster")
    text=original_bytes.decode("utf-8")
    for forbidden in ('"truth_digit"', '"goal_truth_digits"', '"target":', '"native_true_margin"'):
        if forbidden in text:
            raise ValueError("prediction artifact contains forbidden test-score field")


def native_score_case(model, stage, clean, truth, token, feat, kill, sham):
    h=four_arm_logits(model,stage,int(token),kill,sham)
    if not torch.equal(h[0],h[1]):
        raise ArithmeticError("native zero self-patch altered logits")
    if float((h[0]-clean[int(token)]).abs().max())>1e-8:
        raise ArithmeticError("native clean source differed from frozen source")
    if not 1<=int(truth)<=9:
        raise ValueError("invalid Sudoku correct digit")
    target=int(truth)
    rival=clean[int(token)].clone()
    rival[target]=-float("inf")
    competitor=int(rival.argmax())
    margin=(h[:,target]-h[:,competitor]).tolist()
    return {
        "truth_digit":target,"frozen_clean_rival":competitor,
        "clean_correct":bool(int(h[0].argmax())==target),
        "true_D":float(margin[3]-margin[2]),
        "margins":{"clean":margin[0],"zero_self":margin[1],
                   "kill":margin[2],"sham":margin[3],"mirror":margin[4]},
        "kill_correct_to_wrong":bool(int(h[0].argmax())==target and int(h[2].argmax())!=target),
        "sham_correct_to_wrong":bool(int(h[0].argmax())==target and int(h[3].argmax())!=target),
        "native_linf_kill":float((h[2]-h[0]).abs().max()),
        "native_linf_sham":float((h[3]-h[0]).abs().max()),
        "selected_feature":int(feat),
    }


def error_contrast(true_d,pred):
    if set(pred)!={"ridge","mean","shuffled","zero"}:
        raise ValueError("four registered prediction arms required")
    true=float(true_d)
    return {baseline:abs(true-float(pred[baseline]))-abs(true-float(pred["ridge"]))
            for baseline in BASELINES}


def signed_correct(true,pred):
    if float(true)==0 or float(pred)==0:
        return False
    return (float(true)>0)==(float(pred)>0)


def summarize(rows,component,family,draws=BOOT):
    """Resample whole puzzles, using one mean per puzzle with eligible goals."""
    if len(rows)!=PUZZLES:
        raise ValueError("64 independent puzzle clusters required")
    scores=[]
    counts=Counter()
    low_eligibility=Counter()
    sign_ridge=sign_mean=sign_total=0
    kill_flips=sham_flips=0
    for row in rows:
        eligible=[]
        for g in row["goals"]:
            low_eligibility[g["status"]]+=1
            if g["status"]!="eligible":
                continue
            d=float(g["observed"]["true_D"])
            eligible.append((d,g["predictions"]))
            sign_total+=1
            sign_ridge+=int(signed_correct(d,g["predictions"]["ridge"]))
            sign_mean+=int(signed_correct(d,g["predictions"]["mean"]))
            kill_flips+=int(g["observed"]["kill_correct_to_wrong"])
            sham_flips+=int(g["observed"]["sham_correct_to_wrong"])
        if eligible:
            scores.append({
                "count":len(eligible),
                "errors":{mode:sum(abs(d-float(p[mode])) for d,p in eligible)/len(eligible)
                          for mode in ("ridge",)+BASELINES},
                "gains":{b:sum(error_contrast(d,p)[b] for d,p in eligible)/len(eligible)
                         for b in BASELINES},
            })
        else:
            scores.append(None)
    total=sum(s["count"] for s in scores if s is not None)
    nonempty=sum(s is not None for s in scores)
    if total==0:
        return {"status":"no_eligible_goals","eligible_goals":0,
                "eligible_puzzles":0,"registered_pass":False}
    measures={b:sum(s["gains"][b] for s in scores if s is not None)/nonempty
              for b in BASELINES}
    mae={mode:sum(s["errors"][mode] for s in scores if s is not None)/nonempty
         for mode in ("ridge",)+BASELINES}
    rng=random.Random(seed("bootstrap",20262417,family,component))
    samples={b:[] for b in BASELINES}
    valid=0
    for _ in range(draws):
        ids=[rng.randrange(PUZZLES) for _ in range(PUZZLES)]
        chosen=[scores[i] for i in ids if scores[i] is not None]
        if not chosen:
            continue
        valid+=1
        for b in BASELINES:
            samples[b].append(sum(s["gains"][b] for s in chosen)/len(chosen))
    for b in BASELINES:
        samples[b].sort()
    enough=(total>=MIN_GOALS and nonempty>=MIN_PUZZLES and valid>=1990)
    ci95={b:([samples[b][int(.025*valid)],samples[b][int(.975*valid)-1]]
             if valid>=1990 else None) for b in BASELINES}
    ci99={b:([samples[b][int(.003125*valid)],samples[b][int(.996875*valid)-1]]
             if valid>=1990 else None) for b in BASELINES}
    valid_sign=bool(sign_ridge>sign_mean)
    positive=bool(enough and valid_sign and all(ci99[b][0]>0 for b in BASELINES))
    return {
        "status":"measured" if enough else "underpowered_coverage",
        "eligible_goals":total,"eligible_puzzles":nonempty,
        "n_without_eligible":PUZZLES-nonempty,
        "observed_goal_status_counts":dict(low_eligibility),
        "puzzle_mean_MAE":mae,
        "puzzle_mean_ridge_gain":measures,
        "cluster_bootstrap_valid":valid,
        "puzzle_bootstrap95_gain":ci95,
        "puzzle_bootstrap99_375_gain":ci99,
        "ridge_sign_correct":sign_ridge,
        "mean_sign_correct":sign_mean,
        "n_nonzero_truth_goal_signs":sign_total,
        "kill_clean_correct_to_wrong":kill_flips,
        "sham_clean_correct_to_wrong":sham_flips,
        "registered_pass":positive,
    }


def score_native(original, independent, conditioned, prediction, fetch_source):
    torch.set_num_threads(2)
    raw=Path(prediction).read_bytes()
    frozen=json.loads(raw)
    models,components,old,verified=load_and_lock_inputs(
        original,independent,conditioned,allow_fetch=fetch_source)
    snapshot=source_snapshot(models,components,old,verified)
    rosters,manifest=fresh_rosters(old)
    check_input_predictions(frozen,raw,snapshot,manifest)
    ctx,_=source_group_context(models,components,rosters,tuple(FAMILIES))
    score_data={}
    for family,new_rows in rosters.items():
        cells={}
        for component in B_COMPONENTS:
            teacher_id,sae,mean,scale=components[component]
            cov,dec=affine_physical(sae,mean,scale)
            t=ctx[teacher_id]
            predictions=frozen["test_prediction_data"][family][component]
            if len(predictions)!=PUZZLES:
                raise ValueError("pre-frozen input prediction roster incomplete")
            records=[]
            for pi in range(PUZZLES):
                record=predictions[pi]
                r=new_rows[pi]
                if (record["puzzle_index"]!=pi or record["seed"]!=r["seed"] or
                        record["visible_input"]!=r["x"] or len(record["goals"])!=GOALS):
                    raise ValueError("source frozen predictions refer to other puzzle inputs")
                stage=t["cohorts"][family]["stage1"][pi]
                clean=t["cohorts"][family]["clean_logits"][pi]
                goals=t["cohorts"][family]["positions"][pi]
                truth=t["cohorts"][family]["truth_at_goals"][pi]
                if record["positions"]!=goals.tolist():
                    raise ValueError("test goal-position selection changed")
                checked=[]
                for j,g in enumerate(record["goals"]):
                    tok=int(goals[j])
                    if g["token"]!=tok or g["goal_index"]!=j:
                        raise ValueError("wrong frozen goal order")
                    info,kill,sham=plan_target_native(
                        stage[tok],sae,mean,scale,cov,dec,
                        f"SMC24|{family}|{pi}|{component}|{j}")
                    if info!=g["input_plan"] or info["status"]!=g["status"]:
                        raise ValueError("source geometry changed after archived prediction")
                    record_goal={"goal_index":j,"token":tok,"status":g["status"],
                                 "feature":g["feature"],"input_plan":info}
                    if g["status"]=="eligible":
                        assert kill is not None and sham is not None
                        if not torch.equal(kill,torch.tensor(g["edit_kill"],dtype=torch.float64)):
                            raise ValueError("source frozen erasure edit bytes changed")
                        if not torch.equal(sham,torch.tensor(g["edit_sham"],dtype=torch.float64)):
                            raise ValueError("source frozen sham edit bytes changed")
                        record_goal["predictions"]=g["predictions"]
                        record_goal["observed"]=native_score_case(
                            t["model64"],stage,clean,int(truth[j]),tok,
                            g["feature"],kill,sham)
                        record_goal["registered_MAE_gains"]=error_contrast(
                            record_goal["observed"]["true_D"],g["predictions"])
                    checked.append(record_goal)
                records.append({"puzzle_index":pi,"seed":r["seed"],
                                "goal_positions":goals.tolist(),"goals":checked})
            summary=summarize(records,component,family)
            cells[component]={"summary":summary,"per_puzzle":records}
            print("SMC24_SCORED",component,family,summary["eligible_goals"],
                  summary.get("puzzle_mean_ridge_gain"),summary["registered_pass"],flush=True)
        score_data[family]=cells
    decisions={c:all(score_data[family][c]["summary"]["registered_pass"]
                     for family in FAMILIES) for c in B_COMPONENTS}
    return {
        "schema":SCORE_SCHEMA,"protocol_sha256":proto_hash(),
        "source_snapshot":snapshot,"prediction_artifact_sha256":sha(raw),
        "test_manifest":manifest,
        "n_fresh_puzzles":PUZZLES*2,
        "n_components":len(B_COMPONENTS),
        "n_goal_slots":GOALS*PUZZLES*2*len(B_COMPONENTS),
        "calibration_used_for_predictions_only":True,
        "runtime":"native original 3-block FP64 PyTorch CPU after source FP32 stage1",
        "test_data":score_data,
        "pair_success":decisions,
        "program_success":any(decisions[x] for x in B_COMPONENTS[:2]) and any(
            decisions[x] for x in B_COMPONENTS[2:]),
        "limitations":"A learned calibration-only statistical gradient approximation, not native test gradient access or new teacher weights."
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--original-artifact",type=Path,required=True)
    parser.add_argument("--independent-artifact",type=Path,required=True)
    parser.add_argument("--conditioned-receipt",type=Path,required=True)
    parser.add_argument("--prediction",type=Path,required=True)
    parser.add_argument("--fetch-source",action="store_true")
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    result=score_native(args.original_artifact,args.independent_artifact,
                        args.conditioned_receipt,args.prediction,args.fetch_source)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print("SMC24_SCORE_COMPLETE",result["program_success"],str(args.out),flush=True)


if __name__=="__main__":
    main()

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
from smc23_native_first import affine_physical, four_arm_logits, fp32_code, TAU, MAX_DOSE_MULTIPLE, CODE_TOL
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
    if set(frozen.get("archived_source_stage1",{}))!=set(FAMILIES):
        raise ValueError("canonical pretest FP32 source-stage archive missing")
    if any(set(frozen["archived_source_stage1"][fam])!=set(TEACHERS) for fam in FAMILIES):
        raise ValueError("canonical source teacher anchors missing")
    text=original_bytes.decode("utf-8")
    for forbidden in ('"truth_digit"', '"goal_truth_digits"', '"target":', '"native_true_margin"'):
        if forbidden in text:
            raise ValueError("prediction artifact contains forbidden test-score field")


def admitted_frozen_stage1(frozen,ctx):
    """Input-only physical FP32 source state is canonical for native scoring.

    Native model recomputation checks cross-runner numerical parity, but no
    source-reoptimization or post-label edits are allowed after prediction.
    """
    store={}
    drift={}
    for family in FAMILIES:
        store[family]={}
        drift[family]={}
        for teacher in TEACHERS:
            part=frozen["archived_source_stage1"][family][teacher]
            archived=torch.tensor(part["tensor"],dtype=torch.float32)
            if tuple(part["shape"])!=(PUZZLES,81,48) or tuple(archived.shape)!=(PUZZLES,81,48):
                raise ValueError("saved source stage-one physical shape differs")
            if sha(archived.contiguous().numpy().tobytes())!=part["tensor_sha256"]:
                raise ValueError("canonical archived FP32 tensor bytes altered")
            recomputed=ctx[teacher]["cohorts"][family]["stage1"]
            mismatch=float((archived-recomputed).abs().max())
            if mismatch>1e-4:
                raise ArithmeticError("cross-runner stage-one source drift exceeds preoutcome threshold")
            teacher64=ctx[teacher]["model64"]
            with torch.no_grad():
                state=archived.double()
                for _ in range(3):
                    state=teacher64.block(state)
                clean=teacher64.head(state).detach()
            if not bool(torch.isfinite(clean).all()):
                raise ArithmeticError("nonfinite native clean task outputs")
            drift[family][teacher]=mismatch
            store[family][teacher]={"stage":archived,"clean":clean}
    return store,drift


def check_archived_edit(physical, info, sae, mean, scale, cov, dec):
    """Re-validate archived feature choice and input constraints, outcome-blind."""
    if info["status"]!="eligible":
        raise ValueError("only eligible frozen edits can enter native scoring")
    target=int(info["feature"])
    active=[int(a) for a in info["active_ids"]]
    if len(active)!=4 or len(set(active))!=4 or target!=active[0]:
        raise ValueError("frozen target-native top-four selection changed")
    kill=torch.tensor(physical["edit_kill"],dtype=torch.float64)
    sham=torch.tensor(physical["edit_sham"],dtype=torch.float64)
    if kill.shape!=(48,) or sham.shape!=(48,):
        raise ValueError("wrong frozen physical intervention size")
    if not bool(torch.isfinite(kill).all()) or not bool(torch.isfinite(sham).all()):
        raise ValueError("nonfinite frozen input-only edit")
    dose=float(kill.norm())
    if (abs(dose-float(sham.norm()))>1e-8 or
        abs(dose-float(info["dose"]))>1e-8):
        raise ValueError("frozen hard sham is not equal physical dose")
    code=fp32_code(sae,mean,scale,physical["archived_h"])
    values,ids=code.topk(5,sorted=True)
    if active!=ids[:4].tolist() or bool((values[:4]<=0).any()) or not float(values[0])>float(values[1]):
        raise ValueError("archived source SAE target-native top-four changed")
    if float((values-torch.tensor(info["top5_activities"],dtype=torch.float64)).abs().max())>1e-4:
        raise ValueError("source FP32 SAE reencoding drifts from frozen source")
    if not 0<dose<=MAX_DOSE_MULTIPLE*float(code[target])*float(dec[target].norm())+1e-8:
        raise ValueError("frozen erasure violates physical decoder dose cap")
    A=cov[active]
    affine=torch.tensor([float(code[target])+TAU,0,0,0],dtype=torch.float64)
    if float((A@kill-affine).abs().max())>1e-8:
        raise ValueError("archived kill violates physical selective encoder constraints")
    if float((A@sham).abs().max())>1e-8:
        raise ValueError("archived sham is not tangent to all four active encoder rows")
    killed=fp32_code(sae,mean,scale,physical["archived_h"].double()-kill)
    sham_code=fp32_code(sae,mean,scale,physical["archived_h"].double()-sham)
    if float(killed[target])>1e-7:
        raise ValueError("archived target-native code did not erase")
    for i in active[1:]:
        if float(killed[i])<=0 or abs(float(killed[i]-code[i]))>CODE_TOL*(1+float(code[i])):
            raise ValueError("archived kill fails other active SAE support")
    for i in active:
        if float(sham_code[i])<=0 or abs(float(sham_code[i]-code[i]))>CODE_TOL*(1+float(code[i])):
            raise ValueError("archived sham lost original active SAE support")
    return kill,sham


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
    canonical,drift=admitted_frozen_stage1(frozen,ctx)
    score_data={}
    for family,new_rows in rosters.items():
        cells={}
        for component in B_COMPONENTS:
            teacher_id,sae,mean,scale=components[component]
            cov,dec=affine_physical(sae,mean,scale)
            teacher=ctx[teacher_id]["model64"]
            saved_stage=canonical[family][teacher_id]["stage"]
            saved_clean=canonical[family][teacher_id]["clean"]
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
                stage=saved_stage[pi]
                clean=saved_clean[pi]
                goals=t["cohorts"][family]["positions"][pi]
                truth=t["cohorts"][family]["truth_at_goals"][pi]
                if record["positions"]!=goals.tolist():
                    raise ValueError("test goal-position selection changed")
                checked=[]
                for j,g in enumerate(record["goals"]):
                    tok=int(goals[j])
                    if g["token"]!=tok or g["goal_index"]!=j:
                        raise ValueError("wrong frozen goal order")
                    info=g["input_plan"]
                    if info["status"]!=g["status"] or info["feature"]!=g["feature"]:
                        raise ValueError("registered pre-outcome eligibility or feature identity changed")
                    record_goal={"goal_index":j,"token":tok,"status":g["status"],
                                 "feature":g["feature"],"input_plan":info}
                    if g["status"]=="eligible":
                        # Task labels are never passed into this input check.
                        physical={"archived_h":stage[tok],
                                  "edit_kill":g["edit_kill"],"edit_sham":g["edit_sham"]}
                        kill,sham=check_archived_edit(physical,info,sae,mean,scale,cov,dec)
                        record_goal["predictions"]=g["predictions"]
                        record_goal["observed"]=native_score_case(
                            teacher,stage,clean,int(truth[j]),tok,
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
        "source_stage1_cross_runner_max_abs":drift,
        "canonical_physical_source":"archived input-only FP32 stage1 from frozen prediction job",
        "n_fresh_puzzles":PUZZLES*2,
        "n_components":len(B_COMPONENTS),
        "n_goal_slots":GOALS*PUZZLES*2*len(B_COMPONENTS),
        "calibration_used_for_predictions_only":True,
        "runtime":"native original 3-block FP64 PyTorch CPU after archived source FP32 stage1",
        "test_data":score_data,
        "pair_success":decisions,
        "program_success":any(decisions[x] for x in B_COMPONENTS[:2]) and any(
            decisions[x] for x in B_COMPONENTS[2:]),
        "limitations":"Calibration-frozen hidden-state gradient prediction; no test-label-gradient fit or new teacher weights."
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

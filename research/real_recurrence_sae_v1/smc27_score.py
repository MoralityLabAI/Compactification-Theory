"""SMC-27 Stage C: native finite erasure scoring of frozen abstention actions.

Loads four unchanged original SMC26 teachers/SAEs and the separately locked
SMC27 calibration-only threshold choices. Test digits enter only after
the input-only source state and physical edits are revalidated.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from smc14_causal_transfer import goal_positions
from smc23_native_first import affine_physical
from smc24_native_score import check_archived_edit,native_score_case
from smc25_candidate_effects import candidate_predictions,weighted_median
from smc26_train import load_all_models,TEACHERS,digest
from smc26_score import decision_utility,target_class
from smc27_calibrate import (SCHEMA as CAL_SCHEMA,OLD_BUNDLE_SHA,
                            OLD_CAL_SHA,protocol_sha,policy)
from smc27_predict import (FAMILIES,SIZE,GOALS,POLICIES,
                           PRED_SCHEMA,actions_for_effects,new_rosters)

SCHEMA="SMC27/frozen-calibration-abstention-native-scored-v1"
CONTROLS=("always_abstain","calibrated_fixed015","uniform_selected")
SECONDARY=("shifted_selected","top_digit_selected")
BOOTS=5000
MIN_GOALS=200
MIN_PUZZLES=80
RATE_LOW=.05
RATE_HIGH=.90


def summary(rows,teacher,family,cal_admitted,capability,draws=BOOTS):
    """All eight originally selected slots in every 128-puzzle bootstrap."""
    if len(rows)!=SIZE or any(len(row["goals"])!=GOALS for row in rows):
        raise ValueError("source eight-goal 128-puzzle roster must remain complete")
    utilities=[]
    elig=elig_puzzles=acted=neutral=0
    statuses={}
    effects=[]
    for puzzle in rows:
        per={name:0. for name in POLICIES}
        local=0
        for goal in puzzle["goals"]:
            status=goal["status"]
            statuses[status]=statuses.get(status,0)+1
            if status!="eligible":
                continue
            local+=1;elig+=1
            d=float(goal["observed"]["true_D"])
            effects.append(d)
            truth=target_class(d)
            neutral+=int(truth==0)
            actions=goal["actions"]
            acted+=int(actions["calibrated_selected"]!=0)
            if set(actions)!=set(POLICIES):
                raise ValueError("frozen action list changed")
            for name in POLICIES:
                score=decision_utility(int(actions[name]),truth)
                if abs(goal["utility"][name]-score)>1e-10:
                    raise ValueError("stored native utility failed recurrence-independent replay")
                per[name]+=score
        elig_puzzles+=int(local>0)
        utilities.append({name:per[name]/GOALS for name in POLICIES})
    if len(effects)!=elig or sum(statuses.values())!=SIZE*GOALS:
        raise ValueError("source original slot accounting failed")
    if elig==0:
        return {"status":"UNDERPOWERED","registered_pass":False,
                "eligible_goals":0,"eligible_puzzles":0}
    means={name:sum(u[name] for u in utilities)/SIZE for name in POLICIES}
    differences=[{name:u["calibrated_selected"]-u[name]
                  for name in CONTROLS} for u in utilities]
    gains={name:sum(d[name] for d in differences)/SIZE for name in CONTROLS}
    root=f"SMC27|20263717|{teacher}|{family}".encode()
    rng=random.Random(int.from_bytes(hashlib.sha256(root).digest()[:8],"big"))
    quantiles={c:[] for c in CONTROLS}
    for _ in range(draws):
        ids=[rng.randrange(SIZE) for i in range(SIZE)]
        for c in CONTROLS:
            quantiles[c].append(sum(differences[idx][c] for idx in ids)/SIZE)
    for v in quantiles.values():
        v.sort()
    ci95={c:[quantiles[c][int(.025*draws)],quantiles[c][int(.975*draws)-1]]
          for c in CONTROLS}
    # 8 teacher/family cells times 3 controls => 24 simultaneous contrasts;
    # 0.05 / (2*24) = 0.0010416666667 in each tail.
    alpha_tail=.05/(2*24)
    family_ci={c:[quantiles[c][int(alpha_tail*draws)],
                  quantiles[c][int((1-alpha_tail)*draws)-1]] for c in CONTROLS}
    covered=elig>=MIN_GOALS and elig_puzzles>=MIN_PUZZLES
    action_rate=acted/elig
    rate_good=RATE_LOW<=action_rate<=RATE_HIGH
    valid=bool(cal_admitted and capability and covered and rate_good)
    passed=bool(valid and all(family_ci[c][0]>.02 for c in CONTROLS))
    return {
        "status":"CAPABILITY_MISMATCH" if not capability
                 else "CALIBRATION_INADMISSIBLE" if not cal_admitted
                 else "UNDERPOWERED" if not covered
                 else "ACTION_RATE_OUT_OF_BAND" if not rate_good
                 else "MEASURED",
        "registered_pass":passed,
        "teacher":teacher,"family":family,
        "n_puzzles":SIZE,"n_original_blank_slots":SIZE*GOALS,
        "eligible_goals":elig,"eligible_puzzles":elig_puzzles,
        "n_ineligible":SIZE*GOALS-elig,"eligibility_counts":statuses,
        "calibration_selected_threshold_admissible":bool(cal_admitted),
        "old_teacher_capability_match":bool(capability),
        "calibrated_selected_test_action_rate":action_rate,
        "neutral_true_effect_goals":neutral,
        "teacher_mean_true_D":sum(effects)/len(effects),
        "mean_policy_utility_original_slots":means,
        "mean_primary_utility_gain":gains,
        "bootstrap_draws":draws,"bootstrap_puzzles_per_draw":SIZE,
        "bootstrap95_gain":ci95,"bootstrap99_79167_gain":family_ci,
    }


def source_check(oldcal,thresholds,pred,bundle_bytes,stage_bytes,pre_bytes):
    if (digest(bundle_bytes)!=OLD_BUNDLE_SHA or digest(oldcal)!=OLD_CAL_SHA or
        pred["schema"]!=PRED_SCHEMA or pred["protocol_sha256"]!=protocol_sha() or
        thresholds["schema"]!=CAL_SCHEMA or thresholds["protocol_sha256"]!=protocol_sha()):
        raise ValueError("original registered teacher/old-only calibration data changed")
    if (pred["old_only_selected_threshold_calibration_sha256"]!=digest(pre_bytes) or
        pred["selected_source_stage1_sha256"]!=digest(stage_bytes) or
        pred["old_smc26_bundle_sha256"]!=OLD_BUNDLE_SHA):
        raise ValueError("frozen original source file or predictor artifact replaced")
    if (pred["test_labels_present"] is not False or
        pred["test_effects_present"] is not False or
        pred["test_answer_conditioned_gradients_present"] is not False):
        raise ValueError("pre-scoring target labels leaked to source prediction")
    if (pred["policy_threshold_fixed_baseline"]!=.15 or
        pred["correct_effect_neutral_band"]!=.15 or pred["offline_action_cost"]!=.10):
        raise ValueError("original fixed SMC26 reward or baseline changed")
    if (set(pred["test_prediction_data"])!=set(FAMILIES) or
        tuple(pred["teacher_names"])!=TEACHERS or tuple(pred["policy_names"])!=POLICIES):
        raise ValueError("original frozen teacher/policy population changed")
    if (tuple(thresholds["original_teacher_names"])!=TEACHERS or
        thresholds["has_generated_fresh_smc27_puzzles"] is not False):
        raise ValueError("old calibration stage used new puzzles")


def score(original,independent,conditioned,bundle,oldcal,calibration,
          prediction,source_state,allow_fetch):
    torch.set_num_threads(2)
    old_raw=Path(oldcal).read_bytes()
    cal_raw=Path(calibration).read_bytes()
    pred_raw=Path(prediction).read_bytes()
    bundle_bytes=Path(bundle).read_bytes()
    state_bytes=Path(source_state).read_bytes()
    c=json.loads(cal_raw)
    pred=json.loads(pred_raw)
    source_check(old_raw,c,pred,bundle_bytes,state_bytes,cal_raw)
    models,comps,old,old_meta=load_all_models(
        original,independent,conditioned,bundle,oldcal,allow_fetch)
    if (old_meta["source_snapshot"]!=pred["original_teacher_source_snapshot"] or
        old_meta["source_snapshot"]!=c["old_smc26_source_snapshot"]):
        raise ValueError("source old teacher model provenance differs")
    rosters,manifest=new_rosters(old)
    if pred["test_source_manifest"]!=manifest:
        raise ValueError("new puzzle/source selection changed after predictions frozen")
    archive=torch.load(source_state,weights_only=True,map_location="cpu")
    if set(archive)!=set(FAMILIES):
        raise ValueError("source state archive families differ")
    results={}
    drift={}
    for family,cases in rosters.items():
        x=torch.tensor([r["x"] for r in cases],dtype=torch.long)
        y=torch.tensor([r["target"] for r in cases],dtype=torch.long)
        positions=goal_positions(x)
        drift[family]={}
        results[family]={}
        for teacher in TEACHERS:
            key=f"{teacher}_SAE71"
            tid,sae,mu,scale=comps[key]
            stage=archive[family][teacher]
            if stage.dtype!=torch.float32 or tuple(stage.shape)!=(SIZE,81,48):
                raise ValueError("pre-scoring canonical FP32 state geometry wrong")
            with torch.no_grad():
                reproduced=models[teacher].reason_states(x)[1].detach().float()
            mismatch=float((stage-reproduced).abs().max())
            if mismatch>1e-4:
                raise ArithmeticError("pre-outcome canonical source state mismatch >1e-4")
            drift[family][teacher]=mismatch
            native=copy.deepcopy(models[teacher]).double().eval()
            with torch.no_grad():
                clean=stage.double()
                for _ in range(3):
                    clean=native.block(clean)
                clean=native.head(clean)
            cov,dec=affine_physical(sae,mu,scale)
            records=[]
            predrows=pred["test_prediction_data"][family][key]
            old_choice=c["teacher_calibration"][teacher]["threshold_selection"]
            chosen=old_choice["selected_threshold"]
            if len(predrows)!=SIZE:
                raise ValueError("source predictor changed puzzle count")
            for pi,p in enumerate(predrows):
                if (p["puzzle_index"]!=pi or p["seed"]!=cases[pi]["seed"] or
                    p["visible_input"]!=cases[pi]["x"] or
                    p["blank_positions"]!=positions[pi].tolist() or
                    len(p["goals"])!=GOALS):
                    raise ValueError("source predictor selected another puzzle or goal")
                verified=[]
                for j,g in enumerate(p["goals"]):
                    tok=int(positions[pi,j])
                    if (g["slot"]!=j or g["token"]!=tok or
                        g["feature"]!=g["input_plan"]["feature"] or
                        g["status"]!=g["input_plan"]["status"]):
                        raise ValueError("pre-outcome selected source feature changed")
                    case={"slot":j,"token":tok,"feature":g["feature"],
                          "status":g["status"],"input_plan":g["input_plan"]}
                    if g["status"]=="eligible":
                        physical={"archived_h":stage[pi,tok],
                                  "edit_kill":g["edit_kill"],
                                  "edit_sham":g["edit_sham"]}
                        kill,sham=check_archived_edit(
                            physical,g["input_plan"],sae,mu,scale,cov,dec)
                        source_clean=torch.tensor(g["native_clean_logits"],dtype=torch.float64)
                        if float((source_clean-clean[pi,tok]).abs().max())>1e-4:
                            raise ValueError("native clean source logits drifted")
                        check=candidate_predictions(source_clean,
                                 torch.zeros(10,dtype=torch.float64),2.)
                        if (check["rival_for_digit"]!=g["rival_for_digit"] or
                            g["model_top_digit"]!=check["model_top_digit"] or
                            max(abs(a-b) for a,b in zip(
                                check["calibrated_digit_distribution"],
                                g["calibrated_digit_distribution"]))>1e-10):
                            raise ValueError("label-blind source distribution or rivals differ")
                        d=g["classwise_effects"]
                        q=g["calibrated_digit_distribution"]
                        shifted=[q[(i-1)%9] for i in range(9)]
                        estimate={
                            "calibrated_median":weighted_median(d,q),
                            "uniform_median":weighted_median(d,[1/9]*9),
                            "shifted_median":weighted_median(d,shifted),
                            "top_digit_effect":d[g["model_top_digit"]-1]
                        }
                        if any(abs(estimate[k]-g["predictions"][k])>1e-9
                               for k in estimate):
                            raise ValueError("calibrated source candidate aggregation changed")
                        from smc27_predict import actions_for_effects
                        actions=actions_for_effects(g["predictions"],chosen)
                        if (actions!=g["actions"] or
                            g["frozen_selected_threshold"]!=chosen):
                            raise ValueError("frozen old-selected threshold or policy action altered")
                        truth=int(y[pi,tok])
                        measured=native_score_case(native,stage[pi],clean[pi],
                                                   truth,tok,g["feature"],kill,sham)
                        if measured["frozen_clean_rival"]!=g["rival_for_digit"][truth-1]:
                            raise ValueError("native true-digit clean rival differed from frozen prediction")
                        state=target_class(measured["true_D"])
                        case.update({
                            "predictions":g["predictions"],
                            "actions":actions,
                            "observed":measured,
                            "true_effect_class":state,
                            "utility":{
                                policyname:decision_utility(int(actions[policyname]),state)
                                for policyname in POLICIES},
                            "answer_conditioned_tangent":float(d[truth-1]),
                            "answer_conditioned_finite_residual":
                                float(measured["true_D"]-d[truth-1]),
                        })
                    verified.append(case)
                records.append({"puzzle_index":pi,"seed":cases[pi]["seed"],
                                "blank_positions":positions[pi].tolist(),"goals":verified})
            meta=old_meta["all_teacher_temperatures"][teacher]
            results[family][key]={
                "summary":summary(records,teacher,family,old_choice["admissible"],
                                   meta["capability_match"]),
                "per_puzzle":records,
            }
            out=results[family][key]["summary"]
            print("SMC27_NATIVE",teacher,family,"threshold",chosen,
                  "eligible",out["eligible_goals"],"gain",
                  out.get("mean_primary_utility_gain"),"pass",out["registered_pass"],flush=True)
    pairs={teacher:all(results[family][f"{teacher}_SAE71"]["summary"]["registered_pass"]
                       for family in FAMILIES) for teacher in TEACHERS}
    capability=bool(old_meta["all_four_capability_matched"])
    calibration_admitted=bool(c["all_selected_thresholds_admissible"])
    complete=all(results[f][f"{teacher}_SAE71"]["summary"]["status"] not in
                 ("UNDERPOWERED",) for f in FAMILIES for teacher in TEACHERS)
    success=(capability and calibration_admitted and complete and
             sum(pairs.values())>=3 and any(pairs[t] for t in ("B811","B919")))
    status=("CAPABILITY_MISMATCH" if not capability else
            "CALIBRATION_INADMISSIBLE" if not calibration_admitted else
            "UNDERPOWERED" if not complete else "PASS" if success else "FAIL")
    return {
        "schema":SCHEMA,"protocol_sha256":protocol_sha(),
        "old_bundle_sha256":digest(bundle_bytes),
        "old_calibration_sha256":digest(old_raw),
        "policy_calibration_sha256":digest(cal_raw),
        "input_only_prediction_sha256":digest(pred_raw),
        "canonical_FP32_state_sha256":digest(state_bytes),
        "source_snapshot":old_meta["source_snapshot"],
        "test_source_manifest":manifest,
        "canonical_stage1_host_max_abs":drift,
        "n_new_puzzles":2*SIZE,
        "n_preselected_goal_measurement_slots":2*SIZE*GOALS*len(TEACHERS),
        "all_original_teacher_competence_matched":capability,
        "all_calibration_thresholds_admissible":calibration_admitted,
        "teacher_two_cohort_success":pairs,
        "registered_scientific_status":status,
        "data":results,
        "original_SMC26_scientific_gate":"FAIL_0_OF_4",
    }


def main():
    parser=argparse.ArgumentParser()
    for name in ("original-artifact","independent-artifact","conditioned-receipt",
                 "bundle","old-calibration","calibration","prediction","source-state","out"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--fetch-source",action="store_true")
    a=parser.parse_args()
    report=score(a.original_artifact,a.independent_artifact,a.conditioned_receipt,
                 a.bundle,a.old_calibration,a.calibration,a.prediction,
                 a.source_state,a.fetch_source)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    print("SMC27_NATIVE_COMPLETE",report["registered_scientific_status"],
          report["n_preselected_goal_measurement_slots"],flush=True)


if __name__=="__main__":
    main()

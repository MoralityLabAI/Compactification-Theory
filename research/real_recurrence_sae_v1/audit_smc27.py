"""SMC-27 Stage D: independent stdlib source and registered utility receipt audit.

Verifies original SMC26 teacher/calibration bytes, full old-only policy
selection, 256 source-distinct Sudoku solutions, archived target-free physical
edits, native finite scored effects, original eight-slot utility and all
5000 puzzle-bootstrap decisions. No Torch or trained teacher is imported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from smc13_family import (new_full_solution,dig_unique,count_solutions,
                          rectangle_count,fingerprint,build_rosters)
from audit_smc26 import PRIOR_SOURCE_COHORTS

TEACHERS=("B501","B701","B811","B919")
FAMILIES={"smc27_novel30":(202637030000,30),
          "smc27_novel34":(202637040000,34)}
GRID=(.15,.30,.50,.75,1.,1.5,2.,3.,5.)
POLICIES=("calibrated_selected","uniform_selected","calibrated_fixed015",
          "always_abstain","shifted_selected","top_digit_selected")
CONTROLS=("always_abstain","calibrated_fixed015","uniform_selected")
OLD_BUNDLE_SHA="408348eeecc300e7518ee7e8ea14d71caac29821aff39aaaa5ee819c794231f1"
OLD_CAL_SHA="145312c04648847c409b506aecdf1a3464af855d4ec8aa638eb0d604185512c9"
BOOT=5000
SOURCE_SIZE=128
SOURCE_GOALS=8


def digest(raw):return hashlib.sha256(raw).hexdigest()


def protocol_sha():
    return digest((HERE/"SMC27_PROTOCOL.md").read_bytes())


def policy(p,t):
    p=float(p)
    return 1 if p>t else -1 if p< -t else 0


def source_truth(d):
    return policy(d,.15)


def utility(a,t):
    if a not in (-1,0,1) or t not in (-1,0,1):
        raise ValueError("unregistered ternary action/truth")
    return (1 if a==0 else -1) - .1*int(a!=0) if t==0 else (
        1 if a==t else -1 if a==-t else 0)-.1*int(a!=0)


def weighted_median(values,weights):
    pairs=sorted((float(v),i,float(w)) for i,(v,w) in enumerate(zip(values,weights)))
    if len(pairs)!=9 or any(w<0 for _,_,w in pairs) or abs(sum(w for _,_,w in pairs)-1)>1e-10:
        raise ValueError("registered nine-class weights invalid")
    cumul=0.
    for v,i,w in pairs:
        cumul+=w
        if cumul>=.5-1e-15:
            return v
    raise ValueError("class posterior median failed")


def softmax(values,T):
    z=[float(x)/float(T) for x in values]
    m=max(z);exp=[math.exp(x-m) for x in z]
    d=sum(exp)
    return [x/d for x in exp]


def _close(x,y,name="value"):
    if isinstance(x,dict) and isinstance(y,dict):
        if set(x)!=set(y):
            raise ValueError("independent results have differing keys: "+name)
        for k in x:_close(x[k],y[k],name+"."+k)
    elif isinstance(x,list) and isinstance(y,list):
        if len(x)!=len(y):
            raise ValueError("independent source results differ in length: "+name)
        for i,(a,b) in enumerate(zip(x,y)):_close(a,b,name+f".{i}")
    elif isinstance(x,(float,int)) and isinstance(y,(float,int)):
        if abs(float(x)-float(y))>1e-9:
            raise ValueError("independent result changed: "+name)
    elif x!=y:
        raise ValueError("independent result changed: "+name)


def old_calibration_replay(cal,old_metadata):
    if (cal["schema"]!="SMC27/calibration-only-selected-abstention-v1" or
        cal["protocol_sha256"]!=protocol_sha() or
        cal["has_generated_fresh_smc27_puzzles"] is not False or
        tuple(cal["original_teacher_names"])!=TEACHERS):
        raise ValueError("old-only pretest threshold calibration changed")
    source,original_manifest=build_rosters()
    rows=source["feature_calibration"]
    if (len(rows)!=64 or
        fingerprint(rows)!=cal["old_feature_calibration_roster_sha256"] or
        fingerprint(rows)!=old_metadata["source_calibration_roster_sha256"]):
        raise ValueError("original old calibration puzzle source differs")
    if set(cal["teacher_calibration"])!=set(TEACHERS):
        raise ValueError("some original source teacher was omitted from calibration")
    selected={}
    for teacher in TEACHERS:
        rec=cal["teacher_calibration"][teacher]
        per=rec["per_puzzle"]
        if len(per)!=64:
            raise ValueError("old-only calibration puzzle count changed")
        for pi,p in enumerate(per):
            if (p["puzzle_index"]!=pi or p["grid_seed"]!=rows[pi]["seed"] or
                len(p["goals"])!=8 or len(p["selected_positions"])!=8):
                raise ValueError("original 64x8 source calibration roster changed")
            blanks=[idx for idx,v in enumerate(rows[pi]["x"]) if v==0]
            positions=[blanks[(len(blanks)*j)//8] for j in range(8)]
            if p["selected_positions"]!=positions:
                raise ValueError("old calibration selected goal was not visible-mask-only")
            for j,g in enumerate(p["goals"]):
                if g["goal_index"]!=j or g["position"]!=positions[j]:
                    raise ValueError("calibration goal changed")
                if g["status"]!="eligible":
                    if "native_finite_D" in g:
                        raise ValueError("ineligible original calibration goal was scored")
                    continue
                if g["true_digit"]!=rows[pi]["target"][positions[j]]:
                    raise ValueError("calibration label changed")
                if g["true_class"]!=source_truth(g["native_finite_D"]):
                    raise ValueError("source old-only decision target changed")
        scores=[]
        for threshold in GRID:
            n=active=0;sum_utility=0.
            for p in per:
                for g in p["goals"]:
                    if g["status"]!="eligible":continue
                    n+=1
                    action=policy(g["calibrated_median"],threshold)
                    active+=int(action!=0)
                    sum_utility+=utility(action,g["true_class"])
            rate=active/n if n else 0
            scores.append({
                "threshold":threshold,
                "eligible_goals":n,"n_old_calibration_slots":512,
                "calibrated_action_rate":rate,
                "mean_utility_all_original_slots":sum_utility/512,
                "in_range":bool(n and .05<=rate<=.90),
            })
        stored=rec["threshold_selection"]
        _close(scores,stored["threshold_grid_scores"],f"old_{teacher}_threshold_grid")
        allowed=[x for x in scores if x["in_range"]]
        winner=max(allowed,key=lambda r:(r["mean_utility_all_original_slots"],r["threshold"])) if allowed else scores[-1]
        if (stored["selected_threshold"]!=winner["threshold"] or
            bool(stored["admissible"])!=bool(allowed) or
            stored["status"]!=("CALIBRATED" if allowed else "CALIBRATION_INADMISSIBLE")):
            raise ValueError("calibration selected threshold with test-informed data")
        selected[teacher]=winner["threshold"]
    if cal["all_selected_thresholds_admissible"]!=all(
        c["threshold_selection"]["admissible"] for c in cal["teacher_calibration"].values()):
        raise ValueError("pre-outcome four-teacher calibration admission was modified")
    return selected


def source_checks(old_raw,bundle_raw,cal_raw,pred_raw,stage_raw,score_raw):
    if digest(old_raw)!=OLD_CAL_SHA or digest(bundle_raw)!=OLD_BUNDLE_SHA:
        raise ValueError("wrong original SMC26 training and old-calibration artifacts")
    prior=json.loads(old_raw)
    cal=json.loads(cal_raw);pred=json.loads(pred_raw);scored=json.loads(score_raw)
    if (not prior["all_four_capability_matched"] or
        tuple(prior["teacher_names"])!=TEACHERS or
        any(x["temperature"]!=2. for x in prior["all_teacher_temperatures"].values())):
        raise ValueError("original four-teacher capability/source changed")
    h=protocol_sha()
    if (cal["protocol_sha256"]!=h or pred["protocol_sha256"]!=h or
        scored["protocol_sha256"]!=h):
        raise ValueError("registration was altered")
    if (pred["schema"]!="SMC27/calibration-frozen-input-only-causal-sign-policies-v1" or
        scored["schema"]!="SMC27/frozen-calibration-abstention-native-scored-v1"):
        raise ValueError("wrong source stage schema")
    if (cal["old_smc26_calibration_sha256"]!=OLD_CAL_SHA or
        cal["old_smc26_training_bundle_sha256"]!=OLD_BUNDLE_SHA or
        cal["old_smc26_source_snapshot"]!=prior["source_snapshot"]):
        raise ValueError("source old teacher calibration metadata changed")
    if (pred["old_only_selected_threshold_calibration_sha256"]!=digest(cal_raw) or
        scored["policy_calibration_sha256"]!=digest(cal_raw) or
        scored["old_bundle_sha256"]!=OLD_BUNDLE_SHA or
        scored["old_calibration_sha256"]!=OLD_CAL_SHA):
        raise ValueError("old-only fitted threshold or original SMC26 source bytes changed")
    if (pred["selected_source_stage1_sha256"]!=digest(stage_raw) or
        scored["canonical_FP32_state_sha256"]!=digest(stage_raw) or
        scored["input_only_prediction_sha256"]!=digest(pred_raw)):
        raise ValueError("original frozen target-free prediction physical state changed")
    if (not (pred["test_labels_present"] is False and pred["test_effects_present"] is False
             and pred["test_answer_conditioned_gradients_present"] is False) or
        '"truth_digit":' in pred_raw.decode() or '"target":' in pred_raw.decode()):
        raise ValueError("future correct digit leaked into pre-outcome prediction")
    if (pred["policy_threshold_fixed_baseline"]!=.15 or
        pred["offline_action_cost"]!=.10 or
        pred["correct_effect_neutral_band"]!=.15 or
        tuple(pred["policy_names"])!=POLICIES or
        tuple(pred["teacher_names"])!=TEACHERS):
        raise ValueError("frozen original SMC26 utility/policy admission changed")
    if (pred["test_source_manifest"]!=scored["test_source_manifest"] or
        set(pred["test_prediction_data"])!=set(FAMILIES) or
        set(scored["data"])!=set(FAMILIES)):
        raise ValueError("test cohorts changed since frozen prediction")
    return prior,cal,pred,scored


def independent_test_rosters():
    old,_=build_rosters()
    prior_x={tuple(r["x"]) for rows in old.values() for r in rows}
    prior_y={tuple(r["target"]) for rows in old.values() for r in rows}
    all_registered=((202636030000,30,128),(202636040000,34,128))+PRIOR_SOURCE_COHORTS
    for base,holes,n in all_registered:
        for i in range(n):
            answer=new_full_solution(base+i)
            x=dig_unique(answer,base+i+1000000,holes)
            prior_x.add(tuple(x));prior_y.add(tuple(answer))
    previous_count=len(prior_y)
    seen_x=set();seen_y=set()
    cohorts={}
    for family,(base,holes) in FAMILIES.items():
        cases=[]
        for i in range(SOURCE_SIZE):
            sd=base+i
            y=new_full_solution(sd)
            x=dig_unique(y,sd+1000000,holes)
            if (count_solutions(x,2)!=1 or x.count(0)!=holes or
                rectangle_count(y)<=0 or tuple(x) in prior_x or
                tuple(y) in prior_y or tuple(x) in seen_x or tuple(y) in seen_y):
                raise ValueError("fresh SMC27 source reused original or historic puzzle")
            seen_x.add(tuple(x));seen_y.add(tuple(y))
            cases.append({"seed":sd,"x":x,"target":y,
                          "holes":holes,"rectangles":rectangle_count(y)})
        cohorts[family]=cases
    return cohorts,previous_count


def verify_cases(prior,cal,pred,scored):
    prior_thresholds=old_calibration_replay(cal,prior)
    fresh,previous_count=independent_test_rosters()
    manifest=pred["test_source_manifest"]
    if (manifest["seed_bases"]!={k:list(v) for k,v in FAMILIES.items()} or
        manifest["n_per_family"]!=SOURCE_SIZE or
        manifest["n_prior_distinct_target_sources_excluded"]!=previous_count):
        raise ValueError("frozen Sudoku seed inventory differs from independent generator")
    n_orig=n_eligible=0
    for family,cases in fresh.items():
        if manifest["cohorts"][family]["roster_sha256"]!=fingerprint(cases):
            raise ValueError("full new Sudoku source target/puzzle SHA mismatch")
        for teacher in TEACHERS:
            component=f"{teacher}_SAE71"
            preds=pred["test_prediction_data"][family][component]
            scored_rows=scored["data"][family][component]["per_puzzle"]
            if len(preds)!=SOURCE_SIZE or len(scored_rows)!=SOURCE_SIZE:
                raise ValueError("teacher omitted a preregistered fresh puzzle")
            temp=prior["all_teacher_temperatures"][teacher]["temperature"]
            choice=prior_thresholds[teacher]
            for pi,(p,s) in enumerate(zip(preds,scored_rows)):
                source=cases[pi]
                if (p["puzzle_index"]!=pi or s["puzzle_index"]!=pi or
                    p["seed"]!=source["seed"] or s["seed"]!=source["seed"] or
                    p["visible_input"]!=source["x"]):
                    raise ValueError("native scoring changed original Sudoku source puzzle")
                blanks=[i for i,v in enumerate(source["x"]) if v==0]
                positions=[blanks[j*len(blanks)//8] for j in range(8)]
                if (p["blank_positions"]!=positions or
                    s["blank_positions"]!=positions or len(p["goals"])!=8 or
                    len(s["goals"])!=8):
                    raise ValueError("predeclared input-mask blank goal selection changed")
                for j,(g,h) in enumerate(zip(p["goals"],s["goals"])):
                    n_orig+=1
                    if (g["slot"]!=j or h["slot"]!=j or
                        g["token"]!=positions[j] or h["token"]!=positions[j] or
                        g["status"]!=h["status"] or g["feature"]!=h["feature"] or
                        g["input_plan"]!=h["input_plan"]):
                        raise ValueError("source selected feature or input-only edit plan mutated")
                    if g["status"]!="eligible":
                        if "observed" in h:raise ValueError("ineligible goal was scored")
                        continue
                    n_eligible+=1
                    if not 0<=g["feature"]<96 or len(g["classwise_effects"])!=9:
                        raise ValueError("original SAE selected feature or nine candidate effects invalid")
                    logits=g["native_clean_logits"]
                    if len(logits)!=10 or any(not math.isfinite(t) for t in logits):
                        raise ValueError("invalid clean source native full logits")
                    q=softmax(logits[1:10],temp)
                    if max(abs(a-b) for a,b in zip(q,g["calibrated_digit_distribution"]))>1e-10:
                        raise ValueError("frozen class distribution changed")
                    rivals=[max((k for k in range(10) if k!=y),key=lambda k:logits[k])
                            for y in range(1,10)]
                    if rivals!=g["rival_for_digit"]:
                        raise ValueError("native clean nine-digit rivals changed")
                    best=max(range(1,10),key=lambda y:logits[y])
                    if best!=g["model_top_digit"]:
                        raise ValueError("original clean model top digit differed")
                    shifted=[q[(i-1)%9] for i in range(9)]
                    effects=g["classwise_effects"]
                    expected={
                        "calibrated_median":weighted_median(effects,q),
                        "uniform_median":weighted_median(effects,[1/9]*9),
                        "shifted_median":weighted_median(effects,shifted),
                        "top_digit_effect":effects[best-1],
                    }
                    if any(abs(expected[k]-g["predictions"][k])>1e-10 for k in expected):
                        raise ValueError("classwise native candidate aggregator changed")
                    actions={
                        "calibrated_selected":policy(g["predictions"]["calibrated_median"],choice),
                        "uniform_selected":policy(g["predictions"]["uniform_median"],choice),
                        "calibrated_fixed015":policy(g["predictions"]["calibrated_median"],.15),
                        "always_abstain":0,
                        "shifted_selected":policy(g["predictions"]["shifted_median"],choice),
                        "top_digit_selected":policy(g["predictions"]["top_digit_effect"],choice),
                    }
                    if (actions!=g["actions"] or actions!=h["actions"] or
                        g["frozen_selected_threshold"]!=choice):
                        raise ValueError("calibration-selected threshold changed on fresh puzzle")
                    kill=g["edit_kill"];sham=g["edit_sham"]
                    if len(kill)!=48 or len(sham)!=48:
                        raise ValueError("physical source 48d edit missing")
                    norm1=math.sqrt(sum(t*t for t in kill))
                    norm2=math.sqrt(sum(t*t for t in sham))
                    if abs(norm1-norm2)>1e-8 or abs(norm1-g["input_plan"]["dose"])>1e-8:
                        raise ValueError("original physical sham equal-dose gate violated")
                    measured=h["observed"]
                    true_digit=source["target"][positions[j]]
                    if measured["truth_digit"]!=true_digit or measured["frozen_clean_rival"]!=rivals[true_digit-1]:
                        raise ValueError("future true Sudoku digit or clean rival incorrect")
                    margins=measured["margins"]
                    if margins["clean"]!=margins["zero_self"]:
                        raise ValueError("native self-patch must equal clean")
                    D=margins["sham"]-margins["kill"]
                    if abs(D-measured["true_D"])>1e-10:
                        raise ValueError("native finite effect D differs from margin identity")
                    true=source_truth(D)
                    if h["true_effect_class"]!=true:
                        raise ValueError("native ternary truth effect class changed")
                    for k in POLICIES:
                        if abs(h["utility"][k]-utility(actions[k],true))>1e-10:
                            raise ValueError("original per-goal native offline reward differs")
                    if abs(h["answer_conditioned_tangent"]-effects[true_digit-1])>1e-10:
                        raise ValueError("posthoc candidate local tangent source value changed")
    if n_orig!=8192:
        raise ValueError("source lost original 8,192 selected model-goal slots")
    return n_orig,n_eligible,prior_thresholds


def independently_resample(rows,teacher,family,old_admitted,capability):
    if len(rows)!=SOURCE_SIZE or any(len(p["goals"])!=SOURCE_GOALS for p in rows):
        raise ValueError("original fresh 128 puzzle clusters and 8 slots missing")
    all_util=[]
    eligible=eligible_puzzles=action_count=neutral=0
    statuses={}
    native=[]
    for row in rows:
        totals={k:0. for k in POLICIES}
        local=0
        for g in row["goals"]:
            statuses[g["status"]]=statuses.get(g["status"],0)+1
            if g["status"]!="eligible":continue
            local+=1;eligible+=1
            d=float(g["observed"]["true_D"])
            native.append(d)
            t=source_truth(d)
            neutral+=int(t==0)
            action_count+=int(g["actions"]["calibrated_selected"]!=0)
            for k in POLICIES:totals[k]+=utility(g["actions"][k],t)
        eligible_puzzles+=int(local>0)
        all_util.append({k:totals[k]/SOURCE_GOALS for k in POLICIES})
    if eligible==0:
        return {"status":"UNDERPOWERED","registered_pass":False,
                "eligible_goals":0,"eligible_puzzles":0}
    means={k:sum(row[k] for row in all_util)/SOURCE_SIZE for k in POLICIES}
    differences=[{k:u["calibrated_selected"]-u[k] for k in CONTROLS} for u in all_util]
    gains={k:sum(x[k] for x in differences)/SOURCE_SIZE for k in CONTROLS}
    stable=int.from_bytes(hashlib.sha256(f"SMC27|20263717|{teacher}|{family}".encode()).digest()[:8],"big")
    rng=random.Random(stable)
    boot={k:[] for k in CONTROLS}
    for _ in range(BOOT):
        indices=[rng.randrange(SOURCE_SIZE) for __ in range(SOURCE_SIZE)]
        for key in CONTROLS:
            boot[key].append(sum(differences[i][key] for i in indices)/SOURCE_SIZE)
    for vals in boot.values():vals.sort()
    ci95={k:[boot[k][int(.025*BOOT)],boot[k][int(.975*BOOT)-1]] for k in CONTROLS}
    tail=.05/(2*24)
    ci_adjusted={k:[boot[k][int(tail*BOOT)],
                    boot[k][int((1-tail)*BOOT)-1]] for k in CONTROLS}
    covered=eligible>=200 and eligible_puzzles>=80
    rate=action_count/eligible
    rate_good=.05<=rate<=.90
    valid=bool(old_admitted and capability and covered and rate_good)
    return {
        "status":"CAPABILITY_MISMATCH" if not capability
                 else "CALIBRATION_INADMISSIBLE" if not old_admitted
                 else "UNDERPOWERED" if not covered
                 else "ACTION_RATE_OUT_OF_BAND" if not rate_good
                 else "MEASURED",
        "registered_pass":bool(valid and all(ci_adjusted[k][0]>.02 for k in CONTROLS)),
        "teacher":teacher,"family":family,"n_puzzles":SOURCE_SIZE,
        "n_original_blank_slots":SOURCE_SIZE*SOURCE_GOALS,
        "eligible_goals":eligible,"eligible_puzzles":eligible_puzzles,
        "n_ineligible":SOURCE_SIZE*SOURCE_GOALS-eligible,
        "eligibility_counts":statuses,
        "calibration_selected_threshold_admissible":bool(old_admitted),
        "old_teacher_capability_match":bool(capability),
        "calibrated_selected_test_action_rate":rate,
        "neutral_true_effect_goals":neutral,
        "teacher_mean_true_D":sum(native)/len(native),
        "mean_policy_utility_original_slots":means,
        "mean_primary_utility_gain":gains,
        "bootstrap_draws":BOOT,"bootstrap_puzzles_per_draw":SOURCE_SIZE,
        "bootstrap95_gain":ci95,"bootstrap99_79167_gain":ci_adjusted,
    }


def audit(old_bytes,bundle_bytes,cal_bytes,pred_bytes,stage_bytes,scored_bytes):
    original,cal,pred,scored=source_checks(
        old_bytes,bundle_bytes,cal_bytes,pred_bytes,stage_bytes,scored_bytes)
    n_orig,n_eligible,selected=verify_cases(original,cal,pred,scored)
    result=[]
    for family in FAMILIES:
        for teacher in TEACHERS:
            stored=scored["data"][family][f"{teacher}_SAE71"]
            calok=cal["teacher_calibration"][teacher]["threshold_selection"]["admissible"]
            capable=original["all_teacher_temperatures"][teacher]["capability_match"]
            calculated=independently_resample(stored["per_puzzle"],teacher,family,calok,capable)
            _close(calculated,stored["summary"],f"{family}.{teacher}")
            result.append({"family":family,"teacher":teacher,
                           "calibrated_threshold":selected[teacher],
                           "eligible_goals":calculated["eligible_goals"],
                           "action_rate":calculated.get("calibrated_selected_test_action_rate"),
                           "mean_gain":calculated.get("mean_primary_utility_gain"),
                           "adjusted_CI":calculated.get("bootstrap99_79167_gain"),
                           "registered_pass":calculated["registered_pass"],
                           "status":calculated["status"]})
    pair={t:all(next(r["registered_pass"] for r in result
                     if r["teacher"]==t and r["family"]==family)
                 for family in FAMILIES) for t in TEACHERS}
    cap=original["all_four_capability_matched"]
    old_admitted=cal["all_selected_thresholds_admissible"]
    coverage=all(r["status"]!="UNDERPOWERED" for r in result)
    passed=cap and old_admitted and coverage and sum(pair.values())>=3 and any(
        pair[x] for x in ("B811","B919"))
    status=("CAPABILITY_MISMATCH" if not cap else
            "CALIBRATION_INADMISSIBLE" if not old_admitted else
            "UNDERPOWERED" if not coverage else "PASS" if passed else "FAIL")
    if (scored["teacher_two_cohort_success"]!=pair or
        scored["registered_scientific_status"]!=status):
        raise ValueError("scorer changed frozen program-level gate")
    return {
        "schema":"SMC27/independent-standard-library-all-stage-audit-v1",
        "audit_status":"PASS",
        "original_SMC26_training_bundle_sha256":digest(bundle_bytes),
        "old_SMC26_calibration_sha256":digest(old_bytes),
        "SMC27_policy_calibration_sha256":digest(cal_bytes),
        "label_free_prediction_sha256":digest(pred_bytes),
        "canonical_stage_state_sha256":digest(stage_bytes),
        "native_scored_sha256":digest(scored_bytes),
        "registered_protocol_sha256":protocol_sha(),
        "new_unique_Sudoku_puzzles":256,
        "all_prespecified_teacher_goal_slots":n_orig,
        "preoutcome_admitted_goal_measurements":n_eligible,
        "selected_old_calibration_thresholds":selected,
        "teacher_two_cohort_success":pair,
        "scientific_verdict":status,
        "results":result,
        "native_teacher_forward_recomputed":False,
        "limitations":"Verifies archived original-model native response arithmetic and checkpoint SHA, not an independent second neural forward.",
    }


def main():
    parser=argparse.ArgumentParser()
    for name in ("old-calibration","bundle","calibration","prediction",
                 "source-state","scored","out"):
        parser.add_argument("--"+name,type=Path,required=True)
    a=parser.parse_args()
    r=audit(a.old_calibration.read_bytes(),a.bundle.read_bytes(),
            a.calibration.read_bytes(),a.prediction.read_bytes(),
            a.source_state.read_bytes(),a.scored.read_bytes())
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(r,sort_keys=True,indent=2)+"\n")
    print("SMC27_INDEPENDENT_AUDIT_PASS",r["scientific_verdict"],
          r["preoutcome_admitted_goal_measurements"],flush=True)


if __name__=="__main__":
    main()

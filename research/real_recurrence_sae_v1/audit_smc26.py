"""SMC-26 independent Python standard-library replay of native sign utility.

Reconstructs each registered puzzle from fixed seeds, target-free candidate
decisions, recorded finite true-digit margins, all 8,192 original goal slots,
5,000 puzzle-cluster bootstrap draws and the full capability/teacher gate.
Checks the binary archived physical source SHA, but cannot independently run
the original PyTorch frozen teacher forward.
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
from smc13_family import (
    new_full_solution,dig_unique,count_solutions,rectangle_count,fingerprint
)

TEACHERS=("B501","B701","B811","B919")
FAMILIES={"smc26_novel30":(202636030000,30),
          "smc26_novel34":(202636040000,34)}
POLICIES=("calibrated_median","uniform_median","shifted_median",
          "top_digit_effect","always_abstain")
CONTROLS=("uniform_median","always_abstain")
TEMPS=(.5,1.,2.,4.,8.)
PUZZLES=128
GOALS=8
BOOTS=5000
TOL=.15
COST=.10
SCHEMA="SMC26/independent-stdlib-source-and-scored-utility-replay-v1"
LOCKS={
    "original_sae_receipt":"7caafcb54e36f508b49a082892b689789a751f86b52d6d206e955e3782431087",
    "independent_teacher_receipt":"1fc69e15c24087f631806a89d55beaf519c2f6ab58ac7b8193cf8fdc36d240e3",
    "conditioned_mapping_receipt":"3b4637699d5af236af2d3b8aa5d04e024025b16e0d59395535892c104d472d07",
}


def digest(raw):return hashlib.sha256(raw).hexdigest()


def action(x):
    x=float(x)
    return 1 if x>TOL else -1 if x< -TOL else 0


def utility(a,t):
    if a not in (-1,0,1) or t not in (-1,0,1):raise ValueError("bad ternary decision")
    if t==0:base=1 if a==0 else -1
    else:base=0 if a==0 else 1 if a==t else -1
    return base-COST*int(a!=0)


def weighted_median(v,w):
    values=sorted((float(x),i,float(p)) for i,(x,p) in enumerate(zip(v,w)))
    if len(values)!=9 or any(p<0 for _,_,p in values) or abs(sum(p for _,_,p in values)-1)>1e-10:
        raise ValueError("invalid label effect distribution")
    acc=0.
    for x,i,p in values:
        acc+=p
        if acc>=.5-1e-15:return x
    raise ArithmeticError("median distribution missing mass")


def softmax(v,T):
    s=[float(x)/float(T) for x in v]
    m=max(s);e=[math.exp(x-m) for x in s]
    denom=sum(e)
    return [x/denom for x in e]


def protocol_sha():
    return digest((HERE/"SMC26_PROTOCOL.md").read_bytes())


def confirm_source(raw_cal,raw_bundle,raw_pred,raw_stage,raw_score):
    cal=json.loads(raw_cal);pred=json.loads(raw_pred);score=json.loads(raw_score)
    h=protocol_sha()
    if any(x["protocol_sha256"]!=h for x in (cal,pred,score)):
        raise ValueError("prospective source protocol hash differs")
    if (cal["schema"]!="SMC26/original-source-trained-four-teachers-v1" or
        pred["schema"]!="SMC26/label-free-native-sign-actions-v1" or
        score["schema"]!="SMC26/registered-native-offline-sign-utility-v1"):
        raise ValueError("different original source schema")
    if (cal["bundle_sha256"]!=digest(raw_bundle) or
        pred["new_teacher_bundle_sha256"]!=digest(raw_bundle) or
        score["new_teacher_bundle_sha256"]!=digest(raw_bundle)):
        raise ValueError("teacher/SAE training bundle was replaced")
    if pred["calibration_file_sha256"]!=digest(raw_cal) or score["calibration_sha256"]!=digest(raw_cal):
        raise ValueError("source calibration source replaced after freeze")
    if pred["source_stage1_binary_sha256"]!=digest(raw_stage) or score["source_stage1_sha256"]!=digest(raw_stage):
        raise ValueError("original FP32 archived source state file was replaced")
    if score["prediction_sha256"]!=digest(raw_pred):
        raise ValueError("frozen action policy prediction replaced before scoring")
    if not (cal["new_puzzles_generated"] is False and pred["contains_test_true_labels"] is False
            and pred["contains_test_outcomes"] is False and
            pred["contains_test_true_digit_gradient"] is False):
        raise ValueError("heldout label appears in earlier calibration or prediction")
    if (set(cal["all_teacher_temperatures"])!=set(TEACHERS) or
        set(pred["teacher_names"])!=set(TEACHERS) or
        set(pred["test_prediction_data"])!=set(FAMILIES) or
        set(score["data"])!=set(FAMILIES)):
        raise ValueError("four-teacher or test-family population drift")
    if (cal["source_snapshot"]!=pred["source_snapshot"] or
        pred["source_snapshot"]!=score["source_snapshot"]):
        raise ValueError("old checkpoint source SHA drift")
    for name,required in LOCKS.items():
        if cal["source_snapshot"][name]!=required:
            raise ValueError("original prior SAE source SHA mismatch")
    if sorted(cal["new_teacher_seed_roster"])!=[811,919]:
        raise ValueError("new teacher training seeds changed")
    if (cal["all_four_capability_matched"]!=all(
        item["capability_match"] for item in cal["all_teacher_temperatures"].values())):
        raise ValueError("capability selection admission changed")
    for teacher,meta in cal["all_teacher_temperatures"].items():
        correct=meta["calibration_correct"]
        if (meta["n_calibration_goals"]!=512 or not 0<=correct<=512 or
            abs(meta["calibration_goal_accuracy"]-correct/512.)>1e-14 or
            bool(.40<=correct/512.<=.60)!=bool(meta["capability_match"])):
            raise ValueError("calibration teacher competence gate changed")
        vals=meta["temperature_nll"]
        if (set(vals)!={str(t) for t in TEMPS} or
            float(min(TEMPS,key=lambda t:(vals[str(t)],t)))!=meta["temperature"]):
            raise ValueError("temperature-selected model differs from old calibration")
    if '"target":' in raw_pred.decode() or '"truth_digit":' in raw_pred.decode():
        raise ValueError("future correct digit leaked into frozen prediction JSON")
    if pred["frozen_policy_threshold"]!=TOL:
        raise ValueError("signed abstention decision threshold moved")
    if score["n_predeclared_goal_slots"]!=8192 or score["n_unique_new_sudoku_puzzles"]!=256:
        raise ValueError("future cohort sample size changed")
    if pred["test_manifest"]!=score["test_manifest"]:
        raise ValueError("source puzzle manifest changed after predictions")
    return cal,pred,score


# Independently enumerate every explicitly registered earlier fresh test cohort,
# including those whose exclusion was indirect in the original SMC-26 producer.
# This source-only check reads no model output and never selects a replacement.
PRIOR_SOURCE_COHORTS = (
    (202615030000,30,64), (202615040000,34,64),
    (202616030000,30,64), (202616040000,34,64),
    (202633030000,30,64), (202633040000,34,64),
    (202634030000,30,64), (202634040000,34,64),
    (202635030000,30,64), (202635040000,34,64),
)


def audit_all_historical_fresh_cohorts(current_x, current_y):
    past_x=set()
    past_y=set()
    for base,holes,size in PRIOR_SOURCE_COHORTS:
        for i in range(size):
            grid_seed=base+i
            target=new_full_solution(grid_seed)
            clues=dig_unique(target,grid_seed+1000000,holes)
            xt=tuple(clues);yt=tuple(target)
            if xt in current_x or yt in current_y:
                raise ValueError("SMC26 test solution/puzzle reused a prior SMC-15/16/23/24/25 source")
            if xt in past_x or yt in past_y:
                raise ValueError("historical prior source inventory repeated its own Sudoku grid")
            past_x.add(xt)
            past_y.add(yt)
    if len(past_y)!=640:
        raise ValueError("historical ten-cohort source inventory is incomplete")
    return len(past_y)


def check_cases(cal,pred,scored):
    manifest=pred["test_manifest"]
    if manifest["seed_bases"]!={n:list(pair) for n,pair in FAMILIES.items()} or manifest["n_per_family"]!=128:
        raise ValueError("test seeds changed")
    seen_x=set();seen_y=set()
    n_all=n_eligible=0
    for family,(base,holes) in FAMILIES.items():
        cases=[]
        for i in range(PUZZLES):
            sd=base+i
            target=new_full_solution(sd)
            puzzle=dig_unique(target,sd+1000000,holes)
            if (count_solutions(puzzle,2)!=1 or rectangle_count(target)<=0 or
                tuple(puzzle) in seen_x or tuple(target) in seen_y or
                puzzle.count(0)!=holes):
                raise ValueError("invalid or repeated new noncanonical Sudoku case")
            seen_x.add(tuple(puzzle));seen_y.add(tuple(target))
            cases.append({"seed":sd,"x":puzzle,"target":target,
                          "holes":holes,"rectangles":rectangle_count(target)})
        if fingerprint(cases)!=manifest["cohorts"][family]["roster_sha256"]:
            raise ValueError("original full-solution roster SHA256 differs")
        for teacher in TEACHERS:
            key=f"{teacher}_SAE71"
            original=pred["test_prediction_data"][family][key]
            assessed=scored["data"][family][key]["per_puzzle"]
            if len(original)!=PUZZLES or len(assessed)!=PUZZLES:
                raise ValueError("a teacher omitted source puzzles")
            temperature=cal["all_teacher_temperatures"][teacher]["temperature"]
            for pi,(a,b) in enumerate(zip(original,assessed)):
                case=cases[pi]
                if (a["puzzle_index"]!=pi or b["puzzle_index"]!=pi or
                    a["seed"]!=case["seed"] or b["seed"]!=case["seed"] or
                    a["visible_input"]!=case["x"]):
                    raise ValueError("test outcome scored a different source input")
                blanks=[j for j,c in enumerate(case["x"]) if c==0]
                positions=[blanks[(j*len(blanks))//8] for j in range(8)]
                if a["blank_positions"]!=positions or b["blank_positions"]!=positions:
                    raise ValueError("selected blank goals depend on nonvisible data")
                if len(a["goals"])!=GOALS or len(b["goals"])!=GOALS:
                    raise ValueError("unreported original source goal")
                if b["teacher_capability_matched"]!=cal["all_teacher_temperatures"][teacher]["capability_match"]:
                    raise ValueError("capability gating changed between stages")
                for j,(g,z) in enumerate(zip(a["goals"],b["goals"])):
                    n_all+=1
                    if (g["slot"]!=j or z["slot"]!=j or g["token"]!=positions[j] or
                        g["status"]!=z["status"] or g["feature"]!=z["feature"] or
                        g["input_plan"]!=z["input_plan"]):
                        raise ValueError("pre-outcome source feature, token or edit plan changed")
                    if g["status"]!="eligible":
                        if "observed" in z:
                            raise ValueError("ineligible goal entered native scoring")
                        continue
                    n_eligible+=1
                    if not 0<=g["feature"]<96 or len(g["classwise_effects"])!=9 or len(g["rival_for_digit"])!=9:
                        raise ValueError("bad frozen nine-class or SAE feature index")
                    logits=g["native_clean_logits"]
                    if len(logits)!=10 or any(not math.isfinite(v) for v in logits):
                        raise ValueError("invalid clean source full logits")
                    q=softmax(logits[1:10],temperature)
                    if any(abs(x-y)>1e-10 for x,y in zip(q,g["calibrated_digit_distribution"])):
                        raise ValueError("classwise calibrated predictor changed")
                    effects=g["classwise_effects"]
                    rivals=[max((k for k in range(10) if k!=y),key=lambda k:logits[k])
                            for y in range(1,10)]
                    if rivals!=g["rival_for_digit"]:
                        raise ValueError("original top rival candidate mask changed")
                    top=max(range(1,10),key=lambda y:logits[y])
                    if g["model_top_digit"]!=top:
                        raise ValueError("model's predicted Sudoku digit changed")
                    rotated=[q[(i-1)%9] for i in range(9)]
                    predvals={
                        "calibrated_median":weighted_median(effects,q),
                        "uniform_median":weighted_median(effects,[1/9]*9),
                        "shifted_median":weighted_median(effects,rotated),
                        "top_digit_effect":effects[top-1],
                    }
                    if any(abs(predvals[p]-g["predictions"][p])>1e-10 for p in predvals):
                        raise ValueError("post calibration candidate median changed")
                    predicted_actions={p:action(g["predictions"][p]) for p in predvals}
                    predicted_actions["always_abstain"]=0
                    if predicted_actions!=g["actions"] or z["actions"]!=g["actions"]:
                        raise ValueError("test-time action policy was changed")
                    dk=g["edit_kill"];sh=g["edit_sham"]
                    if len(dk)!=48 or len(sh)!=48:
                        raise ValueError("original physical edit width altered")
                    n1=math.sqrt(sum(float(v)*v for v in dk))
                    n2=math.sqrt(sum(float(v)*v for v in sh))
                    if abs(n1-n2)>1e-8 or abs(n1-g["input_plan"]["dose"])>1e-8:
                        raise ValueError("same physical edit dose mismatch")
                    observed=z["observed"]
                    truth=case["target"][positions[j]]
                    if (observed["truth_digit"]!=truth or
                        observed["frozen_clean_rival"]!=rivals[truth-1]):
                        raise ValueError("scored native correct digit/rival changed")
                    margins=observed["margins"]
                    if margins["clean"]!=margins["zero_self"]:
                        raise ValueError("source clean/zero self-patch mismatch")
                    d=float(margins["sham"]-margins["kill"])
                    if abs(d-observed["true_D"])>1e-10:
                        raise ValueError("native finite margin effect inconsistent")
                    decision=action(d)
                    if z["true_effect_class"]!=decision:
                        raise ValueError("scored intervention-sign class changed")
                    for p in POLICIES:
                        if abs(z["offline_policy_utility"][p]-utility(predicted_actions[p],decision))>1e-10:
                            raise ValueError("frozen offline policy gain arithmetic differs")
                    if abs(z["answer_conditioned_local_tangent"]-effects[truth-1])>1e-10:
                        raise ValueError("descriptive oracle tangent changed")
    historical_checked=audit_all_historical_fresh_cohorts(seen_x,seen_y)
    return n_all,n_eligible,historical_checked


def independent_summary(rows,teacher,family):
    if len(rows)!=PUZZLES:
        raise ValueError("128 independent puzzle clusters required")
    per=[]
    elig=elig_ids=action_count=neutral=positive=negative=clean=0
    freq={}
    sign_correct={k:0 for k in POLICIES}
    for row in rows:
        if len(row["goals"])!=GOALS:
            raise ValueError("eight task-free selected goals required")
        u={k:0. for k in POLICIES}
        local_count=0
        for g in row["goals"]:
            freq[g["status"]]=freq.get(g["status"],0)+1
            if g["status"]!="eligible":
                continue
            local_count+=1;elig+=1
            truth=g["true_effect_class"]
            neutral+=int(truth==0);positive+=int(truth==1);negative+=int(truth==-1)
            clean+=int(g["observed"]["clean_correct"])
            action_count+=int(g["actions"]["calibrated_median"]!=0)
            for key in POLICIES:
                u[key]+=utility(g["actions"][key],truth)
                sign_correct[key]+=int(g["actions"][key]==truth and truth!=0)
        elig_ids+=int(local_count>0)
        per.append({k:u[k]/GOALS for k in POLICIES})
    if elig==0:
        return {"eligible_goals":0,"eligible_puzzles":0,"registered_pass":False}
    means={k:sum(r[k] for r in per)/PUZZLES for k in POLICIES}
    gains={k:means["calibrated_median"]-means[k] for k in CONTROLS}
    diffs=[{k:r["calibrated_median"]-r[k] for k in CONTROLS} for r in per]
    stable=int.from_bytes(hashlib.sha256(
        f"SMC26|20263617|{teacher}|{family}".encode()).digest()[:8],"big")
    rng=random.Random(stable)
    boots={k:[] for k in CONTROLS}
    for _ in range(BOOTS):
        ids=[rng.randrange(PUZZLES) for _ in range(PUZZLES)]
        for k in CONTROLS:
            boots[k].append(sum(diffs[i][k] for i in ids)/PUZZLES)
    for k in CONTROLS:boots[k].sort()
    c95={k:[boots[k][int(.025*BOOTS)],boots[k][int(.975*BOOTS)-1]] for k in CONTROLS}
    c99={k:[boots[k][int(.0015625*BOOTS)],boots[k][int(.9984375*BOOTS)-1]] for k in CONTROLS}
    match=bool(rows[0]["teacher_capability_matched"])
    if any(row["teacher_capability_matched"]!=match for row in rows):
        raise ValueError("teacher calibration capability label changed")
    covered=elig>=200 and elig_ids>=80
    actrate=action_count/elig
    success=bool(covered and match and .10<=actrate<=.90 and
                 all(c99[k][0]>.02 for k in CONTROLS))
    return {
        "status":"capability_mismatch" if not match else "measured" if covered else "underpowered",
        "teacher":teacher,"family":family,
        "teacher_capability_matched":match,
        "n_puzzles":PUZZLES,"n_prespecified_goals":PUZZLES*GOALS,
        "eligible_goals":elig,"eligible_puzzles":elig_ids,
        "n_ineligible_goals":PUZZLES*GOALS-elig,"status_counts":freq,
        "calibrated_action_rate":actrate,
        "neutral_true_effect_goals":neutral,
        "positive_true_effect_goals":positive,
        "negative_true_effect_goals":negative,
        "teacher_clean_correct_eligible":clean,
        "policy_sign_correct_nonzero":sign_correct,
        "puzzle_mean_policy_utility":means,
        "puzzle_mean_gain_vs_primary_control":gains,
        "bootstrap_draws":BOOTS,
        "bootstrap_cluster_count":PUZZLES,
        "bootstrap95_gain":c95,
        "bootstrap99_6875_gain":c99,
        "registered_pass":success,
    }


def _close(a,b,path="root"):
    if isinstance(a,dict) and isinstance(b,dict):
        if set(a)!=set(b):
            raise ValueError("receipt keys changed at "+path+":"+str(set(a)^set(b)))
        for k in a:_close(a[k],b[k],path+"."+k)
    elif isinstance(a,list) and isinstance(b,list):
        if len(a)!=len(b):raise ValueError("receipt length changed at "+path)
        for i,(x,y) in enumerate(zip(a,b)):_close(x,y,path+f".{i}")
    elif isinstance(a,(int,float)) and isinstance(b,(int,float)):
        if abs(float(a)-float(b))>1e-8:
            raise ValueError("independent numerical replay differs at "+path+f":{a} vs {b}")
    elif a!=b:
        raise ValueError("registered source/decision identity changed at "+path)


def audit(raw_cal,raw_bundle,raw_pred,raw_stage,raw_score):
    cal,pred,score=confirm_source(raw_cal,raw_bundle,raw_pred,raw_stage,raw_score)
    n_all,n_admit,historical_count=check_cases(cal,pred,score)
    if n_all!=8192:
        raise ValueError("scored wrong number of prespecified slots")
    results=[]
    for family in FAMILIES:
        for teacher in TEACHERS:
            rows=score["data"][family][f"{teacher}_SAE71"]["per_puzzle"]
            computed=independent_summary(rows,teacher,family)
            published=score["data"][family][f"{teacher}_SAE71"]["summary"]
            _close(computed,published)
            results.append({"family":family,"teacher":teacher,
                            "eligible":computed["eligible_goals"],
                            "calibrated_action_rate":computed.get("calibrated_action_rate"),
                            "mean_gain":computed.get("puzzle_mean_gain_vs_primary_control"),
                            "corrected_CI":computed.get("bootstrap99_6875_gain"),
                            "registered_pass":computed["registered_pass"]})
    pair={t:all(next(x["registered_pass"] for x in results if
                     x["family"]==fam and x["teacher"]==t)
                for fam in FAMILIES) for t in TEACHERS}
    match=all(x["capability_match"] for x in cal["all_teacher_temperatures"].values())
    global_pass=match and sum(pair.values())>=3 and any(pair[t] for t in ("B811","B919"))
    status=("CAPABILITY_MISMATCH" if not match else
            "UNDERPOWERED" if any(score["data"][fam][f"{t}_SAE71"]["summary"]["status"]=="underpowered"
                                  for fam in FAMILIES for t in TEACHERS)
            else "PASS" if global_pass else "FAIL")
    if pair!=score["teacher_pair_success"] or status!=score["registered_scientific_status"]:
        raise ValueError("reported program-wide success was post-result changed")
    return {
        "schema":SCHEMA,"audit_status":"PASS",
        "source_protocol_sha256":protocol_sha(),
        "calibration_sha256":digest(raw_cal),"new_bundle_sha256":digest(raw_bundle),
        "prediction_sha256":digest(raw_pred),"binary_source_SHA256":digest(raw_stage),
        "scored_native_sha256":digest(raw_score),
        "n_new_puzzles":256,"n_original_goal_slots":n_all,
        "n_historical_fresh_puzzles_rechecked_independently":historical_count,
        "n_preoutcome_eligible_goal_measurements":n_admit,
        "n_independent_teacher_initializations":4,
        "capability_matched":match,
        "teacher_pair_success":pair,
        "original_scientific_status":status,
        "source_models_replayed":False,
        "results":results,
        "limitations":"The original native teacher forward is accepted via SHA-locked receipts and source-stage/SAE parity from scoring. This Python stdlib audit checks state-file bytes and arithmetic but cannot independently execute Torch checkpoints.",
    }


def main():
    p=argparse.ArgumentParser()
    for name in ("calibration","bundle","prediction","source-state","scored","out"):
        p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args()
    report=audit(a.calibration.read_bytes(),a.bundle.read_bytes(),
                 a.prediction.read_bytes(),a.source_state.read_bytes(),
                 a.scored.read_bytes())
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    print("SMC26_INDEPENDENT_RECEIPT_PASS",report["original_scientific_status"],
          report["n_preoutcome_eligible_goal_measurements"],flush=True)


if __name__=="__main__":
    main()

"""SMC-26 native sign-utility scorer after frozen target-free prediction archive.

Every original puzzle/token is retained. Ineligible edits have zero utility
under every action policy; eligible edits are scored only against original
frozen native teacher full-logit continuations. No policy is retrained here.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import math
import random
import sys
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from smc23_native_first import affine_physical
from smc24_native_score import check_archived_edit,native_score_case
from smc25_candidate_effects import candidate_predictions,weighted_median
from smc26_train import load_all_models,TEACHERS,NEW_TEACHERS,digest,protocol_sha
from smc26_predict import (FAMILIES,SIZE,GOALS,POLICIES,SCHEMA as PRED_SCHEMA,
                           THRESHOLD,policy,fresh_rosters)
from smc14_causal_transfer import goal_positions

SCHEMA="SMC26/registered-native-offline-sign-utility-v1"
RESAMPLES=5000
COST=0.10
CONTROLS=("uniform_median","always_abstain")
SECONDARY=("shifted_median","top_digit_effect")
MIN_ELIGIBLE=200
MIN_PUZZLES=80
BOOT_SEED=20263617


def target_class(effect):
    d=float(effect)
    if not math.isfinite(d):
        raise ValueError("nonfinite scored teacher-native effect")
    return 1 if d>THRESHOLD else -1 if d< -THRESHOLD else 0


def decision_utility(action,truth):
    """Exactly the precommitted ternary cost-aware offline reward."""
    if action not in (-1,0,1) or truth not in (-1,0,1):
        raise ValueError("policy action and truth are ternary")
    if truth==0:
        reward=1 if action==0 else -1
    else:
        reward=0 if action==0 else 1 if action==truth else -1
    return reward-COST*int(action!=0)


def case_summary(rows,teacher,family,draws=RESAMPLES):
    """Cluster all 128 puzzles, 8 blank slots each; ineligible slot gain=0."""
    if len(rows)!=SIZE:
        raise ValueError("original 128 puzzle clusters required")
    delta=[]
    all_utility=[]
    elig_count=0
    elig_puzzles=0
    n_nonzero_action=0
    neutral=0
    true_pos=true_neg=0
    counts={}
    model_correct=0
    signed_correct={p:0 for p in POLICIES}
    for row in rows:
        if len(row["goals"])!=GOALS:
            raise ValueError("eight preselected goals required per puzzle")
        u={p:0.0 for p in POLICIES}
        n_p=0
        for g in row["goals"]:
            status=g["status"]
            counts[status]=counts.get(status,0)+1
            if status!="eligible":
                continue
            elig_count+=1;n_p+=1
            d=float(g["observed"]["true_D"])
            truth=target_class(d)
            neutral+=int(truth==0)
            true_pos+=int(truth==1)
            true_neg+=int(truth==-1)
            model_correct+=int(g["observed"]["clean_correct"])
            action=g["actions"]
            if set(action)!=set(POLICIES):
                raise ValueError("frozen sign policies do not match preregistration")
            n_nonzero_action+=int(action["calibrated_median"]!=0)
            for p in POLICIES:
                u[p]+=decision_utility(int(action[p]),truth)
                signed_correct[p]+=int(int(action[p])==truth and truth!=0)
        elig_puzzles+=int(n_p>0)
        u={p:float(value/GOALS) for p,value in u.items()}
        all_utility.append(u)
        delta.append({c:u["calibrated_median"]-u[c] for c in CONTROLS})
    assert sum(counts.values())==SIZE*GOALS
    if elig_count==0:
        return {"status":"underpowered","eligible_goals":0,
                "eligible_puzzles":0,"registered_pass":False}
    means={p:sum(r[p] for r in all_utility)/SIZE for p in POLICIES}
    gains={c:sum(v[c] for v in delta)/SIZE for c in CONTROLS}
    rng=random.Random(
        int.from_bytes(hashlib.sha256(f"SMC26|{BOOT_SEED}|{teacher}|{family}".encode()).digest()[:8],"big")
    )
    distributions={c:[] for c in CONTROLS}
    for _ in range(draws):
        ids=[rng.randrange(SIZE) for _ in range(SIZE)]
        for c in CONTROLS:
            distributions[c].append(sum(delta[idx][c] for idx in ids)/SIZE)
    for v in distributions.values():v.sort()
    ci95={c:[distributions[c][int(.025*draws)],
              distributions[c][int(.975*draws)-1]]
          for c in CONTROLS}
    ci_family={c:[
        distributions[c][int(.0015625*draws)],
        distributions[c][int(.9984375*draws)-1]] for c in CONTROLS}
    capability_matched=bool(row["teacher_capability_matched"] for row in rows)
    # All rows carry the same teacher calibration boolean.
    if len({row["teacher_capability_matched"] for row in rows})!=1:
        raise ValueError("capability admission changed within source cohort")
    capability_matched=bool(rows[0]["teacher_capability_matched"])
    rate=n_nonzero_action/elig_count
    covered=(elig_count>=MIN_ELIGIBLE and elig_puzzles>=MIN_PUZZLES)
    positive=bool(covered and capability_matched and .10<=rate<=.90 and
                  all(ci_family[c][0]>.02 for c in CONTROLS))
    return {
        "status":"capability_mismatch" if not capability_matched
                 else "measured" if covered else "underpowered",
        "teacher":teacher,"family":family,"teacher_capability_matched":capability_matched,
        "n_puzzles":SIZE,"n_prespecified_goals":SIZE*GOALS,
        "eligible_goals":elig_count,"eligible_puzzles":elig_puzzles,
        "n_ineligible_goals":SIZE*GOALS-elig_count,"status_counts":counts,
        "calibrated_action_rate":rate,"neutral_true_effect_goals":neutral,
        "positive_true_effect_goals":true_pos,"negative_true_effect_goals":true_neg,
        "teacher_clean_correct_eligible":model_correct,
        "policy_sign_correct_nonzero":signed_correct,
        "puzzle_mean_policy_utility":means,
        "puzzle_mean_gain_vs_primary_control":gains,
        "bootstrap_draws":draws,"bootstrap_cluster_count":SIZE,
        "bootstrap95_gain":ci95,"bootstrap99_6875_gain":ci_family,
        "registered_pass":positive,
    }


def check_prediction(frozen,cal_raw,cal,bundle_raw):
    if frozen["schema"]!=PRED_SCHEMA or frozen["protocol_sha256"]!=protocol_sha():
        raise ValueError("prediction protocol changed")
    if frozen["calibration_file_sha256"]!=digest(cal_raw) or cal["bundle_sha256"]!=digest(bundle_raw):
        raise ValueError("original training/calibration bytes mismatch")
    if (frozen["source_snapshot"]!=cal["source_snapshot"] or
        frozen["new_teacher_bundle_sha256"]!=cal["bundle_sha256"] or
        frozen["contains_test_true_labels"] is not False or
        frozen["contains_test_outcomes"] is not False or
        frozen["contains_test_true_digit_gradient"] is not False):
        raise ValueError("target-free source identity or prediction constraints changed")
    if frozen["frozen_policy_threshold"]!=THRESHOLD:
        raise ValueError("registered action threshold changed")
    if set(frozen["teacher_names"])!=set(TEACHERS) or set(frozen["test_prediction_data"])!=set(FAMILIES):
        raise ValueError("different independent teachers or test families")
    if '"target":' in json.dumps(frozen) or '"truth_digit":' in json.dumps(frozen):
        raise ValueError("prediction artifact includes heldout target labels")


def score(original,independent,conditioned,bundle,calibration,prediction,
          source_state,allow_fetch):
    torch.set_num_threads(2)
    cal_raw=Path(calibration).read_bytes()
    bundle_raw=Path(bundle).read_bytes()
    pred_raw=Path(prediction).read_bytes()
    frozen=json.loads(pred_raw)
    models,comps,old,cal=load_all_models(
        original,independent,conditioned,bundle,calibration,allow_fetch)
    check_prediction(frozen,cal_raw,cal,bundle_raw)
    rosters,manifest=fresh_rosters(old)
    if manifest!=frozen["test_manifest"]:
        raise ValueError("novel source manifest changed after prediction freeze")
    stage_raw=Path(source_state).read_bytes()
    if (Path(source_state).name!=frozen["source_stage1_file"] or
        digest(stage_raw)!=frozen["source_stage1_binary_sha256"]):
        raise ValueError("canonical FP32 stage artifact SHA mismatch")
    source=torch.load(source_state,weights_only=True,map_location="cpu")
    results={}
    drift={}
    for family,cases in rosters.items():
        x=torch.tensor([r["x"] for r in cases],dtype=torch.long)
        positions=goal_positions(x)
        y=torch.tensor([r["target"] for r in cases],dtype=torch.long)
        drift[family]={}
        results[family]={}
        for teacher in TEACHERS:
            if teacher not in source[family] or tuple(source[family][teacher].shape)!=(SIZE,81,48):
                raise ValueError("missing input-only FP32 canonical states")
            stage=source[family][teacher]
            if stage.dtype!=torch.float32:
                raise ValueError("source FP32 numerical contract changed")
            with torch.no_grad():
                replay=models[teacher].reason_states(x)[1].detach().float()
            err=float((stage-replay).abs().max())
            if err>1e-4:
                raise ArithmeticError("source stage-one host replay exceeds 1e-4 preoutcome gate")
            drift[family][teacher]=err
            sae_name=f"{teacher}_SAE71"
            _,sae,mu,sig=comps[sae_name]
            cov,dec=affine_physical(sae,mu,sig)
            model64=copy.deepcopy(models[teacher]).double().eval()
            with torch.no_grad():
                clean=stage.double()
                for _ in range(3):
                    clean=model64.block(clean)
                clean=model64.head(clean).detach()
            frozen_rows=frozen["test_prediction_data"][family][sae_name]
            rows=[]
            for pi,p in enumerate(frozen_rows):
                if (pi!=p["puzzle_index"] or
                    p["seed"]!=cases[pi]["seed"] or
                    p["visible_input"]!=cases[pi]["x"] or
                    p["blank_positions"]!=positions[pi].tolist() or len(p["goals"])!=GOALS):
                    raise ValueError("original selected visible puzzle inputs changed")
                frozen_goals=[]
                for j,g in enumerate(p["goals"]):
                    token=int(positions[pi,j])
                    if g["slot"]!=j or g["token"]!=token:
                        raise ValueError("goal token selected outside original source blank mask")
                    result={"slot":j,"token":token,"feature":g["feature"],
                            "status":g["status"],"input_plan":g["input_plan"]}
                    if g["status"]=="eligible":
                        physical={"archived_h":stage[pi,token],
                                  "edit_kill":g["edit_kill"],"edit_sham":g["edit_sham"]}
                        kill,sham=check_archived_edit(physical,g["input_plan"],
                                                       sae,mu,sig,cov,dec)
                        source_clean=torch.tensor(g["native_clean_logits"],dtype=torch.float64)
                        if float((source_clean-clean[pi,token]).abs().max())>1e-4:
                            raise ValueError("frozen native clean logits drifted")
                        recomputed=candidate_predictions(source_clean,
                            torch.zeros(10,dtype=torch.float64),
                            cal["all_teacher_temperatures"][teacher]["temperature"])
                        if (recomputed["rival_for_digit"]!=g["rival_for_digit"] or
                            g["model_top_digit"]!=recomputed["model_top_digit"] or
                            max(abs(a-b) for a,b in zip(
                                g["calibrated_digit_distribution"],
                                recomputed["calibrated_digit_distribution"]))>1e-10):
                            raise ValueError("frozen candidate label probabilities drifted")
                        d=g["classwise_effects"];q=g["calibrated_digit_distribution"]
                        shift=[q[(i-1)%9] for i in range(9)]
                        weights={"calibrated_median":q,"uniform_median":[1/9]*9,
                                 "shifted_median":shift}
                        expectation={name:weighted_median(d,probs)
                                     for name,probs in weights.items()}
                        expectation["top_digit_effect"]=d[g["model_top_digit"]-1]
                        if any(abs(expectation[k]-g["predictions"][k])>1e-9 for k in expectation):
                            raise ValueError("frozen candidate effect aggregation changed")
                        actions={k:policy(g["predictions"][k]) for k in POLICIES if k!="always_abstain"}
                        actions["always_abstain"]=0
                        if g["actions"]!=actions:
                            raise ValueError("test predicted actions changed after source freeze")
                        truth=int(y[pi,token])
                        native=native_score_case(model64,stage[pi],clean[pi],
                                                truth,token,g["feature"],kill,sham)
                        if native["frozen_clean_rival"]!=g["rival_for_digit"][truth-1]:
                            raise ValueError("clean correct-digit rival differs from target-blind archive")
                        class_truth=target_class(native["true_D"])
                        result.update({
                            "actions":actions,"predictions":g["predictions"],
                            "observed":native,"true_effect_class":class_truth,
                            "offline_policy_utility":{
                                key:decision_utility(int(actions[key]),class_truth)
                                for key in POLICIES
                            },
                            "answer_conditioned_local_tangent":float(d[truth-1]),
                            "answer_conditioned_finite_taylor_residual":float(
                                native["true_D"]-d[truth-1])
                        })
                    frozen_goals.append(result)
                rows.append({"puzzle_index":pi,"seed":cases[pi]["seed"],
                             "blank_positions":positions[pi].tolist(),
                             "teacher_capability_matched":
                                 bool(cal["all_teacher_temperatures"][teacher]["capability_match"]),
                             "goals":frozen_goals})
            stats=case_summary(rows,teacher,family)
            results[family][sae_name]={"summary":stats,"per_puzzle":rows}
            print("SMC26_NATIVE",teacher,family,"eligible",stats["eligible_goals"],
                  "gain",stats.get("puzzle_mean_gain_vs_primary_control"),
                  "pass",stats["registered_pass"],flush=True)
    pair={teacher:all(results[f][f"{teacher}_SAE71"]["summary"]["registered_pass"]
                      for f in FAMILIES) for teacher in TEACHERS}
    match=bool(cal["all_four_capability_matched"])
    passed=match and sum(pair.values())>=3 and any(pair[t] for t in NEW_TEACHERS)
    status=("CAPABILITY_MISMATCH" if not match else
            "UNDERPOWERED" if any(results[f][f"{t}_SAE71"]["summary"]["status"]=="underpowered"
                                  for f in FAMILIES for t in TEACHERS)
            else "PASS" if passed else "FAIL")
    return {
        "schema":SCHEMA,"protocol_sha256":protocol_sha(),
        "calibration_sha256":digest(cal_raw),
        "new_teacher_bundle_sha256":digest(bundle_raw),
        "prediction_sha256":digest(pred_raw),
        "source_stage1_sha256":digest(stage_raw),
        "test_manifest":manifest,"source_snapshot":cal["source_snapshot"],
        "stage1_host_drift_max_abs":drift,
        "n_unique_new_sudoku_puzzles":2*SIZE,
        "n_predeclared_goal_slots":len(FAMILIES)*len(TEACHERS)*SIZE*GOALS,
        "calibration_capability_pass":match,"teacher_pair_success":pair,
        "registered_scientific_status":status,
        "data":results,
        "original_smc25_mae_gate_unchanged":"FAIL_0_OF_4",
    }


def main():
    p=argparse.ArgumentParser()
    for name in ("original-artifact","independent-artifact","conditioned-receipt",
                 "bundle","calibration","prediction","source-state","out"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--fetch-source",action="store_true")
    a=p.parse_args()
    d=score(a.original_artifact,a.independent_artifact,a.conditioned_receipt,
            a.bundle,a.calibration,a.prediction,a.source_state,a.fetch_source)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(d,indent=2,sort_keys=True)+"\n")
    print("SMC26_NATIVE_COMPLETE",d["registered_scientific_status"],
          d["n_predeclared_goal_slots"],str(a.out),flush=True)


if __name__=="__main__":
    main()

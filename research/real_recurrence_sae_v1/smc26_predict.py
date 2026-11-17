"""SMC-26: freeze input-only native sign decisions on 256 source-disjoint Sudoku.

No held-out correct digit is supplied to teacher, SAE, intervention planning,
classwise causal tangent, or predicted sign-policy action.
"""
from __future__ import annotations
import argparse
import copy
import json
import math
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
from smc26_train import load_all_models,TEACHERS,digest,protocol_sha

FAMILIES={"smc26_novel30":(202636030000,30),
          "smc26_novel34":(202636040000,34)}
SIZE=128
GOALS=8
THRESHOLD=0.15
POLICIES=("calibrated_median","uniform_median","shifted_median","top_digit_effect","always_abstain")
SCHEMA="SMC26/label-free-native-sign-actions-v1"


def policy(value):
    """Fixed symmetric offline action policy; zero means abstain."""
    x=float(value)
    if not math.isfinite(x):
        raise ValueError("nonfinite classwise native prediction")
    return 1 if x>THRESHOLD else -1 if x< -THRESHOLD else 0


def fresh_rosters(old):
    """Regenerate prior source inventories; never read an earlier test score."""
    from smc25_candidate_effects import generate_fresh as smc25_fresh
    old25,_=smc25_fresh(old)
    seen_x={tuple(z["x"]) for group in list(old.values())+list(old25.values()) for z in group}
    seen_y={tuple(z["target"]) for group in list(old.values())+list(old25.values()) for z in group}
    n_old=len(seen_y)
    fresh={}
    for name,(base,holes) in FAMILIES.items():
        cases=[]
        for i in range(SIZE):
            grid_seed=base+i
            answer=new_full_solution(grid_seed)
            clues=dig_unique(answer,grid_seed+1000000,holes)
            if (not valid_solution(answer) or rectangle_count(answer)<=0 or
                count_solutions(clues,2)!=1 or clues.count(0)!=holes or
                any(v and v!=answer[j] for j,v in enumerate(clues))):
                raise ValueError("fresh Sudoku input violates unique or noncanonical condition")
            if tuple(clues) in seen_x or tuple(answer) in seen_y:
                raise ValueError("SMC26 overlapped old source training/test inventory")
            seen_x.add(tuple(clues));seen_y.add(tuple(answer))
            cases.append({"seed":grid_seed,"x":clues,"target":answer,
                          "holes":holes,"rectangles":rectangle_count(answer)})
        fresh[name]=cases
    manifest={
        "schema":"SMC26/two-fresh-source-disjoint-cohorts-v1",
        "seed_bases":{name:[base,h] for name,(base,h) in FAMILIES.items()},
        "n_per_family":SIZE,"n_previous_full_solution_exclusions":n_old,
        "cohorts":{name:{
            "n":SIZE,
            "roster_sha256":fingerprint(rows),
            "input_sha256":digest(json.dumps([r["x"] for r in rows],
                    sort_keys=True,separators=(",",":")).encode()),
            "target_sha256":digest(json.dumps([r["target"] for r in rows],
                    sort_keys=True,separators=(",",":")).encode()),
            "min_rectangle":min(r["rectangles"] for r in rows),
        } for name,rows in fresh.items()}
    }
    return fresh,manifest


def predict(original,independent,conditioned,bundle,calibration,fetch_source,source_state):
    torch.set_num_threads(2)
    models,comps,old,cal=load_all_models(original,independent,conditioned,
                                         bundle,calibration,fetch_source)
    cohorts,manifest=fresh_rosters(old)
    original_calibration=Path(calibration).read_bytes()
    state_archive={}
    result={}
    for family,rows in cohorts.items():
        x=torch.tensor([r["x"] for r in rows],dtype=torch.long)
        positions=goal_positions(x)
        state_archive[family]={}
        out={}
        for teacher in TEACHERS:
            sae_name=f"{teacher}_SAE71"
            tid,sae,mu,sig=comps[sae_name]
            if tid!=teacher:
                raise ValueError("source SAE belongs to a different teacher")
            with torch.no_grad():
                stage=models[teacher].reason_states(x)[1].detach().float().cpu().contiguous()
            if tuple(stage.shape)!=(SIZE,81,48):
                raise ValueError("wrong FP32 stage-one native source")
            state_archive[family][teacher]=stage
            model64=copy.deepcopy(models[teacher]).double().eval()
            cov,dec=affine_physical(sae,mu,sig)
            temperature=cal["all_teacher_temperatures"][teacher]["temperature"]
            per=[]
            for pi,row in enumerate(rows):
                records=[]
                for j,token in enumerate(positions[pi].tolist()):
                    info,kill,sham=plan_target_native(
                        stage[pi,token],sae,mu,sig,cov,dec,
                        f"SMC26|{family}|{pi}|{sae_name}|{j}")
                    v={"slot":j,"token":token,"feature":int(info["feature"]),
                       "status":info["status"],"input_plan":info}
                    if info["status"]=="eligible":
                        clean,tangent=label_blind_tangent(model64,stage[pi],token,kill,sham)
                        candidates=candidate_predictions(clean,tangent,temperature)
                        frozen_actions={k:policy(candidates["predictions"][k])
                                        for k in POLICIES if k!="always_abstain"}
                        frozen_actions["always_abstain"]=0
                        v.update(candidates)
                        v.update({
                            "native_clean_logits":clean.tolist(),
                            "edit_kill":kill.tolist(),"edit_sham":sham.tolist(),
                            "actions":frozen_actions,
                        })
                    records.append(v)
                per.append({"puzzle_index":pi,"seed":row["seed"],
                            "visible_input":row["x"],
                            "blank_positions":positions[pi].tolist(),
                            "goals":records})
            out[sae_name]=per
            count=sum(g["status"]=="eligible" for row in per for g in row["goals"])
            print("SMC26_LABEL_FREE",family,teacher,"eligible",count,
                  "of",SIZE*GOALS,flush=True)
        result[family]=out
    source_state.parent.mkdir(parents=True,exist_ok=True)
    torch.save(state_archive,source_state)
    stage_sha=digest(source_state.read_bytes())
    return {
        "schema":SCHEMA,"protocol_sha256":protocol_sha(),
        "calibration_file_sha256":digest(original_calibration),
        "source_snapshot":cal["source_snapshot"],
        "new_teacher_bundle_sha256":cal["bundle_sha256"],
        "source_stage1_file":source_state.name,
        "source_stage1_binary_sha256":stage_sha,
        "contains_test_true_labels":False,
        "contains_test_outcomes":False,
        "contains_test_true_digit_gradient":False,
        "frozen_policy_threshold":THRESHOLD,
        "action_labels":{"1":"choose sham","-1":"choose kill","0":"abstain"},
        "teacher_names":list(TEACHERS),
        "one_sae_per_teacher":True,
        "test_manifest":manifest,
        "test_prediction_data":result,
        "precision":"source FP32 then true FP64 three-step recurrence at +/-0.08 direction",
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--original-artifact",type=Path,required=True)
    p.add_argument("--independent-artifact",type=Path,required=True)
    p.add_argument("--conditioned-receipt",type=Path,required=True)
    p.add_argument("--bundle",type=Path,required=True)
    p.add_argument("--calibration",type=Path,required=True)
    p.add_argument("--fetch-source",action="store_true")
    p.add_argument("--source-state",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    value=predict(a.original_artifact,a.independent_artifact,
                  a.conditioned_receipt,a.bundle,a.calibration,
                  a.fetch_source,a.source_state)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(value,sort_keys=True,indent=2)+"\n")
    print("SMC26_INPUT_ONLY_PREDICTION_COMPLETE",value["source_stage1_binary_sha256"],flush=True)


if __name__=="__main__":
    main()

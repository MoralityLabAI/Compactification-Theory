"""SMC-25 source-frozen calibration and label-blind all-digit causal predictions.

The calibration output is written before fresh cohorts are generated.
Prediction code only passes visible Sudoku inputs to model/SAE computations
and archives all nine possible digit-margin effects without actual answers.
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
from smc13_family import (new_full_solution,dig_unique,valid_solution,
                          rectangle_count,count_solutions,fingerprint)
from smc14_causal_transfer import goal_positions
from smc15_tangent_conditioning import load_and_lock_inputs
from smc23_native_first import affine_physical,plan_target_native
from smc24_gradient_predict import (
    source_snapshot, fresh_rosters as old_smc24_rosters,
    sha, seed, B_COMPONENTS, TEACHERS, GOALS, PUZZLES
)

CAL_SCHEMA="SMC-25/teacher-temperature-calibration-before-test-20261117-v1"
PRED_SCHEMA="SMC-25/label-blind-all-class-native-causal-prediction-20261117-v1"
FAMILIES={"smc25_novel30":(202635030000,30),
          "smc25_novel34":(202635040000,34)}
TEMPERATURES=(0.5,1.0,2.0,4.0,8.0)
SMALL=0.08


def protocol_digest():
    return sha((HERE/"SMC25_PROTOCOL.md").read_bytes())


def weighted_median(values,weights):
    """Left weighted median of effects, stable numeric/class tie-break."""
    pairs=sorted((float(v),int(i),float(w)) for i,(v,w) in
                 enumerate(zip(values,weights)))
    if len(pairs)!=9 or not all(math.isfinite(x) for a,b,c in pairs for x in (a,c)):
        raise ValueError("exactly nine finite class effects and probabilities required")
    if any(w<0 for _,_,w in pairs) or abs(sum(w for _,_,w in pairs)-1)>1e-10:
        raise ValueError("candidate weights must be probabilities")
    cumulative=0.
    for v,i,w in pairs:
        cumulative+=w
        if cumulative>=0.5-1e-15:
            return v
    raise ArithmeticError("distribution weighted median failed")


def class_effects(clean_logits,logit_tangent):
    """Nine candidates, with CLEAN highest competing digit including class zero."""
    clean=torch.as_tensor(clean_logits,dtype=torch.float64)
    t=torch.as_tensor(logit_tangent,dtype=torch.float64)
    if clean.shape!=(10,) or t.shape!=(10,) or not bool(torch.isfinite(clean).all()) or not bool(torch.isfinite(t).all()):
        raise ValueError("source scorer requires ten finite native logits")
    values=[];rivals=[]
    for digit in range(1,10):
        competitor=clean.clone()
        competitor[digit]=-float("inf")
        rival=int(competitor.argmax())
        rivals.append(rival)
        values.append(float(t[digit]-t[rival]))
    return values,rivals


def candidate_predictions(clean,raw_tangent,temperature):
    """All candidates and predictors share the same two native tangent forwards."""
    clean=torch.as_tensor(clean,dtype=torch.float64)
    raw=torch.as_tensor(raw_tangent,dtype=torch.float64)
    values,rivals=class_effects(clean,raw)
    if temperature<=0 or not math.isfinite(temperature):
        raise ValueError("frozen calibration temperature positive")
    q=torch.softmax(clean[1:10]/temperature,dim=-1).tolist()
    uniform=[1./9.]*9
    shifted=[q[(i-1)%9] for i in range(9)]
    model_top=int(torch.argmax(clean[1:10]))+1
    pred={
        "calibrated_median":weighted_median(values,q),
        "uniform_median":weighted_median(values,uniform),
        "top_digit_effect":values[model_top-1],
        "shifted_median":weighted_median(values,shifted),
        "calibrated_mean":sum(v*p for v,p in zip(values,q)),
        "uniform_mean":sum(values)/9,
        "zero_effect":0.,
    }
    return {"classwise_effects":values,"rival_for_digit":rivals,
            "calibrated_digit_distribution":q,"calibration_temperature":float(temperature),
            "model_top_digit":model_top,
            "predictions":pred}


def generate_fresh(old):
    """Exclusion cover includes all SMC13/15/16/23/24 source seeds."""
    prior24,_=old_smc24_rosters(old)
    used_x={tuple(r["x"]) for group in list(old.values())+list(prior24.values())
            for r in group}
    used_y={tuple(r["target"]) for group in list(old.values())+list(prior24.values())
            for r in group}
    old_count=len(used_y)
    result={}
    for family,(base,holes) in FAMILIES.items():
        cases=[]
        for index in range(PUZZLES):
            full_seed=base+index
            y=new_full_solution(full_seed)
            x=dig_unique(y,full_seed+1000000,holes)
            if (not valid_solution(y) or rectangle_count(y)<=0 or
                x.count(0)!=holes or count_solutions(x,2)!=1):
                raise ValueError("fresh SMC25 Sudoku source invalid")
            if tuple(x) in used_x or tuple(y) in used_y:
                raise ValueError("SMC25 accidentally reused earlier source")
            used_x.add(tuple(x));used_y.add(tuple(y))
            cases.append({"seed":full_seed,"x":x,"target":y,"holes":holes,
                          "rectangles":rectangle_count(y)})
        result[family]=cases
    manifest={
        "schema":"SMC25/source-disjoint-fresh-Sudoku-v1",
        "seed_bases":{f:list(v) for f,v in FAMILIES.items()},
        "n_previous_excluded":old_count,"n_per_family":PUZZLES,
        "cohorts":{
            f:{
                "n":PUZZLES,"roster_sha256":fingerprint(rows),
                "input_sha256":sha(json.dumps([r["x"] for r in rows],
                                sort_keys=True,separators=(",",":")).encode()),
                "target_sha256":sha(json.dumps([r["target"] for r in rows],
                                sort_keys=True,separators=(",",":")).encode()),
                "min_intercalates":min(r["rectangles"] for r in rows)
            } for f,rows in result.items()
        }}
    return result,manifest


def load_source(original,independent,conditioned,allow_fetch):
    models,components,rosters,locked=load_and_lock_inputs(
        original,independent,conditioned,allow_fetch=allow_fetch)
    snapshot=source_snapshot(models,components,rosters,locked)
    return models,components,rosters,snapshot


def clean_native(model64,stage):
    with torch.no_grad():
        state=stage.double()
        for _ in range(3):
            state=model64.block(state)
        logits=model64.head(state)
    if logits.shape[-1]!=10 or not bool(torch.isfinite(logits).all()):
        raise ArithmeticError("ten native full-logit outputs required")
    return logits


def calibrate(original,independent,conditioned,allow_fetch):
    torch.set_num_threads(2)
    models,components,rosters,snapshot=load_source(
        original,independent,conditioned,allow_fetch)
    cases=rosters["feature_calibration"]
    x=torch.tensor([r["x"] for r in cases],dtype=torch.long)
    y=torch.tensor([r["target"] for r in cases],dtype=torch.long)
    pos=goal_positions(x)
    temps={}
    for tid in TEACHERS:
        with torch.no_grad():
            stage=models[tid].reason_states(x)[1].detach()
        teacher=copy.deepcopy(models[tid]).double().eval()
        clean=clean_native(teacher,stage)
        labels=y.gather(1,pos).long()-1
        choices=clean.gather(1,pos.unsqueeze(-1).expand(-1,-1,10))[:,:,1:10]
        nlls={}
        for T in TEMPERATURES:
            logp=torch.log_softmax(choices/T,dim=-1)
            loss=-logp.gather(2,labels.unsqueeze(-1)).mean()
            nlls[str(T)]=float(loss)
        selected=min(TEMPERATURES,key=lambda t:(nlls[str(t)],t))
        temps[tid]={
            "temperature":float(selected),
            "temperature_candidate_nll":nlls,
            "calibration_goals":len(cases)*GOALS,
            "calibration_native_correct":int((choices.argmax(-1)==labels).sum()),
        }
        print("SMC25_CALIBRATION",tid,"T",selected,"NLL",nlls,flush=True)
    return {"schema":CAL_SCHEMA,"protocol_sha256":protocol_digest(),
            "source_snapshot":snapshot,
            "old_calibration_roster_sha256":fingerprint(cases),
            "old_calibration_targets_only":True,
            "fresh_source_generated":False,
            "teacher_temperatures":temps,
            "precision":"FP32 source stage and FP64 native 3-step clean",
            "torch_version":torch.__version__}


def label_blind_tangent(model,stage,token,kill,sham):
    """Full native 10-logit directional finite tangent, no target digit used."""
    v=(kill-sham).double()
    dose=float(v.norm())
    if dose<=1e-10:
        raise ValueError("zero paired physical difference")
    u=v/dose
    h=stage.unsqueeze(0).double().repeat(3,1,1)
    h[1,int(token),:]+=SMALL*u
    h[2,int(token),:]-=SMALL*u
    with torch.no_grad():
        for _ in range(3):
            h=model.block(h)
        output=model.head(h)[:,int(token),:].detach()
    if output.shape!=(3,10) or not bool(torch.isfinite(output).all()):
        raise ArithmeticError("native directional finite forward invalid")
    tangent=(output[1]-output[2])*(dose/(2*SMALL))
    return output[0],tangent


def check_frozen_calibration(cal,snapshot):
    if cal["schema"]!=CAL_SCHEMA or cal["protocol_sha256"]!=protocol_digest():
        raise ValueError("calibration earlier protocol mismatch")
    if cal["source_snapshot"]!=snapshot or cal["fresh_source_generated"] is not False:
        raise ValueError("pretest temperature source snapshot changed")
    if set(cal["teacher_temperatures"])!=set(TEACHERS):
        raise ValueError("temperature fit teacher list incomplete")
    for rec in cal["teacher_temperatures"].values():
        if rec["calibration_goals"]!=512 or rec["temperature"] not in TEMPERATURES:
            raise ValueError("temperature outside source calibration grid")


def predict(original,independent,conditioned,calibration,allow_fetch):
    torch.set_num_threads(2)
    raw=Path(calibration).read_bytes()
    cal=json.loads(raw)
    models,components,old,snapshot=load_source(
        original,independent,conditioned,allow_fetch)
    check_frozen_calibration(cal,snapshot)
    rosters,manifest=generate_fresh(old)
    result={}
    archived={}
    for family,cases in rosters.items():
        x=torch.tensor([r["x"] for r in cases],dtype=torch.long)
        positions=goal_positions(x)
        stage={}
        teacher64={}
        for tid in TEACHERS:
            with torch.no_grad():
                stage[tid]=models[tid].reason_states(x)[1].detach()
            teacher64[tid]=copy.deepcopy(models[tid]).double().eval()
        archived[family]={
            tid:{"shape":list(stage[tid].shape),
                 "tensor_sha256":sha(stage[tid].contiguous().numpy().tobytes()),
                 "tensor":stage[tid].tolist()} for tid in TEACHERS}
        data={}
        for component in B_COMPONENTS:
            tid,sae,mean,scale=components[component]
            cov,dec=affine_physical(sae,mean,scale)
            temperature=cal["teacher_temperatures"][tid]["temperature"]
            records=[]
            for pi in range(PUZZLES):
                goals=[]
                for j,tok in enumerate(positions[pi].tolist()):
                    info,kill,sham=plan_target_native(
                        stage[tid][pi,tok],sae,mean,scale,cov,dec,
                        f"SMC25|{family}|{pi}|{component}|{j}")
                    row={"goal_index":j,"token":tok,
                         "feature":info["feature"],"status":info["status"],
                         "input_plan":info}
                    if info["status"]=="eligible":
                        clean,tangent=label_blind_tangent(
                            teacher64[tid],stage[tid][pi],tok,kill,sham)
                        candidates=candidate_predictions(clean,tangent,temperature)
                        row.update(candidates)
                        row["edit_kill"]=kill.tolist()
                        row["edit_sham"]=sham.tolist()
                        row["native_clean_logits"]=clean.tolist()
                    goals.append(row)
                records.append({
                    "puzzle_index":pi,"seed":cases[pi]["seed"],
                    "visible_input":cases[pi]["x"],
                    "positions":positions[pi].tolist(),"goals":goals})
            data[component]=records
            admitted=sum(g["status"]=="eligible" for r in records for g in r["goals"])
            print("SMC25_PREDICTED",family,component,admitted,flush=True)
        result[family]=data
    return {
        "schema":PRED_SCHEMA,"protocol_sha256":protocol_digest(),
        "calibration_sha256":sha(raw),"source_snapshot":snapshot,
        "test_manifest":manifest,"archived_source_stage1":archived,
        "test_prediction_data":result,
        "contains_test_labels":False,"uses_test_true_digit_gradient":False,
        "finite_tangent_radius":SMALL,"rival_source":"clean full ten logits 0..9",
        "calibration_grid":TEMPERATURES,
        "limitations":"Two-sided numeric local native logit tangent at 0.08 physical dose, not exact JVP or trained curved SAE."
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--stage",choices=("calibration","prediction"),required=True)
    parser.add_argument("--original-artifact",type=Path,required=True)
    parser.add_argument("--independent-artifact",type=Path,required=True)
    parser.add_argument("--conditioned-receipt",type=Path,required=True)
    parser.add_argument("--calibration",type=Path)
    parser.add_argument("--fetch-source",action="store_true")
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    if args.stage=="calibration":
        value=calibrate(args.original_artifact,args.independent_artifact,
                        args.conditioned_receipt,args.fetch_source)
    else:
        if args.calibration is None:
            parser.error("prediction requires frozen earlier calibration")
        value=predict(args.original_artifact,args.independent_artifact,
                      args.conditioned_receipt,args.calibration,args.fetch_source)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(value,sort_keys=True,indent=2)+"\n")
    print("SMC25_STAGE_COMPLETE",args.stage,str(args.out),flush=True)


if __name__=="__main__":
    main()

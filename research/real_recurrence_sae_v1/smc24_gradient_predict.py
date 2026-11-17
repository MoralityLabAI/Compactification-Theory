"""SMC-24 calibration-frozen, test-label-free gradient prediction.

The protocol was committed BEFORE this source and BEFORE any SMC-24 data.
Stages: calibration (64 OLD puzzle targets allowed), prediction (NEW INPUTS
ONLY; no target digits or task-margin outcomes appear in its output).
A separate native scorer uses the frozen prediction artifact later.
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from smc13_family import (new_full_solution, dig_unique, valid_solution,
                          rectangle_count, count_solutions, fingerprint)
from smc14_causal_transfer import (goal_positions, ORIGINAL_SOURCE_RECEIPT_SHA256,
                                  NEW_TEACHER_RECEIPT_SHA256, CONDITIONED_JSON_SHA256)
from smc15_tangent_conditioning import load_and_lock_inputs
from smc23_native_first import (generate_fresh as smc23_rosters, affine_physical,
                                plan_target_native)

CAL_SCHEMA = "SMC-24/labelled-calibration-gradient-ridge-20261117-v1"
PRED_SCHEMA = "SMC-24/label-free-archived-native-predictions-20261117-v1"
FAMILIES = {"smc24_novel30": (202634030000, 30),
            "smc24_novel34": (202634040000, 34)}
B_COMPONENTS = ("newB_501_SAE_71", "newB_501_SAE_89",
                "newB_701_SAE_71", "newB_701_SAE_89")
TEACHERS = ("B501", "B701")
RIDGE = 32.
GOALS = 8
PUZZLES = 64
BOOT = 2000
MIN_GOALS = 60
MIN_PUZZLES = 30


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def proto_hash():
    return sha((HERE / "SMC24_PROTOCOL.md").read_bytes())


def seed(*items):
    raw = "SMC24|" + "|".join(map(str, items))
    return int.from_bytes(hashlib.sha256(raw.encode()).digest()[:8], "big") & ((1 << 63)-1)


def source_snapshot(models, components, old, frozen):
    if set(TEACHERS) - set(models) or set(B_COMPONENTS) - set(components):
        raise ValueError("missing preregistered independent teacher or SAE")
    if len(frozen) != 4 or not all(
        row["hashes"]["matched_to_original_smc13c_receipt"]
        for row in frozen.values()
    ):
        raise ValueError("source conditioned assignment verification failed")
    return {
        "original_sae_receipt": ORIGINAL_SOURCE_RECEIPT_SHA256,
        "independent_teacher_receipt": NEW_TEACHER_RECEIPT_SHA256,
        "conditioned_mapping_receipt": CONDITIONED_JSON_SHA256,
        "matched_mapping_sha256": {
            str(i): row["hashes"]["matched_assignment_sha256"]
            for i, row in frozen.items()
        },
        "clue_shuffled_mapping_sha256": {
            str(i): row["hashes"]["clue_shuffled_assignment_sha256"]
            for i, row in frozen.items()
        },
        "calibration_roster_sha256": fingerprint(old["feature_calibration"]),
    }


def selected_goal_gradient(teacher64, stage, positions, targets):
    """Independent 8-copy FP64 VJP, one gradient per actual goal token.

    Targets are accessible ONLY from the existing 64 calibration puzzles.
    Every copy has the entire 81-token native recurrence and its own rival.
    """
    if stage.shape != (81, 48) or positions.shape != (GOALS,) or targets.shape != (GOALS,):
        raise ValueError("calibration VJP requires 8 complete 48D goal states")
    h = stage.detach().double().unsqueeze(0).repeat(GOALS, 1, 1).requires_grad_(True)
    with torch.enable_grad():
        state = h
        for _ in range(3):
            state = teacher64.block(state)
        outputs = teacher64.head(state)
        j = torch.arange(GOALS)
        scores = outputs[j, positions.long(), :]
        rival = scores.detach().clone()
        rival[j, targets.long()] = -float("inf")
        rival_ids = rival.argmax(-1)
        margin = scores[j, targets.long()] - scores[j, rival_ids]
        full_gradient = torch.autograd.grad(margin.sum(), h)[0]
        grads = full_gradient[j, positions.long(), :].detach().clone()
    if grads.shape != (GOALS, 48) or not bool(torch.isfinite(grads).all()):
        raise ArithmeticError("nonfinite calibration native-gradient readout")
    return grads, scores.detach().argmax(-1).eq(targets).tolist()


def fit_ridge(H, G, *, seed_id):
    """One prespecified 49x48 ridge fit, and slot-shuffled equal-budget control."""
    H = torch.as_tensor(H, dtype=torch.float64)
    G = torch.as_tensor(G, dtype=torch.float64)
    if H.ndim != 3 or G.shape != H.shape or H.shape[1:] != (GOALS, 48):
        raise ValueError("expected puzzle x eight-goal x 48d paired arrays")
    if len(H) != PUZZLES or not bool(torch.isfinite(H).all()) or not bool(torch.isfinite(G).all()):
        raise ValueError("fit restricted to 64 finite source calibration puzzles")
    m = H.reshape(-1, 48)
    y = G.reshape(-1, 48)
    mean = m.mean(dim=0)
    std = m.std(dim=0, unbiased=False).clamp_min(1e-4)
    X = torch.cat((torch.ones((len(m), 1),dtype=torch.float64),
                   (m-mean)/std),dim=-1)
    reg = RIDGE * torch.eye(49,dtype=torch.float64)
    reg[0,0] = 0.
    gram = X.T@X+reg
    beta = torch.linalg.solve(gram,X.T@y)
    gen = torch.Generator().manual_seed(seed_id)
    shuffled = G.clone()
    for goal in range(GOALS):
        permutation = torch.randperm(PUZZLES, generator=gen)
        shuffled[:, goal, :] = G[permutation, goal, :]
    y_null = shuffled.reshape(-1, 48)
    beta_null = torch.linalg.solve(gram,X.T@y_null)
    gradient_mean = y.mean(dim=0)
    if not all(bool(torch.isfinite(item).all()) for item in (beta,beta_null,gradient_mean,std)):
        raise ArithmeticError("nonfinite pre-test calibration coefficients")
    return {
        "features":49,"targets":48,"lambda":RIDGE,
        "n_calibration_puzzles":PUZZLES,"n_calibration_goals":PUZZLES*GOALS,
        "mean":mean.tolist(),"std":std.tolist(),
        "ridge_beta":beta.tolist(),"slot_shuffled_ridge_beta":beta_null.tolist(),
        "gradient_mean":gradient_mean.tolist(),
        "calibration_gradient_mean_norm":float(G.norm(dim=-1).mean()),
        "slot_shuffled_control":True,
        "calibration_X_sha256":sha(m.contiguous().numpy().tobytes()),
        "calibration_g_sha256":sha(y.contiguous().numpy().tobytes()),
    }


def predict_gradient(fit, hidden):
    """Return ridge, slot-shuffled ridge, mean and zero gradient proposals."""
    h = torch.as_tensor(hidden,dtype=torch.float64)
    if h.shape != (48,):
        raise ValueError("wrong physical input size")
    mu = torch.tensor(fit["mean"],dtype=torch.float64)
    std = torch.tensor(fit["std"],dtype=torch.float64)
    design = torch.cat((torch.ones(1,dtype=torch.float64),(h-mu)/std))
    beta = torch.tensor(fit["ridge_beta"],dtype=torch.float64)
    null = torch.tensor(fit["slot_shuffled_ridge_beta"],dtype=torch.float64)
    avg = torch.tensor(fit["gradient_mean"],dtype=torch.float64)
    ans = {"ridge":design@beta,"shuffled":design@null,
           "mean":avg,"zero":torch.zeros(48,dtype=torch.float64)}
    if not all(bool(torch.isfinite(v).all()) for v in ans.values()):
        raise ArithmeticError("nonfinite frozen test prediction")
    return ans


def fresh_rosters(old):
    """New cases exclude every historical SMC-13, 15, 16 and 23 source."""
    old23, _ = smc23_rosters(old)
    used_targets = {tuple(row["target"]) for group in
                    (list(old.values()) + list(old23.values())) for row in group}
    used_inputs = {tuple(row["x"]) for group in
                   (list(old.values()) + list(old23.values())) for row in group}
    # SMC-23's generator excludes SMC-15 and SMC-16 already; still
    # explicitly regenerate those two fixed seed families for source audit.
    from smc15_tangent_conditioning import fresh_rosters as older15
    fifteen, _ = older15(old)
    from smc23_native_first import OLDER_SMC16
    for group in fifteen.values():
        used_targets.update(tuple(r["target"]) for r in group)
        used_inputs.update(tuple(r["x"]) for r in group)
    for base, holes in OLDER_SMC16.values():
        for i in range(PUZZLES):
            solution = new_full_solution(base+i)
            inputs = dig_unique(solution,base+i+1000000,holes)
            used_targets.add(tuple(solution))
            used_inputs.add(tuple(inputs))
    excluded_count = len(used_targets)
    output = {}
    for name,(base,holes) in FAMILIES.items():
        rows = []
        for i in range(PUZZLES):
            grid_seed = base+i
            target = new_full_solution(grid_seed)
            puzzle = dig_unique(target,grid_seed+1000000,holes)
            if not valid_solution(target) or rectangle_count(target)<=0:
                raise ValueError("source target lacks noncanonical Sudoku certificate")
            if count_solutions(puzzle,2)!=1 or puzzle.count(0)!=holes:
                raise ValueError("source test puzzle underdug or nonunique")
            if tuple(target) in used_targets or tuple(puzzle) in used_inputs:
                raise ValueError("SMC24 overlaps earlier source or current test cohort")
            used_targets.add(tuple(target))
            used_inputs.add(tuple(puzzle))
            rows.append({"seed":grid_seed,"x":puzzle,"target":target,
                         "holes":holes,"rectangles":rectangle_count(target)})
        output[name] = rows
    manifest = {
        "schema":"SMC24/fresh-noncanonical-unique-disjoint-v1",
        "seed_bases":FAMILIES,"n_previous_excluded":excluded_count,
        "n_per_family":PUZZLES,
        "cohorts":{
            name:{
                "roster_sha256":fingerprint(rows),
                "input_sha256":sha(json.dumps([r["x"] for r in rows],separators=(",",":"),sort_keys=True).encode()),
                "target_sha256":sha(json.dumps([r["target"] for r in rows],separators=(",",":"),sort_keys=True).encode()),
                "min_intercalates":min(r["rectangles"] for r in rows),
                "n":len(rows),
            } for name,rows in output.items()
        }}
    return output,manifest


def verify_calibration_receipt(frozen, snapshot):
    if frozen["schema"] != CAL_SCHEMA or frozen["protocol_sha256"] != proto_hash():
        raise ValueError("calibration-before-test protocol hash mismatch")
    if frozen["source_snapshot"] != snapshot:
        raise ValueError("frozen teacher or SAE original source fingerprints differ")
    if set(frozen["fits"]) != set(TEACHERS):
        raise ValueError("wrong independent teacher fit population")
    for teacher,fit in frozen["fits"].items():
        if fit["lambda"] != RIDGE or fit["n_calibration_goals"] != 512:
            raise ValueError("calibration fit hyperparameter or example count drift")
        for name,shape in (("ridge_beta",(49,48)),("slot_shuffled_ridge_beta",(49,48)),
                           ("mean",(48,)),("std",(48,)),("gradient_mean",(48,))):
            if tuple(torch.tensor(fit[name]).shape) != shape:
                raise ValueError("calibration coefficient geometry changed")


def calibrate(original,independent,conditioned,fetch_source):
    torch.set_num_threads(2)
    models, components, old, frozen = load_and_lock_inputs(
        original,independent,conditioned,allow_fetch=fetch_source)
    snap = source_snapshot(models,components,old,frozen)
    rows = old["feature_calibration"]
    x = torch.tensor([r["x"] for r in rows],dtype=torch.long)
    y = torch.tensor([r["target"] for r in rows],dtype=torch.long)
    positions = goal_positions(x)
    fits = {}
    for teacher in TEACHERS:
        net=models[teacher].eval()
        native=copy.deepcopy(net).double().eval().requires_grad_(False)
        with torch.no_grad():
            stage=net.reason_states(x)[1].detach()
        hidden=[];gradients=[];correct=0
        for p in range(PUZZLES):
            pos = positions[p]
            truth = y[p,pos]
            g,acc = selected_goal_gradient(native,stage[p],pos,truth)
            hidden.append(stage[p,pos,:].double().detach())
            gradients.append(g)
            correct+=sum(acc)
        H=torch.stack(hidden)
        G=torch.stack(gradients)
        fits[teacher]=fit_ridge(H,G,seed_id=seed("gradient_shuffle",teacher))
        fits[teacher]["calibration_clean_correct_goals"]=correct
        fits[teacher]["calibration_total_goals"]=PUZZLES*GOALS
        print("SMC24_CALIBRATED",teacher,"calibration_acc",correct/(PUZZLES*GOALS),
              "mean_grad_norm",fits[teacher]["calibration_gradient_mean_norm"],flush=True)
    return {"schema":CAL_SCHEMA,"protocol_sha256":proto_hash(),
            "source_snapshot":snap,"fits":fits,
            "calibration_use_of_target_digits":"allowed only on 64 old feature_calibration puzzles",
            "new_test_rosters_generated":False,"torch_version":torch.__version__}


def predict(original,independent,conditioned,calibration,fetch_source):
    torch.set_num_threads(2)
    raw = Path(calibration).read_bytes()
    frozen_fit = json.loads(raw)
    models, components, old, verified = load_and_lock_inputs(
        original,independent,conditioned,allow_fetch=fetch_source)
    snapshot = source_snapshot(models,components,old,verified)
    verify_calibration_receipt(frozen_fit,snapshot)
    families,manifest = fresh_rosters(old)
    result={}
    for family,rows in families.items():
        x=torch.tensor([r["x"] for r in rows],dtype=torch.long)
        positions=goal_positions(x)
        stage={}
        for teacher in TEACHERS:
            with torch.no_grad():
                stage[teacher]=models[teacher].reason_states(x)[1].detach()
        components_out={}
        for component in B_COMPONENTS:
            teacher_id,sae,mean,scale=components[component]
            cov,dec=__import__("smc23_native_first").affine_physical(sae,mean,scale)
            fit=frozen_fit["fits"][teacher_id]
            puzzle_rows=[]
            for p in range(PUZZLES):
                goals=[]
                for j,tok in enumerate(positions[p].tolist()):
                    state=stage[teacher_id][p,tok]
                    info,kill,sham=plan_target_native(
                        state,sae,mean,scale,cov,dec,
                        f"SMC24|{family}|{p}|{component}|{j}")
                    case={"goal_index":j,"token":tok,"feature":info["feature"],
                          "status":info["status"],"input_plan":info}
                    if kill is not None and sham is not None:
                        d=kill-sham
                        grad=predict_gradient(fit,state)
                        case.update({
                            "edit_kill":kill.tolist(),"edit_sham":sham.tolist(),
                            "predictions":{name:float(vector@d) for name,vector in grad.items()}
                        })
                    goals.append(case)
                puzzle_rows.append({"puzzle_index":p,
                                    "seed":rows[p]["seed"],
                                    "visible_input":rows[p]["x"],
                                    "positions":positions[p].tolist(),
                                    "goals":goals})
            components_out[component]=puzzle_rows
            elig=sum(g["status"]=="eligible" for r in puzzle_rows for g in r["goals"])
            print("SMC24_PREDICTION_ONLY",family,component,elig,flush=True)
        result[family]=components_out
    return {"schema":PRED_SCHEMA,"protocol_sha256":proto_hash(),
            "calibration_artifact_sha256":sha(raw),
            "source_snapshot":snapshot,
            "test_manifest":manifest,
            "test_targets_present":False,
            "test_gradient_from_test_labels":False,
            "teacher_models":TEACHERS,"sae_components":B_COMPONENTS,
            "test_prediction_data":result,
            "precision":"FP32 stage1, FP64 input-sham geometry and frozen ridge",
            "limitations":"The original teacher stage-one embeddings can encode task predictions, but no new correct-digit labels enter this prediction API."}


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
        report=calibrate(args.original_artifact,args.independent_artifact,
                         args.conditioned_receipt,args.fetch_source)
    else:
        if args.calibration is None:
            parser.error("prediction requires frozen calibration artifact")
        report=predict(args.original_artifact,args.independent_artifact,
                       args.conditioned_receipt,args.calibration,args.fetch_source)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    print("SMC24_STAGE_COMPLETE",args.stage,str(args.out),flush=True)


if __name__=="__main__":
    main()

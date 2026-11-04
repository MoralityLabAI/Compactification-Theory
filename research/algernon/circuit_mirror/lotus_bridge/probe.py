"""Probe actual LOTUS post-loop representations against exact pointer-machine states.

Freeze the 1..8-step training range, choose pooling on calibration only, then
score unseen graph families and 9..16 extrapolation. This probes whether the
published LOTUS model already encodes pointer transitions. It does NOT train LOTUS.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

from ..spec import PointerTask, build, witness_for_trace
from .capture import file_sha256
from .roster import validate_roster


def wilson(p: float, n: int, z: float=1.96):
    if not n:
        return [None, None]
    d = 1 + z*z/n
    m = (p + z*z/(2*n))/d
    h = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return [max(0,m-h), min(1,m+h)]


def verify_capture(feature_path, roster_path):
    import numpy as np
    roster = json.loads(Path(roster_path).read_text(encoding="utf-8"))
    validate_roster(roster)
    manifest_path = Path(feature_path).with_suffix(".manifest.json")
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("roster_sha256") != roster["roster_sha256"]:
        raise ValueError("feature manifest roster hash mismatch")
    if manifest.get("feature_sha256") != file_sha256(Path(feature_path)):
        raise ValueError("captured feature bytes changed")
    if not manifest.get("complete_roster"):
        raise ValueError("refuse scoring incomplete smoke roster")
    with np.load(feature_path,allow_pickle=False) as z:
        feat = z["features"].astype(np.float32, copy=True)
        ids = z["ids"].tolist()
    if ids != [r["id"] for r in roster["cases"]]:
        raise ValueError("captured items or ordering differ from frozen roster")
    if list(feat.shape) != manifest["feature_shape"]:
        raise ValueError("feature shape differs from manifest")
    if feat.ndim != 4 or feat.shape[:3] != (len(ids), roster["parameters"]["steps"]+1, 2):
        raise ValueError("expected [items, steps+1, pools=2, hidden_dim]")
    if not np.isfinite(feat).all():
        raise ValueError("nonfinite hidden features")
    return roster, manifest, feat


def fit_probe(features, labels, *, seed: int, updates: int, lr: float, weight_decay: float):
    import torch
    import torch.nn.functional as F
    if features.ndim != 3 or labels.shape != features.shape[:2]:
        raise ValueError("features [cases,stages,dim] and labels [cases,stages] required")
    if features.shape[1] < 2:
        raise ValueError("need at least two latent stages")
    torch.manual_seed(seed)
    X = torch.as_tensor(features[:, 1:9, :], dtype=torch.float32).reshape(-1,features.shape[2])
    y = torch.as_tensor(labels[:, 1:9], dtype=torch.long).reshape(-1)
    mean = X.mean(0)
    std = X.std(0,unbiased=False).clamp_min(1e-4)
    X = (X - mean) / std
    nclasses = int(labels.max())+1
    model = torch.nn.Linear(features.shape[2],nclasses)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    for i in range(updates):
        logits = model(X)
        loss = F.cross_entropy(logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.eval()
    return model,mean,std,float(loss.detach())


def predict(model, mean, std, features):
    import numpy as np
    import torch
    with torch.inference_mode():
        x=torch.as_tensor(features, dtype=torch.float32)
        arr=model((x-mean)/std).argmax(-1).cpu().numpy()
    return np.asarray(arr,dtype=np.int64)


def group_metrics(predicted, rows, steps):
    """Detailed transition/trace diagnostics, including exact R1CS verification."""
    import numpy as np
    predicted=np.asarray(predicted)
    if predicted.shape != (len(rows),steps+1):
        raise ValueError("wrong prediction shape")
    if not len(rows):
        raise ValueError("empty scoring group")
    truth=np.asarray([r["trace"] for r in rows],dtype=np.int64)
    accuracy=(predicted==truth).mean(0)
    n=truth.shape[1]-1
    circuit=build(PointerTask(tuple(rows[0]["successors"]),rows[0]["initial"], n))
    valid_8=valid_full=0
    depth_8=min(8,n)
    # Exact state transition checks on candidate decoded trajectories.
    # The verifier is NOT used to adjust predictions or train the probe.
    for row,p in zip(rows,predicted):
        task=PointerTask(tuple(row["successors"]),row["initial"],n)
        if any(v<0 or v>=task.n for v in p.tolist()):
            raise ValueError("decoded pointer out of domain")
        w, public=witness_for_trace(task,circuit,p.tolist())
        valid_full += int(circuit.check(w,public)["valid"])
        t8=PointerTask(tuple(row["successors"]),row["initial"],depth_8)
        c8=build(t8)
        w8,p8=witness_for_trace(t8,c8,p[:depth_8+1].tolist())
        valid_8 += int(c8.check(w8,p8)["valid"])
    step_defects=0
    for row,p in zip(rows,predicted):
        step_defects+=sum(int(p[k+1]!=row["successors"][int(p[k])]) for k in range(n))
    return {"n":len(rows),
            "accuracy_by_step":[round(float(x),6) for x in accuracy],
            "trained_steps_1_8_accuracy":float(accuracy[1:depth_8+1].mean()),
            "heldout_steps_9_16_accuracy":float(accuracy[9:17].mean()) if n>=16 else None,
            "accuracy_step_16":float(accuracy[16]) if n>=16 else None,
            "step16_wilson_95":wilson(float(accuracy[16]),len(rows)) if n>=16 else None,
            "commuting_square_defect_rate":step_defects/(len(rows)*n),
            "exact_trace_valid_1_8_rate":valid_8/len(rows),
            "exact_trace_valid_full_rate":valid_full/len(rows)}


def evaluate_probe(roster, feature_array, *, seed:int=20261104, updates:int=250,
                   lr:float=0.008, weight_decay:float=0.01):
    """Fit on train only, select readout pooling on calibration, score heldout once."""
    import numpy as np
    import torch
    torch.set_num_threads(min(4,torch.get_num_threads()))
    cases=roster["cases"]
    steps=roster["parameters"]["steps"]
    if steps<16:
        raise ValueError("full depth-extrapolation protocol needs >=16 steps")
    truth=np.asarray([r["trace"] for r in cases],dtype=np.int64)
    train=np.array([i for i,r in enumerate(cases) if r["split"]=="train"])
    cal=np.array([i for i,r in enumerate(cases) if r["split"]=="calibration"])
    test=np.array([i for i,r in enumerate(cases) if r["split"]=="test"])
    if not min(len(train),len(cal),len(test)):
        raise ValueError("all three splits required")
    pool_names=["last_latent","mean_last_block"]
    models=[]
    cal_scores=[]
    for pool in range(2):
        z=feature_array[:,:,pool,:]
        model,mu,sd,loss=fit_probe(z[train],truth[train],seed=seed+pool,
                               updates=updates,lr=lr,weight_decay=weight_decay)
        pred_cal=predict(model,mu,sd,z[cal]); mc=group_metrics(pred_cal,[cases[i] for i in cal],steps)
        cal_scores.append(mc["trained_steps_1_8_accuracy"])
        models.append((model,mu,sd,loss))
    best=int(np.argmax(cal_scores))
    model,mean,std,loss=models[best]
    z=feature_array[:,:,best,:]
    pred=predict(model,mean,std,z[test])
    test_cases=[cases[i] for i in test]
    results={}
    for family in ("long_cycle","sink"):
        mask=np.asarray([i for i,r in enumerate(test_cases) if r["family"]==family])
        results[family]=group_metrics(pred[mask],[test_cases[i] for i in mask],steps)
    results["all_test"]=group_metrics(pred,test_cases,steps)
    # Fixed no-recurrence controls, evaluated with exactly the same trained probe.
    repeated=np.repeat(pred[:, :1],steps+1,axis=1)
    results["constant_stage0_control"]=group_metrics(repeated,test_cases,steps)
    rng=np.random.default_rng(seed+10000)
    shuffle=pred.copy()
    for i in range(len(shuffle)):
        shuffle[i,1:]=shuffle[i,rng.permutation(np.arange(1,steps+1))]
    results["shuffled_stage_control"]=group_metrics(shuffle,test_cases,steps)
    # Parameter reporting and a checkable output signature; no weights from LOTUS are trained.
    linear_params=sum(p.numel() for p in model.parameters())
    state={"weight":model.weight.detach().numpy(),"bias":model.bias.detach().numpy(),
           "mean":mean.numpy(),"std":std.numpy()}
    fingerprint=hashlib.sha256(b"".join(state[k].tobytes() for k in sorted(state))).hexdigest()
    report={"status":"neural_probe_requires_real_lotus_capture",
            "training_procedure":"frozen LOTUS features; train only linear readout on train stages 1..8",
            "seed":seed,"updates":updates,"lr":lr,"weight_decay":weight_decay,
            "train_cases":len(train),"calibration_cases":len(cal),"heldout_cases":len(test),
            "selected_pool":pool_names[best],"calibration_pool_accuracy_1_8":dict(zip(pool_names,cal_scores)),
            "linear_parameters":linear_params,"final_train_loss":loss,
            "probe_sha256":fingerprint,"metrics":results,
            "controls_note":"constant-stage and shuffled-stage replace representations at evaluation only; heldout truth never enters training"}
    return report,state


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--roster",type=Path,required=True)
    ap.add_argument("--features",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--seed",type=int,default=20261104)
    ap.add_argument("--updates",type=int,default=250)
    ap.add_argument("--lr",type=float,default=0.008)
    ap.add_argument("--weight-decay",type=float,default=0.01)
    args=ap.parse_args()
    roster,manifest,features=verify_capture(args.features,args.roster)
    report,state=evaluate_probe(roster,features,seed=args.seed,updates=args.updates,
                                lr=args.lr,weight_decay=args.weight_decay)
    report["status"]="executed_on_captured_lotus_hidden_states"
    report["source_manifest_sha256"]=file_sha256(args.features.with_suffix(".manifest.json"))
    report["source_model_revision"]=manifest["source"]["model_revision"]
    report["roster_sha256"]=roster["roster_sha256"]
    report["feature_sha256"]=manifest["feature_sha256"]
    args.out.parent.mkdir(parents=True,exist_ok=True)
    import numpy as np
    weights_path=args.out.with_suffix(".probe.npz")
    np.savez_compressed(weights_path,**state)
    report["probe_file_sha256"]=file_sha256(weights_path)
    args.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(f"Readout from actual LOTUS vectors: test step16 accuracy "
          f"{report['metrics']['all_test']['accuracy_step_16']:.4f}; "
          f"commuting defect {report['metrics']['all_test']['commuting_square_defect_rate']:.4f}")
    print(f"metrics {args.out}; probe weights {weights_path}")


if __name__=="__main__":
    main()

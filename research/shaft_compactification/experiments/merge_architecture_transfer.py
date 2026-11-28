"""Merge preregistered SHAFT-COMP-18 per-cell receipts, never refit models.

Calibration and heldout are separate. Before heldout execution, the globally
calibrated thresholds must be committed to the repo and supplied unchanged.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np

from research.shaft_compactification.experiments.architecture_vocab_transfer import (
    SEEDS, REGISTRATION_COMMIT, quantile_threshold, operating_point, auc,
    summarize, summarize_calibration
)

FEATURES=("instant","history","instant_norm","history_norm","distance")


def hash_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_receipts(files,phase):
    paths=[Path(x) for x in files]
    if len(paths)!=len(SEEDS):raise ValueError("exactly four registered cells required")
    seen=set(); merged={"experiment":"SHAFT-COMP-18","phase":phase,
           "registration_commit":REGISTRATION_COMMIT,"sources":{},"models":{},"cells":{},"rows":[]}
    for path in paths:
        rec=json.loads(path.read_text())
        if rec["phase"]!=phase or rec["registration_commit"]!=REGISTRATION_COMMIT:
            raise ValueError(f"phase/registration conflict: {path}")
        model_cells={x["cell"] for x in rec["models"].values()}
        if len(model_cells)!=1 or (c:=next(iter(model_cells))) in seen or c not in SEEDS:
            raise ValueError(f"duplicate or unknown architecture/vocabulary cell: {path}")
        seen.add(c)
        if set(map(int,rec["models"]))!=set(SEEDS[c][phase]):
            raise ValueError(f"wrong seed set for {c}")
        merged["sources"][c]={"path":str(path),"sha256":hash_file(path)}
        merged["models"].update(rec["models"])
        merged["cells"].update(rec["cells"])
        merged["rows"].extend(rec["rows"])
    if seen!=set(SEEDS):raise ValueError(f"missing registered cells {set(SEEDS)-seen}")
    merged["rows"].sort(key=lambda r:(r["cell"],r["seed"],r["depth"],r["pair"]))
    merged["aggregate"]=summarize(merged["rows"])
    merged["by_arch_vocab"]={c:summarize([x for x in merged["rows"] if x["cell"]==c]) for c in SEEDS}
    merged["by_model"]={str(seed):summarize([x for x in merged["rows"] if x["seed"]==seed])
        for c in SEEDS for seed in SEEDS[c][phase]}
    return merged


def exact_auc_counts(rows,score):
    e=[r for r in rows if r["eligible"]]
    p=np.asarray([r[score] for r in e if r["diverged_32"]==1])
    n=np.asarray([r[score] for r in e if r["diverged_32"]==0])
    if len(p)==0 or len(n)==0:return None
    bigger=int((p[:,None]>n[None,:]).sum())
    tied=int((p[:,None]==n[None,:]).sum())
    check=(bigger+.5*tied)/(len(p)*len(n))
    assert abs(check-auc([r[score] for r in e],[r["diverged_32"] for r in e]))<1e-10
    return {"positives":len(p),"negatives":len(n),"strict_wins":bigger,"ties":tied}


def uncertainty(rows,permutations=1000,bootstraps=1000):
    eligible=[r for r in rows if r["eligible"]]
    y=np.asarray([r["diverged_32"] for r in eligible],int)
    one=np.asarray([r["instant"] for r in eligible],float)
    two=np.asarray([r["history"] for r in eligible],float)
    group={}
    for idx,r in enumerate(eligible):
        group.setdefault((r["cell"],r["seed"],r["depth"]),[]).append(idx)
    primary=auc(two,y)-auc(one,y)
    rng=np.random.default_rng(180018)
    null=[]
    for _ in range(permutations):
        yp=y.copy()
        for inds in group.values():yp[inds]=rng.permutation(yp[inds])
        a,b=auc(two,yp),auc(one,yp)
        if a is not None and b is not None:null.append(a-b)
    mean=float(np.mean(null))
    p=(1+sum(abs(x-mean)>=abs(primary-mean)-1e-12 for x in null))/(1+len(null))
    vals=[]
    for _ in range(bootstraps):
        indices=np.concatenate([rng.choice(inds,len(inds),replace=True) for inds in group.values()])
        a,b=auc(two[indices],y[indices]),auc(one[indices],y[indices])
        if a is not None and b is not None:vals.append((a,b,a-b))
    ci=np.quantile(np.array(vals),[.025,.975],axis=0)
    return {"primary_delta":round(primary,6),"permutation_p_two_sided":round(p,6),
            "permutation_null_mean":round(mean,6),"permutations_valid":len(null),
            "bootstraps_valid":len(vals),
            "history_auc_ci95":[round(float(x),6) for x in ci[:,0]],
            "instant_auc_ci95":[round(float(x),6) for x in ci[:,1]],
            "delta_ci95":[round(float(x),6) for x in ci[:,2]]}


def metrics_for_heldout(held,cal,thresholds):
    rows=held["rows"]
    ref=thresholds["global_thresholds"]
    held["source_thresholds_reference"]={"registration_commit":thresholds["registration_commit"],
        "calibration_full_sha256":thresholds["calibration_full_sha256"],
        "source_numeric_thresholds":ref,
        "threshold_schema":"0.9 higher quantile of source calibration negatives, strict greater-than"}
    held["operating_global"]={c:{f:operating_point([r for r in rows if r["cell"]==c],f,ref[f])
            for f in ref} for c in SEEDS}
    held["operating_global"]["pooled"]={f:operating_point(rows,f,ref[f]) for f in ref}
    held["operating_adapted"]={c:{f:operating_point([r for r in rows if r["cell"]==c],f,
                          thresholds["cell_thresholds"][c][f]) for f in ref} for c in SEEDS}
    held["pairwise_counts"]={f:exact_auc_counts(rows,f) for f in FEATURES}
    held["inference"]=uncertainty(rows)
    delta=lambda group:None if any(group["auc"][x] is None for x in ("history","instant")) else round(group["auc"]["history"]-group["auc"]["instant"],6)
    held["by_cell_delta"]={c:delta(held["by_arch_vocab"][c]) for c in SEEDS}
    held["by_model_delta"]={str(s):delta(held["by_model"][str(s)]) for c in SEEDS for s in SEEDS[c]["heldout"]}
    crit=held["inference"]["delta_ci95"][0]>0 and sum(v is not None and v>0 for v in held["by_cell_delta"].values())>=3 and sum(v is not None and v>0 for v in held["by_model_delta"].values())>=6
    fpr_crit=all(held["operating_global"][c]["history"]["fpr"] is not None and held["operating_global"][c]["history"]["fpr"]<=.15 for c in SEEDS)
    held["preregistered_decision"]={"auc_support_criteria_met":crit,
        "global_fixed_fpr_transfer_criteria_met":fpr_crit,
        "positive_cell_count":sum(v is not None and v>0 for v in held["by_cell_delta"].values()),
        "positive_model_count":sum(v is not None and v>0 for v in held["by_model_delta"].values())}
    return held


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--phase",required=True,choices=("calibration","heldout"))
    p.add_argument("--receipts",required=True,nargs=4)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--thresholds-out",type=Path)
    p.add_argument("--frozen-thresholds",type=Path)
    a=p.parse_args()
    m=read_receipts(a.receipts,a.phase)
    if a.phase=="calibration":
        if a.thresholds_out is None:raise ValueError("calibration needs thresholds-out")
        t=summarize_calibration(m)
        t["calibration_full_sha256"]="UNSET_PENDING_FILE_HASH"
        # Save receipt once, then persist its exact SHA in threshold file.
        a.out.write_text(json.dumps(m,indent=2,sort_keys=True)+"\n")
        t["calibration_full_sha256"]=hash_file(a.out)
        a.thresholds_out.write_text(json.dumps(t,indent=2,sort_keys=True)+"\n")
    else:
        if a.frozen_thresholds is None:raise ValueError("heldout needs frozen threshold receipt")
        t=json.loads(a.frozen_thresholds.read_text())
        if t["registration_commit"]!=REGISTRATION_COMMIT or t["phase"]!="calibration_thresholds":
            # summarize_calibration() stores the phase label; fail closed if absent.
            raise ValueError("wrong preregistered thresholds")
        m=metrics_for_heldout(m,None,t)
        a.out.write_text(json.dumps(m,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"phase":a.phase,"aggregate":m["aggregate"],"models":len(m["models"]),
        "heldout_decision":m.get("preregistered_decision")},sort_keys=True))

if __name__=="__main__":main()

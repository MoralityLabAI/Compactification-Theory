"""SHAFT-COMP-13: preregistered observability rank and divergence experiment.

Uses frozen synthetic four-bit GRUs. Independent diagnostic and outcome
continuations. Seeds 303/404 are held out from prior SHAFT work. No external
devices, persistence, secret sharing or external requests.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from research.shaft_compactification.experiments.trained_recurrence import (
    fit, GOALS, HIDDEN, TOKENS,
)
from research.shaft_compactification.experiments.causal_probe import train_probe, counterfactual
from research.shaft_compactification.experiments.gramian_probe import analytic_jacobian

MODEL_SEEDS=(101,202,303,404)
DEPTHS=(12,40)
PER_GOAL=16
DIAGNOSTIC_STEPS=5
OUTCOME_STEPS=32
RANK_RELATIVE_THRESHOLD=1e-3
PERMUTATIONS=1000
BOOTSTRAPS=1000

def model_digest(model):
    digest=hashlib.sha256()
    for key,weight in model.state_dict().items():
        digest.update(key.encode("utf-8"))
        digest.update(weight.detach().contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()

def sample_pair(model,seed,depth,n_per_goal):
    goals=torch.arange(GOALS).repeat_interleave(n_per_goal)
    gen=torch.Generator().manual_seed(1300000+seed*1000+depth)
    goals=goals[torch.randperm(len(goals),generator=gen)]
    n=len(goals)
    a=torch.randint(TOKENS,(n,depth),generator=gen)
    b=torch.randint(TOKENS,(n,depth),generator=gen)
    with torch.no_grad():
        ha=model.embedding(goals)
        hb=model.embedding(goals)
        for step in range(depth):
            ha=model.cell(F.one_hot(a[:,step],TOKENS).float(),ha)
            hb=model.cell(F.one_hot(b[:,step],TOKENS).float(),hb)
    return goals,ha,hb,gen

def score_observability(model,probe,states,schedules):
    n=len(states)
    with torch.no_grad():
        h=states.clone()
        W=probe.weight.detach()
        Phi=torch.eye(HIDDEN).repeat(n,1,1)
        blocks=[torch.matmul(W,Phi)]
        for t in range(DIAGNOSTIC_STEPS):
            x=F.one_hot(schedules[:,t],TOKENS).float()
            J=analytic_jacobian(model.cell,x,h)
            Phi=J@Phi
            blocks.append(torch.matmul(W,Phi))
            h=model.cell(x,h)
        stacked=torch.cat(blocks,dim=1).double()
        sv=torch.linalg.svdvals(stacked)
        rank=(sv>RANK_RELATIVE_THRESHOLD*sv[:,[0]]).sum(dim=1)
        weak=-torch.log10((sv[:,-1]/sv[:,0]).clamp_min(1e-12))
        margin=probe(states).abs().min(dim=1).values
    return rank.cpu().numpy(),weak.cpu().numpy(),margin.cpu().numpy()

def auc(scores,labels):
    """Mann-Whitney AUC with average ranks for tied scores."""
    scores=np.asarray(scores,dtype=np.float64)
    labels=np.asarray(labels,dtype=np.int64)
    pos=int(labels.sum());neg=len(labels)-pos
    if not pos or not neg:return None
    ix=np.argsort(scores,kind="stable")
    ss=scores[ix];ranks=np.empty(len(scores),float)
    i=0
    while i<len(scores):
        j=i+1
        while j<len(scores) and ss[j]==ss[i]:j+=1
        ranks[ix[i:j]]=(i+1+j)/2
        i=j
    return float((ranks[labels==1].sum()-pos*(pos+1)/2)/(pos*neg))

def per_cell(rows):
    eligible=[r for r in rows if r["eligible"]]
    metrics={}
    for key in ("rank_deficit","weak_singular","low_probe_margin","state_distance"):
        metrics[key]=auc([r[key] for r in eligible],[r["diverged_32"] for r in eligible])
    return {"eligible":len(eligible),"total":len(rows),
            "positives":sum(r["diverged_32"] for r in eligible),
            "prevalence":round(sum(r["diverged_32"] for r in eligible)/max(1,len(eligible)),6),
            "auroc":{k:(None if v is None else round(v,6)) for k,v in metrics.items()},
            "rank_deficit_histogram":{str(v):sum(r["rank_deficit"]==v for r in eligible)
                                      for v in sorted({r["rank_deficit"] for r in eligible})}}

def evaluate_seed_depth(model,probe,seed,depth,n_per_goal):
    goals,ha,hb,gen=sample_pair(model,seed,depth,n_per_goal)
    n=len(goals)
    with torch.no_grad():
        a0=(counterfactual(model,ha,torch.empty((n,0),dtype=torch.long))>=0)
        b0=(counterfactual(model,hb,torch.empty((n,0),dtype=torch.long))>=0)
        ca=(probe(ha)>=0);cb=(probe(hb)>=0)
        teacher_agree=(a0==b0).all(dim=1)
        probe_agree=(ca==cb).all(dim=1)
        diagnostic_a=torch.randint(TOKENS,(n,DIAGNOSTIC_STEPS),generator=gen)
        diagnostic_b=torch.randint(TOKENS,(n,DIAGNOSTIC_STEPS),generator=gen)
        outcome=torch.randint(TOKENS,(n,OUTCOME_STEPS),generator=gen)
    rank_a,weak_a,margin_a=score_observability(model,probe,ha,diagnostic_a)
    rank_b,weak_b,margin_b=score_observability(model,probe,hb,diagnostic_b)
    with torch.no_grad():
        final_a=(counterfactual(model,ha,outcome)>=0)
        final_b=(counterfactual(model,hb,outcome)>=0)
        disagree=(final_a!=final_b).any(dim=1).cpu().numpy()
        separation=(ha-hb).norm(dim=1).cpu().numpy()
    rows=[]
    for i in range(n):
        rows.append({
            "seed":int(seed),"depth":int(depth),"pair":int(i),
            "goal_for_audit_only":int(goals[i]),
            "teacher_equal_0":bool(teacher_agree[i]),
            "probe_equal_0":bool(probe_agree[i]),
            "eligible":bool(teacher_agree[i] and probe_agree[i]),
            "diverged_32":int(disagree[i]),
            "rank_a":int(rank_a[i]),"rank_b":int(rank_b[i]),
            "rank_deficit":float(HIDDEN-(rank_a[i]+rank_b[i])/2),
            "weak_singular":float((weak_a[i]+weak_b[i])/2),
            "low_probe_margin":float(-(margin_a[i]+margin_b[i])/2),
            "state_distance":float(separation[i]),
        })
    return rows

def uncertainty(rows,permutations=PERMUTATIONS,bootstraps=BOOTSTRAPS):
    rows=[r for r in rows if r["eligible"] and r["seed"] in (303,404)]
    if not rows:return {"error":"no eligible test pairs"}
    groups={}
    for i,r in enumerate(rows):groups.setdefault((r["seed"],r["depth"]),[]).append(i)
    labels=np.array([r["diverged_32"] for r in rows],dtype=int)
    scores=np.array([r["rank_deficit"] for r in rows],dtype=float)
    observed=auc(scores,labels)
    rng=np.random.default_rng(24019013)
    null=[]
    for _ in range(permutations):
        perm=labels.copy()
        for idx in groups.values():perm[idx]=rng.permutation(perm[idx])
        x=auc(scores,perm)
        if x is not None:null.append(x)
    center=float(np.mean(null)) if null else None
    p=(1+sum(abs(x-center)>=abs(observed-center)-1e-12 for x in null))/(len(null)+1) if null and observed is not None else None
    boot=[]
    for _ in range(bootstraps):
        idx=np.concatenate([rng.choice(indices,len(indices),replace=True)
                            for indices in groups.values()])
        x=auc(scores[idx],labels[idx])
        if x is not None:boot.append(x)
    lo,hi=np.quantile(boot,[.025,.975]) if boot else (float("nan"),float("nan"))
    return {"primary_auc":None if observed is None else round(observed,6),
            "permutation_p_two_sided":None if p is None else round(float(p),6),
            "permutation_null_mean":None if center is None else round(center,6),
            "permutation_trials":len(null),
            "bootstrap_trials_valid":len(boot),
            "bootstrap_95_percentile":[round(float(lo),6),round(float(hi),6)],
            "note":"within-seed-depth permutation and bootstrap; only two held-out model seeds"}

def run(per_goal=PER_GOAL,permutations=PERMUTATIONS,bootstraps=BOOTSTRAPS):
    torch.set_num_threads(3)
    torch.use_deterministic_algorithms(True)
    report={"experiment":"SHAFT-COMP-13","status":"locally_trained_synthetic_gru",
            "torch":torch.__version__,"numpy":np.__version__,
            "protocol":"SHAFT_COMP_13_PROTOCOL.md","monitor_outputs":4,
            "state_dimension":HIDDEN,"diagnostic_horizon":DIAGNOSTIC_STEPS,
            "rank_relative_threshold":RANK_RELATIVE_THRESHOLD,
            "outcome_continuation_length":OUTCOME_STEPS,
            "calibration_model_seeds":[101,202],"heldout_model_seeds":[303,404],
            "pairs_per_goal":per_goal,"model_hashes":{},"cell_metrics":{},
            "test_results":{},"rows":[]}
    for seed in MODEL_SEEDS:
        model,_=fit(seed)
        probe,_=train_probe(model,seed)
        report["model_hashes"][str(seed)]={"base":model_digest(model),"probe":model_digest(probe)}
        for depth in DEPTHS:
            rows=evaluate_seed_depth(model,probe,seed,depth,per_goal)
            report["cell_metrics"][f"{seed}_d{depth}"]=per_cell(rows)
            report["rows"].extend(rows)
    report["test_results"]["heldout_pooled"]=per_cell(
        [r for r in report["rows"] if r["seed"] in (303,404)])
    report["test_results"]["uncertainty"]=uncertainty(report["rows"],permutations,bootstraps)
    report["test_results"]["calibration_pooled"]=per_cell(
        [r for r in report["rows"] if r["seed"] in (101,202)])
    return report

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--per-goal",type=int,default=PER_GOAL)
    p.add_argument("--permutations",type=int,default=PERMUTATIONS)
    p.add_argument("--bootstraps",type=int,default=BOOTSTRAPS)
    p.add_argument("--out",type=Path)
    a=p.parse_args()
    if not 1<=a.per_goal<=64:raise ValueError("per-goal must be 1..64")
    result=run(a.per_goal,a.permutations,a.bootstraps)
    payload=json.dumps(result,indent=2,sort_keys=True)+"\n"
    if a.out:
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(payload)
    else:print(payload,end="")
if __name__=="__main__":main()

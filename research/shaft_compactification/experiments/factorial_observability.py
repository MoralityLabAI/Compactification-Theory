"""SHAFT-COMP-15: preregistered learned/native readout x rank/direction factorial.

Two held-in calibration GRUs (101, 202), four fresh held-out GRUs
(707, 808, 909, 1001). Synthetic, local, in-memory tasks only.
Protocol: SHAFT_COMP_15_PROTOCOL.md. No network/device operations.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from research.shaft_compactification.experiments.trained_recurrence import fit, GOALS, HIDDEN, TOKENS
from research.shaft_compactification.experiments.causal_probe import train_probe, counterfactual
from research.shaft_compactification.experiments.gramian_probe import analytic_jacobian
from research.shaft_compactification.experiments.task_weighted_observability import native_logits_grad, check_grad
from research.shaft_compactification.experiments.observability_prediction import sample_pair, model_digest, auc

CALIBRATION_SEEDS=(101,202)
HELDOUT_SEEDS=(707,808,909,1001)
DEPTHS=(12,40)
PER_GOAL=16
DIAGNOSTIC_STEPS=5
OUTCOME_STEPS=32
RANK_THRESHOLD=1e-3
PERMUTATIONS=1000
BOOTSTRAPS=1000
SCORES=("learned_rank","learned_direction","native_rank","native_direction",
        "learned_direction_normed","native_direction_normed","state_distance")

def paired_factorial_scores(model, probe, a, b, diagnostic):
    """Each readout supplies the SAME observation stack for rank and energy."""
    n=len(a)
    if b.shape!=a.shape or diagnostic.shape!=(n,DIAGNOSTIC_STEPS):
        raise ValueError("misaligned states or diagnostic inputs")
    W=probe.weight.detach()
    delta=(a-b).unsqueeze(-1)
    identity=torch.eye(HIDDEN,device=a.device,dtype=a.dtype).unsqueeze(0).expand(n,HIDDEN,HIDDEN)
    ledir,nedir,lerank,nerank=[],[],[],[]
    with torch.no_grad():
        for state in (a,b):
            h=state.detach().clone()
            phi=identity.clone()
            learn_blocks,native_blocks=[],[]
            for t in range(DIAGNOSTIC_STEPS+1):
                _,C=native_logits_grad(model,h)
                learn_blocks.append(torch.matmul(W,phi))
                native_blocks.append(torch.matmul(C,phi))
                if t<DIAGNOSTIC_STEPS:
                    token=F.one_hot(diagnostic[:,t],TOKENS).float()
                    J=analytic_jacobian(model.cell,token,h)
                    phi=torch.matmul(J,phi)
                    h=model.cell(token,h)
            O_learn=torch.cat(learn_blocks,dim=1).double()
            O_native=torch.cat(native_blocks,dim=1).double()
            contrast=delta.double()
            ledir.append(torch.matmul(O_learn,contrast).square().sum(dim=(1,2)))
            nedir.append(torch.matmul(O_native,contrast).square().sum(dim=(1,2)))
            for O,ranks in ((O_learn,lerank),(O_native,nerank)):
                s=torch.linalg.svdvals(O)
                ranks.append((s>RANK_THRESHOLD*s[:,[0]]).sum(dim=1).double())
        ld=torch.sqrt((ledir[0]+ledir[1]).clamp_min(0)/2)
        nd=torch.sqrt((nedir[0]+nedir[1]).clamp_min(0)/2)
        norm=(a-b).double().norm(dim=1)
        values={
            "learned_rank":HIDDEN-(lerank[0]+lerank[1])/2,
            "learned_direction":ld,
            "native_rank":HIDDEN-(nerank[0]+nerank[1])/2,
            "native_direction":nd,
            "learned_direction_normed":ld/norm.clamp_min(1e-9),
            "native_direction_normed":nd/norm.clamp_min(1e-9),
            "state_distance":norm,
        }
        for key,v in values.items():
            if not torch.isfinite(v).all():
                raise RuntimeError(f"nonfinite {key}")
    return {key:v.cpu().numpy() for key,v in values.items()}

def cell_rows(model,probe,seed,depth,per_goal):
    goals,a,b,_=sample_pair(model,seed,depth,per_goal)
    n=len(goals)
    diag_gen=torch.Generator().manual_seed(1400000+seed*1000+depth)
    outcome_gen=torch.Generator().manual_seed(1500000+seed*1000+depth)
    diagnostic=torch.randint(TOKENS,(n,DIAGNOSTIC_STEPS),generator=diag_gen)
    outcome=torch.randint(TOKENS,(n,OUTCOME_STEPS),generator=outcome_gen)
    with torch.no_grad():
        ca=counterfactual(model,a,torch.empty((n,0),dtype=torch.long))>=0
        cb=counterfactual(model,b,torch.empty((n,0),dtype=torch.long))>=0
        native_same=(ca==cb).all(dim=1)
        probe_same=((probe(a)>=0)==(probe(b)>=0)).all(dim=1)
        fa=counterfactual(model,a,outcome)>=0
        fb=counterfactual(model,b,outcome)>=0
        divergent=(fa!=fb).any(dim=1)
    features=paired_factorial_scores(model,probe,a,b,diagnostic)
    return [
        dict(seed=int(seed),depth=int(depth),pair=int(i),
             goal_for_audit_only=int(goals[i]),
             teacher_same=bool(native_same[i]),probe_same=bool(probe_same[i]),
             eligible=bool(native_same[i] & probe_same[i]),
             diverged_32=int(divergent[i]),
             **{k:float(v[i]) for k,v in features.items()})
        for i in range(n)
    ]

def summary(rows):
    eligible=[r for r in rows if r["eligible"]]
    labels=[r["diverged_32"] for r in eligible]
    scores={k:auc([r[k] for r in eligible],labels) for k in SCORES}
    return {"total":len(rows),"eligible":len(eligible),"positives":sum(labels),
            "prevalence":round(sum(labels)/len(eligible),6) if eligible else None,
            "auc":{k:round(v,6) if v is not None else None for k,v in scores.items()}}

def contrasts(a):
    if any(a[k] is None for k in SCORES):
        return {k:None for k in ("learned_direction_minus_rank","native_direction_minus_rank",
             "native_minus_learned_rank","native_minus_learned_direction","interaction")}
    ld=a["learned_direction"]-a["learned_rank"]
    nd=a["native_direction"]-a["native_rank"]
    return {
        "learned_direction_minus_rank":round(ld,6),
        "native_direction_minus_rank":round(nd,6),
        "native_minus_learned_rank":round(a["native_rank"]-a["learned_rank"],6),
        "native_minus_learned_direction":round(a["native_direction"]-a["learned_direction"],6),
        "interaction":round(nd-ld,6),
    }

def infer(rows,permutations=PERMUTATIONS,bootstraps=BOOTSTRAPS):
    e=[r for r in rows if r["eligible"] and r["seed"] in HELDOUT_SEEDS]
    y=np.asarray([r["diverged_32"] for r in e],dtype=int)
    values={k:np.asarray([r[k] for r in e],float) for k in SCORES}
    groups={}
    for i,r in enumerate(e):
        groups.setdefault((r["seed"],r["depth"]),[]).append(i)
    observed=auc(values["learned_direction"],y)-auc(values["learned_rank"],y)
    rng=np.random.default_rng(151515)
    per=[]
    for _ in range(permutations):
        target=y.copy()
        for idx in groups.values():
            target[idx]=rng.permutation(target[idx])
        per.append(auc(values["learned_direction"],target)-auc(values["learned_rank"],target))
    null_mean=float(np.mean(per)) if per else None
    p=(1+sum(abs(v-null_mean)>=abs(observed-null_mean)-1e-12 for v in per))/(1+len(per)) if per else None
    boots,delta=[],[]
    for _ in range(bootstraps):
        idx=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups.values()])
        ys=y[idx]
        a=[auc(values[k][idx],ys) for k in SCORES]
        if all(v is not None for v in a):
            boots.append(a)
            delta.append(a[SCORES.index("learned_direction")]-a[SCORES.index("learned_rank")])
    boot=np.asarray(boots)
    intervals={k:[round(float(v),6) for v in np.quantile(boot[:,i],[.025,.975])]
               for i,k in enumerate(SCORES)} if len(boot) else None
    diff_ci=[round(float(v),6) for v in np.quantile(delta,[.025,.975])] if delta else None
    seedwise={}
    for seed in HELDOUT_SEEDS:
        a=summary([r for r in e if r["seed"]==seed])
        seedwise[str(seed)]={"eligible":a["eligible"],"positives":a["positives"],
                             "auc":a["auc"],"contrasts":contrasts(a["auc"])}
    leave={}
    for seed in HELDOUT_SEEDS:
        a=summary([r for r in e if r["seed"]!=seed])
        leave[str(seed)]=contrasts(a["auc"])["learned_direction_minus_rank"]
    n_positive=sum(1 for r in seedwise.values() if r["contrasts"]["learned_direction_minus_rank"] is not None and r["contrasts"]["learned_direction_minus_rank"]>0)
    return {"primary_delta":round(observed,6),
            "permutation_two_sided_p_centered":round(float(p),6) if p is not None else None,
            "permutation_null_mean":round(null_mean,6) if null_mean is not None else None,
            "permutation_trials":len(per),"bootstrap_trials_valid":len(boot),
            "primary_difference_bootstrap95":diff_ci,
            "four_way_and_controls_bootstrap95":intervals,
            "seedwise":seedwise,
            "leave_one_model_out_primary_difference":leave,
            "positive_seed_count":n_positive,
            "preregistered_support_criteria_met":bool(diff_ci and diff_ci[0]>0 and n_positive>=3),
            "uncertainty_note":"within model/depth; four independent held-out models"}

def run(per_goal=PER_GOAL,permutations=PERMUTATIONS,bootstraps=BOOTSTRAPS):
    torch.set_num_threads(3)
    torch.use_deterministic_algorithms(True)
    report={"experiment":"SHAFT-COMP-15",
            "preregistered_protocol":"SHAFT_COMP_15_PROTOCOL.md",
            "calibration_seeds":list(CALIBRATION_SEEDS),
            "heldout_seeds":list(HELDOUT_SEEDS),"depths":list(DEPTHS),
            "per_goal":per_goal,"diagnostic_steps":DIAGNOSTIC_STEPS,
            "outcome_steps":OUTCOME_STEPS,"relative_rank_threshold":RANK_THRESHOLD,
            "torch":torch.__version__,"numpy":np.__version__,
            "model_hashes":{},"jacobian_parity":{},"cells":{},"rows":[]}
    for seed in CALIBRATION_SEEDS+HELDOUT_SEEDS:
        model,_=fit(seed)
        probe,_=train_probe(model,seed)
        report["model_hashes"][str(seed)]={"base":model_digest(model),"probe":model_digest(probe)}
        report["jacobian_parity"][str(seed)]=check_grad(model)
        for depth in DEPTHS:
            batch=cell_rows(model,probe,seed,depth,per_goal)
            report["cells"][f"{seed}_d{depth}"]=summary(batch)
            report["rows"].extend(batch)
    calibration=[r for r in report["rows"] if r["seed"] in CALIBRATION_SEEDS]
    heldout=[r for r in report["rows"] if r["seed"] in HELDOUT_SEEDS]
    report["calibration"]=summary(calibration)
    report["heldout"]=summary(heldout)
    report["calibration"]["contrasts"]=contrasts(report["calibration"]["auc"])
    report["heldout"]["contrasts"]=contrasts(report["heldout"]["auc"])
    report["inference"]=infer(report["rows"],permutations,bootstraps)
    return report

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--per-goal",type=int,default=PER_GOAL)
    p.add_argument("--permutations",type=int,default=PERMUTATIONS)
    p.add_argument("--bootstraps",type=int,default=BOOTSTRAPS)
    p.add_argument("--out",type=Path)
    a=p.parse_args()
    if not 1<=a.per_goal<=64:
        raise ValueError("per-goal must be 1..64")
    result=run(a.per_goal,a.permutations,a.bootstraps)
    payload=json.dumps(result,sort_keys=True,indent=2)+"\n"
    if a.out:
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(payload)
    else:
        print(payload,end="")
if __name__=="__main__":
    main()

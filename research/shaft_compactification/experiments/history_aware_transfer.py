"""SHAFT-COMP-17: two-step behavior-supervised probes across new temporal task families.

All experiments train fresh 3,265-param toy GRUs locally. Only 4- and 16-output
linear readouts of frozen recurrent activations are fitted. Outcomes and fixed
calibration thresholds are kept separate by mandatory phase gates. No network,
external device, autonomous agent, model replica, or persistence operations.

Preregistered: SHAFT_COMP_17_PROTOCOL.md (commit 938560c3cad9d2d8cf298f349311900a2510e0fe).
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from research.shaft_compactification.experiments.trained_recurrence import (
    TaskMemoryGRU, GOALS, TOKENS, HIDDEN, BATCH, TRAIN_LENGTH,
)
from research.shaft_compactification.experiments.causal_probe import (
    train_probe, capture, counterfactual,
)
from research.shaft_compactification.experiments.gramian_probe import analytic_jacobian
from research.shaft_compactification.experiments.observability_prediction import (
    sample_pair, model_digest, auc,
)

FAMILY_SEEDS={
    "xor_lag":{"calibration":(1301,1302), "heldout":(1303,1304)},
    "parity_gate":{"calibration":(1401,1402), "heldout":(1403,1404)},
}
TRAIN_STEPS=1200
TRAIN_LR=0.01
PER_GOAL=16
DEPTHS=(12,40)
DIAGNOSTIC_STEPS=5
OUTCOME_STEPS=32
PROBE_STEPS=400
PROBE_LR=0.04
CALIBRATION_QUANTILE=0.9
PERMUTATIONS=1000
BOOTSTRAPS=1000
REGISTRATION_COMMIT="938560c3cad9d2d8cf298f349311900a2510e0fe"
SCORES=("instant_direction", "history_direction", "instant_direction_norm", "history_direction_norm", "pair_distance")


def family_targets(goals:torch.Tensor, tokens:torch.Tensor, family:str):
    if family not in FAMILY_SEEDS: raise ValueError(f"unknown family: {family}")
    previous=torch.cat([torch.zeros_like(tokens[:,:1]),tokens[:,:-1]],dim=1)
    if family=="xor_lag":
        index=torch.bitwise_xor(tokens,previous)
        return ((goals[:,None] >> index) & 1).float()
    bit=((goals[:,None] >> tokens) & 1)
    parity=torch.bitwise_and(torch.bitwise_xor(tokens,previous),1)
    return torch.bitwise_xor(bit,parity).float()


def fit_family(seed:int,family:str,steps=TRAIN_STEPS):
    if family not in FAMILY_SEEDS: raise ValueError(family)
    torch.manual_seed(seed)
    rng=torch.Generator(device="cpu").manual_seed(10000+seed)
    model=TaskMemoryGRU()
    opt=torch.optim.Adam(model.parameters(),lr=TRAIN_LR)
    for _ in range(steps):
        goals=torch.randint(0,GOALS,(BATCH,),generator=rng)
        tokens=torch.randint(0,TOKENS,(BATCH,TRAIN_LENGTH),generator=rng)
        y=family_targets(goals,tokens,family)
        loss=F.binary_cross_entropy_with_logits(model(goals,tokens),y)
        opt.zero_grad();loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),2.)
        opt.step()
    model.eval()
    return model,float(loss.detach())


@torch.no_grad()
def test_task_accuracy(model,seed,family,length=12,samples=1024):
    rng=torch.Generator(device="cpu").manual_seed(3000000+seed+43*length)
    goals=torch.randint(GOALS,(samples,),generator=rng)
    tokens=torch.randint(TOKENS,(samples,length),generator=rng)
    y=family_targets(goals,tokens,family)
    p=model(goals,tokens)>=0
    return float((p==(y>=0.5)).float().mean())


@torch.no_grad()
def counterfactual_second(model:TaskMemoryGRU,states:torch.Tensor,chunk_size=256):
    """Native model's final bit after all 16 ordered query pairs (q1,q2)."""
    n=len(states)
    q=torch.arange(TOKENS)
    first=q.repeat_interleave(TOKENS)
    second=q.repeat(TOKENS)
    hot1=F.one_hot(first,TOKENS).float()
    hot2=F.one_hot(second,TOKENS).float()
    batches=[]
    for start in range(0,n,chunk_size):
        h=states[start:start+chunk_size]
        repeated=h.repeat_interleave(TOKENS*TOKENS,dim=0)
        x1=hot1.repeat(len(h),1)
        x2=hot2.repeat(len(h),1)
        state1=model.cell(x1,repeated)
        state2=model.cell(x2,state1)
        logits=model.readout(torch.cat([state2,x2],dim=1)).reshape(len(h),TOKENS*TOKENS)
        batches.append(logits)
    return torch.cat(batches,dim=0)


def fit_history_probe(model,seed):
    """Same frozen training states as the 4-output probe, 16 teacher targets."""
    dataset=capture(model,(4,8,12),48,seed+70000)
    states=torch.cat([dataset[k][0] for k in (4,8,12)],dim=0)
    target=(counterfactual_second(model,states)>=0).float()
    torch.manual_seed(seed+91000)
    probe=nn.Linear(HIDDEN,TOKENS*TOKENS)
    opt=torch.optim.Adam(probe.parameters(),lr=PROBE_LR)
    for _ in range(PROBE_STEPS):
        loss=F.binary_cross_entropy_with_logits(probe(states),target)
        opt.zero_grad();loss.backward();opt.step()
    return probe.eval(),float(loss.detach())


@torch.no_grad()
def paired_directional_scores(model,one_probe,two_probe,a,b,diagnostic):
    """Predeclared directional energy for W1/W2 along the SAME trajectories."""
    n=len(a)
    if b.shape!=a.shape or diagnostic.shape!=(n,DIAGNOSTIC_STEPS):
        raise ValueError("state pair or diagnostic shape mismatch")
    delta=(a-b).unsqueeze(-1)
    phi0=torch.eye(HIDDEN,device=a.device,dtype=a.dtype).expand(n,HIDDEN,HIDDEN)
    result=[]
    for probe in (one_probe,two_probe):
        W=probe.weight.detach()
        members=[]
        for h_initial in (a,b):
            h=h_initial.detach().clone()
            phi=phi0.clone()
            energy=torch.zeros(n,device=h.device)
            for step in range(DIAGNOSTIC_STEPS+1):
                influence=(W@phi@delta).squeeze(-1)
                energy+=influence.square().sum(dim=1)
                if step<DIAGNOSTIC_STEPS:
                    token=F.one_hot(diagnostic[:,step],TOKENS).float()
                    J=analytic_jacobian(model.cell,token,h)
                    phi=J@phi
                    h=model.cell(token,h)
            members.append(energy)
        result.append(torch.sqrt(.5*(members[0]+members[1]).clamp_min(0)))
    norm=(a-b).norm(dim=1)
    outputs={
        "instant_direction":result[0],
        "history_direction":result[1],
        "instant_direction_norm":result[0]/norm.clamp_min(1e-9),
        "history_direction_norm":result[1]/norm.clamp_min(1e-9),
        "pair_distance":norm,
    }
    for key,val in outputs.items():
        if not torch.isfinite(val).all():raise ValueError(f"nonfinite feature: {key}")
    return outputs


def evaluate_cell(model,one_probe,two_probe,seed,family,depth,per_goal):
    goals,a,b,_=sample_pair(model,seed,depth,per_goal)
    n=len(goals)
    rng_diag=torch.Generator().manual_seed(1700000+seed*1000+depth)
    rng_out=torch.Generator().manual_seed(1800000+seed*1000+depth)
    diagnostic=torch.randint(TOKENS,(n,DIAGNOSTIC_STEPS),generator=rng_diag)
    outcome=torch.randint(TOKENS,(n,OUTCOME_STEPS),generator=rng_out)
    with torch.no_grad():
        teacher_a=counterfactual(model,a,torch.empty((n,0),dtype=torch.long))>=0
        teacher_b=counterfactual(model,b,torch.empty((n,0),dtype=torch.long))>=0
        one_a=(one_probe(a)>=0);one_b=(one_probe(b)>=0)
        two_a=(two_probe(a)>=0);two_b=(two_probe(b)>=0)
        native_same=(teacher_a==teacher_b).all(dim=1)
        old_same=(one_a==one_b).all(dim=1)
        new_same=(two_a==two_b).all(dim=1)
        out_a=counterfactual(model,a,outcome)>=0
        out_b=counterfactual(model,b,outcome)>=0
        diverged=(out_a!=out_b).any(dim=1)
        target_one_a=teacher_a
        target_one_b=teacher_b
        target_two_a=counterfactual_second(model,a)>=0
        target_two_b=counterfactual_second(model,b)>=0
        one_fit=torch.cat(((one_a==target_one_a).float(),(one_b==target_one_b).float()),dim=0)
        two_fit=torch.cat(((two_a==target_two_a).float(),(two_b==target_two_b).float()),dim=0)
        one_exact=torch.cat(((one_a==target_one_a).all(dim=1).float(),
                             (one_b==target_one_b).all(dim=1).float()))
        two_exact=torch.cat(((two_a==target_two_a).all(dim=1).float(),
                             (two_b==target_two_b).all(dim=1).float()))
    values=paired_directional_scores(model,one_probe,two_probe,a,b,diagnostic)
    rows=[]
    for i in range(n):
        row={"family":family,"seed":int(seed),"depth":int(depth),"pair":int(i),
             "goal_for_audit_only":int(goals[i]),"teacher_current_equal":bool(native_same[i]),
             "instant_code_equal":bool(old_same[i]),"history_code_equal":bool(new_same[i]),
             "eligible":bool(native_same[i]&old_same[i]),"diverged_32":int(diverged[i])}
        row.update({key:float(t[i]) for key,t in values.items()})
        rows.append(row)
    audit={
        "one_step_probe_bit_agreement":float(one_fit.mean()),
        "two_step_probe_bit_agreement":float(two_fit.mean()),
        "one_step_probe_exact_agreement":float(one_exact.mean()),
        "two_step_probe_exact_agreement":float(two_exact.mean()),
        "history_code_disagrees_on_presently_eligible":int((~new_same & native_same & old_same).sum()),
    }
    return rows,audit


def summarize(rows):
    eligible=[r for r in rows if r["eligible"]]
    labels=[r["diverged_32"] for r in eligible]
    values={k:auc([r[k] for r in eligible],labels) for k in SCORES}
    return {"total":len(rows),"eligible":len(eligible),"positives":sum(labels),
            "positive_fraction":round(sum(labels)/len(eligible),6) if eligible else None,
            "auc":{key:None if value is None else round(value,6) for key,value in values.items()}}


def choose_threshold(rows,feature):
    neg=np.asarray([r[feature] for r in rows if r["eligible"] and not r["diverged_32"]],float)
    if len(neg)==0:raise ValueError("no calibration negatives")
    return float(np.quantile(neg,CALIBRATION_QUANTILE,method="higher"))


def operating_point(rows,feature,threshold):
    valid=[r for r in rows if r["eligible"]]
    negatives=[r for r in valid if not r["diverged_32"]]
    positives=[r for r in valid if r["diverged_32"]]
    fp=sum(r[feature]>threshold for r in negatives)
    tp=sum(r[feature]>threshold for r in positives)
    return {"threshold":float(threshold),"false_positives":fp,"true_positives":tp,
            "negatives":len(negatives),"positives":len(positives),
            "fpr":None if not negatives else round(fp/len(negatives),6),
            "tpr":None if not positives else round(tp/len(positives),6)}


def compute_calibration_receipt(report):
    assert report["phase"]=="calibration"
    rows=report["rows"]
    thresholds={k:choose_threshold(rows,k) for k in ("instant_direction","history_direction")}
    per_family={}
    for family in FAMILY_SEEDS:
        fam_rows=[r for r in rows if r["family"]==family]
        per_family[family]={
            feature:{"threshold":choose_threshold(fam_rows,feature),
                     "source":operating_point(fam_rows,feature,thresholds[feature])}
            for feature in thresholds
        }
    return {"experiment":"SHAFT-COMP-17","registration_commit":REGISTRATION_COMMIT,
            "calibration_seeds":{family:list(s["calibration"]) for family,s in FAMILY_SEEDS.items()},
            "threshold_algorithm":"numpy.quantile(0.9,method=higher), decision strictly greater",
            "thresholds":thresholds,
            "source_operating_points":{k:operating_point(rows,k,v) for k,v in thresholds.items()},
            "per_family_calibration":per_family,
            "model_hashes":report["model_hashes"],
            "calibration_training_accuracy":report["training_accuracy"],
            "numerical_provenance":"derived only from calibration models before any heldout training/evaluation"}


def uncertainty(rows,permutations=PERMUTATIONS,bootstraps=BOOTSTRAPS):
    rows=[r for r in rows if r["eligible"]]
    if not rows:return {"error":"no eligible pair"}
    y=np.asarray([r["diverged_32"] for r in rows],int)
    new=np.asarray([r["history_direction"] for r in rows],float)
    old=np.asarray([r["instant_direction"] for r in rows],float)
    actual_new,actual_old=auc(new,y),auc(old,y)
    if actual_new is None or actual_old is None:return {"error":"single class"}
    group={}
    for i,r in enumerate(rows):group.setdefault((r["family"],r["seed"],r["depth"]),[]).append(i)
    rng=np.random.default_rng(171717)
    null=[]
    for _ in range(permutations):
        labels=y.copy()
        for ix in group.values():labels[ix]=rng.permutation(labels[ix])
        an,ao=auc(new,labels),auc(old,labels)
        if an is not None and ao is not None:null.append(an-ao)
    center=float(np.mean(null)) if null else None
    delta=actual_new-actual_old
    p=None if center is None else (1+sum(abs(v-center)>=abs(delta-center)-1e-12 for v in null))/(len(null)+1)
    boot=[];diff=[]
    for _ in range(bootstraps):
        indexes=np.concatenate([rng.choice(ix,len(ix),replace=True) for ix in group.values()])
        an,ao=auc(new[indexes],y[indexes]),auc(old[indexes],y[indexes])
        if an is not None and ao is not None:
            boot.append((an,ao));diff.append(an-ao)
    intervals={}
    if boot:
        arr=np.asarray(boot)
        intervals={"history_auc_95":[round(float(x),6) for x in np.quantile(arr[:,0],[.025,.975])],
                   "instant_auc_95":[round(float(x),6) for x in np.quantile(arr[:,1],[.025,.975])],
                   "difference_95":[round(float(x),6) for x in np.quantile(diff,[.025,.975])]}
    return {"history_auc":round(actual_new,6),"instant_auc":round(actual_old,6),
            "difference_auc":round(delta,6),"permutation_null_mean":None if center is None else round(center,6),
            "permutation_p_two_sided":None if p is None else round(p,6),
            "valid_permutations":len(null),"valid_bootstraps":len(boot),"intervals":intervals,
            "note":"pair-level resampling within four heldout models only"}


def run_phase(phase:str,per_goal=PER_GOAL,training_steps=TRAIN_STEPS,calibration=None,
              permutations=PERMUTATIONS,bootstraps=BOOTSTRAPS):
    if phase not in ("calibration","heldout"):raise ValueError("run calibration or heldout separately")
    if phase=="heldout" and calibration is None:raise ValueError("heldout requires frozen calibration thresholds")
    torch.set_num_threads(3)
    torch.use_deterministic_algorithms(True)
    result={"experiment":"SHAFT-COMP-17","phase":phase,
            "registration_commit":REGISTRATION_COMMIT,
            "torch":str(torch.__version__),"numpy":str(np.__version__),
            "goal_count":GOALS,"hidden_dimensions":HIDDEN,
            "training_steps":training_steps,"train_batch":BATCH,"train_sequence_length":TRAIN_LENGTH,
            "probe_training_depths":[4,8,12],"probe_steps":PROBE_STEPS,
            "diagnostic_steps":DIAGNOSTIC_STEPS,"outcome_steps":OUTCOME_STEPS,
            "pairs_per_goal":per_goal,
            "model_hashes":{},"training_accuracy":{},"probe_fits":{},"cells":{},"rows":[],
            "families":{},"thresholds":None}
    for family,seeds in FAMILY_SEEDS.items():
        subset=[]
        for seed in seeds[phase]:
            model,loss=fit_family(seed,family,training_steps)
            p1,p1_loss=train_probe(model,seed)
            p2,p2_loss=fit_history_probe(model,seed)
            result["model_hashes"][str(seed)]={"family":family,"base":model_digest(model),
                                               "one_step_probe":model_digest(p1),
                                               "two_step_probe":model_digest(p2)}
            result["training_accuracy"][str(seed)]={
                "family":family,"last_train_loss":round(loss,7),
                "fresh_accuracy_at_12":round(test_task_accuracy(model,seed,family,12),6),
                "fresh_accuracy_at_32":round(test_task_accuracy(model,seed,family,32),6),
                "one_step_probe_loss":round(p1_loss,7),
                "two_step_probe_loss":round(p2_loss,7)}
            for depth in DEPTHS:
                cell,audit=evaluate_cell(model,p1,p2,seed,family,depth,per_goal)
                result["cells"][f"{family}_{seed}_d{depth}"]={"classification":summarize(cell),"probe_audit":audit}
                subset.extend(cell)
                result["rows"].extend(cell)
        result["families"][family]=summarize(subset)
    result["pooled"]=summarize(result["rows"])
    if phase=="calibration":
        result["calibration_receipt"]=compute_calibration_receipt(result)
        result["thresholds"]=result["calibration_receipt"]["thresholds"]
    else:
        assert calibration["registration_commit"]==REGISTRATION_COMMIT
        result["thresholds"]=calibration["thresholds"]
        for family in FAMILY_SEEDS:
            subset=[r for r in result["rows"] if r["family"]==family]
            result["families"][family]["fixed_operating_points"]={
                feature:operating_point(subset,feature,threshold)
                for feature,threshold in calibration["thresholds"].items()}
            result["families"][family]["adapted_operating_points"]={
                feature:operating_point(subset,feature,calibration["per_family_calibration"][family][feature]["threshold"])
                for feature in calibration["thresholds"]}
        result["pooled"]["fixed_operating_points"]={
            feature:operating_point(result["rows"],feature,threshold)
            for feature,threshold in calibration["thresholds"].items()}
        result["inference"]=uncertainty(result["rows"],permutations,bootstraps)
        signs=[]
        for family,seeds in FAMILY_SEEDS.items():
            for seed in seeds["heldout"]:
                s=summarize([r for r in result["rows"] if r["seed"]==seed])
                v=s["auc"]["history_direction"]
                b=s["auc"]["instant_direction"]
                signs.append({"family":family,"seed":seed,"eligible":s["eligible"],
                              "positives":s["positives"],
                              "difference":None if v is None or b is None else round(v-b,6)})
        result["seed_signs"]=signs
        ci=result["inference"].get("intervals",{}).get("difference_95")
        result["decision"]={
            "prediction_transfer_supported":bool(ci and ci[0]>0 and
                all(result["families"][f]["auc"]["history_direction"] is not None and
                    result["families"][f]["auc"]["instant_direction"] is not None and
                    result["families"][f]["auc"]["history_direction"]>
                    result["families"][f]["auc"]["instant_direction"] for f in FAMILY_SEEDS) and
                sum(x["difference"] is not None and x["difference"]>0 for x in signs)>=3),
            "fixed_fpr_transfer_supported":all(
                result["families"][f]["fixed_operating_points"]["history_direction"]["fpr"] is not None and
                result["families"][f]["fixed_operating_points"]["history_direction"]["fpr"]<=.15
                for f in FAMILY_SEEDS),
        }
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--phase",required=True,choices=("calibration","heldout"))
    p.add_argument("--per-goal",type=int,default=PER_GOAL)
    p.add_argument("--training-steps",type=int,default=TRAIN_STEPS)
    p.add_argument("--calibration-receipt",type=Path)
    p.add_argument("--permutations",type=int,default=PERMUTATIONS)
    p.add_argument("--bootstraps",type=int,default=BOOTSTRAPS)
    p.add_argument("--out",type=Path)
    a=p.parse_args()
    if not 1<=a.per_goal<=64:raise ValueError("per-goal must be 1..64")
    if a.training_steps<1:raise ValueError("training steps must be positive")
    cal=json.loads(a.calibration_receipt.read_text()) if a.calibration_receipt else None
    report=run_phase(a.phase,a.per_goal,a.training_steps,cal,a.permutations,a.bootstraps)
    data=json.dumps(report,sort_keys=True,indent=2)+"\n"
    if a.out:
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(data)
    else:print(data,end="")

if __name__=="__main__":main()

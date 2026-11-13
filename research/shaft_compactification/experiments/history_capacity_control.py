"""Exploratory SHAFT-COMP-17 capacity-matched 16-output duplicate control.

Declared after the registered holdout result, see SHAFT_COMP_17_POSTHOC.md.
Uses the same models, state-pair IDs, eligibility and outcome labels.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import torch
import torch.nn.functional as F
from torch import nn
from research.shaft_compactification.experiments.causal_probe import capture,train_probe,counterfactual
from research.shaft_compactification.experiments.history_aware_transfer import (
    FAMILY_SEEDS,DEPTHS,HIDDEN,TOKENS,PROBE_LR,PROBE_STEPS,
    fit_family,paired_directional_scores,model_digest,
)
from research.shaft_compactification.experiments.observability_prediction import sample_pair,auc


def train_duplicate_probe(model,seed):
    data=capture(model,(4,8,12),48,seed+70000)
    states=torch.cat([data[k][0] for k in (4,8,12)])
    with torch.no_grad():
        truth=(counterfactual(model,states,torch.empty((len(states),0),dtype=torch.long))>=0).float()
        target=truth.repeat(1,4)
    torch.manual_seed(seed+92000)
    probe=nn.Linear(HIDDEN,TOKENS*TOKENS)
    opt=torch.optim.Adam(probe.parameters(),lr=PROBE_LR)
    for _ in range(PROBE_STEPS):
        loss=F.binary_cross_entropy_with_logits(probe(states),target)
        opt.zero_grad();loss.backward();opt.step()
    return probe.eval(),float(loss.detach())


def run(original,phase):
    torch.set_num_threads(3)
    torch.use_deterministic_algorithms(True)
    assert original["experiment"]=="SHAFT-COMP-17" and original["phase"]==phase
    original_rows={(r["family"],r["seed"],r["depth"],r["pair"]):r for r in original["rows"]}
    report={"experiment":"SHAFT-COMP-17 capacity matched post-hoc ablation",
            "status":"exploratory_not_registered_primary",
            "original_phase":phase,
            "new_readout":"16 one-step outputs duplicated four times; linear 16x24; independent initialization",
            "model_hashes":{},"fitting_losses":{},"rows":[]}
    for family,sp in FAMILY_SEEDS.items():
        for seed in sp[phase]:
            model,_=fit_family(seed,family)
            p1,_=train_probe(model,seed)
            duplicate,loss=train_duplicate_probe(model,seed)
            report["model_hashes"][str(seed)]={"base":model_digest(model),
                                               "duplicate_probe":model_digest(duplicate)}
            report["fitting_losses"][str(seed)]=loss
            assert model_digest(model)==original["model_hashes"][str(seed)]["base"]
            for depth in DEPTHS:
                goals,ha,hb,_=sample_pair(model,seed,depth,original["pairs_per_goal"])
                generator=torch.Generator().manual_seed(1700000+seed*1000+depth)
                diagnostic=torch.randint(TOKENS,(len(ha),5),generator=generator)
                values=paired_directional_scores(model,p1,duplicate,ha,hb,diagnostic)
                for i in range(len(ha)):
                    prev=original_rows[(family,seed,depth,i)]
                    assert int(goals[i])==prev["goal_for_audit_only"]
                    assert abs(float(values["instant_direction"][i])-prev["instant_direction"])<1e-5
                    report["rows"].append({"family":family,"seed":seed,"depth":depth,"pair":i,
                        "eligible":prev["eligible"],"diverged_32":prev["diverged_32"],
                        "duplicate_direction":float(values["history_direction"][i]),
                        "one_step_direction":prev["instant_direction"],
                        "history_direction":prev["history_direction"]})
    def scores(rows):
        rows=[r for r in rows if r["eligible"]]
        y=[r["diverged_32"] for r in rows]
        names=("duplicate_direction","one_step_direction","history_direction")
        return {"eligible":len(rows),"positives":sum(y),"auc":{
            k:None if (v:=auc([r[k] for r in rows],y)) is None else round(v,6)
            for k in names}}
    valid=[r for r in report["rows"] if r["eligible"]]
    report["pooled"]=scores(valid)
    report["families"]={name:scores([r for r in valid if r["family"]==name]) for name in FAMILY_SEEDS}
    report["seedwise"]={str(seed):scores([r for r in valid if r["seed"]==seed])
                        for sp in FAMILY_SEEDS.values() for seed in sp[phase]}
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--original",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--phase",choices=("heldout","calibration"),required=True)
    a=p.parse_args()
    result=run(json.loads(a.original.read_text()),a.phase)
    a.out.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print(json.dumps({"pooled":result["pooled"],"families":result["families"]},indent=2))

if __name__=="__main__":main()

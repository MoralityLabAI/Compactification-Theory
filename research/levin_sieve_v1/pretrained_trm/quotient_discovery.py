"""LSV-17: preregistered finite-horizon quotient discovery on frozen recurrence.

This is an empirical Torch probe, not a finite-radius neural certificate.
Splits are puzzle-disjoint and fixed before scoring. Probe training uses
teacher-generated logits as targets. See LSV17_FUTURE_QUOTIENT.md.
"""
from __future__ import annotations
import argparse
import copy
import json
import math
import sys
from pathlib import Path

import torch
from torch import nn

HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0,str(HERE))
SCHEMA="LSV-17/future-behavior-quotient-v1"
DIRECTIONS=("random","null_one","null_all","terminal_gradient","common_mode")
HORIZONS=(1,2,3)

def unit(v:torch.Tensor)->torch.Tensor:
    norm=torch.linalg.vector_norm(v)
    if not bool(torch.isfinite(norm)) or norm.item()<=1e-14:
        raise ValueError("zero or nonfinite intervention direction")
    return v/norm

def null_projection(v:torch.Tensor,gradients:torch.Tensor,atol:float=1e-10)->torch.Tensor:
    """Project onto tangent kernel of the supplied future margin gradients."""
    if gradients.ndim!=2 or gradients.shape[1]!=v.numel():
        raise ValueError("gradient matrix shape mismatch")
    if gradients.numel()==0:
        return unit(v)
    _,s,vt=torch.linalg.svd(gradients.to(torch.float64),full_matrices=False)
    rank=int((s>max(atol,float(s.max())*1e-8)).sum().item()) if len(s) else 0
    basis=vt[:rank]
    residual=v.to(torch.float64)-(v.to(torch.float64)@basis.T)@basis
    return unit(residual)

def goal_history(model,state:torch.Tensor,cell:int,steps:int=3)->torch.Tensor:
    """Logits after each subsequent recurrent block, shape [steps,10]."""
    if state.ndim!=3 or state.shape[0]!=1 or state.shape[1]!=81:
        raise ValueError("expected 1x81xd latent state")
    if not 0<=cell<81 or not 1<=steps<=3:
        raise ValueError("invalid cell or continuation horizon")
    rows=[]
    z=state
    for _ in range(steps):
        z=model.block(z)
        rows.append(model.head(z)[0,cell,:])
    return torch.stack(rows)

def margin_gradients(model,state:torch.Tensor,cell:int):
    z=state.detach().clone().requires_grad_(True)
    history=goal_history(model,z,cell)
    ordering=history.detach().topk(2,dim=1).indices
    grads=[]
    for i,pair in enumerate(ordering):
        gap=history[i,pair[0]]-history[i,pair[1]]
        g=torch.autograd.grad(gap,z,retain_graph=i<2)[0][0,cell]
        grads.append(g.detach())
    return history.detach(),ordering,torch.stack(grads)

def directions_for_case(grads:torch.Tensor,seed:int):
    g=torch.Generator(device="cpu").manual_seed(seed)
    random_vector=unit(torch.randn(grads.shape[-1],generator=g,dtype=torch.float64))
    centered=random_vector-random_vector.mean()
    if torch.linalg.vector_norm(centered)<1e-12:
        raise ValueError("degenerate centered noise")
    return {
        "random":unit(centered),
        "null_one":null_projection(centered,grads[:1]),
        "null_all":null_projection(centered,grads),
        "terminal_gradient":unit(grads[-1]),
        "common_mode":unit(torch.ones_like(centered))
    }

def distance_metrics(clean,changed,ordering,gradients,direction,radius,target):
    clean_p,changed_p=clean.softmax(-1),changed.softmax(-1)
    clean_label,changed_label=clean.argmax(-1),changed.argmax(-1)
    tv=(clean_p-changed_p).abs().sum(-1)/2
    kl=(clean_p*(clean.log_softmax(-1)-changed.log_softmax(-1))).sum(-1)
    responses=[]
    for h,pair in enumerate(ordering):
        top,second=int(pair[0]),int(pair[1])
        actual=float((changed[h,top]-changed[h,second])-(clean[h,top]-clean[h,second]))
        tangent=float(radius*torch.dot(gradients[h],direction))
        responses.append({"horizon":h+1,"finite_margin_delta":actual,
                          "tangent_margin_delta":tangent,
                          "taylor_residual":abs(actual-tangent)})
    return {
        "flips":[bool(x) for x in (changed_label!=clean_label).tolist()],
        "clean_correct":[bool(x==target) for x in clean_label.tolist()],
        "perturbed_correct":[bool(x==target) for x in changed_label.tolist()],
        "tv":[float(x) for x in tv],
        "kl":[max(0.0,float(x)) for x in kl],
        "fixed_pair_response":responses
    }

def pre_ln2_gauge_control(model,state,cell:int,radius:float):
    """Independent exact-real symmetry control, measured in FP64 at pre-LN2."""
    block=model.block
    first=block.ln1(state+block.tok_mix(state.transpose(1,2)).transpose(1,2))
    pre=first+block.chan_mix(first)
    clean=block.ln2(pre)
    d=int(pre.shape[-1])
    v=torch.arange(d,dtype=pre.dtype,device=pre.device)
    v=unit(v-v.mean())
    shift=torch.zeros_like(pre)
    shift[0,cell]=radius/math.sqrt(d)
    transverse=torch.zeros_like(pre)
    transverse[0,cell]=radius*v
    changed_gauge=block.ln2(pre+shift)
    changed_transverse=block.ln2(pre+transverse)
    def future(z):
        out=[model.head(z)[0,cell]]
        for _ in range(2):
            z=block(z)
            out.append(model.head(z)[0,cell])
        return torch.stack(out)
    f0=future(clean)
    fg,ft=future(changed_gauge),future(changed_transverse)
    return {
        "radius":radius,
        "gauge_bus_max_hidden_abs":float((clean-changed_gauge).detach().abs().max()),
        "gauge_future_max_logit_abs":float((f0-fg).detach().abs().max()),
        "transverse_future_max_logit_abs":float((f0-ft).detach().abs().max()),
        "gauge_flips":[bool(x) for x in (fg.argmax(-1)!=f0.argmax(-1)).tolist()],
        "transverse_flips":[bool(x) for x in (ft.argmax(-1)!=f0.argmax(-1)).tolist()],
        "scope":"FP64 at pre-LN2; hardware arithmetic is only measured"
    }

def partition(case_index:int):
    return ("train","train","validation","test")[case_index%4]

def train_probes(features:torch.Tensor,targets:torch.Tensor,splits:list[str],*,
                 seed:int=1704,max_epochs:int=180):
    """Linear versus 16-wide tanh quotient on identical puzzle-disjoint samples."""
    if features.ndim!=2 or targets.ndim!=2 or targets.shape[1]!=30:
        raise ValueError("invalid probe data")
    groups={label:torch.tensor([i for i,s in enumerate(splits) if s==label])
            for label in ("train","validation","test")}
    if min(len(v) for v in groups.values())==0:
        raise ValueError("all three puzzle-disjoint partitions required")
    torch.set_num_threads(1)
    features=features.detach().float().cpu()
    targets=targets.detach().float().cpu()
    xm=features[groups["train"]].mean(0)
    xs=features[groups["train"]].std(0).clamp_min(0.001)
    ym=targets[groups["train"]].mean(0)
    ys=targets[groups["train"]].std(0).clamp_min(0.001)
    X=((features-xm)/xs).clamp(-50,50)
    Y=((targets-ym)/ys).clamp(-50,50)
    results={}
    for i,architecture in enumerate(("linear","nonlinear")):
        torch.manual_seed(seed+i)
        model=(nn.Linear(X.shape[1],Y.shape[1]) if architecture=="linear"
               else nn.Sequential(nn.Linear(X.shape[1],16),nn.Tanh(),
                                  nn.Linear(16,Y.shape[1])))
        count=sum(p.numel() for p in model.parameters())
        opt=torch.optim.AdamW(model.parameters(),lr=0.01,weight_decay=0.001)
        best,best_epoch,best_state,patience=float("inf"),0,None,0
        for epoch in range(max_epochs):
            model.train()
            opt.zero_grad(set_to_none=True)
            err=(model(X[groups["train"]])-Y[groups["train"]]).square().mean()
            err.backward()
            opt.step()
            model.eval()
            with torch.no_grad():
                val=float((model(X[groups["validation"]])-Y[groups["validation"]]).square().mean())
            if val<best-1e-6:
                best,best_epoch=val,epoch+1
                best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
                patience=0
            else:
                patience+=1
                if patience>=25:
                    break
        model.load_state_dict(best_state)
        with torch.no_grad():
            prediction=model(X[groups["test"]])*ys+ym
            truth=targets[groups["test"]]
            mse=float((prediction-truth).square().mean())
            agreement=[]
            for h in range(3):
                pred_labels=prediction[:,h*10:(h+1)*10].argmax(-1)
                truth_labels=truth[:,h*10:(h+1)*10].argmax(-1)
                agreement.append(float((pred_labels==truth_labels).float().mean()))
        results[architecture]={"trainable_parameters":count,"best_epoch":best_epoch,
               "validation_normalized_mse":best,
               "test_logit_mse":mse,"test_teacher_label_agreement":agreement}
    return {"samples":{k:len(v) for k,v in groups.items()},"models":results,
            "meaning":"held-out supervised teacher prediction, not replacement of recurrent computation"}

def run(model,inputs,answers,checkpoint_sha:str,roster_sha:str,*,
        max_cases:int=16,radii=(0.2,1.0,3.0),probe_epochs:int=180):
    if type(max_cases) is not int or max_cases<8 or max_cases>len(inputs):
        raise ValueError("max_cases must be an integer 8..roster length")
    if any(not 0<float(r)<=10 for r in radii):
        raise ValueError("radii must be finite and positive and at most 10")
    torch.set_num_threads(1)
    # Match LSV-09/13 by running step 1 with the original checkpoint dtype.
    raw_model=model.eval()
    with torch.no_grad():
        stage1=raw_model.reason_states(inputs[:max_cases])[1].detach()
    model=copy.deepcopy(raw_model).double().eval()
    rows,gauge_rows=[],[]
    all_x,all_y,all_s=[],[],[]
    for i in range(max_cases):
        cell=int((inputs[i]==0).long().argmax())
        if int(inputs[i,cell])!=0:
            raise ValueError("first missing Sudoku cell not found")
        z1=stage1[i:i+1].double().detach()
        clean,top2,grads=margin_gradients(model,z1,cell)
        dirs=directions_for_case(grads,170400+i)
        all_x.append(z1[0,cell].clone())
        all_y.append(clean.reshape(-1).clone())
        all_s.append(partition(i))
        with torch.no_grad():
            gauge_rows.append({"case_index":i,**pre_ln2_gauge_control(model,z1,cell,1.0)})
            for name in DIRECTIONS:
                direction=dirs[name]
                for radius in radii:
                    edit=z1.clone()
                    edit[0,cell]+=float(radius)*direction
                    changed=goal_history(model,edit,cell)
                    metrics=distance_metrics(clean,changed,top2,grads,direction,float(radius),
                                             int(answers[i,cell]))
                    rows.append({"case_index":i,"partition":partition(i),"cell":cell,
                                 "method":name,"radius":float(radius),
                                 "direction_l2":float(torch.linalg.vector_norm(direction)),
                                 **metrics})
                    all_x.append(edit[0,cell].clone())
                    all_y.append(changed.reshape(-1).clone())
                    all_s.append(partition(i))
    probe=train_probes(torch.stack(all_x),torch.stack(all_y),all_s,max_epochs=probe_epochs)
    aggregates={}
    for name in DIRECTIONS:
        for radius in radii:
            relevant=[r for r in rows if r["method"]==name and
                      r["radius"]==float(radius) and r["partition"]=="test"]
            aggregates[f"{name}/{float(radius)}"]={
                "heldout_puzzles":len(relevant),
                "horizon_flip_fractions":[sum(r["flips"][h] for r in relevant)/len(relevant) for h in range(3)],
                "horizon_mean_tv":[sum(r["tv"][h] for r in relevant)/len(relevant) for h in range(3)],
                "horizon_mean_taylor_residual":[sum(r["fixed_pair_response"][h]["taylor_residual"] for r in relevant)/len(relevant) for h in range(3)],
                "early_stable_late_flip_count":sum((not r["flips"][0]) and r["flips"][2] for r in relevant),
                "clean_correct_final_count":sum(r["clean_correct"][2] for r in relevant),
                "perturbed_correct_final_count":sum(r["perturbed_correct"][2] for r in relevant)
            }
    return {
        "schema":SCHEMA,"status":"empirical_source_locked_continuation_pilot",
        "checkpoint_sha256":checkpoint_sha,"roster_sha256":roster_sha,
        "max_cases":max_cases,"radii":[float(r) for r in radii],
        "horizons":list(HORIZONS),"directions":list(DIRECTIONS),
        "partition_rule":"i mod 4: train={0,1}, validation={2}, test={3}",
        "intervention_site":"first missing cell post-first-block state",
        "exact_symmetry_control_site":"pre-LN2 of second block",
        "probe":probe,"heldout_aggregates":aggregates,
        "gauge_bus_max_logit_abs":max(r["gauge_future_max_logit_abs"] for r in gauge_rows),
        "transverse_bus_max_logit_abs":max(r["transverse_future_max_logit_abs"] for r in gauge_rows),
        "source":"external TRM-inspired 44k Sudoku model, not official TRM or Qwen",
        "claim_boundary":"no finite-region robustness certificate, Jacobian-null is pointwise",
        "case_rows":rows,"gauge_rows":gauge_rows
    }

def main():
    from run_checkpoint import locked_bytes,load_pretrained,prepare_roster
    p=argparse.ArgumentParser()
    p.add_argument("--lock",type=Path,default=HERE/"source_lock.json")
    p.add_argument("--checkpoint",type=Path)
    p.add_argument("--fetch",action="store_true")
    p.add_argument("--max-cases",type=int,default=16)
    p.add_argument("--radii",default="0.2,1.0,3.0")
    p.add_argument("--probe-epochs",type=int,default=180)
    p.add_argument("--out",type=Path,default=Path("runs/lsv17/receipt.json"))
    a=p.parse_args()
    lock=json.loads(a.lock.read_text())
    if lock.get("schema")!="LSV-05/pretrained-trm-source-lock-v1":
        raise ValueError("source lock mismatch")
    data,sha=locked_bytes(lock,a.checkpoint,a.fetch)
    model=load_pretrained(lock,data)
    (x,y,_),roster=prepare_roster(lock)
    receipt=run(model,x,y,sha,roster,max_cases=a.max_cases,
                radii=tuple(float(v) for v in a.radii.split(",")),
                probe_epochs=a.probe_epochs)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(receipt,sort_keys=True,indent=2)+"\n")
    summary={k:v for k,v in receipt.items() if k not in ("case_rows","gauge_rows")}
    print("LSV17",json.dumps(summary,sort_keys=True))

if __name__=="__main__":
    main()

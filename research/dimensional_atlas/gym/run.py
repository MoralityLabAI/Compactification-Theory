"""SCM-15 executable synthetic training loop, or pinned EvalMath adapter.

Benchmark categories remain separate: exact verifier future correctness,
resource costs, SAE reconstruction, VPD ablations and diagnostic geometry.
The official EvalMath test roster is first opened after training is frozen.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from hashlib import sha256
import io
import json
import math
from pathlib import Path
import random
import time

import torch
import torch.nn.functional as F
from .model import (LoopedMathTransformer, set_base_trainable, set_adapter_trainable,
                    set_sae_trainable, atlas_probe)
from .tasks import (PROGRAMS, make_toy, make_evalmath_case, encode, check_candidate,
                    verify_order, select_frontier, seal_new_evals, load_evalmath)


def model_digest(model) -> str:
    buffer=io.BytesIO()
    torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},buffer)
    return sha256(buffer.getvalue()).hexdigest()


def _make_pool(n:int, seed_offset:int, *, backend:str, oracle=None, split:str):
    family_cycle=("iid","sparse") if backend=="toy" else ("signed","product","sparse")
    tasks=[]; seen=set()
    for i in range(n*4):
        family=family_cycle[i%len(family_cycle)]
        seed=seed_offset+i
        item=(make_toy(seed,case_id=f"{split}:{seed}",family=family) if backend=="toy"
              else make_evalmath_case(oracle,seed,case_id=f"{split}:{seed}",family=family))
        if item.signature not in seen:
            tasks.append(item);seen.add(item.signature)
        if len(tasks)>=n:break
    if len(tasks)<n: raise ValueError("insufficient unique task signatures")
    return tasks


def tensorize(tasks,device='cpu'):
    inputs=torch.stack([encode(t) for t in tasks]).to(device)
    labels=torch.tensor([PROGRAMS.index(t.source_graph) for t in tasks],dtype=torch.long,device=device)
    return inputs,labels


def optimizer_steps(model, train, *, stage:str, steps:int, batch:int,
                    seed:int, lr:float, penalty:float=1e-3):
    if steps<=0:return {"steps":0,"last_loss":None}
    set_base_trainable(model,stage=="base")
    set_adapter_trainable(model,stage=="vpd")
    set_sae_trainable(model,stage=="sae")
    x,y=tensorize(train)
    rng=torch.Generator().manual_seed(seed)
    if stage=="base":params=list(model.embed.parameters())+list(model.core.parameters())+list(model.head.parameters())+list(model.head_norm.parameters())+[model.positions]
    elif stage=="sae":params=list(model.sae.parameters())
    elif stage=="vpd":params=list(model.vpd.parameters())
    else:raise ValueError(stage)
    optimizer=torch.optim.AdamW(params,lr=lr,weight_decay=0.01)
    model.train()
    last=0.
    for _ in range(steps):
        ids=torch.randint(len(train),(batch,),generator=rng)
        xb,yb=x[ids],y[ids]
        if stage=="sae":
            with torch.no_grad():
                states=model(xb,use_vpd=False).states
                target=torch.cat([F.layer_norm(s,(model.hidden,)).reshape(-1,model.hidden)
                                  for s in states],dim=0).detach()
            reconstruction, codes=model.sae(target)
            # MSE and small activation sparsity penalty; TopK already limits support.
            loss=F.mse_loss(reconstruction,target)+0.0001*codes.abs().mean()
        else:
            result=model(xb,use_vpd=(stage=="vpd"))
            loss=F.cross_entropy(result.logits,yb)
            if stage=="vpd":loss=loss+penalty*result.correction_energy
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params,max_norm=2.0)
        optimizer.step()
        last=float(loss.detach())
    model.eval()
    return {"steps":steps,"last_loss":last}


def inspect_discovery(model,tasks,*,oracle=None):
    # One top-choice test per task. The independent demo-prefix judge gives
    # first fault; future/gold fields never influence the mining proxy.
    model.eval()
    out=[]
    for task in tasks:
        with torch.no_grad():
            prob=model(encode(task).unsqueeze(0),use_vpd=True).logits.softmax(dim=-1)[0]
        idx=int(prob.argmax());q=check_candidate(task,PROGRAMS[idx],oracle)
        errors=[i for i in q["first_faults"] if i is not None]
        out.append({"split":"discovery","case_id":task.case_id,"family":task.family,
                    "topology":task.topology,"graph":list(task.source_graph),
                    "predicted_graph":list(PROGRAMS[idx]),
                    "confidence":float(prob[idx]),"first_fault":min(errors) if errors else None,
                    "public_sha256":task.signature})
    return out


def generate_replay(peaks, *, backend:str,oracle=None,seed_offset:int=600000,per_peak:int=4,
                    forbidden:set[str]):
    generated=[];seen=set(forbidden)
    for idx,peak in enumerate(peaks):
        for j in range(per_peak):
            seed=seed_offset+1000*idx+j
            task=(make_toy(seed,case_id=f"replay:{idx}:{j}",family=peak["family"],
                           topology=peak["topology"],graph=tuple(peak["graph"])) if backend=="toy" else
                  make_evalmath_case(oracle,seed,case_id=f"replay:{idx}:{j}",family=peak["family"],
                                     topology=peak["topology"],graph=tuple(peak["graph"])))
            if task.signature in seen:raise ValueError("replay signature collision")
            generated.append(task); seen.add(task.signature)
    return generated


def score(model,tasks,*,oracle=None,method='base',budget=12,visit_order=None,code_override=None):
    rows=[]; model.eval()
    for task in tasks:
        with torch.no_grad():
            logits=model(encode(task).unsqueeze(0),use_vpd=(method!='base'),
                         visit_order=visit_order,code_override=code_override).logits[0]
            order=logits.argsort(descending=True).tolist()
        # The selected candidate is independent of future gold by construction.
        grade=verify_order(task,order,budget=budget,oracle=oracle)
        rows.append({"id":task.case_id,"family":task.family,"signature":task.signature,
                     "queries":grade["queries"],"accepted":grade["accepted"],
                     "future_correct":grade["future_correct"],
                     "false_future_acceptance":grade["false_future_acceptance"]})
    groups=defaultdict(list)
    for row in rows:groups[row["family"]].append(row)
    def aggregate(items):
        n=len(items)
        return {"n":n,"future_accuracy":sum(x["future_correct"] for x in items)/n,
                "prefix_acceptance":sum(x["accepted"] for x in items)/n,
                "false_future_acceptances":sum(x["false_future_acceptance"] for x in items),
                "mean_exact_queries":sum(x["queries"] for x in items)/n}
    return {"overall":aggregate(rows),"per_family":{k:aggregate(v) for k,v in sorted(groups.items())},
            "records":rows}


def run(args):
    if args.backend=="evalmath":
        if args.evalmath_source is None: raise ValueError("--evalmath-source required")
        oracle=load_evalmath(args.evalmath_source)
    else:oracle=None
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    if args.output.exists(): raise FileExistsError("refusing to overwrite existing receipt")
    train=_make_pool(args.train_count,110000,backend=args.backend,oracle=oracle,split="train")
    discovery=_make_pool(args.discovery_count,220000,backend=args.backend,oracle=oracle,split="discovery")
    known={t.signature for t in train+discovery}
    if len(known)!=len(train)+len(discovery):raise AssertionError("training/discovery overlap")
    model=LoopedMathTransformer(hidden=args.hidden,visits=args.visits,sae_width=args.sae_width,
                                 sae_k=args.sae_k,vpd_rank=args.vpd_rank)
    started=time.monotonic()
    phases={}
    phases['base']=optimizer_steps(model,train,stage="base",steps=args.steps,batch=args.batch,seed=args.seed+1,lr=args.lr)
    phases['sae']=optimizer_steps(model,train,stage="sae",steps=args.sae_steps,batch=args.batch,seed=args.seed+2,lr=args.lr)
    phases['vpd']=optimizer_steps(model,train,stage="vpd",steps=args.vpd_steps,batch=args.batch,seed=args.seed+3,lr=args.lr)
    discovery_rows=inspect_discovery(model,discovery,oracle=oracle)
    peaks=select_frontier(discovery_rows,limit=args.peaks)
    replay=generate_replay(peaks,backend=args.backend,oracle=oracle,forbidden=known)
    phases['frontier_replay']=optimizer_steps(model,train+replay,stage="vpd",steps=args.replay_steps,
                                              batch=args.batch,seed=args.seed+4,lr=args.lr)
    all_training_signatures=known|{t.signature for t in replay}
    # Freeze all models and candidate generation decisions BEFORE loading readout.
    checkpoint_sha=model_digest(model)
    adapter_sha=sha256(str([(k,v.detach().cpu().tolist()) for k,v in model.vpd.state_dict().items()]).encode()).hexdigest()
    if args.backend=="toy":
        # Novel source generator family, never used for training/discovery.
        readout=[make_toy(330000+i,case_id=f"readout:{i}",family="periodic") for i in range(args.readout_count)]
    else:
        rosters=oracle.roster(include_test=True)
        readout=[] # populated below, without exposing future to train
        from .tasks import MathTask
        for key,items in rosters.items():
            if key=="calibration": continue
            for case in items:
                graph=tuple(case.source_graph) if case.source_graph is not None else PROGRAMS[0]
                readout.append(MathTask(case.identifier,case.family,case.topology,
                      tuple((tuple(x[:13]),tuple(y[:13])) for x,y in case.demos),
                      tuple(case.query[:13]),tuple(case.future),graph,"evalmath_arg01",case))
    if any(t.signature in all_training_signatures for t in readout):
        raise ValueError("readout signature overlaps with feedback/training")
    if len({t.signature for t in readout}) != len(readout):
        # EvalMath's deliberate indistinguishable-future collision pairs share
        # visible signatures: group rather than treating them as independent.
        if args.backend!="evalmath":raise ValueError("duplicate readout signatures")
    report={}
    report['tied_base']=score(model,readout,oracle=oracle,method="base",budget=args.budget)
    report['sae_atvpd']=score(model,readout,oracle=oracle,method="vpd",budget=args.budget)
    report['shuffled_visit_gates']=score(model,readout,oracle=oracle,method="vpd",budget=args.budget,
                                         visit_order=tuple(reversed(range(args.visits))))
    report['zero_sae_codes']=score(model,readout,oracle=oracle,method="vpd",budget=args.budget,
                                    code_override="zero")
    assert model_digest(model)==checkpoint_sha, "evaluation modified the frozen model"
    new_public,new_gold=seal_new_evals(peaks,oracle=oracle,per_peak=2,forbidden=all_training_signatures|{t.signature for t in readout})
    # Write gold to separate file; receipt carries only its SHA-256.
    args.output.parent.mkdir(parents=True,exist_ok=True)
    gold_path=args.output.with_suffix('.gold.private.json')
    public_path=args.output.with_suffix('.generated.public.json')
    for path in (gold_path,public_path):
        if path.exists():raise FileExistsError(f"refusing to overwrite: {path}")
    gold_payload=json.dumps(new_gold,sort_keys=True,indent=2)+'\n'
    public_payload=json.dumps(new_public,sort_keys=True,indent=2)+'\n'
    gold_path.write_text(gold_payload,encoding='utf-8')
    public_path.write_text(public_payload,encoding='utf-8')
    test_counts={k:v['overall'] for k,v in report.items()}
    receipt={"schema":"scm15.atlas_vpd.v0","status":"executed_synthetic_gym" if args.backend=="toy" else "executed_pinned_evalmath_gym",
             "backend":args.backend,"seed":args.seed,"train_count":len(train),"discovery_count":len(discovery),
             "frontier_peak_count":len(peaks),"new_replay_count":len(replay),"new_evaluation_count":len(new_public),
             "model_sha256":checkpoint_sha,"adapter_sha256":adapter_sha,"math_source_verified":args.backend=="evalmath",
             "phases":phases,"results":test_counts,"by_family":{k:v['per_family'] for k,v in report.items()},
             "frontier_peaks":[{k:r[k] for k in ('case_id','family','topology','graph','first_fault','frontier_proxy')} for r in peaks],
             "generated_public_sha256":sha256(public_payload.encode()).hexdigest(),
             "generated_gold_sha256":sha256(gold_payload.encode()).hexdigest(),
             "atlas_diagnostic":atlas_probe(model,discovery[0]),
             "measured_training_seconds":round(time.monotonic()-started,2),
             "provenance_warning":"finite-difference geometry is diagnostic only; frontier tuned on discovery; exact test future used only after freeze"}
    args.output.write_text(json.dumps(receipt,sort_keys=True,indent=2)+'\n',encoding='utf-8')
    return receipt


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend',choices=('toy','evalmath'),default='toy')
    parser.add_argument('--evalmath-source',type=Path)
    parser.add_argument('--output',type=Path,default=Path('/tmp/scm15_atlas_vpd.json'))
    parser.add_argument('--seed',type=int,default=15)
    parser.add_argument('--train-count',type=int,default=160)
    parser.add_argument('--discovery-count',type=int,default=40)
    parser.add_argument('--readout-count',type=int,default=48)
    parser.add_argument('--steps',type=int,default=40)
    parser.add_argument('--sae-steps',type=int,default=25)
    parser.add_argument('--vpd-steps',type=int,default=45)
    parser.add_argument('--replay-steps',type=int,default=20)
    parser.add_argument('--batch',type=int,default=24)
    parser.add_argument('--lr',type=float,default=0.002)
    parser.add_argument('--hidden',type=int,default=32)
    parser.add_argument('--visits',type=int,default=3)
    parser.add_argument('--sae-width',type=int,default=64)
    parser.add_argument('--sae-k',type=int,default=6)
    parser.add_argument('--vpd-rank',type=int,default=4)
    parser.add_argument('--budget',type=int,default=12)
    parser.add_argument('--peaks',type=int,default=4)
    args=parser.parse_args(argv)
    if min(args.steps,args.sae_steps,args.vpd_steps,args.replay_steps)<0:parser.error('steps must be nonnegative')
    if args.budget<1 or args.train_count<4 or args.discovery_count<4 or args.readout_count<2:parser.error('invalid run size')
    result=run(args)
    print(json.dumps({"receipt":str(args.output),"status":result['status'],"results":result['results'],
                      "new_evals":result['new_evaluation_count'],"frontier_peaks":result['frontier_peak_count']},indent=2))

if __name__=='__main__':main()

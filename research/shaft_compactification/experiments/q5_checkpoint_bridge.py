"""Optional SHAFT-COMP-13 Q5 bridge for verified local LoopedTransformers weights.

Before measuring, preflight requires all twelve original Q5 TCA receipts and
their SHA-256-matched checkpoint files. Full recurrent state includes *all*
token positions. Random projected-rank is a sketch, never the full-state rank.
No downloading, remote access, checkpoint writing, or agent persistence.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch

REFPATH=Path("research/shaft_compactification/results/q5_registered_reference.json")
TOY="experiments/recurrence_monitorability_v0/loopscope_toy_q5"
TCA="experiments/recurrence_monitorability_v0/loopscope_toy_q5_tca"
THRESHOLD=1e-3
HORIZON=5
SKETCH=16
ITEMS=4

def sha256_file(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()

def load_ref(path=REFPATH):
    r=json.loads(Path(path).read_text())
    assert r["n_valid"]==12 and r["correct"]==9 and r["n_extrap"]==8
    assert sum((x["f1_f8"]<.05)==x["extrap"] for x in r["rows"])==9
    return r

def preflight(root,ref):
    root=Path(root)
    report=[]
    for row in ref["rows"]:
        arm=row["arm"]
        artifact=root/TCA/(arm+".json")
        checkpoint=root/TOY/"ckpt"/(arm+".final.pt")
        status={"arm":arm,"tca_path":str(artifact),"checkpoint_path":str(checkpoint),
                "tca_exists":artifact.exists(),"checkpoint_exists":checkpoint.exists()}
        if artifact.exists():
            receipt=json.loads(artifact.read_text())
            status["expected_sha256"]=receipt["checkpoint_sha256"]
            if checkpoint.exists():
                status["actual_sha256"]=sha256_file(checkpoint)
                status["sha256_matches"]=(status["actual_sha256"]==status["expected_sha256"])
        report.append(status)
    ready=all(x.get("sha256_matches") for x in report)
    return {"status":"ready" if ready else "checkpoint_incomplete_or_hash_mismatch",
            "ready_count":sum(bool(x.get("sha256_matches")) for x in report),
            "expected_count":len(report),"checkpoint_status":report}

def prompt_embeds(model,tokens):
    n=model.n_nodes
    edges=tokens[:,:2*n].reshape(tokens.shape[0],n,2)
    h=torch.cat([model.tok(edges[...,0])+model.dst(edges[...,1]),
                 model.tok(tokens[:,2*n:])],dim=1)
    return h+model.pos(torch.arange(h.shape[1],device=tokens.device))

def transition(model,h,e):
    h=model.inject(h)+e
    for blk in model.core:h=blk(h)
    return h

def sketch_rank(model,e,starting_state,item_seed,k=SKETCH,horizon=HORIZON):
    h=starting_state.detach()
    state_dim=h.numel()
    gen=torch.Generator().manual_seed(item_seed)
    rand=torch.randn((state_dim,k),generator=gen)
    directions=torch.linalg.qr(rand,mode="reduced").Q.to(device=h.device,dtype=h.dtype)
    def observed(x):
        outputs=[]
        for t in range(horizon+1):
            logits=model.readout(x[:,-1])
            outputs.append(logits-logits.mean(dim=-1,keepdim=True))
            if t<horizon:x=transition(model,x,e)
        return torch.cat(outputs,dim=-1).reshape(-1)
    deriv=[]
    for i in range(k):
        vector=directions[:,i].reshape(h.shape)
        # Some fused attention kernels lack forward AD. Use math SDPA if needed.
        _,jvp=torch.func.jvp(observed,(h,),(vector,))
        deriv.append(jvp.detach().double())
    O=torch.stack(deriv,dim=1)
    sv=torch.linalg.svdvals(O)
    return {"rank_relative_1e3":int((sv>sv[0]*THRESHOLD).sum()),
            "rank_relative_1e5":int((sv>sv[0]*1e-5).sum()),
            "min_to_max_sv_ratio":float(sv[-1]/sv[0]) if float(sv[0])>0 else 0.,
            "state_dimensions":state_dim,"sketch_dimensions":k,
            "observation_dimensions":int(O.shape[0]),"horizon":horizon}

def avg_rank(data):
    x=np.asarray(data,float)
    order=np.argsort(x,kind="stable")
    rank=np.empty(len(x),float)
    i=0
    while i<len(x):
        j=i+1
        while j<len(x) and x[order[j]]==x[order[i]]:j+=1
        rank[order[i:j]]=(i+1+j)/2
        i=j
    return rank

def spearman(x,y):
    a,b=avg_rank(x),avg_rank(y)
    if a.std()==0 or b.std()==0:return None
    return float(np.corrcoef(a,b)[0,1])

def auc(scores,labels):
    x=np.asarray(scores,float);y=np.asarray(labels,int)
    n1=int(y.sum());n0=len(y)-n1
    if not n1 or not n0:return None
    return float((avg_rank(x)[y==1].sum()-n1*(n1+1)/2)/(n1*n0))

def run(root,ref,device="cpu",items=ITEMS,sketch=SKETCH):
    check=preflight(root,ref)
    if check["status"]!="ready":
        raise FileNotFoundError(f"Q5 checkpoints/receipts missing or hash mismatch: {check['ready_count']}/12; use --preflight")
    sys.path.insert(0,str(Path(root).resolve()))
    from research_gym.scripts import loopscope_toy as L
    torch.set_num_threads(3)
    rows=[]
    for index,row in enumerate(ref["rows"]):
        path=Path(root)/TOY/"ckpt"/(row["arm"]+".final.pt")
        ck=torch.load(path,map_location="cpu",weights_only=True)
        model=L.build_model(L.model_args_from(ck["cfg"]))
        model.load_state_dict(ck["model"])
        model.to(device).eval().requires_grad_(False)
        values=[]
        for n in range(items):
            tokens,_=L.make_batch(model.n_nodes,1,900013+index*100+n,8)
            tokens=tokens.to(device)
            with torch.no_grad():
                e=prompt_embeds(model,tokens)
                h=transition(model,torch.zeros_like(e),e)
            values.append(sketch_rank(model,e,h,2200013+index*100+n,sketch))
        rows.append({"arm":row["arm"],"f1_f8":row["f1_f8"],
                     "heldout_accuracy":row["heldout_accuracy"],"extrap":row["extrap"],
                     "rank_relative_1e3_median":float(np.median([v["rank_relative_1e3"] for v in values])),
                     "rank_relative_1e5_median":float(np.median([v["rank_relative_1e5"] for v in values])),
                     "min_to_max_sv_ratio_median":float(np.median([v["min_to_max_sv_ratio"] for v in values])),
                     "items":items,"sketch_dimensions":sketch,"details":values})
    score=[r["rank_relative_1e3_median"] for r in rows]
    heldout=[r["heldout_accuracy"] for r in rows]
    return {"status":"Q5_checkpoint_sketch_completed",
            "registered_Q5":{"n":12,"correct":9,"fixed_rule":"F1/F8 < 0.05","positive_count":8},
            "sketch":{"k":sketch,"L":HORIZON,"relative_rank_threshold":THRESHOLD,
                      "state_site":"all token positions after loop 1",
                      "readout":"centered answer-position class logits",
                      "note":"projected rank in k dimensions, not full-state observability"},
            "rank_vs_heldout_spearman":spearman(score,heldout),
            "rank_vs_extrap_auroc":auc(score,[r["extrap"] for r in rows]),"rows":rows}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--looped-repo",type=Path,required=True)
    p.add_argument("--reference",type=Path,default=REFPATH)
    p.add_argument("--device",default="cpu")
    p.add_argument("--items",type=int,default=ITEMS)
    p.add_argument("--sketch",type=int,default=SKETCH)
    p.add_argument("--preflight",action="store_true")
    p.add_argument("--out",type=Path)
    a=p.parse_args()
    if a.items<1 or not 1<=a.sketch<=64:raise ValueError("items>=1 and 1<=sketch<=64 required")
    reference=load_ref(a.reference)
    data=preflight(a.looped_repo,reference) if a.preflight else run(a.looped_repo,reference,a.device,a.items,a.sketch)
    payload=json.dumps(data,indent=2,sort_keys=True)+"\n"
    if a.out:
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(payload)
    else:print(payload,end="")
if __name__=="__main__":main()

"""CPU-only recurrent history encoder; no approximate class is a certificate."""
import argparse
from fractions import Fraction
from pathlib import Path
import time
import json

import numpy as np

from .core import digest, require, write_json
from .program_student import ACTIONS
from .history_memory import (SPLITS,COUNTS,TRIPLETS,CTX,EVENT,
                             specs,context,root_data,verify)

WIDTH=24
EPOCHS=18
BATCH=32
SEEDS=(311,719)
ARMS=(("history_pair",False,2.0),("history_no_pair",False,0.0),
      ("bag_pair",True,2.0),("bag_no_pair",True,0.0))


def arrays(roots,bag=False):
    contexts=[]
    histories=[]
    belief=[]
    action=[]
    values=[]
    risk=[]
    trips=[]
    trip_root=[]
    for rid,public,ref in roots:
        indices={}
        for key,x in public["records"].items():
            indices[key]=len(contexts)
            contexts.append(context(public["spec"]))
            events=[(ACTIONS.index(a),0 if bit is None else bit+1) for a,bit in x["history"]]
            histories.append(sorted(events) if bag else events)
            y=ref["records"][key]
            b=[float(Fraction(v)) for v in y["belief"]]
            belief.append(b+[0.]*(32-len(b)))
            action.append(ACTIONS.index(y["action"]))
            values.append(float(Fraction(y["value"]))/40)
            risk.append(float(Fraction(y["risk"]))/4)
        for t in ref["triplets"]:
            trips.append([indices[t[k]] for k in ("anchor","positive","negative")])
            trip_root.append(rid)
    n=len(contexts)
    T=max(1,max(map(len,histories)))
    events=np.zeros((n,T,EVENT),dtype=np.float32)
    mask=np.zeros((n,T,1),dtype=np.float32)
    for i,h in enumerate(histories):
        for j,(a,b) in enumerate(h):
            events[i,j,a]=1
            events[i,j,len(ACTIONS)+b]=1
            mask[i,j]=1
    return {"x":np.asarray(contexts,dtype=np.float32),"e":events,"m":mask,
            "belief":np.asarray(belief,dtype=np.float32),"action":np.asarray(action),
            "reg":np.asarray([values,risk],dtype=np.float32).T,
            "trip":np.asarray(trips),"roots":trip_root}


def softmax(x):
    y=np.exp(x-np.max(x,axis=1,keepdims=True))
    return y/y.sum(axis=1,keepdims=True)


class RNN:
    def __init__(self,seed):
        rng=np.random.default_rng(seed)
        w=lambda n,m: rng.normal(0,1/np.sqrt(n),(n,m)).astype(np.float32)
        z=lambda n: np.zeros(n,dtype=np.float32)
        self.p={"init":w(CTX,WIDTH),"ib":z(WIDTH),
                "event":w(EVENT,WIDTH),"rec":w(WIDTH,WIDTH),"hb":z(WIDTH),
                "out":w(WIDTH,44),"ob":z(44)}
        self.v={k:np.zeros_like(v) for k,v in self.p.items()}
        self.s={k:np.zeros_like(v) for k,v in self.p.items()}
        self.step=0

    def forward(self,d,ids):
        p=self.p
        x,e,m=d["x"][ids],d["e"][ids],d["m"][ids]
        h0=np.tanh(x@p["init"]+p["ib"])
        h=h0
        trace=[]
        for t in range(e.shape[1]):
            mt=m[:,t]
            et=e[:,t]
            c=np.tanh(h@p["rec"]+et@p["event"]+p["hb"])
            trace.append((h,c,et,mt))
            h=mt*c+(1-mt)*h
        o=h@p["out"]+p["ob"]
        return h,o,(x,h0,trace)

    def train_batch(self,d,trip_ids,pair_weight):
        p=self.p
        ids=d["trip"][trip_ids].reshape(-1)
        n=len(ids)
        h,o,(x,h0,trace)=self.forward(d,ids)
        pb=softmax(o[:,:32])
        pa=softmax(o[:,32:42])
        b=d["belief"][ids]
        a=d["action"][ids]
        y=d["reg"][ids]
        gb=(pb-b)/n
        ga=pa.copy()
        ga[np.arange(n),a]-=1
        ga/=n
        gr=2*(o[:,42:]-y)/n
        go=np.concatenate((gb,ga,gr),axis=1)
        dh=go@p["out"].T
        g={k:np.zeros_like(v) for k,v in p.items()}
        g["out"]=h.T@go
        g["ob"]=go.sum(axis=0)
        if pair_weight:
            zz=h.reshape(-1,3,WIDTH)
            dz=dh.reshape(-1,3,WIDTH)
            pos=zz[:,0]-zz[:,1]
            neg=zz[:,0]-zz[:,2]
            scale=2*pair_weight/(len(zz)*WIDTH)
            active=(np.mean(neg*neg,axis=1)<0.10)[:,None]
            dz[:,0]+=scale*(pos-active*neg)
            dz[:,1]-=scale*pos
            dz[:,2]+=scale*active*neg
        for prev,c,et,mt in reversed(trace):
            dc=dh*mt*(1-c*c)
            g["rec"]+=prev.T@dc
            g["event"]+=et.T@dc
            g["hb"]+=dc.sum(axis=0)
            dh=dh*(1-mt)+dc@p["rec"].T
        di=dh*(1-h0*h0)
        g["init"]=x.T@di
        g["ib"]=di.sum(axis=0)
        self.step+=1
        for k in p:
            q=np.clip(g[k],-2,2)
            self.v[k]=.9*self.v[k]+.1*q
            self.s[k]=.999*self.s[k]+.001*q*q
            p[k]-=.003*(self.v[k]/(1-.9**self.step))/(np.sqrt(self.s[k]/(1-.999**self.step))+1e-8)

    def evaluate(self,d):
        h,o,_=self.forward(d,np.arange(len(d["x"])))
        pb=softmax(o[:,:32])
        pa=softmax(o[:,32:42])
        acc=(np.argmax(pa,axis=1)==d["action"])
        tv=.5*np.abs(pb-d["belief"]).sum(axis=1)
        v=np.abs(o[:,42]-d["reg"][:,0])*40
        r=np.abs(o[:,43]-d["reg"][:,1])*4
        t=d["trip"]
        pos=np.mean((h[t[:,0]]-h[t[:,1]])**2,axis=1)
        neg=np.mean((h[t[:,0]]-h[t[:,2]])**2,axis=1)
        rows={}
        for root in sorted(set(d["roots"])):
            ti=np.flatnonzero(np.asarray(d["roots"])==root)
            i=np.unique(t[ti])
            rows[root]={"triplet_accuracy":float(np.mean((pos[ti]<neg[ti])+.5*(pos[ti]==neg[ti]))),
                        "positive_distance":float(pos[ti].mean()),"negative_distance":float(neg[ti].mean()),
                        "action_accuracy":float(acc[i].mean()),"belief_tv":float(tv[i].mean()),
                        "value_mae":float(v[i].mean()),"risk_mae":float(r[i].mean()),
                        "records":len(i),"triplets":len(ti)}
        metrics={k:float(np.mean([x[k] for x in rows.values()])) for k in next(iter(rows.values())) if k not in ("records","triplets")}
        return {"root_mean":metrics,"roots":rows}

    def weights(self):
        return {k:v.tolist() for k,v in self.p.items()}


def campaign(out,counts=COUNTS,epochs=EPOCHS,limit=TRIPLETS,seeds=SEEDS):
    out=Path(out)
    out.mkdir(parents=True,exist_ok=False)
    for name in ("public","referee","models"):
        (out/name).mkdir()
    roots={}
    seen=set()
    inventory={}
    for split,n in zip(SPLITS,counts):
        rows=[]
        for spec in specs(split,n):
            rid=digest(spec)
            require(rid not in seen,"split root overlap")
            seen.add(rid)
            p,r=root_data(spec,limit)
            verify(p,r)
            write_json(out/"public"/(rid+".json"),p)
            write_json(out/"referee"/(rid+".json"),r)
            rows.append((rid,p,r))
        roots[split]=rows
        inventory[split]={"roots":[x[0] for x in rows],
                          "literal_states":sum(r["literal_states"] for _,_,r in rows),
                          "quotient_classes":sum(r["quotient_classes"] for _,_,r in rows),
                          "records":sum(len(r["records"]) for _,_,r in rows),
                          "triplets":sum(len(r["triplets"]) for _,_,r in rows),
                          "hard_negatives":sum(t["same_prefix_negative"] for _,_,r in rows for t in r["triplets"]),
                          "different_mask_positives":sum(t["different_mask_positive"] for _,_,r in rows for t in r["triplets"])}
    datasets={bag:{split:arrays(rows,bag) for split,rows in roots.items()} for bag in (False,True)}
    results={}
    for name,bag,pair_weight in ARMS:
        results[name]={}
        d=datasets[bag]["train"]
        for seed in seeds:
            model=RNN(seed)
            rng=np.random.default_rng(seed+100)
            start=time.process_time_ns()
            for ep in range(epochs):
                order=rng.permutation(len(d["trip"]))
                for i in range(0,len(order),BATCH):
                    model.train_batch(d,order[i:i+BATCH],pair_weight)
            elapsed=time.process_time_ns()-start
            results[name][str(seed)]={"params":sum(v.size for v in model.p.values()),
                                     "updates":model.step,"train_cpu_ns":elapsed,
                                     "evaluation":{split:model.evaluate(datasets[bag][split]) for split in SPLITS}}
            write_json(out/"models"/(name+"-"+str(seed)+".json"),model.weights())
    report={"schema_version":"distill.history_memory.v1","source_kind":"synthetic_finite_model",
            "training_admission":False,"epochs":epochs,"width":WIDTH,"seed_list":list(seeds),
            "inventory":inventory,"results":results}
    # Per-root detail is split into files to stay within the canonical size limit.
    for name,by_seed in results.items():
        for seed,metrics in by_seed.items():
            write_json(out/"models"/(name+"-"+seed+"-evaluation.json"),metrics)
            for split in SPLITS:
                metrics["evaluation"][split].pop("roots")
    write_json(out/"report.json",report)
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",required=True)
    args=p.parse_args()
    print(json.dumps(campaign(args.out),sort_keys=True))


if __name__=="__main__":
    main()

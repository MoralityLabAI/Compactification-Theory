"""RB-CG-01 small autoregressive critique decoders; no gold in inference inputs."""
import argparse
import hashlib
import json
import math
import random
import re
import time
from collections import Counter
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from corpus import NAME, MAX_LINES, Parsed, SPLITS, render_critique

FIXED = ("PAD", "BOS", "EOS", "CLEAN", "AX", "SEP", "MATCH", "BRIDGE", "COUNTER", "UNLINKED", "0", "1")
OUT = FIXED + tuple(f"L{i}" for i in range(1,MAX_LINES+1))
OI = {x:i for i,x in enumerate(OUT)}
TOKEN = re.compile(NAME+r"|[A-Za-z_]\w*|<;>|:=|=>|&&|\|\||[^\s]")
WIDTHS=(32,80)
SEEDS=(41,43)
EPOCHS=10
BATCH=32
MAX_WORDS=128
MAX_OUT=32

def load_rows(path,split,training=False):
    items=[json.loads(l) for l in (path/f"{split}-inputs.jsonl").read_text().splitlines()]
    for row in items:
        assert set(row)=={"id","source","root","target"}
    if not training:
        return items
    gold={x["id"]:x["certificate"] for x in map(json.loads,(path/f"{split}-gold.jsonl").read_text().splitlines())}
    assert set(gold)=={x["id"] for x in items}
    return [(x,gold[x["id"]]) for x in items]

def tokenize(row):
    raw=row["source"]
    names=list(dict.fromkeys(re.findall(NAME,raw)))
    mapping={n:"ID"+str(i) for i,n in enumerate(names)}
    assert len(names)<=MAX_LINES
    query=["selected",mapping[row["root"]],"advertised",mapping[row["target"]]]
    lines=[query]+[[mapping.get(w,w) for w in TOKEN.findall(s)] for s in raw.splitlines()]
    assert len(lines)<=MAX_LINES+1
    assert max(map(len,lines))<=MAX_WORDS
    return lines

def vocabulary(train):
    words=set(w for row,_ in train for line in tokenize(row) for w in line)
    words.update("ID"+str(i) for i in range(MAX_LINES))
    vocab=["<PAD>","<UNK>"]+sorted(words)
    return {w:i for i,w in enumerate(vocab)}

def encode(row,vocab):
    return [[vocab.get(w,1) for w in line] for line in tokenize(row)]

def pack(rows,targets=None):
    n=len(rows); line_count=max(map(len,rows)); word_count=max(len(s) for x in rows for s in x)
    x=torch.zeros(n,line_count,word_count,dtype=torch.long)
    mask=torch.zeros(n,line_count,dtype=torch.bool)
    for i,lines in enumerate(rows):
        for j,line in enumerate(lines):
            x[i,j,:len(line)]=torch.tensor(line)
            mask[i,j]=True
    if targets is None:
        return x,mask
    maxlen=max(map(len,targets))
    assert maxlen<=MAX_OUT
    y=torch.zeros(n,maxlen,dtype=torch.long)
    for i,t in enumerate(targets):
        y[i,:len(t)]=torch.tensor(t)
    prev=torch.cat((torch.full((n,1),OI["BOS"],dtype=torch.long),y[:,:-1]),dim=1)
    return x,mask,prev,y

class Critic(nn.Module):
    def __init__(self,vocab,width):
        super().__init__()
        self.width=width
        self.word=nn.Embedding(vocab,width,padding_idx=0)
        self.position=nn.Embedding(MAX_WORDS,width)
        self.line=nn.Linear(width,width)
        self.encoder=nn.GRU(width,width,batch_first=True,bidirectional=True)
        self.enc_projection=nn.Linear(2*width,width)
        self.output=nn.Embedding(len(OUT),width,padding_idx=0)
        self.decoder=nn.GRU(width,width,batch_first=True)
        self.init=nn.Linear(2*width,width)
        self.query=nn.Linear(width,width,bias=False)
        self.keys=nn.Linear(width,width,bias=False)
        self.pointer_query=nn.Linear(2*width,width,bias=False)
        self.pointer_keys=nn.Linear(width,width,bias=False)
        self.fixed=nn.Linear(2*width,len(FIXED))

    def read(self,x,mask):
        pos=self.position(torch.arange(x.shape[-1])).view(1,1,x.shape[-1],self.width)
        words=self.word(x)*(1+torch.tanh(pos))
        lines=(words*(x!=0).unsqueeze(-1)).sum(2)/(x!=0).sum(2).clamp_min(1).unsqueeze(-1)
        lengths=mask.sum(1).cpu()
        packed=nn.utils.rnn.pack_padded_sequence(torch.tanh(self.line(lines)),lengths,batch_first=True,enforce_sorted=False)
        enc,_=self.encoder(packed)
        enc,_=nn.utils.rnn.pad_packed_sequence(enc,batch_first=True,total_length=x.shape[1])
        enc=torch.tanh(self.enc_projection(enc))
        avg=(enc*mask.unsqueeze(-1)).sum(1)/lengths.to(enc.device).unsqueeze(-1)
        h=torch.tanh(self.init(torch.cat((enc[:,0],avg),dim=-1))).unsqueeze(0)
        return enc,h

    def decode(self,enc,mask,prev,h):
        pointer=prev>=len(FIXED)
        line_ids=(prev-len(FIXED)+1).clamp(0,enc.shape[1]-1)
        copied=torch.gather(enc,1,line_ids.unsqueeze(-1).expand(-1,-1,self.width))
        inp=self.output(prev)+copied*pointer.unsqueeze(-1)
        dec,h=self.decoder(inp,h)
        weights=torch.bmm(self.query(dec),self.keys(enc).transpose(1,2))/math.sqrt(self.width)
        weights=weights.masked_fill(~mask.unsqueeze(1),-1e9)
        ctx=torch.bmm(weights.softmax(-1),enc)
        state=torch.cat((dec,ctx),dim=-1)
        fixed=self.fixed(state)
        logits=torch.bmm(self.pointer_query(state),self.pointer_keys(enc[:,1:]).transpose(1,2))/math.sqrt(self.width)
        logits=logits.masked_fill(~mask[:,None,1:],-1e9)
        logits=F.pad(logits,(0,MAX_LINES-logits.shape[-1]),value=-1e9)
        return torch.cat((fixed,logits),dim=-1),h

    def forward(self,x,mask,prev):
        enc,h=self.read(x,mask)
        return self.decode(enc,mask,prev,h)[0]

    @torch.no_grad()
    def generate(self,x,mask):
        enc,h=self.read(x,mask)
        prev=torch.full((x.shape[0],1),OI["BOS"],dtype=torch.long)
        tokens=[[] for _ in range(x.shape[0])]
        for _ in range(MAX_OUT):
            logits,h=self.decode(enc,mask,prev,h)
            prev=logits.argmax(-1)
            for row,k in zip(tokens,prev[:,0].tolist()):
                if not row or row[-1]!="EOS":
                    row.append(OUT[k])
            if all(row and row[-1]=="EOS" for row in tokens):
                break
        return tokens

def fit(train_encoded,targets,vocab,width,seed,out):
    torch.manual_seed(seed)
    rng=random.Random(seed)
    model=Critic(len(vocab),width)
    opt=torch.optim.AdamW(model.parameters(),lr=.003)
    start=time.process_time()
    history=[]
    for epoch in range(EPOCHS):
        model.train()
        order=list(range(len(targets)))
        rng.shuffle(order)
        loss_sum=0.; count=0
        for offset in range(0,len(order),BATCH):
            ids=order[offset:offset+BATCH]
            x,m,prev,y=pack([train_encoded[i] for i in ids],[targets[i] for i in ids])
            opt.zero_grad()
            logits=model(x,m,prev)
            loss=F.cross_entropy(logits.flatten(0,1),y.flatten(),ignore_index=OI["PAD"])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step()
            n=int((y!=OI["PAD"]).sum())
            loss_sum+=float(loss.detach())*n;count+=n
        history.append(loss_sum/count)
        print("EPOCH",width,seed,epoch+1,history[-1],flush=True)
    tag=f"gru-{width}-{seed}"
    path=out/(tag+".pt")
    torch.save(model.state_dict(),path)
    result={"model":tag,"parameters":sum(p.numel() for p in model.parameters()),"loss":history,
            "cpu_seconds":time.process_time()-start,"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
    print("FIT",json.dumps(result),flush=True)
    return model.eval(),result

def predict(model,encoded):
    res=[]
    for off in range(0,len(encoded),BATCH):
        x,m=pack(encoded[off:off+BATCH])
        res.extend(model.generate(x,m))
    return res

def bag(encoded,vocab):
    a=torch.zeros(len(encoded),vocab)
    for i,lines in enumerate(encoded):
        counts=Counter(v for line in lines for v in line)
        for k,n in counts.items():
            a[i,k]=n
    return F.normalize(a,dim=-1)

def skeleton(tokens):
    return tuple("L" if x.startswith("L") else "BIT" if x in ("0","1") else x for x in tokens)

def evaluate(rows,predictions,split,policy,out):
    counts=Counter(); details=[]
    assert len(rows)==len(predictions)
    parser_cache={}
    for row,pred in zip(rows,predictions):
        raw=row["source"]
        if raw not in parser_cache:
            parser_cache[raw]=Parsed(raw)
        p=parser_cache[raw]
        root,target=row["root"],row["target"]
        res=p.verify(root,target,pred)
        # Use an independent raw source interpretation, not the gold answer key.
        expected=p.symbolic(root,target)
        sem=expected[expected.index("SEP")+1]
        supported=expected[0]=="CLEAN" and sem in ("MATCH","BRIDGE")
        counts["n"]+=1
        counts["complete"]+=res["complete"]
        counts["axiom"]+=res["axiom"]
        counts["semantic"]+=res["semantic"]
        counts["malformed"]+=not res["syntax"]
        counts["supported_n"]+=supported
        counts["support"]+=res["support"]
        counts["unlinked_n"]+=sem=="UNLINKED"
        counts["unlinked"]+=res["unlinked"]
        counts["counter_n"]+=sem=="COUNTER"
        counts["verified_counter"]+=res["semantic"] and sem=="COUNTER"
        counts["false_accusation"]+=supported and ("AX" in pred or "COUNTER" in pred)
        counts["unverified"]+=not res["complete"]
        counts["tokens"]+=len(pred)
        details.append({"id":row["id"],"split":split,"policy":policy,"generated":pred,
                        **res,"critique":render_critique(raw,root,target,pred)})
    (out/f"{split}-{policy}.jsonl").write_text("".join(json.dumps(x)+"\n" for x in details))
    return {"split":split,"policy":policy,**counts}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--corpus",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    train=load_rows(args.corpus,"train",True)
    assert len(train)==2048
    vocab=vocabulary(train)
    (args.out/"vocabulary.json").write_text(json.dumps(vocab,sort_keys=True))
    train_enc=[encode(x,vocab) for x,_ in train]
    train_targets=[[OI[w] for w in y] for _,y in train]
    train_bag=bag(train_enc,len(vocab))
    common=Counter(skeleton(y) for _,y in train).most_common(1)[0][0]
    constant=next(y for _,y in train if skeleton(y)==common)
    eval_rows={split:load_rows(args.corpus,split) for split in SPLITS if split!="train"}
    eval_enc={split:[encode(x,vocab) for x in rows] for split,rows in eval_rows.items()}
    scores=[];fits=[]
    for split,rows in eval_rows.items():
        symbolic=[Parsed(x["source"]).symbolic(x["root"],x["target"]) for x in rows]
        nearest=(bag(eval_enc[split],len(vocab))@train_bag.T).argmax(1).tolist()
        for policy,preds in (("symbolic",symbolic),("constant",[constant]*len(rows)),("lexical",[train[i][1] for i in nearest])):
            s=evaluate(rows,preds,split,policy,args.out);scores.append(s)
            print("SCORE",json.dumps(s),flush=True)
            if policy=="symbolic":
                assert s["complete"]==s["n"]
    for width in WIDTHS:
        for seed in SEEDS:
            model,info=fit(train_enc,train_targets,vocab,width,seed,args.out);fits.append(info)
            for split,rows in eval_rows.items():
                start=time.process_time()
                preds=predict(model,eval_enc[split])
                s=evaluate(rows,preds,split,info["model"],args.out)
                s["inference_cpu_seconds"]=time.process_time()-start
                scores.append(s);print("SCORE",json.dumps(s),flush=True)
            (args.out/"scores.json").write_text(json.dumps(scores,indent=2))
            (args.out/"training.json").write_text(json.dumps(fits,indent=2))
    gate={}
    for width in WIDTHS:
        test=[x for x in scores if x["policy"].startswith(f"gru-{width}-") and x["split"].startswith("test")]
        assert len(test)==4
        gate[str(width)]=all(2*x["complete"]>=x["n"] and 20*x["false_accusation"]<=x["supported_n"] for x in test)
    (args.out/"gate.json").write_text(json.dumps(gate,indent=2))
    print("GATE",json.dumps(gate),flush=True)

if __name__=="__main__":
    main()

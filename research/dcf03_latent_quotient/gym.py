"""DCF-03: observation-only affine automata. No hidden-state labels in fit/extract."""
import os
os.environ.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
import argparse, json, time, collections, itertools, hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
torch.set_num_threads(1)
A=tuple(range(32)); HOLD=tuple(a*8+b for a in range(1,4) for b in (2,5,7)); TRAIN=tuple(x for x in A if x not in HOLD)
# These are in TRAIN; every held-out composite is represented as two known primitives.
GEN=tuple(dict.fromkeys([0,8,16,24]+list(range(8))))
SEEDS=(131,137,139)

def trans(s,o): return ((2*(o//8)+1)*s+o%8)%8

def observe(word):
    s=0
    for o in word:s=trans(s,o)
    return int(s==0)

def sample(g,n,l,alphabet):
    # Oracle returns only binary observations, never the state sequence.
    choices=torch.tensor(alphabet); x=choices[torch.randint(len(choices),(n,l),generator=g)]
    s=torch.zeros(n,dtype=torch.long); bits=[]
    for o in x.T:
        s=trans(s,o);bits.append(s==0)
    return x,torch.stack(bits,1).float()

class RNN(nn.Module):
    def __init__(self,d=32,r=0,composed=True):
        super().__init__();self.d=d;self.r=r;self.composed=composed
        self.start=nn.Parameter(torch.randn(d)*.2); self.emb=nn.Embedding(12 if composed else 32,2*d)
        self.core=nn.Linear(d,d,bias=False) if not r else nn.Sequential(nn.Linear(d,r,bias=False),nn.Linear(r,d,bias=False))
        self.out=nn.Linear(d,1)
    def initial(self,n):return self.start.tanh().expand(n,-1)
    def atom(self,h,o):
        a,b=self.emb(o).chunk(2,-1); return torch.tanh(self.core(h)*(1+a)+b)
    def step(self,h,o):
        if self.composed:
            h1=self.atom(h,o//8);h=torch.where((o//8==0)[:,None],h,h1)
            h2=self.atom(h,4+o%8); return torch.where((o%8==0)[:,None],h,h2)
        return self.atom(h,o)
    def read(self,h):return self.out(h).squeeze(-1)
    def forward(self,x):
        h=self.initial(len(x));out=[]
        for o in x.T:h=self.step(h,o);out.append(self.read(h))
        return torch.stack(out,1)

class Automaton(nn.Module):
    """Eight anonymous states; only initial/accepting state 0 is anchored."""
    def __init__(self,k=8):
        super().__init__(); self.k=k; self.logits=nn.Parameter(torch.randn(12,k,k)*.3)
    def matrices(self):
        z=self.logits
        for _ in range(4):
            z=z-torch.logsumexp(z,-1,keepdim=True);z=z-torch.logsumexp(z,-2,keepdim=True)
        return z.softmax(-1)
    def initial(self,n):return F.one_hot(torch.zeros(n,dtype=torch.long,device=self.logits.device),self.k).float()
    def forward(self,x):
        h=self.initial(len(x)); m=self.matrices();out=[]
        for o in x.T:
            h1=torch.bmm(h[:,None],m[o//8]).squeeze(1);h=torch.where((o//8==0)[:,None],h,h1)
            h2=torch.bmm(h[:,None],m[4+o%8]).squeeze(1);h=torch.where((o%8==0)[:,None],h,h2)
            out.append(torch.logit(h[:,0].clamp(1e-6,1-1e-6)))
        return torch.stack(out,1)
    def table(self):
        m=self.matrices().detach().argmax(-1).numpy();return [[int(m[4+o%8, m[o//8,s] if o//8 else s]) if o%8 else int(m[o//8,s] if o//8 else s) for o in A] for s in range(self.k)]

ARMS={'dense32':lambda:RNN(), 'rank4':lambda:RNN(r=4), 'monolithic32':lambda:RNN(composed=False), 'latent8':lambda:Automaton()}

def fit(seed,arm,steps,batch):
    torch.manual_seed(seed);m=ARMS[arm]();g=torch.Generator().manual_seed(10000+seed); opt=torch.optim.AdamW(m.parameters(),lr=.015 if arm=='latent8' else .003)
    start=time.perf_counter()
    for t in range(steps):
        x,y=sample(g,batch,4,TRAIN);opt.zero_grad(set_to_none=True);p=m(x)
        loss=F.binary_cross_entropy_with_logits(p,y,pos_weight=torch.tensor(3.))
        # Weak permutation regularization, no semantic label or known transition.
        if arm=='latent8':
            mm=m.matrices();loss=loss+.02*(-mm*mm.clamp_min(1e-8).log()).sum(-1).mean()
        loss.backward();opt.step()
    return m.eval(),time.perf_counter()-start,float(loss.detach())

@torch.inference_mode()
def metrics(m,x,y):
    p=m(x)>0;z=y.bool();correct=p==z
    return dict(final_correct=int(correct[:,-1].sum()),trace_correct=int(correct.all(1).sum()),n=len(x),pos_correct=int((p&z).sum()),positives=int(z.sum()),neg_correct=int((~p&~z).sum()),negatives=int((~z).sum()))

def certify(table,accept,alphabet=A):
    """Product-automaton BFS. Oracle is used ONLY for audit; shortest counterexample."""
    q=collections.deque([(0,0,())]);seen=set()
    while q:
        s,h,w=q.popleft()
        if (s,h) in seen:continue
        seen.add((s,h))
        if int(s==0)!=int(accept[h]):return dict(exact=False,visited=len(seen),counterexample=list(w),oracle=int(s==0),candidate=int(accept[h]))
        for o in alphabet:q.append((trans(s,o),table[h][o],w+(o,)))
    return dict(exact=True,visited=len(seen),counterexample=None)

@torch.inference_mode()
def extract(m,threshold=0.,max_states=64):
    """Use learned outputs only: signatures from empty and seven additive probe words."""
    dev=m.start.device
    def signature(h):
        z=h.expand(8,-1); ops=torch.arange(8,device=dev);n=m.step(z,ops)
        return tuple([int(m.read(h)[0]>threshold)]+(m.read(n)[1:]>threshold).int().tolist())
    h0=m.initial(1);reps=[h0]; sigs={signature(h0):0};table=[];accept=[];words=[()]
    while len(table)<len(reps):
        h=reps[len(table)]; w=words[len(table)];accept.append(int(m.read(h)[0]>threshold));new=m.step(h.expand(32,-1),torch.arange(32,device=dev));row=[]
        for o,v in enumerate(new):
            v=v[None];sg=signature(v)
            if sg not in sigs:
                if len(reps)>=max_states:return None,None,dict(states=len(reps),overflow=True)
                sigs[sg]=len(reps);reps.append(v);words.append(w+(o,))
            row.append(sigs[sg])
        table.append(row)
    return table,accept,dict(states=len(reps),overflow=False,representatives=[list(w) for w in words])

def main():
    p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=600);p.add_argument('--batch',type=int,default=96);p.add_argument('--out',type=Path,required=True);p.add_argument('--seed',type=int,nargs='*',default=SEEDS);p.add_argument('--arms',nargs='*',default=list(ARMS));args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True);rows=[];audits=[]
    for seed in args.seed:
        for arm in args.arms:
            m,t,loss=fit(seed,arm,args.steps,args.batch);torch.save(m.state_dict(),args.out/f'{seed}_{arm}.pt')
            for l in (4,8,16,64):
                for split,alph in [('seen',TRAIN),('mixed',A),('held_only',HOLD)]:
                    x,y=sample(torch.Generator().manual_seed(30000+100*seed+l),1024,l,alph);rows.append(dict(seed=seed,arm=arm,length=l,split=split,**metrics(m,x,y)))
            if arm=='latent8': tab=m.table();acc=[1]+[0]*(m.k-1);meta={'states':m.k,'overflow':False}
            else:tab,acc,meta=extract(m)
            audit=dict(seed=seed,arm=arm,params=sum(v.numel() for v in m.parameters()),train_s=t,loss=loss,**meta,table=tab,accept=acc)
            if tab is not None:audit.update(seen=certify(tab,acc,TRAIN),all=certify(tab,acc))
            audits.append(audit); print(seed,arm,round(t,2),round(loss,4),audit.get('all'),flush=True)
            (args.out/'receipt.json').write_text(json.dumps(dict(steps=args.steps,batch=args.batch,train_ops=TRAIN,hold_ops=HOLD,metrics=rows,audits=audits),indent=2)+'\n')
if __name__=='__main__':main()

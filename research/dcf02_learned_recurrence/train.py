"""DCF-02A: exact modular affine composition, small CPU pilot."""
import argparse, json, os, statistics, time
from pathlib import Path
os.environ.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
torch.set_num_threads(1)
SEEDS=(11,23,47)
ARMS={'dense64':(64,0),'rank16':(64,16),'rank8':(64,8),'dense48':(48,0)}
LENGTHS=(4,8,16)

def data(g,n,length):
    s=torch.randint(8,(n,),generator=g); x=torch.randint(32,(n,length),generator=g)
    h=s; ys=[]
    for op in x.T:
        h=((2*(op//8)+1)*h+op%8)%8; ys.append(h)
    return s,x,torch.stack(ys,dim=1)

class Model(nn.Module):
    def __init__(self,d,r):
        super().__init__(); self.d=d; self.r=r
        self.start=nn.Embedding(8,d); self.op=nn.Embedding(32,2*d)
        self.core=nn.Linear(d,d,bias=False) if not r else nn.Sequential(nn.Linear(d,r,bias=False),nn.Linear(r,d,bias=False))
        self.head=nn.Linear(d,8)
    def forward(self,s,x):
        h=self.start(s); out=[]
        for step in self.op(x).unbind(1):
            a,b=step.chunk(2,dim=-1); h=torch.tanh(self.core(h)*(1+a)+b); out.append(self.head(h))
        return torch.stack(out,dim=1)

def fit(seed,arm,steps,batch):
    torch.manual_seed(seed); m=Model(*ARMS[arm]); opt=torch.optim.AdamW(m.parameters(),lr=.003)
    g=torch.Generator().manual_seed(1000+seed); t=time.perf_counter()
    for _ in range(steps):
        s,x,y=data(g,batch,4); opt.zero_grad(set_to_none=True)
        loss=F.cross_entropy(m(s,x).reshape(-1,8),y.flatten()); loss.backward(); opt.step()
    return m.eval(),time.perf_counter()-t,float(loss.detach())

def svd_arm(m,r):
    z=Model(m.d,r); z.load_state_dict({k:v for k,v in m.state_dict().items() if k!='core.weight'},strict=False)
    u,s,v=torch.linalg.svd(m.core.weight.detach(),full_matrices=False)
    with torch.no_grad():
        z.core[0].weight.copy_(v[:r]*s[:r,None].sqrt()); z.core[1].weight.copy_(u[:,:r]*s[:r].sqrt())
    return z.eval()

@torch.inference_mode()
def assess(m,fixture):
    s,x,y=fixture; logits=m(s,x)[:,-1]; target=y[:,-1]
    return (logits.argmax(-1)==target).numpy(),float(F.cross_entropy(logits,target))

def lower(diff,seed):
    rng=np.random.default_rng(seed)
    return float(np.quantile(rng.choice(diff,size=(2000,len(diff)),replace=True).mean(axis=1),.05))

@torch.inference_mode()
def timing(models,seed,reps):
    out=[]
    for b in (1,16,128):
        for length in (4,16):
            s,x,_=data(torch.Generator().manual_seed(3000+seed),b,length)
            for m in models.values():
                for _ in range(3): m(s,x)
            samples={a:[] for a in models}; rng=np.random.default_rng(seed+b+length)
            for _ in range(reps):
                for a in rng.permutation(list(models)):
                    t=time.perf_counter_ns(); models[a](s,x); samples[a].append(time.perf_counter_ns()-t)
            for a,v in samples.items():
                out.append(dict(seed=seed,arm=a,batch=b,length=length,median_ns=statistics.median(v),p95_ns=float(np.quantile(v,.95)),samples_ns=v))
    return out

def main():
    p=argparse.ArgumentParser(); p.add_argument('--steps',type=int,default=400); p.add_argument('--batch',type=int,default=96); p.add_argument('--test-n',type=int,default=1024); p.add_argument('--reps',type=int,default=21); p.add_argument('--out',type=Path,default=Path('dcf02_run')); args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True); rows=[]; latency=[]
    # Independent scalar oracle on actual generated fixtures.
    s,x,y=data(torch.Generator().manual_seed(9),128,16)
    for i in range(len(s)):
        h=int(s[i])
        for j,op in enumerate(x[i].tolist()):
            h=([1,3,5,7][op//8]*h+op%8)&7; assert h==int(y[i,j])
    for seed in SEEDS:
        models={}; meta={}
        for a in ARMS:
            m,t,loss=fit(seed,a,args.steps,args.batch); models[a]=m; meta[a]=(t,loss); print(seed,a,round(t,2),round(loss,4),flush=True)
        for r in (8,16):
            a=f'svd{r}'; models[a]=svd_arm(models['dense64'],r); meta[a]=(None,None)
        for length in LENGTHS:
            fixture=data(torch.Generator().manual_seed(2000+seed*100+length),args.test_n,length); base,_=assess(models['dense64'],fixture)
            for a,m in models.items():
                hit,nll=assess(m,fixture); diff=hit.astype(float)-base.astype(float); t,loss=meta[a]; n=sum(v.numel() for v in m.parameters())
                rows.append(dict(seed=seed,arm=a,length=length,n=args.test_n,correct=int(hit.sum()),accuracy=float(hit.mean()),nll=nll,delta=float(diff.mean()),paired_lcb95=lower(diff,seed+length),wins=int((diff>0).sum()),losses=int((diff<0).sum()),params=n,core_params=m.d*m.d if not m.r else 2*m.d*m.r,weight_bytes=n*4,train_s=t,last_loss=loss))
        latency.extend(timing(models,seed,args.reps))
        for a,m in models.items(): torch.save(dict(d=m.d,r=m.r,state=m.state_dict()),args.out/f'{seed}_{a}.pt')
    receipt=dict(config=dict(seeds=SEEDS,arms=ARMS,lengths=LENGTHS,steps=args.steps,batch=args.batch,test_n=args.test_n,reps=args.reps,torch=torch.__version__,numpy=np.__version__,threads=torch.get_num_threads(),device='cpu'),accuracy=rows,latency=latency)
    (args.out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')

if __name__=='__main__': main()

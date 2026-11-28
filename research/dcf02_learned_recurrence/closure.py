"""DCF-02C: canonical-state loss on fresh seeds, paired objective ablation."""
from train import *

FRESH=(101,103,107)
LENS=(4,8,16,32)

class ClosedModel(Model):
    def trace(self,s,x):
        h=torch.tanh(self.start(s)); out=[]; states=[]
        for step in self.op(x).unbind(1):
            a,b=step.chunk(2,dim=-1); h=torch.tanh(self.core(h)*(1+a)+b)
            states.append(h); out.append(self.head(h))
        return torch.stack(out,1),torch.stack(states,1)
    def forward(self,s,x):
        return self.trace(s,x)[0]

def fit_obj(seed,arm,lam):
    torch.manual_seed(seed); m=ClosedModel(*ARMS[arm]); opt=torch.optim.AdamW(m.parameters(),lr=.003)
    g=torch.Generator().manual_seed(1000+seed); t=time.perf_counter()
    for _ in range(400):
        s,x,y=data(g,96,4); opt.zero_grad(set_to_none=True)
        logits,h=m.trace(s,x); ce=F.cross_entropy(logits.reshape(-1,8),y.flatten())
        target=torch.tanh(m.start(y)).detach(); loss=ce+lam*F.mse_loss(h,target)
        loss.backward(); opt.step()
    return m.eval(),time.perf_counter()-t

@torch.inference_mode()
def closure_error(m):
    s=torch.arange(8).repeat_interleave(32); x=torch.arange(32).repeat(8)[:,None]
    y=((2*(x[:,0]//8)+1)*s+x[:,0]%8)%8
    logits,h=m.trace(s,x); target=torch.tanh(m.start(y)); err=(h[:,0]-target).norm(dim=-1)
    return dict(single_step_correct=int((logits[:,0].argmax(-1)==y).sum()),mean_l2=float(err.mean()),max_l2=float(err.max()))

def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',type=Path,required=True); a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=True); rows=[]; closure=[]
    for seed in FRESH:
        for arm in ('dense64','rank8'):
            models={}
            for lam in (0.,1.):
                m,t=fit_obj(seed,arm,lam); models[lam]=m
                torch.save(dict(d=m.d,r=m.r,state=m.state_dict(),bounded_initial=True),a.out/f'{seed}_{arm}_{int(lam)}.pt')
                closure.append(dict(seed=seed,arm=arm,lam=lam,train_s=t,**closure_error(m)))
            for l in LENS:
                fixture=data(torch.Generator().manual_seed(110000+seed*100+l),2048,l)
                base,nll0=assess(models[0.],fixture); hit,nll1=assess(models[1.],fixture); diff=hit.astype(float)-base.astype(float)
                rows.append(dict(seed=seed,arm=arm,length=l,n=len(hit),ce_correct=int(base.sum()),closure_correct=int(hit.sum()),ce_nll=nll0,closure_nll=nll1,delta=float(diff.mean()),lcb95=lower(diff,seed+l)))
            print(seed,arm,flush=True)
    (a.out/'receipt.json').write_text(json.dumps(dict(seeds=FRESH,lambda_values=[0,1],steps=400,batch=96,train_length=4,eval_n=2048,eval_lengths=LENS,rows=rows,closure=closure),indent=2)+'\n')

if __name__=='__main__': main()

"""SG-02 CPU development: source-bound spectral width and gated recurrent toy."""
import argparse
import itertools
import json
import math
import time
import torch
import torch.nn.functional as F


def anchored_width(gates, sites):
    """Independent Bernoulli consumers, fixed producer at sites[0]. Last axis T."""
    if sites.ndim != 1 or sites.numel() != gates.shape[-1]+1:
        raise ValueError('sites must include producer and all consumers')
    if sites.requires_grad or not torch.isfinite(sites).all() or (sites.diff()<0).any():
        raise ValueError('fixed finite sorted sites required')
    if not torch.is_floating_point(gates) or not torch.isfinite(gates).all() or ((gates<0)|(gates>1)).any():
        raise ValueError('invalid probabilities')
    suffix = 1-(1-gates).flip(-1).cumprod(-1).flip(-1)
    return (suffix*sites.to(gates).diff()).sum(-1)


def joint_bounds(gates, sites):
    """Marginal-only bounds on E anchored width for arbitrary dependence."""
    gaps=sites.to(gates).diff()
    lower=gates.flip(-1).cummax(-1).values.flip(-1)
    upper=gates.flip(-1).cumsum(-1).flip(-1).clamp(max=1)
    return (lower*gaps).sum(-1),(upper*gaps).sum(-1)


def span(g):
    q=1-g
    return ((1-q.cumprod(-1)[...,:-1])*(1-q.flip(-1).cumprod(-1).flip(-1)[...,1:])).sum(-1)


def hard_cost(mask):
    return (mask*torch.arange(1,mask.shape[-2]+1,dtype=mask.dtype,device=mask.device)[:,None]).amax(-2).sum(-1)


def arithmetic_checks():
    n=0; sites=torch.tensor([0.,1.,3.,7.],dtype=torch.float64)
    for ps in itertools.product([0.,.25,.5,1.],repeat=3):
        p=torch.tensor(ps,dtype=torch.float64); want=0.
        for v in itertools.product([0.,1.],repeat=3):
            x=torch.tensor(v,dtype=p.dtype); prob=torch.where(x.bool(),p,1-p).prod()
            want+=float(prob*(x*sites[1:]).max())
        got=float(anchored_width(p,sites)); lo,hi=joint_bounds(p,sites)
        assert abs(got-want)<1e-12 and lo-1e-12<=got<=hi+1e-12
        n+=1
    # Enumerate every binary 3 visits x 3 labels schedule; group by row budgets.
    minima={}
    for v in itertools.product([0.,1.],repeat=9):
        m=torch.tensor(v).reshape(3,3); k=tuple(map(int,m.sum(-1)))
        val=int(hard_cost(m)); bound=sum(max(k[t:]) for t in range(3))
        assert val>=bound
        minima[k]=min(val,minima.get(k,100)); n+=1
    for k,val in minima.items():
        assert val==sum(max(k[t:]) for t in range(3)); n+=1
    p=torch.tensor([.5,.5],dtype=torch.float64); s=torch.arange(3,dtype=p.dtype)
    lo,hi=joint_bounds(p,s)
    # Comonotone gates: (0,0)/(1,1); exclusive gates: (1,0)/(0,1).
    assert (float(lo),float(anchored_width(p,s)),float(hi))==(1.,1.25,1.5); n+=1
    return {'checks':n,'budget_profiles':len(minima),'correlated_example':[float(lo),float(anchored_width(p,s)),float(hi)]}


def run(roots=8, teacher_steps=160, gate_steps=100):
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    T,D,H,N,E,k=6,10,12,192,512,3
    S=torch.arange(T+1,dtype=torch.float32)
    rows=[]; clock=time.process_time()
    for regime in ['persistent','rotating']:
        ids=[57000+(1000 if regime=='rotating' else 0)+i for i in range(roots)]
        data=[]
        for seed in ids:
            r=torch.Generator().manual_seed(seed)
            q=torch.randn(T,D,generator=r)*.20
            common=torch.randperm(D,generator=r)[:k]
            for t in range(T):
                use=common if regime=='persistent' else torch.randperm(D,generator=r)[:k]
                q[t,use]+=torch.where(torch.rand(k,generator=r)>.5,1.,-1.)
            x=torch.randint(0,2,(N+E,D),generator=r).float()*2-1
            w=torch.randn(D,H,generator=r)/math.sqrt(D)
            u=torch.randn(H,H,generator=r)/math.sqrt(H)*.3
            v=torch.randn(H,1,generator=r)/math.sqrt(H)
            init=torch.randn(T,D,generator=r)*.025
            perm=torch.stack([torch.randperm(T,generator=r) for _ in range(D)])
            data.append((q,x,w,u,v,init,perm))
        q,x,w,u,v,init,perm=map(torch.stack,zip(*data)); y=((x[:, :,None,:]*q[:,None,:,:]).sum(-1).cumsum(-1)>0).float()
        w,u,v=[torch.nn.Parameter(a) for a in (w,u,v)]
        def logits(x,g):
            h=x.new_zeros(roots,x.shape[1],H); out=[]
            for t in range(T):
                h=torch.tanh(h@u+(x*(q[:,t]*g[:,t])[:,None,:])@w)
                out.append((h@v).squeeze(-1))
            return torch.stack(out,-1)
        dense=torch.ones_like(init); opt=torch.optim.Adam([w,u,v],lr=.035)
        for _ in range(teacher_steps):
            opt.zero_grad(); loss=F.binary_cross_entropy_with_logits(logits(x[:,:N],dense),y[:,:N]); loss.backward();opt.step()
        w.requires_grad_(False);u.requires_grad_(False);v.requires_grad_(False)
        def metrics(g):
            with torch.no_grad():
                l=logits(x[:,N:],g); acc=((l>=0)==y[:,N:].bool()).float().mean((1,2));nll=F.binary_cross_entropy_with_logits(l,y[:,N:],reduction='none').mean((1,2));return acc.tolist(),nll.tolist(),hard_cost(g).tolist()
        arms={'dense':metrics(dense)}
        for arm in ['baseline','anchored','unanchored','shuffled']:
            a=torch.nn.Parameter(init.clone()); opt=torch.optim.Adam([a],lr=.10)
            for _ in range(gate_steps):
                opt.zero_grad();g=a.sigmoid(); gt=g.transpose(-1,-2)
                loss=F.binary_cross_entropy_with_logits(logits(x[:,:N],g),y[:,:N])+3*(g.mean(-1)-k/D).square().mean()
                if arm=='anchored': loss=loss+.5*anchored_width(gt,S).mean()/T
                elif arm=='unanchored': loss=loss+.5*span(gt).mean()/T
                elif arm=='shuffled': loss=loss+.5*anchored_width(gt.gather(-1,perm),S).mean()/T
                loss.backward();opt.step()
            hard=torch.zeros_like(a).scatter_(-1,a.topk(k,dim=-1).indices,1.)
            assert torch.all(hard.sum(-1)==k)
            arms[arm]=metrics(hard)
        for i,seed in enumerate(ids):
            rows.append({'root':seed,'regime':regime,'eval_cases':E,'arms':{a:{'accuracy':v[0][i],'nll':v[1][i],'source_cost':v[2][i],'reads':T*(D if a=='dense' else k)} for a,v in arms.items()}})
    summary={}
    for regime in ['persistent','rotating']:
        rr=[r for r in rows if r['regime']==regime];summary[regime]={a:{m:sum(r['arms'][a][m] for r in rr)/len(rr) for m in ['accuracy','nll','source_cost','reads']} for a in rr[0]['arms']}
    return {'evidence':'designed recurrent toy, exposed development; no native model claim','roots_per_regime':roots,'training_cases_per_root':N,'eval_cases_per_root':E,'visits':T,'teacher_steps':teacher_steps,'gate_steps':gate_steps,'arithmetic':arithmetic_checks(),'summary':summary,'rows':rows,'cpu_seconds':time.process_time()-clock}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--roots',type=int,default=8);p.add_argument('--out',default='sg02.json');a=p.parse_args()
    result=run(a.roots);open(a.out,'w').write(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))

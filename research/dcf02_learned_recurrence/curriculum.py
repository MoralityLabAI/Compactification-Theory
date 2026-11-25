"""DCF-02B: equal-update/equal-step-label curriculum, fresh evaluation streams."""
from train import *

LENS=(4,8,16,32)

def fit_curr(seed,arm):
    torch.manual_seed(seed); m=Model(*ARMS[arm]); opt=torch.optim.AdamW(m.parameters(),lr=.003)
    g=torch.Generator().manual_seed(1000+seed); t=time.perf_counter()
    for length,batch,steps in ((4,96,100),(8,48,150),(16,24,150)):
        for _ in range(steps):
            s,x,y=data(g,batch,length); opt.zero_grad(set_to_none=True)
            loss=F.cross_entropy(m(s,x).reshape(-1,8),y.flatten()); loss.backward(); opt.step()
    return m.eval(),time.perf_counter()-t

def main():
    p=argparse.ArgumentParser(); p.add_argument('--prior',type=Path,required=True); p.add_argument('--out',type=Path,required=True); a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True); rows=[]
    for seed in SEEDS:
        for arm in ARMS:
            old=Model(*ARMS[arm]); old.load_state_dict(torch.load(a.prior/f'{seed}_{arm}.pt',weights_only=True)['state']); old.eval()
            new,t=fit_curr(seed,arm); torch.save(dict(d=new.d,r=new.r,state=new.state_dict()),a.out/f'{seed}_{arm}.pt')
            for l in LENS:
                fixture=data(torch.Generator().manual_seed(70000+seed*100+l),2048,l)
                prior,nll0=assess(old,fixture); cur,nll1=assess(new,fixture); diff=cur.astype(float)-prior.astype(float)
                rows.append(dict(seed=seed,arm=arm,length=l,n=len(cur),old_correct=int(prior.sum()),new_correct=int(cur.sum()),old_nll=nll0,new_nll=nll1,delta=float(diff.mean()),lcb95=lower(diff,seed+l),train_s=t))
            print(seed,arm,round(t,2),flush=True)
    (a.out/'receipt.json').write_text(json.dumps(dict(schedule=[[4,96,100],[8,48,150],[16,24,150]],steps=400,step_labels=153600,eval_n=2048,eval_lengths=LENS,rows=rows),indent=2)+'\n')

if __name__=='__main__': main()

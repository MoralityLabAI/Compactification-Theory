"""Known coupling-graph prior, mean-field approximation; exact laws remain hidden."""
from gym import *
class Coupled(Factored):
 def __init__(self,k):
  super().__init__(k)
  with torch.no_grad():self.flip.fill_(-2.)
 def forward(self,j,c,r,y):
  p=torch.full((len(j),self.k),.5,device=j.device);out=[];fl=.49*self.flip.sigmoid();err=.49*self.em.sigmoid()
  for a,b,d,z in zip(j.T,c.T,r.T,y.T):
   a,b,d,z=a[:,None],b[:,None],d[:,None],z[:,None].float();q=p.gather(1,a);control=p.gather(1,(a+1)%self.k);noise=fl[a,b];f=noise+(1-2*noise)*b*control;p=p.scatter(1,a,q+f*(1-2*q));q=p.gather(1,d);e=err[d];pred=e+(1-2*e)*q;out.append(pred[:,0]);l1=torch.where(z.bool(),1-e,e);l0=torch.where(z.bool(),e,1-e);p=p.scatter(1,d,q*l1/(q*l1+(1-q)*l0).clamp_min(1e-8))
  return torch.stack(out,1)
def run(root,steps=200,batch=64):
 rows=[];fits=[];root.mkdir(parents=True,exist_ok=True)
 for k,kind in CASES:
  for seed in SEEDS:
   p=law(k,seed);train=generate(k,kind,p,2048,16,seed+10);torch.manual_seed(seed);m=Coupled(k);opt=torch.optim.Adam(m.parameters(),lr=.035);g=torch.Generator().manual_seed(seed+909);start=time.perf_counter()
   for _ in range(steps):
    idx=torch.randint(len(train[0]),(batch,),generator=g);j,c,r,y=(x[idx] for x in train);opt.zero_grad(set_to_none=True);loss=F.binary_cross_entropy(m(j,c,r,y).clamp(1e-6,1-1e-6),y.float());loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
   fits.append(dict(k=k,kind=kind,seed=seed,params=sum(v.numel() for v in m.parameters()),train_s=time.perf_counter()-start,loss=float(loss.detach())));m.eval();torch.save(m.state_dict(),root/f'{k}_{kind}_{seed}_coupled_mf.pt')
   for l in (16,64,128):
    test=generate(k,kind,p,256,l,seed+10000+l);oracle=oracle_ind(test,k,p) if kind=='independent' else oracle_joint(test,k,kind,p)
    with torch.inference_mode():pred=m(*test).numpy()
    rows.append(dict(k=k,kind=kind,seed=seed,length=l,arm='coupled_mf',**scores(pred,test[-1].numpy(),oracle)))
   print(k,kind,seed,round(rows[-1]['nll'],4),round(rows[-1]['oracle_kl'],5),flush=True)
   (root/'receipt.json').write_text(json.dumps(dict(config=dict(steps=steps,batch=batch),fits=fits,rows=rows),separators=(',',':'))+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--steps',type=int,default=200);a=p.parse_args();run(a.out,a.steps)

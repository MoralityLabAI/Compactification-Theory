"""Sensitivity: 5x optimizer budget for dense GRU64 on four-bit systems."""
from gym import *
def run(root,steps=1000):
 root.mkdir(parents=True,exist_ok=True);rows=[];fits=[]
 for k,kind in ((4,'independent'),(4,'coupled')):
  for seed in SEEDS:
   p=law(k,seed);train=generate(k,kind,p,2048,16,seed+10);m,t,l=fit('gru64',k,train,seed,steps,64);fits.append(dict(k=k,kind=kind,seed=seed,train_s=t,params=sum(v.numel() for v in m.parameters()),loss=l));torch.save(m.state_dict(),root/f'{k}_{kind}_{seed}_gru64long.pt')
   for horizon in (16,64,128):
    test=generate(k,kind,p,256,horizon,seed+10000+horizon);oracle=oracle_ind(test,k,p) if kind=='independent' else oracle_joint(test,k,kind,p)
    with torch.inference_mode():pred=m(*test).numpy()
    rows.append(dict(k=k,kind=kind,seed=seed,length=horizon,arm='gru64long',**scores(pred,test[-1].numpy(),oracle)))
   print(k,kind,seed,t,rows[-1]['nll'],flush=True);(root/'receipt.json').write_text(json.dumps(dict(config=dict(steps=steps,batch=64),fits=fits,rows=rows),separators=(',',':'))+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--steps',type=int,default=1000);a=p.parse_args();run(a.out,a.steps)

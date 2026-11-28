"""Quantize persistent beliefs after each update; model parameters stay FP32."""
from gym import *
from meanfield import Coupled

def predict(data,k,flip,err,coupled,bits=None):
 j,c,r,y=[x.numpy() for x in data];n,l=j.shape;p=np.full((n,k),.5);out=np.zeros((n,l));rows=np.arange(n)
 for t in range(l):
  a,b,d=j[:,t],c[:,t],r[:,t];f=flip[a,b]
  if coupled:f=f+(1-2*f)*b*p[rows,(a+1)%k]
  q=p[rows,a];p[rows,a]=q+f*(1-2*q);q=p[rows,d];e=err[d];out[:,t]=e+(1-2*e)*q;l1=np.where(y[:,t],1-e,e);l0=np.where(y[:,t],e,1-e);p[rows,d]=q*l1/(q*l1+(1-q)*l0)
  if bits is not None:
   levels=(1<<bits)-1;p=np.rint(p*levels)/levels
 return out

def run(root):
 a=json.loads((root/'audit_receipt.json').read_text());rows=[];checks=[]
 for s in a['selection']:
  k,kind,seed,name=(s[v] for v in ('k','kind','seed','selected'));coupled=name=='coupled_mf';m=Coupled(k) if coupled else Factored(k);folder='meanfield_run' if coupled else 'run';m.load_state_dict(torch.load(root/folder/f'{k}_{kind}_{seed}_{name}.pt',weights_only=True));m.eval();p=law(k,seed);test=generate(k,kind,p,256,128,seed+60000);oracle=oracle_ind(test,k,p) if kind=='independent' else oracle_joint(test,k,kind,p)
  with torch.inference_mode():flip=m.flip.sigmoid().numpy()*(.49 if coupled else 1.);err=(.49*m.em.sigmoid()).numpy();fp=m(*test).numpy()
  ref=predict(test,k,flip,err,coupled);delta=float(np.max(np.abs(ref-fp)));assert delta<2e-5;checks.append(delta)
  for b in (None,2,3,4,8,16):
   pred=predict(test,k,flip,err,coupled,b);rows.append(dict(k=k,kind=kind,seed=seed,arm=name,bits=b,belief_bits=None if b is None else k*b,**scores(pred,test[-1].numpy(),oracle)))
 (root/'quantization_receipt.json').write_text(json.dumps(dict(rows=rows,max_fp_replay_delta=max(checks)),separators=(',',':'))+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);run(p.parse_args().root)

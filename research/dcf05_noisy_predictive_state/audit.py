"""Fresh validation selection, causal-prefix check and descriptive paired bootstrap."""
from gym import *
from meanfield import Coupled

def main(root):
 main=json.loads((root/'run/receipt.json').read_text());mf=json.loads((root/'meanfield_run/receipt.json').read_text());rows=main['rows']+mf['rows'];selected=[];checks=[];intervals=[]
 for k,kind in CASES:
  for seed in SEEDS:
   p=law(k,seed);val=generate(k,kind,p,256,32,seed+30000);y=val[-1].numpy();oracle=oracle_ind(val,k,p) if kind=='independent' else oracle_joint(val,k,kind,p);scores_val={}
   for name in ('factor','coupled_mf','gru16','gru64'):
    if name=='coupled_mf':m=Coupled(k);folder='meanfield_run'
    else:m=Factored(k) if name=='factor' else GRU(k,int(name[3:]));folder='run'
    m.load_state_dict(torch.load(root/folder/f'{k}_{kind}_{seed}_{name}.pt',weights_only=True));m.eval()
    with torch.inference_mode():
     a=m(*val).numpy();z=val[-1].clone();z[:,8:]=1-z[:,8:];b=m(*val[:-1],z).numpy()
    delta=float(np.max(np.abs(a[:,:9]-b[:,:9])));assert delta<1e-6;checks.append(dict(k=k,kind=kind,seed=seed,arm=name,max_prefix_delta=delta));scores_val[name]=scores(a,y,oracle)['nll']
   best=min(('factor','coupled_mf'),key=lambda a:scores_val[a]);row=next(r for r in rows if r['k']==k and r['kind']==kind and r['seed']==seed and r['length']==128 and r['arm']==best);selected.append(dict(k=k,kind=kind,seed=seed,selected=best,validation_nll=scores_val,test_nll=row['nll'],test_oracle_kl=row['oracle_kl']))
   rr={r['arm']:r for r in rows if r['k']==k and r['kind']==kind and r['seed']==seed and r['length']==128};g=np.random.default_rng(seed+888*k)
   # paired bootstrap of word means within system, not independent system-level inference.
   a=np.array(rr['factor']['nll_by_word']);b=np.array(rr['coupled_mf']['nll_by_word']);delta=a-b;boot=delta[g.integers(len(delta),size=(2000,len(delta)))].mean(1);intervals.append(dict(k=k,kind=kind,seed=seed,factor_minus_coupled_mf=float(delta.mean()),ci95=np.quantile(boot,[.025,.975]).tolist()))
 out=dict(selection=selected,causal_checks=checks,paired_word_bootstrap=intervals);(root/'audit_receipt.json').write_text(json.dumps(out,indent=2)+'\n');print('selection',[(s['k'],s['kind'],s['seed'],s['selected']) for s in selected]);print('max causal delta',max(c['max_prefix_delta'] for c in checks))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)

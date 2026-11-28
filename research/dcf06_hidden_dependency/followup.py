"""Exploratory DCF-06B: passive graph clusters, graph data ablation, independent oracle and causality audits."""
from gym import *

def brute(data,graph,law):
 j,c,r,y=data;n,l=j.shape;k=len(graph);out=np.zeros((n,l));N=1<<k
 for i in range(n):
  p=np.full(N,1/N)
  for t in range(l):
   a,b,d=int(j[i,t]),int(c[i,t]),int(r[i,t]);f=law['fail'][a] if b else law['idle'][a];q=np.zeros(N)
   for s,m in enumerate(p):
    z=s^((b*(1 if graph[a]<0 else ((s>>graph[a])&1)))<<a);q[z]+=m*(1-f);q[z^(1<<a)]+=m*f
   obs=np.array([law['err'][d] if ((s>>d)&1)==0 else 1-law['err'][d] for s in range(N)]);out[i,t]=q@obs;p=q*(obs if y[i,t] else 1-obs);p/=p.sum()
 return out

def main():
 root=Path(__file__).parent;rows=[];mixed=[]
 for path in sorted((root/'runs').glob('*/receipt.json')):
  a=json.loads(path.read_text());k,family,seed=a['k'],a['family'],a['seed'];g,law=system(k,family,seed);assert g.tolist()==a['true_graph'];train=sample(k,g,law,SPEC['train_words'],SPEC['train_length'],seed+10000);abl=[]
  for n in (128,512,1024,2048,4096):
   gg,_=passive(tuple(x[:n] for x in train),k);abl.append(dict(words=n,graph=gg.tolist(),correct=int((gg==g).sum())))
  pg=np.array(a['graphs']['passive']);ag=np.array(a['graphs']['active']);models={};causal=[]
  fixture=sample(k,g,law,32,16,seed+90000);alt=(*fixture[:-1],fixture[-1].copy());alt[-1][:,8:]^=1
  for arm,gg in a['graphs'].items():
   m=Filter(np.array(gg));m.load_state_dict(torch.load(path.parent/f'{arm}.pt',weights_only=True));m.eval();models[arm]=m
   with torch.inference_mode():p=m(*(torch.from_numpy(x) for x in fixture)).numpy();q=m(*(torch.from_numpy(x) for x in alt)).numpy()
   d=float(np.max(np.abs(p[:,:9]-q[:,:9])));assert d<1e-6;causal.append(d)
  m=GRU(k);m.load_state_dict(torch.load(path.parent/'gru64.pt',weights_only=True));m.eval()
  with torch.inference_mode():p=m(*(torch.from_numpy(x) for x in fixture)).numpy();q=m(*(torch.from_numpy(x) for x in alt)).numpy()
  d=float(np.max(np.abs(p[:,:9]-q[:,:9])));assert d<1e-6;causal.append(d)
  fixture=tuple(x[:8,:12] for x in fixture);d=float(np.max(np.abs(brute(fixture,g,law)-predict(fixture,g,law,[list(range(k))]))));assert d<1e-12
  # Extra candidate family: use passive graph, cluster caps chosen on a new validation stream.
  vals=[];candidates={};val=sample(k,g,law,256,64,seed+120000);conf=np.array(a['passive']['score']);conf=conf[np.arange(k),np.maximum(pg,0)]
  for cap in (1,2,4):
   group=clusters(pg,conf,cap);coords=sum((1<<len(v))-1 for v in group);cfg=(pg,models['passive'].rates(),group);name=f'passive_c{cap}';candidates[name]=cfg;nll=score(predict(val,*cfg),val)['nll'];vals.append(dict(arm=name,nll=nll,coords=coords,groups=group,objective=nll+SPEC['penalty']*(coords-k)))
  active=a['selected'];activegroup=next(v['groups'] for v in a['validation'] if v['arm']==active);activegraph=a['graphs'][active.split('_c')[0]];candidates['prior_selected']=(np.array(activegraph),models[active.split('_c')[0]].rates(),activegroup)
  pick=min(vals,key=lambda v:(v['objective'],v['coords'],v['arm']));candidates['passive_selected']=candidates[pick['arm']];tests=[]
  for policy in SPEC['test_policies']:
   test=sample(k,g,law,256,128,seed+130000+(0 if policy=='random' else 1000),policy);oracle=predict(test,g,law,[list(range(k))]);tests.extend(dict(arm=arm,policy=policy,**score(predict(test,*cfg),test,oracle)) for arm,cfg in candidates.items())
  rows.append(dict(k=k,family=family,seed=seed,ablation=abl,passive_validation=vals,passive_selected=pick['arm'],passive_coords=pick['coords'],causal_max=max(causal),brute_max=d,tests=tests))
 # Post-hoc stress test: mixed sparse graphs with independent roots and heterogeneous directed couplings.
 for k,seed in itertools.product((4,8),(401,409,419)):
  _,law=system(k,'independent',seed);rng=np.random.default_rng(seed+567*k);order=rng.permutation(k);g=np.full(k,-1)
  for pos in range(1,k):
   if pos%3!=0:g[order[pos]]=order[int(rng.integers(pos))]
  train=sample(k,g,law,SPEC['train_words'],SPEC['train_length'],seed+10000);p,_=passive(train,k);a,_,meta=discover(k,g,law,seed+1000000*k);mixed.append(dict(k=k,seed=seed,true_graph=g.tolist(),passive=p.tolist(),active=a.tolist(),probe_words=meta['probe_words']))
 (root/'followup_receipt.json').write_text(json.dumps(dict(rows=rows,mixed=mixed),separators=(',',':'))+'\n')
if __name__=='__main__':main()

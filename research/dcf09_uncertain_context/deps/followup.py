"""Post-primary switch-aware allocation, fixed union partition and projection audit."""
from gym import *

def union_partition(graphs,cap):
 k=len(graphs[0]);groups=[{j} for j in range(k)]
 edges=[(j,ps) for g in graphs for j,ps in enumerate(g) if ps]
 # repeated hyperedges are counted; deterministic order, no truth access.
 counts={edge:edges.count(edge) for edge in set(edges)}
 for (j,ps),_ in sorted(counts.items(),key=lambda x:(-x[1],-len(x[0][1]),x[0])):
  ids=[g for g in groups if g.intersection((j,*ps))];u=set.union(*ids)
  if len(ids)>1 and len(u)<=cap:
   for g in ids:groups.remove(g)
   groups.append(u)
  else:
   for p in ps:
    a=next(g for g in groups if j in g);b=next(g for g in groups if p in g)
    if a is not b and len(a|b)<=cap:groups.remove(a);groups.remove(b);groups.append(a|b)
 return sorted(tuple(sorted(g)) for g in groups)

def proj_check(k,seed):
 rng=np.random.default_rng(seed);old=[tuple(range(0,k//2)),tuple(range(k//2,k))];new=[tuple(range(0,k,2)),tuple(range(1,k,2))];ps=[rng.dirichlet(np.ones(1<<len(g)),size=16) for g in old];q=repartition(ps,old,new);s=0.
 for p in q:s=max(s,float(np.max(np.abs(p.sum(1)-1))))
 for j in range(k):
  a=next(i for i,g in enumerate(old) if j in g);b=next(i for i,g in enumerate(new) if j in g);p=ps[a]@bitmat(old[a])[old[a].index(j)];r=q[b]@bitmat(new[b])[new[b].index(j)];s=max(s,float(np.max(np.abs(p-r))))
 assert s<1e-12
 merged=repartition(ps,old,[tuple(range(k))])[0];prod=ps[0][:,np.arange(1<<k)%ps[0].shape[1]]*ps[1][:,np.arange(1<<k)//ps[0].shape[1]];e=float(np.max(np.abs(merged-prod)));assert e<1e-12
 return dict(marginal_max=s,coarsening_max=e)

def main(root):
 rows=[]
 for path in sorted((root/'runs').glob('*/receipt.json')):
  a=json.loads(path.read_text());k,seed=a['k'],a['seed'];graphs,true=system(k,seed);g=[[tuple(p) for p in graph] for graph in a['guesses']['full']];rate={n:np.array(v) for n,v in a['estimated'].items()};base=[[tuple(p) for p in pp] for pp in a['parts']];full=[[tuple(range(k))] for _ in g];cfg={'reset_selected':base,'full':full}
  for cap in SPEC['caps']:
   cfg[f'dynamic{cap}']=[partition(h,cap) for h in g];cfg[f'fixed{cap}']=[union_partition(g,cap)]*4
  vals=[];ctx=(np.arange(96)//8)%4;v=sample(k,graphs,true,128,96,seed+9000000,ctx)
  for name,pp in cfg.items():
   if name=='full':continue
   d=np.mean([coords(pp[z]) for z in ctx]);nll=score(predict(v,g,rate,pp),v)['nll'];vals.append(dict(arm=name,nll=nll,mean_coords=float(d),objective=nll+SPEC['penalty']*(d-k)))
  fixed=min((r for r in vals if r['arm'].startswith('fixed')),key=lambda r:(r['objective'],r['mean_coords'],r['arm']))['arm'];dyn=min((r for r in vals if not r['arm'].startswith('fixed')),key=lambda r:(r['objective'],r['mean_coords'],r['arm']))['arm'];cfg['switch_selected']=cfg[dyn];cfg['fixed_selected']=cfg[fixed];rr=[]
  for period in (1,8,32):
   ctx=(np.arange(128)//period)%4;test=sample(k,graphs,true,192,128,seed+9100000+period,ctx);oracle=predict(test,graphs,true,full)
   for name in ('reset_selected','switch_selected','fixed_selected','dynamic1','dynamic2','dynamic3','dynamic4','full'):
    pp=cfg[name];d=[coords(pp[z]) for z in ctx];pred=predict(test,g,rate,pp);rr.append(dict(period=period,arm=name,mean_coords=float(np.mean(d)),max_coords=int(max(d)),partition_switches=sum(pp[x]!=pp[y] for x,y in zip(ctx,ctx[1:])),**score(pred,test,oracle)))
  rows.append(dict(k=k,seed=seed,switch_selected=dyn,fixed_selected=fixed,validation=vals,projection_check=proj_check(k,seed),rows=rr));print(k,seed,dyn,fixed,flush=True)
 (root/'followup_receipt.json').write_text(json.dumps(rows,separators=(',',':'))+'\n')
if __name__=='__main__':main(Path(__file__).parent)

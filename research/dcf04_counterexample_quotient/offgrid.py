"""Fresh target cardinalities outside the neural model-selection grid."""
from gym import *
root=pathlib.Path(sys.argv[1]);root.mkdir(parents=True,exist_ok=True);folder=root/'checkpoints';folder.mkdir(exist_ok=True);rows=[]
for n,kind in I.product((5,9,13,20),('perm','map')):
 t,b,seed,attempt=gen(n,kind,2);u,c,nnr=neural(t,b,seed,200,folder);ss=probes(u,c);g=np.random.default_rng(seed+5000);rr=[tuple(map(int,g.integers(4,size=len(w)))) for w in ss];arms={}
 for mode in ('exact','sampled'):
  for arm,warm,pp in [('cold',(),None),('random',closure(rr),None),('neural_suffix',closure(ss),None),('neural_repair',closure(ss),(u,c))]:arms[mode+'_'+arm]=learn(t,b,mode,seed+7000,warm,pp)
 rows.append(dict(n=n,kind=kind,rep=2,seed=seed,generation_attempts=attempt,oracle_table=t.tolist(),oracle_accept=b.tolist(),neural=nnr,neural_suffixes=ss,random_suffixes=rr,arms=arms));(root/'receipt.json').write_text(json.dumps(dict(config=dict(steps=200,ks=KS,mq_budget=BUDGET,eq_cap=EQCAP),rows=rows),indent=2)+'\n')
 print(n,kind,[(a,r['exact'],r['mq']) for a,r in arms.items()],flush=True)

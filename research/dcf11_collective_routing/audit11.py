"""Independent assignment enumeration, frozen fixed replay and future-observation causality."""
from router11 import *

def objective(cost,a,gamma):return float(cost[np.arange(len(a)),a].sum()+gamma*len(set(a)))
def main():
    rng=np.random.default_rng(991); checks=[]; replays=[]
    for i in range(64):
        n=int(rng.integers(2,6)); m=int(rng.integers(2,5));c=rng.uniform(-.2,.4,(n,m));c[rng.random(c.shape)<.3]=np.inf;c[:,0]=rng.uniform(-.2,.4,n);gamma=float(rng.choice([0.,.04,.16,.3]));a=schedule(c,gamma);v=objective(c,a,gamma);b=min(objective(c,list(p),gamma) for p in itertools.product(range(m),repeat=n));err=abs(v-b);assert err<1e-12;checks.append(dict(case=i,error=err))
    for f in sorted((HERE/'results').glob('*.json')):
        rec=json.loads(f.read_text());k,seed=rec['k'],rec['seed'];g,r=system(k,seed)
        for s in rec['selection']:
            ps=[tuple(tuple(g) for g in p) for p in s['parts']];lo=np.array(s['local']);fixed=s['fixed'];d,lab,e=fixture(k,g,r,'noisy30',8,32,seed+P11['seed_offset']+3000);p=route11(d,lab,e,g,r,ps,lo,fixed,'fixed','fused')[0];q=filter_belief(d,lab,e,g,r,ps[fixed],'mixture')[0];err=float(np.max(np.abs(p-q)));assert err<1e-10
            alt=(*d[:-1],d[-1].copy());alt[-1][:,8:]^=1
            a,x,_=route11(d,lab,e,g,r,ps,lo,fixed,'cohort','fused',.04);b,y,_=route11(alt,lab,e,g,r,ps,lo,fixed,'cohort','fused',.04);causal=float(np.max(np.abs(a[:,:9]-b[:,:9])));assert causal==0 and np.array_equal(x[:,:9],y[:,:9]);replays.append(dict(k=k,seed=seed,cap=s['cap'],frozen_fixed_error=err,causal_error=causal))
    (HERE/'audit_receipt.json').write_text(json.dumps(dict(scheduler=checks,replay=replays),indent=2)+'\n');print(len(checks),max(x['error'] for x in checks),len(replays),max(x['frozen_fixed_error'] for x in replays),max(x['causal_error'] for x in replays))
if __name__=='__main__':main()

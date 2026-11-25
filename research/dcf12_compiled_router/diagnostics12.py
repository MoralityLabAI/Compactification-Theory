"""Post-primary diagnostic: shared lookup storage and existing particle baseline."""
from router12 import *

def main():
    table=[];particles=[]
    for path in sorted((HERE/'results').glob('*.json')):
        x=json.loads(path.read_text());k,seed=x['k'],x['seed'];g,r=system(k,seed)
        for s in x['selection']:
            parts=[tuple(tuple(v) for v in p) for p in s['parts']];cap=s['cap'];local=np.array(s['local']);fixed=s['fixed'];reps=[]
            for _ in range(5):
                t=time.perf_counter();cfg=config(g,parts);tab=projection_tables(parts);reps.append(time.perf_counter()-t)
            table.append(dict(k=k,seed=seed,cap=cap,layouts=len(parts),kernel_table_bytes=sum(a.nbytes for a in cfg),projection_table_bytes=sum(a.nbytes for a in tab),prep_s=float(np.median(reps))))
            for i,sc in enumerate(P12['scenarios']):
                d,lab,e=fixture(k,g,r,sc,32,64,seed+P12['seed_offset']+i*1000+32);ref=filter_belief(d,lab,e,g,r,(tuple(range(k)),),'mixture')[0]
                for rep in (0,1):
                    p,_,_,m=particle(d,lab,e,g,r,cap//2,seed+700000+100*i+rep);assert m['bytes']==cap
                    particles.append(dict(k=k,seed=seed,cap=cap,scenario=sc,rep=rep,kl=float(kl(ref,p).mean()),nll=float(loss(p,d[-1]).mean()),**m))
    (HERE/'diagnostics_receipt.json').write_text(json.dumps(dict(tables=table,particles=particles),indent=2)+'\n');print(len(table),len(particles),flush=True)
if __name__=='__main__':main()

"""Independent frozen DCF-09 replay and profile of DCF-10 routing."""
from router10 import *
from adaptive import dynamic
import cProfile,pstats,io,tracemalloc

def run_audit():
    rows=[];profiles=[]
    for path in sorted((HERE/'results').glob('*.json')):
        rec=json.loads(path.read_text());k,seed=rec['k'],rec['seed'];g,r=system(k,seed)
        for s in rec['selection']:
            parts=[tuple(tuple(h) for h in p) for p in s['parts']];local=np.array(s['local']);fixed=s['fixed']
            for i,sc in enumerate(P['scenarios']):
                d,lab,e=fixture(k,g,r,sc,8,32,seed+P['seed_offset']+2000+i)
                base=filter_belief(d,lab,e,g,r,parts[fixed],'mixture')[0];old=dynamic(d,lab,e,g,r,parts,local,fixed,1.)[0]
                f=route(d,lab,e,g,r,parts,local,fixed,'fixed')[0];p=route(d,lab,e,g,r,parts,local,fixed,'step')[0];err=max(float(np.max(np.abs(base-f))),float(np.max(np.abs(old-p))));assert err<1e-12
                rows.append(dict(k=k,seed=seed,budget=s['budget'],scenario=sc,max_delta=err))
                if (k,seed,s['budget'],sc)==(8,709,1280,'noisy30'):
                    d,lab,e=fixture(k,g,r,sc,P['test_words'],P['test_length'],seed+P['seed_offset']+1000+i)
                    for a in ('fixed','step','screen','event'):
                        pr=cProfile.Profile();pr.enable();route(d,lab,e,g,r,parts,local,fixed,a);pr.disable();buf=io.StringIO();pstats.Stats(pr,stream=buf).sort_stats('cumulative').print_stats(16)
                        tracemalloc.start();route(d,lab,e,g,r,parts,local,fixed,a);cur,peak=tracemalloc.get_traced_memory();tracemalloc.stop();profiles.append(dict(arm=a,tracemalloc_peak=peak,profile=buf.getvalue()))
    (HERE/'audit_receipt.json').write_text(json.dumps(dict(rows=rows,profiles=profiles),indent=2)+'\n');print('replay',len(rows),max(x['max_delta'] for x in rows))
    for x in profiles:print(x['arm'],x['tracemalloc_peak'],x['profile'])
if __name__=='__main__':run_audit()

"""Independent equivalence via partition refinement of disjoint union, checkpoint replay."""
from gym import *

def partition(t,b):
    p=list(map(int,b))
    while True:
        signatures=[(p[i],*(p[j] for j in t[i])) for i in range(len(t))]
        k={v:i for i,v in enumerate(dict.fromkeys(signatures))};q=[k[v] for v in signatures]
        if len(set(q))==len(set(p)):return q
        p=q

def independent(t,b,u,c):
    n=len(t);p=partition(t+[[j+n for j in row] for row in u],b+c)
    return p[0]==p[n],len(set(p[:n])),len(set(p[n:]))

def audit(root):
    summaries=[]
    for name in ('run','offgrid'):
        r=json.loads((root/name/'receipt.json').read_text());candidate_checks=active_checks=0
        for row in r['rows']:
            t,b=row['oracle_table'],row['oracle_accept'];n=row['n'];assert len(reach(t))==len(set(partition(t,b)))==n
            for cand in row['neural']['candidates']:
                m=Soft(cand['k']);m.load_state_dict(torch.load(root/name/'checkpoints'/f"{row['seed']}_{cand['k']}.pt",weights_only=True));u,c=m.hard();u,c=u.tolist(),c.tolist();same,_,_=independent(t,b,u,c);assert same==cand['exact'];candidate_checks+=1
                if cand['ce'] is not None:assert out(t,b,cand['ce'])!=out(u,c,cand['ce'])
            for a in row['arms'].values():
                assert a['mq']<=BUDGET and a['counts']['eq']<=EQCAP
                assert a['table'] is not None
                u,c=a['table'],a['accept'];same,_,_=independent(t,b,u,c);assert same==a['exact'];active_checks+=1
                if same:assert a['minimal']==n
                else:assert out(t,b,a['final_ce'])!=out(u,c,a['final_ce'])
        summaries.append(dict(split=name,systems=len(r['rows']),candidate_checks=candidate_checks,active_checks=active_checks))
    (root/'audit_receipt.json').write_text(json.dumps(summaries,indent=2)+'\n');print(summaries)
if __name__=='__main__':audit(pathlib.Path(sys.argv[1]))

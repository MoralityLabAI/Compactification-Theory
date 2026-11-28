"""Observation-query control, minimum-state witness, and independent checkpoint audit."""
from gym import *

def active():
    # Learn signatures and transitions from membership queries over primitive operations only.
    cache={}
    def query(w):
        w=tuple(w)
        assert all(o in GEN for o in w)
        if w not in cache:cache[w]=observe(w)
        return cache[w]
    probes=[()]+[(i,) for i in range(1,8)]
    def signature(w):return tuple(query(w+p) for p in probes)
    reps=[()];sigs={signature(()):0};prim=[];accept=[]
    while len(prim)<len(reps):
        w=reps[len(prim)];accept.append(query(w));row={}
        for o in GEN:
            w1=w+(o,);sig=signature(w1)
            if sig not in sigs:sigs[sig]=len(reps);reps.append(w1)
            row[o]=sigs[sig]
        prim.append(row)
        assert len(reps)<=64
    table=[[prim[prim[s][(o//8)*8]][o%8] for o in A] for s in range(len(reps))]
    return dict(states=len(reps),queries=len(cache),queried_tokens=sum(map(len,cache)),max_query_length=max(map(len,cache)),reps=reps,signatures=[signature(w) for w in reps],table=table,accept=accept,certificate=certify(table,accept))

def independent_check(table,accept):
    # Use only integer arithmetic; derive a bijection via reachability and all transition edges.
    assert len(table)==8 and len(accept)==8
    phi=[table[0][i] for i in range(8)]
    assert len(set(phi))==8 and phi[0]==0
    assert all(accept[phi[s]]==int(s==0) for s in range(8))
    for s in range(8):
        for o in range(32):
            a=[1,3,5,7][o//8];b=o&7
            assert table[phi[s]][o]==phi[(a*s+b)&7]
    return phi

@torch.inference_mode()
def compare(root):
    results=[]
    for folder in ('run','long_run'):
        rec=json.loads((root/folder/'receipt.json').read_text())
        for a in rec['audits']:
            if a['arm']!='latent8':continue
            m=Automaton();m.load_state_dict(torch.load(root/folder/f"{a['seed']}_latent8.pt",weights_only=True));m.eval()
            table=m.table();accept=[1]+[0]*7;assert table==a['table'];cert=certify(table,accept)
            row=dict(seed=a['seed'],steps=rec['steps'],cert=cert,table=table,accept=accept,permutation_rows=all(len(set(col))==8 for col in zip(*table)),tests=[])
            if cert['exact']:row['isomorphism']=independent_check(table,accept)
            for l in (4,16,64,256):
                for split,alph in [('mixed',A),('held_only',HOLD)]:
                    x,y=sample(torch.Generator().manual_seed(80000+1000*a['seed']+l),256,l,alph)
                    tab=torch.tensor(table);state=torch.zeros(len(x),dtype=torch.long);p=[]
                    for op in x.T:state=tab[state,op];p.append(state==0)
                    p=torch.stack(p,1);z=y.bool();soft=metrics(m,x,y)
                    row['tests'].append(dict(length=l,split=split,n=len(x),hard_trace_correct=int((p==z).all(1).sum()),hard_final_correct=int((p[:,-1]==z[:,-1]).sum()),soft_trace_correct=soft['trace_correct'],soft_final_correct=soft['final_correct']))
            results.append(row)
    return results

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args();a=active();a['isomorphism']=independent_check(a['table'],a['accept']);r=compare(args.root)
    # Each reachable state has a distinguishing additive suffix: for state i, add -i mod 8.
    witnesses=[]
    for i,j in itertools.combinations(range(8),2):
        suffix=(-i)%8;assert observe((i,suffix))!=observe((j,suffix));witnesses.append([i,j,suffix])
    out=dict(active=a,learned=r,minimality_witnesses=witnesses)
    (args.root/'audit_receipt.json').write_text(json.dumps(out,indent=2)+'\n'); print('active',a['states'],a['queries'],a['queried_tokens'],a['max_query_length'],a['certificate'])
    for row in r:print(row['seed'],row['cert']['exact'],row.get('isomorphism'),[(t['length'],t['split'],t['hard_trace_correct'],t['soft_trace_correct']) for t in row['tests'] if t['length'] in (64,256)])
if __name__=='__main__':main()

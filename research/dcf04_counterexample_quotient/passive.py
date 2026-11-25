"""Passive red-blue state merging; observation-only, no membership/equivalence teacher."""
from gym import *

def samples(t,b,seed,include_val):
    labels={():int(b[0])}
    streams=[data(np.array(t),np.array(b),np.random.default_rng(seed+90),200*48,4)]
    if include_val:streams.append(data(np.array(t),np.array(b),np.random.default_rng(seed+91),256,8))
    for x,y in streams:
        for w,z in zip(x.tolist(),y.tolist()):
            for i,bit in enumerate(z,1):
                p=tuple(w[:i]);v=int(bit);assert labels.get(p,v)==v;labels[p]=v
    return labels

def merge(labels):
    words=sorted(labels,key=lambda w:(len(w),w));idx={w:i for i,w in enumerate(words)};n=len(words);trans=[{a:idx[w+(a,)] for a in A if w+(a,) in idx} for w in words];accept=[labels[w] for w in words];parent=list(range(n));red={0};attempts=0
    def find(p,i):
        while p[i]!=i:i=p[i]
        return i
    def trial(r,b):
        p=parent.copy();t=[v.copy() for v in trans];q=[(r,b)]
        for i,j in q:
            i,j=find(p,i),find(p,j)
            if i==j:continue
            if accept[i]!=accept[j]:return None
            p[j]=i
            for a,h in t[j].items():
                if a in t[i]:q.append((t[i][a],h))
                else:t[i][a]=h
        return p,t
    while True:
        red={find(parent,i) for i in red};blue={find(parent,h) for i in red for h in trans[i].values()}-red
        if not blue:break
        b=min(blue);success=False
        for r in sorted(red):
            attempts+=1;result=trial(r,b)
            if result is not None:parent,trans=result;success=True;break
        if not success:red.add(b)
    start=find(parent,0);q=[start];seen={start:0};u=[];c=[];sink=-1
    for s in q:
        c.append(0 if s==sink else accept[s]);row=[]
        for a in A:
            h=find(parent,trans[s][a]) if s!=sink and a in trans[s] else sink
            if h not in seen:seen[h]=len(q);q.append(h)
            row.append(seen[h])
        u.append(row)
    assert all(out(u,c,w)==v for w,v in labels.items())
    return u,c,attempts

def main(root):
    result=[]
    for split in ('run','offgrid'):
        rec=json.loads((root/split/'receipt.json').read_text())
        for row in rec['rows']:
            t,b,seed=row['oracle_table'],row['oracle_accept'],row['seed']
            for mode in (False,True):
                labels=samples(t,b,seed,mode);start=time.perf_counter();u,c,attempts=merge(labels);sec=time.perf_counter()-start;ce,p,e=eq(t,b,u,c)
                result.append(dict(split=split,seed=seed,n=row['n'],kind=row['kind'],train_plus_validation=mode,unique_words=len(labels),states=len(u),minimal=minimal(u,c),merge_attempts=attempts,seconds=sec,exact=ce is None,ce=ce,table=u,accept=c))
    (root/'passive_receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    for mode in (False,True):
        rows=[r for r in result if r['train_plus_validation']==mode];print(mode,sum(r['exact'] for r in rows),len(rows),sum(r['unique_words'] for r in rows),sum(r['seconds'] for r in rows),[(r['n'],r['kind'],r['split']) for r in rows if r['exact']])
if __name__=='__main__':main(pathlib.Path(sys.argv[1]))

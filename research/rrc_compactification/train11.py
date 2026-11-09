"""RRC-11 training-only linear preview ranker, deterministic standard library."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
try:
    from .graph11 import GRAPHS,TRAIN_GRAPHS,data,preview_score,digest
except ImportError:
    from graph11 import GRAPHS,TRAIN_GRAPHS,data,preview_score,digest
HERE=Path(__file__).resolve().parent

def features(case,graph):
    s=preview_score(case,graph)
    return (s[0]/8.,s[1]/9.,s[2]/100.,s[3]/100.)

def train():
    cases=data(include_test=False)['train']
    # Label is a canonical training-grammar program; full prefix execution
    # supplies candidate correctness, only on the training roster.
    from graph11 import execute,truncate
    w=[0.]*4
    updates=0
    for epoch in range(12):
        for c in cases:
            candidates=[]
            for g in TRAIN_GRAPHS:
                f=features(c,g)
                candidate=all(execute(truncate(x,12),g)==truncate(y,12) for x,y in c.demos)
                candidates.append((g,f,candidate))
            valid=[z for z in candidates if z[2]]
            if not valid:raise AssertionError('missing training graph')
            positive=valid[0]
            negative=max((z for z in candidates if not z[2]),key=lambda z:sum(a*b for a,b in zip(w,z[1])))
            margin=sum(a*(p-n) for a,p,n in zip(w,positive[1],negative[1]))
            if margin<.1:
                w=[a+.04*(p-n) for a,p,n in zip(w,positive[1],negative[1])]
                updates+=1
    return {'schema':'rrc11.rank.v1','weights':w,'updates':updates,'train_count':len(cases),
            'train_digest':digest(cases),'allowed_supervised_graphs':[list(x) for x in TRAIN_GRAPHS],
            'graph11_sha256':hashlib.sha256((HERE/'graph11.py').read_bytes()).hexdigest(),
            'train11_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'training_uses_graph_labels':True,
            'inference_executes_all_public_graph_previews':True}

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=HERE/'results/rrc11_ranker.json');a=p.parse_args()
    r=train();a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n');print('ranker',r['weights'],r['updates'])
if __name__=='__main__':main()

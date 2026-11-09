"""RRC-10 independent held-out exact verifier and source-locked receipt."""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path
try:
    from . import primitive10 as p
except ImportError:
    import primitive10 as p
HERE=Path(__file__).resolve().parent
SOURCES=('primitive10.py','train10.py','bench10.py','test_primitive10.py')

def sha(path):return sha256(Path(path).read_bytes()).hexdigest()

def load():
    out={}
    for k in ('recurrent','local'):
        path=HERE/'results'/f'rrc10_{k}.json'
        d=json.loads(path.read_text())
        assert d['primitive10_sha256']==sha(HERE/'primitive10.py')
        assert d['train10_sha256']==sha(HERE/'train10.py')
        assert d['training_input_sha256']==p.digest({'train':p.episodes(include_test=False)['train']})['train']
        out[k]=(d,sha(path))
    return out

def score(case,ans):
    accepted=bool(ans['accepted'])
    return {'accepted':int(accepted),
            'future_correct':int(accepted and ans['prediction']==case.future),
            'false_accept':int(accepted and ans['prediction']!=case.future),
            'abstention':int(not accepted)}

def summarize(rows):
    return {'cases':len(rows), **{key:sum(row[key] for row in rows) for key in ('accepted','future_correct','false_accept','abstention')}}

def run():
    m=load();groups=p.episodes()
    methods={'recurrent':p.propose,'local':lambda case,w:p.propose(case,w,local=True),
             'exact_inversion':lambda case,w:p.reference(case),
             'no_demonstration':lambda case,w: p.propose(case,(0,0))}
    result={}
    for name,func in methods.items():
        weights=m[name][0]['compiled_integer_weights'] if name in m else [0,0]
        if weights is None:weights=(0,0)
        split={}
        for label,cases in groups.items():
            if label=='train':continue
            split[label]=summarize([score(c,func(c,weights)) for c in cases])
        result[name]={'splits':split,'aggregate':summarize([score(c,func(c,weights)) for key,v in groups.items() if key!='train' for c in v])}
    pairs=groups['late_collision']
    assert len(pairs)==16
    collision=[]
    for left,right in zip(pairs[::2],pairs[1::2]):
        assert left.demos==right.demos and left.query==right.query and left.future!=right.future
        a=p.reference(left);b=p.reference(right)
        assert a==b
        collision.append({'pair':left.pair,'indistinguishable':True,'correct_members':int(a['prediction']==left.future)+int(a['prediction']==right.future)})
    return {'schema':'rrc10.receipt.v1','source_sha256':{s:sha(HERE/s) for s in SOURCES},
            'checkpoint_sha256':{name:v[1] for name,v in m.items()},
            'roster_sha256':p.digest(groups),
            'parameters':{'demo_degree':p.DEMO,'future_degree':p.HORIZON,
                          'train_k':p.TRAIN_K,'test_k':p.TEST_K,
                          'one_hidden_multiplier_per_episode':True,
                          'two_demonstrations_per_episode':True,
                          'learned_integer_weights':m['recurrent'][0]['compiled_integer_weights'],
                          'public_causal_basis':True,'public_wrapper_interpreter':True},
            'metrics':result,'collision_pairs':collision,
            'scope':'Small polynomial multiplier family, exact teacher-forced causal basis; valid prefix is a finite certificate only; not novel algebra or a new primitive graph.'}

def main():
    par=argparse.ArgumentParser();par.add_argument('--out',type=Path,default=HERE/'results'/'rrc10_pilot.json');par.add_argument('--check',action='store_true');a=par.parse_args()
    o=json.dumps(run(),indent=2,sort_keys=True)+'\n'
    if a.check:
        if not a.out.exists() or a.out.read_text()!=o:raise SystemExit('RRC-10 stale or missing receipt')
        print('RRC-10 exact receipt verified')
    else:
        a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(o)
        print(json.dumps(json.loads(o)['metrics'],indent=2))
if __name__=='__main__':main()

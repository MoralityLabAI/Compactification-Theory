"""RRC-10 exact rational least-squares operator fitting, training-only data.

Two fitted coefficients feed a reusable recurrent causal state update. No
PyTorch dependency, GPU, or held-out primitive labels are used in fitting.
"""
from __future__ import annotations
from fractions import Fraction
import argparse
import hashlib
import json
from pathlib import Path
try:
    from .primitive10 import episodes,DEMO,digest
except ImportError:
    from primitive10 import episodes,DEMO,digest
HERE=Path(__file__).resolve().parent


def fit(*,local=False):
    train=episodes(include_test=False)['train']
    x11=x12=x22=y1=y2=0
    rows=0
    for case in train:
        for f,g in case.demos:
            # Teacher h is derived solely from training pairs, independently of
            # source template labels, via exact finite formal-series division.
            from_h=[1]
            for n in range(1,DEMO+1):
                from_h.append(g[n]-sum(f[j]*from_h[n-j] for j in range(1,n+1)))
            for n in range(1,DEMO+1):
                features=(g[n],f[n]) if local else (g[n],sum(f[j]*from_h[n-j] for j in range(1,n+1)))
                a,b=features;target=from_h[n]
                x11+=a*a;x12+=a*b;x22+=b*b
                y1+=a*target;y2+=b*target;rows+=1
    det=x11*x22-x12*x12
    if det<=0:raise AssertionError('rank deficient design')
    w=(Fraction(y1*x22-y2*x12,det),Fraction(x11*y2-x12*y1,det))
    snapped=tuple(round(q) for q in w)
    accepted=all(abs(q-s)<=Fraction(1,8) and abs(s)<=2 for q,s in zip(w,snapped))
    return {'schema':'rrc10.weights.v1','kind':'raw_local' if local else 'causal_recurrent',
            'samples':len(train),'supervised_rows':rows,'degree_supervised':DEMO,
            'training_input_sha256':digest({'train':train})['train'],
            'primitive10_sha256':hashlib.sha256((HERE/'primitive10.py').read_bytes()).hexdigest(),
            'train10_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'raw_fraction_weights':[[v.numerator,v.denominator] for v in w],
            'compiled_integer_weights':list(snapped) if accepted else None,
            'compilation_threshold':'absolute error <= 1/8; signed weights <= 2',
            'uses_teacher_traces_during_training':True,'uses_teacher_at_inference':False}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out-dir',type=Path,default=HERE/'results')
    a=p.parse_args();a.out_dir.mkdir(parents=True,exist_ok=True)
    for kind,local in [('recurrent',False),('local',True)]:
        d=fit(local=local)
        path=a.out_dir/f'rrc10_{kind}.json'
        path.write_text(json.dumps(d,sort_keys=True,indent=2)+'\n')
        print(kind,d['raw_fraction_weights'],d['compiled_integer_weights'])
if __name__=='__main__':main()

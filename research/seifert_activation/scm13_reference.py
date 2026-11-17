"""Source-fixed synthetic reference for the affine-disk / provenance SCM-13 campaign."""
from __future__ import annotations
import argparse
import json
import math
import random
from pathlib import Path

from scm13_affine_disk import affine_disk_certificate,taylor_errors_from_hessian
from scm13_diagnostic import screen_frames
from knot import synthetic_braid
from nonabelian import artin_witness
from winding import pairwise_windings


def _frames(base, matrices,radius,n=256):
    ans=[]
    for k in range(n+1):
        c,s=(1.,0.) if k==n else (math.cos(2*math.pi*k/n),math.sin(2*math.pi*k/n))
        ans.append([(float(a[0])+radius*(float(B[0][0])*c+float(B[0][1])*s),
                     float(a[1])+radius*(float(B[1][0])*c+float(B[1][1])*s)) for a,B in zip(base,matrices)])
    return ans


def generate():
    a=[[0,0],[2,0]]
    B=[[[0,0],[0,0]],[[1,0],[0,1]]]
    below=affine_disk_certificate(a,B,1,[0,0])
    above=affine_disk_certificate(a,B,3,[0,0])
    screen=screen_frames(_frames(a,B,3,64))
    rng=random.Random(1313)
    checked=0
    null=0
    nontrivial=0
    mismatch=0
    attempted=0
    while checked<80 and attempted<3000:
        attempted+=1
        a=[[rng.randint(-4,4),rng.randint(-4,4)] for _ in range(2)]
        B=[[[rng.randint(-3,3),rng.randint(-3,3)] for _ in range(2)] for _ in range(2)]
        r=rng.choice((0.25,0.5,1,2,4))
        cert=affine_disk_certificate(a,B,r,[0,0],margin='1/100')
        if cert['result']=='inconclusive':
            continue
        obs=pairwise_windings(_frames(a,B,r,256))[(0,1)]
        if obs!=cert['per_pair']['0,1']['pair_winding']:
            mismatch+=1
        if cert['result']=='conditional_full_null_braid':
            null+=1
        else:
            nontrivial+=1
        checked+=1
    C=[1,1,2,2,-1,-1,-2,-2]
    braid=synthetic_braid(3,C)
    commutator_pair=pairwise_windings(braid)
    artin=artin_witness(3,C)
    return {
        'schema':'scm13/synthetic-reference/v1',
        'evidence':'exact-rational geometry and deliberately synthetic polygonal braids',
        'two_strand_below_radius_2':{'result':below['result'],'winding':below['per_pair']['0,1']['pair_winding']},
        'two_strand_above_radius_2':{'result':above['result'],'winding':above['per_pair']['0,1']['pair_winding']},
        'first_harmonic_screen':{'status':screen['status'],'predicted':screen['affine_conditional_check']['per_pair']['0,1']['pair_winding'],
                                 'observed':screen['polygonal_pair_windings']['0,1']},
        'seeded_affine_cases':{'seed':1313,'passed':checked,'mismatched':mismatch,'null':null,'winding_nonzero':nontrivial},
        'nonabelian_commutator':{'pairwise_windings':{f'{i},{j}':v for (i,j),v in commutator_pair.items()},
                                 'artin_nonidentity':not artin['is_identity'],
                                 'interpretation':'nontrivial zero-winding braid requires departure from any certified full-rank affine-disk approximation'},
        'trained_neural_evidence':False,
    }


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--check',action='store_true')
    args=p.parse_args(argv)
    raw=json.dumps(generate(),indent=2,sort_keys=True,allow_nan=False)+'\n'
    if args.check:
        if args.out.read_text(encoding='utf-8')!=raw:
            raise SystemExit('SCM-13 synthetic reference replay changed')
        print('SCM-13 reference replay OK')
    else:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        with args.out.open('x',encoding='utf-8') as f:f.write(raw)
        print('SCM-13 synthetic reference written')

if __name__=='__main__':
    main()

"""NHB-01: exact cost of binary decisions and posterior-threshold interfaces.

Encoder computation, posterior tables and coefficient descriptions are
additional resources beyond output-state bits. Finite laws are exhaustive;
reported Shannon information is numerical.
"""
from __future__ import annotations
from collections import defaultdict
from fractions import Fraction as F
import argparse
import hashlib
import json
import math
from pathlib import Path
try:
    from .reference import (affine_projection, crossover, entropy,
        examples, influence, noise_rate, noise_signal, signs)
except ImportError:
    from reference import (affine_projection, crossover, entropy,
        examples, influence, noise_rate, noise_signal, signs)


def encoded_channel(posterior,codes):
    if not posterior or len(posterior)!=len(codes) or any(abs(x)>1 for x in posterior):
        raise ValueError('matched finite conditional means and codes required')
    cells=defaultdict(lambda:[F(),F()]); N=len(posterior)
    for value,code in zip(posterior,codes):
        cells[code][0]+=(1-value)/(2*N)
        cells[code][1]+=(1+value)/(2*N)
    prior=sum((row[1] for row in cells.values()),F())
    error=sum((min(row) for row in cells.values()),F())
    info=entropy(prior)-math.fsum(float(sum(row))*entropy(row[1]/sum(row))
                                for row in cells.values() if sum(row))
    return {'states':len(cells),'fixed_length_bits':(len(cells)-1).bit_length(),
            'optimal_decoding_error':str(error),'optimal_decoding_error_float':float(error),
            'information_bits_numeric':max(0.0,info)}


def all_threshold_certificate(posterior):
    if not posterior or any(abs(x)>1 for x in posterior):
        raise ValueError('nonempty bounded conditional-mean vector required')
    levels=sorted(set(F(1+x)/2 for x in posterior))
    codes={p:i for i,p in enumerate(levels)}
    thresholds=[(a+b)/2 for a,b in zip(levels,levels[1:])]
    responses=[tuple(p>=t for t in thresholds) for p in levels]
    if len(set(responses))!=len(levels):
        raise AssertionError('distinct posterior levels lack a separating threshold')
    return {'posterior_levels':[str(x) for x in levels],
            'separating_thresholds':[str(x) for x in thresholds],
            'minimum_code_states':len(levels),
            'minimum_fixed_length_bits':(len(levels)-1).bit_length(),
            'code':[codes[F(1+x)/2] for x in posterior],
            'preserves_all_posterior_thresholds':True}


def comparison(truth,p,eta=F(1,10)):
    truth=signs(truth); p=noise_rate(p); eta=noise_rate(eta); a=1-2*eta
    posterior=noise_signal(tuple(a*x for x in truth),p)
    optimal_bit=tuple(1 if v>=0 else -1 for v in posterior)
    affine=affine_projection(truth)
    channels={
        'optimal_decision_bit':encoded_channel(posterior,optimal_bit),
        'same_function_bit':encoded_channel(posterior,truth),
        'affine_surrogate_bit':encoded_channel(posterior,affine),
        'full_state':encoded_channel(posterior,tuple(range(len(truth)))),
    }
    if channels['optimal_decision_bit']['optimal_decoding_error']!=channels['full_state']['optimal_decoding_error']:
        raise AssertionError('Bayes decision bit lost its optimal accuracy')
    cert=all_threshold_certificate(posterior)
    threshold_channel=encoded_channel(posterior,cert['code'])
    if abs(threshold_channel['information_bits_numeric']-channels['full_state']['information_bits_numeric'])>1e-11:
        raise AssertionError('posterior-value code lost information')
    alpha=F(sum(x!=y for x,y in zip(truth,affine)),len(truth))
    Ah=sum(influence(affine))
    sufficient=min(F(1,2),eta+a*(alpha+min(F(1,2),p*Ah)))
    if F(channels['affine_surrogate_bit']['optimal_decoding_error'])>sufficient:
        raise AssertionError('approximate-probe accuracy certificate failed')
    return {'noise':str(p),'target_flip':str(eta),'channels':channels,
            'uniform_surrogate_error':str(alpha),'surrogate_influence':str(Ah),
            'surrogate_error_upper':str(sufficient),
            'threshold_menu':{k:v for k,v in cert.items() if k!='code'},
            'static_encoder_and_decoder_cost_charged_separately':True}


def report():
    rows=[]
    for name,n,terms,truth in examples():
        row=comparison(truth,crossover(F(1,20),4))
        row.update(name=name,n=n)
        rows.append(row)
    return {'schema':'nhb01.decision_memory.v1','status':'EXACT_FINITE_DECISION_MEMORY',
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'reference_sha256':hashlib.sha256(Path(__file__).with_name('reference.py').read_bytes()).hexdigest(),
            'horizon':4,'cases':rows,'trained_checkpoint_evaluated':False}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path)
    args=p.parse_args(argv); r=report()
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(r,sort_keys=True,indent=2)+'\n')
    brief=[{'name':x['name'],'Bayes_error':x['channels']['full_state']['optimal_decoding_error_float'],
            'affine_error':x['channels']['affine_surrogate_bit']['optimal_decoding_error_float'],
            'full_information':x['channels']['full_state']['information_bits_numeric'],
            'decision_bit_information':x['channels']['optimal_decision_bit']['information_bits_numeric'],
            'all_threshold_states':x['threshold_menu']['minimum_code_states'],
            'all_threshold_bits':x['threshold_menu']['minimum_fixed_length_bits']} for x in r['cases']]
    print(json.dumps({'status':r['status'],'cases':brief},indent=2))
    return 0

if __name__=='__main__':raise SystemExit(main())

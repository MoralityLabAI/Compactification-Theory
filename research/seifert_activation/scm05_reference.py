"""Reproducible SCM-05 exact-rational CPU reference on synthetic trajectories."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from knot import synthetic_braid, extract_closed_braid, InvalidBraidTrace
from milnor import triple_milnor_nilpotent
from continuous import continuous_sampling_certificate, allowable_pair_lipschitz_ceiling, _sqrt_bounds, norm_squared, rat
from quantization import uniform_quantize, quantization_bit_knee
from winding import pairwise_windings

HERE=Path(__file__).resolve().parent
C=(1,1,2,2,-1,-1,-2,-2)


def ring(n:int,k:int,R:float=2.0):
    out=[]
    for index in range(n+1):
        angle=2*math.pi*k*index/n
        out.append([[R*math.cos(angle),R*math.sin(angle)],[0.,0.]])
    out[-1]=[p[:] for p in out[0]]
    return out


def gap(n:int):
    return [str(math.nextafter(2*math.pi/n,math.inf))]*n


def refine_poly(frames,subdiv=4):
    refined=[]
    for start,end in zip(frames,frames[1:]):
        for step in range(subdiv):
            x=step/subdiv
            refined.append([[a*(1-x)+b*x for a,b in zip(p,q)] for p,q in zip(start,end)])
    refined.append([list(z) for z in frames[-1]])
    return refined


def exact_PL_upper_bounds(frames):
    """For *this fixture*, q is defined as PL on unit-length time intervals.

    Each pair's exact chord norm is a Lipschitz bound. We round the bound
    upward in float before decimal encoding; the fixture's true interval
    function is precisely the piecewise-linear interpolation of these samples.
    """
    n=len(frames[0]); pairs={f'{i},{j}':[] for i in range(n) for j in range(i+1,n)}
    for a,b in zip(frames,frames[1:]):
        for i in range(n):
            for j in range(i+1,n):
                delta=tuple((rat(b[i][d])-rat(b[j][d])) - (rat(a[i][d])-rat(a[j][d])) for d in range(2))
                # Bound the *exact rational* recorded PL derivative upward.
                bound=_sqrt_bounds(norm_squared(delta),18)[1]
                pairs[f'{i},{j}'].append(str(bound))
    return pairs


def produce():
    sparse=ring(4,4)
    dense=ring(64,4)
    bs={'0,1':'8.000001'}
    sparse_cert=continuous_sampling_certificate(sparse,interval_lengths=gap(4),
                                               pair_lipschitz_bounds=bs)
    dense_cert=continuous_sampling_certificate(dense,interval_lengths=gap(64),
                                              pair_lipschitz_bounds=bs)
    assert not sparse_cert['passes'] and dense_cert['passes']
    assert pairwise_windings(sparse)[0,1]==0
    assert pairwise_windings(dense)[0,1]==4
    borromean=refine_poly(synthetic_braid(3,C),4)
    original=extract_closed_braid(synthetic_braid(3,C))
    assert triple_milnor_nilpotent(3,original.word)['mu123']==1
    plbounds=exact_PL_upper_bounds(borromean)
    knee=quantization_bit_knee(borromean,range_abs='3',
            interval_lengths=['1']*(len(borromean)-1),
            pair_lipschitz_bounds=plbounds,max_bits=16)
    assert knee['certified']
    quantized=uniform_quantize(borromean,bits=knee['bit_knee'],range_abs='3')
    try:
        extract_closed_braid(quantized)
    except InvalidBraidTrace as exc:
        extractor='rejected: '+str(exc)
    else:
        extractor='accepted'
    receipt={
        'schema':'scm05_cpu_reference_v1',
        'evidence':'analytically defined synthetic circles and PL pure braid, no neural data',
        'model_artifacts_available':False,
        'aliasing':{
            'hidden_winding':4,
            'sparse_samples':4,
            'sparse_reported_winding':pairwise_windings(sparse)[0,1],
            'sparse_conditional_certified':sparse_cert['passes'],
            'dense_samples':64,
            'dense_reported_winding':pairwise_windings(dense)[0,1],
            'dense_conditional_certified':dense_cert['passes'],
            'assumed_analytic_pair_L':8.000001,
        },
        'borromean_compactification':{
            'original_braid_word':list(original.word),
            'mu123_of_original':triple_milnor_nilpotent(3,original.word)['mu123'],
            'refined_frame_count':len(borromean),
            'independently_known_source_type':'synthetic_piecewise_linear',
            'quantization_range':3,
            'worst_case_certified_bit_knee':knee['bit_knee'],
            'quantization_certificate_passed':knee['certificate']['passes'],
            'quantization_worst_case_passed':knee['certificate']['worst_case_passes'],
            'post_quantization_crossing_extractor':extractor,
            'invariant_preservation':'implied_by_continuous_configuration_homotopy',
        },
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    return receipt


def main():
    out=HERE/'results'/'scm05_cpu_reference.json'
    out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():
        raise FileExistsError('Refusing to overwrite existing immutable reference')
    result=produce()
    out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__=='__main__':
    main()

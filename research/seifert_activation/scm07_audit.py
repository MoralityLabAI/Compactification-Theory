"""SCM-07: inspect existing RMP braid samples using derived tokenwise bounds.

Source-global token bounds are real-analytic architecture theorems. Optional
center-local bounds additionally assume independently verified ideal-center
error envelopes. This reader cannot prove that PyTorch coordinates enclose the
ideal real-valued output, so no observed-link certificate is marked final.
"""
from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
import json
from pathlib import Path

from continuous import continuous_sampling_certificate
from rmp_lipschitz import angular_gap_upper


def audit_trace(trace: Mapping, bounds: Mapping, *, margin='1/10000000') -> dict:
    if bounds.get('schema') != 'scm07_tokenwise_real_rmp_v1':
        raise ValueError('Wrong source-locked tokenwise bound schema')
    frames=trace.get('frames_by_visit')
    if not isinstance(frames,list) or not frames:
        raise ValueError('Missing activation samples')
    n=len(frames[0])-1
    if n<8 or len(bounds['per_visit_pair_bounds'])!=len(frames):
        raise ValueError('Angular sample/visit counts mismatch')
    if any(len(v)!=n+1 for v in frames):
        raise ValueError('Inconsistent angle count')
    # The capture protocol is uniform theta_k=2*pi*k/n. We must test the
    # supplied actual angles rather than accepting arbitrary samples.
    angles=trace.get('angles')
    if not isinstance(angles,list) or len(angles)!=n+1:
        raise ValueError('Missing uniform angular grid')
    import math
    for k,v in enumerate(angles):
        if abs(float(v)-2*math.pi*k/n)>1e-9:
            raise ValueError('Nonuniform cyclic angles')
    gap=angular_gap_upper(n)
    inspected=[]
    for visit, (observations,pair) in enumerate(zip(frames,bounds['per_visit_pair_bounds']),1):
        result=continuous_sampling_certificate(observations,
                interval_lengths=[str(gap)]*n,
                pair_lipschitz_bounds=pair,extra_margin=margin)
        # Even a positive geometric inequality does not close the proof gap
        # from floating measurements to ideal continuous real model.
        inspected.append({'visit':visit,
                          'passes_ideal_sampling_inequality':bool(result['passes']),
                          'observed_link_certified':False,
                          'status':result['reason'],
                          'worst_margins':{p:{'source':v['worst_source_margin_lower']}
                                           for p,v in result['pairs'].items()}})
    return {'schema':'scm07_source_bound_sampling_audit_v1',
            'source_bound_scope':bounds['bound_scope'],
            'source_global_derivative_proved':bounds['bound_scope']=='source_global_unconditional',
            'ideal_center_enclosures_proved':False,
            'observed_link_certified':False,
            'requires_exact_ideal_activation_or_roundoff_enclosure':True,
            'uniform_gap_upper':str(gap),
            'visits':inspected}


def audit_jsonl(source: Path, output: Path, *, bounds: Mapping):
    """Source-locked immutable input: create a new report, never mutate data."""
    if output.exists():
        raise FileExistsError('Refusing to replace SCM audit output')
    rows=[json.loads(x) for x in source.read_text().splitlines() if x.strip()]
    if not rows:
        raise ValueError('Empty captured RMP records')
    report=[audit_trace(row['capture'],bounds) for row in rows]
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as writer:
        for row in report:
            writer.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
    return {'records':len(report), 'observed_link_certificates':0}


def main(argv=None):
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--captures',type=Path,required=True,
                        help='Immutable SCM-04 RMP JSONL capture, read-only')
    parser.add_argument('--bounds',type=Path,required=True,
                        help='SCM-07 model-derived tokenwise bounds JSON, one model/task contract')
    parser.add_argument('--out',type=Path,required=True,
                        help='Fresh output path; existing files are refused')
    args=parser.parse_args(argv)
    bounds=json.loads(args.bounds.read_text(encoding='utf-8'))
    print(json.dumps(audit_jsonl(args.captures,args.out,bounds=bounds),indent=2))


if __name__=='__main__':
    main()

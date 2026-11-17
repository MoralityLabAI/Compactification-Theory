"""Post-process SCM-04 RMP JSONL receipts with SCM-05 conditional topology bounds.

Read-only: does not load model weights, alter any checkpoints, or touch sealed
RMP evaluation splits. An external, independent analysis must justify all
interval Lipschitz upper bounds supplied in the bound-claim file.

Example:
  python scm05_audit.py --input exploratory.jsonl --claims claims.json \
      --out scm05_audit.jsonl --quant-range 3 --max-bits 16

Claim format:
{
  "schema": "scm05_interval_lipschitz_claims_v1",
  "scope": "registered_external_upper_bounds",
  "derivation_reference": "a content-addressed mathematical proof or validated bounder report",
  "by_visit": {"1":{"0,1":"1.5", "0,2":"2.5", "1,2":"2.0"}, ...}
}

A missing claim gives an explicit unverified status; the supplied bounds
are assumptions, not evidence of an independently validated proof.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping

from continuous import continuous_sampling_certificate, allowable_pair_lipschitz_ceiling
from quantization import quantization_bit_knee

SCHEMA='scm05_continuous_braid_conditional_audit_v1'
CLAIMS='scm05_interval_lipschitz_claims_v1'


def audit_record(record: Mapping, claims: Mapping | None, *,
                 quant_range: str | None = None, max_bits: int = 16,
                 extra_margin='0') -> dict:
    trace=record.get('capture')
    if not isinstance(trace,Mapping):
        raise ValueError('Input record requires RMP capture')
    angles=trace['angles']
    frames_by_visit=trace['frames_by_visit']
    if len(angles)<2 or len(frames_by_visit)<1:
        raise ValueError('Invalid capture dimensions')
    intervals=[angles[i+1]-angles[i] for i in range(len(angles)-1)]
    if any(x<=0 for x in intervals):
        raise ValueError('Angular coordinates must increase')
    if any(len(frames)!=len(angles) for frames in frames_by_visit):
        raise ValueError('Visit capture frame/angle count mismatch')
    if claims is not None and claims.get('schema') != CLAIMS:
        raise ValueError('Invalid external-claim schema')
    if claims is not None:
        locked=claims.get('source_sha256',{})
        if not isinstance(locked,Mapping):
            raise ValueError('source_sha256 must be a mapping')
        for key in ('checkpoint_sha256','dictionary_sha256','chart_sha256'):
            if key in locked and record.get(key)!=locked[key]:
                raise ValueError(f'Bound claim {key} does not match source capture')
    by_visit={} if claims is None else claims.get('by_visit',{})
    if not isinstance(by_visit,Mapping):
        raise ValueError('Claim by_visit must be a mapping')
    reports=[]
    for visit,frames in enumerate(frames_by_visit,1):
        bounds=by_visit.get(str(visit))
        diagnostic=allowable_pair_lipschitz_ceiling(frames,interval_lengths=intervals,
                          extra_margin=extra_margin)
        if bounds is None:
            reports.append({'visit':visit,'status':'missing_external_interval_bounds',
                            'independently_certified':False,'diagnostic':diagnostic})
            continue
        sampling=continuous_sampling_certificate(frames,interval_lengths=intervals,
                        pair_lipschitz_bounds=bounds,extra_margin=extra_margin)
        output={'visit':visit,
                'status':'conditional_pass' if sampling['passes'] else 'conditional_refusal',
                'conditional_on_unverified_external_bounds':True,
                'conditional_math_satisfied':sampling['passes'],
                'independently_certified':False,
                'diagnostic':diagnostic,
                'sampling':sampling}
        if quant_range is not None:
            output['quantization']=quantization_bit_knee(frames,range_abs=quant_range,
                      interval_lengths=intervals,pair_lipschitz_bounds=bounds,
                      max_bits=max_bits,extra_margin=extra_margin)
        reports.append(output)
    return {'schema':SCHEMA,'source_schema':record.get('schema','unknown'),
            'checkpoint_sha256':record.get('checkpoint_sha256'),
            'dictionary_sha256':record.get('dictionary_sha256'),
            'chart_sha256':record.get('chart_sha256'),
            'source_task_sha256':record.get('task_id_sha256'),
            'external_bounds_provenance':None if claims is None else
                {'scope':claims.get('scope'),'derivation_reference':claims.get('derivation_reference')},
            'visits':reports,
            'scientific_scope':'conditional mathematics on chart-projected trajectories; no causal or 4D surface claim'}


def run(input_path:Path, output_path:Path, *, claims_path:Path | None=None,
        quant_range:str | None=None, max_bits:int=16,extra_margin='0') -> dict:
    if output_path.exists():
        raise FileExistsError('Refusing to overwrite immutable audit receipt')
    if input_path.resolve()==output_path.resolve():
        raise ValueError('Audit must write separately from source')
    claims=json.loads(claims_path.read_text(encoding='utf-8')) if claims_path else None
    if claims is not None and not claims.get('derivation_reference'):
        raise ValueError('Externally provided bounds need a provenance reference')
    input_sha=hashlib.sha256(input_path.read_bytes()).hexdigest()
    claim_sha=hashlib.sha256(claims_path.read_bytes()).hexdigest() if claims_path else None
    output_path.parent.mkdir(parents=True,exist_ok=True)
    count=0
    with input_path.open('r',encoding='utf-8') as source, output_path.open('x',encoding='utf-8') as dest:
        for line in source:
            if not line.strip():
                continue
            record=json.loads(line)
            result=audit_record(record,claims,quant_range=quant_range,
                                max_bits=max_bits,extra_margin=extra_margin)
            result['source_receipt_sha256']=input_sha
            result['claim_file_sha256']=claim_sha
            dest.write(json.dumps(result,sort_keys=True,allow_nan=False)+'\n')
            count+=1
    return {'source_receipt_sha256':input_sha,'claim_file_sha256':claim_sha,
            'records':count,'output':str(output_path)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--claims',type=Path)
    p.add_argument('--quant-range')
    p.add_argument('--max-bits',type=int,default=16)
    p.add_argument('--extra-margin',default='0')
    a=p.parse_args()
    print(json.dumps(run(a.input,a.out,claims_path=a.claims,quant_range=a.quant_range,
              max_bits=a.max_bits,extra_margin=a.extra_margin),indent=2))


if __name__=='__main__':
    main()

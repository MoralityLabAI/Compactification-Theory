"""SCM-13 campaign-wide provenance integrity gate for SCM-12 paired captures.

This is a *format and consistency* audit, not authenticity proof: SHA-256
strings are verified syntactically but provenance authenticity still requires
source-locked checkpoint and frozen-manifest verification by SCM-12's launcher.
The audit prevents a mixed-checkpoint, mixed-source, cross-task-chart or
changed-feature campaign from being summarized as one experiment.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from scm12_compare import MODELS, CHARTS, _load, compare_files

HASH=re.compile(r'[0-9a-f]{64}\Z')
REAL='hash_verified_RMP_checkpoint'
SYNTH='synthetic_fixture'


def _hash(value):
    return isinstance(value,str) and bool(HASH.fullmatch(value))


def strict_compare_files(paths: dict[tuple[str,str],Path], *, expected_tasks=64) -> dict:
    if set(paths) != {(m,c) for m in MODELS for c in CHARTS}:
        raise ValueError('Require complete four-arm registered pairing')
    inputs={(m,c):_load(paths[m,c],m,c) for m in MODELS for c in CHARTS}
    types={row['source_type'] for rows in inputs.values() for row in rows.values()}
    if len(types)!=1:
        raise ValueError('Mixed synthetic and trained source types across tasks or arms')
    source=next(iter(types))
    if source not in (REAL,SYNTH):
        raise ValueError('Unknown source classification')
    ids=set(next(iter(inputs.values())))
    for model in MODELS:
        for chart in CHARTS:
            if set(inputs[model,chart])!=ids:
                raise ValueError('Incomplete matched tasks')
            rows=list(inputs[model,chart].values())
            frozen=('checkpoint_sha256','dictionary_sha256','intervention_axes_sha256','features',
                    'chart_sha256')
            for field in frozen:
                # Represent lists/tuples consistently without hashing JSON keys.
                frozen_values={json.dumps(row.get(field),sort_keys=True) for row in rows}
                if len(frozen_values)!=1:
                    raise ValueError(f'Campaign-wide {field} drift within {model}/{chart}')
            for row in rows:
                cap=row.get('capture',{})
                if row.get('features') is None or len(row['features'])!=2 or row['features'][0]==row['features'][1]:
                    raise ValueError('Invalid or unfrozen feature pair')
                if source==REAL:
                    for field in ('task_id_sha256','group_id_sha256','token_sequence_sha256',
                                  'checkpoint_sha256','dictionary_sha256',
                                  'intervention_axes_sha256','chart_sha256'):
                        if not _hash(row.get(field)):
                            raise ValueError('Real RMP record has malformed '+field)
                    if cap.get('injected_position')!=4 or cap.get('injected_after_visit')!=0:
                        raise ValueError('Injection site differs from registration')
                    if cap.get('observed_token_positions')!=[16,32,48]:
                        raise ValueError('Observed token positions differ from registration')
                    if cap.get('radius_in_token_rms')!=0.05:
                        raise ValueError('Intervention radius differs from registration')
                    if cap.get('chart_sha256')!=row.get('chart_sha256') or cap.get('intervention_axes_sha256')!=row.get('intervention_axes_sha256'):
                        raise ValueError('Capture chart or axes hashes disagree with receipt')
                    angles=cap.get('angles')
                    predictions=cap.get('predictions')
                    frames=cap.get('frames_by_visit')
                    if (not isinstance(angles,list) or len(angles)!=33 or
                        not isinstance(predictions,list) or len(predictions)!=33 or
                        not isinstance(frames,list) or len(frames)!=8):
                        raise ValueError('Incomplete 32-angle, eight-visit RMP capture')
                    if any(not isinstance(x,(int,float)) or not math.isfinite(x) for x in angles):
                        raise ValueError('Nonfinite capture angle')
                    for visit in frames:
                        if not isinstance(visit,list) or len(visit)!=33:
                            raise ValueError('Incomplete visit angular frames')
                        for frame in visit:
                            if (not isinstance(frame,list) or len(frame)!=3 or
                                any(not isinstance(p,list) or len(p)!=2 or
                                    any(not isinstance(v,(int,float)) or not math.isfinite(v) for v in p)
                                    for p in frame)):
                                raise ValueError('Malformed or nonfinite projected activation frame')
                    if not isinstance(row.get('topology'),list) or len(row['topology'])!=8:
                        raise ValueError('Eight visit topology decisions required')
                    if not isinstance(row.get('double_resolution'),list) or len(row['double_resolution'])!=8:
                        raise ValueError('Registered 32/64 angular comparison missing')
                    if row.get('source_type')!=REAL:
                        raise ValueError('Real source record retyped')
    for key in ids:
        # Both models process exactly the same token input for the given task.
        tensor_hashes={inputs[m,c][key].get('token_sequence_sha256') for m in MODELS for c in CHARTS}
        if len(tensor_hashes)!=1:
            raise ValueError('Cross-model token sequence drift within paired task')
    report=compare_files(paths,expected_tasks=expected_tasks)
    report['integrity_schema']='scm13/campaign-wide-consistency/v1'
    report['source_type_consistent']=True
    report['frozen_model_chart_provenance']=True
    report['cross_model_input_identity']=True
    report['authenticity_scope']='consistency_only; independent checkpoint and selected-dictionary provenance required'
    report['neural_causal_claim_certified']=False
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for model,key in (('LOOP-T s1','loop'),('FF-U s3','ffu')):
        for chart,short in (('sae','sae'),('random-orthogonal','random')):
            parser.add_argument('--'+key+'-'+short,type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.out.exists():
        parser.error('Refusing to overwrite a previous audit')
    files={(m,c):getattr(args,key+'_'+short)
           for m,key in (('LOOP-T s1','loop'),('FF-U s3','ffu'))
           for c,short in (('sae','sae'),('random-orthogonal','random'))}
    result=strict_compare_files(files)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x',encoding='utf-8') as writer:
        json.dump(result,writer,indent=2,sort_keys=True,allow_nan=False)
        writer.write('\n')
    print(json.dumps({'status':'analyzed','n_matched_tasks':result['n_matched_tasks']}))

if __name__=='__main__':
    main()

"""SCM-08: read-only source-locked clean-tube and null-braid auditor.

Reconstructs tasks ONLY from RMP's selection-half monitor_calibration. Checks
checkpoint, dictionary and chart provenance. Certifies clean pre/mid source
states for an ideal real model with a pinned dyadic position-frequency table.
Physical floating execution and capture-coordinate errors remain unverified.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
from rmp_lipschitz import rational
import hashlib
import json
from pathlib import Path
import sys

import torch

from clean_tubes import (clean_state_tubes,certified_injection_amplitude,
                         certified_zero_disk,verified_local_pair_bounds)
from exact_input import rmp_input_enclosure
from rmp_capture import (chart_from_decoder,sha256_file,sha256_tensor)
from scm06_audit import verify_rmp_source,_same_source

SCHEMA='scm08_ideal_real_source_clean_tube_audit_v1'


def examine_record(record, analysis, tokens:torch.Tensor, chart:torch.Tensor,
                   axes:torch.Tensor, *, margin='1/10000000') -> dict:
    """Model- and task-derived source proof; independent of captured float RMS."""
    capture=record.get('capture')
    if not isinstance(capture,dict):
        raise ValueError('missing RMP capture')
    model=analysis.model
    if model.training:
        raise ValueError('loaded model must be eval')
    if tokens.ndim!=1 or tokens.numel()!=int(capture.get('sequence_length',-1)):
        raise ValueError('task sequence contract mismatch')
    if sha256_tensor(tokens)!=record.get('token_sequence_sha256'):
        raise ValueError('token sequence hash mismatch')
    if sha256_tensor(axes)!=record.get('intervention_axes_sha256') or \
            sha256_tensor(chart)!=record.get('chart_sha256'):
        raise ValueError('injection/readout provenance mismatch')
    if capture.get('intervention_axes_sha256')!=record.get('intervention_axes_sha256'):
        raise ValueError('capture axes mismatch')
    observed=tuple(int(t) for t in capture['observed_token_positions'])
    inj=int(capture['injected_position']); after=int(capture['injected_after_visit'])
    # Capture records contain Python floats. Use their exact binary value;
    # decimal re-parsing could underestimate the true injected amplitude.
    radius=rational(float(capture['radius_in_token_rms']))
    if radius<=0:
        raise ValueError('injection radius must be positive')
    # Exactly the source-config reinjection used in AnalysisModel.run.
    reinject=bool(analysis.config.input_injection)
    initial=rmp_input_enclosure(model,tokens.cpu())
    clean=clean_state_tubes(model,initial,fixed_input_reinjection=reinject)
    certified_amplitude=certified_injection_amplitude(initial,clean,
            injected_token=inj,after_visit=after,radius_fraction=radius)
    # The derived source amplitude replaces an untrusted recorded float RMS.
    result=verified_local_pair_bounds(model,initial,
            chart=chart,axes=axes,amplitude=certified_amplitude,
            observed_tokens=observed,injected_token=inj,inject_after_visit=after,
            fixed_input_reinjection=reinject,clean=clean)
    null_disk=certified_zero_disk(initial,clean,chart,result,observed,margin=margin)
    return {
        'schema':SCHEMA,'status':'ideal_real_source_bound_derived',
        'model':record.get('model'),'checkpoint_sha256':record.get('checkpoint_sha256'),
        'dictionary_sha256':record.get('dictionary_sha256'),
        'task_id_sha256':record.get('task_id_sha256'),
        'input_frequency_table_sha256':initial.frequency_sha256,
        'input_scope':initial.scope,
        'input_max_radius':str(max(initial.radii)),
        'max_ideal_source_injection_amplitude':str(certified_amplitude),
        'source_injection_after_visit':after,
        'fixed_input_reinjection':reinject,
        'null_braid_ideal_real':null_disk,
        'bounds':result,
        'source_theorem_verified_by_this_program':True,
        'floating_capture_certified':False,
        'observed_link_certified':False,
        'requires_frequency_table_match_to_execution':True,
        'requires_output_coordinate_roundoff_enclosures':True,
    }


def run(input_path:Path, output_path:Path, *, rmp_root:Path, runs:Path,
        artifacts:Path, dictionary:Path, monitor_bundle:Path, margin='1/10000000') -> dict:
    """Immutable audit of selected calibration receipts. Never reads sealed tests."""
    if output_path.exists():
        raise FileExistsError('Refusing to overwrite an SCM-08 receipt')
    records=[json.loads(line) for line in input_path.read_text().splitlines() if line.strip()]
    if not records:
        raise ValueError('empty capture source')
    labels={record.get('model') for record in records}
    if len(labels)!=1:
        raise ValueError('exactly one source model per audit')
    if not rmp_root.is_dir():
        raise FileNotFoundError('RMP source checkout unavailable')
    verified=verify_rmp_source(rmp_root)
    sys.path.insert(0,str(rmp_root.resolve()))
    from research_gym.interp.phase2 import MODELS,core_examples,load_model,selection_half
    from research_gym.interp.dictionaries import load_dictionary
    name=next(iter(labels))
    if name not in MODELS:
        raise ValueError('unknown RMP model')
    cell,label,_=MODELS[name]
    checkpoint=artifacts/cell/'checkpoints'/f'{label}.pt'
    for file in (checkpoint,dictionary):
        if not file.is_file():
            raise FileNotFoundError(f'missing source artifact: {file}')
    analysis=load_model(name,runs,artifacts,torch.device('cpu'))
    dictobj,summary=load_dictionary(dictionary,device='cpu')
    if summary.get('config',{}).get('kind')!='shared':
        raise ValueError('only frozen RMP D-shared dictionary supported')
    data=core_examples(monitor_bundle,'monitor_calibration')
    eligible=selection_half(data['groups'])
    by_hash={}
    for i in range(len(data['ids'])):
        if not eligible[i]:
            continue
        token_id=hashlib.sha256(str(data['ids'][i]).encode()).hexdigest()
        if token_id in by_hash:
            raise ValueError('ambiguous calibration task ID')
        by_hash[token_id]=i
    output=[];seen=set()
    for record in records:
        key=record['task_id_sha256']
        if key in seen or key not in by_hash:
            raise ValueError('duplicate or non-selection-half task')
        seen.add(key)
        idx=by_hash[key]
        if hashlib.sha256(str(data['groups'][idx]).encode()).hexdigest()!=record.get('group_id_sha256'):
            raise ValueError('calibration group provenance mismatch')
        if record.get('family')!=str(data['families'][idx]):
            raise ValueError('calibration family provenance mismatch')
        features=tuple(record['features'])
        axes=chart_from_decoder(dictobj.decoder,features)
        if record['chart_control']=='sae':
            chart=axes
        elif record['chart_control']=='random-orthogonal':
            generator=torch.Generator(device='cpu').manual_seed(int(record['seed'])+91247)
            chart=torch.linalg.qr(torch.randn(axes.shape,generator=generator),mode='reduced')[0]
        else:
            raise ValueError('unsupported chart mode')
        _same_source(record,checkpoint=checkpoint,dictionary=dictionary,chart=chart,axes=axes)
        derived=examine_record(record,analysis,data['tokens'][idx],chart,axes,margin=margin)
        derived['rmp_pinned_source']=verified
        output.append(derived)
    output_path.parent.mkdir(parents=True,exist_ok=True)
    with output_path.open('x',encoding='utf-8') as writer:
        for item in output:
            writer.write(json.dumps(item,sort_keys=True,allow_nan=False)+'\n')
    return {'schema':SCHEMA,'source_sha256':sha256_file(input_path),
            'receipt_sha256':sha256_file(output_path),'records':len(output),
            'ideal_null_disk_visits':sum(1 for r in output for x in r['null_braid_ideal_real']
                                         if x['null_braid_for_ideal_real_model']),
            'observed_link_certificates':0,'out':str(output_path)}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--captures',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--rmp-root',type=Path,required=True)
    parser.add_argument('--runs',type=Path,required=True)
    parser.add_argument('--artifacts',type=Path,required=True)
    parser.add_argument('--dictionary',type=Path,required=True)
    parser.add_argument('--monitor-bundle',type=Path,required=True)
    parser.add_argument('--margin',default='1/10000000')
    args=parser.parse_args(argv)
    print(json.dumps(run(args.captures,args.out,rmp_root=args.rmp_root,
                          runs=args.runs,artifacts=args.artifacts,dictionary=args.dictionary,
                          monitor_bundle=args.monitor_bundle,margin=args.margin),indent=2))

if __name__=='__main__':
    main()

"""SCM-09 source-locked offline audit of validated centered RMP real trajectories.

Scope: exact-real arithmetic on frozen dyadic weights, configured frozen
sin/cos frequency table, and real exp/erf/sqrt. Physical PyTorch execution
roundoff and chart capture errors remain outside the proof. Read-only input;
new reports are created exclusively and cannot overwrite earlier receipts.

The default small verifier intentionally rejects large RMP cells until a
verified vectorized backend is available. Raising caps is an explicit opt-in.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

from exact_input import rmp_input_enclosure
from rmp_capture import chart_from_decoder,sha256_file,sha256_tensor
from rmp_lipschitz import rational
from scm06_audit import verify_rmp_source,_same_source
from validated_forward import (
    verified_centered_tubes,verified_centered_pair_bounds,
    verified_injection_amplitude,centered_null_disk,
)

SCHEMA='scm09_rmp_exact_real_centered_interval_v1'


def examine_record(record,analysis,tokens:torch.Tensor,chart:torch.Tensor,
                   axes:torch.Tensor,*,margin='1/10000000',
                   max_seq=16,max_dim=32,max_visits=8):
    capture=record.get('capture')
    if not isinstance(capture,dict):
        raise ValueError('missing RMP capture')
    model=analysis.model
    if model.training or tokens.ndim!=1:
        raise ValueError('expect an eval model and one token sequence')
    seq=capture.get('sequence_length')
    if seq is not None and int(seq)!=tokens.numel():
        raise ValueError('capture sequence length mismatch')
    if sha256_tensor(tokens)!=record.get('token_sequence_sha256'):
        raise ValueError('task token hash mismatch')
    if sha256_tensor(chart)!=record.get('chart_sha256') or \
            sha256_tensor(axes)!=record.get('intervention_axes_sha256') or \
            capture.get('intervention_axes_sha256')!=record.get('intervention_axes_sha256'):
        raise ValueError('chart or injection-axis provenance mismatch')
    obs=tuple(int(pos) for pos in capture['observed_token_positions'])
    injected=int(capture['injected_position'])
    visit=int(capture['injected_after_visit'])
    # The capture's Python float is interpreted as its EXACT stored binary rational.
    # This prevents undershooting the actual source parameter via decimal parsing.
    r=rational(float(capture['radius_in_token_rms']))
    if r<=0:
        raise ValueError('injection amplitude must be positive')
    reinject=bool(analysis.config.input_injection)
    source=rmp_input_enclosure(model,tokens.detach().cpu())
    tube=verified_centered_tubes(model,source,
             fixed_input_reinjection=reinject,max_seq=max_seq,
             max_dim=max_dim,max_visits=max_visits)
    amp=verified_injection_amplitude(tube,injected_token=injected,
                                    after_visit=visit,radius_fraction=r)
    bound=verified_centered_pair_bounds(model,tube,chart=chart,axes=axes,
             amplitude=amp,observed_tokens=obs,injected_token=injected,
             inject_after_visit=visit)
    null=centered_null_disk(tube,chart,obs,bound['per_visit_pair_bounds'],margin=margin)
    return {
        'schema':SCHEMA,
        'status':'source_verified_exact_real_model_only',
        'model':record.get('model'),
        'task_id_sha256':record.get('task_id_sha256'),
        'checkpoint_sha256':record.get('checkpoint_sha256'),
        'dictionary_sha256':record.get('dictionary_sha256'),
        'frequency_table_sha256':source.frequency_sha256,
        'rmp_input_scope':source.scope,
        'input_reinjection':reinject,
        'max_injection_amplitude':str(amp),
        'clean_post_radii':tuple(str(max(x)) for x in tube.post_center_errors),
        'clean_center_theorem_verified':True,
        'null_braid_ideal_real':null,
        'bounds':bound,
        'legacy_missing_sequence_length':seq is None,
        'floating_execution_certified':False,
        'observed_link_certified':False,
        'requires_execution_frequency_table_match':True,
        'requires_float_execution_error_proof':True,
    }


def run(input_path:Path, output_path:Path,*,rmp_root:Path,runs:Path,
        artifacts:Path,dictionary:Path,monitor_bundle:Path,margin='1/10000000',
        max_seq:int=16,max_dim:int=32,max_visits:int=8)->dict:
    if output_path.exists():
        raise FileExistsError('refusing to overwrite a previous SCM-09 audit')
    records=[json.loads(line) for line in input_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    if not records:
        raise ValueError('empty RMP input receipts')
    labels={x.get('model') for x in records}
    if len(labels)!=1:
        raise ValueError('exactly one source model required')
    if not rmp_root.is_dir():
        raise FileNotFoundError('RMP source checkout unavailable')
    verified=verify_rmp_source(rmp_root)
    sys.path.insert(0,str(rmp_root.resolve()))
    from research_gym.interp.phase2 import MODELS,core_examples,load_model,selection_half
    from research_gym.interp.dictionaries import load_dictionary
    name=next(iter(labels))
    if name not in MODELS:
        raise ValueError('unknown RMP source model')
    cell,label,_=MODELS[name]
    checkpoint=artifacts/cell/'checkpoints'/f'{label}.pt'
    for path in (checkpoint,dictionary):
        if not path.is_file():
            raise FileNotFoundError(f'missing immutable artifact: {path}')
    analysis=load_model(name,runs,artifacts,torch.device('cpu'))
    sparse,summary=load_dictionary(dictionary,device='cpu')
    if summary.get('config',{}).get('kind')!='shared':
        raise ValueError('requires source D-shared dictionary')
    data=core_examples(monitor_bundle,'monitor_calibration')
    eligible=selection_half(data['groups'])
    byid={}
    for i,example_id in enumerate(data['ids']):
        if not eligible[i]:
            continue
        h=hashlib.sha256(str(example_id).encode()).hexdigest()
        if h in byid:
            raise ValueError('duplicate calibration example ID')
        byid[h]=i
    output=[];seen=set()
    for record in records:
        key=record['task_id_sha256']
        if key in seen or key not in byid:
            raise ValueError('duplicate or held-out task receipt')
        seen.add(key)
        idx=byid[key]
        if hashlib.sha256(str(data['groups'][idx]).encode()).hexdigest()!=record.get('group_id_sha256'):
            raise ValueError('group hash mismatch')
        if record.get('family')!=str(data['families'][idx]):
            raise ValueError('source family mismatch')
        axes=chart_from_decoder(sparse.decoder,tuple(record['features']))
        if record['chart_control']=='sae':
            chart=axes
        elif record['chart_control']=='random-orthogonal':
            generator=torch.Generator(device='cpu').manual_seed(int(record['seed'])+91247)
            chart=torch.linalg.qr(torch.randn(axes.shape,generator=generator),mode='reduced')[0]
        else:
            raise ValueError('unsupported chart control')
        _same_source(record,checkpoint=checkpoint,dictionary=dictionary,chart=chart,axes=axes)
        receipt=examine_record(record,analysis,data['tokens'][idx],chart,axes,
                               margin=margin,max_seq=max_seq,
                               max_dim=max_dim,max_visits=max_visits)
        receipt['rmp_pinned_source']=verified
        output.append(receipt)
    output_path.parent.mkdir(parents=True,exist_ok=True)
    with output_path.open('x',encoding='utf-8') as writer:
        for r in output:
            writer.write(json.dumps(r,sort_keys=True,allow_nan=False)+'\n')
    return {'schema':SCHEMA,'source_sha256':sha256_file(input_path),
            'receipt_sha256':sha256_file(output_path),'records':len(output),
            'proven_null_visits':sum(int(x['null_braid_ideal_real']) for r in output
                                     for x in r['null_braid_ideal_real']),
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
    parser.add_argument('--max-seq',type=int,default=16)
    parser.add_argument('--max-dim',type=int,default=32)
    parser.add_argument('--max-visits',type=int,default=8)
    args=parser.parse_args(argv)
    print(json.dumps(run(args.captures,args.out,rmp_root=args.rmp_root,runs=args.runs,
             artifacts=args.artifacts,dictionary=args.dictionary,
             monitor_bundle=args.monitor_bundle,margin=args.margin,
             max_seq=args.max_seq,max_dim=args.max_dim,
             max_visits=args.max_visits),indent=2))

if __name__=='__main__':
    main()

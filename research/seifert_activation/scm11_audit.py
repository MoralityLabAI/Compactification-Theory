"""SCM-11 offline source-only RMP attention-response audit.

Reads immutable SCM-04 capture receipts but deliberately derives bounds from
source checkpoint, dictionary, selected calibration task, and SCM-10 clean-state
intervals. Captured floats are not treated as a mathematical proof. This is
an exact-real, CPU IEEE/BLAS-conditional research certificate, not a GPU or
four-dimensional Seifert-surface result. Source G0 and held-out splits are
never written or changed.
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
from scm06_audit import verify_rmp_source, _same_source
from validated_forward import verified_injection_amplitude
from ball_centered import verified_ball_tubes
from fast_centered import ieee_preflight
from refined_response import refined_disk_bounds
from gauge_null import gauge_axis_null_control

SCHEMA='scm11_anchored_attention_source_disk_v1'


def examine_record(record,analysis,tokens:torch.Tensor,chart:torch.Tensor,
                   axes:torch.Tensor,*,margin='1/10000000'):
    """Read-only verification for one already source-matched calibration case."""
    capture=record.get('capture')
    if not isinstance(capture,dict) or analysis.model.training or tokens.ndim!=1:
        raise ValueError('RMP evaluation source, one task and capture required')
    if int(capture.get('sequence_length',tokens.numel()))!=tokens.numel():
        raise ValueError('capture length differs from task')
    if sha256_tensor(tokens)!=record.get('token_sequence_sha256'):
        raise ValueError('task provenance mismatch')
    if sha256_tensor(chart)!=record.get('chart_sha256') or \
       sha256_tensor(axes)!=record.get('intervention_axes_sha256') or \
       capture.get('intervention_axes_sha256')!=record.get('intervention_axes_sha256'):
        raise ValueError('chart or intervention provenance mismatch')
    model=analysis.model
    if chart.ndim!=2 or axes.shape!=chart.shape or chart.shape[1]!=2:
        raise ValueError('chart/axes must be paired 2D readouts')
    if model.hidden_size&(model.hidden_size-1):
        # The inherited fast LayerNorm point implementation uses a stored
        # binary64 1/d; that is an exact rational only for powers of two.
        raise ValueError('current source-certified fast backend requires power-of-two hidden dimension')
    injected=int(capture['injected_position'])
    visit=int(capture['injected_after_visit'])
    obs=tuple(int(x) for x in capture['observed_token_positions'])
    rad=rational(float(capture['radius_in_token_rms']))
    if rad<=0:
        raise ValueError('strictly positive finite injection radius required')
    ieee_preflight()
    source=rmp_input_enclosure(model,tokens.detach().cpu())
    tubes=verified_ball_tubes(model,source,
                fixed_input_reinjection=bool(analysis.config.input_injection))
    amp=verified_injection_amplitude(tubes,injected_token=injected,
                                    after_visit=visit,radius_fraction=rad)
    proof=refined_disk_bounds(model,tubes,chart=chart,axes=axes,
          amplitude=amp,observed_tokens=obs,injected_token=injected,
          inject_after_visit=visit,margin=margin)
    try:
        gauge=gauge_axis_null_control(model,axes,injected_token=injected,
                                      observed_tokens=obs,sequence=tokens.numel())
    except ValueError:
        gauge={'status':'not_applicable_to_registered_axes'}
    return {'schema':SCHEMA,
      'model':record.get('model'),'task_id_sha256':record.get('task_id_sha256'),
      'checkpoint_sha256':record.get('checkpoint_sha256'),
      'dictionary_sha256':record.get('dictionary_sha256'),
      'frequency_table_sha256':source.frequency_sha256,
      'max_injection_amplitude':str(amp),
      'source_clean_max_post_error':[str(max(row)) for row in tubes.post_center_errors],
      'result':proof,'gauge_control':gauge,
      'clean_source_center_verified_under_CPU_IEEE_BLAS_contract':True,
      'source_backend_sha256':{name:sha256_file(Path(__file__).with_name(name)) for name in
         ('fast_centered.py','ball_centered.py','refined_response.py','gauge_null.py')},
      'physical_float_execution_certified':False,'observed_link_certified':False,
      'four_dimensional_surface_obstruction_certified':False}


def run(captures:Path,out:Path,*,rmp_root:Path,runs:Path,artifacts:Path,
        dictionary:Path,monitor_bundle:Path,margin='1/10000000'):
    if out.exists():
        raise FileExistsError('refusing to overwrite source-only audit receipt')
    entries=[json.loads(s) for s in captures.read_text().splitlines() if s.strip()]
    if not entries or len(entries)>64:
        raise ValueError('registered calibration audit expects 1 to 64 records')
    labels={x.get('model') for x in entries}
    if len(labels)!=1:
        raise ValueError('one model source per audit required')
    if not rmp_root.is_dir():
        raise FileNotFoundError('RMP source checkout missing')
    pinned=verify_rmp_source(rmp_root)
    sys.path.insert(0,str(rmp_root.resolve()))
    from research_gym.interp.phase2 import MODELS,core_examples,load_model,selection_half
    from research_gym.interp.dictionaries import load_dictionary
    name=next(iter(labels))
    if name not in MODELS:
        raise ValueError('unregistered RMP model')
    cell,label,_=MODELS[name]
    checkpoint=artifacts/cell/'checkpoints'/f'{label}.pt'
    for p in (checkpoint,dictionary):
        if not p.is_file():
            raise FileNotFoundError(f'missing external source artifact {p}')
    analysis=load_model(name,runs,artifacts,torch.device('cpu'))
    sparse,summary=load_dictionary(dictionary,device='cpu')
    if summary.get('config',{}).get('kind')!='shared':
        raise ValueError('registered D-shared dictionary required')
    data=core_examples(monitor_bundle,'monitor_calibration')
    selection=selection_half(data['groups'])
    byid={}
    for i,task_id in enumerate(data['ids']):
        if not selection[i]:
            continue
        key=hashlib.sha256(str(task_id).encode()).hexdigest()
        if key in byid:
            raise ValueError('duplicate task identity in calibration selection')
        byid[key]=i
    receipts=[];seen=set()
    for record in entries:
        tid=record['task_id_sha256']
        if tid in seen or tid not in byid:
            raise ValueError('duplicate or unregistered/held-out calibration task')
        seen.add(tid)
        index=byid[tid]
        if hashlib.sha256(str(data['groups'][index]).encode()).hexdigest()!=record.get('group_id_sha256'):
            raise ValueError('group hash mismatch')
        if record.get('family')!=str(data['families'][index]):
            raise ValueError('task family mismatch')
        axes=chart_from_decoder(sparse.decoder,tuple(record['features']))
        if record['chart_control']=='sae':
            chart=axes
        elif record['chart_control']=='random-orthogonal':
            gen=torch.Generator(device='cpu').manual_seed(int(record['seed'])+91247)
            chart=torch.linalg.qr(torch.randn(axes.shape,generator=gen),mode='reduced')[0]
        else:
            raise ValueError('unknown chart control')
        _same_source(record,checkpoint=checkpoint,dictionary=dictionary,
                     chart=chart,axes=axes)
        item=examine_record(record,analysis,data['tokens'][index],chart,axes,margin=margin)
        item['rmp_pinned_source']=pinned
        receipts.append(item)
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',encoding='utf-8') as f:
        for item in receipts:
            f.write(json.dumps(item,sort_keys=True,allow_nan=False)+'\n')
    return {'schema':SCHEMA,'captures_sha256':sha256_file(captures),
            'output_sha256':sha256_file(out),'audited':len(receipts),
            'trained_gpu_inference_certified':False,
            'out':str(out)}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('captures','out','rmp-root','runs','artifacts','dictionary','monitor-bundle'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--margin',default='1/10000000')
    args=p.parse_args(argv)
    print(json.dumps(run(args.captures,args.out,rmp_root=args.rmp_root,runs=args.runs,
            artifacts=args.artifacts,dictionary=args.dictionary,
            monitor_bundle=args.monitor_bundle,margin=args.margin),indent=2))

if __name__=='__main__':main()

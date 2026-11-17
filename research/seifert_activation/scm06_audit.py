"""SCM-06: checkpoint-derived real-model Lipschitz audit of SCM-04/05 captures.

Run on a machine with the RMP checkout and checkpoint artifacts. The auditor
reads existing captures and writes a separate immutable JSONL report. Each
machine-level continuity claim remains unproved until an external floating-
point evaluation/roundoff certificate is supplied. Structural derivative
bounds are proved for the real-valued network with the stored checkpoint
weights interpreted as exact rational coefficients.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Mapping

import torch

from continuous import continuous_sampling_certificate
from rmp_capture import chart_from_decoder, sha256_file, sha256_tensor
from rmp_lipschitz import (angular_gap_upper, rational, topology_zero_disk_certificate,
                           visit_bounds)

SCHEMA = 'scm06_checkpoint_real_analytic_audit_v1'
# Git blob hashes verified from the rmp-v0-phase2 source branch on 2026-11-13.
# The analytic proof assumes exactly these decoder/hook implementations.
PINNED_RMP_SOURCE = {
    'research_gym/neural/looped_decoder.py':'d94017ceb42cac12833dbd04a18c423014a0172a',
    'research_gym/neural/schedule_executor.py':'1123add33e55161074b08b6ef1841bd8fc8ceb59',
    'research_gym/interp/models.py':'195822c972f7e68b9ba918fdf4b48c13f821dc35',
    'research_gym/interp/analysis.py':'424edbae0eee3b50158e29ecfa81b2e12db6be3b',
}


def verify_rmp_source(root: Path) -> dict[str,str]:
    """Require the exact audited RMP architecture, permitting CRLF checkouts."""
    observed={}
    for rel, expected in PINNED_RMP_SOURCE.items():
        raw=(root/rel).read_bytes().replace(b'\r\n',b'\n')
        git_object=b'blob '+str(len(raw)).encode()+b'\x00'+raw
        sha=hashlib.sha1(git_object).hexdigest()
        if sha!=expected:
            raise ValueError(f'RMP source hash changed for {rel}; bound derivation needs review')
        observed[rel]=sha
    return observed


def _same_source(record: Mapping, *, checkpoint: Path, dictionary: Path, chart, axes) -> None:
    expected = {'checkpoint_sha256':sha256_file(checkpoint),
                'dictionary_sha256':sha256_file(dictionary),
                'chart_sha256':sha256_tensor(chart),
                'intervention_axes_sha256':sha256_tensor(axes)}
    for key, digest in expected.items():
        if key not in record or record[key] != digest:
            raise ValueError(f'Provenance mismatch in {key}; refusing to use unpinned artifacts')
    capture = record['capture']
    if capture['chart_sha256'] != expected['chart_sha256'] or \
       capture['intervention_axes_sha256'] != expected['intervention_axes_sha256']:
        raise ValueError('Capture chart/axes do not match outer record')


def examine_record(record: Mapping, model, chart: torch.Tensor, axes: torch.Tensor,
                   *, margin: str = '1/10000000') -> dict:
    """Inspect one record with all source matrices and axes independently matched.

    The `model` is an RMP AnalysisModel with loaded frozen model.model. This
    function is also usable in CPU mock-model tests. Source SHA enforcement
    occurs in `run` before calling this function.
    """
    trace = record.get('capture')
    if not isinstance(trace, Mapping):
        raise ValueError('RMP capture missing')
    required = ('injection_baseline_rms','sequence_length','clean_chart_by_visit',
                'frames_by_visit','angles','observed_token_positions','injected_position',
                'injected_after_visit','radius_in_token_rms')
    absent=[name for name in required if name not in trace]
    if absent:
        return {'schema':SCHEMA,'status':'missing_new_capture_fields',
                'missing':absent,'real_model_certified':False,
                'machine_level_certified':False}
    frames=trace['frames_by_visit']
    n = len(trace['angles']) - 1
    if n < 8 or len(frames) != len(model.model.module_indices):
        raise ValueError('Unexpected RMP sample/visit count')
    # Only uniform sampling of the exact ideal circle is covered by the
    # rational upper bound 44/(7*N) on each angular interval.
    import math
    if any(abs(float(ang)-2*math.pi*k/n)>1e-9 for k,ang in enumerate(trace['angles'])):
        raise ValueError('Angles deviate from prescribed uniform cyclic interventions')
    if any(len(row)!=n+1 for row in frames):
        raise ValueError('Visit-frame lengths do not match angular grid')
    amp = rational(trace['radius_in_token_rms']) * rational(trace['injection_baseline_rms'])
    proof=visit_bounds(model.model, sequence=int(trace['sequence_length']),
                       chart=chart, axes=axes, amplitude=amp,
                       observed_tokens=trace['observed_token_positions'],
                       injected_token=int(trace['injected_position']),
                       injected_after_visit=int(trace['injected_after_visit']))
    gap=angular_gap_upper(n)
    intervals=[str(gap)]*n
    disk=topology_zero_disk_certificate(trace['clean_chart_by_visit'],
                                        pair_bounds_result=proof,margin=margin)
    inspected=[]
    for j, (points,bounds) in enumerate(zip(frames,proof['per_visit_pair_bounds'])):
        if points[0]!=points[-1]:
            inspected.append({'visit':j+1,'status':'sampled_endpoints_not_exactly_equal',
                              'real_model_certified':False})
            continue
        result=continuous_sampling_certificate(points,interval_lengths=intervals,
                                   pair_lipschitz_bounds=bounds,extra_margin=margin)
        topology = record.get('topology')
        topo_entry = topology[j] if isinstance(topology,list) and len(topology)>j else {}
        observed_nontrivial = False
        if isinstance(topo_entry,dict) and topo_entry.get('valid'):
            winding=topo_entry.get('pair_winding',{})
            artin=topo_entry.get('artin',{})
            observed_nontrivial=bool(any(int(v)!=0 for v in winding.values()) or
                                     (isinstance(artin,dict) and artin.get('status')=='exact' and
                                      artin.get('nonidentity')))
        null_proved=disk[j]['conditional_null_if_clean_baseline_exact']
        discrepancy = ('conditional_ideal_topology_inconsistency' if result['passes'] else
                       'nontrivial_polygon_without_continuous_certificate') if (
                          null_proved and observed_nontrivial) else None
        inspected.append({'visit':j+1,'status':'conditional_ideal_sampling_inequality_pass'
                          if result['passes'] else 'insufficient_global_bound',
                          'sampling_inequality_passed':bool(result['passes']),
                          'real_model_certified':False,
                          'null_disk_inequality_passed':null_proved,
                          'observed_polygon_nontrivial':observed_nontrivial,
                          'consistency_alarm':discrepancy,
                          'sampling':result})
    return {'schema':SCHEMA,'status':'analysed',
            'arithmetic_scope':'smooth_real_model_with_exact_stored_float_weights',
            'independently_derived_weights_bound':True,
            'real_model_certified':False,
            'requires_exact_or_enclosed_real_activation_samples':True,
            'machine_level_certified':False,
            'machine_uncertainty':'finite_precision_forward_roundoff_and_trig_input_not_bounded',
            'source_model':record.get('model'),
            'checkpoint_sha256':record.get('checkpoint_sha256'),
            'dictionary_sha256':record.get('dictionary_sha256'),
            'chart_sha256':record.get('chart_sha256'),
            'task_id_sha256':record.get('task_id_sha256'),
            'uniform_interval_upper':str(gap),
            'exact_bound_details':proof,
            'disk_certificates':disk,
            'visits':inspected}


def run(input_path: Path, output_path: Path, *, rmp_root:Path, runs:Path,
        artifacts:Path, dictionary:Path, margin='1/10000000') -> dict:
    if output_path.exists():
        raise FileExistsError('SCM-06 refuses to overwrite a receipt')
    if not rmp_root.is_dir():
        raise FileNotFoundError('RMP source checkout required')
    sys.path.insert(0,str(rmp_root.resolve()))
    from research_gym.interp.phase2 import MODELS, load_model
    from research_gym.interp.dictionaries import load_dictionary
    inputs=[json.loads(line) for line in input_path.read_text().splitlines() if line.strip()]
    if not inputs:
        raise ValueError('Empty RMP capture file')
    labels={row.get('model') for row in inputs}
    if len(labels)!=1 or next(iter(labels)) not in MODELS:
        raise ValueError('One known RMP model per source file required')
    name=next(iter(labels))
    cell,label,_=MODELS[name]
    checkpoint=artifacts/cell/'checkpoints'/f'{label}.pt'
    for path in (checkpoint,dictionary):
        if not path.is_file():
            raise FileNotFoundError(f'Missing source artifact: {path}')
    source_checksums=verify_rmp_source(rmp_root)
    # CPU is deterministic for this read-only bound derivation; the capture
    # can have been produced on GPU, but exact stored bytes are matched.
    model=load_model(name,runs,artifacts,torch.device('cpu'))
    dictobj,_=load_dictionary(dictionary,device='cpu')
    output=[]
    for record in inputs:
        features=tuple(record['features'])
        axes=chart_from_decoder(dictobj.decoder,features)
        mode=record.get('chart_control','sae')
        if mode=='sae':
            chart=axes
        elif mode=='random-orthogonal':
            seed=int(record['seed'])
            g=torch.Generator(device='cpu').manual_seed(seed+91247)
            chart=torch.linalg.qr(torch.randn(axes.shape,generator=g),mode='reduced')[0]
        else:
            raise ValueError('Unknown chart control')
        _same_source(record,checkpoint=checkpoint,dictionary=dictionary,chart=chart,axes=axes)
        audit=examine_record(record,model,chart,axes,margin=margin)
        audit['rmp_source_sha256']=source_checksums
        output.append(audit)
    output_path.parent.mkdir(parents=True,exist_ok=True)
    with output_path.open('x',encoding='utf-8') as writer:
        for record in output:
            writer.write(json.dumps(record,sort_keys=True,allow_nan=False)+'\n')
    return {'schema':SCHEMA,'input_sha256':sha256_file(input_path),
            'output_sha256':sha256_file(output_path), 'source_sha256':source_checksums,
            'records':len(output),'out':str(output_path),
            'conditional_inequalities_passed':sum(v.get('sampling_inequality_passed',False)
                                  for row in output for v in row.get('visits',[]))}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--rmp-root',type=Path,required=True)
    p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--artifacts',type=Path,required=True)
    p.add_argument('--dictionary',type=Path,required=True)
    p.add_argument('--margin',default='1/10000000')
    args=p.parse_args(argv)
    print(json.dumps(run(args.input,args.out,rmp_root=args.rmp_root,runs=args.runs,
              artifacts=args.artifacts,dictionary=args.dictionary,margin=args.margin),indent=2))

if __name__=='__main__':
    main()

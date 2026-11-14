"""SCM-12 frozen paired RMP capture: preflight, lock, and deterministic launch.

This runner only invokes existing SCM-04 capture and SCM-12 comparison code.
RMP source, G0, splits, metrics, dictionaries and checkpoints remain read-only.
The source-only bound auditor is a separate step. The script never assumes
that missing out-of-repo trained weights can be reconstructed from results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from scm12_compare import compare_files, CHARTS, MODELS

SCHEMA='scm12/frozen-campaign/v1'
MODEL_CELLS={
    'LOOP-T s1':('LOOP-T-core4-ds-s1','C3'),
    'FF-U s3':('FF-U-core4-ds-inj-s3','C3'),
}
PINNED_SOURCE=(
    'research_gym/neural/looped_decoder.py',
    'research_gym/neural/schedule_executor.py',
    'research_gym/interp/analysis.py',
    'research_gym/interp/phase2.py',
    'research_gym/interp/dictionaries.py',
)


def digest(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as file:
        for buf in iter(lambda:file.read(2**20),b''):
            h.update(buf)
    return h.hexdigest()


def _select_dictionary(metrics:dict, model:str)->tuple[str,bool]:
    if model not in metrics:
        raise ValueError(f'Missing dictionary metrics for {model}')
    grid=[]
    for name,entry in metrics[model].items():
        cfg=entry.get('config',{})
        if cfg.get('kind')=='shared' and cfg.get('seed')==0 and cfg.get('site')==-1:
            grid.append((int(cfg['width']),int(cfg['k']),name,float(entry['splice']['min_ce_recovered'])))
    if not grid:
        raise ValueError(f'No selected D-shared dictionary candidates for {model}')
    passes=sorted(x for x in grid if x[3]>=.95)
    chosen=passes[0] if passes else max(grid,key=lambda x:x[3])
    return chosen[2],bool(passes)


def preflight(*,rmp_root:Path,runs:Path,artifacts:Path,bundle:Path,
              metrics:Path,dictionaries:dict[str,Path|None],
              features:dict[str,tuple[int,int]|None],
              feature_provenance:Path|None=None,examples:int=64,seed:int=20261113,
              angles:int=32)->dict:
    if examples!=64 or seed!=20261113 or angles!=32:
        raise ValueError('SCM-12 is fixed to 64 selection-half tasks, seed 20261113, and 32/64 angles')
    missing=[]
    hashes={}
    def require(path:Path|None,label:str):
        if path is None or not path.is_file():
            missing.append(label)
            return
        hashes[label]=digest(path)
    for relative in PINNED_SOURCE:
        require(rmp_root/relative,'rmp_source/'+relative)
    for label,path in (('dictionary_metrics',metrics),('feature_selection_provenance',feature_provenance)):
        require(path,label)
    if not bundle.is_dir():
        missing.append('monitor_calibration_bundle')
    if not runs.is_dir():
        missing.append('RMP_training_runs')
    selected={}
    contents={}
    if metrics.is_file():
        contents=json.loads(metrics.read_text(encoding='utf-8'))
        for model in MODELS:
            name,met=_select_dictionary(contents,model)
            selected[model]={'dictionary_config':name,'selection_bar_met':met}
    for model in MODELS:
        cell,stage=MODEL_CELLS[model]
        check=artifacts/cell/'checkpoints'/f'{stage}.pt'
        require(check,f'{model}/checkpoint')
        require(check.with_name(check.name+'.receipt.json'),f'{model}/checkpoint_receipt')
        require(runs/cell/'result.json',f'{model}/run_config')
        dictionary=dictionaries.get(model)
        require(dictionary,f'{model}/selected_dictionary')
        if dictionary is not None and dictionary.is_file() and model in selected:
            try:
                # The existing RMP archive is a simple PyTorch dict. weights_only
                # avoids arbitrary pickle class import while checking metadata.
                import torch
                saved=torch.load(dictionary,map_location='cpu',weights_only=True)
                wanted=contents[model][selected[model]['dictionary_config']]['config']
                got=saved['summary']['config']; shape=saved['shape']
                fields=('kind','seed','site','width','k')
                if any(got.get(field)!=wanted.get(field) for field in fields):
                    raise ValueError('Selected dictionary metadata differs from RMP metrics')
                if shape.get('width')!=wanted['width'] or shape.get('k')!=wanted['k'] or shape.get('skip') or shape.get('dim')!=128:
                    raise ValueError('Dictionary tensor dimensions mismatch RMP configuration')
                weights=saved['state']['decoder']
                if tuple(weights.shape)!=(shape['width'],128) or not bool(torch.isfinite(weights).all()):
                    raise ValueError('Dictionary decoder contains invalid weights')
                selected[model]['dictionary_metadata_verified']=True
            except Exception as exc:
                # Preserve the reason as a blocked gate without executing any
                # inference; corrupted/mismatched artifacts cannot be frozen.
                selected[model]['dictionary_metadata_error']=str(exc)
                missing.append(f'{model}/dictionary_metadata_valid')
        feat=features.get(model)
        if feat is None or len(feat)!=2 or feat[0]==feat[1] or min(feat)<0:
            missing.append(f'{model}/frozen_feature_pair')
    return {
        'schema':SCHEMA,
        'status':'ready_for_external_RMP_capture' if not missing else 'blocked_external_artifacts',
        'missing':sorted(set(missing)),
        'required_models':list(MODELS),
        'paths':{'rmp_root':str(rmp_root.resolve()),'runs':str(runs.resolve()),
                 'artifacts':str(artifacts.resolve()),'monitor_bundle':str(bundle.resolve()),
                 'metrics':str(metrics.resolve()),
                 'dictionaries':{model:str(path.resolve()) if path else None
                                  for model,path in dictionaries.items()},
                 'feature_provenance':str(feature_provenance.resolve()) if feature_provenance else None},
        'features':{model:list(features[model]) if features.get(model) is not None else None
                    for model in MODELS},
        'selected_configs':selected,
        'hashes':hashes,
        'protocol':{'examples':examples,'seed':seed,'angles':angles,
                    'angle_control':64,'observed_tokens':[16,32,48],
                    'injected_token':4,'radius':0.05,
                    'chart_controls':list(CHARTS),
                    'calibration_half':'selection_only',
                    'external_verifier':'scm11_audit.py'},
        'scientific_claims':{'trained_model_captured':False,
                             'GPU_float_link_certified':False,
                             'heldout_evaluation':False},
    }


def freeze(plan:dict,lock_path:Path)->None:
    if plan['status']!='ready_for_external_RMP_capture':
        raise ValueError('Refusing to freeze an incomplete source manifest: '+', '.join(plan['missing']))
    lock_path.parent.mkdir(parents=True,exist_ok=True)
    with lock_path.open('x',encoding='utf-8') as file:
        json.dump(plan,file,indent=2,sort_keys=True,allow_nan=False)
        file.write('\n')


def launch(lock_path:Path,out_dir:Path,*,device:str='cpu',python:str=sys.executable):
    if device not in ('cpu','cuda'):
        raise ValueError('Only registered CPU/CUDA capture device names are accepted')
    lock=json.loads(lock_path.read_text())
    paths=lock['paths'];features=lock['features']
    fresh=preflight(rmp_root=Path(paths['rmp_root']),runs=Path(paths['runs']),
        artifacts=Path(paths['artifacts']),bundle=Path(paths['monitor_bundle']),
        metrics=Path(paths['metrics']),
        dictionaries={m:Path(paths['dictionaries'][m]) for m in MODELS},
        features={m:tuple(features[m]) for m in MODELS},
        feature_provenance=Path(paths['feature_provenance']))
    if lock!=fresh:
        raise ValueError('Source/dictionary/selection identity has changed since freezing; refusing launch')
    if out_dir.exists():
        raise FileExistsError('Refusing to overwrite an existing campaign directory')
    out_dir.mkdir(parents=True)
    paths_out={}
    for model in MODELS:
        key='loop' if model=='LOOP-T s1' else 'ffu'
        for chart in CHARTS:
            label='sae' if chart=='sae' else 'random'
            output=out_dir/f'{key}_{label}.jsonl'
            cmd=[python,str(Path(__file__).with_name('rmp_capture.py')),
                '--rmp-root',paths['rmp_root'],'--runs',paths['runs'],
                '--artifacts',paths['artifacts'],'--monitor-bundle',paths['monitor_bundle'],
                '--dictionary',paths['dictionaries'][model],'--model',model,
                '--features',*[str(n) for n in features[model]],
                '--chart-control',chart,'--inject-token','4',
                '--observed-tokens','16','32','48','--radius','0.05',
                '--angles','32','--double-sampling','--examples','64',
                '--seed','20261113','--device',device,'--out',str(output)]
            subprocess.run(cmd,check=True)
            paths_out[model,chart]=output
    # The comparison is strict: any model mismatch, altered chart intervention,
    # or missing group causes a failure rather than a cherry-picked partial result.
    from scm13_receipt_integrity import strict_compare_files
    report=strict_compare_files(paths_out)
    with (out_dir/'paired_summary.json').open('x',encoding='utf-8') as file:
        json.dump(report,file,indent=2,sort_keys=True)
        file.write('\n')
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['preflight','freeze','run'],required=True)
    p.add_argument('--rmp-root',type=Path)
    p.add_argument('--runs',type=Path)
    p.add_argument('--artifacts',type=Path)
    p.add_argument('--monitor-bundle',type=Path)
    p.add_argument('--metrics',type=Path)
    p.add_argument('--feature-provenance',type=Path)
    p.add_argument('--loop-dictionary',type=Path)
    p.add_argument('--ffu-dictionary',type=Path)
    p.add_argument('--loop-features',type=int,nargs=2)
    p.add_argument('--ffu-features',type=int,nargs=2)
    p.add_argument('--lock',type=Path)
    p.add_argument('--out',type=Path)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args=p.parse_args(argv)
    if args.mode=='run':
        if args.lock is None or args.out is None:
            p.error('--run requires --lock and --out')
        report=launch(args.lock,args.out,device=args.device)
        print(json.dumps({'status':'completed','matched':report['n_matched_tasks']}))
        return
    needed=['rmp_root','runs','artifacts','monitor_bundle','metrics']
    if any(getattr(args,name) is None for name in needed):
        p.error('preflight/freeze requires source and artifact directory arguments')
    plan=preflight(rmp_root=args.rmp_root,runs=args.runs,artifacts=args.artifacts,
        bundle=args.monitor_bundle,metrics=args.metrics,
        dictionaries={'LOOP-T s1':args.loop_dictionary,'FF-U s3':args.ffu_dictionary},
        features={'LOOP-T s1':tuple(args.loop_features) if args.loop_features else None,
                  'FF-U s3':tuple(args.ffu_features) if args.ffu_features else None},
        feature_provenance=args.feature_provenance)
    if args.mode=='freeze':
        if args.lock is None:p.error('--freeze requires --lock')
        freeze(plan,args.lock)
    print(json.dumps(plan,indent=2,sort_keys=True))

if __name__=='__main__':
    main()

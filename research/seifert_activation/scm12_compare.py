"""SCM-12: read-only paired RMP chart-control and LOOP-T/FF-U analysis.

Input is four SCM-04-style JSONL captures: one SAE-chart and one random-chart
capture for each model, using the *same intervention axes* within each model.
Pairing by task and group hashes is strict. No causal discovery claim follows
from topology/accuracy correlations; output is descriptive plus a sign test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

SCHEMA='scm12/paired-control-analysis/v1'
MODELS=('LOOP-T s1', 'FF-U s3')
CHARTS=('sae','random-orthogonal')
CORE=('pointer_chase','gated_pointer_chase','reachability','sudoku4')


def file_digest(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda:file.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def _load(path: Path, model: str, chart: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    items={}
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        item=json.loads(line)
        if item.get('model') != model or item.get('chart_control')!=chart:
            raise ValueError('Model or chart mismatch in receipt')
        if item.get('schema')!='scm04/rmp-cyclic/v3':
            raise ValueError('Unexpected capture schema')
        if item.get('source_type') not in ('hash_verified_RMP_checkpoint','synthetic_fixture'):
            raise ValueError('Unknown source type; refuse unaudited record')
        key=item.get('task_id_sha256')
        if not isinstance(key,str) or len(key)<8 or key in items:
            raise ValueError('Missing or duplicated task identity')
        if item.get('family') not in CORE:
            raise ValueError('Unregistered task family')
        capture=item.get('capture',{})
        if (not isinstance(capture.get('predictions'),list) or
             len(capture['predictions'])<2 or
             capture['predictions'][0]!=capture['predictions'][-1]):
            raise ValueError('Endpoint prediction mismatch')
        items[key]=item
    if not items:
        raise ValueError('Empty capture file')
    return items


def _topology(row: dict) -> tuple[bool|None, int]:
    """None if any visit unresolved. True if any exact tested braid detects nontriviality.

    An exact *identity* Artin action proves the based braid identity for that
    polygonal model. An unknown or budget-exceeded Artin action does not prove
    triviality, even when all winding values vanish.
    """
    visits=row.get('topology')
    if not isinstance(visits,list) or not visits:
        return None,0
    flags=[]
    for visit in visits:
        if not visit.get('valid'):
            flags.append(None)
            continue
        artin=visit.get('artin',{})
        if artin.get('status')!='exact':
            flags.append(None)
            continue
        flags.append(bool(artin.get('nonidentity')))
    return (None if any(x is None for x in flags) else any(flags)),sum(x is not None for x in flags)


def _exact_two_sided_sign_test(left_only:int,right_only:int)->float:
    n=left_only+right_only
    if n==0:
        return 1.0
    k=min(left_only,right_only)
    return min(1.0, sum(math.comb(n,j) for j in range(k+1))*2/(2**n))


def compare_files(files:dict[tuple[str,str],Path], *, expected_tasks:int=64) -> dict:
    if set(files)!={(a,b) for a in MODELS for b in CHARTS}:
        raise ValueError('Require exactly two models times two chart controls')
    loaded={(a,b):_load(path,a,b) for (a,b),path in files.items()}
    ids=set(next(iter(loaded.values())))
    if any(set(data)!=ids for data in loaded.values()):
        raise ValueError('Group-disjoint paired task IDs differ across arms')
    if len(ids)!=expected_tasks or expected_tasks<1 or expected_tasks>64:
        raise ValueError(f'Expected exactly {expected_tasks} preregistered paired tasks')
    summaries={}
    first_row=next(iter(loaded[('LOOP-T s1','sae')].values()))
    source_type=first_row['source_type']
    if len({r['group_id_sha256'] for r in loaded[('LOOP-T s1','sae')].values()}) != len(ids):
        raise ValueError('Different task IDs share a group; grouped calibration leak')
    if expected_tasks==64:
        family_counts={family:sum(r['family']==family for r in loaded[('LOOP-T s1','sae')].values()) for family in CORE}
        if any(count!=16 for count in family_counts.values()):
            raise ValueError('Registered balanced calibration requires 16 tasks per family')
    for key in sorted(ids):
        reference=loaded[('LOOP-T s1','sae')][key]
        for model,chart in loaded:
            row=loaded[(model,chart)][key]
            if (row['group_id_sha256'],row['family'],row['target'],row['source_type']) != (
                reference['group_id_sha256'],reference['family'],reference['target'],source_type):
                raise ValueError('Paired targets, group or source provenance differ')
        for model in MODELS:
            a=loaded[(model,'sae')][key]
            b=loaded[(model,'random-orthogonal')][key]
            for field in ('checkpoint_sha256','dictionary_sha256','intervention_axes_sha256',
                          'features','target','token_sequence_sha256'):
                if a.get(field)!=b.get(field):
                    raise ValueError(f'Selected/controls source or intervention differs: {field}')
            if a['chart_sha256']==b['chart_sha256']:
                raise ValueError('Control chart must differ from SAE chart')
            ca,cb=a['capture'],b['capture']
            for field in ('predictions','baseline_prediction','max_abs_logit_change_from_clean',
                          'changed_prediction_fraction','intervention_axes_sha256'):
                if ca.get(field)!=cb.get(field):
                    raise ValueError(f'Changing readout chart changed intervention behavior: {field}')
    for model in MODELS:
        rows=[]
        for key in sorted(ids):
            a,b=(loaded[(model,chart)][key] for chart in CHARTS)
            x,nx=_topology(a)
            y,ny=_topology(b)
            flip=a['capture']['changed_prediction_fraction']
            if not isinstance(flip,(float,int)) or not (0<=flip<=1):
                raise ValueError('Invalid flip fraction')
            rows.append((x,y,nx,ny,float(flip),a['capture']['baseline_prediction']==a['target']))
        considered=[(x,y) for x,y,*_ in rows if x is not None and y is not None]
        lonly=sum(x and not y for x,y in considered)
        ronly=sum(y and not x for x,y in considered)
        summaries[model]={
          'n_tasks':len(ids),'n_topology_comparable':len(considered),
          'n_sae_nontrivial':sum(x is True for x,*_ in rows),
          'n_random_chart_nontrivial':sum(y is True for _,y,*_ in rows),
          'n_sae_only':lonly,'n_random_only':ronly,
          'exact_sign_test_p_descriptive':_exact_two_sided_sign_test(lonly,ronly),
          'mean_prediction_flip_fraction':sum(r[4] for r in rows)/len(rows),
          'n_clean_correct':sum(r[5] for r in rows),
          'n_unresolved_sae':sum(r[0] is None for r in rows),
          'n_unresolved_random':sum(r[1] is None for r in rows),
        }
    return {
        'schema':SCHEMA,
        'source_classification': 'synthetic_only' if source_type=='synthetic_fixture' else 'real_RMP_capture_uncertified_floating_topology',
        'models':summaries,
        'input_sha256':{f'{a}/{b}':file_digest(path) for (a,b),path in sorted(files.items())},
        'n_matched_tasks':len(ids),
        'limits':{'angular_agreement_is_not_continuous_path_proof':True,
                  'causal_feature_claims_require_heldout_interventions':True,
                  'source_level_theorems_do_not_certify_GPU_float_capture':True},
    }


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for model,key in (('LOOP-T s1','loop'),('FF-U s3','ffu')):
        for chart,suffix in (('sae','sae'),('random-orthogonal','random')):
            p.add_argument(f'--{key}-{suffix}',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args(argv)
    if args.out.exists():
        p.error('Refusing to overwrite existing comparison')
    paths={(model,chart):getattr(args,f'{key}_{suffix}')
           for model,key in (('LOOP-T s1','loop'),('FF-U s3','ffu'))
           for chart,suffix in (('sae','sae'),('random-orthogonal','random'))}
    report=compare_files(paths)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x',encoding='utf-8') as file:
        json.dump(report,file,sort_keys=True,indent=2,allow_nan=False)
        file.write('\n')
    print(json.dumps({'out':str(args.out),'sha256':file_digest(args.out),'n_matched_tasks':report['n_matched_tasks']},sort_keys=True))

if __name__=='__main__':
    main()

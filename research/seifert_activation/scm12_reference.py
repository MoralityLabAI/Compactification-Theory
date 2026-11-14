"""Deterministic, explicitly synthetic SCM-12 64-task matched-controls receipt.

Pure test fixture. It does not represent observed activations or measured RMP
behavior. Its only purpose is reproducible checking of source/provenance gates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from scm12_compare import MODELS, CHARTS, compare_files
from scm12_rank_one import rank_one_null_certificate

FAMILIES=('pointer_chase','gated_pointer_chase','reachability','sudoku4')


def source_record(task,model,chart):
    # Pairs are deliberately nontrivial on a small subset; all receipts remain
    # tagged synthetic. In the same task, readout changes leave predictions
    # completely fixed because the intervention axes are held unchanged.
    nontrivial=(task%8==0) if chart=='sae' else (task%8==1)
    flip=0.0 if task%3 else 1/4
    return {
        'schema':'scm04/rmp-cyclic/v3','source_type':'synthetic_fixture',
        'model':model,'chart_control':chart,
        'task_id_sha256':hashlib.sha256(f'scm12-task-{task}'.encode()).hexdigest(),
        'group_id_sha256':hashlib.sha256(f'scm12-group-{task}'.encode()).hexdigest(),
        'family':FAMILIES[task%4],'target':1,
        'checkpoint_sha256':f'synthetic-{model}-ckpt',
        'dictionary_sha256':f'synthetic-{model}-dict',
        'intervention_axes_sha256':f'synthetic-{model}-axes',
        'token_sequence_sha256':f'synthetic-task-{task}-tokens',
        'features':[10,11],'chart_sha256':f'synthetic-{model}-{chart}-chart',
        'capture':{'predictions':[0,2 if flip else 0,0,0,0],
                   'baseline_prediction':0,'max_abs_logit_change_from_clean':float(flip),
                   'changed_prediction_fraction':float(flip),
                   'intervention_axes_sha256':f'synthetic-{model}-axes'},
        'topology':[{'visit':1,'valid':True,
                     'artin':{'status':'exact','nonidentity':bool(nontrivial)}}],
    }


def generate():
    with TemporaryDirectory() as tmp:
        files={}
        for model in MODELS:
            for chart in CHARTS:
                path=Path(tmp)/('loop' if model==MODELS[0] else 'ffu')/('sae' if chart=='sae' else 'random')
                path.parent.mkdir(parents=True,exist_ok=True)
                with path.open('w',encoding='utf-8') as out:
                    for task in range(64):
                        out.write(json.dumps(source_record(task,model,chart),sort_keys=True,separators=(',',':'))+'\n')
                files[model,chart]=path
        data=compare_files(files)
    data['rank_one_exact_reference']=rank_one_null_certificate(
        [('0','0'),('3','0'),('0','3')],
        [('0','0'),('1/10','0'),('0','1/10')],
        ('-1','1'),['1/10']*3,margin='1/100')
    data['reference_evidence']='synthetic_only; none of these 64 tasks were run on trained RMP'
    return data


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--check',action='store_true')
    args=p.parse_args(argv)
    generated=json.dumps(generate(),sort_keys=True,indent=2,allow_nan=False)+'\n'
    if args.check:
        if generated!=args.out.read_text(encoding='utf-8'):
            raise SystemExit('SCM-12 synthetic receipt changed')
        print('SCM-12 synthetic receipt source-bound replay OK')
    else:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        with args.out.open('x',encoding='utf-8') as out:
            out.write(generated)
        print(f'SCM-12 synthetic receipt written to {args.out}')

if __name__=='__main__':main()

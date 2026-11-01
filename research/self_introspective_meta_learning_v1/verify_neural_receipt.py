"""Offline receipt-structure and source checker; --retrain verifies model replay.

Statistical acquisition isolation is an external requirement. The non-retrain
check reconstructs budgets and mean scores without opening test generator code
through the student. --retrain additionally regenerates trained checkpoints.
"""
from __future__ import annotations
import argparse
import hashlib
import gzip
import json
import math
from pathlib import Path


def verify(path:Path, *, retrain:bool=False)->dict:
    payload=gzip.decompress(path.read_bytes()) if path.suffix==".gz" else path.read_bytes()
    obj=json.loads(payload)
    if obj.get('schema')!='siml01.neural-curriculum.v0.2':
        raise ValueError('unexpected receipt schema')
    root=Path(__file__).resolve().parent
    expected=obj['source_sha256']
    for name,sig in expected.items():
        if name not in ('neural_student.py','neural_curriculum.py','microlearner.py','gate_math.py','local_proposer.py'):
            raise ValueError('unrecognized source')
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=sig:
            raise ValueError(f'source digest mismatch: {name}')
    cfg=obj['configuration']
    seeds=cfg['seeds']; arms=cfg['arms']
    outcomes=obj['outcomes']
    if len(outcomes)!=len(seeds)*len(arms):
        raise ValueError('missing outcomes')
    groups={}
    for result in outcomes:
        seed=result['seed']; arm=result['arm']; key=(seed,arm)
        if seed not in seeds or arm not in arms or key in groups:
            raise ValueError('mismatched seed/arm')
        groups[key]=result
        cost=result['resources']
        if cost['training_updates']!=cfg['steps'] or cost['examples_drawn']!=cfg['steps']*cfg['batch']:
            raise ValueError('unmatched sample budget')
        if cost['recurrent_step_examples']!=cfg['steps']*cfg['batch']*2:
            raise ValueError('unmatched recurrent compute proxy')
        if sum(result['updates_by_node'].values())!=cfg['steps']:
            raise ValueError('task allocation inconsistency')
        weighted=sum(obj['graph']['weights'][n]*result['final_test'][n] for n in ('T0','T1','T4'))
        if not math.isclose(weighted,result['weighted_test_score'],abs_tol=1e-12):
            raise ValueError('weighted score mismatch')
        if not (0<=result['secondary_unseen_triples']<=1):
            raise ValueError('invalid unseen triple accuracy')
    for seed in seeds:
        runs=[groups[(seed,a)] for a in arms]
        if len({r['initial_checkpoint'] for r in runs})!=1:
            raise ValueError('init checkpoint mismatch')
        for field in ('parameters','diagnostic_rounds','training_updates','recurrent_step_examples'):
            if len({r['resources'][field] for r in runs})!=1:
                raise ValueError(f'budget mismatch: {field}')
    for arm in arms:
        avg=sum(groups[(seed,arm)]['weighted_test_score'] for seed in seeds)/len(seeds)
        if not math.isclose(avg,obj['comparison_means'][arm],abs_tol=1e-12):
            raise ValueError('summary score mismatch')
    if retrain:
        from neural_curriculum import Config,run_arm
        config=Config(**{**cfg,'seeds':tuple(seeds),'arms':tuple(arms)})
        for seed in seeds:
            for arm in arms:
                saved=groups[(seed,arm)]
                fresh=run_arm(config,seed,arm)
                if fresh['trained_checkpoint']!=saved['trained_checkpoint']:
                    raise ValueError(f'checkpoint replay mismatch: {seed}/{arm}')
                if fresh['final_test']!=saved['final_test']:
                    raise ValueError(f'evaluation replay mismatch: {seed}/{arm}')
    return {'valid':True,'seeds':len(seeds),'arms':len(arms),'runs':len(outcomes),
            'retrained':retrain,'meaning':'receipt arithmetic and optional deterministic neural replay'}


def main():
    cli=argparse.ArgumentParser()
    cli.add_argument('--receipt',required=True)
    cli.add_argument('--retrain',action='store_true')
    args=cli.parse_args()
    print(json.dumps(verify(Path(args.receipt),retrain=args.retrain),sort_keys=True))

if __name__=='__main__':
    main()

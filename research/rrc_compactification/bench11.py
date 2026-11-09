"""RRC-11 held-out exact graph induction and measured preview work."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
try:
    from .graph11 import data,search,guided,score,digest,GRAPHS,TRAIN_GRAPHS,preview_score
    from .train11 import train
except ImportError:
    from graph11 import data,search,guided,score,digest,GRAPHS,TRAIN_GRAPHS,preview_score
    from train11 import train
HERE=Path(__file__).resolve().parent

def hashes():
    return {name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in ('graph11.py','train11.py','bench11.py')}

def run():
    model_path=HERE/'results/rrc11_ranker.json'
    model=json.loads(model_path.read_text())
    checks=hashes()
    if model['graph11_sha256']!=checks['graph11.py'] or model['train11_sha256']!=checks['train11.py']:
        raise AssertionError('model source drift')
    groups=data()
    if model['train_digest']!=digest(groups['train']):raise AssertionError('train roster mismatch')
    if set(TRAIN_GRAPHS).intersection(set(GRAPHS)-set(TRAIN_GRAPHS)):
        raise AssertionError('split overlap')
    results={}
    for family,cases in groups.items():
        if family=='train':continue
        scores={k:[] for k in ('exact','preview','learned_preview','blind_first')}
        for c in cases:
            funcs={
                'exact':search(c),
                'preview':guided(c,None,budget=4),
                'learned_preview':guided(c,model['weights'],budget=4),
                'blind_first':search(c,TRAIN_GRAPHS,budget=1)}
            for k,r in funcs.items():scores[k].append(score(c,r))
        results[family]={}
        for k,rows in scores.items():
            results[family][k]={
                'cases':len(rows),
                'accepted':sum(z['accepted'] for z in rows),
                'correct_future':sum(z['future_correct'] for z in rows),
                'false_accept':sum(z['false_accept'] for z in rows),
                'mean_full_candidate_checks':round(sum(z['attempts'] for z in rows)/len(rows),6),
                'mean_primitive_calls':round(sum(z['primitive_calls'] for z in rows)/len(rows),6),
                'mean_nominal_work':round(sum(z['work_proxy'] for z in rows)/len(rows),6)}
    return {'schema':'rrc11.receipt.v1','status':'exact_finite_cpu_operator_graph_identification',
            'source_sha256':checks,'ranker_file_sha256':hashlib.sha256(model_path.read_bytes()).hexdigest(),
            'rosters_sha256':{k:digest(v) for k,v in groups.items()},
            'parameters':{'observed_degree':12,'forecast_degree':32,'preview_degree':4,
                           'graph_candidates':len(GRAPHS),'training_grammar_candidates':len(TRAIN_GRAPHS),
                           'train_cases':len(groups['train']),'development_cases':len(groups['dev'])},
            'model_weights':model['weights'],'metrics':results,
            'claims_boundary':'Public interpreter executes each graph candidate; learned ranker sees its outputs. No direct neural operator-graph discovery or proof beyond observed horizon.'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');p.add_argument('--out',type=Path,default=HERE/'results/rrc11_pilot.json');a=p.parse_args()
    value=run();payload=json.dumps(value,sort_keys=True,indent=2)+'\n'
    if a.check:
        if not a.out.is_file() or a.out.read_text()!=payload:raise SystemExit('RRC-11 receipt missing or stale')
        print('RRC-11 exact receipt verified')
    else:
        a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(payload)
        print(json.dumps(value['metrics'],indent=2))
if __name__=='__main__':main()

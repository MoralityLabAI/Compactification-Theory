"""RRC-12 exact post-proposal verification, model/search-cost ledger and receipt."""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path

try:
    from . import graph11 as g, graph12 as b, model12, train12
except ImportError:
    import graph11 as g
    import graph12 as b
    import model12
    import train12

HERE=Path(__file__).resolve().parent
KINDS=('recurrent','feedforward','shuffled_recurrent')


def source_hashes():
    files=('graph11.py','graph12.py','model12.py','train12.py','bench12.py')
    return {n:sha256((HERE/n).read_bytes()).hexdigest() for n in files}


def checkpoints():
    hashes=source_hashes();models={};file_hashes={}
    train_roster=b.dataset(include_test=False)
    for name in KINDS:
        path=HERE/'results'/f'rrc12_{name}.json'
        model=json.loads(path.read_text())
        if model['source_sha256']!={k:hashes[k] for k in ('graph12.py','model12.py','train12.py','graph11.py')}:
            raise AssertionError(f'{name}: source hash drift')
        if model['train_roster_sha256']!=b.roster_digest(train_roster['train']):
            raise AssertionError(f'{name}: training roster drift')
        if model['dev_roster_sha256']!=b.roster_digest(train_roster['dev']):
            raise AssertionError(f'{name}: development roster drift')
        expected=4044 if name!='feedforward' else 3990
        if model['parameter_count']!=expected or model['kind']!=name:raise AssertionError('model budget changed')
        models[name]=model
        file_hashes[name]=sha256(path.read_bytes()).hexdigest()
    return models,file_hashes


def metrics(rows):
    if not rows:return {}
    fields=('accepted','correct_future','false_accept')
    result={'cases':len(rows),**{name:sum(z[name] for z in rows) for name in fields},
            'mean_full_verifier_candidates':round(sum(z['full_checks'] for z in rows)/len(rows),6),
            'mean_verifier_coefficient_work':round(sum(z['verifier_work'] for z in rows)/len(rows),6),
            'mean_inference_macs':round(sum(z['model_macs'] for z in rows)/len(rows),6)}
    return result


def run():
    models,file_hashes=checkpoints()
    data=b.dataset()
    methods=('recurrent_1','recurrent_2','recurrent_4','feedforward',
             'shuffled_recurrent','untrained_preview_4','exact_all_64','blind_1')
    results={};decision_hashes={};first_fail=[]
    # A direct model issues ONE graph proposal from visible demonstrations. The
    # public graph interpreter is called only after proposal for verification.
    for family,cases in data.items():
        if family=='train':continue
        accum={k:[] for k in methods}
        decisions={k:[] for k in methods}
        for c in cases:
            proposal={
                'recurrent_1':(model12.predict(c,models['recurrent'],scans=1),model12.inference_mac(models['recurrent'],1)),
                'recurrent_2':(model12.predict(c,models['recurrent'],scans=2),model12.inference_mac(models['recurrent'],2)),
                'recurrent_4':(model12.predict(c,models['recurrent'],scans=4),model12.inference_mac(models['recurrent'],4)),
                'feedforward':(model12.predict(c,models['feedforward']),model12.inference_mac(models['feedforward'])),
                'shuffled_recurrent':(model12.predict(c,models['shuffled_recurrent']),model12.inference_mac(models['shuffled_recurrent'])),
            }
            for label,(graph,macs) in proposal.items():
                score=b.verify(c,graph)
                row={**score,'full_checks':1,'verifier_work':2*3*(b.DEGREE+1)**2,'model_macs':macs}
                accum[label].append(row)
                decisions[label].append((c.identifier,graph,score))
                if family=='unseen_composition' and label=='recurrent_1' and not score['accepted'] and len(first_fail)<12:
                    first_fail.append({'id':c.identifier,'true_graph':c.hidden_graph,'proposed':graph})
            for label,solved in (('untrained_preview_4',g.guided(c,budget=4)),
                                 ('exact_all_64',g.search(c)),('blind_1',g.search(c,g.TRAIN_GRAPHS,budget=1))):
                sc=b.verify(c,solved['graph'])
                assert sc['accepted']==int(solved['graph'] is not None)
                row={**sc,'full_checks':solved['attempts'],'verifier_work':solved['work_proxy'],'model_macs':0}
                accum[label].append(row)
                decisions[label].append((c.identifier,solved['graph'],sc))
        results[family]={key:metrics(rows) for key,rows in accum.items()}
        decision_hashes[family]={k:sha256(json.dumps(v,separators=(',',':')).encode()).hexdigest()
                                 for k,v in decisions.items()}
    train_sig={c.visible for c in data['train']}
    alias={f:sum(c.visible in train_sig for c in cases) for f,cases in data.items() if f!='train'}
    pairs=data['late_collision']
    for p,q in zip(pairs[::2],pairs[1::2]):
        assert p.visible==q.visible and p.query==q.query and p.future!=q.future
        assert model12.predict(p,models['recurrent'])==model12.predict(q,models['recurrent'])
    return {'schema':'rrc12.receipt.v1','status':'cpu_few_shot_direct_graph_proposal',
            'source_sha256':source_hashes(),'checkpoint_sha256':file_hashes,
            'roster_sha256':{k:b.roster_digest(v) for k,v in data.items()},
            'train_observed_prefix_aliases':alias,
            'parameters':{'seed':b.SEED,'train_count':len(data['train']),'dev_count':len(data['dev']),
                          'test_counts':b.TEST_COUNTS,'graph_candidates':len(g.GRAPHS),
                          'semantic_classes':len(b.CANONICAL),'trained_classes':len(b.TRAIN_GRAPHS),
                          'unseen_composition_classes':len(b.TEST_GRAPHS),
                          'degree':b.DEGREE,'horizon':b.HORIZON,
                          'gru_parameters':4044,'feedforward_parameters':3990,
                          'trained_depth':1,'optional_depths':[1,2,4]},
            'model_development_diagnostics':{k:{'loss':round(v['development_loss'],6),
                            'token_accuracy':round(v['development_token_accuracy'],6)} for k,v in models.items()},
            'scores':results,'decision_digest':decision_hashes,'first_recurrent_failures':first_fail,
            'cost_units':{'model_inference':'dense_fp_multiply_accumulates_excluding_nonlinearities',
                          'verifier':'nominal_integer_coefficient_loop_work',
                          'not_interchangeable':True},
            'claims_boundary':'Operators and fixed fork/join topology are hand-coded. Direct neural inference only reads two demonstrations; exact interpreter is post-proposal only. No infinite identity or proof of general algebra discovery.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true')
    p.add_argument('--out',type=Path,default=HERE/'results'/'rrc12_pilot.json');a=p.parse_args()
    record=run();s=json.dumps(record,sort_keys=True,indent=2)+'\n'
    if a.check:
        if not a.out.is_file() or a.out.read_text()!=s:raise SystemExit('RRC-12 source/checkpoint receipt changed')
        print('RRC-12 exact receipt and frozen model replay verified')
    else:
        a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(s)
        for family in ('familiar','unseen_composition','source_shift','out_of_grammar','late_collision'):
            print(family,{k:(v['accepted'],v['correct_future']) for k,v in record['scores'][family].items()})

if __name__=='__main__':main()

"""RRC-13 source-bound active graph construction benchmark and cost ledger."""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path

try:
    from . import graph11 as g, graph12 as b12, graph13 as b, policy13 as p, train13 as t, model12
except ImportError:
    import graph11 as g
    import graph12 as b12
    import graph13 as b
    import policy13 as p
    import train13 as t
    import model12

HERE=Path(__file__).resolve().parent
ARMS=('active','active_no_feedback','passive','frozen_rrc12','greedy_stagewise','preview_4','exact_search')


def sources():
    paths=('graph11.py','graph12.py','graph13.py','policy13.py','train13.py',
           'bench13.py','model12.py','train12.py')
    return {n:sha256((HERE/n).read_bytes()).hexdigest() for n in paths}


def checkpoints():
    expected=t.hashes()
    saved={};shas={}
    for kind in ('active','passive'):
        path=HERE/'results'/f'rrc13_{kind}.json'
        raw=path.read_bytes()
        obj=json.loads(raw)
        if obj['kind']!=kind or obj['parameter_count']!=6148 or obj['hidden_size']!=p.HIDDEN:
            raise AssertionError('wrong checkpoint dimensions')
        if obj['source_hashes']!=expected:raise AssertionError('training source drift')
        saved[kind]=obj;shas[kind]=sha256(raw).hexdigest()
    prior=HERE/'results'/'rrc12_recurrent.json'
    older=json.loads(prior.read_text())
    # RRC-12 is a frozen prior-model comparator, trained with a different seed/roster.
    if older['parameter_count']!=4044 or older['kind']!='recurrent':
        raise AssertionError('RRC-12 frozen comparator invalid')
    for filename,expected in older['source_sha256'].items():
        if sha256((HERE/filename).read_bytes()).hexdigest()!=expected:
            raise AssertionError('RRC-12 comparator source drift')
    saved['frozen_rrc12']=older;shas['frozen_rrc12']=sha256(prior.read_bytes()).hexdigest()
    return saved,shas


def candidate(case,arm,models):
    if arm in ('active','active_no_feedback','passive'):
        ck=models['passive' if arm=='passive' else 'active']
        pred=p.infer(case,ck,feedback_enabled=(arm=='active'))
        estimate=b.policy_work(partial_queries=pred['partial_queries'])
        return pred['graph'],estimate,pred['model_mac_proxy']
    if arm=='frozen_rrc12':
        pred=model12.predict(case,models['frozen_rrc12'])
        return pred,b.policy_work(),model12.inference_mac(models['frozen_rrc12'])
    if arm=='greedy_stagewise':
        pred=b.greedy_construct(case)
        return pred,b.policy_work(partial_queries=12),0
    if arm=='preview_4':
        found=g.guided(case,None,budget=4)
        cost={'partial_queries':64, 'full_verifications':found['attempts'],
              'primitive_calls':found['primitive_calls'],
              'integer_loop_proxy':found['work_proxy']}
        return found['graph'],cost,0
    if arm=='exact_search':
        found=g.search(case)
        cost={'partial_queries':0,'full_verifications':found['attempts'],
              'primitive_calls':found['primitive_calls'],
              'integer_loop_proxy':found['work_proxy']}
        return found['graph'],cost,0
    raise ValueError('unknown arm')


def rows(cases,models,arm):
    records=[]
    for case in cases:
        graph,work,fp=candidate(case,arm,models)
        # For search arms, the enumerator already checked candidate prefixes.
        # Independently score every result without charging the grading oracle.
        verified=b12.verify(case,graph)
        row={'id':case.identifier,'graph':list(graph) if graph else None,
             'accepted':verified['accepted'],'future_correct':verified['correct_future'],
             'false_accept':verified['false_accept'],
             'mac_proxy':fp,**work}
        records.append(row)
    return records


def aggregate(records):
    count=len(records)
    def total(k):return sum(r[k] for r in records)
    return {'cases':count,'accepted':total('accepted'),
            'correct_future':total('future_correct'),'false_accept':total('false_accept'),
            'abstentions':count-total('accepted'),
            'mean_partial_queries':round(total('partial_queries')/count,6),
            'mean_full_verifier_attempts':round(total('full_verifications')/count,6),
            'mean_primitive_calls':round(total('primitive_calls')/count,6),
            'mean_integer_loop_proxy':round(total('integer_loop_proxy')/count,6),
            'mean_fp_mac_proxy':round(total('mac_proxy')/count,6),
            'decision_sha256':sha256(json.dumps([(r['id'],r['graph'],r['accepted'],
                 r['future_correct']) for r in records],separators=(',',':')).encode()).hexdigest()}


def run():
    models,ck_sha=checkpoints()
    groups=b.dataset()
    if models['active']['train_digest']!=b.digest(groups['train']) or models['active']['dev_digest']!=b.digest(groups['dev']):
        raise AssertionError('active train/dev roster drift')
    if models['passive']['train_digest']!=b.digest(groups['train']) or models['passive']['dev_digest']!=b.digest(groups['dev']):
        raise AssertionError('passive train/dev roster drift')
    train_sig={x.visible for x in groups['train']}
    aliases={name:sum(c.visible in train_sig for c in cases) for name,cases in groups.items() if name!='train'}
    if any(aliases.values()):raise AssertionError('unexpected train/test input alias')
    if any(c.hidden_graph not in b.TEST_GRAPHS for name in ('unseen_composition','source_shift') for c in groups[name]):
        raise AssertionError('test family boundary failed')
    for first,second in zip(groups['late_collision'][::2],groups['late_collision'][1::2]):
        if first.visible!=second.visible or first.query!=second.query or first.future==second.future:
            raise AssertionError('collision construction invalid')
    metrics={};failures={};collision_audit={}
    for split,cases in groups.items():
        if split=='train':continue
        metrics[split]={}
        for arm in ARMS:
            rr=rows(cases,models,arm)
            metrics[split][arm]=aggregate(rr)
            if split in ('unseen_composition','source_shift') and arm in ('active','passive'):
                failures[f'{split}:{arm}']=[{'id':r['id'],'proposed':r['graph']}
                     for r in rr if not r['accepted']][:8]
            if split=='late_collision':
                for x,y in zip(rr[::2],rr[1::2]):
                    if x['graph']!=y['graph']:
                        raise AssertionError('prefix-only policy differs on collision pair')
                collision_audit[arm]={
                    'paired_identical_proposals':True,
                    'pair_count':len(rr)//2,
                    'false_accept':sum(q['false_accept'] for q in rr),
                    'correct_future':sum(q['future_correct'] for q in rr)}
    return {
        'schema':'rrc13.receipt.v1',
        'status':'exploratory_cpu_teacher_supervised_active_construction',
        'source_sha256':sources(),'checkpoint_sha256':ck_sha,
        'rosters_sha256':{key:b.digest(cases) for key,cases in groups.items()},
        'train_prefix_aliases':aliases,
        'parameters':{'seed':b.SEED,'training_count':len(groups['train']),
                      'development_count':len(groups['dev']),'test_counts':b.TEST_COUNTS,
                      'graph_syntax_candidates':len(g.GRAPHS),
                      'training_semantic_classes':len(b.TRAIN_GRAPHS),
                      'heldout_semantic_classes':len(b.TEST_GRAPHS),
                      'degree':b.DEGREE,'future_degree':b.HORIZON,
                      'trained_updates':t.STEPS,
                      'active_parameters':models['active']['parameter_count'],
                      'passive_parameters':models['passive']['parameter_count'],
                      'old_frozen_model_parameters':models['frozen_rrc12']['parameter_count'],
                      'active_partial_queries_per_case':2,
                      'greedy_partial_candidate_queries_per_case':12,
                      'preview_candidate_count':64,
                      'feedback_from_own_partial_execution':True},
        'development_teacher_forced_diagnostics':{
            kind:{'loss':round(models[kind]['development_teacher_forced_loss'],8),
                  'token_accuracy':round(models[kind]['development_teacher_forced_token_accuracy'],8)}
            for kind in ('active','passive')},
        'metrics':metrics,'example_failures':failures,
        'collision_audit':collision_audit,
        'accounting':{'fp_mac':'dense matrix multiplications; excludes nonlinearities and Python overhead',
                      'integer_loop_proxy':'RRC11 3*(observed_degree+1)^2 per graph demonstration; excludes bigint bit cost',
                      'grading_excluded':True,'training_counterfactual_feedback_calls_per_case':20,
                      'trained_active_partial_queries':2,'no_compute_unit_conversion':True},
        'claims_boundary':'The program DSL, graph skeleton, residual features, and supervised teacher graph labels are human-specified. Active feedback uses exact interpreter on its own partial graph only. Finite prefix acceptance is not an infinite equivalence proof.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--out',type=Path,default=HERE/'results'/'rrc13_pilot.json')
    args=parser.parse_args()
    obj=run();s=json.dumps(obj,indent=2,sort_keys=True)+'\n'
    if args.check:
        if not args.out.is_file() or args.out.read_text()!=s:raise SystemExit('RRC13 receipt missing or stale')
        print('RRC-13 complete source-bound replay verified')
    else:
        args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(s)
        for k in ('familiar','unseen_composition','source_shift','out_of_grammar','late_collision'):
            print(k,{j:(v['accepted'],v['correct_future']) for j,v in obj['metrics'][k].items()},flush=True)

if __name__=='__main__':main()

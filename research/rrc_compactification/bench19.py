"""RRC-19: family-balanced learned class search, exact verification and costs."""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path

try:
    from . import family19 as f, learn18 as l, model18 as m, train19, train18, topology16 as t, graph11 as g
except ImportError:
    import family19 as f
    import learn18 as l
    import model18 as m
    import train19
    import train18
    import topology16 as t
    import graph11 as g

HERE=Path(__file__).resolve().parent
SOURCE_NAMES=('family19.py','train19.py','bench19.py','learn18.py','model18.py',
              'train18.py','certify18.py','topology16.py','topology15.py',
              'graph11.py','graph12.py','graph13.py','graph14.py','quotient17.py')
ARMS=('balanced_gru12','skewed_gru12','balanced_ff12','shuffled_gru12',
      'frozen_rrc18_gru12','frozen_rrc18_ff12','hybrid_gru6_mass6',
      'certified_mass12','fixed12','complete_classes')


def source_hashes():
    return {name:sha256((HERE/name).read_bytes()).hexdigest() for name in SOURCE_NAMES}


def checkpoints():
    current={};digests={};required=train19.source_hashes()
    for role in train19.ROLES:
        path=HERE/'results'/f'rrc19_{role}.json'
        raw=path.read_bytes();obj=json.loads(raw)
        kind='feedforward' if role=='balanced_feedforward' else 'recurrent'
        if obj['schema']!='rrc19.weights.v1' or obj['role']!=role or obj['kind']!=kind:
            raise AssertionError('incorrect checkpoint role/kind')
        if obj['source_sha256']!=required or obj['optimizer_steps']!=train19.STEPS:
            raise AssertionError('training source or optimization drift')
        if obj['parameter_count']!=m.parameter_count(kind):raise AssertionError('parameter budget drift')
        if obj['training_families']!=list(f.TRAIN_FAMILIES) or obj['unseen_families']!=list(f.OOD_FAMILIES):
            raise AssertionError('training family leak')
        current[role]=obj;digests[role]=sha256(raw).hexdigest()
    historical={}
    for kind in ('recurrent','feedforward'):
        path=HERE/'results'/f'rrc18_{kind}.json'
        raw=path.read_bytes();obj=json.loads(raw)
        if obj['source_sha256']!=train18.sources() or obj['parameter_count']!=m.parameter_count(kind):
            raise AssertionError('RRC-18 comparator source drift')
        historical[kind]=obj;digests['rrc18_'+kind]=sha256(raw).hexdigest()
    return current,historical,digests


def hybrid_order(neural,topology,classes,k=6):
    """Fixed no-test-tuning union: first six neural, then mass-order up to 12."""
    if k<1 or k>=f.BUDGET:raise ValueError('hybrid partition')
    first=l.candidates(neural,topology,classes,count=k)
    extras=tuple(p for p in l.mass_order(topology,classes) if p not in first)
    return (first+extras)[:f.BUDGET]


def propose(case,arm,logits,classes):
    if arm=='balanced_gru12':order=l.candidates(logits['balanced_recurrent'],case.topology,classes,count=f.BUDGET)
    elif arm=='skewed_gru12':order=l.candidates(logits['skewed_recurrent'],case.topology,classes,count=f.BUDGET)
    elif arm=='balanced_ff12':order=l.candidates(logits['balanced_feedforward'],case.topology,classes,count=f.BUDGET)
    elif arm=='shuffled_gru12':order=l.candidates(logits['shuffled_recurrent'],case.topology,classes,count=f.BUDGET)
    elif arm=='frozen_rrc18_gru12':order=l.candidates(logits['frozen_gru'],case.topology,classes,count=f.BUDGET)
    elif arm=='frozen_rrc18_ff12':order=l.candidates(logits['frozen_ff'],case.topology,classes,count=f.BUDGET)
    elif arm=='hybrid_gru6_mass6':order=hybrid_order(logits['balanced_recurrent'],case.topology,classes)
    elif arm in ('certified_mass12','complete_classes'):order=l.mass_order(case.topology,classes)
    elif arm=='fixed12':order=g.GRAPHS
    else:raise ValueError('unknown arm')
    budget=len(order) if arm=='complete_classes' else f.BUDGET
    proposed=l.search(case,order,budget=budget)
    role={'balanced_gru12':'recurrent','skewed_gru12':'recurrent',
          'balanced_ff12':'feedforward','shuffled_gru12':'recurrent',
          'frozen_rrc18_gru12':'recurrent','frozen_rrc18_ff12':'feedforward',
          'hybrid_gru6_mass6':'recurrent'}.get(arm)
    proposed['model_macs']=l.infer_cost(role) if role else 0
    return proposed


def aggregate(rows):
    n=len(rows)
    return {'cases':n,'accepted':sum(x['accepted'] for x in rows),
            'correct_future':sum(x['future_correct'] for x in rows),
            'false_accept':sum(x['false_accept'] for x in rows),
            'abstentions':sum(1-x['accepted'] for x in rows),
            'mean_queries':round(sum(x['queries'] for x in rows)/n,5),
            'mean_integer_loop_proxy':round(sum(x['integer_loop_proxy'] for x in rows)/n,5),
            'mean_dense_model_macs':round(sum(x['model_macs'] for x in rows)/n,5)}


def replications(current,old,classes,already_visible):
    """Post-primary-evaluation robustness check on frozen weights and new seeds.

    Source tags are scoring metadata only. This is not a blind preregistration;
    neither weights nor rank-order rules are adjusted from these scores.
    """
    output={};seen=set(already_visible)
    for offset in (5100,6100):
        summaries={}
        for j,family in enumerate(('product','rational','squared')):
            batch=[]
            for k,topology in enumerate(l.TOPOLOGIES):
                rng=__import__('random').Random(f.SEED+offset+j*1201+k*53)
                attempts=0;accepted=0
                while accepted<20:
                    attempts+=1
                    if attempts>20000:raise RuntimeError('replicate roster exhaustion')
                    case=f.case(rng,rng.choice(g.GRAPHS),topology,'replication',family,attempts)
                    if case.visible in seen:continue
                    seen.add(case.visible);batch.append(case);accepted+=1
            arms=('balanced_gru12','skewed_gru12','balanced_ff12','shuffled_gru12',
                  'frozen_rrc18_gru12','certified_mass12','complete_classes')
            rows={arm:[] for arm in arms}
            for case in batch:
                logits={role:m.logits(case,ck) for role,ck in current.items()}
                logits['frozen_gru']=m.logits(case,old['recurrent'])
                for arm in arms:
                    proposed=propose(case,arm,logits,classes)
                    grade=t.certificate(case,proposed['graph'])
                    rows[arm].append({**grade,'queries':len(proposed['trace']),
                                      'integer_loop_proxy':proposed['integer_loop_proxy'],
                                      'model_macs':proposed['model_macs']})
            if rows['complete_classes'] and aggregate(rows['complete_classes'])['accepted']!=len(batch):
                raise AssertionError('replication class-complete search failed')
            summaries[family]={'roster_sha256':f.digest(batch),
                               'scores':{arm:aggregate(rows[arm]) for arm in arms}}
        output[str(f.SEED+offset)]=summaries
    return output


def run():
    current,old,digests=checkpoints()
    groups=f.roster()
    train_visible={case.visible for case in groups['train']}
    aliases={s:sum(case.visible in train_visible for case in cases)
             for s,cases in groups.items() if s!='train'}
    if any(aliases.values()):raise AssertionError('train/test visible prefix collision')
    # The exact algebraic classes and their mass prior were frozen in RRC-18.
    classes=l.class_map()
    scores={};by_topology={};traces={};paired={};collision={}
    for split,cases in groups.items():
        if split in ('train','dev'):continue
        results={arm:[] for arm in ARMS}
        selected={arm:[] for arm in ARMS}
        per_topology={top:{arm:[] for arm in ARMS} for top in l.TOPOLOGIES}
        for case in cases:
            all_logits={role:m.logits(case,ck) for role,ck in current.items()}
            all_logits['frozen_gru']=m.logits(case,old['recurrent'])
            all_logits['frozen_ff']=m.logits(case,old['feedforward'])
            for arm in ARMS:
                out=propose(case,arm,all_logits,classes)
                trace=out['trace']
                if len(set(trace))!=len(trace):raise AssertionError('repeated paid graph query')
                if arm!='complete_classes' and len(trace)>f.BUDGET:
                    raise AssertionError('budget overflow')
                if out['integer_loop_proxy']!=sum(2*t.graph_cost(p,case.topology) for p in trace):
                    raise AssertionError('unaccounted interpreter work')
                graded=t.certificate(case,out['graph'])
                if out['graph'] is not None and not graded['accepted']:
                    raise AssertionError('unchecked proposed graph accepted')
                row={**graded,'queries':len(trace),'integer_loop_proxy':out['integer_loop_proxy'],
                     'model_macs':out['model_macs']}
                results[arm].append(row)
                per_topology[case.topology][arm].append(row)
                selected[arm].append((case.identifier,[list(p) for p in trace],
                                      None if out['graph'] is None else list(out['graph'])))
        scores[split]={arm:aggregate(rows) for arm,rows in results.items()}
        by_topology[split]={top:{arm:aggregate(rows) for arm,rows in local.items()}
                            for top,local in per_topology.items() if local[ARMS[0]]}
        traces[split]={arm:sha256(json.dumps(records,separators=(',',':')).encode()).hexdigest()
                       for arm,records in selected.items()}
        if split in ('rational_test','squared_test','product_test','sparse_test','signed_test'):
            def only(a,b):return sum(x['future_correct'] and not y['future_correct']
                                      for x,y in zip(results[a],results[b]))
            paired[split]={
                'balanced_only_vs_skewed':only('balanced_gru12','skewed_gru12'),
                'skewed_only_vs_balanced':only('skewed_gru12','balanced_gru12'),
                'balanced_only_vs_ff':only('balanced_gru12','balanced_ff12'),
                'ff_only_vs_balanced':only('balanced_ff12','balanced_gru12'),
                'balanced_only_vs_mass':only('balanced_gru12','certified_mass12'),
                'mass_only_vs_balanced':only('certified_mass12','balanced_gru12'),
            }
        if split=='collision':
            collision={arm:{'same_proposals':all(selected[arm][i][1:]==selected[arm][i+1][1:]
                                 for i in range(0,len(cases),2)),
                            'false_future_acceptances':scores[split][arm]['false_accept']}
                       for arm in ARMS}
    for split in tuple(k+'_test' for k in f.ALL_FAMILIES)+('collision',):
        if scores[split]['complete_classes']['accepted']!=len(groups[split]):
            raise AssertionError('complete bounded grammar misses in-class example')
    if scores['negative']['complete_classes']['accepted']!=0:
        raise AssertionError('bad constructed negative')
    if not all(item['same_proposals'] for item in collision.values()):
        raise AssertionError('future leakage through collision pair')
    if collision['complete_classes']['false_future_acceptances']!=f.COLLISION_PAIRS:
        raise AssertionError('late mutation information gate failed')
    checks=replications(current,old,classes,{c.visible for group in groups.values() for c in group})
    training_macs_per_model={role:train19.STEPS*train19.BATCH*l.infer_cost(ck['kind'])
                             for role,ck in current.items()}
    return {
        'schema':'rrc19.receipt.v1','status':'cpu_balanced_family_class_ranker_one_seed',
        'source_sha256':source_hashes(),'checkpoint_sha256':digests,
        'roster_sha256':{s:f.digest(v) for s,v in groups.items()},
        'train_test_visible_aliases':aliases,
        'parameters':{'seed':f.SEED,'observed_degree':f.PREFIX,'forecast_degree':f.HORIZON,
                      'budget':f.BUDGET,'train_families':list(f.TRAIN_FAMILIES),
                      'ood_families':list(f.OOD_FAMILIES),
                      'train_cases':len(groups['train']),'dev_cases':len(groups['dev']),
                      'train_steps_per_model':train19.STEPS,'batch':train19.BATCH,
                      'recurrent_parameters':m.parameter_count('recurrent'),
                      'feedforward_parameters':m.parameter_count('feedforward'),
                      'offline_symbolic_polynomial_normalizations':4*len(g.GRAPHS),
                      'inference_candidate_previews':0},
        'training_forward_mac_proxy':training_macs_per_model,
        'scores':scores,'by_topology':by_topology,'posthoc_independent_seed_checks':checks,
        'trace_sha256':traces,'paired_results':paired,'collision_pairs':collision,
        'scope':'Observed finite q^12 verification only; teacher supplies certified class labels and public graph interpreter. Training families balanced by topology and source, but no new primitive or graph topology is learned.'
    }


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--check',action='store_true')
    ap.add_argument('--out',type=Path,default=HERE/'results'/'rrc19_pilot.json')
    args=ap.parse_args()
    result=run()
    payload=json.dumps(result,indent=2,sort_keys=True)+'\n'
    if args.check:
        if not args.out.is_file() or args.out.read_text()!=payload:
            raise SystemExit('RRC-19 source-bound receipt mismatch')
        print('RRC-19 CPU receipt verified')
    else:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(payload)
        for group in f.ALL_FAMILIES:
            name=group+'_test'
            print(name,{arm:result['scores'][name][arm]['correct_future']
                        for arm in ('balanced_gru12','skewed_gru12','balanced_ff12',
                                    'shuffled_gru12','certified_mass12','hybrid_gru6_mass6',
                                    'frozen_rrc18_gru12','complete_classes')})
if __name__=='__main__':main()

"""RRC-20 frozen-model family-calibrated ranking and exact CPU replay."""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path

try:
    from . import family20 as f,gate20 as gate,train20,bench19,learn18 as l,model18 as m
    from . import graph11 as g,topology16 as t
except ImportError:
    import family20 as f
    import gate20 as gate
    import train20
    import bench19
    import learn18 as l
    import model18 as m
    import graph11 as g
    import topology16 as t

HERE=Path(__file__).resolve().parent
SOURCE_NAMES=('family20.py','gate20.py','train20.py','bench20.py','family19.py','train19.py',
              'bench19.py','learn18.py','model18.py','certify18.py','topology16.py',
              'graph11.py','graph12.py','graph13.py','graph14.py','quotient17.py')
GATE_ROLES=('constant','adaptive','shuffled')
ARMS=('adaptive_mix12','constant_mix12','uniform_mix12','shuffled_mix12',
      'balanced_gru12','skewed_gru12','balanced_ff12','mass12','first12','complete_classes')


def source_hashes():
    return {name:sha256((HERE/name).read_bytes()).hexdigest() for name in SOURCE_NAMES}


def checkpoints():
    models,history,digests=bench19.checkpoints()
    gates={};expected=train20.source_hashes();frozen=digests.copy()
    for kind in GATE_ROLES:
        path=HERE/'results'/f'rrc20_{kind}.json'
        raw=path.read_bytes();obj=json.loads(raw)
        if obj['schema']!='rrc20.gate.v1' or obj['kind']!=kind or obj['source_sha256']!=expected:
            raise AssertionError('gate source/kind mismatch')
        if obj['checkpoint_sha256']!=frozen or obj['experts']!=list(gate.EXPERTS):
            raise AssertionError('expert drift')
        if obj['parameter_count']!=(4 if kind=='constant' else 60):
            raise AssertionError('wrong gate size')
        if obj['test_families_used'] or not obj['base_models_frozen']:
            raise AssertionError('calibration/test contamination')
        gates[kind]=obj;digests['rrc20_'+kind]=sha256(raw).hexdigest()
    if gates['adaptive']['calibration_digest']!=f.old.digest(f.calibration()):
        raise AssertionError('calibration roster changed')
    return models,history,gates,digests


def candidate_order(case,arm,logits,classes,gates):
    if arm=='adaptive_mix12':return gate.order(case,logits,classes,gates['adaptive'])[:f.BUDGET]
    if arm=='constant_mix12':return gate.order(case,logits,classes,gates['constant'])[:f.BUDGET]
    if arm=='shuffled_mix12':return gate.order(case,logits,classes,gates['shuffled'])[:f.BUDGET]
    if arm=='uniform_mix12':return gate.order(case,logits,classes,fixed=(1,1,1,1))[:f.BUDGET]
    if arm=='balanced_gru12':return l.candidates(logits['balanced_recurrent'],case.topology,classes,count=f.BUDGET)
    if arm=='skewed_gru12':return l.candidates(logits['skewed_recurrent'],case.topology,classes,count=f.BUDGET)
    if arm=='balanced_ff12':return l.candidates(logits['balanced_feedforward'],case.topology,classes,count=f.BUDGET)
    if arm in ('mass12','complete_classes'):return l.mass_order(case.topology,classes)
    if arm=='first12':return g.GRAPHS[:f.BUDGET]
    raise ValueError('unknown candidate method')


def macs(arm):
    if arm.endswith('mix12'):
        return 2*l.infer_cost('recurrent')+l.infer_cost('feedforward')+(gate.FEATURES*4 if arm in ('adaptive_mix12','shuffled_mix12') else 0)
    if arm in ('balanced_gru12','skewed_gru12'):return l.infer_cost('recurrent')
    if arm=='balanced_ff12':return l.infer_cost('feedforward')
    return 0


def aggregate(rows):
    n=len(rows)
    if not n:return {}
    return {'cases':n,'accepted':sum(r['accepted'] for r in rows),
            'correct_future':sum(r['future_correct'] for r in rows),
            'false_accept':sum(r['false_accept'] for r in rows),
            'abstentions':n-sum(r['accepted'] for r in rows),
            'mean_queries':round(sum(r['queries'] for r in rows)/n,5),
            'mean_integer_loop_proxy':round(sum(r['integer_loop_proxy'] for r in rows)/n,5),
            'mean_dense_macs':round(sum(r['model_macs'] for r in rows)/n,5)}


def run():
    base,_,gates,digests=checkpoints()
    classes=l.class_map()
    data=f.roster()
    calibration=set(c.visible for c in f.calibration())
    if any(c.visible in calibration for cases in data.values() for c in cases):
        raise AssertionError('calibration case leaked into evaluation')
    scores={};trace_sha={};by_topology={};collisions={};paired={}
    for split,cases in data.items():
        results={arm:[] for arm in ARMS}
        traces={arm:[] for arm in ARMS}
        per_top={top:{arm:[] for arm in ARMS} for top in l.TOPOLOGIES}
        for c in cases:
            logits={role:m.logits(c,base[role]) for role in gate.EXPERTS[:3]}
            for arm in ARMS:
                candidates=candidate_order(c,arm,logits,classes,gates)
                if len(set(candidates))!=len(candidates):raise AssertionError('duplicate class representatives')
                budget=(len(candidates) if arm=='complete_classes' else f.BUDGET)
                proposal=l.search(c,candidates,budget=budget)
                trace=proposal['trace']
                if len(trace)>f.BUDGET and arm!='complete_classes':raise AssertionError('budget exceeded')
                expected=sum(2*t.graph_cost(gr,c.topology) for gr in trace)
                if proposal['integer_loop_proxy']!=expected:raise AssertionError('unaccounted exact work')
                grade=t.certificate(c,proposal['graph'])
                if proposal['graph'] is not None and not grade['accepted']:
                    raise AssertionError('unverified candidate accepted')
                row={**grade,'queries':len(trace),'integer_loop_proxy':expected,'model_macs':macs(arm)}
                results[arm].append(row);per_top[c.topology][arm].append(row)
                traces[arm].append((c.identifier,[list(v) for v in trace],
                                    None if proposal['graph'] is None else list(proposal['graph'])))
        scores[split]={arm:aggregate(rows) for arm,rows in results.items()}
        by_topology[split]={top:{arm:aggregate(rows) for arm,rows in local.items()}
                            for top,local in per_top.items() if local[ARMS[0]]}
        trace_sha[split]={arm:sha256(json.dumps(rows,separators=(',',':')).encode()).hexdigest()
                          for arm,rows in traces.items()}
        if split not in ('collision','negative'):
            def exclusive(a,b):
                return sum(bool(x['future_correct']) and not bool(y['future_correct'])
                           for x,y in zip(results[a],results[b]))
            paired[split]={
                'adaptive_only_vs_uniform':exclusive('adaptive_mix12','uniform_mix12'),
                'uniform_only_vs_adaptive':exclusive('uniform_mix12','adaptive_mix12'),
                'adaptive_only_vs_constant':exclusive('adaptive_mix12','constant_mix12'),
                'constant_only_vs_adaptive':exclusive('constant_mix12','adaptive_mix12'),
                'adaptive_only_vs_balanced':exclusive('adaptive_mix12','balanced_gru12'),
                'balanced_only_vs_adaptive':exclusive('balanced_gru12','adaptive_mix12')}
        if split=='collision':
            collisions={arm:{'identical_trace':all(traces[arm][i][1:]==traces[arm][i+1][1:]
                                                 for i in range(0,len(cases),2)),
                             'false_future_acceptances':scores[split][arm]['false_accept']}
                        for arm in ARMS}
        if split not in ('collision','negative'):
            if scores[split]['complete_classes']['accepted']!=len(cases):
                raise AssertionError('full class search misses positive observed source')
    if scores['negative']['complete_classes']['accepted'] or not all(c['identical_trace'] for c in collisions.values()):
        raise AssertionError('negative/collision gate failed')
    if scores['collision']['complete_classes']['false_accept']!=f.COLLISION_PAIRS:
        raise AssertionError('future collision impossibility failed')
    summaries={}
    for fams,label in ((f.KNOWN,'known'),(f.NOVEL,'novel'),(f.FAMILIES,'all_positive')):
        summaries[label]={arm:{'cases':sum(scores[split][arm]['cases'] for split in fams),
                               'accepted':sum(scores[split][arm]['accepted'] for split in fams),
                               'correct_future':sum(scores[split][arm]['correct_future'] for split in fams)}
                         for arm in ARMS}
    return {
        'schema':'rrc20.receipt.v1',
        'status':'cpu_frozen_expert_calibration_new_generator_transfer',
        'source_sha256':source_hashes(),'checkpoint_sha256':digests,
        'calibration_sha256':f.old.digest(f.calibration()),
        'roster_sha256':{s:f.digest(cases) for s,cases in data.items()},
        'parameters':{'seed':f.SEED,'observed_degree':t.PREFIX,'forecast_degree':t.HORIZON,
                      'budget':f.BUDGET,'calibration_examples':len(f.calibration()),
                      'new_generator_families':list(f.NOVEL),'old_generator_families':list(f.KNOWN),
                      'gate_parameters':{'constant':4,'adaptive':60,'shuffled':60},
                      'gate_steps':train20.STEPS,'frozen_base_parameters':(6240,6240,6235),
                      'offline_symbolic_normalizations':len(g.GRAPHS)*len(l.TOPOLOGIES),
                      'inference_candidate_previews':0},
        'scores':scores,'summaries':summaries,'by_topology':by_topology,
        'trace_sha256':trace_sha,'collision_pairs':collisions,'paired_outcomes':paired,
        'limits':'Exact certificates cover two degree-12 demos only. Gate is fit on RRC-19 development labels, prior three neural models already trained. All four generator families are constructed from public series transformations. Full class search exceeds 12 queries; classifier and exact interpreter expenses must be counted separately.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--out',type=Path,default=HERE/'results'/'rrc20_pilot.json')
    args=parser.parse_args()
    result=run();payload=json.dumps(result,indent=2,sort_keys=True)+'\n'
    if args.check:
        if not args.out.is_file() or args.out.read_text()!=payload:
            raise SystemExit('RRC-20 source-hashed receipt mismatch')
        print('RRC-20 source-bound CPU replay verified')
    else:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(payload)
        print('Calibration:',{k:checkpoints()[2][k]['calibration_canonical_top12'] for k in GATE_ROLES})
        for group in f.FAMILIES:
            print(group,{arm:result['scores'][group][arm]['correct_future'] for arm in
                             ('adaptive_mix12','constant_mix12','uniform_mix12','balanced_gru12',
                              'skewed_gru12','balanced_ff12','mass12','complete_classes')})
        print('sum:',{group:{arm:x['correct_future'] for arm,x in item.items()} for group,item in result['summaries'].items()})
if __name__=='__main__':main()

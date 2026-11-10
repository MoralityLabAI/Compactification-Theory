"""RRC-18 exact class certificate + direct neural class prediction CPU replay."""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path

try:
    from . import learn18 as l, model18 as m, train18 as t18, certify18 as c18, topology16 as top, graph11 as g
except ImportError:
    import learn18 as l
    import model18 as m
    import train18 as t18
    import certify18 as c18
    import topology16 as top
    import graph11 as g

HERE=Path(__file__).resolve().parent
KINDS=('recurrent','feedforward','shuffled')
ARMS=('recurrent_top1','recurrent_top12','feedforward_top1','feedforward_top12',
      'shuffled_top12','certified_mass12','certified_canonical12','fixed12',
      'complete_certified_quotient','complete_64')
SOURCE_NAMES=('certify18.py','learn18.py','model18.py','train18.py','bench18.py',
              'topology16.py','topology15.py','graph11.py','graph12.py','graph13.py','graph14.py',
              'quotient17.py')

def source_hashes():
    return {n:sha256((HERE/n).read_bytes()).hexdigest() for n in SOURCE_NAMES}


def checkpoints():
    outputs={};digests={}
    for kind in KINDS:
        path=HERE/'results'/f'rrc18_{kind}.json'
        content=path.read_bytes();obj=json.loads(content)
        if obj['kind']!=kind or obj['parameter_count']!=m.parameter_count(kind):
            raise AssertionError('checkpoint kind or parameter count mismatch: '+kind)
        if obj['source_sha256']!=t18.sources() or obj['optimizer_steps']!=t18.STEPS:
            raise AssertionError('checkpoint source/optimizer drift: '+kind)
        outputs[kind]=obj;digests[kind]=sha256(content).hexdigest()
    return outputs,digests


def predict(case,arm,logits,classes):
    if arm.startswith('recurrent'):
        order=l.candidates(logits['recurrent'],case.topology,classes,count=(1 if arm.endswith('top1') else 12))
    elif arm.startswith('feedforward'):
        order=l.candidates(logits['feedforward'],case.topology,classes,count=(1 if arm.endswith('top1') else 12))
    elif arm=='shuffled_top12':order=l.candidates(logits['shuffled'],case.topology,classes,count=12)
    elif arm in ('certified_mass12','complete_certified_quotient'):
        order=l.mass_order(case.topology,classes)
    elif arm=='certified_canonical12':order=classes[case.topology]['representatives']
    elif arm in ('fixed12','complete_64'):order=g.GRAPHS
    else:raise ValueError('unknown arm')
    budget=(len(order) if arm in ('complete_64','complete_certified_quotient') else
            1 if arm.endswith('top1') else l.BUDGET)
    found=l.search(case,order,budget=budget)
    kind=(arm.split('_')[0] if arm.startswith(('recurrent','feedforward','shuffled')) else None)
    found['model_macs']=l.infer_cost(kind) if kind else 0
    return found


def aggregate(rows):
    n=len(rows)
    return {'cases':n,'accepted':sum(x['accepted'] for x in rows),
            'correct_future':sum(x['future_correct'] for x in rows),
            'false_accept':sum(x['false_accept'] for x in rows),
            'abstentions':sum(1-x['accepted'] for x in rows),
            'mean_queries':round(sum(x['queries'] for x in rows)/n,5),
            'mean_integer_work':round(sum(x['integer_loop_proxy'] for x in rows)/n,5),
            'mean_dense_macs':round(sum(x['model_macs'] for x in rows)/n,5)}


def run():
    mapping=l.class_map()
    cert=c18.certificate()
    models,digests=checkpoints()
    data=l.roster()
    assert len(data['train'])==len(l.TOPOLOGIES)*l.TRAIN_PER
    assert all(c.topology in l.TOPOLOGIES for group in data.values() for c in group)
    training={case.visible for case in data['train']}
    aliases={k:sum(c.visible in training for c in arr) for k,arr in data.items() if k!='train'}
    if any(aliases.values()):raise AssertionError('train/test visible alias detected')
    all_results={};trace_hashes={};topology_results={};pairing={};paired_successes={}
    for split,cases in data.items():
        if split in ('train','dev'):continue
        results={arm:[] for arm in ARMS}
        trace_log={arm:[] for arm in ARMS}
        by_topology={topo:{arm:[] for arm in ARMS} for topo in l.TOPOLOGIES}
        for case in cases:
            # Direct models only see observed series/topology; any proposed
            # rule must pass independent exact arithmetic after scoring.
            logits={kind:m.logits(case,ck) for kind,ck in models.items()}
            for arm in ARMS:
                prediction=predict(case,arm,logits,mapping)
                trace=prediction['trace']
                if len(set(trace))!=len(trace):raise AssertionError('duplicate paid query')
                if len(trace)>l.BUDGET and arm not in ('complete_64','complete_certified_quotient'):
                    raise AssertionError('candidate verification budget exceeded')
                if prediction['integer_loop_proxy']!=sum(2*top.graph_cost(p,case.topology) for p in trace):
                    raise AssertionError('arithmetic cost undercharge')
                grade=top.certificate(case,prediction['graph'])
                if prediction['graph'] is not None and not grade['accepted']:
                    raise AssertionError('invalid acceptance')
                row={**grade,'queries':len(trace),'integer_loop_proxy':prediction['integer_loop_proxy'],
                     'model_macs':prediction['model_macs']}
                results[arm].append(row)
                by_topology[case.topology][arm].append(row)
                trace_log[arm].append((case.identifier,[list(g) for g in trace],
                                      None if prediction['graph'] is None else list(prediction['graph'])))
        all_results[split]={arm:aggregate(rows) for arm,rows in results.items()}
        if split in ('signed_holdout','product_shift'):
            def only(a,b):
                return sum(int(x['future_correct'] and not y['future_correct'])
                           for x,y in zip(results[a],results[b]))
            paired_successes[split]={
                'recurrent_only_vs_feedforward':only('recurrent_top12','feedforward_top12'),
                'feedforward_only_vs_recurrent':only('feedforward_top12','recurrent_top12'),
                'recurrent_only_vs_mass':only('recurrent_top12','certified_mass12'),
                'mass_only_vs_recurrent':only('certified_mass12','recurrent_top12')}
        topology_results[split]={topo:{arm:aggregate(rows) for arm,rows in arms.items() if rows}
                                 for topo,arms in by_topology.items() if any(arms.values())}
        trace_hashes[split]={arm:sha256(json.dumps(trace,separators=(',',':')).encode()).hexdigest()
                             for arm,trace in trace_log.items()}
        if split=='future_collision':
            pairing={arm:{'same_trace_and_candidate':all(trace_log[arm][j][1:]==trace_log[arm][j+1][1:]
                                                  for j in range(0,len(cases),2)),
                          'false_accept':all_results[split][arm]['false_accept']}
                     for arm in ARMS}
    for split in ('signed_holdout','product_shift','future_collision'):
        count=len(data[split])
        if all_results[split]['complete_certified_quotient']['accepted']!=count or all_results[split]['complete_64']['accepted']!=count:
            raise AssertionError('certified finite quotient misses in-grammar case')
    if all_results['out_of_grammar']['complete_64']['accepted']!=0 or all_results['out_of_grammar']['complete_certified_quotient']['accepted']!=0:
        raise AssertionError('bad negative generator')
    if not all(x['same_trace_and_candidate'] for x in pairing.values()):raise AssertionError('future leakage')
    if pairing['complete_64']['false_accept']!=l.COUNTS['future_collision']//2:
        raise AssertionError('finite prefix collision gate failed')
    source=source_hashes()
    cert_text=json.dumps(cert,sort_keys=True,separators=(',',':')).encode()
    return {'schema':'rrc18.receipt.v1',
            'status':'exact_degree12_certificate_and_supervised_direct_class_cpu_pilot',
            'source_sha256':source,'checkpoint_sha256':digests,
            'certificate_sha256':sha256(cert_text).hexdigest(),
            'certificate':cert,
            'roster_sha256':{split:l.digest(cases) for split,cases in data.items()},
            'train_test_visible_aliases':aliases,
            'parameters':{'seed':l.SEED,'degree':l.PREFIX,'future_degree':l.HORIZON,
                          'query_budget':l.BUDGET,'training_cases':len(data['train']),
                          'development_cases':len(data['dev']),'test_cases':sum(len(x) for k,x in data.items() if k not in ('train','dev')),
                          'recurrent_params':m.parameter_count('recurrent'),
                          'feedforward_params':m.parameter_count('feedforward'),
                          'symbolic_programs_per_topology':len(g.GRAPHS),
                          'symbolic_offline_graph_normalizations':len(g.GRAPHS)*len(l.TOPOLOGIES),
                          'neural_candidate_previews':0},
            'scores':all_results,'by_topology':topology_results,
            'trace_sha256':trace_hashes,'collision_pairs':pairing,
            'paired_outcomes':paired_successes,
            'claims_boundary':'Exact only for integer coefficient polynomials through q^12. Model learns class labels with all topologies exposed; exact decoder and verifier supplied. Count model MACs separately from symbolic and interpreter work.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check',action='store_true')
    p.add_argument('--out',type=Path,default=HERE/'results'/'rrc18_pilot.json')
    args=p.parse_args()
    value=run();payload=json.dumps(value,indent=2,sort_keys=True)+'\n'
    if args.check:
        if not args.out.is_file() or args.out.read_text()!=payload:
            raise SystemExit('RRC-18 source-bound replay missing or stale')
        print('RRC-18 complete exact receipt verified')
    else:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(payload)
        print('certificate classes:',{k:v['class_count'] for k,v in value['certificate']['topologies'].items()})
        print('test groups:')
        for k in l.COUNTS:
            print(k,{arm:value['scores'][k][arm]['accepted'] for arm in ('recurrent_top1','recurrent_top12','feedforward_top12','shuffled_top12','certified_mass12','fixed12','complete_certified_quotient')})
if __name__=='__main__':main()

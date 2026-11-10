"""RRC-16 exact held-out join transfer, matched edit budgets and source-bound replay."""
from __future__ import annotations
import argparse
from hashlib import sha256
import heapq
import json
from pathlib import Path

try:
    from . import topology16 as b, policy16 as p, policy14 as p14, policy13 as p13
    from . import graph11 as g, graph14 as b14, train14, train16
except ImportError:
    import topology16 as b
    import policy16 as p
    import policy14 as p14
    import policy13 as p13
    import graph11 as g
    import graph14 as b14
    import train14
    import train16

HERE=Path(__file__).resolve().parent
KINDS=('recurrent','feedforward','shuffled')
ARMS=('multi_gru','multi_gru_reset','multi_gru_descriptor_off','multi_ff','shuffled',
      'frozen_gru14','residual_frontier','fixed_first12','exact_64')
SOURCES=('topology16.py','policy16.py','train16.py','bench16.py','topology15.py',
         'graph11.py','graph12.py','graph13.py','graph14.py','policy13.py',
         'policy14.py','train14.py')


def source_hashes():
    return {name:sha256((HERE/name).read_bytes()).hexdigest() for name in SOURCES}


def checkpoints():
    old,digests,prior=train16.prior_models()
    new={};new_hash={}
    for kind in KINDS:
        path=HERE/'results'/f'rrc16_{kind}.json'
        raw=path.read_bytes();ck=json.loads(raw)
        if ck['kind']!=kind or ck['parameter_count']!=p.parameter_count(kind):
            raise AssertionError('checkpoint schema/parameter mismatch: '+kind)
        if ck['source_sha256']!=train16.source_hashes():
            raise AssertionError('checkpoint source drift: '+kind)
        prior_kind='feedforward' if kind=='feedforward' else 'recurrent'
        if ck['pretrained_sha256']!=digests[prior_kind] or ck['initializer_sha256']!=digests['rrc13_frozen_initializer']:
            raise AssertionError('pretrained checkpoint changed: '+kind)
        if ck['heldout_join_in_training'] or ck['topologies_trained']!=list(b.TRAIN_TOPOLOGIES):
            raise AssertionError('heldout joined during training')
        new[kind]=ck;new_hash[kind]=sha256(raw).hexdigest()
    new_hash.update({f'rrc14_{kind}':digests[kind] for kind in ('recurrent','feedforward')})
    new_hash['rrc13_initializer']=digests['rrc13_frozen_initializer']
    return new,old,prior,new_hash


def edits(case,arm,models,old,initializer,budget=b.BUDGET):
    if budget<1 or budget>64:raise ValueError('budget 1..64')
    if arm not in ARMS[:-2]:raise ValueError('unsupported edit arm')
    initial=train14.start_graph(case,initializer)
    current=(initial,None,None)
    frontier=[];visited={};counter=0;backtracks=0;work=0
    macs=p13.inference_macs(initializer)
    trace=[]
    while len(visited)<budget:
        graph,parent_hidden,parent=current
        if graph in visited:
            if not frontier:break
            _,_,graph,parent_hidden,parent=heapq.heappop(frontier)
            current=(graph,parent_hidden,parent)
            continue
        signal=b.feedback(case,graph)
        visited[graph]=signal
        trace.append(graph)
        work+=2*b.graph_cost(graph,case.topology)
        if parent is not None and len(trace)>1 and trace[-2]!=parent:backtracks+=1
        if signal['accepted']:
            return {'graph':graph,'trace':tuple(trace),'integer_loop_proxy':work,
                    'model_macs':macs,'backtracks':backtracks,'initializer':initial}
        if arm in ('multi_gru','multi_gru_reset','multi_gru_descriptor_off','multi_ff','shuffled'):
            kind='feedforward' if arm=='multi_ff' else 'shuffled' if arm=='shuffled' else 'recurrent'
            ck=models[kind]
            x=b.features(case,graph,signal)
            if arm=='multi_gru_descriptor_off':x=x[:-3]+(0.,0.,0.)
            hidden=None if arm=='multi_gru_reset' else parent_hidden
            logits,new_hidden=p.step(x,ck,hidden)
            macs+=p.model_macs(kind)
            prios=[(-logits[index],index,child) for index,(child,_,_) in enumerate(b14.neighborhood(graph))]
        elif arm=='frozen_gru14':
            x=b14.action_features(case,graph,signal)
            logits,new_hidden=p14.step(x,old['recurrent'],parent_hidden)
            macs+=p14.mac_proxy('recurrent',1)
            prios=[(-logits[index],index,child) for index,(child,_,_) in enumerate(b14.neighborhood(graph))]
        else:
            new_hidden=None
            if arm!='residual_frontier':raise ValueError('unknown heuristic arm')
            priority=-(signal['first_fault']*25+signal['matched'])+min(signal['capped_residual'],100000)*1e-8
            prios=[(priority,index,child) for index,(child,_,_) in enumerate(b14.neighborhood(graph))]
        for priority,idx,child in prios:
            if child in visited:continue
            counter+=1
            heapq.heappush(frontier,(priority,counter,child,new_hidden,graph))
        if not frontier:break
        _,_,graph,parent_hidden,parent=heapq.heappop(frontier)
        current=(graph,parent_hidden,parent)
    return {'graph':None,'trace':tuple(trace),'integer_loop_proxy':work,
            'model_macs':macs,'backtracks':backtracks,'initializer':initial}


def catalogue(case,arm,budget=b.BUDGET):
    if arm=='fixed_first12':order=g.GRAPHS[:budget]
    elif arm=='exact_64':order=g.GRAPHS
    else:raise ValueError('invalid catalogue')
    trace=[];work=0;candidate=None
    for graph in order:
        signal=b.feedback(case,graph)
        trace.append(graph);work+=2*b.graph_cost(graph,case.topology)
        if signal['accepted']:
            candidate=graph;break
    return {'graph':candidate,'trace':tuple(trace),'integer_loop_proxy':work,
            'model_macs':0,'backtracks':0,'initializer':None}


def predict(case,arm,models,old,initializer,budget=b.BUDGET):
    if arm in ('fixed_first12','exact_64'):return catalogue(case,arm,budget)
    return edits(case,arm,models,old,initializer,budget)


def group_scores(rows):
    size=len(rows)
    sums={k:sum(row[k] for row in rows) for k in
          ('accepted','future_correct','false_accept','queries','integer_loop_proxy','model_macs','backtracks')}
    return {'cases':size,'accepted':sums['accepted'],'correct_future':sums['future_correct'],
            'false_accept':sums['false_accept'],'abstentions':size-sums['accepted'],
            'mean_graph_queries':round(sums['queries']/size,5),
            'mean_demo_executions':round(2*sums['queries']/size,5),
            'mean_primitive_calls':round(8*sums['queries']/size,5),
            'mean_integer_loop_proxy':round(sums['integer_loop_proxy']/size,5),
            'mean_model_macs':round(sums['model_macs']/size,5),
            'mean_backtracks':round(sums['backtracks']/size,5)}


def run():
    models,old,initializer,digests=checkpoints()
    data=b.roster()
    train_visible={c.visible for c in data['train']}
    aliases={split:sum(c.visible in train_visible for c in cases) for split,cases in data.items() if split!='train'}
    if any(aliases.values()):raise AssertionError('train/test raw alias')
    if any(c.topology=='hadamard' for split in ('train','dev') for c in data[split]):
        raise AssertionError('heldout join in train/dev')
    results={};trace_hashes={};per_case={};alias_old={};paired={}
    for split,cases in data.items():
        if split=='train':continue
        arm_rows={arm:[] for arm in ARMS}
        fingerprints={arm:[] for arm in ARMS}
        per_case[split]=[]
        alias_old[split]=sum(any(b.matching_graphs(case,old_topology) for old_topology in b.TRAIN_TOPOLOGIES)
                             for case in cases)
        for case in cases:
            row={'id':case.identifier,'topology':case.topology}
            for arm in ARMS:
                found=predict(case,arm,models,old,initializer)
                trace=found['trace'];q=len(trace)
                if len(set(trace))!=q or (q>b.BUDGET and arm!='exact_64'):
                    raise AssertionError('repeated or over-budget graph execution')
                if found['integer_loop_proxy']!=sum(2*b.graph_cost(c,case.topology) for c in trace):
                    raise AssertionError('unreported graph interpreter work')
                grade=b.certificate(case,found['graph'])
                if found['graph'] is not None and not grade['accepted']:
                    raise AssertionError('uncertified program returned')
                arm_rows[arm].append({**grade,'queries':q,'integer_loop_proxy':found['integer_loop_proxy'],
                                      'model_macs':found['model_macs'],'backtracks':found['backtracks']})
                fingerprints[arm].append((case.identifier,[list(g) for g in trace],
                                           None if found['graph'] is None else list(found['graph'])))
                row[arm]=grade['future_correct']
            per_case[split].append(row)
        results[split]={arm:group_scores(rows) for arm,rows in arm_rows.items()}
        trace_hashes[split]={arm:sha256(json.dumps(v,separators=(',',':')).encode()).hexdigest()
                             for arm,v in fingerprints.items()}
        if split=='new_join_collision':
            paired[split]={arm:{'identical_prediction_and_trace':all(
                fingerprints[arm][j][1:]==fingerprints[arm][j+1][1:]
                for j in range(0,len(fingerprints[arm]),2)),
                'future_false_accept':results[split][arm]['false_accept']}
                for arm in ARMS}
    pairing={}
    for split in ('new_join_signed','new_join_products'):
        arr=per_case[split]
        def paired_only(a,b):return sum(int(z[a] and not z[b]) for z in arr)
        pairing[split]={'recurrent_only_vs_ff':paired_only('multi_gru','multi_ff'),
                        'feedforward_only_vs_recurrent':paired_only('multi_ff','multi_gru'),
                        'recurrent_only_vs_residual':paired_only('multi_gru','residual_frontier'),
                        'residual_only_vs_recurrent':paired_only('residual_frontier','multi_gru')}
    return {'schema':'rrc16.receipt.v1','status':'exploratory_cpu_multitopology_heldout_join',
            'source_sha256':source_hashes(),'checkpoint_sha256':digests,
            'roster_sha256':{split:b.digest(cases) for split,cases in data.items()},
            'train_aliases':aliases,'old_grammar_aliases':alias_old,
            'parameters':{'seed':b.SEED,'prefix_degree':b.PREFIX,'future_degree':b.HORIZON,
                          'train_topologies':list(b.TRAIN_TOPOLOGIES),'heldout_join':b.HELDOUT_TOPOLOGY,
                          'training_examples':len(data['train']),'development_examples':len(data['dev']),
                          'graph_budget':b.BUDGET,'complete_search_max_queries':64,
                          'recurrent_parameters':p.parameter_count('recurrent'),
                          'feedforward_parameters':p.parameter_count('feedforward'),
                          'frozen_initializer_parameters':6148},
            'scores':results,'trace_sha256':trace_hashes,'paired_comparisons':pairing,
            'collision_pairs':paired,
            'claim_limit':'New Hadamard join implementation and public structural features are supplied; policies use exactly selected candidate executions, and any result is finite through degree 12, not an infinite identity proof.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--out',type=Path,default=HERE/'results'/'rrc16_pilot.json')
    args=parser.parse_args()
    payload=json.dumps(run(),sort_keys=True,indent=2)+'\n'
    if args.check:
        if not args.out.is_file() or args.out.read_text()!=payload:
            raise SystemExit('RRC-16 exact CPU receipt missing or stale')
        print('RRC-16 source-bound exact replay verified')
    else:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(payload)
        report=json.loads(payload)
        print(json.dumps({name:{arm:report['scores'][name][arm]['accepted'] for arm in ARMS}
                          for name in ('seen_wirings','new_join_signed','new_join_products')},indent=2))
if __name__=='__main__':main()

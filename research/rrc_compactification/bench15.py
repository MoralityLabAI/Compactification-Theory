"""RRC-15: frozen RRC-14 cross-topology transfer, exact, source-bound replay."""
from __future__ import annotations
import argparse
from hashlib import sha256
import heapq
import json
from pathlib import Path
import random

try:
    from . import topology15 as t, graph11 as g, graph14 as b14, policy14 as p14
    from . import policy13 as p13, train14, bench14
except ImportError:
    import topology15 as t
    import graph11 as g
    import graph14 as b14
    import policy14 as p14
    import policy13 as p13
    import train14
    import bench14

HERE = Path(__file__).resolve().parent
ARMS = ('frozen_recurrent','recurrent_reset','frozen_feedforward','untrained_residual',
        'untrained_random','fixed_first_12','exact_64')
SOURCES = ('topology15.py','bench15.py','graph11.py','graph12.py','graph13.py',
           'graph14.py','policy13.py','policy14.py','train14.py','bench14.py')


def source_hashes():
    return {name:sha256((HERE/name).read_bytes()).hexdigest() for name in SOURCES}


def checkpoint_locks():
    models,digests,prior = bench14.checkpoints()
    return models,prior,{**digests,'rrc14_pilot':sha256((HERE/'results'/'rrc14_pilot.json').read_bytes()).hexdigest()}


def _priority(signal):
    # Only parent state observed so far, never an unevaluated child response.
    return -(signal['first_fault']*25+signal['matched'])+min(signal['capped_residual'],100000)*1e-8


def search_edits(case,arm,models,prior,budget=t.BUDGET):
    """Reversible best-first search with graph evaluations charged once each.

    Policy chooses among nine unexecuted neighbors of visited programs.
    Caller can mask future/labels; neither is accessed until independent grade.
    """
    if budget<1 or budget>len(g.GRAPHS):raise ValueError('budget 1..64 required')
    if arm not in ARMS[:5]:raise ValueError('unknown edit policy')
    initial = train14.start_graph(case,prior)
    if initial not in g.GRAPHS:raise AssertionError('bad frozen initializer')
    is_neural = arm in ('frozen_recurrent','recurrent_reset','frozen_feedforward')
    ck=models['feedforward' if arm=='frozen_feedforward' else 'recurrent'] if is_neural else None
    frontier=[];visited={};counter=0;model_macs=p13.inference_macs(prior)
    calls=0;work=0;query_trace=[];restores=0
    current=(initial,None,None)
    rng=random.Random(int(sha256((str(t.SEED)+json.dumps(case.visible,separators=(',',':'))).encode()).hexdigest()[:12],16))
    while len(visited)<budget:
        node,parent_h,parent=current
        if node in visited:
            if not frontier:break
            _,_,node,parent_h,parent=heapq.heappop(frontier)
            current=(node,parent_h,parent)
            continue
        signal=t.feedback(case,node)
        visited[node]=signal
        query_trace.append(node)
        calls+=8
        work+=2*t.graph_cost(node,case.topology)
        if parent is not None and len(visited)>1 and parent!=query_trace[-2]:restores+=1
        if signal['accepted']:
            return {'graph':node,'queries':len(visited),'primitive_calls':calls,
                    'integer_loop_proxy':work,'model_macs':model_macs,
                    'backtracks':restores,'trace':tuple(query_trace),'initializer':initial}
        if is_neural:
            x=b14.action_features(case,node,signal)
            h_in=None if arm=='recurrent_reset' else parent_h
            logits,h=p14.step(x,ck,h_in)
            model_macs+=p14.mac_proxy(ck['kind'],1)
            priorities=[(-float(logits[i]),i,child) for i,(child,_,_) in enumerate(b14.neighborhood(node))]
        else:
            h=None
            if arm=='untrained_residual':
                priorities=[(_priority(signal),i,child) for i,(child,_,_) in enumerate(b14.neighborhood(node))]
            else:
                priorities=[(rng.random(),i,child) for i,(child,_,_) in enumerate(b14.neighborhood(node))]
        for priority,index,child in priorities:
            if child in visited:continue
            counter+=1
            heapq.heappush(frontier,(priority,counter,child,h,node))
        if not frontier:break
        _,_,node,parent_h,parent=heapq.heappop(frontier)
        current=(node,parent_h,parent)
    return {'graph':None,'queries':len(visited),'primitive_calls':calls,
            'integer_loop_proxy':work,'model_macs':model_macs,
            'backtracks':restores,'trace':tuple(query_trace),'initializer':initial}


def fixed_search(case,arm,budget=t.BUDGET):
    if arm=='fixed_first_12':order=g.GRAPHS[:budget]
    elif arm=='exact_64':order=g.GRAPHS
    else:raise ValueError('unknown fixed-search arm')
    trace=[];calls=0;work=0
    found=None
    for gr in order:
        trace.append(gr)
        # Every two-demo verification is one complete query, as for edit arms.
        r=t.feedback(case,gr)
        calls+=8
        work+=2*t.graph_cost(gr,case.topology)
        if r['accepted']:
            found=gr;break
    return {'graph':found,'queries':len(trace),'primitive_calls':calls,
            'integer_loop_proxy':work,'model_macs':0,'backtracks':0,
            'trace':tuple(trace),'initializer':None}


def predict(case,arm,models,prior,budget=t.BUDGET):
    if arm in ARMS[:5]:return search_edits(case,arm,models,prior,budget)
    return fixed_search(case,arm,budget)


def metrics(rows):
    n=len(rows)
    sums={k:sum(r[k] for r in rows) for k in ('accepted','correct_future','false_accept','queries',
             'primitive_calls','integer_loop_proxy','model_macs','backtracks')}
    return {'cases':n,'accepted':sums['accepted'],'correct_future':sums['correct_future'],
            'false_accept':sums['false_accept'],'abstentions':n-sums['accepted'],
            'mean_queries':round(sums['queries']/n,5),
            'mean_primitive_calls':round(sums['primitive_calls']/n,5),
            'mean_integer_loop_proxy':round(sums['integer_loop_proxy']/n,5),
            'mean_model_macs':round(sums['model_macs']/n,5),
            'mean_backtracks':round(sums['backtracks']/n,5)}


def run():
    models,prior,digests=checkpoint_locks()
    groups=t.roster()
    results={};trace_hash={};alias={};topology_novel={};paired={}
    for split,cases in groups.items():
        rows={arm:[] for arm in ARMS};traces={arm:[] for arm in ARMS}
        alias[split]=sum(bool(t.matching_graphs(c,'fork')) for c in cases)
        topology_novel[split]=sum(not t.matching_graphs(c,'fork') for c in cases)
        for case in cases:
            for arm in ARMS:
                response=predict(case,arm,models,prior)
                if response['queries']!=len(response['trace']) or len(set(response['trace']))!=len(response['trace']):
                    raise AssertionError('duplicate or uncharged candidate evaluation')
                if arm!='exact_64' and response['queries']>t.BUDGET:
                    raise AssertionError('interpreter budget exceeded')
                if response['primitive_calls']!=8*response['queries']:
                    raise AssertionError('interpreter primitive count invalid')
                if response['integer_loop_proxy']!=sum(2*t.graph_cost(gr,case.topology) for gr in response['trace']):
                    raise AssertionError('work proxy mismatch')
                grade=t.certify_future(case,response['graph'])
                if response['graph'] is not None and not grade['accepted']:
                    raise AssertionError('unverified candidate emitted')
                rows[arm].append({**grade,**{k:response[k] for k in ('queries','primitive_calls',
                                               'integer_loop_proxy','model_macs','backtracks')}})
                traces[arm].append((list(response['graph']) if response['graph'] else None,
                                    [list(x) for x in response['trace']]))
        if split.endswith('collision'):
            for arm in ARMS:
                for a,b in zip(traces[arm][::2],traces[arm][1::2]):
                    if a!=b:raise AssertionError('late future influenced search '+arm)
        if split.endswith('negative') and rows['exact_64'] and any(x['accepted'] for x in rows['exact_64']):
            raise AssertionError('negative case accepted by bounded exact reference')
        results[split]={arm:metrics(v) for arm,v in rows.items()}
        paired[split]={}
        for rival in ('recurrent_reset','frozen_feedforward','untrained_residual'):
            a=[x['accepted'] for x in rows['frozen_recurrent']]
            z=[x['accepted'] for x in rows[rival]]
            paired[split][rival]={
                'both_accepted':sum(x and y for x,y in zip(a,z)),
                'recurrent_only':sum(x and not y for x,y in zip(a,z)),
                'rival_only':sum(y and not x for x,y in zip(a,z)),
                'both_abstained':sum(not x and not y for x,y in zip(a,z)),
            }
        trace_hash[split]={arm:sha256(json.dumps(tr,separators=(',',':')).encode()).hexdigest()
                           for arm,tr in traces.items()}
        print('RRC-15 evaluated',split,len(cases),flush=True)
    return {'schema':'rrc15.receipt.v1','status':'frozen_zero_shot_cross_topology_cpu',
            'source_sha256':source_hashes(),'checkpoint_sha256':digests,
            'roster_sha256':{name:t.digest(v) for name,v in groups.items()},
            'parameters':{'seed':t.SEED,'graph_queries_budget':t.BUDGET,
                'observable_degree':t.DEGREE,'future_degree':t.HORIZON,
                'graph_vocabulary':list(t.OPS),'topology_choices':list(t.TOPOLOGIES),
                'counts':t.COUNTS,'training_updates':0,
                'frozen_recurrent_parameters':models['recurrent']['parameters'],
                'frozen_feedforward_parameters':models['feedforward']['parameters'],
                'frozen_initializer_parameters':prior['parameter_count']},
            'old_fork_expressible':alias,'old_fork_inexpressible':topology_novel,
            'scores':results,'paired_acceptance':paired,'predicted_trace_sha256':trace_hash,
            'scope':'RRC-14 weights never updated. New fixed graph wiring is public. Each query executes one complete candidate on both observed demos; the independent future grader is excluded from policy calls. Distinct topologies have different integer-work proxies. Exact-64 exceeds the budget and is only an upper control. No new algebraic theorem or infinite identity is inferred.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check',action='store_true')
    p.add_argument('--out',type=Path,default=HERE/'results'/'rrc15_pilot.json')
    args=p.parse_args();result=run()
    payload=json.dumps(result,indent=2,sort_keys=True)+'\n'
    if args.check:
        if not args.out.exists() or args.out.read_text()!=payload:
            raise SystemExit('RRC-15 source-bound replay mismatch')
        print('RRC-15 checked exact replay verified')
    else:
        args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(payload)
        print('RRC-15 receipt written',args.out)

if __name__=='__main__':main()

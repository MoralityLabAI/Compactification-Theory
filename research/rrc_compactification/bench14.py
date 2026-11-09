"""RRC-14 exact source-bound CPU replay with reversible edits and call budgets."""
from __future__ import annotations

import argparse
from hashlib import sha256
import heapq
import json
from pathlib import Path
import random

try:
    from . import graph11 as g, graph12 as b12, graph14 as b, policy14 as p, train14 as t
except ImportError:
    import graph11 as g
    import graph12 as b12
    import graph14 as b
    import policy14 as p
    import train14 as t

HERE=Path(__file__).resolve().parent
KINDS=('recurrent','feedforward','shuffled')
ARMS=('recurrent','recurrent_no_history','recurrent_blind_residual','feedforward',
      'shuffled','lexical_frontier','residual_frontier','random_frontier',
      'fixed_catalogue_12','frozen_initializer','preview_4','exact_search')


def hashes():
    names=('graph11.py','graph12.py','graph13.py','policy13.py',
           'graph14.py','policy14.py','train14.py','bench14.py')
    return {name:sha256((HERE/name).read_bytes()).hexdigest() for name in names}


def checkpoints():
    expected=t.source_hashes()
    prior,prior_sha=t.initial_checkpoint()
    out={};digests={}
    for kind in KINDS:
        path=HERE/'results'/f'rrc14_{kind}.json'
        content=path.read_bytes();m=json.loads(content)
        if m['source_hashes']!=expected or m['initializer_sha256']!=prior_sha:
            raise AssertionError('checkpoint source / initializer mismatch: '+kind)
        if m['kind']!=kind or m['parameters']!=(7393 if kind=='feedforward' else 7353):
            raise AssertionError('parameter shape drift: '+kind)
        out[kind]=m;digests[kind]=sha256(content).hexdigest()
    digests['rrc13_frozen_initializer']=prior_sha
    return out,digests,prior


def edits(case,arm,models,prior_ck,budget=b.QUERY_BUDGET):
    """Best-first policy over reversible single-token edits.

    Each popped *new* candidate is executed exactly once on both demo prefixes.
    Frontier edges are suggested using only already evaluated residuals. The
    policy can backtrack to an earlier state's untried edges without calling
    the interpreter again for that already evaluated graph. No candidate
    preview is performed to choose a frontier edit.
    """
    if budget<1 or budget>len(g.GRAPHS):raise ValueError('invalid query budget')
    neural=arm in ('recurrent','recurrent_no_history','recurrent_blind_residual','feedforward','shuffled')
    ck=models['feedforward'] if arm=='feedforward' else models['shuffled'] if arm=='shuffled' else models['recurrent'] if neural else None
    initial=t.start_graph(case,prior_ck)
    if initial not in g.GRAPHS:raise AssertionError('initializer produced non-graph')
    frontier=[];visited={};used=0;inference=0;backtracks=0;counter=0
    rng=random.Random(int(sha256((str(b.SEED)+json.dumps(case.visible,separators=(',',':'))).encode()).hexdigest()[:12],16))
    current=(initial,None,None,0)
    graph=None
    while used<budget:
        candidate,parent_state,parent_graph,depth=current
        if candidate in visited:
            # A previously tested graph can always be revisited at zero calls.
            if not frontier:break
            _,_,*item=heapq.heappop(frontier)
            current=tuple(item)
            continue
        signal=b.feedback(case,candidate)
        used+=1;visited[candidate]={'feedback':signal,'parent':parent_graph,'depth':depth}
        if parent_graph is not None and parent_graph!=list(visited)[-2]:backtracks+=1
        if signal['accepted']:
            graph=candidate
            break
        if neural:
            x=b.action_features(case,candidate,signal)
            if arm=='recurrent_blind_residual':x=x[:-16]+(0.0,)*16
            if arm=='recurrent_no_history':parent_state=None
            logits,h=p.step(x,ck,parent_state)
            inference+=p.mac_proxy(ck['kind'],1)
            priorities=[(-logits[i],i,child) for i,(child,_,_) in enumerate(b.neighborhood(candidate))]
        else:
            h=None
            if arm=='lexical_frontier':
                priorities=[(float(i),i,child) for i,(child,_,_) in enumerate(b.neighborhood(candidate))]
            elif arm=='random_frontier':
                priorities=[(rng.random(),i,child) for i,(child,_,_) in enumerate(b.neighborhood(candidate))]
            elif arm=='residual_frontier':
                # Compare ALREADY EVALUATED parent graph residuals, with no
                # unseen candidate response. Smaller integer residual first.
                priority=-(signal['first_fault']*25+signal['matched'])+min(signal['capped_residual'],100000)*1e-8
                priorities=[(float(priority),i,child) for i,(child,_,_) in enumerate(b.neighborhood(candidate))]
            else:raise ValueError('unknown edit policy')
        for prio,index,child in priorities:
            if child in visited:continue
            counter+=1
            heapq.heappush(frontier,(prio,counter,child,h,candidate,depth+1))
        if not frontier:break
        _,_,*item=heapq.heappop(frontier)
        current=tuple(item)
    return {'graph':graph,'queries':used,'model_macs':inference+p13_macs(prior_ck),
            'backtrack_or_nonlocal_steps':backtracks,
            'unique_graphs':len(visited),'initializer':initial,
            'final_proposal':list(visited)[-1],
            'visited':tuple(visited)}


def p13_macs(prior_ck):
    # Frozen RRC-13 feedback-disabled GRU, charged for every edit arm.
    try:
        from . import policy13
    except ImportError:
        import policy13
    return policy13.inference_macs(prior_ck)


def fixed(case,arm,budget=b.QUERY_BUDGET,prior_ck=None):
    if arm=='frozen_initializer':
        if prior_ck is None:raise ValueError('frozen initializer required')
        candidate=t.start_graph(case,prior_ck)
        return {'graph':candidate if b.feedback(case,candidate)['accepted'] else None,
                'queries':1,'model_macs':p13_macs(prior_ck),
                'backtrack_or_nonlocal_steps':0}
    if arm=='fixed_catalogue_12':
        order=b12.CANONICAL
        if len(order)<budget:raise ValueError('budget exceeds canonicals')
        chosen=None;uses=0
        for candidate in order[:budget]:
            uses+=1
            if b.feedback(case,candidate)['accepted']:
                chosen=candidate;break
        return {'graph':chosen,'queries':uses,'model_macs':0,'backtrack_or_nonlocal_steps':0}
    if arm=='preview_4':
        prior=g.guided(case,budget=4)
        # 64 two-demo PREVIEW executions + up to 4 full checks (with early stop).
        # Store actual work from graph11's declared cost convention separately.
        return {'graph':prior['graph'],'queries':prior['preview_calls']//8+prior['attempts'],
                'model_macs':0,'backtrack_or_nonlocal_steps':0,
                'primitive_calls':prior['primitive_calls'],
                'integer_loop_proxy':prior['work_proxy'],
                'full_candidate_checks':prior['attempts']}
    if arm=='exact_search':
        prior=g.search(case)
        return {'graph':prior['graph'],'queries':prior['attempts'],
                'model_macs':0,'backtrack_or_nonlocal_steps':0,
                'primitive_calls':prior['primitive_calls'],
                'integer_loop_proxy':prior['work_proxy'],
                'full_candidate_checks':prior['attempts']}
    raise ValueError('unsupported non-edit arm')


def grade(case,pred):
    graph=pred['graph']
    # A duplicate independent execution is strictly grading and excluded
    # from the policy's meter; it cannot influence selection or verification.
    certificate=b.verify_future(case,graph)
    if (graph is None)!=(certificate['accepted']==0):
        raise AssertionError('proposed program failed independent certificate')
    return certificate


def score_rows(rows):
    if not rows:return {}
    n=len(rows)
    keys=('accepted','future_correct','false_accept','queries','demo_executions',
          'primitive_calls','integer_loop_proxy','model_macs','backtrack_or_nonlocal_steps')
    sums={key:sum(r[key] for r in rows) for key in keys}
    return {'cases':n,'accepted':sums['accepted'],'correct_future':sums['future_correct'],
            'false_accept':sums['false_accept'],'abstained':n-sums['accepted'],
            'mean_calls':round(sums['queries']/n,6),
            'mean_demo_executions':round(sums['demo_executions']/n,6),
            'mean_primitive_calls':round(sums['primitive_calls']/n,6),
            'mean_integer_loop_proxy':round(sums['integer_loop_proxy']/n,6),
            'mean_model_macs':round(sums['model_macs']/n,6),
            'mean_backtracks':round(sums['backtrack_or_nonlocal_steps']/n,6)}


def run():
    models,digests,prior_ck=checkpoints()
    cases=b.dataset()
    train_signatures={c.visible for c in cases['train']}
    aliases={k:sum(c.visible in train_signatures for c in v) for k,v in cases.items() if k!='train'}
    if any(aliases.values()):raise AssertionError('train/eval exact prefix collision')
    if not all(c.hidden_graph in b.TRAIN_GRAPHS for c in cases['train']+cases['dev']):
        raise AssertionError('test classes entered training')
    if not all(c.hidden_graph in b.TEST_GRAPHS for split in ('unseen_composition','source_shift') for c in cases[split]):
        raise AssertionError('heldout split mismatch')
    metrics={};paths={}
    for family,items in cases.items():
        if family=='train':continue
        rows={name:[] for name in ARMS}
        graph_paths={name:[] for name in ARMS}
        for case in items:
            for arm in ARMS:
                if arm in ('fixed_catalogue_12','frozen_initializer','preview_4','exact_search'):
                    pred=fixed(case,arm,prior_ck=prior_ck)
                else:pred=edits(case,arm,models,prior_ck)
                score=grade(case,pred)
                q=pred['queries']
                calls=pred.get('primitive_calls',b.cost(q)['primitive_calls'])
                work=pred.get('integer_loop_proxy',b.cost(q)['integer_loop_proxy'])
                demos=pred.get('demo_executions', calls//4)
                row={**score,'queries':q,'demo_executions':demos,
                     'primitive_calls':calls,'integer_loop_proxy':work,
                     'model_macs':pred['model_macs'],
                     'backtrack_or_nonlocal_steps':pred['backtrack_or_nonlocal_steps']}
                rows[arm].append(row)
                graph_paths[arm].append(None if pred['graph'] is None else list(pred['graph']))
        metrics[family]={arm:score_rows(v) for arm,v in rows.items()}
        paths[family]={arm:sha256(json.dumps(v,separators=(',',':')).encode()).hexdigest()
                       for arm,v in graph_paths.items()}
        if family=='late_collision':
            for arm,graphs in graph_paths.items():
                for x,y in zip(graphs[::2],graphs[1::2]):
                    if x!=y:raise AssertionError(f'future collision leaked into {arm}')
        print('completed',family,'cases',len(items),flush=True)
    return {'schema':'rrc14.receipt.v1','status':'fixed_budget_reversible_graph_edit_cpu',
            'source_sha256':hashes(),'checkpoint_sha256':digests,
            'roster_sha256':{k:b.digest(v) for k,v in cases.items()},
            'training_prefix_aliases':aliases,
            'parameters':{'seed':b.SEED,'train_cases':len(cases['train']),
                          'dev_cases':len(cases['dev']),'test_counts':b.COUNTS,
                          'observed_degree':b.DEGREE,'forecast_degree':b.HORIZON,
                          'edit_budget_graph_queries':b.QUERY_BUDGET,
                          'recurrent_parameters':7353,'feedforward_parameters':7393,
                          'legal_graphs':len(g.GRAPHS),
                          'edit_neighbors':9,
                          'teacher_neighbor_calls_training_only':models['recurrent']['teacher_neighbor_executions'],
                          'initialization':'frozen RRC-13 active checkpoint, feedback off, zero query calls',
                          'one_graph_query':'both demo executions at observed degree, exact integer residual'},
            'metrics':metrics,'predicted_graph_sha256':paths,
            'claims_boundary':'Only finite exact observed-prefix and degree-32 forecast; the public interpreter supplies graph semantics. Teacher uses nine neighbor executions per supervised state, and neural model receives no unexecuted candidate results at test time. Budgeted exact and untrained controls have separate work accounting.'}


def main():
    p0=argparse.ArgumentParser(description=__doc__)
    p0.add_argument('--out',type=Path,default=HERE/'results'/'rrc14_pilot.json')
    p0.add_argument('--check',action='store_true')
    args=p0.parse_args()
    result=run()
    body=json.dumps(result,indent=2,sort_keys=True)+'\n'
    if args.check:
        if not args.out.exists() or args.out.read_text()!=body:
            raise SystemExit('RRC-14 frozen receipt missing or stale')
        print('RRC-14 CPU replay verified')
    else:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(body)
        print('wrote',args.out)
        print(json.dumps({k:{m:v['accepted'] for m,v in r.items()} for k,r in result['metrics'].items()},indent=2))

if __name__=='__main__':main()

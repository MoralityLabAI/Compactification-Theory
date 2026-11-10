"""RRC-17: source-locked probe-quotient search, frozen learned arms, and costs."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import random

try:
    from . import quotient17 as q, topology16 as b, graph11 as g, bench16
except ImportError:
    import quotient17 as q
    import topology16 as b
    import graph11 as g
    import bench16

HERE = Path(__file__).resolve().parent
SOURCES = ('quotient17.py', 'bench17.py', 'topology16.py', 'graph11.py',
           'graph12.py', 'graph13.py', 'graph14.py', 'topology15.py',
           'bench16.py', 'policy16.py', 'train16.py')
ARMS = ('quotient_mass12', 'quotient_work12', 'quotient_canonical12',
        'quotient_random12', 'first12', 'frozen_gru', 'frozen_reset',
        'frozen_feedforward', 'untrained_frontier', 'quotient_all', 'exact_64')


def source_hashes():
    return {name: sha256((HERE / name).read_bytes()).hexdigest() for name in SOURCES}


def orderings():
    quotients = {topo: q.build_quotient(topo) for topo in q.TOPOLOGIES}
    return quotients, {topo: {
        'quotient_mass12': q.choose_representatives(quotient, 'mass'),
        'quotient_work12': q.choose_representatives(quotient, 'mass_per_work'),
        'quotient_canonical12': q.choose_representatives(quotient, 'canonical'),
        'quotient_random12': q.choose_representatives(quotient, 'canonical'),
        'first12': g.GRAPHS,
        'quotient_all': q.choose_representatives(quotient, 'mass'),
        'exact_64': g.GRAPHS,
    } for topo, quotient in quotients.items()}


def prediction(case, arm, orders, models, old, initializer):
    if arm in ('frozen_gru', 'frozen_reset', 'frozen_feedforward', 'untrained_frontier'):
        mapped = {'frozen_gru': 'multi_gru', 'frozen_reset': 'multi_gru_reset',
                  'frozen_feedforward': 'multi_ff',
                  'untrained_frontier': 'residual_frontier'}[arm]
        item = bench16.predict(case, mapped, models, old, initializer, budget=q.BUDGET)
        return {**item, 'queries': len(item['trace']),
                'primitive_calls': 8 * len(item['trace'])}
    order = orders[case.topology][arm]
    if arm == 'quotient_random12':
        rng = random.Random(int(sha256((str(q.SEED) + repr(case.visible)).encode()).hexdigest()[:16],16))
        order = list(order)
        rng.shuffle(order)
    return q.search(case, order, budget=(len(order) if arm in ('quotient_all','exact_64') else q.BUDGET))


def metrics(rows):
    n = len(rows)
    return {
        'cases': n,
        'accepted': sum(x['accepted'] for x in rows),
        'correct_future': sum(x['future_correct'] for x in rows),
        'false_accept': sum(x['false_accept'] for x in rows),
        'abstentions': n - sum(x['accepted'] for x in rows),
        'mean_queries': round(sum(x['queries'] for x in rows)/n,5),
        'mean_primitive_calls': round(sum(x['primitive_calls'] for x in rows)/n,5),
        'mean_integer_loop_proxy': round(sum(x['integer_loop_proxy'] for x in rows)/n,5),
        'mean_dense_macs': round(sum(x['model_macs'] for x in rows)/n,5),
    }



def replication(models, old, initializer, orders):
    """Three independently seeded Hadamard batches; uniform graph syntax prior.

    This is an exploratory post-design robustness check. None of these cases
    is used to select or alter the class ordering or any frozen weight.
    """
    output = {}
    for offset in (4000, 5000, 6000):
        rng = random.Random(q.SEED + offset)
        cases = []
        seen = set()
        while len(cases) < 128:
            graph = rng.choice(g.GRAPHS)
            product = len(cases) % 4 == 0
            c = b._case(rng, graph, 'hadamard', f'replication_{offset}',
                        f'case-{len(cases):03d}', product=product)
            if c.visible in seen: continue
            seen.add(c.visible)
            cases.append(c)
        arms = ('quotient_mass12','first12','frozen_gru','frozen_feedforward',
                'untrained_frontier','quotient_all','exact_64')
        rows = {arm:[] for arm in arms}
        for case in cases:
            for arm in arms:
                found = prediction(case,arm,orders,models,old,initializer)
                grade = b.certificate(case,found['graph'])
                rows[arm].append({**grade,**{k:found[k] for k in
                                  ('queries','primitive_calls','integer_loop_proxy','model_macs')}})
        output[str(q.SEED+offset)] = {'roster_sha256':q.hash_roster(cases),
                                      'scores':{arm:metrics(rows[arm]) for arm in arms}}
    return output

def run():
    frozen, old, init, digest = bench16.checkpoints()
    quotients, orders = orderings()
    stability = {name: q.verify_probe_stability(part) for name, part in quotients.items()}
    groups = q.roster()
    alias_old = {}
    partition = {}
    scores = {}
    trace_digests = {}
    collisions = {}
    for topo, block in quotients.items():
        partition[topo] = {
            'classes': len(block.classes),
            'class_sizes_desc': sorted((len(cls) for cls in block.classes),reverse=True),
            'top12_syntax_prior_mass_numerator': sum(len(z) for z in block.sorted_classes(mode='mass')[:12]),
            'prior_mass_denominator': len(g.GRAPHS),
            'probe_graph_executions': block.total_probe_graph_executions,
            'probe_integer_loop_proxy': block.total_probe_integer_work,
            'probe_sha256': block.probes_hash,
            'fresh_stress_comparisons': stability[topo],
        }
    for split, cases in groups.items():
        alias_old[split] = sum(bool([x for oldtop in b.TRAIN_TOPOLOGIES
                                     for x in b.matching_graphs(c,oldtop)]) for c in cases)
        rows = {arm: [] for arm in ARMS}
        traces = {arm: [] for arm in ARMS}
        for case in cases:
            for arm in ARMS:
                result = prediction(case,arm,orders,frozen,old,init)
                trace = result['trace']
                if len(trace) != result['queries'] or len(set(trace)) != len(trace):
                    raise AssertionError('query deduplication failure')
                if result['integer_loop_proxy'] != sum(2*b.graph_cost(gr,case.topology) for gr in trace):
                    raise AssertionError('work accounting failure')
                if result['primitive_calls'] != 8*len(trace):
                    raise AssertionError('primitive cost failure')
                if len(trace)>q.BUDGET and arm not in ('quotient_all','exact_64'):
                    raise AssertionError('12-check limit exceeded')
                grade = b.certificate(case,result['graph'])
                if result['graph'] is not None and not grade['accepted']:
                    raise AssertionError('unchecked candidate accepted')
                rows[arm].append({**grade, **{k: result[k] for k in (
                    'queries','primitive_calls','integer_loop_proxy','model_macs')}})
                traces[arm].append((tuple(case.visible), trace, result['graph']))
        scores[split] = {arm: metrics(rows[arm]) for arm in ARMS}
        trace_digests[split] = {arm:sha256(json.dumps(traces[arm],separators=(',',':')).encode()).hexdigest()
                                for arm in ARMS}
        if split=='hadamard_collision':
            collisions = {arm: {
                'identical_traces': all(traces[arm][i][1:] == traces[arm][i+1][1:]
                                        for i in range(0,len(cases),2)),
                'future_false_accept':scores[split][arm]['false_accept']}
                for arm in ARMS}
    for split in ('hadamard_uniform','hadamard_novel','hadamard_product',
                  'fork_uniform','cascade_uniform','additive_uniform','hadamard_collision'):
        if scores[split]['exact_64']['accepted'] != len(groups[split]):
            raise AssertionError('exact reference missed in-class source')
        if scores[split]['quotient_all']['accepted'] != len(groups[split]):
            raise AssertionError('probe quotient dropped a test source')
    repeated=replication(frozen,old,init,orders)
    if any(x['scores']['quotient_all']['accepted']!=128 or
           x['scores']['exact_64']['accepted']!=128 for x in repeated.values()):
        raise AssertionError('independent replication grammar-class miss')
    if scores['hadamard_negative']['exact_64']['accepted'] != 0:
        raise AssertionError('class-negative generator invalid')
    if not all(x['identical_traces'] for x in collisions.values()):
        raise AssertionError('collision trace changed with inaccessible future')
    if scores['hadamard_collision']['exact_64']['false_accept']!=q.COUNTS['hadamard_collision']//2:
        raise AssertionError('finite-prefix impossibility control failed')
    return {
        'schema':'rrc17.receipt.v1',
        'status':'exact_cpu_semantic_probe_quotient_fixed_budget',
        'source_sha256':source_hashes(),
        'rrc16_checkpoint_sha256':digest,
        'seed':q.SEED,'probe_prime':q.PRIME,'probe_count':q.PROBES,
        'prefix_degree':q.PREFIX,'horizon':q.HORIZON,'budget':q.BUDGET,
        'probe_semantic_groups':partition,
        'probe_build_total_executions':sum(x.total_probe_graph_executions for x in quotients.values()),
        'probe_build_total_nominal_work':sum(x.total_probe_integer_work for x in quotients.values()),
        'roster_sha256':{k:q.hash_roster(v) for k,v in groups.items()},
        'observed_old_topology_aliases':alias_old,
        'scores':scores,
        'trace_sha256':trace_digests,
        'late_collisions':collisions,
        'independent_seed_replications':repeated,
        'limitations':'Probe grouping is a high-confidence randomized heuristic over F_p, not universal algebraic equivalence proof. Shared offline build work and the full public interpreter are additional resources. Future certificates concern finite observed prefixes only.',
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--out',type=Path,default=HERE/'results'/'rrc17_pilot.json')
    args=parser.parse_args()
    output=json.dumps(run(),indent=2,sort_keys=True)+'\n'
    if args.check:
        if not args.out.is_file() or args.out.read_text()!=output:
            raise SystemExit('RRC-17 source-bound receipt missing or stale')
        print('RRC-17 complete CPU replay verified')
    else:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(output)
        obj=json.loads(output)
        for split in q.COUNTS:
            print(split, {arm:obj['scores'][split][arm]['accepted'] for arm in
                          ('quotient_mass12','quotient_work12','first12','frozen_gru',
                           'frozen_feedforward','untrained_frontier','quotient_all','exact_64')})

if __name__=='__main__': main()

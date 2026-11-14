"""Exact finite DSL landscape on discovery-only prefix verifier feedback.

Local maxima here refer to the 9-neighbor Hamming-one graph on 64 programs.
They are mathematical maxima of a *declared prefix score*, not of neural loss.
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence
from ..tasks import PROGRAMS, OPS, MathTask, check_candidate, fresh_pool

PREFIX_LIMIT=13


def neighbors(graph:tuple[str,str,str]):
    for site in range(3):
        for op in OPS:
            if op==graph[site]:continue
            changed=list(graph);changed[site]=op
            yield tuple(changed)


def prefix_score(task:MathTask,graph:tuple[str,str,str],*,oracle=None)->int:
    q=check_candidate(task,graph,oracle)
    if len(q['first_faults'])!=2:
        raise ValueError('expected two independent demonstrations')
    return min(PREFIX_LIMIT if fault is None else int(fault) for fault in q['first_faults'])


def local_maxima(task:MathTask,*,oracle=None):
    values={g:prefix_score(task,g,oracle=oracle) for g in PROGRAMS}
    result=[]
    for graph in PROGRAMS:
        own=values[graph]
        group=[values[g] for g in neighbors(graph)]
        if own>=max(group) and any(own>x for x in group):
            result.append({'graph':list(graph),'prefix_score':own,
                           'neighbor_max':max(group),'neighbors':len(group),
                           'is_true_source_graph':graph==task.source_graph})
    return result


def mine_discovery(tasks:Sequence[MathTask],*,oracle=None,max_tasks:int=48,peaks:int=4):
    if peaks<1 or max_tasks<1:raise ValueError('positive limits required')
    seen=set();rows=[]
    for task in tasks[:max_tasks]:
        if task.case_id.startswith('readout') or task.case_id.startswith('test'):
            raise ValueError('reserved readout is forbidden in discovery mining')
        if task.signature in seen:raise ValueError('duplicate public discovery signature')
        seen.add(task.signature)
        maxima=local_maxima(task,oracle=oracle)
        deceptive=[m for m in maxima if not m['is_true_source_graph'] and m['prefix_score'] < PREFIX_LIMIT]
        if not deceptive:
            deceptive=[m for m in maxima if not m['is_true_source_graph']]
        if deceptive:
            best=sorted(deceptive,key=lambda m:(-m['prefix_score'],tuple(m['graph'])))[0]
            rows.append({'case_id':task.case_id,'source_signature':task.signature,
                         'family':task.family,'topology':task.topology,
                         'source_graph':list(task.source_graph),
                         'peak_program':best['graph'],'prefix_score':best['prefix_score'],
                         'neighbor_max':best['neighbor_max'],
                         'peak_type':'accepted_prefix_alias' if best['prefix_score']==PREFIX_LIMIT else 'first_fault_local_peak',
                         'n_local_maxima':len(maxima),
                         'verifier_queries':len(PROGRAMS)})
    grouped={}
    for row in rows:
        key=(row['family'],row['topology'],tuple(row['source_graph']))
        if key not in grouped or (-row['prefix_score'],row['case_id']) < (-grouped[key]['prefix_score'],grouped[key]['case_id']):
            grouped[key]=row
    # Prioritize witnessed first-fault local optima. Accepted aliases of the
    # observed prefix are a separate phenomenon and must not crowd them out.
    chosen=sorted(grouped.values(),key=lambda r:(
        r['peak_type']!='first_fault_local_peak',-r['prefix_score'],r['case_id']))[:peaks]
    return {'landscape_tasks':min(max_tasks,len(tasks)),'verifier_queries':len(PROGRAMS)*min(max_tasks,len(tasks)),
            'scored_discovery_tasks':len(rows),'peak_count':len(chosen), 'peaks':chosen,
            'definition':'Hamming-one local maxima of exact observed-prefix agreement'}


def generate_fresh_challenges(peaks:Sequence[dict],*,backend:str,oracle=None,
                              per_peak:int=4,seed_base:int=990000,forbidden:set[str]):
    """Generated gold remains in private task objects until frozen evaluation."""
    if per_peak<1:raise ValueError('per_peak must be positive')
    seen=set(forbidden);out=[]
    for i,peak in enumerate(peaks):
        if peak.get('prefix_score') is None or peak['prefix_score']<0:
            raise ValueError('invalid peak')
        seeds=[seed_base+i*10000+j for j in range(per_peak*5)]
        pool=fresh_pool('scm16_new_eval',seeds,family=peak['family'],
                        topology=peak['topology'],graph=tuple(peak['source_graph']),
                        oracle=oracle if backend=='evalmath' else None,forbidden=seen)
        if len(pool)<per_peak:raise ValueError('generation exhausted')
        for task in pool[:per_peak]:
            if task.signature in seen:raise ValueError('signature reused')
            seen.add(task.signature);out.append(task)
    return out

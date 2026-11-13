"""FC-08 registered source-only baselines, implemented before neural evaluation.

All policies consume only original graph atoms, facts, rules and goal.
No gold labels, reference proofs, or minimum-proof metadata are consulted.
The best-first source-space limit is exactly 24 popped states and no more
than five distinct novel-rule actions per candidate derivation.
"""
from __future__ import annotations
from collections import deque
import hashlib
import heapq
import random
import time
from typing import Any

from ..fc06.posthoc_backward_slice import priority_trace
from ..fc07.posthoc_unary import unary_shortest_trace

BUDGET=5
MAX_POPPED=24


def legal(task,known):
    return [i for i,r in enumerate(task["rules"])
            if r["head"] not in known and set(r["premises"])<=known]


def relaxed_goal_distances(task:dict)->dict[str,int]:
    reversed_graph={x:[] for x in task["atoms"]}
    for rule in task["rules"]:
        for atom in rule["premises"]:
            reversed_graph[rule["head"]].append(atom)
    distances={task["goal"]:0}
    queue=deque([task["goal"]])
    while queue:
        current=queue.popleft()
        for atom in reversed_graph[current]:
            if atom not in distances:
                distances[atom]=distances[current]+1
                queue.append(atom)
    return distances


def best_first(task:dict,*,budget:int=BUDGET,
               max_popped:int=MAX_POPPED)->dict[str,Any]:
    if not 1<=budget<=5 or max_popped<1:
        raise ValueError("unregistered best-first action or expansion budget")
    distances=relaxed_goal_distances(task)
    known0=frozenset(task["facts"])
    if not known0<=set(task["atoms"]) or task["goal"] not in task["atoms"]:
        raise ValueError("invalid source graph syntax")
    def h(known):
        return min((distances.get(x,len(task["atoms"])+1) for x in known),
                   default=len(task["atoms"])+1)
    tie=0
    queue=[(h(known0),0,tie,known0,())]
    visited={known0}
    popped=0
    recent=[]
    shortest_h=float("inf")
    representative=[]
    start=time.perf_counter()
    while queue and popped<max_popped:
        _,g,__,known,trace=heapq.heappop(queue)
        popped+=1
        recent=list(trace)
        score=h(known)
        if score<shortest_h or (score==shortest_h and len(trace)<len(representative)):
            shortest_h=score
            representative=list(trace)
        if task["goal"] in known:
            return {"trace":list(trace),"states_popped":popped,"found_goal":True,
                    "symbolic_search_ms":round(1000*(time.perf_counter()-start),6)}
        if g>=budget:
            continue
        for i in legal(task,known):
            next_known=known | frozenset([task["rules"][i]["head"]])
            if next_known in visited:
                continue
            visited.add(next_known)
            tie+=1
            next_g=g+1
            heapq.heappush(queue,(next_g+h(next_known),next_g,tie,
                                  next_known,trace+(i,)))
    return {"trace":representative if representative else recent,
            "states_popped":popped,"found_goal":False,
            "symbolic_search_ms":round(1000*(time.perf_counter()-start),6)}


def baseline_trace(task:dict,method:str)->dict:
    start=time.perf_counter()
    if method=="bounded_best_first":
        return best_first(task)
    if method=="backward_slice":
        trace=priority_trace(task,budget=BUDGET)
    elif method=="unary_bfs":
        trace=unary_shortest_trace(task,budget=BUDGET)
    elif method in ("random_legal","goal_first"):
        digest=hashlib.sha256(task["case_id"].encode()).digest()
        rng=random.Random(2917+int.from_bytes(digest[:4],"big"))
        known=set(task["facts"])
        trace=[]
        for _ in range(BUDGET):
            if task["goal"] in known:
                break
            choices=legal(task,known)
            if not choices:
                break
            goals=[i for i in choices if task["rules"][i]["head"]==task["goal"]]
            chosen=min(goals) if method=="goal_first" and goals else rng.choice(choices)
            trace.append(chosen)
            known.add(task["rules"][chosen]["head"])
    else:
        raise ValueError("unregistered source-only control "+method)
    return {"trace":trace,"states_popped":0,
            "found_goal":None,
            "symbolic_search_ms":round(1000*(time.perf_counter()-start),6)}


def first_observed_receipt(records:list[dict],verifier)->dict:
    """Measure all fixed controls on the untouched test positives.

    Labels are accessed only for score reporting after each source-only
    proposed trace has passed through the independent checker.
    """
    methods=("random_legal","goal_first","backward_slice",
             "unary_bfs","bounded_best_first")
    report={}
    for split in ("test_same","test_deep"):
        tasks=[x for x in records if x["split"]==split and x["label"]=="established"]
        per_method={}
        for method in methods:
            checked=0
            states=0
            for task in tasks:
                suggestion=baseline_trace(task,method)
                states+=suggestion["states_popped"]
                checked+=int(verifier(task,suggestion["trace"],budget=BUDGET)["verified"])
            per_method[method]={"certified_valid":checked,
                                "eligible_valid":len(tasks),
                                "positive_recall":round(checked/len(tasks),6),
                                "states_popped":states}
        report[split]=per_method
    return report

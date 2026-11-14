"""FC-09 registered goal-directed AND/OR Horn planner and source-only controls.

The state is the set of proved propositions, represented as a 12-bit mask.
h_max relaxes conjunction by charging a rule once plus the maximum cost of
its prerequisites, making an admissible lower bound for novelty-rule count.
A* runs under either a 24-popped-state cap or a per-example measured time
allowance. It reads graph syntax only, never gold, model outputs or proofs.
"""
from __future__ import annotations

from collections import deque
import hashlib
import heapq
import time

from ..fc08 import controls as old
from ..fc05 import proofs

ACTION_BUDGET=5
SYMBOLIC_FIXED_POPPED=24


def _compile(task:dict):
    atoms=task["atoms"]
    idx={x:i for i,x in enumerate(atoms)}
    if len(idx)!=len(atoms) or len(atoms)>12:
        raise ValueError("invalid atom set")
    start=sum(1<<idx[x] for x in task["facts"])
    goal=(1<<idx[task["goal"]])
    rules=[]
    for original,r in enumerate(task["rules"]):
        indices=tuple(idx[x] for x in r["premises"])
        pre=sum(1<<i for i in indices)
        head=1<<idx[r["head"]]
        rules.append((original,pre,head,indices,idx[r["head"]]))
    return len(atoms),start,goal,rules


def hmax_lower_bound(n:int,known:int,goal:int,compiled_rules)->int:
    """Relaxed conjunctive min(max) cost, from current known-set to goal."""
    infinity=n+10
    cost=[0 if known&(1<<i) else infinity for i in range(n)]
    for _ in range(n):
        modified=False
        for _,__,___,premises,head_idx in compiled_rules:
            prerequisite_cost=max(cost[i] for i in premises)
            if prerequisite_cost>=infinity:
                continue
            candidate=1+prerequisite_cost
            if candidate<cost[head_idx]:
                cost[head_idx]=candidate
                modified=True
        if not modified:
            break
    goal_idx=goal.bit_length()-1
    return cost[goal_idx]


def hmax_search(task:dict,*,budget:int=ACTION_BUDGET,
                max_popped:int|None=None,
                max_runtime_ms:float|None=None)->dict:
    """A* over actually derivable sets, with a hard expansion or time budget.

    For CPU matching the time scalar is determined outside this function
    solely from the frozen model's measured action-scoring runtime.
    A root state can be checked even with a zero allowance. Stop checks
    run before each subsequent pop and within successor generation.
    """
    if not 1<=budget<=5:
        raise ValueError("unsupported action budget")
    if max_popped is not None and max_popped<1:
        raise ValueError("max_popped must be positive")
    if max_runtime_ms is not None and max_runtime_ms<0:
        raise ValueError("negative runtime budget")
    started=time.perf_counter_ns()
    deadline=(started+int(max_runtime_ms*1_000_000)
              if max_runtime_ms is not None else None)
    n,start,goal,rules=_compile(task)
    root_h=hmax_lower_bound(n,start,goal,rules)
    queue=[(root_h,0,0,start,())]
    seen={start}
    tie=0
    popped=0
    latest=[]
    candidate=[]
    best_h=1_000_000
    timed_out=False
    while queue:
        if max_popped is not None and popped>=max_popped:
            break
        if deadline is not None and popped>0 and time.perf_counter_ns()>=deadline:
            timed_out=True
            break
        _,depth,_,known,trace=heapq.heappop(queue)
        popped+=1
        latest=list(trace)
        remaining=hmax_lower_bound(n,known,goal,rules)
        if remaining<best_h or (remaining==best_h and len(trace)<len(candidate)):
            candidate=list(trace)
            best_h=remaining
        if known&goal:
            elapsed=(time.perf_counter_ns()-started)/1_000_000
            return {"trace":list(trace),"found_goal":True,
                "states_popped":popped,"search_ms":round(elapsed,6),
                "allowed_ms":max_runtime_ms,
                "budget_overrun_ms":round(max(0.,elapsed-max_runtime_ms),6)
                    if max_runtime_ms is not None else 0.,
                "timed_out":False}
        if depth>=budget or depth+remaining>budget:
            continue
        for index,pre_mask,head_mask,_,__ in rules:
            if deadline is not None and popped>0 and time.perf_counter_ns()>=deadline:
                timed_out=True
                break
            if known&head_mask or (known&pre_mask)!=pre_mask:
                continue
            nxt=known|head_mask
            if nxt in seen:
                continue
            seen.add(nxt)
            nh=hmax_lower_bound(n,nxt,goal,rules)
            if depth+1+nh>budget:
                continue
            tie+=1
            heapq.heappush(queue,(depth+1+nh,depth+1,tie,nxt,trace+(index,)))
        if timed_out:
            break
    elapsed=(time.perf_counter_ns()-started)/1_000_000
    return {"trace":candidate if candidate else latest,
        "found_goal":False,"states_popped":popped,
        "search_ms":round(elapsed,6),"allowed_ms":max_runtime_ms,
        "budget_overrun_ms":round(max(0.,elapsed-max_runtime_ms),6)
            if max_runtime_ms is not None else 0.,
        "timed_out":timed_out}


def static_baseline(task:dict,method:str)->dict:
    if method=="hmax_24":
        return hmax_search(task,max_popped=SYMBOLIC_FIXED_POPPED)
    source_method={
        "fc08_24":"bounded_best_first",
        "random_legal":"random_legal",
        "goal_first":"goal_first",
        "backward_slice":"backward_slice",
        "unary_bfs":"unary_bfs",
    }.get(method)
    if source_method is None:
        raise ValueError("unregistered baseline method")
    result=old.baseline_trace(task,source_method)
    result["search_ms"]=result.pop("symbolic_search_ms",0.)
    result["allowed_ms"]=None
    result["budget_overrun_ms"]=0.
    result["timed_out"]=False
    return result


def first_source_controls(rows:list[dict])->dict:
    """Frozen pre-model records from every positive case, with exact replay."""
    methods=("hmax_24","fc08_24","random_legal",
             "goal_first","backward_slice","unary_bfs")
    out={}
    for split in ("test_same","test_deep"):
        positives=[r for r in rows if r["split"]==split and r["label"]=="established"]
        data={}
        for method in methods:
            correct=0
            pops=0
            for item in positives:
                proposed=static_baseline(item,method)
                success=proofs.verify_trace(item,proposed["trace"],budget=ACTION_BUDGET)
                correct+=int(success["verified"])
                pops+=proposed.get("states_popped",0)
            data[method]={"valid_cases":len(positives),
                "certified_positive":correct,
                "search_states_popped":pops}
        out[split]=data
    return out

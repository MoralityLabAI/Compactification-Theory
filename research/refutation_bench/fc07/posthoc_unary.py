"""Post-hoc zero-neural unary-implication BFS on the frozen FC-07 corpus.

Added only after the first preregistered FC-07 model result. It never looks at
source labels, reference traces, minimum-proof metadata, or solution oracle.
A rule with exactly ONE premise becomes a directed edge. A BFS from the
initial facts to the goal returns a concrete original rule-index proof trace.
It is a public, cheap symbolic baseline and cannot be retroactively counted
as a preregistered experimental arm.
"""
from __future__ import annotations
import argparse
from collections import deque,Counter
import hashlib
import json
from pathlib import Path

from . import test_data
from ..fc05 import proofs


def unary_shortest_trace(task:dict,budget:int=5)->list[int]:
    """Find shortest path through one-premise Horn implications only.

    Input is original graph syntax, never a gold proof. A graph with
    goal sources requiring conjunctions may be entailed but unresolved
    by this restricted search, which correctly returns no certificate.
    """
    atoms=task["atoms"]
    goal=task["goal"]
    facts=task["facts"]
    if goal in facts:
        return []
    outgoing={node:[] for node in atoms}
    for rule_id,rule in enumerate(task["rules"]):
        antecedents=rule["premises"]
        if len(antecedents)==1:
            outgoing[antecedents[0]].append((rule["head"],rule_id))
    seen=set(facts)
    parent={}
    queue=deque(facts)
    while queue:
        source=queue.popleft()
        for node,rule_id in outgoing[source]:
            if node in seen:
                continue
            seen.add(node)
            parent[node]=(source,rule_id)
            if node==goal:
                trace=[]
                current=node
                while current not in facts:
                    prev,index=parent[current]
                    trace.append(index)
                    current=prev
                trace.reverse()
                return trace if len(trace)<=budget else []
            queue.append(node)
    return []


def evaluate(rows:list[dict])->dict:
    records=[]
    by_split=Counter()
    for task in rows:
        trace=unary_shortest_trace(task,budget=5)
        verdict=proofs.verify_trace(task,trace,budget=5)
        if verdict["verified"] and task["label"]!="established":
            raise AssertionError("false proof verified")
        if task["label"]=="established" and not verdict["verified"]:
            by_split[(task["split"],"missed")]+=1
        if verdict["verified"]:
            by_split[(task["split"],"positive")]+=1
        records.append({
            "case_id":task["case_id"],"split":task["split"],
            "family_id":task["family_id"],"source_label":task["label"],
            "trace":trace,"trace_sha256":proofs.trace_sha256(trace),
            "verified_positive":bool(verdict["verified"]),
            "steps":len(trace),"checker_reason":verdict["reason"],
        })
    return {
      "kind":"posthoc_zero_neural_unary_graph_bfs",
      "frozen_fc07_source":"6cee4fdba91ba9c13ad47c492a127c7dc1884090",
      "source_used_as_input":"graph facts, rule premise lists, rule heads and goal only",
      "preregistered":False,
      "trainable_parameters":0,"neural_recurrent_calls":0,
      "n":len(rows),
      "source_positive":sum(x["label"]=="established" for x in rows),
      "certified_positive":sum(x["verified_positive"] for x in records),
      "wrong_positive_certificates":0,
      "by_split":{s:{
            "valid":sum(t["label"]=="established" for t in rows if t["split"]==s),
            "certified":by_split[s,"positive"],
            "missed":by_split[s,"missed"]}
            for s in ("test_same","test_deep")},
      "predictions":records,
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--lean",type=Path)
    a=p.parse_args()
    data=test_data.generate_tests()
    receipt=evaluate(data)
    if a.lean:
        model_proofs=[(task,unary_shortest_trace(task)) for task in data
                      if task["label"]=="established"]
        a.lean.parent.mkdir(parents=True,exist_ok=True)
        a.lean.write_text(proofs.lean_document(model_proofs),encoding="utf8")
        receipt["generated_lean_sha256"]=hashlib.sha256(a.lean.read_bytes()).hexdigest()
        receipt["generated_lean_theorems"]=len(model_proofs)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(receipt,sort_keys=True,indent=2)+"\n")
    print(json.dumps({k:v for k,v in receipt.items() if k!="predictions"},
                     sort_keys=True,indent=2))


if __name__=="__main__":
    main()

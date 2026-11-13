"""Exact finite behavioral-history quotient for 4/6-symbol rotated lookup.

A one-query signature identifies only a rotated goal truth table. Including
all two-query final outputs distinguishes some states that share an immediate
signature, so the two-step quotient is strictly finer on both vocabularies.
"""
from collections import defaultdict
import json


def one(g,p,v):
    return tuple((g >> ((p+q)%v))&1 for q in range(v))


def two(g,p,v):
    return tuple((g >> ((first+second)%v))&1 for first in range(v) for second in range(v))


def reference():
    results={}
    for v in (4,6):
        states=[(g,p) for g in range(1<<v) for p in range(v)]
        immediate=defaultdict(list)
        joint=defaultdict(list)
        for g,p in states:
            a=one(g,p,v)
            b=two(g,p,v)
            immediate[a].append((g,p))
            joint[(a,b)].append((g,p))
        witness=((1,0),(2,1))
        assert one(1,0,v)==one(2,1,v)
        assert two(1,0,v)!=two(2,1,v)
        assert len(immediate)==1<<v
        assert all(len(members)==v for members in immediate.values())
        assert len(joint)>len(immediate)
        results[str(v)]={
            "alphabet":v,"goals":1<<v,"states":len(states),
            "immediate_signature_classes":len(immediate),
            "joint_one_and_two_step_classes":len(joint),
            "current_equivalence_class_size":v,
            "strict_refinement":len(joint)>len(immediate),
            "witness":[list(x) for x in witness],
            "witness_immediate_signature":list(one(1,0,v)),
            "witness_two_step_disagrees":True,
        }
    return {"evidence":"exhaustive finite history-signature partitions", "results":results}


if __name__=="__main__":
    print(json.dumps(reference(),indent=2,sort_keys=True))

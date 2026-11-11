"""SHAFT-COMP-24: XOR cancellation and previous-input gauge, exact finite test."""
from __future__ import annotations
import itertools
import json

V=6
PERMS=tuple(itertools.permutations(range(V)))
FAMILIES={"repeat_toggle":(0,1,2,3,4,5),"successor_toggle":(1,2,3,4,5,0)}

def one_signature(b):
    return tuple((b>>c)&1 for c in range(V))

def two_signature(b,previous,rule):
    if not 0<=b<64 or not 0<=previous<V or sorted(rule)!=list(range(V)):
        raise ValueError("invalid input")
    return tuple(((b>>c)&1)^int(a==rule[previous])
        for a in range(V) for c in range(V))

def identify_from_exact(b,previous,signature):
    one=one_signature(b)
    return tuple(int(all(one[c]^signature[V*a+c] for c in range(V))) for a in range(V))

def compatible_rules(observations):
    return [r for r in PERMS if all(r[p]==a for p,a in observations.items())]

def rename_rule(rule,latent_label_permutation):
    return tuple(rule[latent_label_permutation[j]] for j in range(V))

def reference():
    cases=0
    for family,rule in FAMILIES.items():
        for b in range(64):
            for p in range(V):
                s=two_signature(b,p,rule)
                t=identify_from_exact(b,p,s)
                assert sum(t)==1 and t[rule[p]]==1
                assert all(one_signature(b)[c]^s[V*a+c]==int(a==rule[p])
                           for a in range(V) for c in range(V))
                cases+=1
        assert compatible_rules(dict(enumerate(rule)))==[rule]
        assert len(compatible_rules(dict(enumerate(rule[:4]))))==2
        assert len(compatible_rules(dict(enumerate(rule[:5]))))==1
        assert len({rename_rule(rule,perm) for perm in PERMS})==720
        for perm in PERMS:
            new_rule=rename_rule(rule,perm)
            for j in range(V):
                assert two_signature(17,j,new_rule)==two_signature(17,perm[j],rule)
    return {"experiment":"SHAFT-COMP-24","finite_canonical_cases":cases,
            "xor_positions_checked":cases*36,"permutation_rules":len(PERMS),
            "visible_previous_all_six_rule_candidates":1,
            "visible_previous_only_four_rule_candidates":2,
            "visible_previous_only_five_rule_candidates":1,
            "unlabeled_previous_gauge_representations":720,
            "family_true_rules":{k:list(v) for k,v in FAMILIES.items()},
            "conclusion":"canonical law is identifiable only with grounded previous-token labels"}

if __name__=="__main__":
    print(json.dumps(reference(),indent=2,sort_keys=True))

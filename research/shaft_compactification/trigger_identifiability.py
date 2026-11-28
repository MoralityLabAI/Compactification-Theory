"""SHAFT-COMP-24: exact identification of an unknown Boolean latch trigger.

Given a known previous input p, a current behavioral table B1 and a two-query
final signature B2, infer each rule bit by B1[c] XOR B2[a,c]. No oracle state
labels or hidden task weights are needed to recover the trigger row.
"""
from __future__ import annotations
import itertools
import json
import random
V=6
FAMILIES=('repeat_toggle','successor_toggle')

def true_trigger(family):
    if family not in FAMILIES: raise ValueError(family)
    return tuple(tuple(int(p==a) if family=='repeat_toggle' else int(a==(p+1)%V)
                       for a in range(V)) for p in range(V))

def one(effective):
    return tuple((effective>>c)&1 for c in range(V))

def two(effective,p,trigger):
    if not 0<=p<V:raise ValueError(p)
    b=one(effective)
    return tuple(b[c]^trigger[p][a] for a in range(V) for c in range(V))

def recover_row(b,t):
    if len(b)!=V or len(t)!=V*V:raise ValueError('wrong signature dimension')
    return tuple(b[0]^t[a*V] for a in range(V))

def pair_gauge_equal(effective,p,trigger,perm):
    # The previous token becomes a latent index when p is unobserved.
    permuted_trigger=tuple(trigger[perm[i]] for i in range(V))
    inverted={value:index for index,value in enumerate(perm)}
    return two(effective,p,trigger)==two(effective,inverted[p],permuted_trigger)

def reference():
    exact_pairs=0
    for family in FAMILIES:
        T=true_trigger(family)
        for effective,p in itertools.product(range(1<<V),range(V)):
            b=one(effective)
            assert recover_row(b,two(effective,p,T))==T[p]
            exact_pairs+=1
        for perm in itertools.permutations(range(V)):
            for p in range(V):
                assert pair_gauge_equal(11,p,T,perm)
    rng=random.Random(240124)
    arbitrary_rules=50
    for i in range(arbitrary_rules):
        T=tuple(tuple(rng.randrange(2) for _ in range(V)) for _ in range(V))
        for p in range(V):
            for effective in (0,1,17,63):
                assert recover_row(one(effective),two(effective,p,T))==T[p]
    return {'test':'exact unknown Boolean 6x6 trigger recovery from one-and-two-query signatures',
            'families':list(FAMILIES),'unknown_entries':V*V,'per_family_states':V*(1<<V),
            'family_state_checks':exact_pairs,'random_arbitrary_matrices_checked':arbitrary_rules,
            'rows_identified_with_observed_previous_token':True,
            'hidden_previous_token_row_permutation_ambiguity':True,
            'unlabelled_row_permutations_checked':2*720*V}

if __name__=='__main__':print(json.dumps(reference(),sort_keys=True,indent=2))

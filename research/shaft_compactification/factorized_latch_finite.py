"""SHAFT-COMP-22 exact behavioral-gauge quotient and factorized decoder.

This is a finite, deterministic reference, independent of any learned models.
Complemented goals and opposite latches are behaviorally gauge-equivalent.
"""
from __future__ import annotations
from itertools import product
import json
from research.shaft_compactification.persistent_latch_finite import (
    V, GOALS, STATES, FAMILIES, trigger, step, one, two,
)

ALL_BITS = GOALS - 1


def canonical(state):
    g, l, p = state
    return (g, l, p)


def reduced(state):
    g, l, p = state
    return (g ^ (ALL_BITS if l else 0), p)


def reduced_step(code, q, family):
    effective, previous = code
    output = (effective >> q) & 1
    toggle = trigger(previous, q, family)
    return (effective ^ (ALL_BITS if toggle else 0), q), output


def canonical_two(goal, latch, previous, family):
    return tuple(((goal >> second) & 1) ^ latch ^ trigger(previous, first, family)
                 for first in range(V) for second in range(V))


def reduced_two(effective, previous, family):
    return tuple(((effective >> second) & 1) ^ trigger(previous, first, family)
                 for first in range(V) for second in range(V))


def reduced_rollout(code, word, family):
    result = []
    for q in word:
        code, y = reduced_step(code, q, family)
        result.append(y)
    return code, tuple(result)


def reference():
    result = {}
    for family in FAMILIES:
        all_red = {reduced(s) for s in STATES}
        direct_two = {two(s, family) for s in STATES}
        assert len(all_red) == GOALS*V
        assert len(direct_two) == len(all_red) == GOALS*V
        # Any two different codes are distinguishable by either one or two probes.
        signature_to_reduced = {}
        for s in STATES:
            g,l,p = s
            assert canonical_two(g,l,p,family) == two(s, family)
            assert reduced_two(*reduced(s), family) == two(s, family)
            for q in range(V):
                next_full, y_full = step(s,q,family)
                next_reduced, y_red = reduced_step(reduced(s),q,family)
                assert reduced(next_full) == next_reduced and y_red == y_full
            sig=two(s,family)
            if sig in signature_to_reduced:
                assert signature_to_reduced[sig] == reduced(s)
            else: signature_to_reduced[sig]=reduced(s)
        assert len(signature_to_reduced) == GOALS*V
        sym_pairs=0
        for g,l,p in STATES:
            partner=(g ^ ALL_BITS, l ^ 1, p)
            assert reduced((g,l,p)) == reduced(partner)
            assert one((g,l,p),family) == one(partner,family)
            assert two((g,l,p),family) == two(partner,family)
            sym_pairs+=1
            # Exactly check every word length <=3; congruence carries to all lengths.
            for size in range(4):
                for word in product(range(V),repeat=size):
                    sa,oa=reduced_rollout(reduced((g,l,p)),word,family)
                    sb,ob=reduced_rollout(reduced(partner),word,family)
                    assert sa==sb and oa==ob
        # A disjoint-previous-token pair proves immediate alias and two-query split.
        g,l=3,0
        a,b=(g,l,0),(g,l,1)
        assert one(a,family)==one(b,family)
        assert two(a,family)!=two(b,family)
        result[family]={
            'canonical_state_count':len(STATES),
            'minimal_behavioral_classes':len(all_red),
            'two_query_signature_classes':len(direct_two),
            'canonical_states_per_reduced_code':len(STATES)//len(all_red),
            'complement_latch_gauge_pairs_checked':sym_pairs,
            'all_full_state_transition_and_output_checks':len(STATES)*V,
            'exhaustive_words_checked_per_state':sum(V**n for n in range(4)),
            'canonical_and_reduced_decoder_exact':True,
            'immediate_alias_two_query_split':True,
            'symmetry':'(g,l,p) and (g XOR 63,l XOR 1,p) have identical behavior for every future input word',
        }
    return {'experiment':'SHAFT-COMP-22', 'evidence':'exhaustive integer finite behavioral gauge and decoder',
            'vocabulary':V,'result':result}


if __name__=='__main__':
    print(json.dumps(reference(), indent=2, sort_keys=True))

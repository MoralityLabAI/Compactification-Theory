"""SHAFT-COMP-19: exhaustive six-symbol equal-capacity signature reference.

Repeating a six-bit immediate response six times adds output dimensions but
cannot split the exact behavioral equivalence classes. Native two-step
responses depend on the task history and define distinct equivalence classes.
"""
import json
from itertools import product

V=6
STATES=tuple(product(range(1<<V),range(V)))


def response(goal,query,previous,family):
    if family=='lag_rotate':
        bit=(query+previous)%V
        return (goal>>bit)&1
    if family=='lag_match':
        return ((goal>>query)&1)^int(query==previous)
    raise ValueError('unsupported family')


def immediate_signature(state,family):
    goal,previous=state
    return tuple(response(goal,q,previous,family) for q in range(V))


def duplicate_signature(state,family):
    return tuple(b for bit in immediate_signature(state,family) for b in (bit,)*V)


def history_signature(state,family):
    goal,_previous=state
    return tuple(response(goal,q2,q1,family)
                 for q1 in range(V) for q2 in range(V))


def partition_count(family,signatures):
    return len({tuple(signature(s,family) for signature in signatures) for s in STATES})


def reference():
    result={'experiment':'SHAFT-COMP-19','type':'exact_six_symbol_capacity_control',
            'states':len(STATES),'vocab':V,'families':{}}
    for family in ('lag_rotate','lag_match'):
        by_immediate={}
        for state in STATES:
            by_immediate.setdefault(immediate_signature(state,family),[]).append(state)
        witness=None
        for row in by_immediate.values():
            for a,b in product(row,repeat=2):
                if history_signature(a,family)!=history_signature(b,family):
                    witness=[list(a),list(b)]
                    break
            if witness is not None:break
        result['families'][family]={
            'immediate_behavior_classes':partition_count(family,(immediate_signature,)),
            'duplicated_immediate_classes':partition_count(family,(duplicate_signature,)),
            'history_behavior_classes':partition_count(family,(history_signature,)),
            'joint_immediate_history_classes':partition_count(family,(immediate_signature,history_signature)),
            'immediate_alias_split_by_history_witness':witness,
            'duplicated_immediate_is_exact_repetition':all(
                duplicate_signature(s,family)==tuple(bit for bit in immediate_signature(s,family) for _ in range(V)) for s in STATES),
            'two_step_history_independent_of_start_previous':all(
                history_signature((goal,p),family)==history_signature((goal,0),family)
                for goal in range(1<<V) for p in range(V)),
        }
    return result


if __name__=='__main__':
    print(json.dumps(reference(),sort_keys=True,indent=2))

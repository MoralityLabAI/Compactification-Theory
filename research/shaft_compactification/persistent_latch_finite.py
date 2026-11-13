"""SHAFT-COMP-20 exact 768-state persistent temporal-latch observer.

For each query, output precedes the latch transition. Immediate responses
depend only on goal and latch, while a two-query response retains the
previous-input register via its effect on the updated latch.
"""
from itertools import product
import json

V=6
GOALS=1 << V
FAMILIES=('repeat_toggle','successor_toggle')
STATES=tuple(product(range(GOALS),(0,1),range(V)))

def trigger(previous,q,family):
    if family=='repeat_toggle':
        return int(previous==q)
    if family=='successor_toggle':
        return int(q==(previous+1)%V)
    raise ValueError(family)

def step(state,q,family):
    g,l,p=state
    y=((g>>q)&1)^l
    return (g,l^trigger(p,q,family),q),y

def output(state,inputs,family):
    for q in inputs:
        state,y=step(state,q,family)
    return y

def one(state,family):
    return tuple(output(state,(q,),family) for q in range(V))

def two(state,family):
    return tuple(output(state,(q1,q2),family)
                 for q1 in range(V) for q2 in range(V))

def goal_code(state):
    return tuple(((state[0]>>q)&1) for q in range(V) for _ in range(V))

def full_signature(state,family):
    return goal_code(state),one(state,family),two(state,family)

def rollout(state,word,family):
    for q in word:
        state,_=step(state,q,family)
    return state

def exact_reference():
    result={}
    for family in FAMILIES:
        group_goal={goal_code(s) for s in STATES}
        group_one={one(s,family) for s in STATES}
        group_two={two(s,family) for s in STATES}
        group_all={full_signature(s,family) for s in STATES}
        prior_pairs=0
        two_separated=0
        long_separated=0
        witness=None
        for g,l in product(range(GOALS),(0,1)):
            for p,pp in product(range(V),repeat=2):
                if p==pp:
                    continue
                a,b=(g,l,p),(g,l,pp)
                prior_pairs+=1
                assert one(a,family)==one(b,family)
                assert goal_code(a)==goal_code(b)
                if two(a,family)!=two(b,family):
                    two_separated+=1
                first=p if family=='repeat_toggle' else (p+1)%V
                word=(first,)+(2,)*31
                aa=rollout(a,word,family)
                bb=rollout(b,word,family)
                assert aa[2]==bb[2]
                if one(aa,family)!=one(bb,family):
                    long_separated+=1
                if witness is None:
                    witness={'a':list(a),'b':list(b),'q_first':first,
                             'same_current':one(a,family)==one(b,family),
                             'two_queries_disagree':two(a,family)!=two(b,family),
                             'after_32_steps':list(word),
                             'output_differs_after_32':one(aa,family)!=one(bb,family)}
        assert [len(group_goal),len(group_one),len(group_two),len(group_all)] == [64,64,384,768]
        assert prior_pairs==two_separated==long_separated==3840
        result[family]={
            'state_count':len(STATES),
            'goal_signature_classes':len(group_goal),
            'immediate_signature_classes':len(group_one),
            'two_query_signature_classes':len(group_two),
            'joint_goal_immediate_two_query_classes':len(group_all),
            'same_goal_latch_ordered_different_previous_pairs':prior_pairs,
            'pairs_two_query_distinguishes':two_separated,
            'pairs_persist_after_32_shared_steps':long_separated,
            'witness':witness}
    return {'experiment':'SHAFT-COMP-20','status':'exact_exhaustive_automata',
            'vocabulary':V,'goals':GOALS,'families':result}

if __name__=='__main__':
    print(json.dumps(exact_reference(),indent=2,sort_keys=True))

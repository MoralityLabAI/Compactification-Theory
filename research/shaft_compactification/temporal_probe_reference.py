"""Exhaustive temporal-signature equivalence on 64 task states.

Two previous-token-dependent binary task families. One-step behavioral signatures
may alias different goals; final outputs of all 16 two-query continuations
separate those goals. Pure Python, deterministic, no neural model operations.
"""
from itertools import product
GOALS=tuple(range(16))
QUERIES=tuple(range(4))

def native_output(goal,previous,query,family):
    if family=="xor_lag":return (goal>>(previous^query))&1
    if family=="parity_gate":return ((goal>>query)&1)^((previous^query)&1)
    raise ValueError(family)

def one_step(goal,previous,family):
    return tuple(native_output(goal,previous,q,family) for q in QUERIES)

def two_step(goal,previous,family):
    return tuple(native_output(goal,q1,q2,family)
                 for q1 in QUERIES for q2 in QUERIES)

def reference():
    result={}
    for family in ("xor_lag","parity_gate"):
        states=list(product(GOALS,QUERIES))
        one=[one_step(g,p,family) for g,p in states]
        two=[two_step(g,p,family) for g,p in states]
        aliases=[(i,j) for i in range(len(states)) for j in range(i+1,len(states))
                 if one[i]==one[j] and two[i]!=two[j]]
        assert aliases
        assert all(two_step(g,0,family)==two_step(g,p,family)
                   for g in GOALS for p in QUERIES)
        assert len({two_step(g,0,family) for g in GOALS})==16
        i,j=aliases[0]
        result[family]={
            "physical_states":len(states),
            "one_step_observation_classes":len(set(one)),
            "two_step_observation_classes":len(set(two)),
            "one_step_alias_pairs_split_by_two_steps":len(aliases),
            "first_alias":[list(states[i]),list(states[j])],
            "first_alias_current_response":list(one[i]),
            "two_step_patterns_differ":two[i]!=two[j],
            "two_step_signature_previous_token_invariant":True}
    return {"evidence":"exhaustive_64_state_temporal_task_reference",
            "state_space":"16 goal truth tables times 4 previous tokens",
            "families":result}

if __name__=="__main__":
    import json
    print(json.dumps(reference(),sort_keys=True,indent=2))

"""Pure standard-library reference for SHAFT-COMP-16 temporal task families.

The first input has previous-token sentinel 0. All 16 goals remain four-bit
truth tables, but their query meaning changes with the preceding input.
"""


def family_trace(goal: int, tokens: tuple[int,...], family: str) -> tuple[int,...]:
    if not 0 <= goal < 16 or any(t not in range(4) for t in tokens):
        raise ValueError("goal must be 0..15 and tokens 0..3")
    if family not in ("lag_rotate","lag_match"):
        raise ValueError("unsupported family")
    previous=0
    outputs=[]
    for current in tokens:
        if family=="lag_rotate":
            y=(goal >> ((current+previous) % 4)) & 1
        else:
            y=((goal >> current)&1) ^ int(current==previous)
        outputs.append(y)
        previous=current
    return tuple(outputs)


def current_reply(goal: int,current: int,previous: int,family: str) -> int:
    if not 0 <= previous < 4:
        raise ValueError("previous token must be in 0..3")
    return family_trace(goal,(previous,current),family)[1]


def reference():
    examples={}
    for family in ("lag_rotate","lag_match"):
        witness=None
        counter=0
        for goal in range(16):
            for current in range(4):
                values={current_reply(goal,current,previous,family) for previous in range(4)}
                if len(values)>1:
                    counter+=1
                    if witness is None:
                        witness={"goal":goal,"current_query":current,
                                 "previous_0_reply":current_reply(goal,current,0,family),
                                 "previous_1_reply":current_reply(goal,current,1,family)}
        assert witness is not None
        examples[family]={"goal_current_combinations_history_sensitive":counter,
                          "total_goal_current_combinations":64,
                          "first_sensitive_witness":witness}
    return {"evidence":"exhaustive finite 16-goal 4-query temporal reference",
            "previous_input_sentinel":0,"family_sensitivity":examples,
            "reason":"same goal ID and current input can yield different answers after different histories"}


if __name__=="__main__":
    import json
    print(json.dumps(reference(),indent=2,sort_keys=True))

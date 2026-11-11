"""SHAFT-COMP-21: exact independent oracle for latch and previous-token supervision.

Native output uses the pre-update latch. Auxiliary labels use post-update
latch and the most recent token. Pure finite arithmetic, no neural dependency.
"""
from itertools import product
from research.shaft_compactification.persistent_latch_finite import V, GOALS, FAMILIES, step, one, two


def supervised_trace(goal, tokens, family):
    state = (goal, 0, 0)
    bits, latches, previous = [], [], []
    for q in tokens:
        state, y = step(state, q, family)
        bits.append(y)
        latches.append(state[1])
        previous.append(state[2])
    return tuple(bits), tuple(latches), tuple(previous)


def independent_trace(goal, tokens, family):
    prev, latch = 0, 0
    bits, latches, previous = [], [], []
    for q in tokens:
        bits.append(((goal >> q) & 1) ^ latch)
        if family == "repeat_toggle":
            toggle = int(q == prev)
        elif family == "successor_toggle":
            toggle = int(q == (prev + 1) % V)
        else:
            raise ValueError(family)
        latch ^= toggle
        prev = q
        latches.append(latch)
        previous.append(prev)
    return tuple(bits), tuple(latches), tuple(previous)


def reference():
    checks = 0
    for family in FAMILIES:
        for goal in range(GOALS):
            for n in range(4):
                for tokens in product(range(V), repeat=n):
                    assert supervised_trace(goal, tokens, family) == independent_trace(goal, tokens, family)
                    checks += 1
    separations = {}
    for family in FAMILIES:
        alias = split = 0
        for goal in range(GOALS):
            for latch in (0, 1):
                for p in range(V):
                    for pp in range(V):
                        if p == pp:
                            continue
                        a, b = (goal, latch, p), (goal, latch, pp)
                        alias += int(one(a, family) == one(b, family))
                        split += int(two(a, family) != two(b, family))
        assert alias == split == 3840
        separations[family] = {"one_query_aliases": alias, "two_query_splits": split}
    return {
        "experiment": "SHAFT-COMP-21",
        "type": "exhaustive_finite_label_oracle",
        "vocabulary": V,
        "goals": GOALS,
        "sequences_per_family_goal": sum(V**n for n in range(4)),
        "traces_checked": checks,
        "post_step_labels": "latch after update and previous token equals current token",
        "native_output_label": "goal bit XOR latch before update",
        "separations": separations,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(reference(), sort_keys=True, indent=2))

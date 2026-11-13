"""Exhaustive delayed-state counterexample to causal compactification.

A current bit may fully describe the visible output while concealing another
bit that affects later behavior. No network, devices, or persistence operations.
"""

from itertools import product

STATES = tuple(product((0, 1), repeat=3))


def step(state, activate):
    return (state[1], state[2], state[2]) if activate else state


def code(state):
    return state[0]


def run(state, actions):
    for action in actions:
        state = step(state, action)
    return state


def observe(state, actions):
    return code(run(state, actions))


def reference():
    s, t = (0, 0, 0), (0, 0, 1)
    same_current = code(s) == code(t)
    same_one_step = observe(s, (1,)) == observe(t, (1,))
    different_two_step = observe(s, (1, 1)) != observe(t, (1, 1))
    distinguished = sum(
        code(a) == code(b) and observe(a, (1, 1)) != observe(b, (1, 1))
        for a, b in product(STATES, repeat=2)
    )
    decoders = list(product(STATES, repeat=2))
    sound_short_decoders = sum(
        all(observe(decoder[code(s)], (1, 1)) == observe(s, (1, 1))
            for s in STATES)
        for decoder in decoders
    )
    full_state_sound = all(
        observe(s, actions) == observe(s, actions)
        for s in STATES for n in range(5)
        for actions in product((0, 1), repeat=n)
    )
    return {
        "evidence": "exhaustive_finite_counterexample",
        "states": len(STATES),
        "same_current_code_and_one_step_witness": same_current and same_one_step,
        "two_step_witness": different_two_step,
        "number_of_ordered_same_code_pairs_distinguishable_at_two_steps": distinguished,
        "possible_visible_bit_decoders": len(decoders),
        "universally_sound_visible_bit_decoders": sound_short_decoders,
        "universally_sound_full_state_decoder": full_state_sound,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(reference(), sort_keys=True, indent=2))

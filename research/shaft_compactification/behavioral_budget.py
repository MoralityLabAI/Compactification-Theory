"""Finite-bit behavioral identifier bounds for uniformly sampled tasks.

Distinct complete signatures require different exact codes. A fixed-length
k-bit code has at most 2**k values. This bounds exact identification of M
uniform tasks at min(M,2**k)/M. No device or persistence mechanism is modeled.
"""

from math import ceil, log2


def necessary_exact_bits(classes: int) -> int:
    if classes < 1:
        raise ValueError("at least one behavior class required")
    return ceil(log2(classes))


def maximum_uniform_exact_recovery(classes: int, bits: int) -> tuple[int, int]:
    if classes < 1 or bits < 0:
        raise ValueError("positive class count and nonnegative bit budget required")
    return min(classes, 2 ** bits), classes


def example_uniform_16_goal_bound() -> dict:
    return {
        "goal_classes": 16,
        "minimum_exact_bits": necessary_exact_bits(16),
        "ceiling_successes_for_16_uniform_goals": {
            str(bits): maximum_uniform_exact_recovery(16, bits)[0]
            for bits in range(5)
        },
        "denominator": 16,
        "evidence": "finite counting identity, no neural performance claims",
    }


if __name__ == '__main__':
    import json
    print(json.dumps(example_uniform_16_goal_bound(), sort_keys=True, indent=2))

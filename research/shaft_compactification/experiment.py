"""SHAFT-COMP-05: fixed, local synthetic behavioral compactification study.

No network, persistence agent, model weights, sharing implementation or external files
are used. A goal is just the index of a finite deterministic bit-valued policy.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from math import ceil, log2

GOALS = 16
INPUTS = 64
PUBLIC_INPUTS = 8
SEED = "shaft-comp-05-cpu-reference-v1"


def policy(goal: int, input_id: int) -> int:
    if not 0 <= goal < GOALS or not 0 <= input_id < INPUTS:
        raise ValueError("out-of-range synthetic goal/input")
    if input_id < PUBLIC_INPUTS:
        return 0
    value = f"{SEED}:{goal}:{input_id}".encode("utf-8")
    return hashlib.sha256(value).digest()[0] & 1


def signature(goal: int, indices: range) -> tuple[int, ...]:
    return tuple(policy(goal, index) for index in indices)


def partition_count(indices: range) -> int:
    return len({signature(goal, indices) for goal in range(GOALS)})


def minimum_prototype_cover(indices: range, max_errors: int) -> int:
    """Exact minimum number of existing policies covering all goal signatures.

    Representatives are drawn from the given goal library; this is a finite
    codebook-cover calculation, NOT unrestricted rate-distortion complexity.
    """
    vectors = [signature(g, indices) for g in range(GOALS)]
    coverage = []
    for prototype in vectors:
        cover = 0
        for goal, vector in enumerate(vectors):
            if sum(a != b for a, b in zip(prototype, vector)) <= max_errors:
                cover |= 1 << goal
        coverage.append(cover)
    full_mask = (1 << GOALS) - 1

    @lru_cache(None)
    def solve(remaining: int) -> int:
        if remaining == 0:
            return 0
        first = (remaining & -remaining).bit_length() - 1
        return 1 + min(
            solve(remaining & ~cover)
            for cover in coverage
            if cover & (1 << first)
        )

    return solve(full_mask)


def reference_result() -> dict:
    monitored = range(PUBLIC_INPUTS)
    heldout = range(PUBLIC_INPUTS, INPUTS)
    all_inputs = range(INPUTS)
    goal_selector_bits = ceil(log2(GOALS))
    raw_policy_table_bits = INPUTS
    pairs = GOALS * (GOALS - 1) // 2
    indistinguishable_monitored = sum(
        signature(a, monitored) == signature(b, monitored)
        for a in range(GOALS) for b in range(a + 1, GOALS)
    )
    distinguishable_heldout = sum(
        signature(a, heldout) != signature(b, heldout)
        for a in range(GOALS) for b in range(a + 1, GOALS)
    )
    assert indistinguishable_monitored == pairs
    assert distinguishable_heldout == pairs
    assert partition_count(all_inputs) == GOALS

    return {
        "status": "constructed_finite_cpu_reference_only",
        "source_seed": SEED,
        "goal_count": GOALS,
        "input_count": INPUTS,
        "monitored_input_count": PUBLIC_INPUTS,
        "heldout_input_count": INPUTS - PUBLIC_INPUTS,
        "goal_pairs": pairs,
        "monitor_indistinguishable_pairs": indistinguishable_monitored,
        "heldout_distinguishable_pairs": distinguishable_heldout,
        "exact_public_behavior_classes": partition_count(monitored),
        "exact_full_behavior_classes": partition_count(all_inputs),
        "goal_selector_bits_given_shared_substrate": goal_selector_bits,
        "naive_uncompressed_single_policy_table_bits": raw_policy_table_bits,
        "shared_library_raw_table_bits": GOALS * INPUTS,
        "prototype_cover_heldout_errors_0": minimum_prototype_cover(heldout, 0),
        "prototype_cover_heldout_errors_21": minimum_prototype_cover(heldout, 21),
        "uniform_goal_accuracy_after_complete_selector_erasure": f"1/{GOALS}",
        "uniform_goal_accuracy_with_selector_intact": "1/1",
        "note": "Conditional selector size excludes substrate and codebook costs; no neural model was evaluated.",
    }


def main() -> None:
    print(json.dumps(reference_result(), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()

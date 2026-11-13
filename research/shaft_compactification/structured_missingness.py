"""SHAFT-COMP-09: deterministic negative controls for structured observation errors.

These finite examples show where MCAR/independent-bit-flip guarantees cease
to apply. They describe abstract behavioral signatures only.
"""

import json

from research.shaft_compactification.gauge_identifiability import GAUGES
from research.shaft_compactification.noisy_gauge import (
    BUDGETS, DESIGNS, TRUE, anchors_for, packed_response,
)


def censored_query_candidates(design: str, budget: int, visible=(0, 1)):
    """Return gauges indistinguishable when designated query slots are always hidden."""
    if not visible or any(q not in range(4) for q in visible):
        raise ValueError("visible query slots must belong to 0,1,2,3")
    anchors = anchors_for(design, budget)
    visibility = sum(1 << (4 * i + q) for i in range(len(anchors)) for q in visible)
    true_code = packed_response(anchors, TRUE)
    return [g for g in GAUGES if ((packed_response(anchors, g) ^ true_code) & visibility) == 0]


def systematic_relabeling_matches_wrong_gauge(design: str, budget: int):
    """Show full-consistent structured error can mimic a different exact gauge."""
    wrong = GAUGES[0]
    assert wrong != TRUE
    anchors = anchors_for(design, budget)
    observed = packed_response(anchors, wrong)
    scores = [
        ((packed_response(anchors, g) ^ observed)).bit_count()
        for g in GAUGES
    ]
    choices = [g for g, score in zip(GAUGES, scores) if score == min(scores)]
    return {"different_from_true": wrong != TRUE, "unique_wrong_gauge": choices == [wrong]}


def reference():
    return {
        "evidence": "finite_exact_counterexamples_to_independent_missingness",
        "invariant_hidden_query_positions": [2, 3],
        "results": {
            design: {
                str(budget): {
                    "compatible_gauges_with_queries_2_3_never_seen":
                        len(censored_query_candidates(design, budget)),
                    "consistent_global_relabeling":
                        systematic_relabeling_matches_wrong_gauge(design, budget),
                }
                for budget in BUDGETS
            } for design in DESIGNS
        },
    }


if __name__ == "__main__":
    print(json.dumps(reference(), sort_keys=True, indent=2))

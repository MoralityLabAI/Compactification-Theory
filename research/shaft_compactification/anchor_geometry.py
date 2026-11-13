"""Exact finite anchor design for synthetic behavioral-gauge experiments.

Scores are expected gap values calibrated for a KNOWN synthetic reference
alignment. The method chooses an informative verification panel. It does not
infer the alignment solely from unpaired data.
"""
from itertools import combinations


def select_minimax_triple(expected_margin, correct_index):
    """Lexicographically first three distinct tasks maximizing worst gap."""
    if len(expected_margin) != 16 or not expected_margin or not 0 <= correct_index < len(expected_margin[0]):
        raise ValueError("need 16 task rows and a valid reference gauge")
    ng = len(expected_margin[0])
    if ng < 2 or any(len(row) != ng for row in expected_margin):
        raise ValueError("expected a rectangular matrix with >=2 gauge candidates")
    best = float("-inf")
    best_triple = None
    for triple in combinations(range(16), 3):
        score = min(sum(expected_margin[t][g] for t in triple)
                    for g in range(ng) if g != correct_index)
        if score > best:
            best = score
            best_triple = triple
    return {"selected_triple": list(best_triple),
            "minimum_expected_score_margin": round(float(best), 6),
            "candidate_triples_considered": 560}


def exact_target_bits(goal, perm, flip):
    """Expected destination signature for source truth-table goal."""
    out = [0] * 4
    for source_query, target_query in enumerate(perm):
        out[target_query] = ((goal >> source_query) & 1) ^ flip
    return tuple(out)


def clean_gauge_margins(gauges, correct_index):
    """Expected score-gap matrix if each paired destination response is exact."""
    result = []
    for goal in range(16):
        truth = exact_target_bits(goal, *gauges[correct_index])
        row = []
        for perm, flip in gauges:
            prediction = exact_target_bits(goal, perm, flip)
            row.append(sum(a != b for a, b in zip(truth, prediction)))
        result.append(row)
    return result

"""Exact finite matching of semantic gauges in a four-query binary task family.

The aligner receives only paired black-box behavioral observations. It never
receives hidden goal labels. The paired references are an explicit identifiability
assumption, and the target interface is a permutation plus global output flip.
"""

from itertools import permutations

QUERIES = range(4)
GOALS = range(16)
PERMS = tuple(permutations(QUERIES))
GAUGES = tuple((perm, flip) for perm in PERMS for flip in (0, 1))
TARGET_PERM = (1, 0, 3, 2)
TARGET_FLIP = 1
ANCHORS = (1, 2, 4)


def signature(goal, perm=(0, 1, 2, 3), flip=0):
    return tuple(((goal >> perm[j]) & 1) ^ flip for j in QUERIES)


def infer_gauges(source_rows, target_rows):
    """Perm maps a source-query index to the matching target-query index."""
    if len(source_rows) != len(target_rows):
        raise ValueError("paired observations require equal row counts")
    return [(perm, flip) for perm, flip in GAUGES if all(
        row_a[q] == (row_b[perm[q]] ^ flip)
        for row_a, row_b in zip(source_rows, target_rows) for q in QUERIES
    )]


def candidates_for_goals(goals):
    return infer_gauges(
        [signature(g) for g in goals],
        [signature(g, TARGET_PERM, TARGET_FLIP) for g in goals],
    )


def reference():
    stages = {str(k): len(candidates_for_goals(ANCHORS[:k])) for k in range(4)}
    pair_counts = [len(candidates_for_goals((g, h))) for g in GOALS for h in GOALS]
    full = {signature(g) for g in GOALS}
    invariant = all({signature(g, p, c) for g in GOALS} == full for p, c in GAUGES)
    unique = candidates_for_goals(ANCHORS)
    assert stages == {"0": 48, "1": 6, "2": 2, "3": 1}
    assert min(pair_counts) == 2
    assert invariant and unique == [((1, 0, 3, 2), 1)]
    return {
        "evidence": "exhaustive_finite_gauge_enumeration",
        "goal_count": len(GOALS),
        "gauge_group_size": len(GAUGES),
        "anchors": list(ANCHORS),
        "candidate_count_by_anchor_number": stages,
        "every_ordered_pair_of_two_anchors_ambiguous": min(pair_counts) > 1,
        "two_anchor_pair_count_examined": len(pair_counts),
        "minimum_candidates_with_two_anchors": min(pair_counts),
        "unpaired_policy_collection_identical_under_all_gauges": invariant,
        "test_target_interface_perm": list(TARGET_PERM),
        "test_output_flip": TARGET_FLIP,
        "recovered_target_query_alignment": list(unique[0][0]),
        "recovered_flip": unique[0][1],
    }


if __name__ == "__main__":
    import json
    print(json.dumps(reference(), indent=2, sort_keys=True))

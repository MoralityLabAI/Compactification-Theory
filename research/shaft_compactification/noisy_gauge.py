"""SHAFT-COMP-09: noisy, partially observed behavioral-gauge identifiability.

Everything is an isolated finite synthetic experiment over 16 binary truth-table
policies and the 48-element S4 x C2 gauge family. Reference source signatures
are assumed accurate; only the four target observations per paired anchor may
be independently erased (MCAR) or flipped (BSC). There is no network or device.

Using packed integers allows exact Hamming scoring and a seed-locked Monte
Carlo comparison of one-hot versus balanced anchor designs.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from pathlib import Path

from research.shaft_compactification.gauge_identifiability import (
    GAUGES, TARGET_PERM, TARGET_FLIP,
)

GOALS = tuple(range(16))
TRUE = (TARGET_PERM, TARGET_FLIP)
ONEHOT = (1, 2, 4)
BALANCED = (3, 5, 6, 9, 10, 12)
DESIGNS = {"onehot": ONEHOT, "balanced": BALANCED}
BUDGETS = (3, 6, 12, 24)
NOISE_PCT = (0, 5, 10, 20)
MISSING_PCT = (0, 25, 50)
TRIALS = 1024
SEED = 20261110


def signature_bits(anchor: int, perm: tuple[int, ...], flip: int) -> tuple[int, ...]:
    if anchor not in GOALS or sorted(perm) != [0, 1, 2, 3] or flip not in (0, 1):
        raise ValueError("invalid anchor or gauge")
    return tuple(((anchor >> perm[j]) & 1) ^ flip for j in range(4))


def packed_response(anchors: tuple[int, ...], gauge: tuple) -> int:
    perm, flip = gauge
    return sum(
        bit << (4 * i + j)
        for i, anchor in enumerate(anchors)
        for j, bit in enumerate(signature_bits(anchor, perm, flip))
    )


def anchors_for(design: str, budget: int) -> tuple[int, ...]:
    if budget < 0 or design not in DESIGNS:
        raise ValueError("unknown design or negative budget")
    seq = DESIGNS[design]
    return (seq * ((budget + len(seq) - 1) // len(seq)))[:budget]


def distance_profile(anchors: tuple[int, ...]) -> list[int]:
    true_code = packed_response(anchors, TRUE)
    return [
        (packed_response(anchors, gauge) ^ true_code).bit_count()
        for gauge in GAUGES if gauge != TRUE
    ]


def minimax_three_anchors():
    """Exhaustively optimize minimum clean Hamming distance over 560 triples."""
    best = -1
    optima = []
    for anchors in itertools.combinations(GOALS, 3):
        separation = min(distance_profile(anchors))
        if separation > best:
            best = separation
            optima = [list(anchors)]
        elif separation == best:
            optima.append(list(anchors))
    return {"max_min_distance": best, "optimal_triples": optima,
            "distinct_triplets_checked": math.comb(len(GOALS), 3)}


def chernoff_union_bound(distances: list[int], p: float, missing: float) -> float:
    """Union bound over incorrect gauges, from Bernoulli erasures and flips.

    For d disagreements, Markov's inequality with the optimized exponential
    tilting parameter gives [m + 2(1-m)*sqrt(p(1-p))]^d when 0 <= p < 0.5.
    This is an upper bound on non-unique/incorrect maximum-likelihood choice.
    """
    if not 0 <= p < 0.5 or not 0 <= missing <= 1:
        raise ValueError("require p<1/2 and missing in [0,1]")
    alpha = missing + (1 - missing) * 2 * math.sqrt(p * (1 - p))
    return min(1.0, sum(alpha ** d for d in distances))


def condition(design: str, noise_pct: int, missing_pct: int, trials: int) -> dict:
    if trials <= 0 or noise_pct not in NOISE_PCT or missing_pct not in MISSING_PCT:
        raise ValueError("unsupported parameters")
    noise, missing = noise_pct / 100, missing_pct / 100
    max_anchors = anchors_for(design, max(BUDGETS))
    codes = [packed_response(max_anchors, gauge) for gauge in GAUGES]
    true_index = GAUGES.index(TRUE)
    true_code = codes[true_index]
    masks = {budget: (1 << (4 * budget)) - 1 for budget in BUDGETS}
    distances = {
        budget: [((code ^ true_code) & masks[budget]).bit_count()
                 for i, code in enumerate(codes) if i != true_index]
        for budget in BUDGETS
    }
    counters = {
        budget: dict(success=0, unique_success=0, true_among_minima=0,
                     tied=0, certificate_sufficient=0, observed_total=0,
                     noisy_total=0, posterior_entropy_sum=0.0)
        for budget in BUDGETS
    }
    rng = random.Random(SEED + 100000 * list(DESIGNS).index(design)
                        + 1000 * noise_pct + missing_pct)
    positions = 4 * max(BUDGETS)
    for _ in range(trials):
        observed = 0
        flipped = 0
        for bit_index in range(positions):
            if rng.random() >= missing:
                observed |= 1 << bit_index
                if rng.random() < noise:
                    flipped |= 1 << bit_index
        noisy_code = true_code ^ flipped
        for budget in BUDGETS:
            prefix = masks[budget]
            visible = observed & prefix
            changed = flipped & visible
            scores = [((code ^ noisy_code) & visible).bit_count() for code in codes]
            minimum = min(scores)
            winners = [i for i, score in enumerate(scores) if score == minimum]
            winner = winners[0]
            c = counters[budget]
            c["success"] += int(winner == true_index)
            c["unique_success"] += int(winners == [true_index])
            c["true_among_minima"] += int(true_index in winners)
            c["tied"] += int(len(winners) > 1)
            c["observed_total"] += visible.bit_count()
            c["noisy_total"] += changed.bit_count()
            observed_min_separation = min(
                ((code ^ true_code) & visible).bit_count()
                for i, code in enumerate(codes) if i != true_index
            )
            c["certificate_sufficient"] += int(
                observed_min_separation > 2 * changed.bit_count()
            )
            if noise == 0:
                nmin = len(winners)
                entropy = math.log2(nmin)
            else:
                likelihood_ratio = noise / (1 - noise)
                weights = [likelihood_ratio ** (s - minimum) for s in scores]
                total = sum(weights)
                entropy = math.log2(total) - sum(
                    (w / total) * math.log2(w) for w in weights if w > 0
                )
            c["posterior_entropy_sum"] += entropy

    report = {}
    for budget in BUDGETS:
        c = counters[budget]
        dist = distances[budget]
        report[str(budget)] = {
            "min_clean_pairwise_distance": min(dist),
            "mean_clean_pairwise_distance": round(sum(dist) / len(dist), 4),
            "successes_lexicographic_tiebreak": c["success"],
            "unique_correct_recoveries": c["unique_success"],
            "true_in_minimum_score_set": c["true_among_minima"],
            "ambiguous_minimum_score_trials": c["tied"],
            "sufficient_margin_certificates": c["certificate_sufficient"],
            "observed_bits_total": c["observed_total"],
            "flipped_observed_bits_total": c["noisy_total"],
            "mean_posterior_gauge_entropy_bits": round(
                c["posterior_entropy_sum"] / trials, 6
            ),
            "chernoff_union_bound_probability": round(
                chernoff_union_bound(dist, noise, missing), 8
            ),
        }
    return report


def reference(trials=TRIALS):
    result = {
        "evidence": "exact_finite_geometry_plus_seeded_mcar_bsc_monte_carlo",
        "source_observations": "clean, exact binary reference signatures",
        "target_observations": "independent coordinate erasures and bit flips",
        "tie_break": "first (lexicographic) among equal minimum-Hamming scores",
        "gauge_count": len(GAUGES),
        "true_gauge": {"target_query_permutation": list(TARGET_PERM),
                       "output_flip": TARGET_FLIP},
        "seed": SEED,
        "trials_per_condition": trials,
        "budgets": list(BUDGETS),
        "noise_percent": list(NOISE_PCT),
        "missing_percent": list(MISSING_PCT),
        "three_anchor_minimax": minimax_three_anchors(),
        "designs": {},
    }
    for design, seed_anchors in DESIGNS.items():
        data = {
            "base_anchor_cycle": list(seed_anchors),
            "conditions": {}
        }
        for p in NOISE_PCT:
            for m in MISSING_PCT:
                key = f"p{p:02d}_missing{m:02d}"
                data["conditions"][key] = condition(design, p, m, trials)
        result["designs"][design] = data
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=TRIALS)
    parser.add_argument("--out")
    args = parser.parse_args()
    result = json.dumps(reference(args.trials), sort_keys=True, indent=2) + "\n"
    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(result)
    else:
        print(result, end="")


if __name__ == "__main__":
    main()

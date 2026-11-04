"""QO-08: paired, equal-query-budget evaluation of three QO-07 probe rules.

This is a frozen digital benchmark, NOT a neural experiment or statistical
proof of superiority. Each policy sees the same per-(state, action, draw-index)
pseudorandom stream for a given scenario and seed. Rates are private to the
simulator and post-run scoring; policies receive only the ProbeLedger.

The pilot reports all predefined cases/seeds/checkpoints, including uncertain,
incorrect and uncovered cases. No seed or scenario postselection.
"""
from collections import defaultdict
from fractions import Fraction as Q
from itertools import product
from random import Random

from research.ingression_dynamics_v1.anytime_active_quotient import (
    ProbeLedger, choose_probe, certificate, frozen_error
)
from research.ingression_dynamics_v1.geometric_quotient import optimum_for_k
from research.ingression_dynamics_v1.rate_sampling import partition_radius

POLICIES = ("round_robin", "width", "bottleneck")
SEEDS = (11, 19, 23, 37, 41, 53)
CHECKPOINTS = (24, 64, 128)
ALPHA = Q(1, 20)
BISECTION_BITS = 10

# Declared before simulation; fixed action/state labels and exact probabilities.
SCENARIOS = (
    {
        "id": "two_cluster_feasible",
        "rates": ((Q(1, 10), Q(3, 20)),
                  (Q(1, 5), Q(1, 10)),
                  (Q(4, 5), Q(17, 20)),
                  (Q(9, 10), Q(4, 5))),
        "classes": 2,
        "tolerance": Q(7, 20),
    },
    {
        "id": "separated_infeasible",
        "rates": ((Q(0), Q(0)), (Q(1), Q(1)),
                  (Q(1, 2), Q(1, 2))),
        "classes": 1,
        "tolerance": Q(1, 10),
    },
    {
        "id": "weak_margin_feasible",
        "rates": ((Q(2, 5), Q(1, 2)),
                  (Q(1, 2), Q(11, 20)),
                  (Q(11, 20), Q(3, 5))),
        "classes": 2,
        "tolerance": Q(1, 10),
    },
)


def stream_seed(case_index, seed, state, action):
    """Constant, public, scenario-specific seed independent of the policy."""
    return int(seed)*1000003 + case_index*10000019 + state*9907 + action*7919


def deterministic_cell_prefix(case_index, seed, rates, state, action, count):
    """Independent oracle for the first count outcomes of a declared cell."""
    rng = Random(stream_seed(case_index, seed, state, action))
    p = float(rates[state][action])
    return tuple(int(rng.random() < p) for _ in range(count))


def run_policy(case_index, case, seed, policy, checkpoints=CHECKPOINTS):
    """One continuous probe run; snapshots are taken without reseeding."""
    if policy not in POLICIES:
        raise ValueError("unregistered policy")
    checkpoints = tuple(checkpoints)
    if (not checkpoints or any(type(t) is not int or t < 0 for t in checkpoints)
            or tuple(sorted(set(checkpoints))) != checkpoints):
        raise ValueError("increasing unique nonnegative checkpoints required")
    rates, nclass = case["rates"], case["classes"]
    ledger = ProbeLedger(len(rates), len(rates[0]), ALPHA, BISECTION_BITS)
    streams = {(i, a): Random(stream_seed(case_index, seed, i, a))
               for i, a in product(range(ledger.n), range(ledger.d))}
    score_true_opt, _, _ = optimum_for_k(rates, nclass)
    feasible_truth = score_true_opt <= case["tolerance"]
    rows = []
    for step in range(checkpoints[-1] + 1):
        if step in checkpoints:
            result = certificate(ledger.snapshot(), nclass, case["tolerance"])
            covered = all(lo <= rates[i][a] <= hi
                          for i, row in enumerate(ledger.bounds)
                          for a, (lo, hi) in enumerate(row))
            decision = result["decision"]
            false_cert = ((decision == "certified-feasible" and not feasible_truth)
                          or (decision == "certified-infeasible" and feasible_truth))
            if covered:
                assert not false_cert, "coverage event with false certificate"
            observed_error = frozen_error(
                rates, result["transient_labels"], result["frozen_midpoints"])
            if covered:
                assert observed_error <= result["robust_macro_error_upper"]
            regret = (partition_radius(rates, result["transient_labels"])
                      - score_true_opt)
            assert regret >= 0
            rows.append({
                "scenario": case["id"],
                "seed": seed,
                "policy": policy,
                "queries": step,
                "decision": decision,
                "true_feasible": feasible_truth,
                "false_cert": false_cert,
                "true_rates_covered": covered,
                "true_optimum": str(score_true_opt),
                "robust_upper": str(result["robust_macro_error_upper"]),
                "true_optimum_lower": str(result["true_optimum_lower"]),
                "true_frozen_error": str(observed_error),
                "partition_regret": str(regret),
                "interval_max_width": str(result["maximum_interval_width"]),
                "queried_cells": sum(c > 0 for row in ledger.counts for c in row),
                "count_by_cell": [list(row) for row in ledger.counts],
            })
        if step == checkpoints[-1]:
            break
        i, a = choose_probe(ledger, nclass, policy)
        draw = int(streams[(i, a)].random() < float(rates[i][a]))
        ledger.observe(i, a, draw)
    assert ledger.query_count() == checkpoints[-1]
    receipts = tuple(ledger.receipts)
    return rows, receipts


def verify_common_stream_prefix(case_index, case, seed, receipts):
    """Compare every per-cell observed prefix to an independent replay."""
    per_cell = defaultdict(list)
    for i, a, hit in receipts:
        per_cell[(i, a)].append(hit)
    for (i, a), hits in per_cell.items():
        expected = deterministic_cell_prefix(
            case_index, seed, case["rates"], i, a, len(hits))
        if expected != tuple(hits):
            return False
    return True


def _mean_fraction(values):
    values = tuple(Q(v) for v in values)
    return sum(values, Q(0)) / len(values)


def build_report(scenarios=SCENARIOS, seeds=SEEDS, policies=POLICIES,
                 checkpoints=CHECKPOINTS):
    """Preregistered finite pilot; exact arithmetic for reported means."""
    scenarios, seeds, policies, checkpoints = (
        tuple(scenarios), tuple(seeds), tuple(policies), tuple(checkpoints))
    if len(set(seeds)) != len(seeds):
        raise ValueError("duplicate seeds")
    rows, matched = [], []
    for ci, case in enumerate(scenarios):
        rates = case["rates"]
        for seed in seeds:
            for policy in policies:
                snapshots, receipts = run_policy(ci, case, seed, policy,
                                                 checkpoints=checkpoints)
                if not verify_common_stream_prefix(ci, case, seed, receipts):
                    raise AssertionError("paired stream prefix mismatch")
                rows.extend(snapshots)
                matched.append((case["id"], seed, policy, len(receipts)))
    groups = []
    for case in scenarios:
        for budget in checkpoints:
            for policy in policies:
                sample = [r for r in rows if r["scenario"] == case["id"] and
                          r["queries"] == budget and r["policy"] == policy]
                assert len(sample) == len(seeds)
                groups.append({
                    "scenario": case["id"],
                    "budget": budget,
                    "policy": policy,
                    "n": len(sample),
                    "true_optimum": sample[0]["true_optimum"],
                    "true_feasible": sample[0]["true_feasible"],
                    "certified_feasible": sum(r["decision"] == "certified-feasible" for r in sample),
                    "certified_infeasible": sum(r["decision"] == "certified-infeasible" for r in sample),
                    "undetermined": sum(r["decision"] == "undetermined" for r in sample),
                    "false_certificates": sum(r["false_cert"] for r in sample),
                    "coverage_failures": sum(not r["true_rates_covered"] for r in sample),
                    "mean_robust_upper": str(_mean_fraction(r["robust_upper"] for r in sample)),
                    "mean_true_frozen_error": str(
                        _mean_fraction(r["true_frozen_error"] for r in sample)),
                    "mean_partition_regret": str(
                        _mean_fraction(r["partition_regret"] for r in sample)),
                    "mean_max_interval_width": str(
                        _mean_fraction(r["interval_max_width"] for r in sample)),
                    "mean_queried_cells": str(
                        _mean_fraction(r["queried_cells"] for r in sample)),
                })
    deltas = []
    for case in scenarios:
        for budget in checkpoints:
            by_policy = {
                p: {r["seed"]: r for r in rows if
                    r["scenario"] == case["id"] and r["queries"] == budget
                    and r["policy"] == p}
                for p in policies
            }
            for baseline in ("round_robin", "width"):
                if "bottleneck" not in by_policy or baseline not in by_policy:
                    continue
                deltas.append({
                    "scenario": case["id"],
                    "budget": budget,
                    "baseline": baseline,
                    "comparison": "bottleneck",
                    "paired_mean_true_error_improvement": str(_mean_fraction(
                        Q(by_policy[baseline][s]["true_frozen_error"]) -
                        Q(by_policy["bottleneck"][s]["true_frozen_error"])
                        for s in seeds)),
                    "paired_mean_robust_upper_improvement": str(_mean_fraction(
                        Q(by_policy[baseline][s]["robust_upper"]) -
                        Q(by_policy["bottleneck"][s]["robust_upper"])
                        for s in seeds)),
                    "paired_wins_true_error": sum(
                        Q(by_policy["bottleneck"][s]["true_frozen_error"]) <
                        Q(by_policy[baseline][s]["true_frozen_error"]) for s in seeds),
                    "paired_ties_true_error": sum(
                        Q(by_policy["bottleneck"][s]["true_frozen_error"]) ==
                        Q(by_policy[baseline][s]["true_frozen_error"]) for s in seeds),
                })
    return {
        "contract": "QO-08 paired fixed-budget synthetic Bernoulli stream experiment",
        "scope": "small finite known-addressable-state hit/hold systems; no neural model",
        "alpha": str(ALPHA),
        "bisection_bits": BISECTION_BITS,
        "policies": list(policies),
        "seeds": list(seeds),
        "checkpoints": list(checkpoints),
        "scenarios": [{
            "id": c["id"], "rates": [[str(x) for x in row] for row in c["rates"]],
            "max_transient_classes": c["classes"], "tolerance": str(c["tolerance"])
        } for c in scenarios],
        "trial_rows": rows,
        "groups": groups,
        "paired_deltas": deltas,
        "run_counts": {
            "policy_trajectories": len(matched),
            "snapshots": len(rows),
            "queries_all_trajectories": sum(n for _, _, _, n in matched),
            "matched_stream_prefix_verified": len(matched),
        },
        "interpretation": "No training or transfer evaluated. Paired differences are descriptive, not significance claims."
    }

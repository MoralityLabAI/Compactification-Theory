"""QO-08: preregistered, paired synthetic anytime-probe policy pilot.

Every policy receives the same named state/action Bernoulli streams for each
seed and family, with identical query budgets. The frozen public family and
seeds are *development material*, not an evaluator-private heldout panel.

No neural model is queried, trained or evaluated.
"""
import argparse
import hashlib
import json
from fractions import Fraction as Q
from pathlib import Path

from research.ingression_dynamics_v1.anytime_active_quotient import run_synthetic
from research.ingression_dynamics_v1.geometric_quotient import optimum_for_k

# Frozen experiment before inspecting any run outcome.
SEEDS = tuple(range(101, 113))
BUDGET = 96
ALPHA = Q(1, 20)
TOLERANCE = Q(1, 4)
POLICIES = ("round_robin", "width", "bottleneck")
FAMILIES = (
    ("two_goal_regimes", (
        (Q(0), Q(0)), (Q(0), Q(1, 4)),
        (Q(1), Q(3, 4)), (Q(1), Q(1))
    ), 2),
    ("three_action_nuisance", (
        (Q(0), Q(1, 2), Q(1, 2)),
        (Q(0), Q(1, 2), Q(1, 2)),
        (Q(1), Q(1, 2), Q(1, 2)),
        (Q(1), Q(1, 2), Q(1, 2)),
    ), 2),
)


def _fraction_string(x):
    return str(Q(x))


def _source_hashes():
    root = Path(__file__).resolve().parents[2]
    paths = (
        "research/ingression_dynamics_v1/sampler_policy_pilot.py",
        "research/ingression_dynamics_v1/anytime_active_quotient.py",
        "research/ingression_dynamics_v1/test_sampler_policy_pilot.py",
    )
    return {p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths}


def run_pilot(seeds=SEEDS, budget=BUDGET):
    if type(budget) is not int or budget < 0 or not seeds:
        raise ValueError("nonnegative integer budget and nonempty seeds required")
    seedset = tuple(seeds)
    if len(set(seedset)) != len(seedset) or any(type(x) is not int for x in seedset):
        raise ValueError("distinct integer seeds required")
    records = []
    for family, rates, k in FAMILIES:
        true_best = optimum_for_k(rates, k)[0]
        for seed in seedset:
            for policy in POLICIES:
                run = run_synthetic(
                    rates, budget, k, policy=policy, seed=seed,
                    alpha=ALPHA, tolerance=TOLERANCE, bisection_bits=12,
                    stop_on_certificate=False,
                )
                # True rates are visible only to this AFTER-run evaluation code.
                # A pilot run may fail the 1-alpha coverage event; retain it.
                bad_cert = (
                    run["decision"] == "certified-feasible"
                    and run["true_frozen_error"] > TOLERANCE
                ) or (
                    run["decision"] == "certified-infeasible"
                    and true_best <= TOLERANCE
                )
                if run["covered_true_rates"] and bad_cert:
                    raise AssertionError("a certified decision failed on covered world")
                records.append({
                    "family": family,
                    "seed": seed,
                    "policy": policy,
                    "budget": budget,
                    "observed_queries": run["observed_queries"],
                    "decision": run["decision"],
                    "covered_true_rates": run["covered_true_rates"],
                    "false_certificate_on_truth": bad_cert,
                    "robust_upper": run["robust_macro_error_upper"],
                    "optimistic_lower": run["true_optimum_lower"],
                    "certificate_gap": run["optimality_gap_upper"],
                    "true_frozen_error": run["true_frozen_error"],
                    "true_optimum": true_best,
                    "transient_labels": run["transient_labels"],
                    "query_counts": run["counts"],
                    "receipt_hash": hashlib.sha256(
                        json.dumps(run["receipts"]).encode("ascii")
                    ).hexdigest(),
                })
    summary = []
    for family, _, _ in FAMILIES:
        for policy in POLICIES:
            rows = [x for x in records if x["family"] == family
                    and x["policy"] == policy]
            n = len(rows)
            summary.append({
                "family": family, "policy": policy, "seeds": n,
                "mean_robust_upper": sum((r["robust_upper"] for r in rows),Q(0))/n,
                "mean_certificate_gap": sum((r["certificate_gap"] for r in rows),Q(0))/n,
                "mean_true_frozen_error": sum((r["true_frozen_error"] for r in rows),Q(0))/n,
                "confidence_coverage_count": sum(int(r["covered_true_rates"]) for r in rows),
                "false_certificates": sum(int(r["false_certificate_on_truth"]) for r in rows),
                "certified_feasible": sum(r["decision"] == "certified-feasible" for r in rows),
                "certified_infeasible": sum(r["decision"] == "certified-infeasible" for r in rows),
            })
    comparison = []
    for family, _, _ in FAMILIES:
        baseline = next(row for row in summary if row["family"] == family
                        and row["policy"] == "round_robin")
        for policy in POLICIES[1:]:
            matched = next(row for row in summary if row["family"] == family
                           and row["policy"] == policy)
            differences = []
            for seed in seedset:
                a = next(r for r in records if r["family"] == family
                         and r["seed"] == seed and r["policy"] == policy)
                b = next(r for r in records if r["family"] == family
                         and r["seed"] == seed and r["policy"] == "round_robin")
                differences.append(a["robust_upper"]-b["robust_upper"])
            comparison.append({
                "family": family, "candidate": policy, "reference": "round_robin",
                "mean_upper_delta": matched["mean_robust_upper"]-baseline["mean_robust_upper"],
                "candidate_better_seed_count": sum(v < 0 for v in differences),
                "candidate_equal_seed_count": sum(v == 0 for v in differences),
                "candidate_worse_seed_count": sum(v > 0 for v in differences),
            })
    def serialize(value):
        if isinstance(value, Q): return str(value)
        if isinstance(value, tuple): return [serialize(x) for x in value]
        if isinstance(value, list): return [serialize(x) for x in value]
        if isinstance(value, dict): return {key: serialize(v) for key,v in value.items()}
        return value
    return serialize({
        "protocol": {
            "kind": "public development-only paired digital hit/hold pilot",
            "seeds": seedset, "budget_per_policy": budget,
            "alpha": ALPHA, "threshold": TOLERANCE,
            "policies": POLICIES, "family_names": tuple(f[0] for f in FAMILIES),
            "policy_selection": "frozen fixed heuristic; no neural model",
            "claim_limit": "descriptive paired small seeded panel; no clinical or neural inference",
        },
        "source_sha256": _source_hashes(),
        "summaries": summary,
        "comparisons": comparison,
        "records": records,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",type=Path)
    parser.add_argument("--smoke",action="store_true",
                        help="first 2 development seeds and budget 12")
    args = parser.parse_args()
    report = run_pilot(SEEDS[:2],12) if args.smoke else run_pilot()
    contents = json.dumps(report, indent=2, sort_keys=True)+"\n"
    if args.out is None:
        print(contents, end="")
    else:
        if args.out.exists():
            if args.out.read_text(encoding="utf-8") != contents:
                raise FileExistsError("existing paired receipt disagrees")
        else:
            args.out.parent.mkdir(parents=True,exist_ok=True)
            args.out.write_text(contents,encoding="utf-8")
        print(args.out)


if __name__ == "__main__":
    main()

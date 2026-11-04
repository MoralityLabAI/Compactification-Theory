"""QO-09: fixed fractional-cover probe policy and paired digital pilot.

The allocation solves a SPECIFIED synthetic QI-03 binary-incidence
baseline/alternative problem, independently of the evaluated QO-08 rate
tables. A rational primal and dual certify that fixed design's optimum.

Normalizing its primal weights produces a probe SCHEDULE heuristic, not an
optimal strategy or a valid information lower bound for QO-08's different
hypothesis/test contract. The exact QO-07 e-process is unchanged.

All policies see the same per-state/action Bernoulli stream prefixes, use
the same query budget, confidence level, and robust quotient scorer.
The pilot includes old QO-08 cases as an explicitly retrospective panel
and two newly fixed synthetic transfer cases; neither is neural evidence.
"""

from collections import defaultdict
from fractions import Fraction as Q
from itertools import product
from math import ceil
from random import Random

from research.ingression_dynamics_v1.anytime_active_quotient import (
    ProbeLedger, certificate, frozen_error
)
from research.ingression_dynamics_v1.geometric_quotient import optimum_for_k
from research.ingression_dynamics_v1.paired_probe_pilot import (
    SCENARIOS as ORIGINAL_SCENARIOS, SEEDS, CHECKPOINTS,
    ALPHA, BISECTION_BITS, POLICIES as EXISTING_POLICIES,
    run_policy as original_run_policy,
    stream_seed, verify_common_stream_prefix
)
from research.ingression_dynamics_v1.rate_sampling import partition_radius

POLICIES = EXISTING_POLICIES + ("fractional_cover", "fractional_cover_rotated")
ALTERNATIVE_BASE_RATE = Q(1, 2)
ALTERNATIVE_CHANGED_RATE = Q(5, 8)
ALTERNATIVE_ERROR_LEVEL = Q(1, 20)
TRAINING_FAMILY = "QI03 disjoint 6-cell blocks: a 3-cycle of pair masks plus singleton masks"

# Both additional synthetic cases are specified before the first complete
# five-policy comparison. Neither is used to construct LP masks or weights.
FRESH_SCENARIOS = (
    {
        "id": "crossed_clusters_feasible",
        "rates": ((Q(1, 10), Q(4, 5)),
                  (Q(3, 20), Q(9, 10)),
                  (Q(4, 5), Q(1, 10)),
                  (Q(9, 10), Q(1, 5))),
        "classes": 2,
        "tolerance": Q(7, 20),
    },
    {
        "id": "crossed_infeasible",
        "rates": ((Q(1, 10), Q(9, 10)),
                  (Q(1, 2), Q(1, 2)),
                  (Q(9, 10), Q(1, 10))),
        "classes": 1,
        "tolerance": Q(1, 10),
    },
)
SCENARIOS = ORIGINAL_SCENARIOS + FRESH_SCENARIOS


def _masks_primal_dual(cell_count):
    """Closed-form QI-03 fractional-cover optimum for disjoint blocks.

    In each block of at most 6 cells, three pair alternatives on the
    first three cells have primal and dual weights 1/2. The remaining
    cells are covered by singleton alternatives, with weights 1.

    A tail of fewer than three cells uses singleton alternatives only.
    Exact primal/dual objective equality certifies global optimality.
    """
    if type(cell_count) is not int or cell_count < 1:
        raise ValueError("positive number of state/action cells required")
    weights = [Q(0)] * cell_count
    masks = []
    dual = []
    for offset in range(0, cell_count, 6):
        length = min(6, cell_count - offset)
        if length >= 3:
            pair_indices = ((0, 1), (1, 2), (0, 2))
            for i, j in pair_indices:
                masks.append((1 << (offset + i)) | (1 << (offset + j)))
                dual.append(Q(1, 2))
            for i in range(3):
                weights[offset + i] = Q(1, 2)
            next_single = 3
        else:
            next_single = 0
        for i in range(next_single, length):
            masks.append(1 << (offset + i))
            dual.append(Q(1))
            weights[offset + i] = Q(1)
    return tuple(masks), tuple(weights), tuple(dual)


def verify_fractional_design(masks, weights, dual):
    """Pure rational feasibility and matching-objective certificate.

    A weak-duality proof for every finite rational incidence matrix
    is kernel-checked in OntologicalProofs/MOR-08.
    """
    masks, weights, dual = tuple(masks), tuple(Q(x) for x in weights), tuple(Q(x) for x in dual)
    n = len(weights)
    if not n or len(masks) != len(dual) or any(
            type(mask) is not int or mask <= 0 or mask >= (1 << n)
            for mask in masks):
        raise ValueError("invalid binary-incidence design")
    if any(x < 0 for x in weights) or any(y < 0 for y in dual):
        raise ValueError("negative primal or dual coordinate")
    if any(sum((weights[i] for i in range(n) if mask & (1 << i)), Q(0)) < 1
           for mask in masks):
        raise ValueError("primal fails to cover an alternative")
    if any(sum((dual[j] for j, mask in enumerate(masks)
                if mask & (1 << i)), Q(0)) > 1 for i in range(n)):
        raise ValueError("dual exceeds unit query cost")
    objective = sum(weights, Q(0))
    if objective != sum(dual, Q(0)):
        raise ValueError("primal and dual objectives differ")
    return objective


def fractional_plan(state_count, action_count, rotated=False):
    """Construct a public, pre-outcome LP plan on named state/action cells.

    Rotated null permutes both the fixed alternative family and its
    optimal weights, preserving the exact LP objective and query budget.
    """
    if (type(state_count) is not int or state_count < 1 or
            type(action_count) is not int or action_count < 1):
        raise ValueError("positive state/action counts required")
    cell_count = state_count * action_count
    masks, primal, dual = _masks_primal_dual(cell_count)
    if rotated:
        shift = max(1, cell_count // 2)
        permutation = tuple((i + shift) % cell_count for i in range(cell_count))
        out = [Q(0)] * cell_count
        for i, x in enumerate(primal):
            out[permutation[i]] = x
        primal = tuple(out)
        masks = tuple(sum(1 << permutation[i] for i in range(cell_count)
                          if mask & (1 << i)) for mask in masks)
    objective = verify_fractional_design(masks, primal, dual)
    return {
        "cell_count": cell_count,
        "state_count": state_count,
        "action_count": action_count,
        "alternative_masks": masks,
        "primal_weights": primal,
        "dual_weights": dual,
        "exact_lp_objective": objective,
        "normalized_shares": tuple(x / objective for x in primal),
        "rotated_control": bool(rotated),
    }


def choose_fractional_cell(counts, plan):
    """Predictable deterministic weighted-fair allocation.

    At every step, query the cell with the smallest observed count
    divided by its QI-03 optimal fractional weight. Positive weights
    ensure all cells are eventually observed. Cell ID breaks ties.
    """
    n, d = plan["state_count"], plan["action_count"]
    if len(counts) != n or any(len(row) != d for row in counts):
        raise ValueError("count matrix does not match declared plan")
    weights = plan["primal_weights"]
    if any(type(x) is not int or x < 0 for row in counts for x in row):
        raise ValueError("counts must be nonnegative integers")
    index = min(range(n * d), key=lambda c: (
        Q(counts[c // d][c % d]) / weights[c], c))
    return divmod(index, d)


def run_fractional_policy(case_index, case, seed, policy, checkpoints=CHECKPOINTS):
    """One complete frozen policy trajectory, with QO-08-compatible rows."""
    if policy not in ("fractional_cover", "fractional_cover_rotated"):
        raise ValueError("invalid fractional-cover policy name")
    checkpoints = tuple(checkpoints)
    if (not checkpoints or any(type(x) is not int or x < 0 for x in checkpoints)
            or tuple(sorted(set(checkpoints))) != checkpoints):
        raise ValueError("unique sorted nonnegative budget checkpoints required")
    rates, max_classes = case["rates"], case["classes"]
    n, d = len(rates), len(rates[0])
    plan = fractional_plan(n, d, rotated=(policy == "fractional_cover_rotated"))
    ledger = ProbeLedger(n, d, ALPHA, BISECTION_BITS)
    streams = {(i, a): Random(stream_seed(case_index, seed, i, a))
               for i, a in product(range(n), range(d))}
    true_optimum, _, _ = optimum_for_k(rates, max_classes)
    true_feasible = true_optimum <= case["tolerance"]
    rows = []
    for step in range(checkpoints[-1] + 1):
        if step in checkpoints:
            found = certificate(ledger.snapshot(), max_classes, case["tolerance"])
            covers_truth = all(lo <= rates[i][a] <= hi
                               for i, row in enumerate(ledger.bounds)
                               for a, (lo, hi) in enumerate(row))
            decision = found["decision"]
            wrong = (decision == "certified-feasible" and not true_feasible or
                     decision == "certified-infeasible" and true_feasible)
            if covers_truth and wrong:
                raise AssertionError("incorrect certificate on the coverage event")
            error = frozen_error(
                rates, found["transient_labels"], found["frozen_midpoints"])
            if covers_truth and error > found["robust_macro_error_upper"]:
                raise AssertionError("frozen macro-model exceeds certified error")
            regret = partition_radius(rates, found["transient_labels"]) - true_optimum
            if regret < 0:
                raise AssertionError("negative regret under true optimum")
            rows.append({
                "scenario": case["id"], "seed": seed, "policy": policy,
                "queries": step,
                "decision": decision,
                "true_feasible": true_feasible,
                "false_cert": bool(wrong),
                "true_rates_covered": covers_truth,
                "true_optimum": str(true_optimum),
                "robust_upper": str(found["robust_macro_error_upper"]),
                "true_optimum_lower": str(found["true_optimum_lower"]),
                "true_frozen_error": str(error),
                "partition_regret": str(regret),
                "interval_max_width": str(found["maximum_interval_width"]),
                "queried_cells": sum(x > 0 for row in ledger.counts for x in row),
                "count_by_cell": [list(row) for row in ledger.counts],
                "hits_by_cell": [list(row) for row in ledger.hits],
            })
        if step == checkpoints[-1]:
            break
        i, a = choose_fractional_cell(ledger.counts, plan)
        result = int(streams[(i, a)].random() < float(rates[i][a]))
        ledger.observe(i, a, result)
    assert ledger.query_count() == checkpoints[-1]
    return rows, tuple(ledger.receipts)


def _mean(values):
    values = tuple(Q(x) for x in values)
    return sum(values, Q(0)) / len(values)


def build_pilot_report(scenarios=SCENARIOS, seeds=SEEDS,
                       policies=POLICIES, checkpoints=CHECKPOINTS):
    """Full paired retrospective-plus-exploratory comparison.

    Produces every predeclared trial row. Original QO-08 baselines are
    invoked unchanged. New policies use only their fixed LP plan.
    """
    scenarios, seeds = tuple(scenarios), tuple(seeds)
    policies, checkpoints = tuple(policies), tuple(checkpoints)
    if (not scenarios or not seeds or len(set(seeds)) != len(seeds) or
            not checkpoints or tuple(sorted(set(checkpoints))) != checkpoints or
            any(p not in POLICIES for p in policies) or len(set(policies)) != len(policies)):
        raise ValueError("invalid fixed pilot protocol")
    rows = []
    trajectories = []
    designs = {}
    for ci, case in enumerate(scenarios):
        n,d = len(case["rates"]), len(case["rates"][0])
        designs[case["id"]] = {
            name: fractional_plan(n, d, name == "fractional_cover_rotated")
            for name in ("fractional_cover", "fractional_cover_rotated")
        }
        for seed in seeds:
            for policy in policies:
                if policy in EXISTING_POLICIES:
                    checkpoints_rows, receipts = original_run_policy(
                        ci, case, seed, policy, checkpoints)
                else:
                    checkpoints_rows, receipts = run_fractional_policy(
                        ci, case, seed, policy, checkpoints)
                if not verify_common_stream_prefix(ci, case, seed, receipts):
                    raise AssertionError("shared per-cell Bernoulli stream replay failed")
                if len(receipts) != checkpoints[-1]:
                    raise AssertionError("query budget discrepancy")
                rows.extend(checkpoints_rows)
                trajectories.append((case["id"], seed, policy, len(receipts)))
    groups=[]
    for case in scenarios:
        for budget in checkpoints:
            for policy in policies:
                sample=[r for r in rows if r["scenario"]==case["id"] and
                        r["queries"]==budget and r["policy"]==policy]
                assert len(sample)==len(seeds)
                groups.append({
                    "scenario":case["id"],"budget":budget,"policy":policy,
                    "n":len(sample),
                    "true_feasible":sample[0]["true_feasible"],
                    "true_optimum":sample[0]["true_optimum"],
                    "certified_feasible":sum(r["decision"]=="certified-feasible" for r in sample),
                    "certified_infeasible":sum(r["decision"]=="certified-infeasible" for r in sample),
                    "undetermined":sum(r["decision"]=="undetermined" for r in sample),
                    "false_certificates":sum(r["false_cert"] for r in sample),
                    "coverage_failures":sum(not r["true_rates_covered"] for r in sample),
                    "mean_robust_upper":str(_mean(r["robust_upper"] for r in sample)),
                    "mean_true_frozen_error":str(_mean(r["true_frozen_error"] for r in sample)),
                    "mean_partition_regret":str(_mean(r["partition_regret"] for r in sample)),
                    "mean_queried_cells":str(_mean(r["queried_cells"] for r in sample)),
                })
    paired=[]
    for case in scenarios:
        for budget in checkpoints:
            indexed={p:{r["seed"]:r for r in rows if r["scenario"]==case["id"]
                        and r["queries"]==budget and r["policy"]==p}
                     for p in policies}
            if "fractional_cover" not in indexed:
                continue
            for baseline in policies:
                if baseline=="fractional_cover":
                    continue
                paired.append({
                    "scenario":case["id"],"budget":budget,
                    "candidate":"fractional_cover","baseline":baseline,
                    "paired_mean_true_error_improvement":str(_mean(
                        Q(indexed[baseline][seed]["true_frozen_error"]) -
                        Q(indexed["fractional_cover"][seed]["true_frozen_error"])
                        for seed in seeds)),
                    "paired_mean_robust_bound_improvement":str(_mean(
                        Q(indexed[baseline][seed]["robust_upper"]) -
                        Q(indexed["fractional_cover"][seed]["robust_upper"])
                        for seed in seeds)),
                    "paired_true_error_wins":sum(
                        Q(indexed["fractional_cover"][seed]["true_frozen_error"]) <
                        Q(indexed[baseline][seed]["true_frozen_error"]) for seed in seeds),
                    "paired_true_error_ties":sum(
                        Q(indexed["fractional_cover"][seed]["true_frozen_error"]) ==
                        Q(indexed[baseline][seed]["true_frozen_error"]) for seed in seeds),
                })
    return {
        "contract":"QO-09 fixed QI-03-derived allocation against unchanged QO-08 baselines",
        "scope":"public QO-08 retrospective panel plus two fixed synthetic extension cases",
        "training_family":TRAINING_FAMILY,
        "training_baseline_rate":str(ALTERNATIVE_BASE_RATE),
        "training_alternative_rate":str(ALTERNATIVE_CHANGED_RATE),
        "training_error_level":str(ALTERNATIVE_ERROR_LEVEL),
        "alpha":str(ALPHA), "bisection_bits":BISECTION_BITS,
        "seeds":list(seeds), "checkpoints":list(checkpoints),
        "policies":list(policies),
        "scenarios":[{"id":c["id"],"max_transient_classes":c["classes"],
                     "tolerance":str(c["tolerance"]),
                     "rates":[[str(p) for p in row] for row in c["rates"]],
                     "panel":"QO-08 retrospective" if i<len(ORIGINAL_SCENARIOS)
                     else "QO-09 fixed exploratory extension"}
                    for i,c in enumerate(scenarios)],
        "certified_fractional_designs":{
            key:{name:{
                "primal":[str(x) for x in p["primal_weights"]],
                "dual":[str(y) for y in p["dual_weights"]],
                "alternative_masks":list(p["alternative_masks"]),
                "objective":str(p["exact_lp_objective"]),
                "normalized_shares":[str(z) for z in p["normalized_shares"]],
            }for name,p in value.items()}for key,value in designs.items()},
        "trial_rows":rows,
        "groups":groups,
        "paired_comparisons":paired,
        "run_counts":{
            "trajectories":len(trajectories),
            "snapshots":len(rows),
            "queries":sum(x[-1] for x in trajectories),
            "paired_stream_verifications":len(trajectories),
        },
        "limitations":"Static alternative-family information lower bound is unrelated to true QO-08 class decision. Allocation weights are a heuristic; no neural transfer or proven policy optimality.",
    }

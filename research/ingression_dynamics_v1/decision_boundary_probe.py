"""QO-10: prospective synthetic test of QI-04's decision-boundary allocation.

The static fractional plan is constructed using a fixed *public*
baseline-versus-alternatives model family from EvalMath QI-04.
Each alternative is independently certified to cross the actual
goal-visible one-step quotient feasibility boundary (k=1,tau=1/5).

The plan never reads evaluation rate tables or held-out Bernoulli bits.
Its fractional information optimality is conditional on its *training*
hypothesis family, not on all feasible unknown environments.
QO-07's anytime e-process remains the sole source of certificate validity.

Comparison uses unchanged QO-08/QO-09 simulators for old baselines,
new unused seeds, fixed synthetic cases, and common per-cell streams.
All results, including negative outcomes and coverage failures, are kept.
"""
from fractions import Fraction as Q
from itertools import combinations, product
from hashlib import sha256
from random import Random
import json

from research.ingression_dynamics_v1.anytime_active_quotient import (
    ProbeLedger, certificate, frozen_error
)
from research.ingression_dynamics_v1.geometric_quotient import optimum_for_k
from research.ingression_dynamics_v1.rate_sampling import partition_radius
from research.ingression_dynamics_v1.paired_probe_pilot import (
    POLICIES as BASE_POLICIES, run_policy as baseline_run_policy,
    stream_seed, verify_common_stream_prefix
)
from research.ingression_dynamics_v1.fractional_cover_probe import (
    run_fractional_policy as unrelated_fractional_run,
    verify_fractional_design, choose_fractional_cell
)

TRAIN_BASE = Q(1,10)
TRAIN_CHANGED = Q(9,10)
CLASS_BUDGET = 1
TOLERANCE = Q(1,5)
ALPHA = Q(1,20)
BISECTION_BITS = 10

POLICIES = BASE_POLICIES + (
    "fractional_cover", "decision_boundary", "decision_boundary_rotated")
SEEDS = (101, 103, 107, 109, 113, 127)
CHECKPOINTS = (32, 96, 192)

# Frozen before the first completed six-policy experiment. All states
# are addressable; one separately observable absorbing goal is implicit.
SCENARIOS = (
    {
        "id":"feasible_uniform_shift",
        "rates":((Q(12,100),Q(11,100)),
                 (Q(9,100),Q(10,100)),
                 (Q(13,100),Q(9,100))),
        "classes":1,"tolerance":TOLERANCE,
        "support_type":"feasible near training baseline",
    },
    {
        "id":"action0_singleton",
        "rates":((Q(88,100),Q(11,100)),
                 (Q(11,100),Q(10,100)),
                 (Q(9,100),Q(12,100))),
        "classes":1,"tolerance":TOLERANCE,
        "support_type":"near structured training singleton",
    },
    {
        "id":"action1_pair",
        "rates":((Q(11,100),Q(87,100)),
                 (Q(10,100),Q(12,100)),
                 (Q(12,100),Q(90,100))),
        "classes":1,"tolerance":TOLERANCE,
        "support_type":"near structured training action1 pair",
    },
    {
        "id":"action1_singleton_out_of_family",
        "rates":((Q(10,100),Q(11,100)),
                 (Q(12,100),Q(90,100)),
                 (Q(13,100),Q(9,100))),
        "classes":1,"tolerance":TOLERANCE,
        "support_type":"out-of-family action1 singleton negative control",
    },
    {
        "id":"feasible_graded_rates",
        "rates":((Q(12,100),Q(10,100)),
                 (Q(30,100),Q(12,100)),
                 (Q(38,100),Q(10,100))),
        "classes":1,"tolerance":TOLERANCE,
        "support_type":"feasible graded-rate near boundary",
    }
)


def declared_alternative_masks():
    """Three action-zero singletons plus all three action-one pairs."""
    singles=tuple(1<<(2*s) for s in range(3))
    pairs=tuple((1<<(2*i+1)) | (1<<(2*j+1))
                for i,j in combinations(range(3),2))
    return singles+pairs


def make_training_world(mask):
    """Pure fixture constructor; used only to audit the public prior."""
    if type(mask) is not int or not 0<=mask<64:
        raise ValueError("six-bit training change mask required")
    return tuple(tuple(TRAIN_CHANGED if mask & (1<<(2*i+a))
                       else TRAIN_BASE for a in range(2))
                 for i in range(3))


def decision_boundary_plan(rotated=False):
    """Closed-form QI-04 primal/dual certificate, independent of test rates.

    The *LP value* is exact for the structured counterfactual family.
    Scheduling the primal as query shares is an unproven policy heuristic.
    """
    masks=declared_alternative_masks()
    weights=(Q(1),Q(1,2),Q(1),Q(1,2),Q(1),Q(1,2))
    dual=(Q(1),Q(1),Q(1),Q(1,2),Q(1,2),Q(1,2))
    if rotated:
        n=6
        permutation=tuple((i+3)%n for i in range(n))
        shifted=[Q(0)]*n
        for i,x in enumerate(weights):
            shifted[permutation[i]]=x
        weights=tuple(shifted)
        masks=tuple(sum(1<<permutation[i] for i in range(n)
                        if mask & (1<<i)) for mask in masks)
    baseline_optimum=optimum_for_k(make_training_world(0),CLASS_BUDGET)[0]
    if baseline_optimum>TOLERANCE:
        raise AssertionError("training baseline must be feasible")
    alternatives=[]
    for mask in masks:
        value=optimum_for_k(make_training_world(mask),CLASS_BUDGET)[0]
        if value<=TOLERANCE:
            raise AssertionError("decision-aware candidate failed to cross the goal quotient boundary")
        alternatives.append((mask,str(value)))
    optimum=verify_fractional_design(masks,weights,dual)
    if optimum!=Q(9,2):
        raise AssertionError("QI-04 exact rational LP objective changed")
    return {
        "state_count":3,"action_count":2,"cell_count":6,
        "alternative_masks":masks,"primal_weights":weights,
        "dual_weights":dual,"exact_lp_objective":optimum,
        "normalized_shares":tuple(x/optimum for x in weights),
        "baseline_error":baseline_optimum,
        "alternative_errors":tuple(alternatives),
        "rotated_control":bool(rotated),
    }


def run_decision_policy(case_index, case, seed, policy,
                        checkpoints=CHECKPOINTS):
    if policy not in ("decision_boundary","decision_boundary_rotated"):
        raise ValueError("unknown decision-boundary policy")
    checkpoints=tuple(checkpoints)
    if (not checkpoints or tuple(sorted(set(checkpoints)))!=checkpoints
            or any(type(t) is not int or t<0 for t in checkpoints)):
        raise ValueError("strictly increasing nonnegative query checkpoints required")
    rates,classes=case["rates"],case["classes"]
    if len(rates)!=3 or any(len(row)!=2 for row in rates) or classes!=1:
        raise ValueError("the QI-04 transfer test fixes a 3-state, 2-action, one-class interface")
    plan=decision_boundary_plan(policy=="decision_boundary_rotated")
    ledger=ProbeLedger(3,2,ALPHA,BISECTION_BITS)
    rngs={(i,a):Random(stream_seed(case_index,seed,i,a))
          for i,a in product(range(3),range(2))}
    true_best,_,_=optimum_for_k(rates,classes)
    is_feasible=true_best<=case["tolerance"]
    rows=[]
    for t in range(checkpoints[-1]+1):
        if t in checkpoints:
            score=certificate(ledger.snapshot(),classes,case["tolerance"])
            covered=all(lo<=rates[i][a]<=hi for i,row in enumerate(ledger.bounds)
                        for a,(lo,hi) in enumerate(row))
            decision=score["decision"]
            incorrect=((decision=="certified-feasible" and not is_feasible)
                       or (decision=="certified-infeasible" and is_feasible))
            if covered and incorrect:
                raise AssertionError("false certificate despite valid confidence interval event")
            error=frozen_error(rates,score["transient_labels"],
                               score["frozen_midpoints"])
            if covered and error>score["robust_macro_error_upper"]:
                raise AssertionError("certificate underestimates frozen one-step error")
            regret=partition_radius(rates,score["transient_labels"])-true_best
            if regret<0:
                raise AssertionError("negative true-optimum regret")
            rows.append({
                "scenario":case["id"],"seed":seed,"policy":policy,"queries":t,
                "decision":decision,"true_feasible":is_feasible,
                "false_cert":bool(incorrect),"true_rates_covered":covered,
                "true_optimum":str(true_best),
                "robust_upper":str(score["robust_macro_error_upper"]),
                "true_optimum_lower":str(score["true_optimum_lower"]),
                "true_frozen_error":str(error),"partition_regret":str(regret),
                "interval_max_width":str(score["maximum_interval_width"]),
                "queried_cells":sum(x>0 for row in ledger.counts for x in row),
                "count_by_cell":[list(r) for r in ledger.counts],
                "hits_by_cell":[list(r) for r in ledger.hits],
            })
        if t==checkpoints[-1]:
            break
        i,a=choose_fractional_cell(ledger.counts,plan)
        hit=int(rngs[i,a].random()<float(rates[i][a]))
        ledger.observe(i,a,hit)
    if ledger.query_count()!=checkpoints[-1]:
        raise AssertionError("budget not consumed")
    return rows,tuple(ledger.receipts)


def _mean(values):
    values=tuple(Q(v) for v in values)
    return sum(values,Q(0))/len(values)


def full_report(scenarios=SCENARIOS,seeds=SEEDS,
                policies=POLICIES,checkpoints=CHECKPOINTS):
    """Predeclared paired pilot, all outcomes retained, no outcome-based tuning."""
    scenarios,seeds,policies,checkpoints=map(
        tuple,(scenarios,seeds,policies,checkpoints))
    if (not scenarios or not seeds or len(set(seeds))!=len(seeds)
            or not policies or len(set(policies))!=len(policies)
            or any(p not in POLICIES for p in policies)
            or not checkpoints or tuple(sorted(set(checkpoints)))!=checkpoints):
        raise ValueError("invalid pilot protocol")
    plans={name:decision_boundary_plan(name=="decision_boundary_rotated")
           for name in ("decision_boundary","decision_boundary_rotated")}
    all_rows=[]
    hashes=[]
    for case_index,case in enumerate(scenarios):
        for seed in seeds:
            for policy in policies:
                if policy in BASE_POLICIES:
                    records,receipts=baseline_run_policy(case_index,case,seed,
                                                         policy,checkpoints)
                elif policy=="fractional_cover":
                    records,receipts=unrelated_fractional_run(
                        case_index,case,seed,policy,checkpoints)
                else:
                    records,receipts=run_decision_policy(
                        case_index,case,seed,policy,checkpoints)
                if len(receipts)!=checkpoints[-1]:
                    raise AssertionError("mismatched query budget")
                if not verify_common_stream_prefix(case_index,case,seed,receipts):
                    raise AssertionError("per-cell paired prefix mismatch")
                all_rows.extend(records)
                hashes.append({
                    "case":case["id"],"seed":seed,"policy":policy,
                    "queries":len(receipts),
                    "receipt_sha256":sha256(
                        json.dumps(receipts,separators=(",",":")).encode("ascii")
                    ).hexdigest(),
                })
    groups=[]
    deltas=[]
    for case in scenarios:
        for budget in checkpoints:
            by_policy={
                policy:{r["seed"]:r for r in all_rows
                        if r["scenario"]==case["id"]
                        and r["queries"]==budget and r["policy"]==policy}
                for policy in policies
            }
            for policy in policies:
                subset=list(by_policy[policy].values())
                if len(subset)!=len(seeds):
                    raise AssertionError("incomplete paired seed group")
                groups.append({
                    "scenario":case["id"],"budget":budget,"policy":policy,
                    "n":len(subset),
                    "true_optimum":subset[0]["true_optimum"],
                    "true_feasible":subset[0]["true_feasible"],
                    "certified_feasible":sum(r["decision"]=="certified-feasible"
                                             for r in subset),
                    "certified_infeasible":sum(r["decision"]=="certified-infeasible"
                                               for r in subset),
                    "undetermined":sum(r["decision"]=="undetermined" for r in subset),
                    "false_certificates":sum(r["false_cert"] for r in subset),
                    "coverage_failures":sum(not r["true_rates_covered"]
                                            for r in subset),
                    "mean_true_error":str(_mean(r["true_frozen_error"] for r in subset)),
                    "mean_upper_bound":str(_mean(r["robust_upper"] for r in subset)),
                    "mean_regret":str(_mean(r["partition_regret"] for r in subset)),
                    "mean_cells_queried":str(_mean(r["queried_cells"] for r in subset)),
                })
            if "decision_boundary" in by_policy:
                for baseline in policies:
                    if baseline=="decision_boundary":
                        continue
                    deltas.append({
                        "scenario":case["id"],"budget":budget,
                        "candidate":"decision_boundary","baseline":baseline,
                        "mean_true_error_improvement":str(_mean(
                            Q(by_policy[baseline][seed]["true_frozen_error"])-
                            Q(by_policy["decision_boundary"][seed]["true_frozen_error"])
                            for seed in seeds)),
                        "mean_robust_upper_improvement":str(_mean(
                            Q(by_policy[baseline][seed]["robust_upper"])-
                            Q(by_policy["decision_boundary"][seed]["robust_upper"])
                            for seed in seeds)),
                        "paired_true_error_wins":sum(
                            Q(by_policy["decision_boundary"][seed]["true_frozen_error"])<
                            Q(by_policy[baseline][seed]["true_frozen_error"])
                            for seed in seeds),
                        "paired_true_error_ties":sum(
                            Q(by_policy["decision_boundary"][seed]["true_frozen_error"])==
                            Q(by_policy[baseline][seed]["true_frozen_error"])
                            for seed in seeds),
                    })
    return {
        "contract":"QO-10 prospective fixed synthetic decision-aware query allocation",
        "scope":"QO-07 Bernoulli hit/hold goal-visible k=1 test; no neural or biological data",
        "training_specification":{
            "states":3,"actions":2,"classes":1,
            "tolerance":str(TOLERANCE),"baseline_rate":str(TRAIN_BASE),
            "changed_rate":str(TRAIN_CHANGED),
            "alternative_family":"three action0 singletons and three action1 pairs",
            "alternative_masks":list(declared_alternative_masks()),
        },
        "designs":{name:{
            "primal_weights":[str(v) for v in plan["primal_weights"]],
            "dual_weights":[str(v) for v in plan["dual_weights"]],
            "exact_objective":str(plan["exact_lp_objective"]),
            "masks":list(plan["alternative_masks"]),
        }for name,plan in plans.items()},
        "seeds":list(seeds),"checkpoints":list(checkpoints),
        "policies":list(policies),"alpha":str(ALPHA),
        "bisection_bits":BISECTION_BITS,
        "scenarios":[{
            "id":case["id"],"support_type":case["support_type"],
            "rates":[[str(p) for p in row] for row in case["rates"]],
            "classes":case["classes"],
            "tolerance":str(case["tolerance"]),
        }for case in scenarios],
        "trial_rows":all_rows,"trajectory_receipt_hashes":hashes,
        "groups":groups,"paired_comparisons":deltas,
        "run_counts":{
            "trajectories":len(hashes),"snapshots":len(all_rows),
            "queries":sum(h["queries"] for h in hashes),
            "stream_replays":len(hashes),
        },
        "limitations":"static decision-aware prior, synthetic public casebook, no new model training; LP lower bound does not certify transfer optimality.",
    }

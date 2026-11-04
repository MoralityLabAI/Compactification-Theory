"""QO-11 paired gap acquisition pilot with fixed source-described cases.

Policies:
  unchanged QO-08 round_robin, width and bottleneck,
  new expected_gap_bottleneck / expected_gap_width.

All streams use the original per-cell QO-08 seed equation; public
registered cases, seeds and budget checkpoints are fixed in source.
The two new policies use only the observed ProbeLedger to plan.

This pilot is synthetic and small; prediction quality, certification
coverage and deterministic planning-work counts are preserved verbatim.
"""
from collections import defaultdict
from fractions import Fraction as Q
from hashlib import sha256
from itertools import product
from random import Random
import json

from research.ingression_dynamics_v1.anytime_active_quotient import (
    ProbeLedger, certificate, frozen_error
)
from research.ingression_dynamics_v1.expected_gap_probe import (
    gap_bounds, choose_expected_gap_probe
)
from research.ingression_dynamics_v1.geometric_quotient import optimum_for_k
from research.ingression_dynamics_v1.paired_probe_pilot import (
    SCENARIOS as QO08_SCENARIOS,
    run_policy as qo08_run_policy, stream_seed,
    verify_common_stream_prefix
)
from research.ingression_dynamics_v1.decision_boundary_probe import (
    SCENARIOS as QO10_SCENARIOS
)
from research.ingression_dynamics_v1.rate_sampling import partition_radius

POLICIES=("round_robin","width","bottleneck",
          "expected_gap_bottleneck","expected_gap_width")
SEEDS=(401,409,419,421)
CHECKPOINTS=(32,96,192)
ALPHA=Q(1,20)
BISECTION_BITS=10

# Existing synthetic cases are explicitly retrospective and have been
# investigated in QO-10. The last two are new fixed cases, committed
# before the first complete QO-11 report. No private model holdout.
SCENARIOS=(
    {
        **QO10_SCENARIOS[1],
        "panel":"retrospective QO-10 action0 singleton",
    },
    {
        **QO10_SCENARIOS[2],
        "panel":"retrospective QO-10 action1 pair",
    },
    {
        **QO10_SCENARIOS[4],
        "panel":"retrospective QO-10 feasible graded",
    },
    {
        "id":"new_feasible_close",
        "rates":((Q(19,100),Q(21,100)),
                 (Q(25,100),Q(22,100)),
                 (Q(28,100),Q(27,100))),
        "classes":1,
        "tolerance":Q(1,5),
        "panel":"QO-11 new source-frozen synthetic",
    },
    {
        "id":"new_infeasible_action_interaction",
        "rates":((Q(18,100),Q(80,100)),
                 (Q(20,100),Q(18,100)),
                 (Q(78,100),Q(20,100))),
        "classes":1,
        "tolerance":Q(1,5),
        "panel":"QO-11 new source-frozen synthetic",
    },
)


def _valid_checkpoints(checkpoints):
    c=tuple(checkpoints)
    if (not c or any(type(t) is not int or t<0 for t in c)
            or tuple(sorted(set(c)))!=c):
        raise ValueError("unique increasing nonnegative budgets required")
    return c


def gap_policy_runner(case_index,case,seed,policy,checkpoints=CHECKPOINTS):
    """Exact paired Bernoulli run with separate hypothesis-scoring receipt."""
    if policy not in ("expected_gap_bottleneck","expected_gap_width"):
        raise ValueError("unknown expected-gap policy")
    checkpoints=_valid_checkpoints(checkpoints)
    rates=case["rates"]
    n,d=len(rates),len(rates[0])
    k=case["classes"]
    ledger=ProbeLedger(n,d,ALPHA,BISECTION_BITS)
    rng={(i,a):Random(stream_seed(case_index,seed,i,a))
         for i,a in product(range(n),range(d))}
    optimal,_,_=optimum_for_k(rates,k)
    feasible=optimal<=case["tolerance"]
    rows=[]
    history_gaps=[]
    plateau_steps=0
    positive_steps=0
    future_optimizer_calls=0
    next_planning_scores=[]
    fallback="bottleneck" if policy=="expected_gap_bottleneck" else "width"
    for t in range(checkpoints[-1]+1):
        bounds=ledger.snapshot()
        upper,lower,gap=gap_bounds(bounds,k)
        if t and gap>history_gaps[-1]:
            raise AssertionError("nested intervals increased certification gap")
        history_gaps.append(gap)
        if t in checkpoints:
            score=certificate(bounds,k,case["tolerance"])
            if upper!=score["robust_macro_error_upper"] or (
                    lower!=score["true_optimum_lower"]
                    or gap!=score["optimality_gap_upper"]):
                raise AssertionError("lookahead objective differs from QO-07 certified gap")
            covered=all(lo<=rates[i][a]<=hi
                        for i,row in enumerate(ledger.bounds)
                        for a,(lo,hi) in enumerate(row))
            decision=score["decision"]
            incorrect=(decision=="certified-feasible" and not feasible or
                       decision=="certified-infeasible" and feasible)
            if covered and incorrect:
                raise AssertionError("wrong certificate on the coverage event")
            error=frozen_error(
                rates,score["transient_labels"],score["frozen_midpoints"])
            if covered and error>upper:
                raise AssertionError("frozen model not contained in robust upper bound")
            regret=partition_radius(rates,score["transient_labels"])-optimal
            if regret<0:
                raise AssertionError("negative true optimum regret")
            rows.append({
                "scenario":case["id"],"seed":seed,"policy":policy,"queries":t,
                "decision":decision,"true_feasible":feasible,
                "false_cert":bool(incorrect),"true_rates_covered":covered,
                "true_optimum":str(optimal),"robust_upper":str(upper),
                "true_optimum_lower":str(lower),
                "certification_gap":str(gap),
                "true_frozen_error":str(error),"partition_regret":str(regret),
                "interval_max_width":str(score["maximum_interval_width"]),
                "queried_cells":sum(c>0 for row in ledger.counts for c in row),
                "count_by_cell":[list(row) for row in ledger.counts],
                "planning_lookahead_calls":future_optimizer_calls,
                "planning_plateau_steps":plateau_steps,
                "planning_positive_steps":positive_steps,
                "planned_queries":t,
            })
        if t==checkpoints[-1]:
            break
        (i,a),info=choose_expected_gap_probe(ledger,k,fallback=fallback)
        future_optimizer_calls+=info["hypothetical_optimizer_calls"]
        if info["mode"].startswith("fallback"):
            plateau_steps+=1
        else:
            positive_steps+=1
        next_planning_scores.append(info["best_expected_gap_reduction"])
        observation=int(rng[i,a].random()<float(rates[i][a]))
        ledger.observe(i,a,observation)
    if ledger.query_count()!=checkpoints[-1] or (
            plateau_steps+positive_steps!=ledger.query_count()):
        raise AssertionError("incorrect query/planning ledger")
    if any(history_gaps[i+1]>history_gaps[i]
           for i in range(len(history_gaps)-1)):
        raise AssertionError("realized gap monotonicity broken")
    return rows,tuple(ledger.receipts),{
        "hypothetical_optimizer_calls":future_optimizer_calls,
        "plateau_steps":plateau_steps,
        "positive_steps":positive_steps,
        "initial_gap":str(history_gaps[0]),
        "final_gap":str(history_gaps[-1]),
        "sum_predicted_gap_gain":str(sum((Q(gain) for gain in next_planning_scores),Q(0))),
        "actual_gap_reduction":str(history_gaps[0]-history_gaps[-1]),
        "gap_nonincreasing":True,
    }


def _mean(values):
    seq=tuple(Q(x) for x in values)
    return sum(seq,Q(0))/len(seq)


def full_report(scenarios=SCENARIOS,seeds=SEEDS,
                policies=POLICIES,checkpoints=CHECKPOINTS):
    """Deterministic complete panel, source-hash ready; no cherry-picked seeds."""
    scenarios,seeds,policies,checkpoints=map(
        tuple,(scenarios,seeds,policies,checkpoints))
    _valid_checkpoints(checkpoints)
    if (not scenarios or not seeds or len(set(seeds))!=len(seeds)
            or not policies or len(set(policies))!=len(policies)
            or any(p not in POLICIES for p in policies)):
        raise ValueError("invalid declared pilot casebook or policies")
    rows=[]
    trajectories=[]
    planning=[]
    for ci,case in enumerate(scenarios):
        for seed in seeds:
            for policy in policies:
                if policy in ("round_robin","width","bottleneck"):
                    r,receipts=qo08_run_policy(
                        ci,case,seed,policy,checkpoints)
                    for x in r:
                        # QO-08 baseline computes R and L but its default
                        # row has no explicit gap; add from its own numbers.
                        x["certification_gap"]=str(
                            Q(x["robust_upper"])-Q(x["true_optimum_lower"]))
                        x["planning_lookahead_calls"]=0
                        x["planning_plateau_steps"]=0
                        x["planning_positive_steps"]=0
                    metrics=None
                else:
                    r,receipts,metrics=gap_policy_runner(
                        ci,case,seed,policy,checkpoints)
                    planning.append({
                        "scenario":case["id"],"seed":seed,"policy":policy,
                        **metrics})
                if len(receipts)!=checkpoints[-1] or not (
                        verify_common_stream_prefix(ci,case,seed,receipts)):
                    raise AssertionError("paired random stream/budget mismatch")
                rows.extend(r)
                trajectories.append({
                    "scenario":case["id"],"seed":seed,"policy":policy,
                    "queries":len(receipts),
                    "receipt_sha256":sha256(
                        json.dumps(receipts,separators=(",",":")).encode("ascii")
                    ).hexdigest()
                })
    groups=[]
    comparisons=[]
    for case in scenarios:
        for budget in checkpoints:
            index={
                policy:{r["seed"]:r for r in rows
                        if r["scenario"]==case["id"]
                        and r["queries"]==budget and r["policy"]==policy}
                for policy in policies
            }
            for policy in policies:
                subset=list(index[policy].values())
                if len(subset)!=len(seeds):
                    raise AssertionError("partial paired policy group")
                groups.append({
                    "scenario":case["id"],"budget":budget,"policy":policy,
                    "n":len(subset),
                    "true_feasible":subset[0]["true_feasible"],
                    "certified_feasible":sum(x["decision"]=="certified-feasible"
                                             for x in subset),
                    "certified_infeasible":sum(x["decision"]=="certified-infeasible"
                                               for x in subset),
                    "undetermined":sum(x["decision"]=="undetermined"
                                      for x in subset),
                    "false_certificates":sum(x["false_cert"] for x in subset),
                    "coverage_failures":sum(not x["true_rates_covered"]
                                            for x in subset),
                    "mean_certification_gap":str(_mean(
                        x["certification_gap"] for x in subset)),
                    "mean_true_error":str(_mean(
                        x["true_frozen_error"] for x in subset)),
                    "mean_robust_upper":str(_mean(
                        x["robust_upper"] for x in subset)),
                    "mean_planning_optimizer_calls":str(_mean(
                        x["planning_lookahead_calls"] for x in subset)),
                    "mean_lookahead_plateau_steps":str(_mean(
                        x["planning_plateau_steps"] for x in subset)),
                })
            for candidate in ("expected_gap_bottleneck","expected_gap_width"):
                if candidate not in index:
                    continue
                for baseline in policies:
                    if baseline==candidate:
                        continue
                    comparisons.append({
                        "scenario":case["id"],"budget":budget,
                        "candidate":candidate,"baseline":baseline,
                        "mean_gap_improvement":str(_mean(
                            Q(index[baseline][s]["certification_gap"])-
                            Q(index[candidate][s]["certification_gap"])
                            for s in seeds)),
                        "mean_true_error_improvement":str(_mean(
                            Q(index[baseline][s]["true_frozen_error"])-
                            Q(index[candidate][s]["true_frozen_error"])
                            for s in seeds)),
                        "candidate_more_certificates":sum(
                            index[candidate][s]["decision"]!="undetermined"
                            and index[baseline][s]["decision"]=="undetermined"
                            for s in seeds),
                        "candidate_fewer_certificates":sum(
                            index[candidate][s]["decision"]=="undetermined"
                            and index[baseline][s]["decision"]!="undetermined"
                            for s in seeds),
                    })
    return {
        "contract":"QO-11 one-step expected certification-gap acquisition",
        "scope":"QO-07 synthetics with addressable IID Bernoulli cells, no neural model",
        "method":"Beta(1,1) planning predictive; unchanged QO-07 CS",
        "confidence_alpha":str(ALPHA),"bisection_bits":BISECTION_BITS,
        "policies":list(policies),"seeds":list(seeds),
        "checkpoints":list(checkpoints),
        "scenarios":[{
            "id":c["id"],"panel":c["panel"],
            "rates":[[str(x) for x in row] for row in c["rates"]],
            "classes":c["classes"],"tolerance":str(c["tolerance"]),
        }for c in scenarios],
        "trial_rows":rows,
        "trajectory_receipt_hashes":trajectories,
        "planning_receipts":planning,
        "groups":groups,
        "paired_comparisons":comparisons,
        "run_counts":{
            "trajectories":len(trajectories),
            "snapshots":len(rows),
            "queries":sum(x["queries"] for x in trajectories),
            "stream_replays":len(trajectories),
            "planning_traces":len(planning),
        },
        "limitations":"One-step Beta planning is myopic; plateau fallback; synthetic fixed cases and nonzero planning compute. No trained controller.",
    }

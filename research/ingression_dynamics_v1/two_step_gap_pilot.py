"""QO-12: paired synthetic two-step Bellman gap acquisition pilot.

The two new policies invoke horizon-two planning ONLY on one-step
plateaus; baseline QO-08 and QO-11 runners are unchanged. Every policy
uses identical Bernoulli stream prefixes for the same scenario/seed.
No biological data or trained recurrent controller is involved.

The pilot counts actual gap optimizer calls, Bellman candidate branches,
two-step plateau visits, fallback steps and all confidence failures.
"""
from fractions import Fraction as Q
from hashlib import sha256
from itertools import product
from random import Random
import json

from research.ingression_dynamics_v1.anytime_active_quotient import (
    ProbeLedger, certificate, frozen_error
)
from research.ingression_dynamics_v1.expected_gap_probe import gap_bounds
from research.ingression_dynamics_v1.two_step_gap_probe import (
    choose_two_step_on_plateau
)
from research.ingression_dynamics_v1.geometric_quotient import optimum_for_k
from research.ingression_dynamics_v1.rate_sampling import partition_radius
from research.ingression_dynamics_v1.paired_probe_pilot import (
    run_policy as qo08_policy, stream_seed, verify_common_stream_prefix
)
from research.ingression_dynamics_v1.expected_gap_pilot import (
    gap_policy_runner as qo11_policy, SCENARIOS as QO11_CASES
)

POLICIES=("bottleneck","expected_gap_bottleneck",
          "two_step_bottleneck","two_step_width")
SEEDS=(449,457,461)
CHECKPOINTS=(24,64,96)
ALPHA=Q(1,20)
BISECTION_BITS=10

SCENARIOS=(
    {
        "id":"two_cell_feasible_plateau",
        "rates":((Q(1,2),),(Q(1,2),)),
        "classes":1,"tolerance":Q(1,5),
        "panel":"QO-12 new source-frozen plateau",
    },
    {
        "id":"two_cell_infeasible_plateau",
        "rates":((Q(1,10),),(Q(9,10),)),
        "classes":1,"tolerance":Q(1,5),
        "panel":"QO-12 new source-frozen plateau",
    },
    {**QO11_CASES[0],"panel":"retrospective QO-11 action0 singleton"},
    {**QO11_CASES[1],"panel":"retrospective QO-11 action1 pair"},
    {**QO11_CASES[4],"panel":"retrospective QO-11 two-action interaction"},
)


def run_two_step(case_index,case,seed,policy,checkpoints=CHECKPOINTS):
    if policy not in ("two_step_bottleneck","two_step_width"):
        raise ValueError("unknown QO-12 policy")
    checkpoints=tuple(checkpoints)
    if (not checkpoints or any(type(t) is not int or t<0 for t in checkpoints)
            or tuple(sorted(set(checkpoints)))!=checkpoints):
        raise ValueError("strictly increasing query checkpoints required")
    rates=case["rates"]
    n,d=len(rates),len(rates[0])
    k=case["classes"]
    ledger=ProbeLedger(n,d,ALPHA,BISECTION_BITS)
    rng={(i,a):Random(stream_seed(case_index,seed,i,a))
         for i,a in product(range(n),range(d))}
    true_opt,_,_=optimum_for_k(rates,k)
    true_feasible=true_opt<=case["tolerance"]
    rows=[]
    cost={
        "one_step_gap_calls":0,
        "two_step_gap_calls":0,
        "two_step_bellman_candidates":0,
        "two_step_branches":0,
        "two_step_invalid_branches":0,
        "one_step_positive":0,
        "two_step_positive":0,
        "fallback_bottleneck":0,
        "fallback_width":0,
    }
    previous_gap=None
    for step in range(checkpoints[-1]+1):
        bounds=ledger.snapshot()
        upper,lower,gap=gap_bounds(bounds,k)
        if previous_gap is not None and gap>previous_gap:
            raise AssertionError("nested confidence intervals enlarged gap")
        previous_gap=gap
        if step in checkpoints:
            score=certificate(bounds,k,case["tolerance"])
            if gap!=score["optimality_gap_upper"]:
                raise AssertionError("gap differs from QO-07 certificate")
            covered=all(lo<=rates[i][a]<=hi
                        for i,row in enumerate(bounds)
                        for a,(lo,hi) in enumerate(row))
            decision=score["decision"]
            incorrect=(decision=="certified-feasible" and not true_feasible or
                       decision=="certified-infeasible" and true_feasible)
            if covered and incorrect:
                raise AssertionError("wrong certificate on valid evidence")
            true_error=frozen_error(
                rates,score["transient_labels"],score["frozen_midpoints"])
            if covered and true_error>upper:
                raise AssertionError("frozen model error exceeded robust bound")
            regret=partition_radius(rates,score["transient_labels"])-true_opt
            if regret<0:
                raise AssertionError("negative partition regret")
            rows.append({
                "scenario":case["id"],"seed":seed,"policy":policy,
                "queries":step,"decision":decision,
                "true_feasible":true_feasible,
                "false_cert":bool(incorrect),
                "true_rates_covered":covered,
                "true_optimum":str(true_opt),
                "robust_upper":str(upper),
                "true_optimum_lower":str(lower),
                "certification_gap":str(gap),
                "true_frozen_error":str(true_error),
                "partition_regret":str(regret),
                "count_by_cell":[list(row) for row in ledger.counts],
                "planning_cost":dict(cost),
            })
        if step==checkpoints[-1]:
            break
        fallback="bottleneck" if policy=="two_step_bottleneck" else "width"
        (i,a),audit=choose_two_step_on_plateau(ledger,k,fallback=fallback)
        for name in ("one_step_gap_calls","two_step_gap_calls",
                     "two_step_bellman_candidates","two_step_branches",
                     "two_step_invalid_branches"):
            cost[name]+=audit[name]
        cost[audit["mode"]]+=1
        observed=int(rng[i,a].random()<float(rates[i][a]))
        ledger.observe(i,a,observed)
    if sum(cost[key] for key in (
            "one_step_positive","two_step_positive",
            "fallback_bottleneck","fallback_width"))!=ledger.query_count():
        raise AssertionError("planner steps did not match actual probes")
    return rows,tuple(ledger.receipts),cost


def _mean(values):
    v=tuple(Q(x) for x in values)
    return sum(v,Q(0))/len(v)


def complete_report(scenarios=SCENARIOS,seeds=SEEDS,
                    policies=POLICIES,checkpoints=CHECKPOINTS):
    scenarios,seeds,policies,checkpoints=map(
        tuple,(scenarios,seeds,policies,checkpoints))
    if (not scenarios or not seeds or len(set(seeds))!=len(seeds)
            or not policies or len(set(policies))!=len(policies)
            or any(p not in POLICIES for p in policies)
            or not checkpoints or tuple(sorted(set(checkpoints)))!=checkpoints):
        raise ValueError("invalid fixed pilot contract")
    rows=[]
    trails=[]
    planner_receipts=[]
    for ci,case in enumerate(scenarios):
        for seed in seeds:
            for policy in policies:
                if policy=="bottleneck":
                    trial,history=qo08_policy(
                        ci,case,seed,policy,checkpoints)
                    for row in trial:
                        row["certification_gap"]=str(
                            Q(row["robust_upper"])-Q(row["true_optimum_lower"]))
                        row["planning_cost"]={
                            "one_step_gap_calls":0,"two_step_gap_calls":0,
                            "two_step_bellman_candidates":0,
                            "two_step_branches":0,
                            "two_step_invalid_branches":0}
                elif policy=="expected_gap_bottleneck":
                    trial,history,cost=qo11_policy(
                        ci,case,seed,policy,checkpoints)
                    for row in trial:
                        row["planning_cost"]={
                            "one_step_gap_calls":row["planning_lookahead_calls"],
                            "two_step_gap_calls":0,
                            "two_step_bellman_candidates":0,
                            "two_step_branches":0,
                            "two_step_invalid_branches":0}
                else:
                    trial,history,cost=run_two_step(
                        ci,case,seed,policy,checkpoints)
                    planner_receipts.append({
                        "scenario":case["id"],"seed":seed,
                        "policy":policy,**cost,
                    })
                if len(history)!=checkpoints[-1] or not (
                        verify_common_stream_prefix(ci,case,seed,history)):
                    raise AssertionError("paired Bernoulli stream mismatch")
                rows.extend(trial)
                trails.append({
                    "scenario":case["id"],"seed":seed,
                    "policy":policy,"queries":len(history),
                    "sha256":sha256(json.dumps(
                        history,separators=(",",":")).encode("ascii")).hexdigest(),
                })
    grouped=[]
    compared=[]
    for case in scenarios:
        for budget in checkpoints:
            lookup={
                policy:{r["seed"]:r for r in rows
                        if r["scenario"]==case["id"]
                        and r["queries"]==budget and r["policy"]==policy}
                for policy in policies
            }
            for policy in policies:
                records=list(lookup[policy].values())
                if len(records)!=len(seeds):
                    raise AssertionError("incomplete case/seed/policy group")
                grouped.append({
                    "scenario":case["id"],"budget":budget,"policy":policy,
                    "n":len(records),
                    "correct_certificates":sum(r["decision"]!="undetermined"
                                               and not r["false_cert"]
                                               for r in records),
                    "unresolved":sum(r["decision"]=="undetermined" for r in records),
                    "false_certificates":sum(r["false_cert"] for r in records),
                    "coverage_failures":sum(not r["true_rates_covered"]
                                            for r in records),
                    "mean_gap":str(_mean(r["certification_gap"] for r in records)),
                    "mean_true_error":str(_mean(r["true_frozen_error"] for r in records)),
                    "mean_one_step_gap_calls":str(_mean(
                        r["planning_cost"]["one_step_gap_calls"] for r in records)),
                    "mean_two_step_gap_calls":str(_mean(
                        r["planning_cost"]["two_step_gap_calls"] for r in records)),
                })
            for candidate in ("two_step_bottleneck","two_step_width"):
                if candidate not in lookup:
                    continue
                for baseline in policies:
                    if baseline==candidate:
                        continue
                    compared.append({
                        "scenario":case["id"],"budget":budget,
                        "candidate":candidate,"baseline":baseline,
                        "paired_mean_gap_improvement":str(_mean(
                            Q(lookup[baseline][s]["certification_gap"])-
                            Q(lookup[candidate][s]["certification_gap"])
                            for s in seeds)),
                        "paired_mean_true_error_improvement":str(_mean(
                            Q(lookup[baseline][s]["true_frozen_error"])-
                            Q(lookup[candidate][s]["true_frozen_error"])
                            for s in seeds)),
                        "paired_certificate_wins":sum(
                            lookup[candidate][s]["decision"]!="undetermined"
                            and lookup[baseline][s]["decision"]=="undetermined"
                            for s in seeds),
                        "paired_certificate_losses":sum(
                            lookup[candidate][s]["decision"]=="undetermined"
                            and lookup[baseline][s]["decision"]!="undetermined"
                            for s in seeds),
                    })
    return {
        "contract":"QO-12 two-step gap Bellman on zero-gain plateaus",
        "scope":"synthetic IID resettable state/action Bernoulli fixtures only",
        "planning":"exact two-step recourse with Beta(1,1) predictive, triggered only when one-step value zero",
        "statistical_certificate":"unchanged QO-07 anytime beta-mixture e-process",
        "seeds":list(seeds),"checkpoints":list(checkpoints),
        "policies":list(policies),
        "scenarios":[{
            "id":case["id"],"panel":case["panel"],
            "rates":[[str(x) for x in r] for r in case["rates"]],
            "classes":case["classes"],"tolerance":str(case["tolerance"]),
        }for case in scenarios],
        "trial_rows":rows,
        "trajectory_hashes":trails,
        "planning_receipts":planner_receipts,
        "groups":grouped,
        "paired_comparisons":compared,
        "run_counts":{
            "trajectories":len(trails),
            "snapshots":len(rows),
            "queries":sum(x["queries"] for x in trails),
            "stream_replays":len(trails),
            "two_step_planning_traces":len(planner_receipts),
        },
        "limitations":"Myopic two-step pilot, not optimal horizon>2; lookahead CPU cost counted as optimizations, no timed neural inference.",
    }

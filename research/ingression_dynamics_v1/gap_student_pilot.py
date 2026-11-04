"""QO-16: train 26/138-parameter students on exact QO-12 gap progress.

All data seen during fitting comes from QO-14's original train/validation
rates, with their original seeds and freshly sampled longer histories.
The independent frozen QO-16 generator is imported ONLY after training.
The unchanged QO-07 e-process alone issues feasibility certificates.
"""
from fractions import Fraction as F
from hashlib import sha256
import json
from random import Random

from research.ingression_dynamics_v1.anytime_active_quotient import (
    ProbeLedger, certificate, choose_probe, frozen_error,
)
from research.ingression_dynamics_v1.certificate_student_data import (
    ALPHA, BISECTION_BITS, TOLERANCE, evidence_features,
)
from research.ingression_dynamics_v1.gap_progress_teacher import (
    READINESS_MIN_PER_CELL, NORMALIZED_GAIN_GATE,
    gap_action_values, gain_to_target, greedy_gain_cell,
)
from research.ingression_dynamics_v1.gap_student_data import (
    train_validation, example_from_ledger, source_stats,
)
from research.ingression_dynamics_v1.tiny_recurrent_certificate_student import (
    create_linear, create_rnn, fit, forward, inference_multiplies,
    trainable_count,
)
from research.ingression_dynamics_v1.paired_probe_pilot import (
    run_policy as old_bottleneck, stream_seed, verify_common_stream_prefix,
)

ARMS=("bottleneck", "exact_qo12_gap_h2",
      "linear_gap_student", "recurrent_gap_student")
STUDENTS=("linear_gap_student", "recurrent_gap_student")
CLUSTER_BOOTSTRAP_REPS=2000
CLUSTER_BOOTSTRAP_SEED=20261104


def metric_for_student(model, rows):
    if not rows:
        raise ValueError("nonempty independent test sample required")
    informative=disagree=0
    total_raw_regret=conditional_raw_regret=0.0
    total_scaled_regret=total_scaled_margin_mae=total_scaled_q_mae=0.0
    engagement_errors=over=under=0
    buckets={"zero":0,"0_to_0.0001":0,"0.0001_to_0.001":0,
             "0.001_to_0.01":0,"gt_0.01":0}
    for ex in rows:
        q=tuple(map(float,ex["raw_q"]))
        y=tuple(map(float,ex["q"]))
        p=forward(model,ex["tokens"])
        chosen=0 if p[0]>=p[1] else 1
        raw_delta=q[0]-q[1]
        scaled_delta=y[0]-y[1]
        pred_delta=p[0]-p[1]
        raw_regret=max(q)-q[chosen]
        scaled_regret=max(y)-y[chosen]
        if scaled_regret > abs(scaled_delta-pred_delta)+1e-11:
            raise AssertionError("MOR-13 margin regret inequality violated")
        if raw_delta != 0:
            informative+=1
            conditional_raw_regret+=raw_regret
            disagree+=int(chosen!=(0 if raw_delta>=0 else 1))
        total_raw_regret+=raw_regret
        total_scaled_regret+=scaled_regret
        total_scaled_margin_mae+=abs(scaled_delta-pred_delta)
        total_scaled_q_mae+=(abs(y[0]-p[0])+abs(y[1]-p[1]))/2
        tr=max(y)>=float(NORMALIZED_GAIN_GATE)
        pr=max(p)>=float(NORMALIZED_GAIN_GATE)
        engagement_errors+=int(tr!=pr)
        over+=int(pr and not tr)
        under+=int(tr and not pr)
        margin=abs(raw_delta)
        if margin==0:buckets["zero"]+=1
        elif margin<=1e-4:buckets["0_to_0.0001"]+=1
        elif margin<=1e-3:buckets["0.0001_to_0.001"]+=1
        elif margin<=1e-2:buckets["0.001_to_0.01"]+=1
        else:buckets["gt_0.01"]+=1
    n=len(rows)
    return {
        "n":n,"informative":informative,
        "informative_action_disagreements":disagree,
        "conditional_exact_progress_regret":(
            conditional_raw_regret/informative if informative else None),
        "unconditional_exact_progress_regret":total_raw_regret/n,
        "mean_normalized_progress_regret":total_scaled_regret/n,
        "mean_normalized_margin_error":total_scaled_margin_mae/n,
        "mean_normalized_q_error":total_scaled_q_mae/n,
        "engagement_errors":engagement_errors,
        "false_engagements":over,"missed_engagements":under,
        "exact_progress_margin_buckets":buckets,
    }


def fit_gap_students():
    """Every fit and epoch-selection operation is independent of frozen test."""
    training,train_stats=train_validation("train")
    validation,val_stats=train_validation("validation")
    if not training or not validation:
        raise AssertionError("empty source training family")
    linear=fit(create_linear(),training,validation)
    recurrent=fit(create_rnn(),training,validation)
    return {
        "models":{
            "linear_gap_student":linear["weights"],
            "recurrent_gap_student":recurrent["weights"],
        },
        "fits":{
            "linear_gap_student":{k:v for k,v in linear.items() if k!="weights"},
            "recurrent_gap_student":{k:v for k,v in recurrent.items() if k!="weights"},
        },
        "source_train":train_stats,
        "source_validation":val_stats,
    }


def select_gap_probe(ledger,arm,models,cost):
    baseline=choose_probe(ledger,1,"bottleneck")
    if min(ledger.counts[i][0] for i in range(2))<READINESS_MIN_PER_CELL:
        cost["readiness_fallbacks"]+=1
        return baseline
    if certificate(ledger.snapshot(),1,TOLERANCE)["decision"]!="undetermined":
        cost["already_certified_fallbacks"]+=1
        return baseline
    cost["eligible_states"]+=1
    if arm=="exact_qo12_gap_h2":
        exact,stats=gap_action_values(ledger)
        p=tuple(gain_to_target(x) for x in exact)
        cost["exact_teacher_calls"]+=1
        cost["exact_gap_evaluations"]+=stats["gap_evaluations"]
        cost["exact_second_action_candidates"]+=stats["candidate_action_evaluations"]
        cost["invalid_hypothetical_branches"]+=stats["invalid_branches"]
        scores=p
    elif arm in STUDENTS:
        model=models[arm]
        scores=forward(model,evidence_features(ledger))
        cost["student_inferences"]+=1
        cost["student_multiply_accumulates"]+=inference_multiplies(model)
    else:
        raise ValueError("unknown learned or exact policy")
    if max(scores)<NORMALIZED_GAIN_GATE:
        cost["low_gain_fallbacks"]+=1
        return baseline
    selected=greedy_gain_cell(ledger,scores)
    cost["model_chosen_actions"]+=1
    cost["selected_cell_"+str(selected)]+=1
    return selected,0


def _cost():
    return {
        "readiness_fallbacks":0, "already_certified_fallbacks":0,
        "eligible_states":0,"exact_teacher_calls":0,
        "exact_gap_evaluations":0,"exact_second_action_candidates":0,
        "invalid_hypothetical_branches":0,
        "student_inferences":0,"student_multiply_accumulates":0,
        "low_gain_fallbacks":0,"model_chosen_actions":0,
        "selected_cell_0":0,"selected_cell_1":0,
    }


def run_gap_policy(ci,case,seed,arm,models,checkpoints):
    """Exactly one source-ordered observed Bernoulli draw per query."""
    if arm not in ARMS[1:]:
        raise ValueError("unknown QO-16 action-selection rule")
    if len(case["rates"])!=2 or any(len(x)!=1 for x in case["rates"]):
        raise ValueError("first-stage QO-16 is two-addressed-cell only")
    ledger=ProbeLedger(2,1,ALPHA,BISECTION_BITS)
    streams=[Random(stream_seed(ci,seed,i,0)) for i in range(2)]
    rates=case["rates"]
    true_feasible=(abs(rates[0][0]-rates[1][0])/2<=case["tolerance"])
    cost=_cost()
    snapshots=[]
    for t in range(checkpoints[-1]+1):
        if t in checkpoints:
            cert=certificate(ledger.snapshot(),1,case["tolerance"])
            covered=all(lo<=rates[i][0]<=hi for i,row in
                        enumerate(ledger.bounds) for lo,hi in row)
            decision=cert["decision"]
            wrong=((decision=="certified-feasible" and not true_feasible)
                   or (decision=="certified-infeasible" and true_feasible))
            if covered and wrong:
                raise AssertionError("wrong certificate while true p covered")
            error=frozen_error(rates,cert["transient_labels"],
                               cert["frozen_midpoints"])
            if covered and error>cert["robust_macro_error_upper"]:
                raise AssertionError("true frozen error above certificate")
            gap=cert["robust_macro_error_upper"]-cert["true_optimum_lower"]
            snapshots.append({
                "scenario":case["id"],"seed":seed,"policy":arm,
                "queries":t,"decision":decision,
                "false_cert":wrong,"true_rates_covered":covered,
                "true_feasible":true_feasible,
                "certification_gap":str(gap),
                "robust_upper":str(cert["robust_macro_error_upper"]),
                "true_optimum_lower":str(cert["true_optimum_lower"]),
                "true_frozen_error":str(error),
                "cell_query_counts":[ledger.counts[i][0] for i in range(2)],
                "policy_cost":dict(cost),
            })
        if t==checkpoints[-1]:
            break
        i,a=select_gap_probe(ledger,arm,models,cost)
        hit=int(streams[i].random()<float(rates[i][0]))
        ledger.observe(i,a,hit)
    if cost["readiness_fallbacks"]+cost["already_certified_fallbacks"]+(
        cost["low_gain_fallbacks"]+cost["model_chosen_actions"])!=checkpoints[-1]:
        raise AssertionError("lost at least one logged policy query")
    if cost["eligible_states"]!=(
        cost["low_gain_fallbacks"]+cost["model_chosen_actions"]):
        raise AssertionError("unaccounted eligible oracle/student inference")
    return snapshots,tuple(ledger.receipts),cost


def _mean(values):
    vals=tuple(F(x) for x in values)
    return sum(vals,F(0))/len(vals)


def _bootstrap_cluster_interval(sample_by_scenario,reps=CLUSTER_BOOTSTRAP_REPS):
    """Deterministic 8-cluster percentile bootstrap, resampling rate pairs."""
    means=[float(_mean(v)) for k,v in sorted(sample_by_scenario.items())]
    if not means:
        raise ValueError("empty paired bootstrap")
    rng=Random(CLUSTER_BOOTSTRAP_SEED)
    vals=[]
    for _ in range(reps):
        vals.append(sum(means[rng.randrange(len(means))]
                        for _ in range(len(means)))/len(means))
    vals.sort()
    return [vals[int(.025*reps)],vals[min(reps-1,int(.975*reps))]]


def _paired_comparisons(rows,arms,checkpoints,scenarios,seeds):
    comparisons=[]
    for at in checkpoints:
        baseline={
            (r["scenario"],r["seed"]):r for r in rows
            if r["queries"]==at and r["policy"]=="bottleneck"
        }
        for arm in arms[1:]:
            arm_rows={
                (r["scenario"],r["seed"]):r for r in rows
                if r["queries"]==at and r["policy"]==arm
            }
            cluster={}
            paired=[]
            for case in scenarios:
                sid=case["id"]
                for seed in seeds:
                    key=(sid,seed)
                    before=F(baseline[key]["certification_gap"])
                    after=F(arm_rows[key]["certification_gap"])
                    improvement=before-after
                    cluster.setdefault(sid,[]).append(improvement)
                    paired.append({
                        "scenario":sid,"seed":seed,
                        "exact_gap_improvement":str(improvement),
                        "improved":bool(improvement>0),
                        "tied":bool(improvement==0),
                        "worsened":bool(improvement<0),
                    })
            diffs=[F(x["exact_gap_improvement"]) for x in paired]
            comparisons.append({
                "budget":at,"policy":arm,
                "against":"bottleneck",
                "matched_n":len(diffs),
                "paired_mean_gap_improvement":str(_mean(diffs)),
                "positive_deltas":sum(x>0 for x in diffs),
                "ties":sum(x==0 for x in diffs),
                "negative_deltas":sum(x<0 for x in diffs),
                "cluster_bootstrap_percentile_95":_bootstrap_cluster_interval(cluster),
                "per_case_seed":paired,
            })
    return comparisons


def full_report():
    trained=fit_gap_students()
    models=trained["models"]
    # Frozen test source becomes accessible ONLY after both models are fitted.
    from research.ingression_dynamics_v1.gap_student_frozen_test import (
        FROZEN_SCENARIOS,FROZEN_SUPERVISED_SEEDS,
        FROZEN_POLICY_SEEDS,FROZEN_POLICY_CHECKPOINTS,
        generate_frozen_ledgers,frozen_contract,frozen_contract_sha256,
    )
    source_ledgers,failures=generate_frozen_ledgers()
    supervised=[]
    work_meta={"teacher_gap_evaluations":0,
               "teacher_candidate_evaluations":0,
               "invalid_hypothetical_branches":0}
    for idx,seed,sched,t,ledger in source_ledgers:
        ex=example_from_ledger(ledger,"frozen_test",idx,seed,sched,t)
        if ex["q"]==(F(0),F(0)) and certificate(
                ledger.snapshot(),1,TOLERANCE)["decision"]!="undetermined":
            continue
        supervised.append(ex)
        work_meta["teacher_gap_evaluations"]+=ex["teacher_gap_evaluations"]
        work_meta["teacher_candidate_evaluations"]+=ex["teacher_candidate_evaluations"]
        work_meta["invalid_hypothetical_branches"]+=ex["teacher_invalid_branches"]
    if not supervised:
        raise AssertionError("frozen test has no undecided public states")
    test_stats=source_stats(supervised,failures)
    test_stats.update(work_meta)
    metrics={arm:metric_for_student(model,supervised)
             for arm,model in models.items()}
    rows,traces=[],[]
    costs={arm:_cost() for arm in ARMS}
    for ci,case in enumerate(FROZEN_SCENARIOS):
        for seed in FROZEN_POLICY_SEEDS:
            for arm in ARMS:
                if arm=="bottleneck":
                    observed,receipts=old_bottleneck(
                        ci,case,seed,"bottleneck",FROZEN_POLICY_CHECKPOINTS)
                    cost=_cost()
                    observed=[dict(r,policy="bottleneck",
                               certification_gap=str(
                                   F(r["robust_upper"])-F(r["true_optimum_lower"])),
                               policy_cost=dict(cost)) for r in observed]
                else:
                    observed,receipts,cost=run_gap_policy(
                        ci,case,seed,arm,models,FROZEN_POLICY_CHECKPOINTS)
                if len(receipts)!=FROZEN_POLICY_CHECKPOINTS[-1] or not (
                        verify_common_stream_prefix(ci,case,seed,receipts)):
                    raise AssertionError("matched Bernoulli stream invalid")
                rows.extend(observed)
                traces.append({
                    "scenario":case["id"],"seed":seed,"policy":arm,
                    "queries":len(receipts),
                    "sha256":sha256(json.dumps(
                        receipts,separators=(",",":")).encode("ascii")).hexdigest(),
                })
                for k in costs[arm]:
                    costs[arm][k]+=cost[k]
    grouped=[]
    for at in FROZEN_POLICY_CHECKPOINTS:
        for arm in ARMS:
            subset=[r for r in rows if r["queries"]==at and r["policy"]==arm]
            assert len(subset)==len(FROZEN_SCENARIOS)*len(FROZEN_POLICY_SEEDS)
            grouped.append({
                "queries":at,"policy":arm,"n":len(subset),
                "correct_certificates":sum(
                    r["decision"]!="undetermined" and not r["false_cert"]
                    for r in subset),
                "false_certificates":sum(r["false_cert"] for r in subset),
                "coverage_failures":sum(not r["true_rates_covered"] for r in subset),
                "mean_certification_gap":str(_mean(
                    r["certification_gap"] for r in subset)),
                "mean_true_frozen_error":str(_mean(
                    r["true_frozen_error"] for r in subset)),
                "mean_robust_upper":str(_mean(
                    r["robust_upper"] for r in subset)),
            })
    return {
        "contract":"QO-16 source-frozen exact QO-12 gap-progress distillation",
        "teacher":"two-step expected QO-07 robust gap progress, Beta(1,1) predictive",
        "frozen_contract":frozen_contract(),
        "frozen_contract_sha256":frozen_contract_sha256(),
        "source_training":trained["source_train"],
        "source_validation":trained["source_validation"],
        "trained_models":models,
        "training_receipts":trained["fits"],
        "frozen_supervised_examples":len(supervised),
        "frozen_test_stats":test_stats,
        "frozen_test_failures":list(failures),
        "frozen_supervised_fingerprint":sha256(json.dumps(
            [(ex["rate_index"],ex["seed"],ex["schedule"],ex["time"],
              [str(x) for x in ex["raw_q"]])
             for ex in supervised],separators=(",",":")).encode("ascii")).hexdigest(),
        "supervised_metrics":metrics,
        "model_parameters":{k:trainable_count(v) for k,v in models.items()},
        "model_macs_per_inference":{k:inference_multiplies(v)
                                    for k,v in models.items()},
        "policy_arms":list(ARMS),
        "policy_groups":grouped,
        "policy_costs":costs,
        "paired_comparisons":_paired_comparisons(
            rows,ARMS,FROZEN_POLICY_CHECKPOINTS,
            FROZEN_SCENARIOS,FROZEN_POLICY_SEEDS),
        "policy_rows":rows,
        "trajectory_hashes":traces,
        "counts":{
            "trajectories":len(traces),
            "snapshots":len(rows),
            "queries":sum(t["queries"] for t in traces),
        },
        "limitations":(
            "Finite digital known-addressable Bernoulli states; "
            "source-frozen empirical evidence only, horizon-2 planning "
            "gain does not certify predictive calibration or transfer."
        ),
    }

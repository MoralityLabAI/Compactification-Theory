"""QO-14 frozen CPU distillation and paired probe-policy evaluation.

Models train on QO-13 horizon-two oracle first-action Q-values,
with rate-family and seed splits defined in recurrent_teacher_data.
The only learned parameters are the reservoir readout and linear
control coefficients. All test-rate tables are unseen during training.

The QO-07 beta-mixture confidence ledger is unchanged and decides
certifications. Predictions may only choose the NEXT probe. Every
query receives a matched per-cell Bernoulli stream, independently of
whether the student/teacher chooses that cell.

Teacher planning costs, frozen students' MAC counts and the model
bytes are recorded. No model/API/GPU work or hidden real measurements.
"""
from fractions import Fraction as Q
from hashlib import sha256
from itertools import product
from random import Random
import json
import numpy as np

from research.ingression_dynamics_v1.anytime_active_quotient import (
    ProbeLedger,choose_probe,certificate,frozen_error
)
from research.ingression_dynamics_v1.certificate_reachability import (
    optimal_certificate_reachability
)
from research.ingression_dynamics_v1.recurrent_teacher_data import (
    dataset,dataset_summary,TRAIN_RATES,TEST_RATES,TRAIN_SEEDS,TEST_SEEDS,
    TAU,ALPHA,BITS,encode_cell_tokens,audit_fixed_split,teacher_record
)
from research.ingression_dynamics_v1.recurrent_probe_student import (
    train_students,predict_q,evaluate_student_on_teacher,
    VALUE_THRESHOLD
)

POLICIES=("bottleneck","linear","recurrent","oracle_h2")
ROLLOUT_SEEDS=(809,811,821,823)
ROLLOUT_RATE_CASES=TEST_RATES[:4]
CHECKPOINTS=(24,64,96)
MIN_SAMPLE_PER_CELL=8
DECISION_MARGIN=0.005


def rollout_cell_stream(case_index,seed,rate,cell):
    """Identical QO-08-style response prefixes under every policy."""
    rng=Random(20261104*1000003 + case_index*10000019 +
               seed*9907 + cell*7919)
    while True:
        yield int(rng.random()<float(rate))


def rollout(policy,case_index,rates,seed,checkpoint,model):
    if policy not in POLICIES or checkpoint<0:
        raise ValueError("registered policy and nonnegative budget required")
    ledger=ProbeLedger(2,1,ALPHA,BITS)
    streams={
        (0,0):rollout_cell_stream(case_index,seed,rates[0],0),
        (1,0):rollout_cell_stream(case_index,seed,rates[1],1),
    }
    teacher_calls=0;teacher_candidates=0
    student_calls=0;learned_choices=0
    fallback_calls=0;model_MACs=0
    trace=[]
    for t in range(checkpoint):
        report=certificate(ledger.snapshot(),1,TAU)
        if report["decision"]!="undetermined":
            # Fixed-budget continuation preserves all response ledgers,
            # while no further expensive learned planning is needed.
            chosen=choose_probe(ledger,1,"bottleneck")
            fallback_calls+=1
        elif policy=="bottleneck" or min(ledger.counts[0][0],ledger.counts[1][0])<MIN_SAMPLE_PER_CELL:
            chosen=choose_probe(ledger,1,"bottleneck")
            fallback_calls+=1
        elif policy=="oracle_h2":
            exact=teacher_record(ledger)
            teacher_calls+=exact["teacher_gap_calls"]
            teacher_candidates+=exact["teacher_bellman_candidates"]
            qa=float(exact["q"][0]);qb=float(exact["q"][1])
            if max(qa,qb)>=VALUE_THRESHOLD and abs(qa-qb)>DECISION_MARGIN:
                chosen=(int(qb>qa),0)
                learned_choices+=1
            else:
                chosen=choose_probe(ledger,1,"bottleneck")
                fallback_calls+=1
        else:
            qa,qb=predict_q(model,encode_cell_tokens(ledger),policy)
            student_calls+=1
            model_MACs+=(model["recurrent_macs_per_inference_estimate"] if
                         policy=="recurrent" else
                         model["linear_macs_per_inference_estimate"])
            if max(qa,qb)>=VALUE_THRESHOLD and abs(qa-qb)>DECISION_MARGIN:
                chosen=(int(qb>qa),0)
                learned_choices+=1
            else:
                chosen=choose_probe(ledger,1,"bottleneck")
                fallback_calls+=1
        event=next(streams[chosen])
        ledger.observe(chosen[0],chosen[1],event)
        trace.append((chosen[0],chosen[1],event))
    report=certificate(ledger.snapshot(),1,TAU)
    truth=abs(rates[0]-rates[1])/2<=TAU
    decision=report["decision"]
    false=(decision=="certified-feasible" and not truth or
           decision=="certified-infeasible" and truth)
    covered=all(lo<=rates[i]<=hi for i,row in enumerate(ledger.bounds)
                for lo,hi in row)
    if covered and false:
        raise AssertionError("false certificate despite valid confidence interval")
    error=frozen_error(
        tuple((x,) for x in rates),report["transient_labels"],
        report["frozen_midpoints"])
    if covered and error>report["robust_macro_error_upper"]:
        raise AssertionError("true frozen error exceeds confidence certificate")
    return {
        "rate_family":[str(x) for x in rates],
        "case_index":case_index,
        "seed":seed,
        "policy":policy,
        "queries":checkpoint,
        "decision":decision,
        "true_feasible":truth,
        "false_cert":bool(false),
        "true_rate_coverage":covered,
        "certification_gap":str(report["optimality_gap_upper"]),
        "true_frozen_error":str(error),
        "count_by_cell":[list(v) for v in ledger.counts],
        "teacher_gap_evaluations":teacher_calls,
        "teacher_bellman_candidates":teacher_candidates,
        "student_inferences":student_calls,
        "student_MACs_estimate":model_MACs,
        "student_selected_probes":learned_choices,
        "bottleneck_fallbacks":fallback_calls,
        "trace_sha256":sha256(
            json.dumps(trace,separators=(",",":")).encode("ascii")).hexdigest(),
    }


def _mean(values):
    seq=tuple(Q(x) for x in values)
    return str(sum(seq,Q(0))/len(seq))


def build_report(train_rows=None,test_rows=None,
                 policies=POLICIES,
                 rollout_rates=ROLLOUT_RATE_CASES,
                 rollout_seeds=ROLLOUT_SEEDS,
                 checkpoints=CHECKPOINTS):
    """One frozen train, one frozen evaluation. No train-on-test adaptation."""
    audit_fixed_split()
    train_rows=dataset("train") if train_rows is None else train_rows
    test_rows=dataset("test") if test_rows is None else test_rows
    if not train_rows or not test_rows:
        raise ValueError("train and held-out examples required")
    model=train_students(train_rows)
    student_metrics=[
        evaluate_student_on_teacher(test_rows,model,name)
        for name in ("linear","recurrent")
    ]
    rows=[]
    for case_index,rates in enumerate(rollout_rates):
        for seed in rollout_seeds:
            for policy in policies:
                for budget in checkpoints:
                    rows.append(rollout(policy,case_index,rates,seed,
                                        budget,model))
    groups=[]
    for case_index,rates in enumerate(rollout_rates):
        for budget in checkpoints:
            for policy in policies:
                selected=[
                    x for x in rows if x["case_index"]==case_index and
                    x["queries"]==budget and x["policy"]==policy
                ]
                if len(selected)!=len(rollout_seeds):
                    raise AssertionError("incomplete independent holdout")
                groups.append({
                    "rate_family":[str(x) for x in rates],
                    "case_index":case_index,"budget":budget,
                    "policy":policy,"n":len(selected),
                    "correct_certificates":sum(
                        x["decision"]!="undetermined" and not x["false_cert"]
                        for x in selected),
                    "undetermined":sum(x["decision"]=="undetermined" for x in selected),
                    "false_certificates":sum(x["false_cert"] for x in selected),
                    "coverage_failures":sum(not x["true_rate_coverage"] for x in selected),
                    "mean_certification_gap":_mean(
                        x["certification_gap"] for x in selected),
                    "mean_true_error":_mean(x["true_frozen_error"] for x in selected),
                    "mean_teacher_gap_calls":_mean(
                        x["teacher_gap_evaluations"] for x in selected),
                    "mean_student_MACs":_mean(
                        x["student_MACs_estimate"] for x in selected),
                    "mean_student_selected_probes":_mean(
                        x["student_selected_probes"] for x in selected),
                })
    return {
        "contract":"QO-14 reservoir recurrent readout distillation of exact QO-13 Bellman Q values",
        "teacher":"QO-13 horizon=2, exact QO-07 nested evidence",
        "model_kind":"fixed tanh recurrent reservoir, trained ridge readout (not BPTT)",
        "training_split":audit_fixed_split(),
        "train_summary":dataset_summary(train_rows),
        "test_summary":dataset_summary(test_rows),
        "test_student_metrics":student_metrics,
        "model_checkpoint":model,
        "rollout_test_rate_pairs":[[str(x) for x in pair] for pair in rollout_rates],
        "rollout_test_seeds":list(rollout_seeds),
        "rollout_checkpoints":list(checkpoints),
        "policies":list(policies),
        "rows":rows,
        "groups":groups,
        "counts":{
            "training_examples":len(train_rows),
            "test_examples":len(test_rows),
            "policy_trials":len(rows),
            "rollout_queries":sum(x["queries"] for x in rows),
        },
        "limitations":"synthetic fixed rate pairs, only readout learned, frozen QO-07 certificate, approximate MAC counts",
    }

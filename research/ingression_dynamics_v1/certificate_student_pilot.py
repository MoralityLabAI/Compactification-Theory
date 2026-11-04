"""QO-14 paired distillation experiment, deterministic pure-Python training.

Teacher labels: exact QO-13 horizon-two per-action certificate
reachability probabilities. Students: trained linear and BPTT tanh-RNN.
All train/validation/test sources and evaluation seeds are disjoint.

The student chooses probes from its Q-value predictions and the public
evidence ledger only. The true rates stay inside the Bernoulli simulator
and are used again solely for post-run audits. QO-07's unchanged anytime
e-process is the sole certificate authority.

Fixed simulation contract:
    six unseen rate pairs, 3 new seeds, 4 policies (bottleneck, linear,
    trained-RNN, exact horizon-two teacher), checkpoints 32/56/80.
    72 complete trajectories and 5,760 Bernoulli queries.
"""
from fractions import Fraction as Q
from hashlib import sha256
from itertools import product
from random import Random
import json

from research.ingression_dynamics_v1.certificate_student_data import (
    ALPHA, BISECTION_BITS, TOLERANCE, TEST_RATES, TEST_SEEDS,
    TRAIN_RATES, VALIDATION_RATES, TRAIN_SEEDS, VALIDATION_SEEDS,
    generate_split, teacher_q_values, split_signature, evidence_features
)
from research.ingression_dynamics_v1.tiny_recurrent_certificate_student import (
    create_linear,create_rnn,fit,forward,from_receipt,
    evaluate_supervised,inference_multiplies,trainable_count
)
from research.ingression_dynamics_v1.anytime_active_quotient import (
    ProbeLedger,certificate,choose_probe,frozen_error
)
from research.ingression_dynamics_v1.paired_probe_pilot import (
    run_policy as old_bottleneck,stream_seed,verify_common_stream_prefix
)

POLICIES=("bottleneck","linear_student","recurrent_student","exact_h2_teacher")
POLICY_SEEDS=(1301,1307,1319)
CHECKPOINTS=(32,56,80)
READINESS_MIN_PER_CELL=8
MIN_PREDICTED_VALUE=0.05
EVAL_SCENARIOS=tuple({
    "id":"heldout_pair_"+str(i),
    "rates":((pair[0],),(pair[1],)),
    "classes":1,
    "tolerance":TOLERANCE,
    "origin":"rate-disjoint held-out family",
} for i,pair in enumerate(TEST_RATES))


def _numeric_ratio(num,den):
    return num/den if den else 0.0


def _data_stats(rows,failures):
    informative=sum(ex["q"][0]!=ex["q"][1] for ex in rows)
    positive=sum(max(ex["q"])>0 for ex in rows)
    certified=sum(ex["q"]==(Q(1),Q(1)) for ex in rows)
    return {
        "total_records":len(rows),
        "informative_action_labels":informative,
        "positive_value_records":positive,
        "already_certified_records":certified,
        "stream_confidence_inconsistencies":list(failures),
    }


def fit_students():
    """Train only on train split, select epochs by validation loss only."""
    training, train_failed=generate_split("train")
    validation, val_failed=generate_split("validation")
    test, test_failed=generate_split("test")
    # Already-certified states have no meaningful future probe choice.
    train=tuple(ex for ex in training if ex["q"]!=(Q(1),Q(1)))
    val=tuple(ex for ex in validation if ex["q"]!=(Q(1),Q(1)))
    heldout=tuple(ex for ex in test if ex["q"]!=(Q(1),Q(1)))
    if not train or not val or not heldout:
        raise AssertionError("incomplete predeclared supervised splits")
    linear=fit(create_linear(),train,val)
    recurrent=fit(create_rnn(),train,val)
    models={"linear_student":linear["weights"],
            "recurrent_student":recurrent["weights"]}
    performance={
        name:evaluate_supervised(model,heldout)
        for name,model in models.items()
    }
    return {
        "models":models,
        "training_receipts":{
            "linear_student":{k:v for k,v in linear.items() if k!="weights"},
            "recurrent_student":{k:v for k,v in recurrent.items() if k!="weights"},
        },
        "supervised_test_metrics":performance,
        "split_sizes":{
            "train":len(train),"validation":len(val),"test":len(heldout)
        },
        "raw_splits":{
            "train":_data_stats(training,train_failed),
            "validation":_data_stats(validation,val_failed),
            "test":_data_stats(test,test_failed)
        },
        "split_parameters":{
            "train_rates":[list(map(str,p)) for p in TRAIN_RATES],
            "validation_rates":[list(map(str,p)) for p in VALIDATION_RATES],
            "test_rates":[list(map(str,p)) for p in TEST_RATES],
            "train_seeds":list(TRAIN_SEEDS),
            "validation_seeds":list(VALIDATION_SEEDS),
            "supervised_test_seeds":list(TEST_SEEDS),
            "policy_test_seeds":list(POLICY_SEEDS),
        },
        "first_supervised_test_hash":sha256(
            json.dumps([split_signature(ex) for ex in heldout],
                       separators=(",",":")).encode("ascii")).hexdigest(),
    }


def student_choice(ledger,policy,models,counters):
    """Predictable fixed student/teacher action; bottleneck fallback.

    Both trained students and the exact teacher have identical
    readiness conditions and absolute-action-value gate. A tiny
    predicted success probability is treated as an unresolved
    acquisition signal, where bottleneck is the registered fallback.
    """
    if policy not in POLICIES[1:]:
        raise ValueError("unknown learned/teacher policy")
    if (ledger.n,ledger.d)!=(2,1):
        raise ValueError("first student only handles two one-action cells")
    if min(ledger.counts[i][0] for i in range(2))<READINESS_MIN_PER_CELL:
        counters["readiness_fallback"]+=1
        return choose_probe(ledger,1,"bottleneck")
    if certificate(ledger.snapshot(),1,TOLERANCE)["decision"]!="undetermined":
        counters["already_certified_fallback"]+=1
        return choose_probe(ledger,1,"bottleneck")
    if policy=="exact_h2_teacher":
        q=tuple(float(x) for x in teacher_q_values(ledger))
        counters["teacher_q_labels"]+=1
    else:
        model=models[policy]
        q=forward(model,evidence_features(ledger))
        counters["student_inferences"]+=1
        counters["student_multiply_accumulates"]+=inference_multiplies(model)
    if max(q)<MIN_PREDICTED_VALUE:
        counters["low_value_fallback"]+=1
        return choose_probe(ledger,1,"bottleneck")
    selected=0 if q[0]>=q[1] else 1
    counters["chosen_actions"]+=1
    counters["selected_cell_"+str(selected)]+=1
    return selected,0


def run_student_policy(ci,case,seed,policy,models,checkpoints=CHECKPOINTS):
    if policy not in POLICIES[1:]:
        raise ValueError("invalid student/teacher policy")
    rates=case["rates"]
    if len(rates)!=2 or any(len(x)!=1 for x in rates):
        raise ValueError("two-state, one-action rate family required")
    ledger=ProbeLedger(2,1,ALPHA,BISECTION_BITS)
    streams=[Random(stream_seed(ci,seed,i,0)) for i in range(2)]
    counters={
        "readiness_fallback":0,"already_certified_fallback":0,
        "low_value_fallback":0,"teacher_q_labels":0,
        "student_inferences":0,"student_multiply_accumulates":0,
        "chosen_actions":0,"selected_cell_0":0,"selected_cell_1":0,
    }
    rows=[]
    optimum=abs(rates[0][0]-rates[1][0])/2
    true_feasible=optimum<=TOLERANCE
    for t in range(checkpoints[-1]+1):
        if t in checkpoints:
            score=certificate(ledger.snapshot(),1,TOLERANCE)
            covered=all(lo<=rates[i][0]<=hi
                        for i,row in enumerate(ledger.bounds)
                        for lo,hi in row)
            decision=score["decision"]
            wrong=((decision=="certified-feasible" and not true_feasible)
                   or (decision=="certified-infeasible" and true_feasible))
            if covered and wrong:
                raise AssertionError("incorrect certificate on valid confidence set")
            frozen=frozen_error(rates,score["transient_labels"],
                                score["frozen_midpoints"])
            if covered and frozen>score["robust_macro_error_upper"]:
                raise AssertionError("frozen macro error exceeds robust certificate")
            rows.append({
                "scenario":case["id"],"seed":seed,"policy":policy,
                "queries":t,"decision":decision,
                "false_cert":bool(wrong),"true_rates_covered":covered,
                "true_feasible":true_feasible,
                "certification_gap":str(score["optimality_gap_upper"]),
                "robust_upper":str(score["robust_macro_error_upper"]),
                "true_frozen_error":str(frozen),
                "cell_query_counts":[ledger.counts[i][0] for i in range(2)],
                "policy_cost":dict(counters),
            })
        if t==checkpoints[-1]:
            break
        i,a=student_choice(ledger,policy,models,counters)
        hit=int(streams[i].random()<float(rates[i][0]))
        ledger.observe(i,a,hit)
    if sum((counters[k] for k in (
            "readiness_fallback","already_certified_fallback",
            "low_value_fallback","chosen_actions")))!=(checkpoints[-1]):
        raise AssertionError("unaccounted student policy decisions")
    return rows,tuple(ledger.receipts),counters


def _mean_exact(strings):
    x=tuple(Q(s) for s in strings)
    return str(sum(x,Q(0))/len(x))


def full_report():
    trained=fit_students()
    models=trained["models"]
    rows=[]
    receipts=[]
    for ci,case in enumerate(EVAL_SCENARIOS):
        for seed in POLICY_SEEDS:
            for policy in POLICIES:
                if policy=="bottleneck":
                    rr,history=old_bottleneck(ci,case,seed,policy,CHECKPOINTS)
                    for row in rr:
                        row["certification_gap"]=str(
                            Q(row["robust_upper"])-Q(row["true_optimum_lower"]))
                        row["policy_cost"]={"teacher_q_labels":0,
                              "student_inferences":0,
                              "student_multiply_accumulates":0}
                else:
                    rr,history,_=run_student_policy(
                        ci,case,seed,policy,models,CHECKPOINTS)
                if len(history)!=CHECKPOINTS[-1] or not (
                        verify_common_stream_prefix(ci,case,seed,history)):
                    raise AssertionError("query counts or paired stream mismatch")
                rows.extend(rr)
                receipts.append({
                    "scenario":case["id"],"seed":seed,"policy":policy,
                    "queries":len(history),
                    "sha256":sha256(json.dumps(
                        history,separators=(",",":")).encode("ascii")).hexdigest(),
                })
    groups=[]
    for case in EVAL_SCENARIOS:
        for checkpoint in CHECKPOINTS:
            for policy in POLICIES:
                subset=[x for x in rows if x["scenario"]==case["id"] and
                        x["queries"]==checkpoint and x["policy"]==policy]
                if len(subset)!=len(POLICY_SEEDS):
                    raise AssertionError("incomplete test group")
                groups.append({
                    "scenario":case["id"],"checkpoint":checkpoint,
                    "policy":policy,"n":len(subset),
                    "correct_certificates":sum(
                        x["decision"]!="undetermined" and not x["false_cert"]
                        for x in subset),
                    "unresolved":sum(x["decision"]=="undetermined" for x in subset),
                    "wrong_certificates":sum(x["false_cert"] for x in subset),
                    "coverage_failures":sum(not x["true_rates_covered"]
                                            for x in subset),
                    "mean_certification_gap":_mean_exact(
                        x["certification_gap"] for x in subset),
                    "mean_true_frozen_error":_mean_exact(
                        x["true_frozen_error"] for x in subset),
                    "mean_student_multiply_accumulates":_mean_exact(
                        x["policy_cost"].get("student_multiply_accumulates",0)
                        for x in subset),
                    "mean_exact_teacher_lookups":_mean_exact(
                        x["policy_cost"].get("teacher_q_labels",0)
                        for x in subset),
                })
    return {
        "contract":"QO-14 trainable recurrent distillation of exact QO-13 action values",
        "evaluation":"fixed, rate-disjoint synthetic test family and fresh policy seeds",
        "teacher":"horizon-2 exact rational certificate-reachability Q-values",
        "split_sizes":trained["split_sizes"],
        "raw_splits":trained["raw_splits"],
        "split_parameters":trained["split_parameters"],
        "test_input_fingerprint":trained["first_supervised_test_hash"],
        "training":trained["training_receipts"],
        "models":models,
        "supervised_test_metrics":trained["supervised_test_metrics"],
        "inference_parameter_counts":{n:trainable_count(m)
                                      for n,m in models.items()},
        "inference_multiply_accumulates":{n:inference_multiplies(m)
                                         for n,m in models.items()},
        "policies":list(POLICIES),"seeds":list(POLICY_SEEDS),
        "checkpoints":list(CHECKPOINTS),
        "scenarios":[{
            "id":v["id"],"rates":[[str(p) for p in row] for row in v["rates"]],
            "source":v["origin"],
        } for v in EVAL_SCENARIOS],
        "trial_rows":rows,
        "groups":groups,
        "trajectory_hashes":receipts,
        "counts":{"trajectories":len(receipts),
                  "snapshots":len(rows),
                  "queries":sum(x["queries"] for x in receipts)},
        "limitations":"two-state one-action digital hit/hold, teacher horizon 2; no neural LLM transfer, no certified recurrent model error bound",
    }

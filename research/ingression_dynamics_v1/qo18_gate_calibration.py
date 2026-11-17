"""QO-18 validation-only selective engagement-gate calibration.

Student and bottleneck source-policy histories use original QO-14
VALIDATION rate families and freshly source-fixed CALIBRATION seeds.
Each eligible public state receives exact QO-12 progress labels only
AFTER the policy has already chosen and observed its Bernoulli action.
No independent frozen test family or source file is accessed here.
"""
from fractions import Fraction as F
from random import Random
from hashlib import sha256
import json

from research.ingression_dynamics_v1.certificate_student_data import (
    VALIDATION_RATES,ALPHA,BISECTION_BITS,TOLERANCE,evidence_features,
)
from research.ingression_dynamics_v1.anytime_active_quotient import (
    ProbeLedger,choose_probe,
)
from research.ingression_dynamics_v1.qo17_occupancy_data import (
    is_eligible,_public_state_receipt,
)
from research.ingression_dynamics_v1.gap_student_data import (
    example_from_ledger,
)
from research.ingression_dynamics_v1.gap_progress_teacher import (
    greedy_gain_cell,
)
from research.ingression_dynamics_v1.tiny_recurrent_certificate_student import (
    forward,inference_multiplies,
)
from research.ingression_dynamics_v1.paired_probe_pilot import (
    stream_seed,verify_common_stream_prefix,
)

CAL_SEEDS=(6503,6521)
POLICIES=("student_fixed_005","bottleneck")
CHECKPOINTS=(24,32,40,56,72,96,128,160)
THRESHOLDS=(F(5,100),F(10,100),F(15,100),F(20,100),
            F(25,100),F(35,100),F(50,100))
TEACHER_GATE=F(1,20)
MAX_FALSE_RATE=F(1,10)
MIN_RECALL=F(4,5)


def validation_calibration_states(model):
    """Independent of model fitting and any QO-18 frozen test distribution."""
    samples,trace_hashes=[],[]
    cost={
        "student_policy_inferences":0,
        "student_policy_macs":0,
        "student_policy_selected":0,
        "student_low_gain_fallbacks":0,
        "fallback_unready_or_certified":0,
        "posthoc_teacher_labels":0,
        "posthoc_gap_evaluations":0,
        "skipped_ineligible_checkpoints":0,
    }
    for idx,rate in enumerate(VALIDATION_RATES):
        case={"rates":((rate[0],),(rate[1],))}
        for seed in CAL_SEEDS:
            for policy in POLICIES:
                ledger=ProbeLedger(2,1,ALPHA,BISECTION_BITS)
                streams=[Random(stream_seed(idx,seed,i,0)) for i in (0,1)]
                for t in range(CHECKPOINTS[-1]+1):
                    if t in CHECKPOINTS:
                        if is_eligible(ledger):
                            ex=example_from_ledger(
                                ledger,"qo18_validation_calibration",
                                idx,seed,policy,t)
                            pred=forward(model,ex["tokens"])
                            teacher=max(ex["q"])>=TEACHER_GATE
                            samples.append({
                                "rate_index":idx,"seed":seed,
                                "source_policy":policy,"queries":t,
                                "cluster":str(idx)+":"+str(seed)+":"+policy,
                                "teacher_positive":teacher,
                                "true_transformed_q":[str(x) for x in ex["q"]],
                                "predicted_q":list(pred),
                                "model_score":max(pred),
                                "public_evidence":_public_state_receipt(ledger),
                            })
                            cost["posthoc_teacher_labels"]+=1
                            cost["posthoc_gap_evaluations"]+=ex["teacher_gap_evaluations"]
                        else:
                            cost["skipped_ineligible_checkpoints"]+=1
                    if t==CHECKPOINTS[-1]:
                        break
                    fallback=choose_probe(ledger,1,"bottleneck")
                    if policy=="bottleneck":
                        chosen=fallback
                    elif not is_eligible(ledger):
                        chosen=fallback
                        cost["fallback_unready_or_certified"]+=1
                    else:
                        scores=forward(model,evidence_features(ledger))
                        cost["student_policy_inferences"]+=1
                        cost["student_policy_macs"]+=inference_multiplies(model)
                        if max(scores)<float(TEACHER_GATE):
                            chosen=fallback
                            cost["student_low_gain_fallbacks"]+=1
                        else:
                            chosen=(greedy_gain_cell(ledger,scores),0)
                            cost["student_policy_selected"]+=1
                    i,a=chosen
                    ledger.observe(i,a,int(streams[i].random()<float(rate[i])))
                if not verify_common_stream_prefix(idx,case,seed,ledger.receipts):
                    raise AssertionError("nonreplayable validation calibration stream")
                trace_hashes.append({
                    "rate_index":idx,"seed":seed,"source_policy":policy,
                    "queries":len(ledger.receipts),
                    "sha256":sha256(json.dumps(
                        ledger.receipts,separators=(",",":")).encode("ascii")).hexdigest(),
                })
    return tuple(samples),{
        "source_rate_count":len(VALIDATION_RATES),
        "source_seeds":list(CAL_SEEDS),
        "source_policies":list(POLICIES),
        "source_checkpoints":list(CHECKPOINTS),
        "sample_count":len(samples),
        "cluster_count":len(set(x["cluster"] for x in samples)),
        "trajectory_count":len(trace_hashes),
        "trajectory_hashes":trace_hashes,
        "costs":cost,
    }


def confusion(samples,threshold):
    """Exact finite empirical counts; sampled states are cluster-dependent."""
    if threshold is None:
        # The prespecified safe fallback never engages the learned policy.
        predicted=lambda x:False
    else:
        threshold=float(F(threshold))
        predicted=lambda x:x["model_score"]>=threshold
    tp=fp=tn=fn=0
    cluster_counts={}
    for row in samples:
        truth=row["teacher_positive"]
        guess=predicted(row)
        key=row["cluster"]
        z=cluster_counts.setdefault(key,{"tp":0,"fp":0,"tn":0,"fn":0})
        if truth and guess:
            tp+=1;z["tp"]+=1
        elif guess:
            fp+=1;z["fp"]+=1
        elif truth:
            fn+=1;z["fn"]+=1
        else:
            tn+=1;z["tn"]+=1
    return {
        "tp":tp,"fp":fp,"tn":tn,"fn":fn,
        "positives":tp+fn,"negatives":fp+tn,
        "precision":tp/(tp+fp) if tp+fp else None,
        "recall":tp/(tp+fn) if tp+fn else None,
        "false_engagement_rate":fp/(fp+tn) if fp+tn else None,
        "cluster_counts":cluster_counts,
        "observed_clusters":len(cluster_counts),
    }


def select_gate(samples):
    """Frozen grid, 10% false-positive and 80% true-positive recall constraints.

    A valid threshold requires nonempty teacher-positive AND teacher-negative
    reference denominators. If none qualify, explicitly report infeasible
    and route every action to the QO-08 bottleneck fallback at deployment.
    """
    if not samples:
        return {"threshold":None,"feasible":False,
                "reason":"empty_independent_validation",
                "candidates":[],"chosen_validation_confusion":confusion(samples,None)}
    candidates=[]
    feasible=[]
    for threshold in THRESHOLDS:
        cm=confusion(samples,threshold)
        eligible=(cm["positives"]>0 and cm["negatives"]>0 and
            F(cm["fp"],cm["negatives"])<=MAX_FALSE_RATE and
            F(cm["tp"],cm["positives"])>=MIN_RECALL)
        candidates.append({
            "threshold":str(threshold),
            "feasible":eligible,
            "confusion":cm,
        })
        if eligible:
            feasible.append((F(cm["tp"],cm["positives"]),threshold,cm))
    if feasible:
        # Highest true-positive recall, deterministic lowest threshold on ties.
        feasible.sort(key=lambda row:(-row[0],row[1]))
        _,best,cm=feasible[0]
        return {
            "threshold":str(best),"feasible":True,
            "reason":"validation_constraints_satisfied",
            "candidates":candidates,"chosen_validation_confusion":cm,
        }
    return {
        "threshold":None,"feasible":False,
        "reason":"no_grid_threshold_satisfies_validation_constraints",
        "candidates":candidates,
        "chosen_validation_confusion":confusion(samples,None),
    }

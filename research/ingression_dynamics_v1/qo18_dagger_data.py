"""QO-18 train-only DAgger-like student-policy occupancy aggregation.

No QO-18 frozen test import in this module. Every source ledger comes from
the original QO-14 train or validation rate family with new, disjoint
round-specific Bernoulli seeds. Collection uses the *actual* student
predictions and QO-07 bottleneck fallback, with an exact QO-12 label on
each separately recorded eligible public state after its selection.
"""
from fractions import Fraction as F
from random import Random
from hashlib import sha256
import json

from research.ingression_dynamics_v1.certificate_student_data import (
    TRAIN_RATES,VALIDATION_RATES,ALPHA,BISECTION_BITS,TOLERANCE,
    evidence_features,
)
from research.ingression_dynamics_v1.qo17_occupancy_data import (
    is_eligible,_public_state_receipt,training_mixture,
)
from research.ingression_dynamics_v1.gap_student_data import example_from_ledger
from research.ingression_dynamics_v1.gap_progress_teacher import (
    greedy_gain_cell,
)
from research.ingression_dynamics_v1.tiny_recurrent_certificate_student import (
    create_rnn,forward,trainable_count,inference_multiplies,
)
from research.ingression_dynamics_v1.qo17_margin_gate_loss import (
    fit_augmented,
)
from research.ingression_dynamics_v1.anytime_active_quotient import (
    ProbeLedger, choose_probe,
)
from research.ingression_dynamics_v1.paired_probe_pilot import (
    stream_seed,verify_common_stream_prefix,
)

DAGGER_SEEDS={
    "train":((5701,5711),(5903,5917)),
    "validation":((6101,6113),(6301,6311)),
}
CHECKPOINTS=(24,40,56,72,96,128,160)
POLICY_GATE=F(1,20)
NEGATIVE_MAX_UNIQUE=64
NEGATIVE_EXTRA_COPIES=8
SOURCE_TAG="student_policy_fixed_005"


def rate_family(split):
    if split=="train":return TRAIN_RATES
    if split=="validation":return VALIDATION_RATES
    raise ValueError("data from train or validation families only")


def _student_cell(ledger,model,counts):
    fallback=choose_probe(ledger,1,"bottleneck")
    if not is_eligible(ledger):
        counts["unready_or_certified_fallbacks"]+=1
        return fallback
    counts["model_inferences"]+=1
    counts["model_macs"]+=inference_multiplies(model)
    outputs=forward(model,evidence_features(ledger))
    if max(outputs)<float(POLICY_GATE):
        counts["low_gain_fallbacks"]+=1
        return fallback
    counts["model_chosen"]+=1
    return greedy_gain_cell(ledger,outputs),0


def generate_round(split,round_index,model):
    if round_index not in (0,1):
        raise ValueError("exactly two registered DAgger collection rounds")
    if trainable_count(model)!=138:
        raise ValueError("DAgger student must have unchanged RNN capacity")
    rows=[]
    receipts=[]
    costs={
        "model_inferences":0,"model_macs":0,
        "model_chosen":0,"low_gain_fallbacks":0,
        "unready_or_certified_fallbacks":0,
        "teacher_labels":0,"teacher_gap_evaluations":0,
        "teacher_candidate_evaluations":0,
        "ineligible_snapshots":0,
    }
    rates=rate_family(split)
    for idx,rate in enumerate(rates):
        case={"rates":((rate[0],),(rate[1],)),
              "id":"qo18_"+split+"_"+str(idx)}
        for seed in DAGGER_SEEDS[split][round_index]:
            ledger=ProbeLedger(2,1,ALPHA,BISECTION_BITS)
            streams=[Random(stream_seed(idx,seed,i,0)) for i in (0,1)]
            for t in range(CHECKPOINTS[-1]+1):
                if t in CHECKPOINTS:
                    if is_eligible(ledger):
                        ex=example_from_ledger(
                            ledger,split+"_dagger",idx,seed,
                            SOURCE_TAG+"_round_"+str(round_index+1),t)
                        ex["round"]=round_index+1
                        ex["source_policy"]=SOURCE_TAG
                        ex["public_state_receipt"]=_public_state_receipt(ledger)
                        rows.append(ex)
                        costs["teacher_labels"]+=1
                        costs["teacher_gap_evaluations"]+=ex["teacher_gap_evaluations"]
                        costs["teacher_candidate_evaluations"]+=ex[
                            "teacher_candidate_evaluations"]
                    else:
                        costs["ineligible_snapshots"]+=1
                if t==CHECKPOINTS[-1]:
                    break
                i,a=_student_cell(ledger,model,costs)
                observed=int(streams[i].random()<float(rate[i]))
                ledger.observe(i,a,observed)
            if not verify_common_stream_prefix(idx,case,seed,ledger.receipts):
                raise AssertionError("student training source trajectory not replayable")
            receipts.append({
                "rate_index":idx,"seed":seed,"split":split,
                "round":round_index+1,"queries":ledger.query_count(),
                "source_policy":SOURCE_TAG,
                "sha256":sha256(json.dumps(
                    ledger.receipts,separators=(",",":")).encode("ascii")).hexdigest(),
            })
    evidence=[{
        "rate_index":ex["rate_index"],
        "seed":ex["seed"],"round":ex["round"],"time":ex["time"],
        "source_policy":ex["source_policy"],
        "public_evidence":ex["public_state_receipt"],
        "exact_q":[str(x) for x in ex["raw_q"]],
        "transformed_q":[str(x) for x in ex["q"]],
    } for ex in rows]
    return tuple(rows),{
        "split":split,"round":round_index+1,
        "seeds":list(DAGGER_SEEDS[split][round_index]),
        "rate_count":len(rates),
        "eligible_snapshots":len(rows),
        "teacher_negative_states":sum(max(ex["q"])<POLICY_GATE for ex in rows),
        "informative_states":sum(ex["q"][0]!=ex["q"][1] for ex in rows),
        "costs":costs,
        "source_trajectory_hashes":receipts,
        "source_evidence":evidence,
    }


def hard_negative_copies(rows):
    """Stable first-64 unique visited negatives, 8 extra copies each.

    This is training-only augmentation. A negative source state is
    eligible if the exact teacher's normalized max progress <1/20.
    Validation and independent test are evaluated on natural proportions.
    """
    negative=[]
    seen=set()
    for ex in rows:
        if max(ex["q"])>=POLICY_GATE:
            continue
        signature=(ex["rate_index"],ex["seed"],ex["time"],
                   tuple(ex["public_state_receipt"]["bounds"][0]),
                   tuple(ex["public_state_receipt"]["bounds"][1]))
        if signature in seen:
            continue
        seen.add(signature)
        negative.append(ex)
        if len(negative)==NEGATIVE_MAX_UNIQUE:
            break
    extra=tuple(negative)*NEGATIVE_EXTRA_COPIES
    return extra,{"selected_unique":len(negative),
                  "extra_copies":len(extra),
                  "total_candidate_negatives":sum(
                      max(ex["q"])<POLICY_GATE for ex in rows)}


def train_dagger_rounds(initial_model):
    """Return two validation-selected stage weights; never inspect tests."""
    from research.ingression_dynamics_v1.qo17_margin_gate_loss import fit_augmented
    baseline_train,base_train_meta=training_mixture("train")
    baseline_val,base_val_meta=training_mixture("validation")
    cumulative_train,cumulative_val=[],[]
    model=initial_model
    frozen_rounds=[]
    for ri in (0,1):
        new_train,train_meta=generate_round("train",ri,model)
        new_val,val_meta=generate_round("validation",ri,model)
        cumulative_train.extend(new_train)
        cumulative_val.extend(new_val)
        extras,hard_meta=hard_negative_copies(cumulative_train)
        train=tuple(baseline_train)+tuple(cumulative_train)+extras
        validation=tuple(baseline_val)+tuple(cumulative_val)
        fitted=fit_augmented(create_rnn(),train,validation)
        model=fitted["weights"]
        frozen_rounds.append({
            "round":ri+1,
            "model":model,
            "fit":{k:v for k,v in fitted.items() if k!="weights"},
            "train_meta":train_meta,
            "validation_meta":val_meta,
            "hard_negative_oversampling":hard_meta,
            "training_rows":len(train),"validation_rows":len(validation),
            "source_training_rows":len(baseline_train),
            "source_validation_rows":len(baseline_val),
            "cumulative_student_training_states":len(cumulative_train),
            "cumulative_student_validation_states":len(cumulative_val),
        })
    return {
        "rounds":frozen_rounds,
        "source_training_meta":base_train_meta,
        "source_validation_meta":base_val_meta,
        "final_model":model,
    }

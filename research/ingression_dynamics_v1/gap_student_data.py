"""QO-16 training and validation evidence from QO-14's original source families.

Only original QO-14 training/validation rates and seeds are accessed.
Labels come from QO-12 two-step exact expected robust-gap reduction,
then pass through a fixed monotone bounded transform for sigmoid heads.
Never import the frozen QO-16 test module here.
"""
from copy import deepcopy
from fractions import Fraction as F
from random import Random

from research.ingression_dynamics_v1.anytime_active_quotient import ProbeLedger
from research.ingression_dynamics_v1.certificate_student_data import (
    ALPHA, BISECTION_BITS, TRAIN_RATES, TRAIN_SEEDS,
    VALIDATION_RATES, VALIDATION_SEEDS, SCHEDULES, probe_choice,
    evidence_features,
)
from research.ingression_dynamics_v1.gap_progress_teacher import (
    gap_action_values, gain_to_target,
)

SOURCE_CHECKPOINTS = (24,40,64,96)


def example_from_ledger(ledger, split, rate_index, seed, schedule, t):
    raw, work=gap_action_values(ledger)
    labels=tuple(gain_to_target(x) for x in raw)
    return {
        "tokens": evidence_features(ledger),
        "q": labels,
        "raw_q": raw,
        "teacher_action": 0 if raw[0]>=raw[1] else 1,
        "split": split,
        "rate_index": rate_index,
        "seed": seed,
        "schedule": schedule,
        "time": t,
        "teacher_gap_evaluations": work["gap_evaluations"],
        "teacher_candidate_evaluations": work["candidate_action_evaluations"],
        "teacher_invalid_branches": work["invalid_branches"],
    }


def generate_source(split):
    if split=="train":
        rates,seeds=TRAIN_RATES,TRAIN_SEEDS
    elif split=="validation":
        rates,seeds=VALIDATION_RATES,VALIDATION_SEEDS
    else:
        raise ValueError("QO-16 source is only train or validation")
    rows,failures=[],[]
    for idx,rate in enumerate(rates):
        for seed in seeds:
            for schedule in SCHEDULES:
                ledger=ProbeLedger(2,1,ALPHA,BISECTION_BITS)
                rngs=[
                    Random(1000003*seed+100019*idx+9791*i+
                           8599*SCHEDULES.index(schedule))
                    for i in range(2)
                ]
                for t in range(1,SOURCE_CHECKPOINTS[-1]+1):
                    i,a=probe_choice(schedule,t-1)
                    try:
                        ledger.observe(i,a,int(rngs[i].random()<float(rate[i])))
                    except ValueError:
                        failures.append((split,idx,seed,schedule,t))
                        break
                    if t in SOURCE_CHECKPOINTS:
                        rows.append(example_from_ledger(
                            ledger,split,idx,seed,schedule,t))
    return tuple(rows),tuple(failures)


def swapped(ex):
    """Cell label permutation; transformation commutes with swapping."""
    result=dict(ex)
    result["tokens"]=(ex["tokens"][1],ex["tokens"][0])
    result["q"]=(ex["q"][1],ex["q"][0])
    result["raw_q"]=(ex["raw_q"][1],ex["raw_q"][0])
    result["teacher_action"]=0 if result["raw_q"][0]>=result["raw_q"][1] else 1
    result["permuted"]=True
    return result


def source_stats(rows,failures):
    n=len(rows)
    margins=[abs(x["raw_q"][0]-x["raw_q"][1]) for x in rows]
    vals=[max(x["raw_q"]) for x in rows]
    return {
        "original_examples":n,
        "informative_action_margins":sum(x>0 for x in margins),
        "positive_gain_examples":sum(x>0 for x in vals),
        "positive_both_actions":sum(min(x["raw_q"])>0 for x in rows),
        "already_zero_gain_examples":sum(x==0 for x in vals),
        "raw_margin_mean":str(sum(margins,F(0))/n) if n else None,
        "raw_progress_mean":str(sum(vals,F(0))/n) if n else None,
        "teacher_gap_evaluations":sum(x["teacher_gap_evaluations"] for x in rows),
        "teacher_candidate_evaluations":sum(
            x["teacher_candidate_evaluations"] for x in rows),
        "invalid_hypothetical_branches":sum(
            x["teacher_invalid_branches"] for x in rows),
        "confidence_stream_failures":list(failures),
    }


def train_validation(split):
    base,failures=generate_source(split)
    if not base:
        raise AssertionError("missing QO-16 supervised source")
    both=base+tuple(swapped(x) for x in base)
    meta=source_stats(base,failures)
    meta["used_examples_with_swap_augmentation"]=len(both)
    return both,meta

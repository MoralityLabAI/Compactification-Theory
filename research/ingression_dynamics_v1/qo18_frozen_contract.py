"""QO-18 FIRST COMMIT: independent frozen test and all decision rules.

Preregistered 2026-11-05, before QO-18 student training or held-out
measurement. This file may be imported by the QO-18 experiment only AFTER
all student fitting and validation-side threshold selection have finished.
The QO-07 confidence certificate and QO-12 exact planning labels stay
unchanged; full source-order Bernoulli histories are reconstructed.
"""
from fractions import Fraction as F
from random import Random
from copy import deepcopy
from hashlib import sha256
import json

from research.ingression_dynamics_v1.certificate_student_data import (
    ALPHA, BISECTION_BITS, TOLERANCE,
    TRAIN_RATES, VALIDATION_RATES, TEST_RATES,
    TRAIN_SEEDS, VALIDATION_SEEDS, TEST_SEEDS, SCHEDULES, probe_choice,
)
from research.ingression_dynamics_v1.certificate_boundary_test_contract import (
    FROZEN_RATES as QO15_RATES,
    FROZEN_SUPERVISED_SEEDS as QO15_SUP_SEEDS,
    FROZEN_POLICY_SEEDS as QO15_POL_SEEDS,
)
from research.ingression_dynamics_v1.gap_student_frozen_test import (
    FROZEN_RATES as QO16_RATES,
    FROZEN_SUPERVISED_SEEDS as QO16_SUP_SEEDS,
    FROZEN_POLICY_SEEDS as QO16_POL_SEEDS,
)
from research.ingression_dynamics_v1.qo17_frozen_contract import (
    FROZEN_RATES as QO17_RATES,
    FROZEN_SUPERVISED_SEEDS as QO17_SUP_SEEDS,
    FROZEN_POLICY_SEEDS as QO17_POL_SEEDS,
    TRAIN_OCCUPANCY_SEEDS as QO17_TRAIN_OCC,
    VALIDATION_OCCUPANCY_SEEDS as QO17_VAL_OCC,
)
from research.ingression_dynamics_v1.anytime_active_quotient import ProbeLedger

FROZEN_RATES=(
    (F(7,120), F(113,120)), (F(19,120), F(101,120)),
    (F(31,120), F(89,120)), (F(43,120), F(77,120)),
    (F(53,120), F(67,120)), (F(61,120), F(59,120)),
    (F(83,120), F(37,120)), (F(103,120), F(17,120)),
)
FROZEN_SUPERVISED_SEEDS=(4103,4127,4153)
FROZEN_POLICY_SEEDS=(4307,4327,4339)
SUPERVISED_CHECKPOINTS=(24,40,64,96)
POLICY_CHECKPOINTS=(32,80,160)
VISITED_CHECKPOINTS=(24,32,48,80,112,160)
DAGGER_SOURCE_CHECKPOINTS=(24,40,56,72,96,128,160)
TRAIN_DAGGER_SEEDS=((5701,5711),(5903,5917))
VALIDATION_DAGGER_SEEDS=((6101,6113),(6301,6311))
CALIBRATION_SEEDS=(6503,6521)
CALIBRATION_SOURCE_POLICIES=("student_fixed_005","bottleneck")
CALIBRATION_CHECKPOINTS=(24,32,40,56,72,96,128,160)
GATE_CANDIDATES=(F(5,100),F(10,100),F(15,100),F(20,100),
                 F(25,100),F(35,100),F(50,100))
TEACHER_GATE=F(1,20)
MAX_FALSE_ENGAGEMENT=F(1,10)
MIN_TRUE_RECALL=F(4,5)
NEGATIVE_OVERSAMPLE_COPIES=8
NEGATIVE_OVERSAMPLE_MAX_UNIQUE=64
DAGGER_ROUNDS=2
SOURCE_POLICY_GATE=F(1,20)
ORACLE_SEED_BOOTSTRAP=20261104
BOOTSTRAP_REPS=2000
ARMS=("bottleneck","exact_gap_h2","qo17_frozen_rnn",
      "dagger_round1_fixed_005","dagger_round2_fixed_005",
      "dagger_round2_calibrated")
PRIMARY_ARM="dagger_round2_calibrated"
FROZEN_SCENARIOS=tuple({
    "id":"qo18_frozen_"+str(i),
    "rates":((x,),(y,)), "classes":1,"tolerance":TOLERANCE,
} for i,(x,y) in enumerate(FROZEN_RATES))

assert len(set(FROZEN_RATES))==len(FROZEN_RATES)==8
assert not (set(FROZEN_RATES) &
    (set(TRAIN_RATES)|set(VALIDATION_RATES)|set(TEST_RATES)|
     set(QO15_RATES)|set(QO16_RATES)|set(QO17_RATES)))
old_seeds=(set(TRAIN_SEEDS)|set(VALIDATION_SEEDS)|set(TEST_SEEDS)|
    set(QO15_SUP_SEEDS)|set(QO15_POL_SEEDS)|
    set(QO16_SUP_SEEDS)|set(QO16_POL_SEEDS)|
    set(QO17_SUP_SEEDS)|set(QO17_POL_SEEDS)|
    set(QO17_TRAIN_OCC)|set(QO17_VAL_OCC))
new_seeds=(tuple(FROZEN_SUPERVISED_SEEDS)+tuple(FROZEN_POLICY_SEEDS)+
    tuple(sum((list(x) for x in TRAIN_DAGGER_SEEDS),[]))+
    tuple(sum((list(x) for x in VALIDATION_DAGGER_SEEDS),[]))+
    tuple(CALIBRATION_SEEDS))
assert len(set(new_seeds))==len(new_seeds)
assert not (set(new_seeds)&old_seeds)

def contract():
    return {
        "test_rates":[[str(x),str(y)] for x,y in FROZEN_RATES],
        "supervised_test_seeds":list(FROZEN_SUPERVISED_SEEDS),
        "policy_test_seeds":list(FROZEN_POLICY_SEEDS),
        "supervised_checkpoints":list(SUPERVISED_CHECKPOINTS),
        "policy_checkpoints":list(POLICY_CHECKPOINTS),
        "visited_checkpoints":list(VISITED_CHECKPOINTS),
        "dagger_source_checkpoints":list(DAGGER_SOURCE_CHECKPOINTS),
        "dagger_training_seeds":[list(x) for x in TRAIN_DAGGER_SEEDS],
        "dagger_validation_seeds":[list(x) for x in VALIDATION_DAGGER_SEEDS],
        "dagger_rounds":DAGGER_ROUNDS,
        "dagger_policy_gate":str(SOURCE_POLICY_GATE),
        "dagger_training_source":"QO-14 original TRAIN/VALUE families; QO-17 original occupancy mixture",
        "dagger_validation_source":"QO-14 original VALIDATION families; QO-17 original occupancy mixture",
        "calibration_seeds":list(CALIBRATION_SEEDS),
        "calibration_policies":list(CALIBRATION_SOURCE_POLICIES),
        "calibration_checkpoints":list(CALIBRATION_CHECKPOINTS),
        "threshold_candidates":[str(x) for x in GATE_CANDIDATES],
        "teacher_positive_gate":str(TEACHER_GATE),
        "maximum_validation_false_engagement_rate":str(MAX_FALSE_ENGAGEMENT),
        "minimum_validation_recall":str(MIN_TRUE_RECALL),
        "calibration_selection":"maximize validation recall among feasible; tie choose smaller threshold; if none feasible fall back all",
        "hard_negative_max_unique":NEGATIVE_OVERSAMPLE_MAX_UNIQUE,
        "hard_negative_copies":NEGATIVE_OVERSAMPLE_COPIES,
        "hard_negative_definition":"max exact QO-12 transformed first-action value <1/20 on eligible training-family student-visited state",
        "student_architecture":"unchanged 138-parameter 2-token 8-hidden tanh RNN",
        "student_loss":"QO-17 fixed MSE + margin softplus + engagement BCE, unchanged",
        "fit":"QO-17 65-epoch seeded Adam and validation-only early checkpoint",
        "arms":list(ARMS),"primary_policy":PRIMARY_ARM,
        "primary_budget":160,
        "primary_metric":"paired average QO-07 robust-optimistic gap reduction vs bottleneck",
        "bootstrap_reps":BOOTSTRAP_REPS,
        "bootstrap_seed":ORACLE_SEED_BOOTSTRAP,
        "readiness_min_per_cell":8,
        "certificate_source":"unchanged QO-07",
        "teacher_source":"exact QO-12 two-step expected gap",
    }

def contract_sha256():
    return sha256(json.dumps(contract(),sort_keys=True,
        separators=(",",":")).encode("ascii")).hexdigest()

def frozen_supervised_ledgers():
    """Only called after training and validation gate selection."""
    rows,errors=[],[]
    for idx,rates in enumerate(FROZEN_RATES):
        for seed in FROZEN_SUPERVISED_SEEDS:
            for sched in SCHEDULES:
                ledger=ProbeLedger(2,1,ALPHA,BISECTION_BITS)
                streams=[Random(1000003*seed+100019*idx+9791*i+
                                8599*SCHEDULES.index(sched)) for i in (0,1)]
                for t in range(1,SUPERVISED_CHECKPOINTS[-1]+1):
                    i,a=probe_choice(sched,t-1)
                    try:
                        ledger.observe(i,a,int(streams[i].random()<float(rates[i])))
                    except ValueError:
                        errors.append((idx,seed,sched,t))
                        break
                    if t in SUPERVISED_CHECKPOINTS:
                        rows.append((idx,seed,sched,t,deepcopy(ledger)))
    return tuple(rows),tuple(errors)

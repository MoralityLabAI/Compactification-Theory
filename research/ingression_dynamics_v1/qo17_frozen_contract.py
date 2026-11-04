"""QO-17 independently frozen test contract.

This is the FIRST QO-17 experiment commit, prior to any QO-17 fitting,
evaluation, or model choice. Import in the reporting phase only.
New rate/seed families are disjoint from QO-14, QO-15 and QO-16 tests.
"""
from fractions import Fraction as F
from hashlib import sha256
import json
from random import Random
from copy import deepcopy

from research.ingression_dynamics_v1.certificate_student_data import (
    ALPHA, BISECTION_BITS, TOLERANCE, TRAIN_RATES, VALIDATION_RATES,
    TEST_RATES, TRAIN_SEEDS, VALIDATION_SEEDS, TEST_SEEDS,
    SCHEDULES, probe_choice,
)
from research.ingression_dynamics_v1.certificate_boundary_test_contract import (
    FROZEN_RATES as QO15_RATES, FROZEN_SUPERVISED_SEEDS as QO15_SUP_SEEDS,
    FROZEN_POLICY_SEEDS as QO15_POL_SEEDS,
)
from research.ingression_dynamics_v1.gap_student_frozen_test import (
    FROZEN_RATES as QO16_RATES, FROZEN_SUPERVISED_SEEDS as QO16_SUP_SEEDS,
    FROZEN_POLICY_SEEDS as QO16_POL_SEEDS,
)
from research.ingression_dynamics_v1.anytime_active_quotient import ProbeLedger

FROZEN_RATES=(
    (F(1,80),F(79,80)),
    (F(9,80),F(71,80)),
    (F(17,80),F(63,80)),
    (F(25,80),F(55,80)),
    (F(31,80),F(49,80)),
    (F(39,80),F(41,80)),
    (F(53,80),F(27,80)),
    (F(67,80),F(13,80)),
)
FROZEN_SUPERVISED_SEEDS=(3503,3517,3529)
FROZEN_POLICY_SEEDS=(3701,3719,3733)
SUPERVISED_CHECKPOINTS=(24,40,64,96)
POLICY_CHECKPOINTS=(32,80,160)
VISITED_CAL_CHECKPOINTS=(24,32,48,80,112,160)
ARMS=(
    "bottleneck",
    "exact_gap_h2",
    "qo16_linear_fixed_mse",
    "qo16_recurrent_fixed_mse",
    "linear_occupancy_mse",
    "recurrent_occupancy_mse",
    "linear_occupancy_margin_gate",
    "recurrent_occupancy_margin_gate",
)
MAIN_STUDENT="recurrent_occupancy_margin_gate"
PRIMARY_BUDGET=160
TRAIN_OCCUPANCY_SEEDS=(3101,3109)
VALIDATION_OCCUPANCY_SEEDS=(3301,3313)
TRAINING_OCC_CHECKPOINTS=(16,24,32,40,56,72,96,128)
TEACHER_ENGAGEMENT_GATE=F(1,20)
READINESS_MIN_PER_CELL=8
LOSS_COEFFICIENTS={
    "q_mse":1.0,
    "rank":2.0,
    "rank_temperature":0.10,
    "gate_bce":0.015,
    "gate_temperature":0.025,
}
OPTIMIZER={
    "epochs":65,
    "adam_rate":0.012,
    "clip_norm":3.0,
    "initialization_seed":20261104,
    "loss_seed":20261104,
}
BOOTSTRAP_REPS=2000
BOOTSTRAP_SEED=20261104
FROZEN_SCENARIOS=tuple({
    "id":"qo17_independent_"+str(i),
    "rates":((rate[0],),(rate[1],)),
    "classes":1,"tolerance":TOLERANCE,
} for i,rate in enumerate(FROZEN_RATES))

assert len(FROZEN_RATES)==len(set(FROZEN_RATES))==8
assert not (set(FROZEN_RATES) &
    (set(TRAIN_RATES)|set(VALIDATION_RATES)|set(TEST_RATES)|
     set(QO15_RATES)|set(QO16_RATES)))
_old_seeds=(set(TRAIN_SEEDS)|set(VALIDATION_SEEDS)|set(TEST_SEEDS)|
    set(QO15_SUP_SEEDS)|set(QO15_POL_SEEDS)|
    set(QO16_SUP_SEEDS)|set(QO16_POL_SEEDS))
assert not ((set(FROZEN_SUPERVISED_SEEDS)|set(FROZEN_POLICY_SEEDS)|
    set(TRAIN_OCCUPANCY_SEEDS)|set(VALIDATION_OCCUPANCY_SEEDS)) & _old_seeds)
assert len(set(FROZEN_SUPERVISED_SEEDS)|set(FROZEN_POLICY_SEEDS)|
    set(TRAIN_OCCUPANCY_SEEDS)|set(VALIDATION_OCCUPANCY_SEEDS))==10

def frozen_contract():
    return {
        "rates":[[str(x) for x in pair] for pair in FROZEN_RATES],
        "supervised_seeds":list(FROZEN_SUPERVISED_SEEDS),
        "policy_seeds":list(FROZEN_POLICY_SEEDS),
        "supervised_checkpoints":list(SUPERVISED_CHECKPOINTS),
        "policy_checkpoints":list(POLICY_CHECKPOINTS),
        "visited_calibration_checkpoints":list(VISITED_CAL_CHECKPOINTS),
        "supervised_schedules":list(SCHEDULES),
        "arms":list(ARMS),"primary_student":MAIN_STUDENT,
        "primary_budget":PRIMARY_BUDGET,
        "occupancy_seeds_train":list(TRAIN_OCCUPANCY_SEEDS),
        "occupancy_seeds_validation":list(VALIDATION_OCCUPANCY_SEEDS),
        "occupancy_source_policies":["bottleneck","exact_gap_h2"],
        "occupancy_checkpoints":list(TRAINING_OCC_CHECKPOINTS),
        "train_mixture":"source fixed schedule 1x + occupancy 1x; no duplication beyond source cell-swap",
        "source_heldout":"QO-17 supervised and independent policy test never sampled during fitting",
        "loss_coefficients":LOSS_COEFFICIENTS,
        "optimizer":OPTIMIZER,
        "teacher_gate":str(TEACHER_ENGAGEMENT_GATE),
        "readiness_min_per_cell":READINESS_MIN_PER_CELL,
        "teacher_source":"QO-12 exact two-step gap progress Beta(1,1)",
        "certificate_source":"QO-07 unmodified",
        "bootstrap_repetitions":BOOTSTRAP_REPS,
        "bootstrap_seed":BOOTSTRAP_SEED,
    }

def contract_sha256():
    return sha256(json.dumps(frozen_contract(),sort_keys=True,
                             separators=(",",":")).encode("ascii")).hexdigest()

def generate_frozen_supervised_ledgers():
    """Only called from post-training evaluation, never from fit."""
    rows,failures=[],[]
    for idx,rates in enumerate(FROZEN_RATES):
        for seed in FROZEN_SUPERVISED_SEEDS:
            for sched in SCHEDULES:
                ledger=ProbeLedger(2,1,ALPHA,BISECTION_BITS)
                rngs=[Random(1000003*seed+100019*idx+9791*i+
                             8599*SCHEDULES.index(sched)) for i in range(2)]
                for t in range(1,SUPERVISED_CHECKPOINTS[-1]+1):
                    i,a=probe_choice(sched,t-1)
                    try:
                        ledger.observe(i,a,int(rngs[i].random()<float(rates[i])))
                    except ValueError:
                        failures.append((idx,seed,sched,t))
                        break
                    if t in SUPERVISED_CHECKPOINTS:
                        rows.append((idx,seed,sched,t,deepcopy(ledger)))
    return tuple(rows),tuple(failures)

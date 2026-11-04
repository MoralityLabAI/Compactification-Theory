"""QO-16: independently preregistered test contract, committed before training.

These sources are inaccessible to the QO-16 trainer until after all fit()
calls and validation-only checkpoint selection. Every test ledger uses real
QO-07 source-order confidence updates. Rate parameters drive simulation
only and are never model inputs. Neither QO-14 nor QO-15 test families
or seeds are recycled for the primary QO-16 evaluation.
"""
from fractions import Fraction as F
from hashlib import sha256
import json
from random import Random

from research.ingression_dynamics_v1.certificate_student_data import (
    ALPHA, BISECTION_BITS, TOLERANCE, TRAIN_RATES, VALIDATION_RATES,
    TEST_RATES, TRAIN_SEEDS, VALIDATION_SEEDS, TEST_SEEDS,
    SCHEDULES, probe_choice,
)
from research.ingression_dynamics_v1.certificate_boundary_test_contract import (
    FROZEN_RATES as QO15_TEST_RATES,
    FROZEN_SUPERVISED_SEEDS as QO15_TEST_SEEDS,
    FROZEN_POLICY_SEEDS as QO15_POLICY_SEEDS,
)
from research.ingression_dynamics_v1.anytime_active_quotient import ProbeLedger

FROZEN_RATES = (
    (F(3,100), F(97,100)),
    (F(11,100), F(89,100)),
    (F(19,100), F(81,100)),
    (F(27,100), F(73,100)),
    (F(33,100), F(67,100)),
    (F(43,100), F(57,100)),
    (F(57,100), F(43,100)),
    (F(87,100), F(13,100)),
)
FROZEN_SUPERVISED_SEEDS = (2309, 2311, 2333)
FROZEN_POLICY_SEEDS = (2503, 2521, 2539)
FROZEN_SUPERVISED_CHECKPOINTS = (24, 40, 64, 96)
FROZEN_POLICY_CHECKPOINTS = (32, 80, 160)
FROZEN_SCENARIOS = tuple({
    "id": "qo16_frozen_pair_" + str(i),
    "rates": ((pair[0],), (pair[1],)),
    "classes": 1, "tolerance": TOLERANCE,
    "origin": "independent QO-16 registered synthetic family",
} for i, pair in enumerate(FROZEN_RATES))

assert len(set(FROZEN_RATES)) == len(FROZEN_RATES)
assert not (set(FROZEN_RATES) & (
    set(TRAIN_RATES) | set(VALIDATION_RATES) | set(TEST_RATES) |
    set(QO15_TEST_RATES)))
_old_seeds = (
    set(TRAIN_SEEDS) | set(VALIDATION_SEEDS) | set(TEST_SEEDS) |
    set(QO15_TEST_SEEDS) | set(QO15_POLICY_SEEDS))
assert not (set(FROZEN_SUPERVISED_SEEDS) & _old_seeds)
assert not (set(FROZEN_POLICY_SEEDS) & _old_seeds)
assert not (set(FROZEN_POLICY_SEEDS) & set(FROZEN_SUPERVISED_SEEDS))

def frozen_contract():
    return {
        "rates": [[str(p) for p in pair] for pair in FROZEN_RATES],
        "supervised_seeds": list(FROZEN_SUPERVISED_SEEDS),
        "policy_seeds": list(FROZEN_POLICY_SEEDS),
        "supervised_checkpoints": list(FROZEN_SUPERVISED_CHECKPOINTS),
        "policy_checkpoints": list(FROZEN_POLICY_CHECKPOINTS),
        "supervised_schedules": list(SCHEDULES),
        "confidence_alpha": str(ALPHA),
        "confidence_bisection_bits": BISECTION_BITS,
        "tolerance": str(TOLERANCE),
        "two_step_horizon": 2,
        "beta_prior": "Beta(1,1)",
        "student_gain_transform": "gain/(gain+1/64)",
        "readiness_min_per_cell": 8,
        "normalized_gain_gate": "1/20",
        "teacher_action_tie": "maximum width then minimum count then cell ID",
        "paired_cluster_bootstrap": 2000,
        "bootstrap_seed": 20261104,
        "evaluation_arm_names": [
            "bottleneck", "exact_qo12_gap_h2",
            "linear_gap_student", "recurrent_gap_student",
        ],
    }

def frozen_contract_sha256():
    return sha256(json.dumps(frozen_contract(),sort_keys=True,
                             separators=(",",":")).encode("ascii")).hexdigest()

def generate_frozen_ledgers():
    """Supply public state snapshots; label them in evaluation code only."""
    from copy import deepcopy
    rows=[]
    failures=[]
    for idx,rates in enumerate(FROZEN_RATES):
        for seed in FROZEN_SUPERVISED_SEEDS:
            for sched in SCHEDULES:
                ledger=ProbeLedger(2,1,ALPHA,BISECTION_BITS)
                rngs=[
                    Random(1000003*seed + 100019*idx + 9791*i +
                           8599*SCHEDULES.index(sched))
                    for i in range(2)
                ]
                for t in range(1,FROZEN_SUPERVISED_CHECKPOINTS[-1]+1):
                    i,a=probe_choice(sched,t-1)
                    try:
                        ledger.observe(i,a,int(
                            rngs[i].random()<float(rates[i])))
                    except ValueError:
                        failures.append((idx,seed,sched,t))
                        break
                    if t in FROZEN_SUPERVISED_CHECKPOINTS:
                        rows.append((idx,seed,sched,t,deepcopy(ledger)))
    return tuple(rows),tuple(failures)

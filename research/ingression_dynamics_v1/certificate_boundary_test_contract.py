"""QO-15 source-frozen, independent evaluation family.

Declared before QO-15 student training or result inspection. Bernoulli
rates and seeds in this module are held out from QO-14 source families
and all QO-15 augmentation and validation generators. The actual ledger
and QO-07 certificate implementation are unchanged.
"""
from fractions import Fraction as Q
from hashlib import sha256
import json
from random import Random

from research.ingression_dynamics_v1.anytime_active_quotient import ProbeLedger
from research.ingression_dynamics_v1.certificate_student_data import (
    ALPHA, BISECTION_BITS, TOLERANCE, CHECKPOINTS, TRAIN_RATES,
    VALIDATION_RATES, TEST_RATES, TRAIN_SEEDS, VALIDATION_SEEDS, TEST_SEEDS,
    SCHEDULES, probe_choice, training_example,
)

FROZEN_RATES = (
    (Q(7, 40), Q(33, 40)),
    (Q(9, 40), Q(31, 40)),
    (Q(11, 40), Q(29, 40)),
    (Q(13, 40), Q(27, 40)),
    (Q(17, 40), Q(23, 40)),
    (Q(23, 40), Q(17, 40)),
    (Q(29, 40), Q(11, 40)),
    (Q(33, 40), Q(7, 40)),
)
FROZEN_SUPERVISED_SEEDS = (1601, 1607, 1613)
FROZEN_POLICY_SEEDS = (1709, 1721, 1733)
FROZEN_POLICY_CHECKPOINTS = (32, 56, 80)
FROZEN_SCENARIOS = tuple({
    "id": "qo15_frozen_pair_" + str(i),
    "rates": ((rates[0],), (rates[1],)),
    "classes": 1,
    "tolerance": TOLERANCE,
    "origin": "QO-15 independently registered held-out family",
} for i, rates in enumerate(FROZEN_RATES))

assert len(set(FROZEN_RATES)) == len(FROZEN_RATES)
assert not (set(FROZEN_RATES) & set(TRAIN_RATES))
assert not (set(FROZEN_RATES) & set(VALIDATION_RATES))
assert not (set(FROZEN_RATES) & set(TEST_RATES))
assert not (set(FROZEN_SUPERVISED_SEEDS) & (
    set(TRAIN_SEEDS) | set(VALIDATION_SEEDS) | set(TEST_SEEDS)))
assert not (set(FROZEN_POLICY_SEEDS) & (
    set(TRAIN_SEEDS) | set(VALIDATION_SEEDS) | set(TEST_SEEDS) |
    set(FROZEN_SUPERVISED_SEEDS)))


def frozen_contract():
    """Stable, inspectable configuration identity independent of model."""
    return {
        "rates": [[str(x) for x in pair] for pair in FROZEN_RATES],
        "supervised_seeds": list(FROZEN_SUPERVISED_SEEDS),
        "policy_seeds": list(FROZEN_POLICY_SEEDS),
        "supervised_schedules": list(SCHEDULES),
        "supervised_checkpoints": list(CHECKPOINTS),
        "policy_checkpoints": list(FROZEN_POLICY_CHECKPOINTS),
        "teacher_horizon": 2,
        "certificate_alpha": str(ALPHA),
        "certificate_tolerance": str(TOLERANCE),
        "readiness_min_per_cell": 8,
        "action_value_gate": 0.05,
        "baseline_weights": "QO-14 checked-in receipt",
    }


def frozen_contract_digest():
    raw = json.dumps(frozen_contract(), sort_keys=True,
                     separators=(",", ":")).encode("ascii")
    return sha256(raw).hexdigest()


def generate_frozen_supervised():
    """Frozen rate/seed/schedule streams, independent from QO-15 training.

    True rates control the simulator only. Every teacher target is
    calculated from the public evidence ledger.
    """
    rows = []
    failures = []
    for idx, rate in enumerate(FROZEN_RATES):
        for seed in FROZEN_SUPERVISED_SEEDS:
            for schedule in SCHEDULES:
                ledger = ProbeLedger(2, 1, ALPHA, BISECTION_BITS)
                rngs = [
                    Random(1000003 * seed + 100019 * idx + 9791 * i +
                           8599 * SCHEDULES.index(schedule))
                    for i in range(2)
                ]
                for t in range(1, max(CHECKPOINTS) + 1):
                    i, a = probe_choice(schedule, t-1)
                    bit = int(rngs[i].random() < float(rate[i]))
                    try:
                        ledger.observe(i, a, bit)
                    except ValueError:
                        failures.append((idx, seed, schedule, t))
                        break
                    if t in CHECKPOINTS:
                        ex = training_example(
                            ledger, "qo15_frozen_test", idx, seed, schedule, t)
                        if ex["q"] != (Q(1), Q(1)):
                            rows.append(ex)
    return tuple(rows), tuple(failures)

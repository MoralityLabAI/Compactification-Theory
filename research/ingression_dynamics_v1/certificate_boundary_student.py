"""QO-15: deterministic boundary-enrichment and paired probe evaluation.

Frozen held-out contract is imported only inside full_report after fitting.
All teacher labels are QO-13 exact-rational two-step reachability values.
The existing QO-07 confidence engine and QO-14 RNN architecture are reused.
"""
from copy import deepcopy
from fractions import Fraction as Q
from hashlib import sha256
import json
from pathlib import Path
from random import Random

from research.ingression_dynamics_v1.anytime_active_quotient import (
    ProbeLedger, certificate
)
from research.ingression_dynamics_v1.certificate_reachability import (
    certification_status
)
from research.ingression_dynamics_v1.certificate_student_data import (
    TRAIN_RATES, TRAIN_SEEDS, VALIDATION_RATES, VALIDATION_SEEDS,
    SCHEDULES, ALPHA, BISECTION_BITS, TOLERANCE, CHECKPOINTS,
    generate_split, training_example, probe_choice, split_signature
)
from research.ingression_dynamics_v1.tiny_recurrent_certificate_student import (
    create_rnn, fit, forward, from_receipt, inference_multiplies
)
from research.ingression_dynamics_v1.certificate_student_pilot import (
    run_student_policy
)
from research.ingression_dynamics_v1.paired_probe_pilot import (
    run_policy as run_bottleneck, verify_common_stream_prefix
)
from research.ingression_dynamics_v1.two_step_gap_probe import planning_state

PARENT_CHECKPOINTS = (32, 40, 48, 56)
AUGMENT_CAP_PER_RATE = {"train": 96, "validation": 64}
INFORMATIVE_REPEATS = {"train": 24, "validation": 8}
ARMS = ("bottleneck", "qo14_recurrent", "qo15_boundary_recurrent",
        "exact_h2_teacher")
ROOT = Path(__file__).resolve().parents[2]
QO14_RECEIPT = ROOT / "results/certificate_student/receipt.json"


def _rate_seed_split(split):
    if split == "train":
        return TRAIN_RATES, TRAIN_SEEDS
    if split == "validation":
        return VALIDATION_RATES, VALIDATION_SEEDS
    raise ValueError("QO-15 frontier generation is train/validation ONLY")


def parent_ledgers(split):
    """Reconstruct source-original Bernoulli histories and retain ledgers."""
    rates, seeds = _rate_seed_split(split)
    for idx, rate in enumerate(rates):
        for seed in seeds:
            for schedule in SCHEDULES:
                ledger = ProbeLedger(2, 1, ALPHA, BISECTION_BITS)
                rngs = [
                    Random(1000003 * seed + 100019 * idx + 9791 * i +
                           8599 * SCHEDULES.index(schedule))
                    for i in range(2)
                ]
                for t in range(1, max(PARENT_CHECKPOINTS) + 1):
                    i, a = probe_choice(schedule, t - 1)
                    hit = int(rngs[i].random() < float(rate[i]))
                    try:
                        ledger.observe(i, a, hit)
                    except ValueError:
                        break
                    if t in PARENT_CHECKPOINTS:
                        yield idx, seed, schedule, t, deepcopy(ledger)


def _successor_states(parent):
    """Enumerate every valid one- and two-step counterfactual history."""
    frontier = [((), parent)]
    for depth in (1, 2):
        next_frontier = []
        for path, state in frontier:
            for i in (0, 1):
                for bit in (0, 1):
                    changed = deepcopy(state)
                    try:
                        changed.observe(i, 0, bit)
                    except ValueError:
                        continue
                    new_path = path + ((i, bit),)
                    next_frontier.append((new_path, changed))
                    yield depth, new_path, changed
        frontier = next_frontier


def _frontier_distance(ledger):
    """Low-cost public frontier score, independent of both Q labels."""
    if min(ledger.counts[i][0] for i in range(2)) < 8:
        return None
    decision, upper, lower, _ = certification_status(
        ledger.snapshot(), 1, TOLERANCE)
    if decision != "undetermined":
        return None
    return min(abs(upper - TOLERANCE), abs(lower - TOLERANCE))


def frontier_examples(split):
    """Public-only top-K selection; Q labels computed after selection.

    Stratification is per underlying training/validation rate family,
    never across the frozen independent test rates.
    """
    rates, _ = _rate_seed_split(split)
    by_rate = {i: {} for i in range(len(rates))}
    seen = 0
    invalid = 0
    for idx, seed, schedule, t, parent in parent_ledgers(split):
        for depth, path, ledger in _successor_states(parent):
            seen += 1
            distance = _frontier_distance(ledger)
            if distance is None:
                invalid += 1
                continue
            key = planning_state(ledger)
            ordering = (distance, seed, SCHEDULES.index(schedule),
                        t, depth, path)
            current = by_rate[idx].get(key)
            if current is None or ordering < current[0]:
                by_rate[idx][key] = (ordering, ledger, seed, schedule, t)
    examples = []
    for idx in range(len(rates)):
        ranked = sorted(by_rate[idx].values(), key=lambda row: row[0])
        for ordering, ledger, seed, schedule, t in (
                ranked[:AUGMENT_CAP_PER_RATE[split]]):
            example = training_example(
                ledger, split + "_frontier", idx, seed, schedule, t)
            if example["q"] != (Q(1), Q(1)):
                examples.append(example)
    meta = {
        "split": split,
        "counterfactual_candidates": seen,
        "excluded_or_unready": invalid,
        "unique_eligible_by_rate": [
            len(by_rate[i]) for i in range(len(rates))
        ],
        "selected_by_rate": [
            min(len(by_rate[i]), AUGMENT_CAP_PER_RATE[split])
            for i in range(len(rates))
        ],
        "selected": len(examples),
        "informative": sum(x["q"][0] != x["q"][1] for x in examples),
        "positive_value": sum(max(x["q"]) > 0 for x in examples),
        "exact_teacher_labels": len(examples),
    }
    return tuple(examples), meta


def swap_action_cells(example):
    """Exact Q and public-feature equivariance under cell relabeling."""
    switched = dict(example)
    switched["tokens"] = (example["tokens"][1], example["tokens"][0])
    switched["q"] = (example["q"][1], example["q"][0])
    switched["teacher_action"] = (
        0 if switched["q"][0] >= switched["q"][1] else 1)
    switched["permuted"] = True
    return switched


def augmented_split(split):
    """Fixed resampling keeps Q labels exact, with unchanged MSE trainer."""
    original, failures = generate_split(split)
    original = tuple(e for e in original if e["q"] != (Q(1), Q(1)))
    frontier, meta = frontier_examples(split)
    swapped = tuple(swap_action_cells(e) for e in frontier)
    informative = tuple(e for e in frontier if e["q"][0] != e["q"][1])
    informative_swapped = tuple(swap_action_cells(e) for e in informative)
    repeat = INFORMATIVE_REPEATS[split]
    result = (original + frontier + swapped +
              (informative + informative_swapped) * repeat)
    if not result:
        raise AssertionError("empty augmented training/validation split")
    meta.update({
        "base_examples": len(original),
        "base_informative": sum(e["q"][0] != e["q"][1] for e in original),
        "source_stream_failures": list(failures),
        "augmented_rows_after_oversampling": len(result),
        "enrichment_repeat_multiplier": repeat,
        "cell_swap_examples": len(swapped),
        "informative_oversampled_rows": 2 * repeat * len(informative),
    })
    return result, meta


def margin_metrics(model, examples):
    """Exact observable regret with the rare informative denominator kept."""
    n = len(examples)
    if not n:
        raise ValueError("test cannot be empty")
    relevant = errors = 0
    total_regret = total_informative_regret = 0.0
    total_margin_error = total_abs_q_error = 0.0
    wrong_engagement = too_eager = too_quiet = 0
    value_buckets = {"zero": 0, "0_to_0.001": 0, "0.001_to_0.01": 0,
                     "0.01_to_0.1": 0, "gt_0.1": 0}
    for ex in examples:
        true = tuple(map(float, ex["q"]))
        pred = forward(model, ex["tokens"])
        delta = true[0] - true[1]
        pred_delta = pred[0] - pred[1]
        margin = abs(delta)
        action = 0 if pred_delta >= 0 else 1
        regret = max(true) - true[action]
        if delta != 0:
            relevant += 1
            errors += int(action != (0 if delta >= 0 else 1))
            total_informative_regret += regret
        total_regret += regret
        total_margin_error += abs(delta - pred_delta)
        total_abs_q_error += (abs(true[0]-pred[0])+abs(true[1]-pred[1])) / 2
        true_engage = max(true) >= 0.05
        pred_engage = max(pred) >= 0.05
        wrong_engagement += int(true_engage != pred_engage)
        too_eager += int(pred_engage and not true_engage)
        too_quiet += int(true_engage and not pred_engage)
        if margin == 0:
            value_buckets["zero"] += 1
        elif margin <= 0.001:
            value_buckets["0_to_0.001"] += 1
        elif margin <= 0.01:
            value_buckets["0.001_to_0.01"] += 1
        elif margin <= 0.1:
            value_buckets["0.01_to_0.1"] += 1
        else:
            value_buckets["gt_0.1"] += 1
        if regret > abs(delta-pred_delta) + 1e-12:
            raise AssertionError("two-action margin-regret lemma violated")
    return {
        "examples": n, "informative": relevant,
        "informative_action_disagreements": errors,
        "conditional_teacher_regret": (
            total_informative_regret/relevant if relevant else None),
        "mean_teacher_regret": total_regret/n,
        "mean_absolute_q_error": total_abs_q_error/n,
        "mean_margin_error": total_margin_error/n,
        "engagement_errors": wrong_engagement,
        "false_engagements": too_eager,
        "missed_engagements": too_quiet,
        "teacher_margin_buckets": value_buckets,
    }


def _model_from_qo14():
    if not QO14_RECEIPT.is_file():
        raise FileNotFoundError("QO-14 reference receipt must be checked in")
    raw = QO14_RECEIPT.read_bytes()
    original = json.loads(raw.decode("ascii"))
    return from_receipt(original["models"]["recurrent_student"]), sha256(raw).hexdigest()


def full_report():
    """Fit without visiting held-out generator, then score frozen evaluation."""
    baseline, baseline_digest = _model_from_qo14()
    train, training_meta = augmented_split("train")
    val, validation_meta = augmented_split("validation")
    fitted = fit(create_rnn(), train, val)

    # Independent test data and evaluation code are first accessed only here.
    from research.ingression_dynamics_v1.certificate_boundary_test_contract import (
        FROZEN_SCENARIOS, FROZEN_POLICY_SEEDS,
        FROZEN_POLICY_CHECKPOINTS, generate_frozen_supervised,
        frozen_contract, frozen_contract_digest,
    )
    frozen, frozen_failures = generate_frozen_supervised()
    models = {
        "qo14_recurrent": baseline,
        "qo15_boundary_recurrent": fitted["weights"],
    }
    supervised = {
        arm: margin_metrics(weights, frozen) for arm, weights in models.items()
    }
    rows, traces, costs = [], [], {}
    for ci, case in enumerate(FROZEN_SCENARIOS):
        for seed in FROZEN_POLICY_SEEDS:
            for arm in ARMS:
                if arm == "bottleneck":
                    policy_rows, history = run_bottleneck(
                        ci, case, seed, "bottleneck",
                        FROZEN_POLICY_CHECKPOINTS)
                    counters = {}
                elif arm == "exact_h2_teacher":
                    policy_rows, history, counters = run_student_policy(
                        ci, case, seed, "exact_h2_teacher", models,
                        FROZEN_POLICY_CHECKPOINTS)
                else:
                    policy_rows, history, counters = run_student_policy(
                        ci, case, seed, "recurrent_student",
                        {"recurrent_student": models[arm]},
                        FROZEN_POLICY_CHECKPOINTS)
                if len(history) != FROZEN_POLICY_CHECKPOINTS[-1]:
                    raise AssertionError("incomplete paired trajectory")
                if not verify_common_stream_prefix(ci, case, seed, history):
                    raise AssertionError("independent streams were not paired")
                for row in policy_rows:
                    row = dict(row)
                    row["policy"] = arm
                    # QO-14 bottleneck report uses a different gap key;
                    # keep the certificate's exact robust gap consistently.
                    if "certification_gap" not in row:
                        row["certification_gap"] = str(
                            Q(row["robust_upper"]) -
                            Q(row["true_optimum_lower"]))
                    rows.append(row)
                traces.append({
                    "scenario": case["id"], "seed": seed, "policy": arm,
                    "queries": len(history),
                    "sha256": sha256(json.dumps(
                        history, separators=(",", ":")).encode("ascii")).hexdigest()
                })
                total = costs.setdefault(
                    arm, {"teacher_q_labels": 0, "student_inferences": 0,
                          "student_multiply_accumulates": 0,
                          "chosen_actions": 0, "low_value_fallback": 0})
                for k in total:
                    total[k] += counters.get(k, 0)
    groups = []
    for checkpoint in FROZEN_POLICY_CHECKPOINTS:
        for arm in ARMS:
            subset = [r for r in rows if r["queries"] == checkpoint and
                      r["policy"] == arm]
            assert len(subset) == len(FROZEN_SCENARIOS) * len(FROZEN_POLICY_SEEDS)
            groups.append({
                "checkpoint": checkpoint, "policy": arm, "n": len(subset),
                "correct_certificates": sum(
                    r["decision"] != "undetermined" and not r["false_cert"]
                    for r in subset),
                "wrong_certificates": sum(r["false_cert"] for r in subset),
                "coverage_failures": sum(
                    not r["true_rates_covered"] for r in subset),
                "mean_certification_gap": str(
                    sum((Q(r["certification_gap"]) for r in subset),
                        Q(0)) / len(subset)),
                "mean_true_frozen_error": str(
                    sum((Q(r["true_frozen_error"]) for r in subset),
                        Q(0)) / len(subset)),
            })
    return {
        "contract": "QO-15 frozen boundary distillation",
        "frozen_contract": frozen_contract(),
        "frozen_contract_sha256": frozen_contract_digest(),
        "qo14_receipt_sha256": baseline_digest,
        "training": training_meta,
        "validation": validation_meta,
        "selected_epoch": fitted["best_epoch"],
        "training_history": fitted["history"],
        "training_updates": fitted["update_steps"],
        "trained_weights": fitted["weights"],
        "supervised_test_metrics": supervised,
        "frozen_test_examples": len(frozen),
        "frozen_test_failures": list(frozen_failures),
        "frozen_test_fingerprint": sha256(json.dumps(
            [(split_signature(e), [str(v) for v in e["q"]])
             for e in frozen], separators=(",", ":")).encode("ascii")).hexdigest(),
        "arms": list(ARMS),
        "policy_summaries": groups,
        "policy_compute": costs,
        "policy_rows": rows,
        "trajectory_hashes": traces,
        "counts": {
            "trajectories": len(traces),
            "snapshots": len(rows),
            "queries": sum(t["queries"] for t in traces),
        },
        "limitations": (
            "Digital two-state one-action problem; horizon-2 planning "
            "teacher; recurrent policy has no proven uniform approximation "
            "bound, and held-out confidence coverage is empirical only."
        ),
    }

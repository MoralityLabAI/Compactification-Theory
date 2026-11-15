"""NOR-14: train-residualised basin nonlinearities vs recurrent-trajectory cues.

All gates, projections and readout weights use public source features only.
Projection coefficients are fitted on the 96 original training examples.
The original 1000-parameter GRU, 16-class head and NOR-11 basin head are
immutable. Future degree-256 truth is used only in development selection
and final evaluation. Four exact public template checks remain the budget.

Crucial scope: the endpoint basin probability is a deterministic function
of the twelve endpoint hidden coordinates. It has zero conditional mutual
information given the full hidden state. Residualising it against a *linear*
hidden-state span tests nonlinear feature bias, not new information.
The step-four recurrent trajectory is a separate observable which need not
be determined by the endpoint hidden state.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path
import random
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.rrc_compactification import compound25 as corpus, model25
from research.nor11_attractor_controller import controller as prior_probe
from research.nor12_task_readout import readout as old
from research.nor12_task_readout import independent_confirmation as earlier
from research.nor13_gated_readout import interaction as nor13

SCHEMA = "nor14.train_orthogonal_nonlinear_basin_vs_trajectory.v1"
KINDS = ("endpoint_basin", "endpoint_shuffled", "endpoint_random",
         "trajectory_basin", "trajectory_random", "hidden_only")
SLOT_WIDTH = 25
CLASSES = 16
EPOCHS = 150
LR = .023
RIDGE = .075
SCALES = (.25, .5, 1., 2.)
SEED = 2026111514
NEW_TEST_SEED_OFFSET = 92517000
REPETITIONS = 4
PER_FAMILY = 12
TRAIN_ROWS = 96
DEV_ROWS = 36
PUBLIC_TEMPLATES = 16


def trace_at_four(case, header, checkpoint):
    """Exact portable GRU recurrence over nine publicly visible feature rows."""
    h = [0.] * model25.HIDDEN
    s = checkpoint["state_dict"]
    step4 = None
    xs = corpus.features(case, header)
    assert len(xs) == 9
    for t, row in enumerate(xs, 1):
        a = model25.affine(s["rnn.weight_ih_l0"], s["rnn.bias_ih_l0"], row)
        v = model25.affine(s["rnn.weight_hh_l0"], s["rnn.bias_hh_l0"], h)
        new_h = []
        for i in range(model25.HIDDEN):
            reset = model25.sigmoid(a[i] + v[i])
            update = model25.sigmoid(a[i + model25.HIDDEN] +
                                    v[i + model25.HIDDEN])
            candidate = math.tanh(a[i + 2 * model25.HIDDEN] +
                                  reset * v[i + 2 * model25.HIDDEN])
            new_h.append((1. - update) * candidate + update * h[i])
        h = new_h
        if t == 4:
            step4 = h[:]
    assert step4 is not None
    return step4, h


def fixed_negative_projection(probe):
    """Frozen feature-only random sigmoid with the basin head's weight norm."""
    rng = random.Random(SEED)
    r = [rng.gauss(0, 1) for _ in range(12)]
    n = math.sqrt(math.fsum(x*x for x in r))
    target = math.sqrt(math.fsum(x*x for x in probe["weights"][:12]))
    return [v * target / n for v in r] + [probe["weights"][12]]


def synthetic_gate(z, coefficients):
    logit = math.fsum(x*w for x, w in zip(z + [1.], coefficients, strict=True))
    return math.tanh(.5 * logit)


def attach_public_gates(rows, checkpoint, probe, negative):
    random_weights = fixed_negative_projection(probe)
    for row in rows:
        early, end = trace_at_four(row["case"], row["header"], checkpoint)
        assert max(abs(a-b) for a, b in zip(end, row["h"], strict=True)) < 1e-11
        assert prior_probe.public_digest(row["case"], row["header"]) == row["digest"]
        end_z = row["x"]["hidden_only"][:12]
        early_z = prior_probe.features(early, probe["mean"], probe["scale"])[:12]
        row["nor14_gate"] = {
            "endpoint_basin": 2 * row["p"] - 1,
            "endpoint_shuffled": row["x"]["shuffled_basin"][12],
            "endpoint_random": synthetic_gate(end_z, random_weights),
            "trajectory_basin": 2 * prior_probe.head_score(early, probe) - 1,
            "trajectory_random": synthetic_gate(early_z, random_weights),
            "hidden_only": 0.,
        }
        assert all(-1.000000001 <= v <= 1.000000001
                   for v in row["nor14_gate"].values())


def linear_features(row):
    z = row["x"]["hidden_only"][:12]
    assert len(z) == 12
    return z + [1.]


def orthonormal_column_span(design):
    """Twice-reorthogonalised QR, retaining coordinates in the 13-column span."""
    if not design or any(len(x) != 13 for x in design):
        raise ValueError("expected a nonempty twelve-hidden-plus-bias design")
    basis = []
    for j in range(13):
        v = [x[j] for x in design]
        coef = [float(k == j) for k in range(13)]
        for _ in range(2):
            for q, qc in basis:
                dot = math.fsum(a*b for a, b in zip(v, q, strict=True))
                v = [a - dot*b for a, b in zip(v, q, strict=True)]
                coef = [a - dot*b for a, b in zip(coef, qc, strict=True)]
        norm = math.sqrt(math.fsum(x*x for x in v))
        if norm > 1e-9:
            basis.append(([a/norm for a in v], [a/norm for a in coef]))
    return basis


def gate(row, kind):
    if kind not in KINDS:
        raise ValueError("unknown gate family")
    return row["nor14_gate"][kind]


def fit_projection(train, kind):
    """Remove the 13-dimensional LINEAR span from every gated product.

    All coefficients and scale factors are frozen from TRAIN. This is a
    finite-design linear projection, not statistical conditional independence.
    """
    if kind not in KINDS or len(train) < 13:
        raise ValueError("invalid train-only projection")
    xs = [linear_features(row) for row in train]
    basis = orthonormal_column_span(xs)
    coeffs = []
    scales = []
    worst_moment = 0.
    for k in range(12):
        vec = [gate(row, kind) * x[k] for row, x in zip(train, xs, strict=True)]
        beta = [0.] * 13
        for q, qc in basis:
            inner = math.fsum(a*b for a, b in zip(vec, q, strict=True))
            beta = [a + inner*b for a, b in zip(beta, qc, strict=True)]
        residual = [v - math.fsum(a*b for a, b in zip(x, beta, strict=True))
                    for v, x in zip(vec, xs, strict=True)]
        rms = math.sqrt(math.fsum(v*v for v in residual) / len(train))
        scales.append(max(.05, rms))
        coeffs.append(beta)
        worst_moment = max(worst_moment, *(
            abs(math.fsum(r*x[j] for r, x in zip(residual, xs, strict=True))
                / len(train)) for j in range(13)))
    if worst_moment > 1e-7:
        raise ArithmeticError("train projection failed linear orthogonality")
    return {"kind": kind, "linear_rank": len(basis),
            "coefficients": coeffs, "normalizers": scales,
            "max_train_linear_moment": worst_moment,
            "fitted_from": "96 original public TRAIN hidden states and frozen gate values"}


def feature_vector(row, projection, substitute=None, remove_residual=False):
    x = linear_features(row)
    if remove_residual:
        return x[:12] + [0.]*12 + [1.]
    g = gate(row, projection["kind"]) if substitute is None else substitute
    if not -1.000000001 <= g <= 1.000000001:
        raise ValueError("gate substitution outside [-1,1]")
    products = []
    for k in range(12):
        beta = projection["coefficients"][k]
        linear_part = math.fsum(a*b for a, b in zip(beta, x, strict=True))
        products.append((g*x[k] - linear_part) /
                        projection["normalizers"][k])
    return x[:12] + products + [1.]


def raw_scores(row, model, scale, substitute=None, remove_residual=False):
    if not 0 <= scale <= 2:
        raise ValueError("amplitude outside frozen development grid")
    w = model["weights"]
    if len(w) != CLASSES or any(len(v) != SLOT_WIDTH for v in w):
        raise ValueError("400-slot model mismatch")
    x = feature_vector(row, model["projection"], substitute, remove_residual)
    return [base + scale*math.fsum(a*b for a, b in zip(coef, x, strict=True))
            for base, coef in zip(row["base"], w, strict=True)]


def training_gradient(row, model):
    scores = raw_scores(row, model, 1.)
    p = old.softmax(scores)
    accepted = row["visible_mask"]
    assert len(accepted) == CLASSES and any(accepted)
    mass = math.fsum(p[j] for j in range(CLASSES) if accepted[j])
    if mass <= 0:
        raise ArithmeticError("zero public accepted-template mass")
    x = feature_vector(row, model["projection"])
    out = []
    for j in range(CLASSES):
        delta = p[j] - (p[j] / mass if accepted[j] else 0.)
        out.append([delta*v for v in x])
    return -math.log(mass), out


def fit(train, kind, epochs=EPOCHS):
    if len(train) != TRAIN_ROWS or not 1 <= epochs <= EPOCHS:
        raise ValueError("locked 96-case public training set required")
    projection = fit_projection(train, kind)
    model = {"kind": kind, "projection": projection,
             "weights": [[0.]*SLOT_WIDTH for _ in range(CLASSES)]}
    momentum = [[0.]*SLOT_WIDTH for _ in range(CLASSES)]
    squared = [[0.]*SLOT_WIDTH for _ in range(CLASSES)]
    trace = []
    for t in range(1, epochs+1):
        gradients = [[0.]*SLOT_WIDTH for _ in range(CLASSES)]
        total = 0.
        for row in train:
            loss, grad = training_gradient(row, model)
            total += loss
            for j in range(CLASSES):
                for k in range(SLOT_WIDTH):
                    gradients[j][k] += grad[j][k]
        for j in range(CLASSES):
            for k in range(SLOT_WIDTH):
                g = gradients[j][k]/len(train) + RIDGE*model["weights"][j][k]
                momentum[j][k] = .9*momentum[j][k] + .1*g
                squared[j][k] = .999*squared[j][k] + .001*g*g
                model["weights"][j][k] -= LR*(momentum[j][k]/(1-.9**t))/(
                    math.sqrt(squared[j][k]/(1-.999**t)) + 1e-8)
        if t in (1, 40, 80, 120, 150):
            trace.append({"epoch": t, "public_accepted_mass_nll": total/len(train)})
    model.update({"slots": 400, "epochs": epochs, "training_trace": trace,
                  "training_full_public_template_verifications_per_head": 1536,
                  "private_future_truth_used_in_fit": False,
                  "original_checkpoint_updated": False})
    return model


def task_eval(rows, model=None, scale=0., replace_gate=None,
              remove_residual=False):
    if model is None:
        assert scale == 0 and replace_gate is None and not remove_residual
        return old.evaluate(rows)
    changed = []
    for row in rows:
        substitute = (row["nor14_gate"][replace_gate] if replace_gate is not None
                      else None)
        changed.append({**row, "base": raw_scores(
            row, model, scale, substitute, remove_residual)})
    return old.evaluate(changed)


def freeze_dev(dev, fitted):
    grid = []
    base = task_eval(dev)
    utility = lambda r: r["future_correct"] - .5*r["false_accept"] - .015*r["paid_checks_total"]
    grid.append({"kind": "original", "scale": 0., "utility": utility(base),
                 "future_correct": base["future_correct"],
                 "false_accept": base["false_accept"],
                 "paid_checks": base["paid_checks_total"]})
    for kind in KINDS:
        for scale in SCALES:
            v = task_eval(dev, fitted[kind], scale)
            grid.append({"kind": kind, "scale": scale, "utility": utility(v),
                         "future_correct": v["future_correct"],
                         "false_accept": v["false_accept"],
                         "paid_checks": v["paid_checks_total"]})
    selections = {
        kind: max((r for r in grid if r["kind"] == kind),
                  key=lambda r: (r["utility"], -r["scale"]))
        for kind in KINDS
    }
    overall = max(grid, key=lambda r: (r["utility"], r["kind"] == "original",
                                       -r["scale"]))
    return selections, overall, grid


def fresh_cohorts(roster, repetitions=REPETITIONS, per_family=PER_FAMILY):
    """Exclude all NOR-11/12/12B/13R visible cases and nonlate input hashes."""
    if not 1 <= repetitions <= 4 or not 1 <= per_family <= 12:
        raise ValueError("frozen independent test budget exceeded")
    prior12 = earlier.fresh_cohorts(
        roster, replicates=earlier.REPETITIONS, per_family=earlier.CASES_PER_FAMILY)
    prior13 = nor13.new_heldout(
        roster, replicates=nor13.CONFIRMATIONS, per_family=nor13.FAMILY_PER)
    known_visible = {nor13.source_fingerprint(c)
                     for split in roster.values() for c, _, _ in split}
    known_feature = {d for split in roster.values() for _, _, d in split}
    for cohort in prior12 + prior13:
        for case, _, digest in cohort:
            known_visible.add(nor13.source_fingerprint(case))
            known_feature.add(digest)
    output = []
    for rep in range(repetitions):
        cohort = []
        for index, family in enumerate(corpus.TEST_FAMILIES):
            rng = random.Random(corpus.SEED + NEW_TEST_SEED_OFFSET +
                                rep*1000003 + index*10007)
            found = 0
            for trial in range(per_family*1800):
                case = corpus.make_case(
                    rng, family, f"nor14-r{rep}-{family}-{trial:05d}")
                headers = corpus.possible_headers(case)
                if len(headers) != 1:
                    continue
                header = headers[0]
                digest = prior_probe.public_digest(case, header)
                vis = nor13.source_fingerprint(case)
                if vis in known_visible:
                    continue
                if family != "late_source" and digest in known_feature:
                    continue
                known_visible.add(vis)
                if family != "late_source":
                    known_feature.add(digest)
                cohort.append((case, header, digest))
                found += 1
                if found == per_family:
                    break
            if found != per_family:
                raise RuntimeError("new cohort generator exhausted: " + family)
        output.append(cohort)
    return output


def paired(a, b, nonlate_only=True):
    """Exact paired successes and feature-group sign counts."""
    diff = defaultdict(int)
    gains = losses = 0
    n = 0
    for x, y in zip(a, b, strict=True):
        assert x["public_features_sha256"] == y["public_features_sha256"]
        assert x["family"] == y["family"]
        if nonlate_only and x["family"] == "late_source":
            continue
        n += 1
        delta = y["future_correct"] - x["future_correct"]
        gains += delta > 0
        losses += delta < 0
        diff[x["public_features_sha256"]] += delta
    positive = sum(v > 0 for v in diff.values())
    negative = sum(v < 0 for v in diff.values())
    return {"cases": n, "unique_public_inputs": len(diff),
            "paired_gains": gains, "paired_losses": losses,
            "net": gains-losses, "positive_unique_groups": positive,
            "negative_unique_groups": negative,
            "unadjusted_exact_group_sign_p": float(earlier.exact_sign_p(
                positive, negative))}


def aggregate(records):
    return {
        "cases": len(records),
        "future_correct": sum(r["future_correct"] for r in records),
        "false_accept": sum(r["false_accept"] for r in records),
        "paid_checks": sum(r["paid_checks"] for r in records),
        "unique_input_groups": len({r["public_features_sha256"] for r in records}),
        "nonlate": {
            "cases": sum(r["family"] != "late_source" for r in records),
            "future_correct": sum(r["future_correct"] for r in records
                                  if r["family"] != "late_source"),
            "false_accept": sum(r["false_accept"] for r in records
                                if r["family"] != "late_source"),
            "unique_input_groups": len({
                r["public_features_sha256"] for r in records
                if r["family"] != "late_source"}),
        }
    }


def run(repetitions=REPETITIONS, per_family=PER_FAMILY, epochs=EPOCHS):
    ck, probe, negative, old_weights, original, design, roster = nor13.frozen_data()
    assert ck["parameters"] == 1000 and old.BASIN_SHA == nor13.READOUT_SHA
    assert len(original["train"]) == TRAIN_ROWS
    assert len(original["dev"]) == DEV_ROWS
    for row in original["train"]:
        assert row["visible_mask"] == old.visible_mask(row)
    for split in ("train", "dev"):
        attach_public_gates(original[split], ck, probe, negative)
    fitted = {kind: fit(original["train"], kind, epochs) for kind in KINDS}
    selected, global_choice, dev_grid = freeze_dev(original["dev"], fitted)
    cohorts = fresh_cohorts(roster, repetitions, per_family)

    all_records = defaultdict(list)
    reports = []
    for index, cohort in enumerate(cohorts):
        rows = earlier.new_rows(cohort, ck, probe, negative)
        attach_public_gates(rows, ck, probe, negative)
        modes = {"original": task_eval(rows)}
        for kind in KINDS:
            modes[kind] = task_eval(rows, fitted[kind], selected[kind]["scale"])
        if global_choice["kind"] == "original":
            modes["dev_selected"] = modes["original"]
        else:
            modes["dev_selected"] = task_eval(
                rows, fitted[global_choice["kind"]], global_choice["scale"])
        # The four fixed-model interventions use the same trained task weights
        # and the same projection coefficients; only the specified feature changes.
        target = fitted["endpoint_basin"]
        amp = selected["endpoint_basin"]["scale"]
        modes["endpoint_basin_remove_nonlinear"] = task_eval(
            rows, target, amp, remove_residual=True)
        modes["endpoint_basin_replace_shuffled"] = task_eval(
            rows, target, amp, replace_gate="endpoint_shuffled")
        modes["endpoint_basin_replace_random"] = task_eval(
            rows, target, amp, replace_gate="endpoint_random")
        modes["endpoint_basin_replace_trajectory"] = task_eval(
            rows, target, amp, replace_gate="trajectory_basin")
        for kind, result in modes.items():
            assert result["cases"] == len(rows)
            assert result["paid_checks_total"] <= 4*len(rows)
            all_records[kind].extend(result["records"])
        reports.append({
            "replicate": index,
            "cases": len(rows),
            "distinct_input_groups": len({r["digest"] for r in rows}),
            "public_input_manifest_sha256": sha256(
                "\n".join(r["digest"] for r in rows).encode()).hexdigest(),
            "result": {kind: {
                "future_correct": value["future_correct"],
                "false_accept": value["false_accept"],
                "paid_checks": value["paid_checks_total"],
            } for kind, value in modes.items()}
        })
    outcomes = {kind: aggregate(recs) for kind, recs in all_records.items()}
    comparisons = {}
    baserecords = all_records["original"]
    for kind, recs in all_records.items():
        comparisons[kind + "_vs_original_nonlate"] = paired(
            baserecords, recs, nonlate_only=True)
    for control in ("endpoint_shuffled", "endpoint_random", "hidden_only"):
        comparisons["endpoint_basin_vs_" + control] = paired(
            all_records[control], all_records["endpoint_basin"], True)
    comparisons["trajectory_basin_vs_trajectory_random"] = paired(
        all_records["trajectory_random"], all_records["trajectory_basin"], True)
    for intervention in ("endpoint_basin_remove_nonlinear",
                         "endpoint_basin_replace_shuffled",
                         "endpoint_basin_replace_random",
                         "endpoint_basin_replace_trajectory"):
        comparisons["fixed_endpoint_basin_vs_" + intervention] = paired(
            all_records[intervention], all_records["endpoint_basin"], True)
    return {
        "schema": SCHEMA,
        "checkpoint_sha256": old.BASIN_SHA,
        "probe_sha256": design["frozen_probe_json_sha256"],
        "train_manifest": design["manifests"]["train"],
        "source_cases_train": TRAIN_ROWS,
        "source_cases_dev": DEV_ROWS,
        "public_training_template_checks_shared": TRAIN_ROWS*PUBLIC_TEMPLATES,
        "public_training_template_exposures_per_fit": TRAIN_ROWS*PUBLIC_TEMPLATES,
        "all_training_heads_slots": {k: v["slots"] for k, v in fitted.items()},
        "training": {kind: {
            "weight_sha256": sha256(json.dumps(model["weights"],
                sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "projection": model["projection"],
            "training_trace": model["training_trace"],
            "epochs": model["epochs"],
        } for kind, model in fitted.items()},
        "development_per_arm_frozen": selected,
        "development_global_frozen": global_choice,
        "development_grid": dev_grid,
        "new_source_seed_offset": NEW_TEST_SEED_OFFSET,
        "repetitions": repetitions,
        "cases_per_family_per_repetition": per_family,
        "cohorts": reports,
        "outcomes": outcomes,
        "primary_and_intervention_comparisons": comparisons,
        "public_per_case_evidence": {kind: recs for kind, recs in all_records.items()},
        "scope": {
            "nonlate_primary_and_aliased_late_source_secondary": True,
            "frozen_original_gru_and_task_head": True,
            "probe_probability_deterministic_from_endpoint_hidden": True,
            "endpoint_gate_additional_conditional_information_given_h": False,
            "training_only_linear_projection": True,
            "all_model_fits_use_public_demo_template_acceptance_only": True,
            "test_future_grading_after_frozen_development_selection": True,
            "at_most_four_paid_native_checks_per_case": True,
            "float_not_interval_certification": True,
            "statistical_p_values_exploratory_unadjusted": True,
        },
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repetitions", type=int, default=REPETITIONS)
    p.add_argument("--per-family", type=int, default=PER_FAMILY)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--check", action="store_true")
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    result = run(args.repetitions, args.per_family, args.epochs)
    if args.check:
        assert result["checkpoint_sha256"] == nor13.READOUT_SHA
        assert result["public_training_template_checks_shared"] == 1536
        assert set(result["all_training_heads_slots"].values()) == {400}
        assert result["outcomes"]["original"]["cases"] == (
            6*args.per_family*args.repetitions)
        assert result["outcomes"]["original"]["nonlate"]["cases"] == (
            5*args.per_family*args.repetitions)
        assert all(z["paid_checks"] <= 4*z["cases"]
                   for z in result["outcomes"].values())
    serialized = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(serialized)
    else:
        print(serialized, end="")


if __name__ == "__main__":
    main()

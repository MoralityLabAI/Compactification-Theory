"""SMC-15: frozen source-locked native two-SAE-feature tangent experiment.

Pilot uses SMC-13 calibration ONLY; full uses freshly preregistered Sudoku seeds.
No existing sealed cohort labels are used for feature or intervention selection.
All native continuations use PyTorch eager FP64 with frozen teacher weights.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import sys

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from smc12_cones import load_frozen_sources
from smc13_family import (
    new_full_solution, dig_unique, valid_solution, rectangle_count,
    count_solutions, fingerprint,
)
from smc13_train_teachers import tensors, source_model_from_lock
from smc13_transfer import load_independent_models
from smc14_causal_transfer import (
    ORIGINAL_SOURCE_RECEIPT_SHA256, NEW_TEACHER_RECEIPT_SHA256,
    CONDITIONED_JSON_SHA256, SOURCE_TEACHER_SHA256,
    PAIRS, digest, goal_positions, source_group_context,
    physical_decoder_unit_table, load_conditioned, exact_frozen_assignments,
)

SCHEMA = "SMC-15/native-coactive-causal-tangent-20261117-v1"
SOURCES = ("pretrainedA_71", "newB_501_SAE_71", "newB_501_SAE_89",
           "newB_701_SAE_71", "newB_701_SAE_89")
NEW_BASES = {"smc15_novel30": 202615030000, "smc15_novel34": 202615040000}
HOLES = {"smc15_novel30": 30, "smc15_novel34": 34}
SIZE = 64
N_GOALS = 8
SMALL = 0.08
RADIUS = 0.20
SIGMA_CUTOFF = 0.10
BOOT_SEED = 20261517
BOOT_N = 2000
PILOT_N = 4
MIN_PUZZLES = 10
MIN_GOALS = 30


def fresh_rosters(previous: dict) -> tuple[dict, dict]:
    """Construct strictly disjoint new cohorts from prespecified source seeds."""
    old_targets = {tuple(e["target"]) for rows in previous.values() for e in rows}
    old_inputs = {tuple(e["x"]) for rows in previous.values() for e in rows}
    new_targets = set()
    new_inputs = set()
    cohorts = {}
    for family, base in NEW_BASES.items():
        entries = []
        for i in range(SIZE):
            seed = base + i
            solution = new_full_solution(seed)
            if not valid_solution(solution) or rectangle_count(solution) <= 0:
                raise ValueError("new solution invalid or in canonical orbit")
            puzzle = dig_unique(solution, seed + 1000000, HOLES[family])
            if count_solutions(puzzle, 2) != 1:
                raise ValueError("new puzzle is nonunique")
            solution_key, puzzle_key = tuple(solution), tuple(puzzle)
            if solution_key in old_targets or solution_key in new_targets:
                raise ValueError("target duplicated in existing or new cohorts")
            if puzzle_key in old_inputs or puzzle_key in new_inputs:
                raise ValueError("puzzle duplicated in existing or new cohorts")
            new_targets.add(solution_key)
            new_inputs.add(puzzle_key)
            entries.append({"seed": seed, "x": puzzle, "target": solution,
                            "holes": HOLES[family],
                            "rectangles": rectangle_count(solution)})
        cohorts[family] = entries
    manifest = {"type": "SMC-15/source-disjoint-20261117-v1",
                "sources": NEW_BASES, "n_per_family": SIZE,
                "excluded_previous_target_count": len(old_targets),
                "puzzles_unique": len(new_inputs) == 2 * SIZE,
                "targets_unique": len(new_targets) == 2 * SIZE,
                "cohorts": {
                    name: {
                        "fingerprint_sha256": fingerprint(rows),
                        "input_sha256": hashlib.sha256(
                            json.dumps([r["x"] for r in rows],
                                       sort_keys=True, separators=(",", ":")).encode()
                        ).hexdigest(),
                        "target_sha256": hashlib.sha256(
                            json.dumps([r["target"] for r in rows],
                                       sort_keys=True, separators=(",", ":")).encode()
                        ).hexdigest(),
                        "min_rectangles": min(r["rectangles"] for r in rows),
                        "max_rectangles": max(r["rectangles"] for r in rows),
                        "n": len(rows),
                    } for name, rows in cohorts.items()
                }}
    return cohorts, manifest


def frozen_two_features(sae, mean, scale, stage, positions):
    """Choose from natural top4 activations only, with an unused third-place tie gate."""
    with torch.no_grad():
        encoded = sae.encode((stage - mean) / scale)
        selected = encoded[positions.long()].topk(3, dim=-1, sorted=True)
        ids, vals = selected.indices.cpu(), selected.values.cpu()
        admissible = ((vals[:, 0] > vals[:, 1]) &
                      (vals[:, 1] > vals[:, 2]) &
                      (vals[:, 2] >= 0))
        return ids[:, :2].clone(), admissible, vals.clone()


def zero_center_digits(logits):
    nine = logits[..., 1:10].double()
    return nine - nine.mean(dim=-1, keepdim=True)


def native_logits_from_stage(model, anchors):
    """Three native FP64 steps, shared across all counterfactual arms."""
    with torch.no_grad():
        value = anchors.double()
        for _ in range(3):
            value = model.block(value)
        return model.head(value)


def normalised_frame(j1, j2):
    """9-logit response frame; returns None for numerically silent columns."""
    a = torch.as_tensor(j1, dtype=torch.float64)
    b = torch.as_tensor(j2, dtype=torch.float64)
    if a.shape != (9,) or b.shape != (9,):
        raise ValueError("expected two nine-digit response columns")
    a = a - a.mean()
    b = b - b.mean()
    norms = torch.stack([a.norm(), b.norm()])
    if not bool(torch.isfinite(norms).all()) or bool((norms <= 1e-10).any()):
        return None
    A = torch.column_stack([a / norms[0], b / norms[1]])
    _, singular, vh = torch.linalg.svd(A, full_matrices=False)
    return singular, vh, norms


def deterministic_sham(u, v, desired_norm, seed):
    if desired_norm <= 1e-12:
        return None
    basis = torch.stack([u, v], dim=-1)
    if float(torch.linalg.svdvals(basis)[-1]) <= 1e-8:
        return None
    q = torch.linalg.qr(basis, mode="reduced").Q
    gen = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(48, dtype=torch.float64, generator=gen)
    noise = noise - q @ (q.T @ noise)
    denom = float(noise.norm())
    if denom <= 1e-8:
        return None
    result = noise * (desired_norm / denom)
    if not math.isclose(float(result.norm()), desired_norm, rel_tol=1e-11,
                        abs_tol=1e-12):
        raise ArithmeticError("nuisance edit failed exact norm match")
    return result


def _entry_hash_seed(*values):
    seed_bytes = "|".join(map(str, values)).encode()
    return int.from_bytes(hashlib.sha256(seed_bytes).digest()[:8], "big") & ((1 << 63) - 1)


def _patch_batch(stage, token, deltas):
    """All arms built from one anchor with exactly one modified token."""
    h = stage.unsqueeze(0).double().repeat(len(deltas), 1, 1)
    for i, delta in enumerate(deltas):
        if delta is not None:
            h[i, token, :] += delta
    return h


def score_goal(model, stage, clean_reference, token, truth, ids, vals,
               eligible, unit_table, sham_seed):
    """A preselected pair undergoes 6-arm tangent and 8-arm finite-dose tests.

    Selection is fixed by the caller before clean truth or effect inspection.
    """
    head = {"token": int(token), "selected_atoms": [int(v) for v in ids],
            "selected_top3": [float(v) for v in vals],
            "strictly_active": bool(eligible)}
    if not eligible:
        return {**head, "status": "abstain_top2_not_strict"}
    u = unit_table[int(ids[0])].double()
    v = unit_table[int(ids[1])].double()
    physical = torch.stack([u, v], dim=-1)
    physical_sv = torch.linalg.svdvals(physical)
    if float(physical_sv[-1]) <= 1e-8:
        return {**head, "status": "abstain_physical_rank"}
    local_deltas = [
        None, torch.zeros_like(u), SMALL * u, -SMALL * u,
        SMALL * v, -SMALL * v,
    ]
    local = native_logits_from_stage(model, _patch_batch(stage, token, local_deltas))[:, token]
    if not torch.equal(local[0], local[1]):
        raise ArithmeticError("self patch changed native logits")
    err = float((local[0] - clean_reference[token]).abs().max())
    if err > 1e-8:
        raise ArithmeticError("source-clean vs matched-batch native logits differ")
    j1 = zero_center_digits((local[2] - local[3]) / (2 * SMALL))
    j2 = zero_center_digits((local[4] - local[5]) / (2 * SMALL))
    frame = normalised_frame(j1, j2)
    if frame is None:
        return {**head, "status": "abstain_silent_native_column",
                "source_clean_discrepancy": err}
    singular, vh, norms = frame
    d_min = vh[-1, 0] * u + vh[-1, 1] * v
    d_max = vh[0, 0] * u + vh[0, 1] * v
    if float(d_min.norm()) <= 1e-10 or float(d_max.norm()) <= 1e-10:
        return {**head, "status": "abstain_unrealizable_singular_edit"}
    d_min = RADIUS * d_min / d_min.norm()
    d_max = RADIUS * d_max / d_max.norm()
    d_both = RADIUS * (u + v)
    physical_joint_norm = float(d_both.norm())
    sham = deterministic_sham(u, v, physical_joint_norm, sham_seed)
    if sham is None:
        return {**head, "status": "abstain_sham_rank"}

    final_deltas = [
        None, torch.zeros_like(u), d_min, d_max,
        RADIUS * u, RADIUS * v, d_both, sham,
    ]
    actual = native_logits_from_stage(model, _patch_batch(stage, token, final_deltas))[:, token]
    if not torch.equal(actual[0], actual[1]):
        raise ArithmeticError("finite-dose baseline self-patch changed logits")
    if float((actual[0] - local[0]).abs().max()) > 1e-8:
        raise ArithmeticError("matched-arm baseline drift between stages")
    min_change = float(zero_center_digits(actual[2] - actual[0]).norm())
    max_change = float(zero_center_digits(actual[3] - actual[0]).norm())
    if max_change <= 1e-12:
        return {**head, "status": "abstain_silent_finite_dose"}
    digit = int(truth)
    if not 1 <= digit <= 9:
        raise ValueError("target label outside digits 1-9")
    rival_logits = actual[0].clone()
    rival_logits[digit] = -float("inf")
    rival = int(torch.argmax(rival_logits))
    margins = actual[:, digit] - actual[:, rival]
    source_rival = int(torch.argmax(torch.cat((
        clean_reference[token, :digit],
        torch.tensor([-float("inf")], dtype=torch.float64),
        clean_reference[token, digit+1:]
    ))))
    if source_rival != rival:
        raise ArithmeticError("clean rival differs from original source")
    gap = float(margins[6] - margins[4] - margins[5] + margins[0])
    sham_gap = float(margins[7] - margins[0])
    output = {**head, "status": "measured",
              "source_clean_discrepancy": err, "native_response_norms": norms.tolist(),
              "sigma_physical_min": float(physical_sv[-1]),
              "sigma_physical_max": float(physical_sv[0]),
              "sigma_output_min": float(singular[-1]),
              "sigma_output_max": float(singular[0]),
              "condition_low": bool(float(singular[-1]) < SIGMA_CUTOFF),
              "native_finite_min_norm": min_change,
              "native_finite_max_norm": max_change,
              "minmax_finite_response_ratio": min_change / max_change,
              "dose_min": float(d_min.norm()), "dose_max": float(d_max.norm()),
              "joint_physical_norm": physical_joint_norm,
              "sham_physical_norm": float(sham.norm()),
              "native_joint_margin_nonadditivity": gap,
              "native_sham_margin_change": sham_gap,
              "native_clean_margin": float(margins[0]),
              "native_min_margin_change": float(margins[2] - margins[0]),
              "native_max_margin_change": float(margins[3] - margins[0]),
              "native_joint_margin_change": float(margins[6] - margins[0]),
              "native_u_margin_change": float(margins[4] - margins[0]),
              "native_v_margin_change": float(margins[5] - margins[0]),
              "clean_digit_correct": bool(int(actual[0].argmax()) == digit),
              "frozen_clean_rival": rival, "truth_digit": digit}
    if not all(math.isfinite(float(output[k])) for k in (
        "sigma_output_min", "minmax_finite_response_ratio",
        "native_joint_margin_nonadditivity", "native_sham_margin_change")):
        raise ArithmeticError("nonfinite finite-dose effect")
    return output


def score_component_puzzle(model, stage, clean, positions, truth,
                           component, sae, mean, scale, table, family, index):
    with torch.no_grad():
        ids, valid, vals = frozen_two_features(sae, mean, scale, stage, positions)
        trials = []
        for j, pos in enumerate(positions.tolist()):
            trials.append(score_goal(model, stage, clean, pos, int(truth[j]),
                                     ids[j].tolist(), vals[j].tolist(),
                                     bool(valid[j]), table,
                                     _entry_hash_seed("SMC15", family, index, component, j)))
    return {"puzzle_index": index, "family": family,
            "component": component,
            "n_measured": sum(r["status"] == "measured" for r in trials),
            "goal_trials": trials,
            "selected_top2": ids.tolist(),
            "selection_strict_mask": valid.tolist()}


def _pool_group(rows, low):
    values = []
    goal_count = 0
    for row in rows:
        chosen = [g["minmax_finite_response_ratio"] for g in row["goal_trials"]
                  if g["status"] == "measured" and g["condition_low"] == low]
        if chosen:
            goal_count += len(chosen)
            values.append(float(sum(chosen) / len(chosen)))
    return values, goal_count


def summary_component(rows, family, component):
    low, n_low = _pool_group(rows, True)
    high, n_high = _pool_group(rows, False)
    measured = [g for p in rows for g in p["goal_trials"] if g["status"] == "measured"]
    statuses = {}
    for p in rows:
        for g in p["goal_trials"]:
            statuses[g["status"]] = statuses.get(g["status"], 0) + 1
    informative = (len(low) >= MIN_PUZZLES and len(high) >= MIN_PUZZLES and
                   n_low >= MIN_GOALS and n_high >= MIN_GOALS)
    bootstrap = None
    if informative:
        rng = random.Random(_entry_hash_seed(BOOT_SEED, family, component))
        estimates = []
        for _ in range(BOOT_N):
            sample = [rows[rng.randrange(len(rows))] for _ in range(len(rows))]
            lb, _ = _pool_group(sample, True)
            hb, _ = _pool_group(sample, False)
            if not lb or not hb:
                continue
            estimates.append(sum(hb) / len(hb) - sum(lb) / len(lb))
        estimates.sort()
        if len(estimates) == BOOT_N:
            bootstrap = [estimates[int(BOOT_N * 0.025)],
                         estimates[int(BOOT_N * 0.975) - 1]]
    low_mean = sum(low)/len(low) if low else None
    high_mean = sum(high)/len(high) if high else None
    gate = bool(informative and bootstrap is not None and bootstrap[0] > 0)
    return {"family": family, "component": component, "n_puzzles": len(rows),
            "status": "measured" if informative else "underpowered_groups",
            "eligibility_status_counts": statuses,
            "n_measured_goals": len(measured),
            "n_low_conditioning_goals": n_low,
            "n_high_conditioning_goals": n_high,
            "n_low_conditioning_puzzles": len(low),
            "n_high_conditioning_puzzles": len(high),
            "low_puzzle_mean_finite_ratio": low_mean,
            "high_puzzle_mean_finite_ratio": high_mean,
            "high_minus_low_puzzle_mean": (
                high_mean - low_mean if low_mean is not None and high_mean is not None else None),
            "puzzle_bootstrap95_high_minus_low": bootstrap,
            "registered_conditional_gate": gate,
            "mean_native_joint_nonadditivity": (
                sum(g["native_joint_margin_nonadditivity"] for g in measured)/len(measured)
                if measured else None),
            "median_normalized_sigma_min": (
                float(torch.tensor([g["sigma_output_min"] for g in measured]).median())
                if measured else None),
            "source": "native FP64 3-step continuation of frozen teacher"}


def cross_coactivity(rows_by_component, supports, assignments, family):
    """Only active source pairs are eligible; target masks are untouched."""
    reports = []
    for pair_index in range(4):
        a,b = PAIRS[pair_index]
        assignment = assignments[pair_index]["matched"]
        source_rows = rows_by_component[a]
        target_support = supports[b][family]["goal_support"]
        num_source = 0
        num_target_both = 0
        for pi, row in enumerate(source_rows):
            for j, valid in enumerate(row["selection_strict_mask"]):
                if not valid:
                    continue
                num_source += 1
                a0, a1 = row["selected_top2"][j]
                mapped = [assignment[a0], assignment[a1]]
                if bool(target_support[pi,j,mapped].all()):
                    num_target_both += 1
        reports.append({
            "source": a, "target": b, "family": family,
            "n_goals": len(source_rows) * N_GOALS,
            "n_source_strict": num_source,
            "n_both_mapped_target_active": num_target_both,
            "fraction_of_all_goals": num_target_both/(len(source_rows)*N_GOALS),
            "fraction_of_source_selected": num_target_both/max(num_source,1),
            "assignment_hash_sha256": digest(json.dumps(assignment).encode()),
            "status": "descriptive_only",
        })
    return reports


def load_and_lock_inputs(original_artifact, independent_artifact, conditioned_receipt,
                         allow_fetch=False, source_checkpoint=None):
    for location, filename, sha in (
        (original_artifact, "joint-full.json", ORIGINAL_SOURCE_RECEIPT_SHA256),
        (independent_artifact, "full.json", NEW_TEACHER_RECEIPT_SHA256),
    ):
        if digest((Path(location)/filename).read_bytes()) != sha:
            raise ValueError("source artifact receipt SHA256 mismatch: " + filename)
    orig, saes, _ = load_frozen_sources(extracted_dir=original_artifact)
    source, _ = source_model_from_lock(allow_fetch, source_checkpoint)
    independent, rosters, models_new = load_independent_models(independent_artifact)
    conditioned = load_conditioned(conditioned_receipt)
    models = {"A": source}
    components = {"pretrainedA_71": ("A", *saes[71])}
    for seed, group in models_new.items():
        label = f"B{seed}"
        models[label] = group["teacher"]
        for sae_seed, params in group["saes"].items():
            components[f"newB_{seed}_SAE_{sae_seed}"] = (label, *params)
    calib_ctx, calib_support = source_group_context(
        models, components, rosters, ("feature_calibration",))
    frozen = {}
    for index,(a,b) in enumerate(PAIRS[:4]):
        pair_record = conditioned["results"][index]
        if (pair_record["source"], pair_record["target"]) != (a,b):
            raise ValueError("SMC13C correspondence pair order changed")
        calibrated_inputs = calib_ctx[components[a][0]]["cohorts"][
            "feature_calibration"]["inputs"]
        assigns, hashes = exact_frozen_assignments(
            calib_support[a]["feature_calibration"]["support"],
            calib_support[b]["feature_calibration"]["support"],
            calibrated_inputs, pair_record, index)
        frozen[index] = {"permutations": assigns, "hashes": hashes}
    return models, components, rosters, frozen


def run(mode, original_artifact, independent_artifact, conditioned_receipt,
        allow_fetch=False, source_checkpoint=None):
    if mode not in ("pilot", "full"):
        raise ValueError("pilot or full mode required")
    torch.set_num_threads(2)
    models, components, prior, calibration = load_and_lock_inputs(
        original_artifact, independent_artifact, conditioned_receipt,
        allow_fetch, source_checkpoint)
    if mode == "pilot":
        rosters = {"feature_calibration_pilot": prior["feature_calibration"][:PILOT_N]}
        manifest = {"type": "SMC15/calibration-only-pilot", "n": PILOT_N,
                    "source_roster_sha256": fingerprint(rosters["feature_calibration_pilot"])}
    else:
        # Only now, after the original 4 assignment hashes pass, create new cohorts.
        rosters, manifest = fresh_rosters(prior)
    keys = tuple(rosters)
    ctx, supports = source_group_context(models, components, rosters, keys)
    tables = {name: physical_decoder_unit_table(values[1], values[3])
              for name, values in components.items()}
    all_results = {}
    cross_results = {}
    for family in keys:
        per_component = {}
        for component in SOURCES:
            teacher_id, sae, mean, scale = components[component]
            teacher = ctx[teacher_id]
            cohort = teacher["cohorts"][family]
            records = []
            for i in range(len(rosters[family])):
                records.append(score_component_puzzle(
                    teacher["model64"], cohort["stage1"][i],
                    cohort["clean_logits"][i], cohort["positions"][i],
                    cohort["truth_at_goals"][i], component, sae, mean, scale,
                    tables[component], family, i))
            per_component[component] = records
            print("SMC15_NATIVE", mode, family, component,
                  summary_component(records, family, component), flush=True)
        cross_results[family] = cross_coactivity(
            per_component, supports,
            {i: v["permutations"] for i,v in calibration.items()}, family)
        all_results[family] = {
            name: {"summary": summary_component(records,family,name),
                   "per_puzzle": records} for name, records in per_component.items()
        }
    return {
        "schema": SCHEMA, "status": mode, "registered_before_test": True,
        "source_teacher_sha256": SOURCE_TEACHER_SHA256,
        "original_sae_receipt_sha256": ORIGINAL_SOURCE_RECEIPT_SHA256,
        "independent_teacher_receipt_sha256": NEW_TEACHER_RECEIPT_SHA256,
        "source_conditioned_receipt_sha256": CONDITIONED_JSON_SHA256,
        "calibration_permutation_receipts": calibration,
        "torch_version": torch.__version__, "torch_compiled": False,
        "precision": "FP32 post-stage1 anchor, FP64 3-block native continuation",
        "calibration_only_first": True,
        "radius_small": SMALL, "radius_verify": RADIUS,
        "predeclared_sigma_threshold": SIGMA_CUTOFF,
        "new_puzzle_manifest": manifest,
        "components": SOURCES, "data": all_results,
        "source_to_target_coactivity": cross_results,
        "limitations": ("Within-goal two-dose selected SVD, finite FP64 numerical native evidence. "
                        "This is a frozen linear SAE, not a trained curved SAE. "
                        "Test-derived sigma groups are conditional; cross-model coactivity is descriptive."),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("pilot","full"), required=True)
    parser.add_argument("--fetch-source", action="store_true")
    parser.add_argument("--source-checkpoint", type=Path)
    parser.add_argument("--original-artifact", type=Path, required=True)
    parser.add_argument("--independent-artifact", type=Path, required=True)
    parser.add_argument("--conditioned-receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.mode, args.original_artifact,
                 args.independent_artifact, args.conditioned_receipt,
                 args.fetch_source, args.source_checkpoint)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
    print("SMC15_COMPLETE", json.dumps({
        "out": str(args.out), "mode":args.mode,
        "puzzle_manifest": report["new_puzzle_manifest"],
        "summaries": {family:{k:v["summary"]["status"]
                     for k,v in entries.items()} for family,entries in report["data"].items()},
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

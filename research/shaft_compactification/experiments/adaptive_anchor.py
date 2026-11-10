"""SHAFT-COMP-10b: model-aware reference-anchor selection for local toy GRUs.

Calibration accesses the true synthetic alignment, so the optimization is an
oracle-calibrated *reference design*, not unsupervised gauge discovery. Tests
use independent recurrent prefixes and retain both full trained models.
"""
from __future__ import annotations
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from research.shaft_compactification.experiments.activation_noise import (
    GAUGES, CORRECT, TARGET_PERM, TARGET_FLIP,
    true_signatures, batched_signatures, infer_all, score_summary, errors_diagnostics,
    DESIGNS, BUDGETS,
)
from research.shaft_compactification.experiments.trained_recurrence import fit


def mean_margin_matrix(target, depth, seed, cal_per_goal=128):
    goals = np.repeat(np.arange(16, dtype=np.int64), cal_per_goal)[:, None]
    obs = batched_signatures(target, goals, (depth,), seed,
                             perm=TARGET_PERM, flip=TARGET_FLIP)[depth]
    source_truth = true_signatures(goals, (0,1,2,3), 0)
    vals = []
    for perm, flip in GAUGES:
        diff = np.zeros(goals.shape, dtype=np.int16)
        for q in range(4):
            diff += (source_truth[:, :, q] != (obs[:, :, perm[q]] ^ flip)).astype(np.int16)
        vals.append(diff.reshape(16, cal_per_goal).mean(axis=1))
    values = np.stack(vals, axis=1)
    return values - values[:, [CORRECT]]


def select_anchors(margins):
    wrong = np.delete(margins, CORRECT, axis=1)
    ranked = []
    for triple in itertools.combinations(range(16), 3):
        total = wrong[list(triple)].sum(axis=0)
        ranked.append((float(total.min()), triple))
    best = max(score for score, _ in ranked)
    optimal = next(triple for score, triple in ranked if score == best)

    def gap(triple):
        return float(wrong[list(triple)].sum(axis=0).min())
    return {
        "selected_triple": list(optimal),
        "minimum_expected_score_margin": round(best, 6),
        "onehot_minimum_expected_score_margin": round(gap((1,2,4)), 6),
        "balanced_minimum_expected_score_margin": round(gap((3,5,6)), 6),
        "candidate_triples_considered": len(ranked),
    }


def eval_triple(source, target, triple, depth, seed, trials=512):
    cycle = tuple(triple)
    seq = (cycle * ((24 + len(cycle) - 1) // len(cycle)))[:24]
    goals = np.tile(np.asarray(seq, dtype=np.int64), (trials,1))
    source_sig = batched_signatures(source, goals, (12,), seed)[12]
    target_sig = batched_signatures(target, goals, (depth,), seed + 127,
                                    perm=TARGET_PERM, flip=TARGET_FLIP)[depth]
    return {
        "source_error": errors_diagnostics(source_sig, true_signatures(goals, (0,1,2,3), 0)),
        "target_error": errors_diagnostics(target_sig, true_signatures(goals, TARGET_PERM, TARGET_FLIP)),
        "recovery": score_summary(infer_all(source_sig, target_sig)),
    }


def run(trials=512, cal_per_goal=128, evaluation_seed_base=83000):
    torch.set_num_threads(3)
    torch.use_deterministic_algorithms(True)
    models = {seed: fit(seed)[0].eval() for seed in (101,202)}
    results = {}
    for source_seed, target_seed in ((101,202), (202,101)):
        sid = f"{source_seed}_to_{target_seed}"
        source, target = models[source_seed], models[target_seed]
        results[sid] = {}
        for depth in (40,64):
            margins = mean_margin_matrix(target, depth, 72000 + target_seed + depth, cal_per_goal)
            selected = select_anchors(margins)
            designs = {
                "calibration_selected": tuple(selected["selected_triple"]),
                "onehot": (1,2,4),
                "balanced_three": (3,5,6),
                "balanced_six": (3,5,6,9,10,12),
            }
            evaluated = {
                name: eval_triple(source, target, triple, depth,
                                  evaluation_seed_base + target_seed + depth, trials)
                for name, triple in designs.items()
            }
            results[sid][str(depth)] = {"calibration": selected, "heldout": evaluated}
    return {
        "experiment": "SHAFT-COMP-10b",
        "trials": trials,
        "calibration_prefixes_per_goal": cal_per_goal,
        "evaluation_seed_base": evaluation_seed_base,
        "anchor_budget": list(BUDGETS),
        "noisy_process": "trained GRU recurrent state only",
        "selection": "maximize minimum estimated expected score gap over all 47 wrong gauges; distinct 3-task triples; calibrated with true synthetic alignment",
        "results": results,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=512)
    p.add_argument("--cal", type=int, default=128)
    p.add_argument("--evaluation-seed", type=int, default=83000)
    p.add_argument("--out")
    a = p.parse_args()
    s = json.dumps(run(a.trials, a.cal, a.evaluation_seed), indent=2, sort_keys=True) + "\n"
    if a.out:
        Path(a.out).write_text(s)
    else:
        print(s, end="")


if __name__ == "__main__":
    main()

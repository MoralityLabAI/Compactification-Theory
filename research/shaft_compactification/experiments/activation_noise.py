"""SHAFT-COMP-10: endogenous recurrent observation errors and semantic-gauge identifiability.

Only two consented local toy GRUs with 16 four-bit lookup tasks. No networking,
model persistence, private weights or external devices. All goal IDs are used to
make experiment fixtures and audit results; the gauge estimator sees only paired
behavioral output signatures.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from research.shaft_compactification.experiments.trained_recurrence import fit, GOALS, TOKENS
from research.shaft_compactification.gauge_identifiability import GAUGES, TARGET_PERM, TARGET_FLIP

DESIGNS = {"onehot": (1, 2, 4), "balanced": (3, 5, 6, 9, 10, 12)}
DEPTHS = (12, 24, 40, 64)
BUDGETS = (3, 6, 12, 24)
CORRECT = GAUGES.index((TARGET_PERM, TARGET_FLIP))


def fixtures(design: str, n_trials: int):
    cycle = DESIGNS[design]
    return np.tile(np.asarray((cycle * 8)[:24], dtype=np.int64), (n_trials, 1))


@torch.inference_mode()
def batched_signatures(model, goal_matrix: np.ndarray, depths: tuple[int, ...], seed: int,
                       perm: tuple[int, ...] = (0, 1, 2, 3), flip: int = 0,
                       batch_size: int = 2048):
    """Observe the trained model's counterfactual responses after recurrent prefixes."""
    goal_flat = torch.from_numpy(goal_matrix.reshape(-1).copy()).long()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    tokens = torch.randint(TOKENS, (len(goal_flat), max(depths)), generator=generator)
    output = {d: np.empty((len(goal_flat), 4), dtype=np.uint8) for d in depths}
    q = F.one_hot(torch.tensor(perm), TOKENS).float()
    for start in range(0, len(goal_flat), batch_size):
        stop = min(start + batch_size, len(goal_flat))
        g, t = goal_flat[start:stop], tokens[start:stop]
        h = model.embedding(g)
        for depth in range(1, max(depths) + 1):
            h = model.cell(F.one_hot(t[:, depth - 1], TOKENS).float(), h)
            if depth in depths:
                qq = q.repeat(len(g), 1)
                hh = h.repeat_interleave(TOKENS, 0)
                advanced = model.cell(qq, hh)
                logits = model.readout(torch.cat((advanced, qq), dim=1)).reshape(len(g), TOKENS)
                output[depth][start:stop] = (logits >= 0).byte().numpy() ^ flip
    return {d: val.reshape(*goal_matrix.shape, 4) for d, val in output.items()}


def true_signatures(goals: np.ndarray, perm: tuple[int, ...], flip: int):
    a = np.asarray(perm, dtype=np.int64)
    return (((goals[..., None] >> a) & 1) ^ flip).astype(np.uint8)


def errors_diagnostics(observed, truth):
    errors = observed != truth
    nrows = errors.reshape(-1, 4)
    p = nrows.mean()
    rowcnt = nrows.sum(axis=1)
    counts = np.bincount(rowcnt, minlength=5)
    independent_ge2 = 1 - (1-p)**4 - 4*p*(1-p)**3
    query_rates = nrows.mean(axis=0)
    return {
        "mean_bit_error": round(float(p), 6),
        "query_bit_error": [round(float(x), 6) for x in query_rates],
        "row_error_histogram": counts.tolist(),
        "fraction_rows_at_least_two_errors": round(float((rowcnt >= 2).mean()), 6),
        "matched_iid_bsc_fraction_rows_at_least_two_errors": round(float(independent_ge2), 6),
        "two_or_more_error_excess_over_bsc":
            round(float((rowcnt >= 2).mean() - independent_ge2), 6),
    }


def infer_all(source: np.ndarray, target: np.ndarray):
    """Cumulative Hamming-score matrix [trials, 48 gauges, 24 anchors]."""
    values = []
    for perm, flip in GAUGES:
        diff = np.zeros(source.shape[:2], dtype=np.int16)
        for q in range(4):
            diff += (source[:, :, q] != (target[:, :, perm[q]] ^ flip)).astype(np.int16)
        values.append(np.cumsum(diff, axis=1))
    return np.stack(values, axis=1)


def score_summary(scores: np.ndarray):
    out = {}
    for budget in BUDGETS:
        score = scores[:, :, budget - 1]
        minimum = score.min(axis=1)
        winners = score == minimum[:, None]
        true_best = score[:, CORRECT] == minimum
        nbest = winners.sum(axis=1)
        first = winners.argmax(axis=1)
        other = np.delete(score, CORRECT, axis=1).min(axis=1)
        margin = other - score[:, CORRECT]
        out[str(budget)] = {
            "uniform_random_tie_expected_accuracy": round(float(np.mean(true_best / nbest)), 6),
            "lex_tiebreak_correct": int((first == CORRECT).sum()),
            "unique_correct": int((true_best & (nbest == 1)).sum()),
            "true_among_minima": int(true_best.sum()),
            "tie_trials": int((nbest > 1).sum()),
            "mean_true_vs_best_wrong_margin": round(float(margin.mean()), 6),
            "median_true_vs_best_wrong_margin": round(float(np.median(margin)), 6),
        }
    return out


def mock_iid_bsc(goals: np.ndarray, trial_prob: float, rng: np.random.Generator):
    truth = true_signatures(goals, TARGET_PERM, TARGET_FLIP)
    return truth ^ (rng.random(truth.shape) < trial_prob).astype(np.uint8)


def condition(source, target, design: str, n_trials: int, seed: int, cal_trials=192):
    goals = fixtures(design, n_trials)
    source_sign = batched_signatures(source, goals, (12,), seed)[12]
    target_sign = batched_signatures(target, goals, DEPTHS, seed + 123,
                                     perm=TARGET_PERM, flip=TARGET_FLIP)
    cal_goals = fixtures(design, cal_trials)
    cal_sign = batched_signatures(target, cal_goals, DEPTHS, seed + 10501,
                                  perm=TARGET_PERM, flip=TARGET_FLIP)
    cal_truth = true_signatures(cal_goals, TARGET_PERM, TARGET_FLIP)
    source_truth = true_signatures(goals, (0,1,2,3), 0)
    target_truth = true_signatures(goals, TARGET_PERM, TARGET_FLIP)
    source_errors = errors_diagnostics(source_sign, source_truth)
    result = {"source_depth": 12, "source_errors": source_errors, "target_depths": {}}
    for depth in DEPTHS:
        target_obs = target_sign[depth]
        diagnostics = errors_diagnostics(target_obs, target_truth)
        endogenous = score_summary(infer_all(source_sign, target_obs))
        p = float((cal_sign[depth] != cal_truth).mean())
        per_goal = {}
        for a in sorted(set(DESIGNS[design])):
            mask = goals == a
            per_goal[str(a)] = round(float((target_obs != target_truth)[mask].mean()), 6)
        iid = mock_iid_bsc(goals, p, np.random.default_rng(seed + 8000 + depth))
        synthetic = score_summary(infer_all(source_sign, iid))
        result["target_depths"][str(depth)] = {
            "target_errors": diagnostics,
            "endogenous_recovery": endogenous,
            "matched_iid_bsc_recovery": synthetic,
            "iid_calibration_flip_rate": round(p, 6),
            "target_error_by_anchor_task": per_goal,
        }
    return result


def run(trials=512, seed=86244, cal_trials=192):
    torch.set_num_threads(3)
    torch.use_deterministic_algorithms(True)
    models = {i: fit(i)[0].eval() for i in (101, 202)}
    result = {
        "status": "locally_trained_toy_gru_endogenous_errors",
        "torch": torch.__version__, "numpy": np.__version__,
        "seeds_base_models": [101, 202], "experiment_seed": seed,
        "trials_per_condition": trials, "max_anchors": 24,
        "source_depth": 12, "target_depths": list(DEPTHS),
        "anchor_budgets": list(BUDGETS),
        "noise_source": "deterministic neural dynamics from independent random prefixes",
        "matched_iid_control": "independent flips of exact target responses using separate calibration prefixes to estimate marginal bit error",
        "calibration_trials_per_design": cal_trials,
        "assumptions": "source and target share a synthetic four-query task language; source signature error audited; goal IDs only used for generation/measurement",
        "directions": {},
    }
    for direction_idx, (source_seed, target_seed) in enumerate(((101,202), (202,101))):
        direction = {}
        for i, design in enumerate(DESIGNS):
            direction[design] = condition(models[source_seed], models[target_seed],
                                          design, trials, seed + 50000*direction_idx + 1000*i,
                                          cal_trials)
        result["directions"][f"{source_seed}_to_{target_seed}"] = direction
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=512)
    ap.add_argument("--seed", type=int, default=86244)
    ap.add_argument("--cal-trials", type=int, default=192)
    ap.add_argument("--out")
    a = ap.parse_args()
    s = json.dumps(run(a.trials, a.seed, a.cal_trials), sort_keys=True, indent=2) + "\n"
    if a.out:
        p = Path(a.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(s)
    else:
        print(s, end="")


if __name__ == "__main__":
    main()

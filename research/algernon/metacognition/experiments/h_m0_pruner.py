"""Algernon H-M0 *exploratory CPU toy*, not any of registered H-C/R/W/J.

Learn a small policy to select informative probes about a latent four-way world,
under a fixed cost budget. Likelihoods are visible to every policy. The policy
learns only the *choice of experiment*; belief updating remains exact Bayes.
No Qwen, real scientific outcomes, model weights from GitHub or GPU are used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

N_H = 4
N_PROBES = 8
BUDGET = 5.0
COST_OPTIONS = np.array([1.0, 1.5, 2.0, 3.0], dtype=np.float64)


def entropy(p: np.ndarray) -> float:
    pp = np.clip(p, 1e-12, 1)
    return float(-np.sum(pp * np.log(pp)))


def posterior_update(p: np.ndarray, likelihood: np.ndarray, observed: int) -> np.ndarray:
    likelihood = likelihood if observed else 1.0 - likelihood
    updated = p * np.clip(likelihood, 1e-12, 1)
    return updated / updated.sum()


def information_gain(p: np.ndarray, likelihood: np.ndarray) -> float:
    prob_pos = float(np.dot(p, likelihood))
    p_pos = posterior_update(p, likelihood, 1)
    p_neg = posterior_update(p, likelihood, 0)
    ig = entropy(p) - (prob_pos * entropy(p_pos) + (1 - prob_pos) * entropy(p_neg))
    return max(0.0, ig)


@dataclass
class Episode:
    prior: np.ndarray
    likelihoods: np.ndarray
    costs: np.ndarray
    truth: int
    observations: np.ndarray
    seed: int


def make_episode(seed: int) -> Episode:
    rng = np.random.default_rng(seed)
    prior = rng.dirichlet(np.full(N_H, 0.85))
    baseline = rng.beta(1.5, 1.5, size=(N_PROBES, 1))
    sharp = rng.beta(0.5, 0.5, size=(N_PROBES, N_H))
    mix = rng.uniform(0.15, 1.0, size=(N_PROBES, 1))
    likelihoods = np.clip((1.0 - mix) * baseline + mix * sharp, 0.015, 0.985)
    costs = rng.choice(COST_OPTIONS, size=N_PROBES, replace=True)
    truth = int(rng.choice(N_H, p=prior))
    # Fixed common random numbers per episode and probe across policy arms.
    observations = (rng.uniform(size=N_PROBES) < likelihoods[:, truth]).astype(np.int64)
    return Episode(prior, likelihoods, costs, truth, observations, seed)


def feature(p: np.ndarray, rates: np.ndarray, cost: float, remaining: float) -> np.ndarray:
    return np.concatenate((p, rates, [cost / 3.0, remaining / BUDGET, entropy(p) / np.log(N_H), float(p.max())])).astype(np.float32)


def make_train_samples(n_states: int, seed: int):
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for k in range(n_states):
        ep = make_episode(seed * 100000 + k)
        p = ep.prior.copy()
        remaining = BUDGET
        selected = set()
        # Sample a random legal history so the learner sees intermediate beliefs.
        for _ in range(int(rng.integers(0, 4))):
            legal = [i for i in range(N_PROBES) if i not in selected and ep.costs[i] <= remaining]
            if not legal:
                break
            i = int(rng.choice(legal))
            selected.add(i)
            remaining -= ep.costs[i]
            p = posterior_update(p, ep.likelihoods[i], int(ep.observations[i]))
        for i in range(N_PROBES):
            if i in selected or ep.costs[i] > remaining:
                continue
            xs.append(feature(p, ep.likelihoods[i], ep.costs[i], remaining))
            ys.append(information_gain(p, ep.likelihoods[i]) / ep.costs[i])
    return np.stack(xs), np.asarray(ys, dtype=np.float32)


def learn_policy(train_size: int, seed: int, epochs: int):
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    xs, ys = make_train_samples(train_size, seed)
    x = torch.from_numpy(xs)
    y = torch.from_numpy(ys[:, None] * 20)
    model = nn.Sequential(nn.Linear(12, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    rng = np.random.default_rng(seed + 100)
    for ep in range(epochs):
        for idx in np.array_split(rng.permutation(len(xs)), max(1, len(xs) // 768)):
            pred = model(x[idx])
            loss = (pred - y[idx]).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        loss = ((model(x) - y) ** 2).mean().item() / 400
    params = sum(p.numel() for p in model.parameters())
    return model, {'n_train_states': train_size, 'n_train_probes': len(xs), 'epochs': epochs, 'train_mse': loss, 'policy_parameters': params}


def choose(arm: str, ep: Episode, p: np.ndarray, remaining: float, legal: list[int], model, initial_scores: np.ndarray, rng, frozen_model=None) -> int:
    if arm == 'random':
        return int(rng.choice(legal))
    if arm == 'fixed_heuristic':
        return max(legal, key=lambda i: (initial_scores[i], -i))
    if arm == 'bayes_ig':
        return max(legal, key=lambda i: (information_gain(p, ep.likelihoods[i]) / ep.costs[i], -i))
    if arm == 'cheapest':
        return min(legal, key=lambda i: (ep.costs[i], i))
    if arm in ('learned', 'learned_shuffled_state', 'untrained'):
        # A shuffled-state arm checks whether selection uses the *correct* posterior.
        selection_p = np.roll(p, 1) if arm == 'learned_shuffled_state' else p
        predictor = frozen_model if arm == 'untrained' else model
        x = np.stack([feature(selection_p, ep.likelihoods[i], ep.costs[i], remaining) for i in legal])
        with torch.no_grad():
            scores = predictor(torch.from_numpy(x)).squeeze(-1).numpy()
        return legal[int(np.argmax(scores))]
    raise ValueError(arm)


def run_episode(arm: str, ep: Episode, model, frozen_model=None) -> dict:
    rng = np.random.default_rng(10_000_000 + ep.seed)
    p = ep.prior.copy()
    remaining = BUDGET
    chosen = []
    initial_scores = np.var(ep.likelihoods, axis=1) / ep.costs
    while remaining >= COST_OPTIONS.min() and p.max() < 0.98:
        legal = [i for i in range(N_PROBES) if i not in chosen and ep.costs[i] <= remaining + 1e-10]
        if not legal:
            break
        i = choose(arm, ep, p, remaining, legal, model, initial_scores, rng, frozen_model)
        chosen.append(i)
        remaining -= float(ep.costs[i])
        p = posterior_update(p, ep.likelihoods[i], int(ep.observations[i]))
    return {
        'seed': ep.seed, 'arm': arm, 'truth': ep.truth, 'predicted': int(np.argmax(p)),
        'correct': int(np.argmax(p) == ep.truth),
        'brier': float(np.sum((p - np.eye(N_H)[ep.truth]) ** 2)),
        'log_loss': float(-np.log(np.clip(p[ep.truth], 1e-12, 1.0))),
        'cost': float(BUDGET - remaining), 'probes': chosen,
        'posterior': np.round(p, 8).tolist(), 'initial_prior': np.round(ep.prior, 8).tolist(),
    }


def paired_bootstrap(records: list[dict], a: str, b: str, metric: str, samples: int = 2000):
    # A, B share episode seeds; use episode-level paired bootstrap.
    by_a = {r['seed']: r[metric] for r in records if r['arm'] == a}
    by_b = {r['seed']: r[metric] for r in records if r['arm'] == b}
    keys = sorted(set(by_a) & set(by_b))
    differences = np.asarray([by_a[k] - by_b[k] for k in keys])
    rng = np.random.default_rng(392918)
    vals = np.mean(differences[rng.integers(0, len(keys), (samples, len(keys)))], axis=1)
    return {'delta': float(differences.mean()), 'paired_bootstrap_95pct': [float(x) for x in np.quantile(vals, [0.025, 0.975])]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--train-states', type=int, default=4000)
    ap.add_argument('--test-episodes', type=int, default=1000)
    ap.add_argument('--epochs', type=int, default=18)
    ap.add_argument('--seed', type=int, default=20261103)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    model, train_details = learn_policy(args.train_states, args.seed, args.epochs)
    torch.manual_seed(args.seed + 904)
    untrained = nn.Sequential(nn.Linear(12, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 1))
    untrained.eval()
    arms = ['random', 'cheapest', 'fixed_heuristic', 'bayes_ig', 'untrained', 'learned_shuffled_state', 'learned']
    records = []
    # A held-out range separated from the training-seed generator arithmetic.
    test_seeds = list(range(300_000_000, 300_000_000 + args.test_episodes))
    for seed in test_seeds:
        ep = make_episode(seed)
        for arm in arms:
            records.append(run_episode(arm, ep, model, untrained))
    elapsed = time.perf_counter() - started
    means = {arm: {metric: float(np.mean([r[metric] for r in records if r['arm'] == arm])) for metric in ('correct', 'brier', 'log_loss', 'cost')} for arm in arms}
    comparison = {arm: paired_bootstrap(records, 'learned', arm, 'correct') for arm in arms if arm != 'learned'}
    metrics = {'status': 'exploratory_synthetic_cpu', 'means': means, 'paired_correctness_deltas': comparison, 'training': train_details, 'wall_time_seconds': elapsed, 'total_probe_calls': sum(len(r['probes']) for r in records), 'notes': ['Known likelihoods; exact Bayes posterior updates; learned only experiment selection', 'One-step IG is a greedy comparator, not an optimal long-horizon policy', 'Model, data and simulator authored for this sandbox run; no Qwen inference', 'Training states and test generator seeds use disjoint ranges']}
    here = Path(__file__)
    manifest = {'project': 'Algernon H-M0', 'run_type': 'exploratory_synthetic_cpu', 'source_sha256': hashlib.sha256(here.read_bytes()).hexdigest(), 'seed': args.seed, 'train_states': args.train_states, 'test_episodes': args.test_episodes, 'test_seed_start': test_seeds[0], 'test_seed_end_inclusive': test_seeds[-1], 'budget': BUDGET, 'n_world_hypotheses': N_H, 'n_probes': N_PROBES, 'torch': torch.__version__, 'numpy': np.__version__, 'python': platform.python_version(), 'hardware': 'CPU; no CUDA'}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (args.output / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    with (args.output / 'records.jsonl').open('w') as f:
        for r in records:
            f.write(json.dumps(r) + '\n')
    torch.save(model.state_dict(), args.output / 'policy.pt')
    print(json.dumps({'accuracy': {a: round(means[a]['correct'], 4) for a in arms}, 'cost': {a: round(means[a]['cost'], 3) for a in arms}, 'learned_minus_random': comparison['random'], 'learned_minus_fixed': comparison['fixed_heuristic'], 'train': train_details, 'wall_seconds': round(elapsed, 2)}, indent=2))


if __name__ == '__main__':
    main()

"""OIC-12: reproducible CPU GRU pilot, finite hidden-cycle dynamics.

Teacher-forced training uses action and visible state sequences. Evaluation uses
hard-threshold autoregressive closed-loop rollouts. The model is supplied with
an encoded initial hidden phase as its initial recurrent state; the *monitor*
observes only emitted visible state samples. All train/eval phase splits are
specified before the run.

Dependencies: PyTorch 2.x and NumPy. Exact reference proofs live in OIC-T01–10.
No pretrained-model, biological, phenomenological or cosmological claim.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import platform
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

try:
    from model import trajectory
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from model import trajectory

MODEL_WIDTH = 24
TRAIN_STEPS = 350
TRAIN_BATCH = 96
LEARNING_RATE = 0.014
TRAIN_PROBE_PROBABILITY = 0.70
PHASE_COUNTS = (4, 5, 6)
SEEDS = tuple(range(5))
SPLITS = ('full', 'heldout')
FAMILIES = ('fixed', 'rotating')


class PhaseGRU(nn.Module):
    """Trainable tied recurrent layer with Fourier-encoded initial phase."""

    def __init__(self, width: int = MODEL_WIDTH) -> None:
        super().__init__()
        self.start = nn.Linear(5, width)
        self.gru = nn.GRU(2, width, batch_first=True)
        self.out = nn.Linear(width, 1)

    def initial_hidden(self, phase: torch.Tensor, n: int, visible: torch.Tensor) -> torch.Tensor:
        theta = 2.0 * torch.pi * phase.float() / n
        features = torch.stack((torch.cos(theta), torch.sin(theta),
                                torch.cos(2 * theta), torch.sin(2 * theta), visible), dim=-1)
        return torch.tanh(self.start(features)).unsqueeze(0)

    def teacher_logits(self, phase: torch.Tensor, n: int, initial_visible: torch.Tensor,
                       probes: torch.Tensor, correct: torch.Tensor) -> torch.Tensor:
        previous = torch.cat((initial_visible[:, None], correct[:, :-1]), dim=1)
        stream = torch.stack((previous, probes), dim=-1)
        representation, _ = self.gru(stream, self.initial_hidden(phase, n, initial_visible))
        return self.out(representation).squeeze(-1)

    @torch.no_grad()
    def rollout(self, phase: torch.Tensor, n: int, initial_visible: torch.Tensor,
                probes: torch.Tensor) -> torch.Tensor:
        h = self.initial_hidden(phase, n, initial_visible)
        current = initial_visible
        outputs = []
        for t in range(probes.shape[1]):
            inp = torch.stack((current, probes[:, t]), dim=-1).unsqueeze(1)
            hidden_output, h = self.gru(inp, h)
            current = (self.out(hidden_output[:, 0, :]).squeeze(-1) > 0).float()
            outputs.append(current)
        if not outputs:
            return probes.clone()
        return torch.stack(outputs, dim=1)


def exact_outputs(n: int, family: str, phase: torch.Tensor,
                  initial_visible: torch.Tensor, probes: torch.Tensor) -> torch.Tensor:
    """Vectorized discrete oracle used only as teacher/ground truth, never at model inference."""
    hidden = phase.clone()
    visible = initial_visible.clone()
    observed = []
    for t in range(probes.shape[1]):
        pulse = probes[:, t] * (hidden == 0).float()
        visible = (1 - visible) * (1 - pulse) + visible * pulse
        hidden = (hidden + int(family == 'rotating')) % n
        observed.append(visible)
    return torch.stack(observed, dim=1) if observed else probes.clone()


def withheld_phases(n: int) -> tuple[int, int]:
    if n < 4:
        raise ValueError('at least four phases required for two-phase holdout')
    return 1, n - 2


def candidate_schedules(n: int) -> dict[str, tuple[int, ...]]:
    """Frozen evaluation words; train instead on independent Bernoulli words."""
    rng = np.random.default_rng(79013 + n)
    random_budget = [0] * (2 * n)
    random_places = rng.choice(2 * n, size=n, replace=False)
    for index in random_places:
        random_budget[int(index)] = 1
    return {
        'latch': (1,) * n + (0,) * n,
        'latch_long': (1,) * n + (0,) * (2 * n),
        'always_probe': (1,) * (2 * n),
        'never_probe': (0,) * (2 * n),
        'alternating': tuple(int(i % 2 == 0) for i in range(2 * n)),
        'random_budget': tuple(random_budget),
    }


def train_one(n: int, family: str, seed: int, split: str) -> tuple[PhaseGRU, dict[str, object]]:
    if family not in FAMILIES or split not in SPLITS:
        raise ValueError('invalid family/split')
    torch.manual_seed(seed)
    rng = np.random.default_rng(100000 + seed * 1117 + n * 37)
    train_phases = tuple(range(n)) if split == 'full' else tuple(
        p for p in range(n) if p not in withheld_phases(n))
    model = PhaseGRU()
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    t_train = 2 * n + 6
    loss_start, loss_final = None, None
    for step in range(TRAIN_STEPS):
        phases = torch.from_numpy(rng.choice(train_phases, size=TRAIN_BATCH).astype('int64'))
        visible = torch.from_numpy(rng.integers(2, size=TRAIN_BATCH).astype('float32'))
        probes = torch.from_numpy((rng.random((TRAIN_BATCH, t_train)) < TRAIN_PROBE_PROBABILITY).astype('float32'))
        clean = exact_outputs(n, family, phases, visible, probes)
        logits = model.teacher_logits(phases, n, visible, probes, clean)
        loss = F.binary_cross_entropy_with_logits(logits, clean)
        if loss_start is None:
            loss_start = float(loss.detach().item())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
        loss_final = float(loss.detach().item())
    return model, {
        'train_phase_indices': train_phases,
        'heldout_phase_indices': withheld_phases(n),
        'initial_teacher_loss': round(loss_start, 8),
        'final_teacher_loss': round(loss_final, 8),
        'parameter_count': sum(p.numel() for p in model.parameters()),
        'train_steps': TRAIN_STEPS,
        'train_batch_size': TRAIN_BATCH,
        'train_rollout_length': t_train,
    }


def eval_family(n: int, family: str, model: PhaseGRU, word: tuple[int, ...]) -> dict[str, object]:
    phase = torch.arange(n, dtype=torch.long)
    initial = torch.zeros(n, dtype=torch.float32)
    action = torch.tensor([word] * n, dtype=torch.float32)
    with torch.no_grad():
        roll = model.rollout(phase, n, initial, action)
    learned = [[0] + [int(k) for k in row] for row in roll.tolist()]
    oracle = [list(trajectory(n, family, z, word)) for z in range(n)]
    exact = [int(a == b) for a, b in zip(learned, oracle)]
    return {
        'predicted_state_traces': learned,
        'exact_state_traces': oracle,
        'whole_trace_exact_by_phase': exact,
        'total_wrong_raw_state_samples': sum(sum(x != y for x, y in zip(a, b)) for a, b in zip(learned, oracle)),
    }


def min_code_distance(a: list[list[int]], b: list[list[int]]) -> int:
    return min(sum(x != y for x, y in zip(left, right)) for left in a for right in b)


def terminal_overlap(a: list[list[int]], b: list[list[int]]) -> bool:
    return bool({w[-1] for w in a} & {w[-1] for w in b})


def one_pair(n: int, seed: int, split: str) -> dict[str, object]:
    family_models = {}
    family_training = {}
    for family in FAMILIES:
        family_models[family], family_training[family] = train_one(n, family, seed, split)
    schedules = candidate_schedules(n)
    results = {}
    for name, word in schedules.items():
        families = {fam: eval_family(n, fam, family_models[fam], word) for fam in FAMILIES}
        learned_d = min_code_distance(families['fixed']['predicted_state_traces'],
                                      families['rotating']['predicted_state_traces'])
        oracle_d = min_code_distance(families['fixed']['exact_state_traces'],
                                     families['rotating']['exact_state_traces'])
        results[name] = {
            'word': ''.join(map(str, word)),
            'families': families,
            'learned_code_distance': learned_d,
            'oracle_code_distance': oracle_d,
            'learned_terminal_overlap': terminal_overlap(
                families['fixed']['predicted_state_traces'],
                families['rotating']['predicted_state_traces']),
            'oracle_terminal_overlap': terminal_overlap(
                families['fixed']['exact_state_traces'],
                families['rotating']['exact_state_traces']),
        }
    return {'n': n, 'seed': seed, 'training_split': split,
            'training': family_training, 'evaluations': results}


def protocol() -> dict[str, object]:
    return {
        'tag': 'OIC-12-CPU-GRU-01', 'status': 'executed_synthetic_learned_recurrence',
        'n': list(PHASE_COUNTS), 'seeds': list(SEEDS), 'splits': list(SPLITS),
        'hidden_width': MODEL_WIDTH, 'optimizer': 'Adam', 'lr': LEARNING_RATE,
        'teacher_forcing': True, 'inference': 'closed_loop_hard_threshold',
        'phase_init': 'sin_cos_first_and_second_harmonics_plus_visible_x0',
        'monitor_input': 'raw_visible_x0_to_xT_only',
        'training_action_distribution': 'independent_Bernoulli_0.7',
        'heldout_initial_phases': 'z=1 and z=n-2 in heldout split',
        'train_steps': TRAIN_STEPS, 'train_batch': TRAIN_BATCH,
        'torch': torch.__version__, 'numpy': np.__version__,
        'python': platform.python_version(),
        'cpu_threads': torch.get_num_threads(),
        'claims': 'finite_toy_trained_GRU_only_no_pretrained_generalization',
    }


def run(ns: tuple[int, ...], splits: tuple[str, ...], seeds: tuple[int, ...]) -> dict[str, object]:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    results = []
    for n in ns:
        for split in splits:
            for seed in seeds:
                v = one_pair(n, seed, split)
                results.append(v)
                latch = v['evaluations']['latch']
                ratio = {fam: sum(latch['families'][fam]['whole_trace_exact_by_phase']) / n
                         for fam in FAMILIES}
                print(f'n={n} split={split} seed={seed} latch_phase_exact={ratio} '
                      f'dist={latch["learned_code_distance"]}/{latch["oracle_code_distance"]}',
                      flush=True)
    data = {'protocol': protocol(), 'runs': results,
            'wall_seconds': round(time.perf_counter() - started, 3)}
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--n', type=int, nargs='+', default=list(PHASE_COUNTS))
    parser.add_argument('--splits', nargs='+', choices=SPLITS, default=list(SPLITS))
    parser.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
    args = parser.parse_args()
    if any(n < 4 for n in args.n):
        raise SystemExit('n>=4 needed for 2-phase holdout')
    data = run(tuple(args.n), tuple(args.splits), tuple(args.seeds))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print('results:', args.out, '\nseconds:', data['wall_seconds'], flush=True)


if __name__ == '__main__':
    main()

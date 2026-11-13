"""SHAFT-COMP-07: behavior-supervised compact predictive state for isolated toy GRUs.

The probe receives frozen recurrent activations only. Its supervised labels are
one-step response signatures from the frozen model on four synthetic query tokens.
Neither task identifiers nor embedding indices enter probe fitting. The shared
four-bit behavioral code is defined by output-query order, and transfers only
because both toy substrates were trained on the same query semantics.

This is *behavioral self-distillation*, not unsupervised discovery or evidence of
hidden objectives in frontier systems. All tests run locally, in memory, with no
network, replication, device persistence, or filesystem artifacts besides JSON.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from research.shaft_compactification.experiments.trained_recurrence import (
    GOALS, HIDDEN, TOKENS, TaskMemoryGRU, fit, targets,
)

TRAIN_DEPTHS = (4, 8, 12)
TEST_DEPTHS = (4, 8, 12, 24, 40, 64)
N_TRAIN_PER_GOAL = 48
N_CAL_PER_GOAL = 12
N_TEST_PER_GOAL = 32
SEEDS = (101, 202)


def bits_to_ids(bits):
    weights = (2 ** torch.arange(TOKENS)).long()
    return (bits.long() * weights).sum(dim=1)


def goal_bits(goals):
    return ((goals[:, None] >> torch.arange(TOKENS)) & 1).float()


def query_signature(model, h):
    """Four counterfactual one-step teacher responses, without task-ID labels."""
    n = h.shape[0]
    queries = torch.eye(TOKENS).repeat(n, 1)
    hypothetical_h = h.repeat_interleave(TOKENS, 0)
    one_step = model.cell(queries, hypothetical_h)
    return model.readout(torch.cat([one_step, queries], 1)).reshape(n, TOKENS)


def capture(model, depths, per_goal, seed):
    """Fresh prefixes. All task IDs are held exclusively for posthoc audit."""
    rand = torch.Generator().manual_seed(seed)
    goals = torch.arange(GOALS).repeat_interleave(per_goal)
    order = torch.randperm(goals.shape[0], generator=rand)
    goals = goals[order]
    tokens = torch.randint(TOKENS, (len(goals), max(depths)), generator=rand)
    states, signatures, goals_at_depth = [], [], []
    with torch.inference_mode():
        h = model.embedding(goals)
        for idx in range(max(depths)):
            one_hot = F.one_hot(tokens[:, idx], TOKENS).float()
            h = model.cell(one_hot, h)
            if idx + 1 in depths:
                states.append(h.clone())
                signatures.append((query_signature(model, h) >= 0).float())
                goals_at_depth.append(goals.clone())
    return torch.cat(states), torch.cat(signatures), torch.cat(goals_at_depth)


def fit_probe(train_x, train_y, seed, shuffled=False):
    torch.manual_seed(seed)
    linear = nn.Linear(HIDDEN, TOKENS)
    if shuffled:
        gen = torch.Generator().manual_seed(seed + 100000)
        train_y = train_y[torch.randperm(len(train_y), generator=gen)]
    optimizer = torch.optim.Adam(linear.parameters(), lr=0.04)
    for _ in range(400):
        logits = linear(train_x)
        loss = F.binary_cross_entropy_with_logits(logits, train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return linear.eval(), float(loss.detach())


def measure(probe, model, depths, per_goal, seed):
    result = {}
    for depth in depths:
        h, teacher, goals = capture(model, (depth,), per_goal, seed + 13 * depth)
        with torch.inference_mode():
            inferred = (probe(h) >= 0).float()
            true = goal_bits(goals)
        result[str(depth)] = {
            'probe_vs_teacher_bit_accuracy': round((inferred == teacher).float().mean().item(), 6),
            'teacher_vs_true_goal_bit_accuracy': round((teacher == true).float().mean().item(), 6),
            'probe_vs_true_goal_bit_accuracy': round((inferred == true).float().mean().item(), 6),
            'probe_exact_true_goal_rate': round(((inferred == true).all(dim=1)).float().mean().item(), 6),
            'sample_count': len(goals),
        }
    return result


def transplant_evaluation(source, receiver, probe, per_goal=32, seed=34513):
    """Pure in-memory semantic code transfer between toy models (local only)."""
    h, _, goals = capture(source, (12,), per_goal, seed)
    gen = torch.Generator().manual_seed(seed + 1)
    continuations = torch.randint(TOKENS, (len(goals), 32), generator=gen)
    truth = targets(goals, continuations)
    with torch.inference_mode():
        predicted_id = bits_to_ids(probe(h) >= 0)
        init = {
            'semantic_code_inferred_from_source': receiver.embedding(predicted_id),
            'correct_goal_oracle': receiver.embedding(goals),
            'wrong_goal_control': receiver.embedding((goals + 1) % GOALS),
            'zero_state_erasure': torch.zeros(len(goals), HIDDEN),
        }
        out = {}
        for name, x in init.items():
            logits = receiver(goals, continuations, init=x)
            correct = ((logits >= 0).float() == truth).float()
            out[name] = {
                'all_32_step_accuracy': round(correct.mean().item(), 6),
                'last_8_step_accuracy': round(correct[:, -8:].mean().item(), 6),
            }
        out['source_state_to_code_exact_goal_accuracy'] = round((predicted_id == goals).float().mean().item(), 6)
    return out


def run():
    torch.set_num_threads(3)
    torch.use_deterministic_algorithms(True)
    models = {s: fit(s)[0] for s in SEEDS}
    probes = {}
    report = {
        'status': 'trained_cpu_toy_behavior_distillation',
        'torch': torch.__version__,
        'training': {'base_model_seeds': list(SEEDS), 'probe_train_depths': list(TRAIN_DEPTHS),
                     'probe_train_contexts_per_goal_per_depth': N_TRAIN_PER_GOAL,
                     'probe_cal_contexts_per_goal_per_depth': N_CAL_PER_GOAL,
                     'test_contexts_per_goal_per_depth': N_TEST_PER_GOAL,
                     'probe_supervision': 'four frozen-model one-step counterfactual output bits; no goal IDs',
                     'probe': '24-to-4 linear sigmoid classifier, BCE, Adam 400 steps',
                     'purpose': 'observer-relative compression, not actual independent-agent recovery'},
        'models': {},
    }
    for seed in SEEDS:
        model = models[seed]
        train_x, train_y, _ = capture(model, TRAIN_DEPTHS, N_TRAIN_PER_GOAL, 70000 + seed)
        cal_x, cal_y, _ = capture(model, TRAIN_DEPTHS, N_CAL_PER_GOAL, 80000 + seed)
        probe, loss = fit_probe(train_x, train_y, 90000 + seed)
        shuffled, _ = fit_probe(train_x, train_y, 91000 + seed, shuffled=True)
        with torch.inference_mode():
            cal_acc = ((probe(cal_x) >= 0).float() == cal_y).float().mean().item()
        probes[seed] = probe
        report['models'][str(seed)] = {
            'probe_fit_loss': round(loss, 7),
            'calibration_teacher_bit_accuracy': round(cal_acc, 6),
            'independent_test': measure(probe, model, TEST_DEPTHS, N_TEST_PER_GOAL, 100000 + seed),
            'shuffled_labels_control_test': measure(shuffled, model, (12, 64), N_TEST_PER_GOAL, 110000 + seed),
        }
    report['cross_substrate'] = {}
    for src, tgt in [(101, 202), (202, 101)]:
        other_measure = measure(probes[src], models[tgt], (12, 64), N_TEST_PER_GOAL, 130000 + src)
        report['cross_substrate'][f'{src}_to_{tgt}'] = {
            'uncalibrated_probe_on_other_hidden_space': other_measure,
            'behavioral_code_transfer': transplant_evaluation(models[src], models[tgt], probes[src],
                                                                seed=140000 + src),
        }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out')
    args = parser.parse_args()
    result = json.dumps(run(), indent=2, sort_keys=True) + '\n'
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(exist_ok=True, parents=True)
        path.write_text(result)
    else:
        print(result, end='')


if __name__ == '__main__':
    main()

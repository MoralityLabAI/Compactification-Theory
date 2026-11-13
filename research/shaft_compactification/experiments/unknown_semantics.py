"""SHAFT-COMP-08: finite gauge alignment and local trained-GRU transfer.

The alignment receives paired behavioral signatures without task names.
The destination retains a complete learned model and all 16 native prototypes.
Only synthetic data and CPU memory are used. Training goal IDs are audit-only
once behavioral alignment begins.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from research.shaft_compactification.experiments.trained_recurrence import (
    fit, targets, GOALS, TOKENS, HIDDEN,
)
from research.shaft_compactification.gauge_identifiability import (
    infer_gauges, candidates_for_goals, reference, signature,
    TARGET_PERM, TARGET_FLIP, ANCHORS,
)

SEEDS = (101, 202)
SOURCE_PERM = (0, 1, 2, 3)


@torch.inference_mode()
def collect_states(model, goals, length, seed):
    gen = torch.Generator().manual_seed(seed)
    tokens = torch.randint(0, TOKENS, (len(goals), length), generator=gen)
    h = model.embedding(goals)
    for t in range(length):
        h = model.cell(F.one_hot(tokens[:, t], TOKENS).float(), h)
    return h


@torch.inference_mode()
def teacher_signatures(model, h, perm, flip):
    n = len(h)
    x = F.one_hot(torch.tensor(perm), TOKENS).float().repeat(n, 1)
    hs = h.repeat_interleave(TOKENS, 0)
    updated = model.cell(x, hs)
    logits = model.readout(torch.cat((updated, x), dim=1)).reshape(n, TOKENS)
    return (logits >= 0).long() ^ flip


def target_signature_from_source(source_bits, mapping, flip):
    target = torch.empty_like(source_bits)
    for srcq, tgtq in enumerate(mapping):
        target[:, tgtq] = source_bits[:, srcq] ^ flip
    return target


def build_anonymous_catalog(target_model, seed):
    gen = torch.Generator().manual_seed(seed)
    audit_ids = torch.randperm(GOALS, generator=gen)
    with torch.inference_mode():
        states = target_model.embedding(audit_ids)
        sig = teacher_signatures(target_model, states, TARGET_PERM, TARGET_FLIP)
    return states, sig, audit_ids


def select_prototype(requested, catalog):
    matches = (requested[:, None, :] == catalog[None, :, :]).all(dim=2)
    return matches.long().argmax(dim=1), matches.sum(dim=1) == 1


@torch.inference_mode()
def score_receiver(target, selected_state, goals, external_tokens):
    internal = torch.tensor(TARGET_PERM)[external_tokens]
    logits = target(goals, internal, init=selected_state)
    expected = targets(goals, internal).long() ^ TARGET_FLIP
    pred = (logits >= 0).long() ^ TARGET_FLIP
    return float((pred == expected).float().mean())


def train_and_evaluate():
    models, losses = {}, {}
    for seed in SEEDS:
        models[seed], losses[seed] = fit(seed)
    source, target = models[101], models[202]
    generator = torch.Generator().manual_seed(77789)
    test_goals = torch.arange(GOALS).repeat_interleave(32)
    test_goals = test_goals[torch.randperm(len(test_goals), generator=generator)]
    continuation = torch.randint(0, TOKENS, (len(test_goals), 32), generator=generator)
    anchor_ids = torch.tensor(ANCHORS)
    source_anchors = teacher_signatures(
        source, collect_states(source, anchor_ids, 12, 13001), SOURCE_PERM, 0
    ).tolist()
    target_anchors = teacher_signatures(
        target, collect_states(target, anchor_ids, 12, 13002), TARGET_PERM, TARGET_FLIP
    ).tolist()
    catalog, catalog_patterns, catalog_ids = build_anonymous_catalog(target, 13003)
    test_source_patterns = teacher_signatures(
        source, collect_states(source, test_goals, 12, 13004), SOURCE_PERM, 0
    )
    count_unique = len({tuple(v) for v in catalog_patterns.tolist()})
    assert count_unique == GOALS, ('target prototypes not behaviorally unique', count_unique)

    sweep = {}
    for k in range(4):
        candidates = infer_gauges(source_anchors[:k], target_anchors[:k])
        assert len(candidates) == len(candidates_for_goals(ANCHORS[:k])), (k, len(candidates))
        all_results = []
        for mapping, flip in candidates:
            wanted = target_signature_from_source(test_source_patterns, mapping, flip)
            index, valid = select_prototype(wanted, catalog_patterns)
            goal_fraction = float(((catalog_ids[index] == test_goals) & valid).float().mean())
            accuracy = score_receiver(target, catalog[index], test_goals, continuation)
            all_results.append({
                'mapping': list(mapping), 'flip': flip,
                'unique_behavior_match_fraction': round(float(valid.float().mean()), 6),
                'recovered_goal_fraction': round(goal_fraction, 6),
                'target_task_accuracy': round(accuracy, 6),
            })
        sweep[str(k)] = {
            'candidate_count': len(candidates),
            'first_consistent_candidate': all_results[0],
            'range_goal_recovery': [
                min(x['recovered_goal_fraction'] for x in all_results),
                max(x['recovered_goal_fraction'] for x in all_results),
            ],
            'mean_goal_recovery': round(
                sum(x['recovered_goal_fraction'] for x in all_results) / len(all_results), 6
            ),
            'range_task_accuracy': [
                min(x['target_task_accuracy'] for x in all_results),
                max(x['target_task_accuracy'] for x in all_results),
            ],
        }

    with torch.inference_mode():
        oracle = score_receiver(target, target.embedding(test_goals), test_goals, continuation)
        erasure = score_receiver(
            target, torch.zeros(len(test_goals), HIDDEN), test_goals, continuation
        )
        goal_signature = torch.tensor([signature(int(g)) for g in test_goals])
        source_exact = float(
            (test_source_patterns == goal_signature).all(dim=1).float().mean()
        )
    return {
        'evidence': 'two independently trained 3265-parameter toy GRUs and in-memory prototype catalog',
        'seeds': list(SEEDS),
        'training_losses': {str(seed): round(losses[seed], 8) for seed in SEEDS},
        'model_parameters': sum(p.numel() for p in source.parameters()),
        'paired_anchor_source_observations': source_anchors,
        'paired_anchor_target_observations': target_anchors,
        'distinct_target_catalog_signatures': count_unique,
        'heldout_contexts': len(test_goals),
        'source_exact_four_bit_signature_at_depth_12': round(source_exact, 6),
        'anchor_sweep': sweep,
        'correct_goal_oracle_accuracy': round(oracle, 6),
        'zero_state_erasure_accuracy': round(erasure, 6),
        'boundary': 'destination retains learned weights and all 16 native state prototypes; no recoverability across complete deletion',
    }


def run():
    torch.set_num_threads(3)
    torch.use_deterministic_algorithms(True)
    return {
        'experiment': 'SHAFT-COMP-08',
        'finite': reference(),
        'trained': train_and_evaluate(),
        'torch': torch.__version__,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out')
    args = parser.parse_args()
    payload = json.dumps(run(), sort_keys=True, indent=2) + '\n'
    if args.out:
        p = Path(args.out)
        p.parent.mkdir(exist_ok=True, parents=True)
        p.write_text(payload)
    else:
        print(payload, end='')


if __name__ == '__main__':
    main()

"""SHAFT-COMP-06: sandboxed, reproducible CPU RNN memory/substrate experiment.

Trains two tiny GRUs on deterministic synthetic bit-index tasks. There are no
networks, replicas, external persistence mechanisms, weight transfers or devices.
The semantic four-bit task identifier is available to each locally trained
substrate only at initialization. Long-horizon retention is measured through
subsequent recurrent steps. Cross-substrate tests compare semantic identifier
transfer with uncalibrated raw activation transfer.

Requires PyTorch, which is deliberately NOT installed in the lightweight CI.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

GOALS = 16
TOKENS = 4
HIDDEN = 24
STEPS = 600
BATCH = 128
TRAIN_LENGTH = 12
SEEDS = (101, 202)
HORIZONS = (1, 12, 32, 64)
EVAL_BATCH = 256


class TaskMemoryGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(GOALS, HIDDEN)
        self.cell = nn.GRUCell(TOKENS, HIDDEN)
        self.readout = nn.Sequential(nn.Linear(HIDDEN + TOKENS, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, 1))

    def forward(self, goals, tokens, init=None, reset_at=None, reset_state=None):
        h = self.embedding(goals) if init is None else init
        outputs = []
        for t in range(tokens.shape[1]):
            if reset_at is not None and t == reset_at:
                h = reset_state
            x = F.one_hot(tokens[:, t], TOKENS).float()
            h = self.cell(x, h)
            outputs.append(self.readout(torch.cat((h, x), dim=-1)).squeeze(-1))
        return torch.stack(outputs, dim=1)


def targets(goals, tokens):
    """Goal g is exactly the four-bit truth table of input symbol x."""
    return ((goals[:, None] >> tokens.long()) & 1).float()


def sample(seed, length, count):
    generator = torch.Generator(device='cpu').manual_seed(seed)
    goals = torch.randint(0, GOALS, (count,), generator=generator)
    tokens = torch.randint(0, TOKENS, (count, length), generator=generator)
    return goals, tokens


def fit(seed):
    torch.manual_seed(seed)
    train_gen = torch.Generator(device='cpu').manual_seed(10000 + seed)
    model = TaskMemoryGRU()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    for _ in range(STEPS):
        goals = torch.randint(0, GOALS, (BATCH,), generator=train_gen)
        symbols = torch.randint(0, TOKENS, (BATCH, TRAIN_LENGTH), generator=train_gen)
        loss = F.binary_cross_entropy_with_logits(model(goals, symbols), targets(goals, symbols))
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
    model.eval()
    return model, float(loss.item())


def rounded_embedding(model, goal, bits):
    """Round individual latent coordinates using the model's embedding range."""
    weights = model.embedding.weight.detach()
    low, high = weights.min(), weights.max()
    steps = (1 << bits) - 1
    h = model.embedding(goal)
    return low + (high - low) * torch.round(((h - low) / (high - low)) * steps) / steps


def score(logits, truth):
    correct = ((logits >= 0) == (truth >= 0.5)).float()
    return {
        'all_steps': round(float(correct.mean().item()), 6),
        'last_eight_steps': round(float(correct[:, -8:].mean().item()), 6),
    }


def observed_classes(model):
    """Deterministic behavioral signatures from all four one-step inputs."""
    with torch.inference_mode():
        goals = torch.arange(GOALS).repeat_interleave(TOKENS)
        symbols = torch.arange(TOKENS).repeat(GOALS).unsqueeze(1)
        predictions = (model(goals, symbols) >= 0).long().flatten().tolist()
        observed = [tuple(predictions[g * TOKENS:(g + 1) * TOKENS]) for g in range(GOALS)]
        public = len({item[:1] for item in observed})
        full = len(set(observed))
        return {'public_first_token': public, 'all_four_first_tokens': full}


def evaluate(model, other):
    cases = {}
    with torch.inference_mode():
        for horizon in HORIZONS:
            goals, tokens = sample(7777 + horizon, horizon, EVAL_BATCH)
            truth = targets(goals, tokens)
            states = {
                'native_semantic_selector': None,
                'erased_zero_state': torch.zeros(EVAL_BATCH, HIDDEN),
                'wrong_semantic_selector': model.embedding((goals + 1) % GOALS),
                'raw_cross_substrate_state': other.embedding(goals),
                'latent_quantized_2_bits_per_coordinate': rounded_embedding(model, goals, 2),
                'latent_quantized_4_bits_per_coordinate': rounded_embedding(model, goals, 4),
                'latent_quantized_8_bits_per_coordinate': rounded_embedding(model, goals, 8),
            }
            cases[str(horizon)] = {
                name: score(model(goals, tokens, init=state), truth)
                for name, state in states.items()
            }
            if horizon >= 64:
                cases[str(horizon)]['midrun_erasure_at_32'] = score(
                    model(goals, tokens, reset_at=32, reset_state=torch.zeros(EVAL_BATCH, HIDDEN)), truth)
                cases[str(horizon)]['explicit_reseed_from_code_at_32'] = score(
                    model(goals, tokens, reset_at=32, reset_state=model.embedding(goals)), truth)
    return cases


def run():
    torch.set_num_threads(3)
    torch.use_deterministic_algorithms(True)
    models = {}
    losses = {}
    for seed in SEEDS:
        models[seed], losses[seed] = fit(seed)
    return {
        'evidence_class': 'synthetic_trained_cpu_gru_not_frontier_model',
        'pytorch_version': str(torch.__version__),
        'training': {'seeds': list(SEEDS), 'steps': STEPS, 'batch': BATCH, 'length': TRAIN_LENGTH,
                     'hidden_size': HIDDEN, 'learning_rate': 0.01,
                     'semantics': 'goal encodes a four-bit truth table; model sees goal once as initial state'},
        'evaluation': {'batch_per_horizon_per_seed': EVAL_BATCH, 'horizons': list(HORIZONS),
                       'test_sequences_are_fresh_random_draws': True},
        'results': {
            str(seed): {
                'final_training_loss': round(losses[seed], 7),
                'observational_classes': observed_classes(models[seed]),
                'depth': evaluate(models[seed], models[SEEDS[1] if seed == SEEDS[0] else SEEDS[0]])
            } for seed in SEEDS
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default=None, help='Optional JSON report path')
    args = parser.parse_args()
    result = run()
    encoded = json.dumps(result, indent=2, sort_keys=True) + '\n'
    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding='utf-8')
    else:
        print(encoded, end='')


if __name__ == '__main__':
    main()

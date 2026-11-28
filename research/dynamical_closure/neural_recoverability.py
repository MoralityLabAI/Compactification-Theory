"""Small CPU recurrent-attention falsifier on exactly known 3-bit dynamics.

Run with: apy research/dynamical_closure/neural_recoverability.py --steps 300
All states and one-step actions are trained; two action *compositions* are held out.
No pretrained model, GPU, network, or model API is used.
"""
import argparse
import json
import time
from collections import defaultdict
from fractions import Fraction
from itertools import product

import torch
from torch import nn

from recoverability_experiment import STATES, build_system, first_failure
from task_closure import bayes_risk, run_word


class RecurrentAttention(nn.Module):
    """Causal two-token attention block; next action token reads recurrent state."""
    def __init__(self, width=24):
        super().__init__()
        self.encode = nn.Linear(3, width)
        self.actions = nn.Embedding(5, width)
        self.attention = nn.MultiheadAttention(width, 2, batch_first=True)
        self.attn_norm = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(),
                                nn.Linear(2 * width, width))
        self.ff_norm = nn.LayerNorm(width)
        self.readout = nn.Linear(width, 1)

    def initial(self, bits):
        return torch.tanh(self.encode(bits))

    def step(self, state, action):
        a = self.actions(action)
        tokens = torch.stack((state, a), dim=1)
        # The action query sees the earlier state and itself, not future tokens.
        update, _ = self.attention(tokens[:, 1:], tokens, tokens, need_weights=False)
        state = self.attn_norm(state + a + update[:, 0])
        return self.ff_norm(state + self.ff(state))

    def forward_batch(self, bits, action_columns):
        state = self.initial(bits)
        outputs = [self.readout(state).squeeze(-1)]
        for actions in action_columns:
            state = self.step(state, actions)
            outputs.append(self.readout(state).squeeze(-1))
        return torch.stack(outputs, dim=1), state

    def forward(self, bits, word):
        columns = [torch.full((len(bits),), action, dtype=torch.long) for action in word]
        return self.forward_batch(bits, columns)


def words_and_targets(system, names):
    heldout = ((names.index("flip_b"), names.index("reveal_b")),
               (names.index("flip_y"), names.index("reveal_b")))
    all_words = [(a,) for a in range(len(names))]
    all_words += [w for w in product(range(len(names)), repeat=2) if w not in heldout]
    targets = {}
    for word in all_words + list(heldout):
        stages = [tuple(system.task[run_word(system, x, word[:t])]
                        for x in range(len(STATES))) for t in range(len(word) + 1)]
        targets[word] = torch.tensor(stages, dtype=torch.float32).T
    return all_words, heldout, targets


def decode_bits(train_hidden, train_labels, test_hidden):
    """Fixed linear least-squares readout with intercept; one fit, no test refit."""
    a = torch.cat((train_hidden, torch.ones(len(train_hidden), 1)), dim=1)
    b = torch.cat((test_hidden, torch.ones(len(test_hidden), 1)), dim=1)
    weights = torch.linalg.lstsq(a, train_labels).solution
    return (b @ weights >= 0.5).int(), torch.max(torch.abs(a @ weights - train_labels)).item()


def fitted_code_composition(initial_codes, successors, word, targets):
    """Fit squared-loss transition tables from *single-action neural codes*."""
    tables = {}
    conflicts = 0
    for action, codes_after in successors.items():
        groups = defaultdict(list)
        for z, next_z in zip(initial_codes, codes_after):
            groups[z].append(next_z)
        conflicts += sum(len(set(values)) > 1 for values in groups.values())
        tables[action] = {z: tuple(Fraction(sum(v[j] for v in values), len(values))
                                    for j in range(len(z)))
                          for z, values in groups.items()}
    loss = Fraction()
    for z, target in zip(initial_codes, targets):
        for action in word:
            if z not in tables[action]:
                return {"mse": None, "nonclosed_fibers": conflicts,
                        "reason": "intermediate code outside fitted discrete table"}
            z = tables[action][z]
        predicted = z[0] if z else Fraction(1, 2)
        loss += (Fraction(target) - predicted) ** 2
    return {"mse": str(loss / len(targets)), "nonclosed_fibers": conflicts}


def run_seed(seed, steps=300, consistency=0.0):
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    system, names = build_system()
    train_words, heldout, targets = words_and_targets(system, names)
    bits = torch.tensor(STATES, dtype=torch.float32)
    model = RecurrentAttention()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.008, weight_decay=0)
    train_bits = bits.repeat(len(train_words), 1)
    # One vectorized full batch of all 8 states for every training word.
    padded = [w + (names.index("identity"),) * (2 - len(w)) for w in train_words]
    action_columns = [torch.tensor([w[t] for w in padded for _ in STATES])
                      for t in range(2)]
    train_targets = torch.stack([
        torch.cat((targets[w], targets[w][:, -1:].repeat(1, 3 - targets[w].shape[1])), dim=1)
        for w in train_words]).reshape(-1, 3)
    mask = torch.tensor([[True, True, len(w) == 2] for w in train_words
                         for _ in STATES])
    single_bits = bits.repeat(len(names), 1)
    single_actions = torch.tensor([a for a in range(len(names)) for _ in STATES])
    successors = torch.tensor([system.transitions[a][x] for a in range(len(names))
                               for x in range(len(STATES))])
    for _ in range(steps):
        model.train()
        optimizer.zero_grad()
        logits, _ = model.forward_batch(train_bits, action_columns)
        loss = nn.functional.binary_cross_entropy_with_logits(logits[mask], train_targets[mask])
        if consistency:
            # Single-action supervision only: canonicalize the recurrent state
            # to the encoder's state for the known microscopic successor.
            h_next = model.step(model.initial(single_bits), single_actions)
            h_target = model.initial(bits[successors]).detach()
            loss = loss + consistency * nn.functional.mse_loss(h_next, h_target)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        outputs = {word: model(bits, word) for word in train_words + list(heldout)}
        train_correct = sum(int(((torch.sigmoid(outputs[w][0][:, -1]) >= .5) ==
                                 targets[w][:, -1].bool()).sum()) for w in train_words)
        train_total = len(STATES) * len(train_words)
        heldout_counts = {"/".join(names[a] for a in w):
                          int(((torch.sigmoid(outputs[w][0][:, -1]) >= .5) ==
                               targets[w][:, -1].bool()).sum()) for w in heldout}
        # Probe the recurrent state at t=0 and after every trained one-step action.
        train_hidden = torch.cat([model(bits, ())[1]] +
                                 [outputs[(a,)][1] for a in range(len(names))])
        train_state = [tuple(run_word(system, x, w) for x in range(len(STATES)))
                       for w in [()] + [(a,) for a in range(len(names))]]
        train_labels = torch.tensor([STATES[x] for indices in train_state for x in indices],
                                    dtype=torch.float32)
        initial_hidden = model(bits, ())[1]
        initial_codes, probe_max_train_residual = decode_bits(train_hidden, train_labels, initial_hidden)
        codes = [tuple(tuple(map(int, row[:n])) for row in initial_codes.tolist())
                 for n in (3, 2, 1, 0)]
        single_decoded = {a: decode_bits(train_hidden, train_labels, outputs[(a,)][1])[0]
                          for a in range(len(names))}
        single_truth = {a: torch.tensor([STATES[system.transitions[a][x]]
                                         for x in range(len(STATES))], dtype=torch.int)
                        for a in range(len(names))}
        single_correct = sum(int((single_decoded[a] == single_truth[a]).sum())
                             for a in single_decoded)
        pilot = (names.index("reveal_b"),)
        def risks(word):
            ys = tuple(system.task[run_word(system, x, word)] for x in range(len(STATES)))
            return [bayes_risk(code, ys) for code in codes]
        pilot_risks = risks(pilot)
        tests = {}
        for word in heldout:
            key = "/".join(names[a] for a in word)
            predicted, _ = decode_bits(train_hidden, train_labels, outputs[word][1])
            state_after = torch.tensor([STATES[run_word(system, x, word)]
                                        for x in range(len(STATES))], dtype=torch.int)
            rr = risks(word)
            true_target = tuple(int(v) for v in targets[word][:, -1])
            fitted = [fitted_code_composition(code,
                      {a: tuple(tuple(int(v) for v in row[:n])
                                for row in single_decoded[a].tolist())
                       for a in range(len(names))}, word, true_target)
                      for n, code in zip((3, 2, 1, 0), codes)]
            tests[key] = {"model_correct_of_8": heldout_counts[key],
                          "probe_bits_correct_of_24": int((predicted == state_after).sum()),
                          "oracle_risk_by_code": [str(v) for v in rr],
                          "fitted_code_composition": fitted,
                          "threshold_k": first_failure(rr)}
        unseen_to_pilot = (names.index("reveal_c"),)
        rr = risks(unseen_to_pilot)
        tests["reveal_c_not_in_pilot"] = {"oracle_risk_by_code": [str(v) for v in rr],
                                            "threshold_k": first_failure(rr),
                                            "note": "Action was trained singly; excluded only from pilot risk."}
        return {"seed": seed, "steps": steps, "consistency_weight": consistency,
                "training_final_correct": train_correct,
                "training_final_total": train_total, "training_loss": round(loss.item(), 6),
                "initial_decoded_states_correct_of_8": int((initial_codes == bits.int()).all(1).sum()),
                "single_action_probe_bits_correct_of_120": single_correct,
                "probe_max_training_abs_residual": round(probe_max_train_residual, 6),
                "pilot_risk_by_code": [str(v) for v in pilot_risks],
                "pilot_threshold_k": first_failure(pilot_risks), "heldout": tests}


def run(seeds=(0, 1), steps=300, consistency=0.0):
    started = time.monotonic()
    return {"setup": "8 complete states; 5 trained single actions; 23 trained two-action words; 2 held-out two-action words",
            "model": "24-wide, 2-head recurrent attention block; AdamW lr=0.008",
            "codes": ["decoded (y,b,c)", "decoded (y,b)", "decoded y", "constant"],
            "risk_tolerance": "1/16", "seed_results": [run_seed(s, steps, consistency) for s in seeds],
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "torch_version": torch.__version__}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--consistency", type=float, default=0.0,
                        help="weight of single-action latent successor alignment")
    args = parser.parse_args()
    if args.steps < 1 or args.consistency < 0:
        parser.error("steps must be positive and consistency nonnegative")
    print(json.dumps(run(args.seeds, args.steps, args.consistency), indent=2, sort_keys=True))

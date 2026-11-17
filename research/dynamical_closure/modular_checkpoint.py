"""CPU reduced-scale Nanda-style modular addition checkpoint experiment.

Run: apy research/dynamical_closure/modular_checkpoint.py --steps 6000 --output results.json
Calibration seed 91 selects Fourier frequencies from its *final* logit spectrum;
confirmation seeds 0 and 1 never select frequencies from their own outcomes.
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from cyclic_addition import onehot_risk


class TinyTransformer(nn.Module):
    """Single causal attention layer + ReLU MLP, no LayerNorm or tied embeddings."""
    def __init__(self, p=31, d=32, width=64, heads=2):
        super().__init__()
        self.p, self.heads, self.dh = p, heads, d // heads
        self.embed = nn.Embedding(p + 1, d)
        self.pos = nn.Parameter(torch.randn(3, d) * .02)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.attn_out = nn.Linear(d, d, bias=False)
        self.mlp_in = nn.Linear(d, width)
        self.mlp_out = nn.Linear(width, d)
        self.unembed = nn.Linear(d, p, bias=False)

    def forward(self, pairs, hidden=False):
        eq = torch.full((len(pairs), 1), self.p, dtype=torch.long)
        x = self.embed(torch.cat((pairs, eq), dim=1)) + self.pos
        b, seq, d = x.shape
        q, k, v = self.qkv(x).reshape(b, seq, 3, self.heads, self.dh).unbind(2)
        q, k, v = (z.transpose(1, 2) for z in (q, k, v))
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.dh)
        scores = scores.masked_fill(torch.triu(torch.ones(3, 3, dtype=torch.bool), diagonal=1), -1e9)
        attn = (scores.softmax(-1) @ v).transpose(1, 2).reshape(b, seq, d)
        x = x + self.attn_out(attn)
        h = F.relu(self.mlp_in(x[:, 2]))
        logits = self.unembed(x[:, 2] + self.mlp_out(h))
        return (logits, h) if hidden else logits


def data(p, seed):
    a = torch.arange(p).repeat_interleave(p)
    b = torch.arange(p).repeat(p)
    pairs = torch.stack((a, b), -1)
    labels = (a + b) % p
    perm = torch.randperm(p * p, generator=torch.Generator().manual_seed(seed))
    train = perm[:round(.3 * p * p)].sort().values
    test = perm[round(.3 * p * p):].sort().values
    return pairs, labels, train, test


def select_calibration_frequencies(logits, p, count=3):
    ft = torch.fft.fft2(logits.reshape(p, p, p).double(), dim=(0, 1))
    strength = [(float(ft[j, j].abs().square().sum() + ft[-j, -j].abs().square().sum()), j)
                for j in range(1, (p + 1) // 2)]
    return tuple(j for _, j in sorted(strength, key=lambda x: (-x[0], x[1]))[:count])


def spectral_losses(logits, labels, train, test, p, selected, control):
    ft = torch.fft.fft2(logits.reshape(p, p, p).double(), dim=(0, 1))
    keep = torch.zeros((p, p), dtype=torch.bool)
    keep[0, 0] = True
    for j in selected:
        keep[j, j] = keep[-j, -j] = True
    restricted = torch.fft.ifft2(ft * keep[:, :, None], dim=(0, 1)).real.reshape(-1, p).float()
    excluded = torch.fft.ifft2(ft * (~keep)[:, :, None], dim=(0, 1)).real.reshape(-1, p).float()
    control_mask = torch.zeros_like(keep)
    control_mask[0, 0] = True
    for j in control:
        control_mask[j, j] = control_mask[-j, -j] = True
    control_logits = torch.fft.ifft2(ft * control_mask[:, :, None], dim=(0, 1)).real.reshape(-1, p).float()
    diag = torch.zeros_like(keep)
    for j in range(p):
        diag[j, j] = True
    all_diagonal = torch.fft.ifft2(ft * diag[:, :, None], dim=(0, 1)).real.reshape(-1, p).float()
    return {"restricted_test_ce": float(F.cross_entropy(restricted[test], labels[test])),
            "excluded_train_ce": float(F.cross_entropy(excluded[train], labels[train])),
            "matched_other_test_ce": float(F.cross_entropy(control_logits[test], labels[test])),
            "all_diagonal_test_ce": float(F.cross_entropy(all_diagonal[test], labels[test]))}


def quantized_codes(hidden, train, k, basis, mean, bits=8):
    if k == 0:
        return torch.zeros(len(hidden), 0, dtype=torch.int16)
    coords = (hidden - mean) @ basis[:, :k]
    sd = coords[train].std(0, unbiased=False).clamp_min(1e-6)
    radius = 2 ** (bits - 1) - 1
    return torch.round((coords / (3 * sd)) * radius).clamp(-radius, radius).to(torch.int16)


def compression_metrics(hidden, labels, train, test, p, seed, ks=(16, 8, 4, 2, 1, 0)):
    h = hidden.double()
    mean = h[train].mean(0)
    centered = h[train] - mean
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    pca = vh.T
    gen = torch.Generator().manual_seed(10000 + seed)
    random, _ = torch.linalg.qr(torch.randn(h.shape[1], max(ks), generator=gen, dtype=torch.double))
    target = (labels + 1) % p  # the same future-output target at every k and t
    target_onehot = F.one_hot(target, p).double()
    result = {}
    for name, basis in (("pca", pca), ("random", random)):
        bayes, linear_objective, decoder_test = [], [], []
        for k in ks:
            q = quantized_codes(h, train, k, basis, mean)
            codes = [tuple(int(v) for v in row) for row in q[train].tolist()]
            bayes.append(float(onehot_risk(codes, target[train].tolist())))
            # Constant-width affine decoder: unused inputs are zero-padded to 16.
            features = torch.zeros(len(h), max(ks) + 1, dtype=torch.double)
            features[:, 0] = 1
            features[:, 1:k+1] = q.double() / 127
            penalty = torch.diag(torch.tensor([0.] + [1e-3] * max(ks), dtype=torch.double))
            w = torch.linalg.solve(features[train].T @ features[train] + penalty,
                                   features[train].T @ target_onehot[train])
            train_error = ((features[train] @ w - target_onehot[train]) ** 2).sum(1).mean()
            linear_objective.append(float(train_error + (w * (penalty @ w)).sum() / len(train)))
            decoder_test.append(float(((features[test] @ w - target_onehot[test]) ** 2).sum(1).mean()))
        # Both threshold definitions and tolerances are frozen before evaluation.
        bstar = next((i for i, risk in enumerate(bayes) if risk > .1), None)
        lstar = next((i for i, value in enumerate(linear_objective) if value > .5), None)
        result[name] = {"bayes_train_risk": bayes, "bayes_k_star_index": bstar,
                        "linear_train_objective": linear_objective,
                        "linear_k_star_index": lstar, "linear_decoder_test_mse": decoder_test}
    return result


def train_run(seed, p, steps, checkpoints, selection=None, control=None, output=None):
    torch.manual_seed(seed)
    pairs, labels, train, test = data(p, 100 + seed)
    model = TinyTransformer(p)
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=1.)
    records = []
    for epoch in range(steps + 1):
        if epoch in checkpoints:
            model.eval()
            with torch.no_grad():
                logits, h = model(pairs, hidden=True)
                row = {"epoch": epoch, "train_ce": float(F.cross_entropy(logits[train], labels[train])),
                       "test_ce": float(F.cross_entropy(logits[test], labels[test])),
                       "train_accuracy": float((logits[train].argmax(-1) == labels[train]).float().mean()),
                       "test_accuracy": float((logits[test].argmax(-1) == labels[test]).float().mean()),
                       "compression": compression_metrics(h, labels, train, test, p, seed)}
                if selection is not None:
                    row.update(spectral_losses(logits, labels, train, test, p, selection, control))
                records.append(row)
                if output:
                    output(records)
        if epoch == steps:
            break
        model.train()
        opt.zero_grad()
        loss = F.cross_entropy(model(pairs[train]), labels[train])
        loss.backward()
        opt.step()
    with torch.no_grad():
        final_logits = model(pairs)
    return records, select_calibration_frequencies(final_logits, p)


def experiment(p=31, steps=6000, path=None, seeds=(0, 1), fixed_selection=None, calibration_ref=None):
    torch.set_num_threads(1)
    checkpoints = tuple(x for x in (0, 100, 250, 500, 1000, 1500, 2000, 3000, 4000, 5000,
                                    6000, 8000, 10000, 12000, 15000, 20000)
                        if x <= steps)
    if steps not in checkpoints:
        checkpoints += (steps,)
    started = time.monotonic()
    result = {"configuration": {"p": p, "steps": steps, "train_fraction": .3, "dim": 32,
                "mlp_width": 64, "heads": 2, "optimizer": "AdamW", "lr": .001,
                "weight_decay": 1., "code_dimensions": [16, 8, 4, 2, 1, 0],
                "quantization_bits_per_coordinate": 8, "risk_target": "one-hot sum+1",
                "risk_measure": "uniform on train pairs", "bayes_risk_tolerance": .1,
                "linear_objective_tolerance": .5, "decoder": "fixed 17 inputs including intercept; ridge 1e-3, intercept unpenalized",
                "calibration_seed": 91 if fixed_selection is None else None,
                "calibration_ref": calibration_ref,
                "confirmation_seeds": list(seeds),
                "selection": "three strongest diagonal logit frequencies at independent calibration final checkpoint; frozen before confirmation"},
              "checkpoints": checkpoints, "runs": {}, "torch": torch.__version__}
    def save():
        if path:
            Path(path).write_text(json.dumps(result, indent=2))
    if fixed_selection is None:
        calibration, selection = train_run(91, p, steps, checkpoints,
                                          output=lambda rows: (result["runs"].update({"91": rows}), save()))
        result["runs"]["91"] = calibration
    else:
        selection = tuple(fixed_selection)
        if len(set(selection)) != len(selection) or any(j < 1 or j > (p - 1) // 2 for j in selection):
            raise ValueError("fixed frequencies must be distinct nonzero real Fourier frequencies")
    result["selected_frequencies"] = selection
    control = tuple(j for j in range(1, (p + 1) // 2) if j not in selection)[:len(selection)]
    result["matched_other_frequencies"] = control
    save()
    for seed in seeds:
        rows, _ = train_run(seed, p, steps, checkpoints, selection, control,
                            output=lambda rows, seed=seed: (result["runs"].update({str(seed): rows}), save()))
        result["runs"][str(seed)] = rows
        save()
    result["elapsed_seconds"] = round(time.monotonic() - started, 3)
    save()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--p", type=int, default=31)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--output", type=str)
    parser.add_argument("--confirm-seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--fixed-frequencies", type=int, nargs="+")
    parser.add_argument("--calibration-ref", type=str)
    args = parser.parse_args()
    if args.p < 3 or args.steps < 1:
        parser.error("p>=3 and steps>=1 required; use a prime modulus")
    if args.fixed_frequencies and not args.calibration_ref:
        parser.error("a fixed-frequency run must name its independent calibration ref")
    print(json.dumps(experiment(args.p, args.steps, args.output, args.confirm_seeds,
                                args.fixed_frequencies, args.calibration_ref), indent=2))

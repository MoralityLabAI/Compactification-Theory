"""CPU torch checks for the reduced-scale modular checkpoint assay."""
import unittest
from math import cos, pi

import torch

from cyclic_addition import onehot_risk
from modular_checkpoint import (TinyTransformer, compression_metrics, data,
                                experiment, quantized_codes, select_calibration_frequencies,
                                spectral_losses)


class ModularCheckpointTests(unittest.TestCase):
    def test_splits_and_model_shapes(self):
        pairs, labels, tr, te = data(17, 100)
        self.assertEqual(len(tr) + len(te), 17 * 17)
        self.assertFalse(set(tr.tolist()) & set(te.tolist()))
        self.assertTrue(torch.equal((pairs[:, 0] + pairs[:, 1]) % 17, labels))
        model = TinyTransformer(p=17)
        logits, h = model(pairs[:8], hidden=True)
        self.assertEqual(logits.shape, (8, 17))
        self.assertEqual(h.shape, (8, 64))
        logits.sum().backward()
        self.assertGreater(model.qkv.weight.grad.abs().sum(), 0)

    def test_fourier_selection_and_controls(self):
        p = 17
        pairs, labels, tr, te = data(p, 100)
        c = torch.arange(p).float()
        s = (pairs[:, 0] + pairs[:, 1]) % p
        logits = torch.cos(2 * pi * 3 * (s[:, None] - c[None, :]) / p)
        chosen = select_calibration_frequencies(logits, p, count=1)
        self.assertEqual(chosen, (3,))
        measures = spectral_losses(logits, labels, tr, te, p, chosen, (1,))
        self.assertAlmostEqual(measures['restricted_test_ce'],
                               float(torch.nn.functional.cross_entropy(logits[te], labels[te])), places=5)
        self.assertGreater(measures['matched_other_test_ce'], measures['restricted_test_ce'])

    def test_external_frequency_selection_skips_calibration(self):
        result = experiment(p=17, steps=1, seeds=(2,), fixed_selection=(1, 2, 3),
                            calibration_ref='independent-fixture')
        self.assertEqual(result['selected_frequencies'], (1, 2, 3))
        self.assertEqual(set(result['runs']), {'2'})
        self.assertEqual(len(result['runs']['2']), 2)

    def test_quantization_and_fixed_onehot_target(self):
        p = 17
        pairs, labels, tr, te = data(p, 100)
        torch.manual_seed(3)
        h = torch.randn(len(pairs), 64)
        mean = h[tr].double().mean(0)
        basis = torch.eye(64, dtype=torch.double)
        code = quantized_codes(h.double(), tr, 2, basis, mean)
        self.assertEqual(code.shape, (len(pairs), 2))
        self.assertEqual(quantized_codes(h.double(), tr, 0, basis, mean).shape[1], 0)
        summary = compression_metrics(h, labels, tr, te, p, 0)
        for name in ('pca', 'random'):
            self.assertEqual(len(summary[name]['bayes_train_risk']), 6)
            self.assertEqual(len(summary[name]['linear_decoder_test_mse']), 6)
            self.assertEqual(summary[name]['bayes_train_risk'][-1],
                             float(onehot_risk([()] * len(tr), ((labels[tr] + 1) % p).tolist())))


if __name__ == '__main__':
    unittest.main()

"""Small architecture and split checks; run with apy (CPU torch)."""
import unittest

import torch

from neural_recoverability import (RecurrentAttention, fitted_code_composition,
                                   words_and_targets)
from recoverability_experiment import STATES, build_system, code_path


class NeuralRecoverabilityTests(unittest.TestCase):
    def test_words_and_targets_have_no_heldout_compositions(self):
        system, names = build_system()
        train, heldout, targets = words_and_targets(system, names)
        self.assertEqual(len(train), 28)
        self.assertEqual(len(heldout), 2)
        self.assertFalse(set(train) & set(heldout))
        self.assertEqual(len(set(train)), len(train))
        self.assertEqual(len(targets), 30)
        for word in train + list(heldout):
            self.assertEqual(targets[word].shape, (8, len(word) + 1))

    def test_fitted_code_tables_reproduce_exact_fixture(self):
        system, names = build_system()
        word = (names.index('flip_b'), names.index('reveal_b'))
        targets = tuple(system.task[system.transitions[word[1]][system.transitions[word[0]][x]]]
                        for x in range(len(STATES)))
        for code, mse in zip(code_path(), ('0', '0', '1/4', '1/4')):
            successors = {a: tuple(code[system.transitions[a][x]] for x in range(len(STATES)))
                          for a in range(len(names))}
            self.assertEqual(fitted_code_composition(code, successors, word, targets)['mse'], mse)

    def test_recurrent_attention_receives_gradient(self):
        torch.manual_seed(0)
        model = RecurrentAttention(width=8)
        inputs = torch.tensor(STATES, dtype=torch.float32)
        logits, state = model(inputs, (2, 3))
        self.assertEqual(logits.shape, (8, 3))
        self.assertEqual(state.shape, (8, 8))
        logits.sum().backward()
        self.assertGreater(model.attention.in_proj_weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.encode.weight.grad.abs().sum().item(), 0)


if __name__ == '__main__':
    unittest.main()

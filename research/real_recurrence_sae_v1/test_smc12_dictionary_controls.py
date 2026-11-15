"""SMC-12B dictionary-wide exact control and original initialization hash tests."""
import sys
from pathlib import Path
import unittest

import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from smc12_dictionary_controls import (
    initialization_control,exact_dictionary_cones,EXPECTED_INITIAL_DECODER_SHA
)


class DictionaryControlsTests(unittest.TestCase):
    def test_frozen_original_initializer_matches_previous_training_receipt(self):
        scale=torch.ones(48,dtype=torch.float32)
        for seed in (71,89,113):
            model,digest=initialization_control(seed,scale)
            self.assertEqual(digest,EXPECTED_INITIAL_DECODER_SHA[seed])
            self.assertEqual(tuple(model.encoder.weight.shape),(96,48))
            self.assertEqual(tuple(model.decoder.weight.shape),(48,96))

    def test_exact_all_96_initial_atoms_classified(self):
        scale=torch.ones(48,dtype=torch.float32)
        model,_=initialization_control(71,scale)
        result=exact_dictionary_cones(model,scale)
        self.assertEqual(sum(result['counts'].values()),96)
        self.assertEqual(result['all_source_atom_count'],96)
        self.assertEqual(result['strict_cone_interior_count'],1)
        self.assertTrue(all(0<=j<96 for j in result['infinite_self_selection_atom_ids']))

    def test_reject_unpinned_seed_and_wrong_architecture(self):
        scale=torch.ones(48,dtype=torch.float32)
        with self.assertRaises(ValueError):
            initialization_control(999,scale)
        model,_=initialization_control(71,scale)
        with self.assertRaises(ValueError):
            exact_dictionary_cones(model,torch.ones(47,dtype=torch.float32))
        model.encoder=torch.nn.Linear(48,64)
        with self.assertRaises(ValueError):
            exact_dictionary_cones(model,scale)


if __name__=='__main__':
    unittest.main()

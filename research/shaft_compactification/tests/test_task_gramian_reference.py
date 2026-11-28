import unittest
from research.shaft_compactification.task_gramian_reference import (
    dot,weighted_energy,cumulative_energies,reference
)

class TaskGramianTests(unittest.TestCase):
    def test_fixed_rank_can_have_different_directional_energy(self):
        self.assertEqual(weighted_energy([(1,0)],(1,0),[1]),1)
        self.assertEqual(weighted_energy([(1,0)],(0,1),[1]),0)
    def test_added_output_observes_hidden_axis(self):
        self.assertEqual(cumulative_energies([(1,0),(0,1)],(0,1),[1,1]),[0,0,1])
    def test_weight_zero_removes_observability(self):
        self.assertEqual(weighted_energy([(1,0)],(5,1),[0]),0)
    def test_invalid_inputs(self):
        with self.assertRaises(ValueError): dot((1,),(1,2))
        with self.assertRaises(ValueError): weighted_energy([(1,0)],(0,1),[])
        with self.assertRaises(ValueError): weighted_energy([(1,0)],(0,1),[-1])
    def test_full_exhaustive_receipt(self):
        self.assertEqual(reference()['exact_weighted_monotonicity_cases'],81)

if __name__=="__main__":unittest.main()

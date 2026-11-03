import unittest
import torch
from experiments.toy_contracts import ToyRV32, FrozenToyBase, ToyGraft, run_smokes

class ToyContractTests(unittest.TestCase):
    def test_randomized_arithmetic_and_reversible_gate(self):
        d = run_smokes(trials=128)
        self.assertEqual(d['reference_mismatches'], 0)
        self.assertTrue(d['reversible_bridge_zero_gate_exact'])
        self.assertTrue(d['recurrence_output_changes_with_K'])
        self.assertTrue(d['gradient_finite'])

    def test_branch_and_reset(self):
        p = [('li',1,4),('li',2,4),('beq',1,2,5),('li',3,99),('halt',),('li',3,33),('halt',)]
        m = ToyRV32().run(p)
        self.assertEqual(m.reg[3], 33)
        self.assertEqual(ToyRV32().reg,[0]*8)

    def test_bounds_and_invalid_instruction(self):
        with self.assertRaises(ValueError): ToyRV32().run([('li',1,1),('sw',1,127)])
        with self.assertRaises(ValueError): ToyRV32().run([('bogus',)])
        with self.assertRaises(ValueError): ToyRV32().run([('beq',0,0,100)])

    def test_corrupted_instruction_detected(self):
        valid = [('li',1,2),('li',2,5),('add',3,1,2),('halt',)]
        self.assertEqual(ToyRV32().run(valid).reg[3],7)
        corrupt= [('li',1,2),('li',2,5),('sub',3,1,2),('halt',)]
        self.assertNotEqual(ToyRV32().run(corrupt).reg[3],7)

if __name__ == '__main__': unittest.main()

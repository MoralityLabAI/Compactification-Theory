import unittest

import torch

from trajectory_lab import (
    ResidualMLP, allowed, enforce_gate, grow, make_data,
    interpolation_barrier, macs, rank_layers, resize, run_seed, uniform_indices,
)


class TrajectoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_oracle_and_phase_gate(self):
        x, y, phase = make_data(300, 200)
        self.assertEqual(x.shape, (300, 12))
        self.assertTrue(torch.all(y // 2 == phase))
        pred = torch.arange(6).repeat(50)
        self.assertTrue(torch.all(allowed(phase, enforce_gate(phase, pred))))

    def test_exact_same_architecture_copy(self):
        torch.manual_seed(1)
        model = ResidualMLP(8, 64)
        clone = resize(model, 8, 64)
        x = torch.randn(10, 12)
        self.assertTrue(torch.equal(model(x), clone(x)))
        self.assertNotEqual(model.blocks[0].weight.data_ptr(), clone.blocks[0].weight.data_ptr())

    def test_layer_selection(self):
        self.assertEqual(uniform_indices(8, 4), [0, 2, 5, 7])
        with self.assertRaises(ValueError):
            resize(ResidualMLP(8, 64), 4, 32, [1, 1, 2, 3])
        a = ResidualMLP(8, 64)
        self.assertEqual(len(rank_layers(a, torch.randn(32, 12), 4)), 4)

    def test_growth_copy_and_identity(self):
        torch.manual_seed(0)
        base = ResidualMLP(4, 32)
        clone = grow(base, "copy", 10)
        ident = grow(base, "identity", 10)
        self.assertEqual(clone.depth, 8)
        self.assertEqual(clone.width, base.width)
        self.assertTrue(torch.equal(clone.blocks[1].weight, clone.blocks[3].weight))
        self.assertNotEqual(clone.blocks[1].weight.data_ptr(), clone.blocks[3].weight.data_ptr())
        x = torch.randn(24, 12)
        self.assertTrue(torch.allclose(ident(x), base(x), atol=1e-6, rtol=0))
        copy_scaled = grow(base, "copy_scaled", 10)
        random_scaled = grow(base, "random_scaled", 10)
        self.assertTrue(torch.equal(copy_scaled.gains, random_scaled.gains))
        self.assertTrue(torch.equal(copy_scaled.blocks[1].weight, copy_scaled.blocks[3].weight))
        self.assertAlmostEqual(interpolation_barrier(base, base, (x, torch.zeros(len(x), dtype=torch.long), None)), 0)

    def test_smoke_and_budget(self):
        a = run_seed(0, teacher_steps=3, student_steps=4, batch=8, budget="tokens", growth=True)
        self.assertEqual(len(a["arms"]), 7)
        for row in a["arms"].values():
            self.assertEqual(row["final"], [4, 32])
            self.assertEqual(row["recovery_examples"], 32)
            self.assertEqual(row["validation"]["post_gate_violation"], 0)
            self.assertGreaterEqual(row["validation"]["accuracy"], 0)
        self.assertEqual(len(a["growth"]), 4)
        self.assertEqual(len(a["interpolation_barriers_vs_one_shot_kd"]), 6)
        self.assertLess(a["growth"]["identity"]["initial_function_mse"], 1e-11)
        b = run_seed(0, teacher_steps=3, student_steps=12, batch=8, budget="flops")
        standard = b["arms"]["one_shot_kd"]["train_mac_proxy"]
        for row in b["arms"].values():
            self.assertLessEqual(row["train_mac_proxy"], standard)
            self.assertGreater(row["recovery_examples"], 0)


if __name__ == "__main__":
    unittest.main()

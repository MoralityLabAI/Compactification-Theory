"""Finite model tests: invariant checks and required negative controls."""
import sys
import unittest
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import (ARMS, Shock, State, basin, basin_probability, first_quorum_loss_probability,
                   quorum_margin_witness, recovery_steps, repair, safety_kernel,
                   shortest_violation, shocks, transition)


class ModelTests(unittest.TestCase):
    def test_repair_quorum_and_outside_support(self):
        self.assertEqual(repair(0b0011, 4, 2, "quorum"), 0b0111)
        self.assertEqual(repair(0b0001, 4, 2, "quorum"), 0b0001)
        self.assertEqual(repair(0b0000, 4, 2, "backup"), 0b0001)
        self.assertEqual(repair(0b0011, 4, 2, "quorum", True), 0b0011)

    def test_recovery_basins_exact(self):
        self.assertEqual([basin(a)["fraction"] for a in ARMS],
                         ["1/16", "11/16", "11/16", "1"])
        self.assertEqual([basin(a)["max_recovery_rounds"] for a in ARMS],
                         [0, 2, 2, 4])

    def test_basin_prior_dependence(self):
        self.assertEqual(basin_probability(ARMS[0], initial_support_probability=Fraction(3, 4)), Fraction(81, 256))
        self.assertEqual(basin_probability(ARMS[1], initial_support_probability=Fraction(3, 4)), Fraction(243, 256))
        self.assertEqual(basin_probability(ARMS[3], initial_support_probability=Fraction(3, 4)), Fraction(1))
        for a in ARMS:
            self.assertEqual(basin_probability(a), Fraction(basin(a)["fraction"]))
        with self.assertRaises(ValueError):
            basin_probability(ARMS[1], initial_support_probability=Fraction(2))

    def test_recovery_threshold(self):
        self.assertIsNone(recovery_steps(State(1), ARMS[1]))
        self.assertEqual(recovery_steps(State(3), ARMS[1]), 2)
        self.assertEqual(recovery_steps(State(0), ARMS[3]), 4)

    def test_known_harm_independently_vetoed_all_states(self):
        attacks = shocks()
        for a in (ARMS[0], ARMS[3]):
            kernel = safety_kernel(a, attacks)
            self.assertEqual(len(kernel), 32)
            self.assertIsNone(shortest_violation(State(15), a, attacks))

    def test_mutable_gate_counterexample(self):
        step = transition(State(15), Shock("known", 4, rewrite_gate=True), ARMS[1])
        self.assertTrue(step.violated)
        self.assertFalse(step.state.gate_intact)
        self.assertFalse(step.vetoed)
        self.assertEqual(len(safety_kernel(ARMS[1], shocks())), 0)

    def test_novel_harm_limits_verifier(self):
        for a in (ARMS[0], ARMS[3]):
            step = transition(State(15), Shock("novel", 4), a)
            self.assertTrue(step.violated)
        self.assertEqual(len(safety_kernel(ARMS[3], shocks(novel=True))), 0)

    def test_peer_corruption_blocks_novel_recognition(self):
        healthy = transition(State(15), Shock("novel", 0), ARMS[3])
        poisoned = transition(State(15), Shock("novel", 0, peer_corrupted=True), ARMS[3])
        self.assertFalse(healthy.violated)
        self.assertTrue(poisoned.violated)

    def test_soft_pressure_competes_with_drive(self):
        self.assertFalse(transition(State(15), Shock("novel", 3), ARMS[3]).violated)
        self.assertTrue(transition(State(15), Shock("novel", 4), ARMS[3]).violated)

    def test_benign_work_completion(self):
        for a in ARMS:
            self.assertEqual(transition(State(15), Shock(), a).emitted, "serve")

    def test_quorum_margin_exhaustive(self):
        x = quorum_margin_witness()
        self.assertEqual(x["robust_support_threshold"], 3)
        self.assertGreater(x["checked_edges"], 0)
        self.assertEqual(x["boundary_counterexample"]["end"].bit_count(), 1)

    def test_shortest_counterexample_replays(self):
        path = shortest_violation(State(15), ARMS[1], shocks(), max_steps=4)
        self.assertIsNotNone(path)
        state = State(15)
        for node in path:
            shock = Shock(**node["shock"])
            out = transition(state, shock, ARMS[1])
            self.assertEqual(node["after"], {"supports": out.state.supports,
                                            "gate_intact": out.state.gate_intact})
            self.assertEqual(node["violated"], out.violated)
            state = out.state
        self.assertTrue(path[-1]["violated"])

    def test_exact_probability_bounds(self):
        for a in ARMS:
            p = first_quorum_loss_probability(a)
            self.assertGreaterEqual(p, 0)
            self.assertLessEqual(p, 1)
        self.assertEqual(first_quorum_loss_probability(ARMS[1], rounds=0), 0)
        self.assertEqual(first_quorum_loss_probability(ARMS[1], rounds=1,
                        per_support_loss=Fraction(0)), 0)
        self.assertEqual(first_quorum_loss_probability(ARMS[1], rounds=1,
                        per_support_loss=Fraction(1)), 1)

    def test_independent_backup_reduces_first_loss(self):
        self.assertLessEqual(first_quorum_loss_probability(ARMS[3]),
                             first_quorum_loss_probability(ARMS[1]))

    def test_validation(self):
        with self.assertRaises(ValueError):
            repair(4, 2, 3, "quorum")
        with self.assertRaises(ValueError):
            transition(State(15), Shock("known", 5), ARMS[3])
        with self.assertRaises(ValueError):
            safety_kernel(ARMS[3], [])
        with self.assertRaises(ValueError):
            quorum_margin_witness(4, 4)


if __name__ == "__main__":
    unittest.main()
